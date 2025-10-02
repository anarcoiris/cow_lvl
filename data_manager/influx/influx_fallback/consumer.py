#!/usr/bin/env python3
"""
consumer.py

Kafka -> SQLite persistent queue -> Influx writer

- Consumer encola mensajes rápido en SQLite (WAL) y hace commit de offsets.
- InfluxWriter (thread) lee la cola, intenta escribir a Influx con reintentos
  exponenciales y mueve mensajes fallidos a DLQ.
- Evita acumulación de memoria si Influx está lento; persiste backlog en disco.

Requisitos: kafka-python, influxdb-client
"""
from __future__ import annotations

import os
import json
import time
import sqlite3
import logging
import signal
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point, WritePrecision

# ---------- Config (variables de entorno) ----------
BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC', 'prices')
GROUP_ID = os.getenv('KAFKA_GROUP_ID', 'market-group')

INFLUX_URL = os.getenv('INFLUXDB_URL', 'http://influxdb:8086')
INFLUX_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
INFLUX_ORG = os.getenv('INFLUXDB_ORG', '')
INFLUX_BUCKET = os.getenv('INFLUXDB_BUCKET', '')

# Consumer batching / dedup / retry
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 500))            # máximo mensajes por lote leídos de Kafka
POLL_TIMEOUT_MS = int(os.getenv('POLL_TIMEOUT_MS', 1000)) # tiempo de poll en ms
DEDUP_TTL_SECONDS = int(os.getenv('DEDUP_TTL_SECONDS', 5))# ventana para deduplicación en memoria
WRITE_RETRIES = int(os.getenv('WRITE_RETRIES', 5))
INITIAL_BACKOFF_SECONDS = float(os.getenv('INITIAL_BACKOFF_SECONDS', 1.0))
BACKOFF_MULT = float(os.getenv('BACKOFF_MULT', 2.0))
MAX_ATTEMPTS = int(os.getenv('MAX_ATTEMPTS', WRITE_RETRIES))

# SQLite queue / DLQ
SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', '/tmp/write_queue.db')
DLQ_TABLE = os.getenv('DLQ_TABLE', 'dlq')
DLQ_FILE = os.getenv('DLQ_PATH', '/tmp/kafka_influx_dlq.jsonl')

# When queue rows exceed this threshold, backpressure consumer (sleep)
QUEUE_BACKPRESSURE_THRESHOLD = int(os.getenv('QUEUE_BACKPRESSURE_THRESHOLD', 200_000))
CONSUMER_BACKOFF_SLEEP = float(os.getenv('CONSUMER_BACKOFF_SLEEP', 1.0))

# Control de normalización de tags:
PRESERVE_TAG_CASE = os.getenv('PRESERVE_TAG_CASE', 'false').lower() in ('1', 'true', 'yes')

# Logging básico
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("kafka->influx")

# Graceful shutdown
_shutdown = threading.Event()


# ---------- SQLite persistent queue helpers ----------
def init_db(db_path: str = SQLITE_DB_PATH) -> sqlite3.Connection:
    """Inicializa la DB SQLite con WAL y tablas write_queue + dlq."""
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("""
    CREATE TABLE IF NOT EXISTS write_queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT,
        ts REAL,
        payload TEXT,
        attempts INTEGER DEFAULT 0,
        next_attempt_ts REAL DEFAULT 0,
        created_ts REAL
    );
    """)
    con.execute(f"""
    CREATE TABLE IF NOT EXISTS {DLQ_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT,
        ts REAL,
        payload TEXT,
        error TEXT,
        failed_at REAL
    );
    """)
    # index para seleccionar por next_attempt_ts rápidamente
    con.execute("CREATE INDEX IF NOT EXISTS idx_next_attempt ON write_queue(next_attempt_ts);")
    con.commit()
    log.info("SQLite DB initialized at %s", db_path)
    return con


def enqueue_message(con: sqlite3.Connection, key: str, ts: float, payload: dict) -> int:
    """Inserta mensaje en la cola persistente. Devuelve id row."""
    cur = con.cursor()
    cur.execute(
        "INSERT INTO write_queue (key, ts, payload, attempts, next_attempt_ts, created_ts) VALUES (?, ?, ?, 0, 0, ?)",
        (key, ts, json.dumps(payload, ensure_ascii=False), time.time())
    )
    con.commit()
    rowid = cur.lastrowid
    log.debug("Enqueued id=%s key=%s ts=%s", rowid, key, ts)
    return rowid


def fetch_batch_for_write(con: sqlite3.Connection, limit: int = 500) -> List[Tuple]:
    """
    Recupera filas con next_attempt_ts <= now para procesar (orden por created_ts asc).
    Retorna lista de tuples (id, key, ts, payload, attempts)
    """
    cur = con.cursor()
    now = time.time()
    cur.execute("SELECT id, key, ts, payload, attempts FROM write_queue WHERE next_attempt_ts <= ? ORDER BY created_ts ASC LIMIT ?", (now, limit))
    rows = cur.fetchall()
    return rows


def delete_queue_ids(con: sqlite3.Connection, ids: List[int]) -> None:
    if not ids:
        return
    cur = con.cursor()
    cur.executemany("DELETE FROM write_queue WHERE id = ?", [(i,) for i in ids])
    con.commit()
    log.debug("Deleted %d queued rows", len(ids))


def update_attempts_and_backoff(con: sqlite3.Connection, id_: int, attempts: int, backoff_seconds: float) -> None:
    next_ts = time.time() + backoff_seconds
    cur = con.cursor()
    cur.execute("UPDATE write_queue SET attempts = ?, next_attempt_ts = ? WHERE id = ?", (attempts, next_ts, id_))
    con.commit()
    log.debug("Updated attempts for id=%s attempts=%d next_ts=%s", id_, attempts, next_ts)


def move_to_dlq(con: sqlite3.Connection, id_: int, error: str) -> None:
    cur = con.cursor()
    # move the row to dlq table
    cur.execute("INSERT INTO {} (key, ts, payload, error, failed_at) SELECT key, ts, payload, ?, ? FROM write_queue WHERE id=?".format(DLQ_TABLE),
                (error, time.time(), id_))
    cur.execute("DELETE FROM write_queue WHERE id = ?", (id_,))
    con.commit()
    log.warning("Moved id=%s to DLQ table due to error: %s", id_, error)


def get_queue_length(con: sqlite3.Connection) -> int:
    cur = con.cursor()
    cur.execute("SELECT COUNT(1) FROM write_queue")
    return int(cur.fetchone()[0])


# ---------- Utilities ----------
def parse_ts(ts):
    """Normaliza ts_candle que puede venir como ISO str o epoch ms (int/float). Devuelve epoch seconds float."""
    if isinstance(ts, str):
        try:
            if ts.endswith("Z"):
                ts = ts.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            try:
                num = float(ts)
                return num / 1000.0
            except Exception:
                raise
    elif isinstance(ts, (int, float)):
        # asume epoch ms
        return float(ts) / 1000.0
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    else:
        raise ValueError("Timestamp missing or invalid type")


def write_dlq_file(messages: List[dict], path: str = DLQ_FILE) -> None:
    """Volcado simple a fichero JSONL para debugging / recuperación manual."""
    try:
        with open(path, 'a', encoding='utf-8') as f:
            for m in messages:
                rec = {
                    "ts_dlq": datetime.now(timezone.utc).isoformat(),
                    "message": m
                }
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
        log.warning("Dumped %d messages to DLQ file (%s)", len(messages), path)
    except Exception as e:
        log.exception("Failed writing DLQ file: %s", e)


def normalize_tags(exchange: str, symbol: str) -> Tuple[str, str]:
    """Normalización consistente (o no) según PRESERVE_TAG_CASE."""
    if PRESERVE_TAG_CASE:
        return exchange, symbol
    exch_norm = exchange.lower()
    sym_norm = symbol.replace('/', '').upper()
    return exch_norm, sym_norm


# ---------- Influx writer thread ----------
class InfluxWriter(threading.Thread):
    def __init__(self,
                 db_path: str,
                 influx_url: str,
                 influx_token: str,
                 influx_org: str,
                 influx_bucket: str,
                 batch_size: int = 200,
                 max_attempts: int = MAX_ATTEMPTS):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.influx_url = influx_url
        self.influx_token = influx_token
        self.influx_org = influx_org
        self.influx_bucket = influx_bucket
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self._stop = threading.Event()
        # sqlite connection specific for writer
        self.con = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL;")
        # Influx client
        self.client = InfluxDBClient(url=self.influx_url, token=self.influx_token, org=self.influx_org)
        self.write_api = self.client.write_api()
        log.info("InfluxWriter initialized (bucket=%s org=%s)", influx_bucket, influx_org)

    def stop(self):
        self._stop.set()

    def run(self):
        log.info("InfluxWriter started")
        while not self._stop.is_set():
            try:
                rows = fetch_batch_for_write(self.con, limit=self.batch_size)
                if not rows:
                    time.sleep(0.5)
                    continue

                ids_to_delete = []
                dlq_ids = []
                points_batch = []
                # prepare points and mapping id->payload
                id_payload_map: Dict[int, Tuple[str, float, dict, int]] = {}  # id -> (key, ts, payload_obj, attempts)
                for r in rows:
                    id_, key, ts, payload_json, attempts = r
                    try:
                        payload_obj = json.loads(payload_json)
                    except Exception:
                        payload_obj = None
                    id_payload_map[id_] = (key, ts, payload_obj, attempts)
                    # we'll construct points one-by-one during the write attempt

                # Try write each row sequentially (could be batched if payloads are points)
                for id_, (key, ts, payload_obj, attempts) in id_payload_map.items():
                    if payload_obj is None:
                        # malformed payload -> move to dlq table directly
                        move_to_dlq(self.con, id_, "malformed_payload")
                        continue
                    try:
                        exch = payload_obj.get('exchange')
                        sym = payload_obj.get('symbol')
                        tf = payload_obj.get('timeframe', 'unknown')
                        # normalize tags like consumer does
                        exch_norm, sym_norm = normalize_tags(exch, sym)
                        open_v = float(payload_obj.get('open', 0.0))
                        high_v = float(payload_obj.get('high', 0.0))
                        low_v = float(payload_obj.get('low', 0.0))
                        close_v = float(payload_obj.get('close', 0.0))
                        vol_v = float(payload_obj.get('volume', 0.0))

                        p = (
                            Point("prices")
                                .tag("exchange", exch_norm)
                                .tag("symbol", sym_norm)
                                .tag("timeframe", tf)
                                .field("open", open_v)
                                .field("high", high_v)
                                .field("low", low_v)
                                .field("close", close_v)
                                .field("volume", vol_v)
                                .time(int(ts * 1e9), WritePrecision.NS)  # ts is epoch seconds -> ns
                        )
                        # we'll attempt to write the point
                        success = False
                        attempt = attempts
                        while attempt < self.max_attempts and not success:
                            try:
                                # write single point (could be optimized for batches)
                                self.write_api.write(bucket=self.influx_bucket, org=self.influx_org, record=p)
                                success = True
                            except Exception as e:
                                attempt += 1
                                backoff = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULT ** (attempt - 1))
                                log.warning("Influx write failed id=%s attempt=%d backoff=%.1fs err=%s", id_, attempt, backoff, e)
                                time.sleep(min(backoff, 300))
                        if success:
                            ids_to_delete.append(id_)
                        else:
                            # reached max attempts -> move to dlq table
                            move_to_dlq(self.con, id_, "max_attempts_reached")
                    except Exception as e:
                        log.exception("Error preparing/writing point for id=%s: %s", id_, e)
                        # defensive: move to dlq to avoid loop
                        move_to_dlq(self.con, id_, f"exception:{e}")

                if ids_to_delete:
                    delete_queue_ids(self.con, ids_to_delete)

            except Exception as e:
                log.exception("Unhandled exception in InfluxWriter loop: %s", e)
                time.sleep(5.0)

        # cleanup on stop
        try:
            self.client.close()
        except Exception:
            pass
        log.info("InfluxWriter stopped")


# ---------- Consumer loop (producer -> queue) ----------
def start_kafka_consumer_loop(sqlite_con: sqlite3.Connection):
    """Consume desde Kafka y encola en SQLite. Commit manual al encolar."""
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id=GROUP_ID,
        consumer_timeout_ms=1000
    )

    seen_cache: Dict[str, float] = {}
    stats = {'processed': 0, 'enqueued': 0, 'skipped_dup': 0, 'errors': 0}

    try:
        log.info("Kafka consumer started, topic=%s", TOPIC)
        while not _shutdown.is_set():
            records = consumer.poll(timeout_ms=POLL_TIMEOUT_MS, max_records=BATCH_SIZE)
            if not records:
                # periodic cleanup
                now = time.time()
                # cleanup seen cache entries older than DEDUP_TTL_SECONDS
                old_keys = [k for k, v in seen_cache.items() if now - v > DEDUP_TTL_SECONDS]
                for k in old_keys:
                    del seen_cache[k]
                # backpressure check by queue length
                queue_len = get_queue_length(sqlite_con)
                if queue_len > QUEUE_BACKPRESSURE_THRESHOLD:
                    log.warning("Queue length %d exceeds threshold %d -> consumer will sleep %.2fs",
                                queue_len, QUEUE_BACKPRESSURE_THRESHOLD, CONSUMER_BACKOFF_SLEEP)
                    time.sleep(CONSUMER_BACKOFF_SLEEP)
                continue

            batch_dlq_candidates = []
            for tp, msgs in records.items():
                for m in msgs:
                    stats['processed'] += 1
                    try:
                        data = m.value
                        exch = data.get('exchange')
                        sym = data.get('symbol')
                        tf = data.get('timeframe', 'unknown')
                        ts_raw = data.get('ts_candle') or data.get('ts') or data.get('timestamp')
                        if not (exch and sym and ts_raw):
                            raise ValueError("Missing required keys (exchange/symbol/ts_candle)")

                        exch_norm, sym_norm = normalize_tags(exch, sym)
                        dedup_key = f"{exch_norm}|{sym_norm}|{tf}|{ts_raw}"
                        now_epoch = time.time()
                        last_seen = seen_cache.get(dedup_key)
                        if last_seen and (now_epoch - last_seen) < DEDUP_TTL_SECONDS:
                            stats['skipped_dup'] += 1
                            continue
                        seen_cache[dedup_key] = now_epoch

                        # parse timestamp to epoch seconds for enqueue
                        ts_epoch = parse_ts(ts_raw)
                        # create queue key (could be same as dedup_key)
                        queue_key = dedup_key

                        # enqueue to sqlite quickly
                        enqueue_message(sqlite_con, queue_key, float(ts_epoch), data)
                        stats['enqueued'] += 1

                        # commit kafka offset now that we've safely queued
                        try:
                            consumer.commit()
                        except Exception:
                            log.exception("Failed to commit offsets after enqueue")
                    except Exception as e:
                        stats['errors'] += 1
                        log.exception("Skipping message due to parse/enqueue error: %s", e)
                        # store raw message in DLQ file for later inspection
                        try:
                            batch_dlq_candidates.append(m.value)
                        except Exception:
                            pass

            if batch_dlq_candidates:
                write_dlq_file(batch_dlq_candidates)

            # stats log occasionally
            if stats['processed'] % max(1, BATCH_SIZE * 5) == 0:
                log.info("Stats processed=%d enqueued=%d skipped_dup=%d errors=%d queue_len=%d",
                         stats['processed'], stats['enqueued'], stats['skipped_dup'], stats['errors'], get_queue_length(sqlite_con))

    except KeyboardInterrupt:
        log.info("Consumer stopped by KeyboardInterrupt")
    except Exception as e:
        log.exception("Unhandled exception in consumer loop: %s", e)
    finally:
        try:
            consumer.close()
        except Exception:
            pass
        log.info("Kafka consumer closed")


# ---------- Signal handlers ----------
def _signal_handler(signum, frame):
    log.info("Received signal %s, shutting down...", signum)
    _shutdown.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------- Entrypoint ----------
def main():
    # init sqlite db
    sqlite_con = init_db(SQLITE_DB_PATH)

    # start writer thread
    writer = InfluxWriter(db_path=SQLITE_DB_PATH,
                          influx_url=INFLUX_URL,
                          influx_token=INFLUX_TOKEN,
                          influx_org=INFLUX_ORG,
                          influx_bucket=INFLUX_BUCKET,
                          batch_size=200,
                          max_attempts=MAX_ATTEMPTS)
    writer.start()

    # start kafka consumer loop (main thread)
    try:
        start_kafka_consumer_loop(sqlite_con)
    finally:
        # request writer stop and wait
        writer.stop()
        writer.join(timeout=10)
        log.info("Shutdown complete")


if __name__ == '__main__':
    main()
