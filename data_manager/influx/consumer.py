#!/usr/bin/env python3
import os
import json
import time
import logging
from datetime import datetime, timezone
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
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 500))            # máximo mensajes por lote
POLL_TIMEOUT_MS = int(os.getenv('POLL_TIMEOUT_MS', 1000)) # tiempo de poll en ms
DEDUP_TTL_SECONDS = int(os.getenv('DEDUP_TTL_SECONDS', 5))# ventana para deduplicación en memoria
WRITE_RETRIES = int(os.getenv('WRITE_RETRIES', 3))
RETRY_BACKOFF_SECONDS = float(os.getenv('RETRY_BACKOFF_SECONDS', 1.0))

# DLQ file prefix (si falla escritura repetidamente)
DLQ_PATH = os.getenv('DLQ_PATH', '/tmp/kafka_influx_dlq.jsonl')

# Logging básico
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
log = logging.getLogger("kafka->influx")

# ---------- Kafka consumer (manual commit) ----------
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=False,   # importante: commit manual para garantizar durabilidad
    group_id=GROUP_ID
)

def parse_ts(ts):
    """Normaliza ts_candle que puede venir como ISO str o epoch ms (int/float). Devuelve datetime aware UTC."""
    if isinstance(ts, str):
        # fromisoformat no acepta 'Z' — convertimos
        try:
            if ts.endswith("Z"):
                ts = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(ts)
        except Exception:
            # último recurso: intentar parse manual como número en string
            try:
                num = float(ts)
                return datetime.utcfromtimestamp(num / 1000.0).replace(tzinfo=timezone.utc)
            except Exception:
                raise
    elif isinstance(ts, (int, float)):
        return datetime.utcfromtimestamp(ts / 1000.0).replace(tzinfo=timezone.utc)
    elif isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        raise ValueError("Timestamp missing or invalid type")

def write_dlq(messages):
    """Vuelca mensajes problemáticos a DLQ (jsonl)."""
    try:
        with open(DLQ_PATH, 'a', encoding='utf-8') as f:
            for m in messages:
                json.dump(m, f, ensure_ascii=False)
                f.write("\n")
        log.warning("Dumped %d messages to DLQ (%s)", len(messages), DLQ_PATH)
    except Exception as e:
        log.exception("Failed writing DLQ: %s", e)

def cleanup_seen_cache(seen_cache, ttl):
    """Eliminar claves antiguas de la cache de dedup"""
    now = time.time()
    to_delete = [k for k, v in seen_cache.items() if now - v > ttl]
    for k in to_delete:
        del seen_cache[k]

def main_loop():
    # Influx client y write API (sin opciones adicionales para mantener compatibilidad)
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        write_api = client.write_api()
        log.info("Consumer started. Waiting for messages...")

        seen_cache = {}  # key -> last_seen_epoch_seconds (dedup)
        stats = {'processed': 0, 'written': 0, 'skipped_dup': 0, 'errors': 0}

        try:
            while True:
                # poll: devuelve dict TopicPartition -> [msgs]
                records = consumer.poll(timeout_ms=POLL_TIMEOUT_MS, max_records=BATCH_SIZE)
                if not records:
                    # limpieza periódica de cache para no crecer indefinidamente
                    cleanup_seen_cache(seen_cache, DEDUP_TTL_SECONDS)
                    continue

                points = []
                batch_raw_msgs_for_dlq = []  # si fallamos, volcamos ahí
                any_parsed = False

                # aplanar mensajes y procesar
                for tp, msgs in records.items():
                    for m in msgs:
                        stats['processed'] += 1
                        data = m.value
                        try:
                            # campos obligatorios
                            exch = data.get('exchange')
                            sym = data.get('symbol')
                            tf = data.get('timeframe', 'unknown')
                            ts_raw = data.get('ts_candle') or data.get('ts') or data.get('timestamp')
                            if not (exch and sym and ts_raw):
                                raise ValueError("Missing required keys (exchange/symbol/ts_candle)")

                            # dedup key (por vela exacta)
                            dedup_key = f"{exch}|{sym}|{tf}|{ts_raw}"

                            now_epoch = time.time()
                            last_seen = seen_cache.get(dedup_key)
                            if last_seen and (now_epoch - last_seen) < DEDUP_TTL_SECONDS:
                                stats['skipped_dup'] += 1
                                continue
                            # marcar visto ahora
                            seen_cache[dedup_key] = now_epoch

                            # parse timestamp
                            ts = parse_ts(ts_raw)

                            # campos numericos
                            open_v = float(data.get('open', 0.0))
                            high_v = float(data.get('high', 0.0))
                            low_v = float(data.get('low', 0.0))
                            close_v = float(data.get('close', 0.0))
                            vol_v = float(data.get('volume', 0.0))

                            p = (
                                Point("prices")
                                .tag("exchange", exch)
                                .tag("symbol", sym)
                                .tag("timeframe", tf)
                                .field("open", open_v)
                                .field("high", high_v)
                                .field("low", low_v)
                                .field("close", close_v)
                                .field("volume", vol_v)
                                .time(ts, WritePrecision.NS)
                            )
                            points.append(p)
                            any_parsed = True
                        except Exception as e:
                            stats['errors'] += 1
                            log.exception("Skipping message due to parse error: %s", e)
                            # guardar el mensaje en DLQ batch para no bloquear
                            batch_raw_msgs_for_dlq.append(data)

                # Si no hay puntos válidos, igual commit para avanzar (después de DLQ)
                if not points:
                    if batch_raw_msgs_for_dlq:
                        write_dlq(batch_raw_msgs_for_dlq)
                        # Commit offsets para saltar mensajes problemáticos
                        try:
                            consumer.commit()
                        except Exception:
                            log.exception("Failed committing offsets after DLQ dump")
                    # limpieza y seguir
                    cleanup_seen_cache(seen_cache, DEDUP_TTL_SECONDS)
                    continue

                # Intentar escribir a Influx con reintentos
                write_ok = False
                last_exc = None
                for attempt in range(1, WRITE_RETRIES + 1):
                    try:
                        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=points)
                        write_ok = True
                        stats['written'] += len(points)
                        break
                    except Exception as e:
                        last_exc = e
                        stats['errors'] += 1
                        log.exception("Write attempt %d failed: %s", attempt, e)
                        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

                if not write_ok:
                    # volcar el batch crudo a DLQ y avanzar offsets para no bloquear
                    log.error("Batch write failed after %d attempts, dumping to DLQ", WRITE_RETRIES)
                    # intentar extraer raw messages de records para dlq
                    dlq_msgs = []
                    for tp, msgs in records.items():
                        for m in msgs:
                            try:
                                dlq_msgs.append(m.value)
                            except Exception:
                                pass
                    write_dlq(dlq_msgs)
                    try:
                        consumer.commit()
                    except Exception:
                        log.exception("Failed committing offsets after DLQ dump")
                else:
                    # Commit sólo si el write tuvo éxito
                    try:
                        consumer.commit()
                    except Exception:
                        log.exception("Failed committing offsets after successful write")

                # limpieza periódica
                cleanup_seen_cache(seen_cache, DEDUP_TTL_SECONDS)

                # logging de estadísticas ocasional
                if stats['processed'] % max(1, BATCH_SIZE * 5) == 0:
                    log.info("Stats processed=%d written=%d skipped_dup=%d errors=%d",
                             stats['processed'], stats['written'], stats['skipped_dup'], stats['errors'])

        except KeyboardInterrupt:
            log.info("Consumer stopped by user")
        except Exception:
            log.exception("Unhandled exception in consumer loop")

if __name__ == '__main__':
    main_loop()
