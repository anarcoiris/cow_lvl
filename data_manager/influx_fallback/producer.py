#!/usr/bin/env python3
"""
producer.py - robust Kafka producer with local bounded queue, sender thread, DLQ and WS health checks.

Mínimos cambios respecto a tu versión original:
- import defensivo y correcto de KafkaError (kafka.errors)
- normalización de TIMEFRAMES a minúsculas (evita problemas con ccxt/ws)
- todo lo demás preservado para compatibilidad con tus requirements
"""
from __future__ import annotations

import os
import json
import time
import uuid
import queue
import threading
import traceback
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple, List

# IMPORTS defensivos: kafka-python es la librería esperada en tu requirements
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except Exception as e:
    raise ImportError(
        "kafka client import failed. Make sure 'kafka-python' is installed in the environment.\n"
        "Install with: pip install kafka-python\n"
        f"Original import error: {e}"
    )

import ccxt
import websockets

# ---------------- CONFIG (env vars) ----------------
BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC', 'prices')
EXCHANGES = [e.strip().lower() for e in os.getenv('EXCHANGES', 'binance').split(',') if e.strip()]
SYMBOLS = [s.strip().upper() for s in os.getenv('SYMBOLS', 'BTC/USDT').split(',') if s.strip()]
# Normalizamos timeframes a minúsculas para compatibilidad con ccxt/ws
TIMEFRAMES = [t.strip().lower() for t in os.getenv('TIMEFRAMES', '1m,30m,4h,1d').split(',') if t.strip()]

FETCH_INTERVAL = float(os.getenv('FETCH_INTERVAL', 60))  # seconds for REST polling
PRODUCER_MODE = os.getenv('PRODUCER_MODE', 'ALL').upper()  # HISTORICAL, REST, WS, ALL

# local queue/backpressure
LOCAL_QUEUE_MAXSIZE = int(os.getenv('LOCAL_QUEUE_MAXSIZE', 30000))  # cap memory usage
ENQUEUE_TIMEOUT = float(os.getenv('ENQUEUE_TIMEOUT', 5.0))  # seconds to block when queue full

# sender / DLQ
SENDER_BATCH_SIZE = int(os.getenv('SENDER_BATCH_SIZE', 200))
SENDER_FLUSH_INTERVAL = float(os.getenv('SENDER_FLUSH_INTERVAL', 1.0))  # seconds
SEND_GET_TIMEOUT = float(os.getenv('SEND_GET_TIMEOUT', 10.0))  # wait for ack from kafka
MAX_SEND_ATTEMPTS = int(os.getenv('MAX_SEND_ATTEMPTS', 5))
DLQ_FILE = os.getenv('DLQ_FILE', '/tmp/producer_dlq.jsonl')

# websocket reconnection/backoff
WS_BASE_BACKOFF = float(os.getenv('WS_BASE_BACKOFF', 1.0))
WS_MAX_BACKOFF = float(os.getenv('WS_MAX_BACKOFF', 60.0))
WS_STALL_THRESHOLD = float(os.getenv('WS_STALL_THRESHOLD', 120.0))  # seconds without message -> restart handler

# kafka producer tuning
KAFKA_ACKS = os.getenv('KAFKA_ACKS', 'all')  # 'all' for durability
KAFKA_RETRIES = int(os.getenv('KAFKA_RETRIES', 5))
KAFKA_LINGER_MS = int(os.getenv('KAFKA_LINGER_MS', 10))
KAFKA_COMPRESSION = os.getenv('KAFKA_COMPRESSION', None)  # e.g., 'gzip' or None

# debug
VERBOSE = os.getenv('VERBOSE', 'false').lower() in ('1', 'true', 'yes')

# ---------------- Producer instance (Kafka) ----------------
producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
    key_serializer=lambda k: k if isinstance(k, (bytes, bytearray)) else str(k).encode('utf-8'),
    acks=KAFKA_ACKS,
    retries=KAFKA_RETRIES,
    linger_ms=KAFKA_LINGER_MS,
    compression_type=(KAFKA_COMPRESSION or None),
)

# ---------------- Local bounded queue ----------------
local_queue: "queue.Queue[Dict[str,Any]]" = queue.Queue(maxsize=LOCAL_QUEUE_MAXSIZE)

# ---------------- Helper functions ----------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def epoch_ms_from_ts_like(ts_like) -> int:
    # Accept datetime, iso string or epoch ms/seconds
    if isinstance(ts_like, (int, float)):
        # Heuristic: if > 1e12 assume ms, else seconds
        num = float(ts_like)
        if num > 1e12:
            return int(num)
        elif num > 1e9:
            # seconds with decimals? treat as seconds
            return int(num * 1000)
        else:
            return int(num * 1000)
    if isinstance(ts_like, str):
        try:
            # try ISO
            if ts_like.endswith("Z"):
                ts_like = ts_like.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts_like)
            return int(dt.timestamp() * 1000)
        except Exception:
            # fallback numeric string
            num = float(ts_like)
            return int(num)
    if isinstance(ts_like, datetime):
        dt = ts_like if ts_like.tzinfo else ts_like.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    raise ValueError("Invalid ts_like")

def make_kafka_key(exchange: str, symbol: str, timeframe: str) -> bytes:
    # normalized to lower exchange and symbol without slash uppercase for consumer consistency
    exch = exchange.lower()
    sym = symbol.replace('/', '').upper()
    tf = timeframe
    return f"{exch}|{sym}|{tf}".encode('utf-8')

def normalize_symbol_for_msg(symbol: str) -> str:
    return symbol.replace('/', '').upper()

def safe_enqueue(msg: Dict[str,Any]) -> bool:
    """
    Put message into local_queue WITH backpressure.
    Returns True if enqueued, False if timed out (queue full).
    """
    try:
        local_queue.put(msg, timeout=ENQUEUE_TIMEOUT)
        if VERBOSE:
            print("Enqueued message", msg.get("exchange"), msg.get("symbol"), msg.get("timeframe"), msg.get("ts_ms"))
        return True
    except queue.Full:
        print("Local queue full: enqueue timed out, dropping or handling backpressure upstream")
        return False

def dlq_dump(msg: Dict[str,Any], err: Optional[str]=None) -> None:
    rec = {
        "ts_dlq": now_iso(),
        "err": err,
        "message": msg
    }
    try:
        with open(DLQ_FILE, "a", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        print("Failed to write DLQ file:", e)

# ---------------- Sender thread (drains local_queue -> Kafka) ----------------
class SenderThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        print("SenderThread started")
        batch: List[Dict[str,Any]] = []
        last_flush = time.time()
        while not self._stop.is_set():
            try:
                # collect up to SENDER_BATCH_SIZE or until SENDER_FLUSH_INTERVAL elapsed
                try:
                    item = local_queue.get(timeout=SENDER_FLUSH_INTERVAL)
                    batch.append(item)
                except queue.Empty:
                    pass

                # also try to drain immediate further items without blocking
                while len(batch) < SENDER_BATCH_SIZE:
                    try:
                        item = local_queue.get_nowait()
                        batch.append(item)
                    except queue.Empty:
                        break

                # If nothing to do continue
                if not batch:
                    continue

                # Send batch items one by one with confirmation (get)
                for msg in batch:
                    # send with retries/backoff per message
                    attempts = 0
                    success = False
                    last_err = None
                    while attempts < MAX_SEND_ATTEMPTS and not success:
                        attempts += 1
                        try:
                            key = msg.get("_kafka_key")
                            # if key absent compute it
                            if key is None:
                                key = make_kafka_key(msg.get("exchange", ""), msg.get("symbol", ""), msg.get("timeframe", ""))
                            fut = producer.send(TOPIC, value=msg, key=key)
                            # block until ack or timeout - ensures we know the write outcome
                            fut.get(timeout=SEND_GET_TIMEOUT)
                            success = True
                            if VERBOSE:
                                print(f"Sent msg key={key} ts={msg.get('ts_ms')}")
                        except Exception as e:
                            last_err = str(e)
                            backoff = min(WS_MAX_BACKOFF, 0.5 * (2 ** (attempts - 1)))
                            print(f"Producer send attempt {attempts} failed: {e}; backoff {backoff}s")
                            time.sleep(backoff)
                    if not success:
                        print("Max send attempts reached for message; dumping to DLQ")
                        dlq_dump(msg, last_err)

                # clear batch
                batch.clear()
            except Exception as ex:
                print("SenderThread unhandled exception:", traceback.format_exc())
                time.sleep(1.0)

        # on stop flush remaining
        try:
            print("SenderThread flushing producer before exit...")
            producer.flush(timeout=10)
        except Exception:
            print("Producer flush error on SenderThread stop")
        print("SenderThread stopped")

# ---------------- REST / HIST publishing helpers ----------------
# Build a canonical message body
def build_message(exchange: str, symbol: str, timeframe: str, ts_ms: int, open_v, high_v, low_v, close_v, volume_v) -> Dict[str,Any]:
    msg = {
        "msg_id": str(uuid.uuid4()),
        "exchange": exchange,
        "symbol": normalize_symbol_for_msg(symbol),
        "timeframe": timeframe,
        "ts_ms": int(ts_ms),  # epoch ms
        "ts_iso": datetime.utcfromtimestamp(ts_ms/1000.0).isoformat() + "Z",
        "ts_ingest": now_iso(),
        "open": float(open_v),
        "high": float(high_v),
        "low": float(low_v),
        "close": float(close_v),
        "volume": float(volume_v)
    }
    # attach kafka key for convenience (consumer expects 'exchange' and 'symbol' normalized)
    msg["_kafka_key"] = make_kafka_key(exchange, symbol, timeframe)
    return msg

# historical & polling as before, but using safe_enqueue
exchange_instances: Dict[str, Any] = {}
for name in EXCHANGES:
    if hasattr(ccxt, name):
        try:
            exchange_instances[name] = getattr(ccxt, name)({'enableRateLimit': True})
        except Exception as e:
            print(f"Failed to init REST client for {name}: {e}")
    else:
        print(f"ccxt does not have exchange '{name}' for REST — skipping REST for it")

# last_ts tracking for historical pulls
last_ts: Dict[str, int] = {}

def fetch_historical_and_publish():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    limit = 500
    for exch_name, exch in exchange_instances.items():
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                key = f"{exch_name}_{symbol}_{tf}"
                start_ts = last_ts.get(key, None)
                if start_ts is None:
                    start_ts = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
                try:
                    ohlcv = exch.fetch_ohlcv(symbol, timeframe=tf, since=start_ts, limit=limit)
                    if not ohlcv:
                        continue
                    for candle in ohlcv:
                        ts_ms, open_, high, low, close, volume = candle[:6]
                        if ts_ms >= now_ms:
                            break
                        msg = build_message(exch_name, symbol, tf, ts_ms, open_, high, low, close, volume)
                        ok = safe_enqueue(msg)
                        if not ok:
                            # queue full -> optionally pause or break to apply backpressure
                            print("Queue full during historical ingestion; pausing historical loop briefly")
                            time.sleep(1.0)
                    last_ts[key] = ohlcv[-1][0] + 1
                except Exception as e:
                    print(f"Error fetching historical {symbol} {tf} from {exch_name}: {e}")

def fetch_and_publish():
    for exch_name, exch in exchange_instances.items():
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                try:
                    ohlcv = exch.fetch_ohlcv(symbol, timeframe=tf, limit=1)
                    if not ohlcv:
                        continue
                    ts_ms, open_, high, low, close, volume = ohlcv[0][:6]
                    msg = build_message(exch_name, symbol, tf, ts_ms, open_, high, low, close, volume)
                    ok = safe_enqueue(msg)
                    if not ok:
                        # queue full -> skip/pause
                        print("Queue full during polling; skipping this publish")
                except Exception as e:
                    print(f"Error fetching live {symbol} {tf} from {exch_name}: {e}")

# ---------------- WebSocket handlers (with monitor) ----------------
# track last_msg timestamp per handler to detect stalls
_ws_last_msg_ts: Dict[str, float] = {}
_ws_tasks: List[asyncio.Task] = []

async def binance_handler(symbol: str, timeframe: str):
    # ensure timeframe normalized for Binance streams (binance expects e.g. '1m', '1h', '1d')
    tf = timeframe.lower()
    stream_symbol = symbol.replace('/', '').lower()
    stream = f"{stream_symbol}@kline_{tf}"
    url = f"wss://stream.binance.com:9443/ws/{stream}"
    backoff = WS_BASE_BACKOFF
    handler_name = f"binance:{symbol}:{timeframe}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                print(f"[WS] Connected Binance {stream}")
                backoff = WS_BASE_BACKOFF
                async for raw in ws:
                    _ws_last_msg_ts[handler_name] = time.time()
                    try:
                        data = json.loads(raw)
                        k = data.get('k')
                        if not k:
                            continue
                        ts_ms = int(k['t'])
                        msg = build_message('binance', symbol, timeframe, ts_ms, k['o'], k['h'], k['l'], k['c'], k['v'])
                        ok = safe_enqueue(msg)
                        if not ok:
                            # If queue is full we may want to pause or sleep briefly
                            print("Queue full in binance_handler; sleeping briefly")
                            await asyncio.sleep(0.5)
                    except Exception as e:
                        print("Error parsing Binance message:", e)
                        continue
        except Exception as e:
            print(f"[WS] Binance handler error {symbol} {timeframe}: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, WS_MAX_BACKOFF)
            print(f"[WS] Reconnecting Binance in {backoff}s...")

async def coinbase_handler(symbol: str, timeframe: str):
    url = "wss://ws-feed.exchange.coinbase.com"
    tf_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
    granularity = tf_map.get(timeframe.lower())
    if granularity is None:
        print(f"Coinbase: timeframe {timeframe} not supported")
        return
    product_id = symbol.replace('/', '-')
    payload = {
        "type": "subscribe",
        "channels": [{"name": "candles", "product_ids": [product_id], "granularity": granularity}]
    }
    backoff = WS_BASE_BACKOFF
    handler_name = f"coinbase:{symbol}:{timeframe}"
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                print(f"[WS] Connected Coinbase {product_id} gran={granularity}")
                await ws.send(json.dumps(payload))
                backoff = WS_BASE_BACKOFF
                async for raw in ws:
                    _ws_last_msg_ts[handler_name] = time.time()
                    try:
                        data = json.loads(raw)
                        if data.get("type") == "candles":
                            for c in data.get("candles", []):
                                ts_s = int(c[0])
                                ts_ms = ts_s * 1000
                                # Attempt to map candle fields robustly
                                try:
                                    open_v = float(c[3]); high_v = float(c[2]); low_v = float(c[1]); close_v = float(c[4]); vol_v = float(c[5])
                                except Exception:
                                    try:
                                        open_v = float(c[1]); high_v = float(c[2]); low_v = float(c[3]); close_v = float(c[4]); vol_v = float(c[5])
                                    except Exception:
                                        continue
                                msg = build_message('coinbase', symbol, timeframe, ts_ms, open_v, high_v, low_v, close_v, vol_v)
                                ok = safe_enqueue(msg)
                                if not ok:
                                    await asyncio.sleep(0.5)
                    except Exception as e:
                        print("Coinbase parse error:", e)
                        continue
        except Exception as e:
            print(f"[WS] Coinbase handler error {product_id} gran={granularity}: {e}")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, WS_MAX_BACKOFF)
            print(f"[WS] Reconnecting Coinbase in {backoff}s...")

# monitor to restart stalled handlers (for extra robustness)
async def ws_monitor_task():
    while True:
        now = time.time()
        for k, last in list(_ws_last_msg_ts.items()):
            if now - last > WS_STALL_THRESHOLD:
                print(f"[WS monitor] Handler {k} stale for {now-last:.1f}s; you may want to restart it (handled by reconnect logic).")
        await asyncio.sleep(max(1.0, WS_STALL_THRESHOLD/4.0))

# run websockets tasks
async def run_websockets():
    tasks = []
    for exch in EXCHANGES:
        if exch == 'binance':
            for s in SYMBOLS:
                for tf in TIMEFRAMES:
                    tasks.append(asyncio.create_task(binance_handler(s, tf)))
        elif exch == 'coinbase':
            for s in SYMBOLS:
                for tf in TIMEFRAMES:
                    tasks.append(asyncio.create_task(coinbase_handler(s, tf)))
        else:
            print(f"No WS handler for {exch}")
    tasks.append(asyncio.create_task(ws_monitor_task()))
    if tasks:
        await asyncio.gather(*tasks)

# ---------------- Entrypoint ----------------
def start_thread(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t

def main():
    # start sender thread to drain queue -> kafka
    sender = SenderThread()
    sender.start()

    threads = []
    try:
        # Historical loop thread
        if PRODUCER_MODE in ('HISTORICAL', 'ALL'):
            def hist_loop():
                while True:
                    try:
                        fetch_historical_and_publish()
                    except Exception as e:
                        print("Historical loop error:", e)
                    time.sleep(FETCH_INTERVAL)
            threads.append(start_thread(hist_loop))

        # REST polling loop
        if PRODUCER_MODE in ('REST', 'ALL'):
            def rest_loop():
                while True:
                    try:
                        fetch_and_publish()
                    except Exception as e:
                        print("Rest loop error:", e)
                    time.sleep(FETCH_INTERVAL)
            threads.append(start_thread(rest_loop))

        # Websockets (asyncio) - runs in main thread to use event loop
        if PRODUCER_MODE in ('WS', 'ALL'):
            asyncio.run(run_websockets())

        # join background threads if only HISTORICAL/REST mode
        if PRODUCER_MODE in ('HISTORICAL', 'REST'):
            for t in threads:
                t.join()
    except KeyboardInterrupt:
        print("Producer stopped by user")
    except Exception as e:
        print("Producer unhandled exception:", e)
    finally:
        # graceful shutdown
        print("Shutting down producer...")
        sender.stop()
        sender.join(timeout=5.0)
        try:
            producer.flush(timeout=10)
            producer.close(timeout=10)
        except Exception:
            print("Producer flush/close error")

if __name__ == "__main__":
    main()
