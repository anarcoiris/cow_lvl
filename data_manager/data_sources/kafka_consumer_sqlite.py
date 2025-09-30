"""Advanced Consumer: Kafka -> SQLite with batching, CSV export, and graceful shutdown."""
import asyncio
import json
import signal
import os
import time
from datetime import datetime
from typing import Dict
from aiokafka import AIOKafkaConsumer
import pandas as pd

from ..data_manager import DataManager

# Carpeta exports relativa a data_manager
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # data_manager
EXPORTS_DIR = os.path.join(BASE_DIR, 'exports')
DB_PATH = os.path.join(EXPORTS_DIR, 'marketdata.db')

async def consume_and_store(
    topic: str = 'prices',
    bootstrap_servers: str = 'localhost:9092',
    group_id: str = 'sqlite_saver',
    db: str = DB_PATH,
    replica: str = None,
    nan_strategy: str = 'drop',
    batch_size: int = 200,
    batch_timeout: float = 2.0
):
    os.makedirs(EXPORTS_DIR, exist_ok=True)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler():
        print('Received stop signal, finishing current batch...')
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _signal_handler())

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset='earliest',
        session_timeout_ms=30000,
        heartbeat_interval_ms=10000
    )
    await consumer.start()

    dm = DataManager(db, replica_db_path=replica, nan_strategy=nan_strategy)
    await dm.init_db()

    processed_stats_path = os.path.join(EXPORTS_DIR, 'processed_stats.json')
    stats: Dict[str, int] = {}
    if os.path.exists(processed_stats_path):
        try:
            with open(processed_stats_path, 'r') as f:
                stats = json.load(f)
        except Exception:
            stats = {}

    batch = []
    last_flush = time.time()
    flush_lock = asyncio.Lock()

    async def flush_batch():
        nonlocal batch, stats, last_flush
        async with flush_lock:
            if not batch:
                return
            try:
                grouped = {}
                for item in batch:
                    key = (item['symbol'], item['timeframe'])
                    grouped.setdefault(key, []).append([item['ts'], item['o'], item['h'], item['l'], item['c'], item['v']])

                total_inserted = 0
                for (symbol, timeframe), rows in grouped.items():
                    inserted = await dm.insert_ohlcv(symbol, timeframe, rows, source='kafka')
                    total_inserted += inserted
                    stats_key = f"{symbol}|{timeframe}"
                    stats[stats_key] = stats.get(stats_key, 0) + inserted

                df = pd.DataFrame(batch)
                df = df[['symbol','timeframe','ts','o','h','l','c','v','source']]
                date_str = datetime.utcnow().strftime('%Y-%m-%d')
                csv_path = os.path.join(EXPORTS_DIR, f"ohlcv_{date_str}.csv")
                header = not os.path.exists(csv_path)
                df.to_csv(csv_path, mode='a', index=False, header=header)

                await consumer.commit()
                last_flush = time.time()
                batch = []
                with open(processed_stats_path, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"Flushed {total_inserted} rows to DB and CSV (batch size {len(df)}).")
            except Exception as e:
                print("Error flushing batch:", e)

    try:
        async for msg in consumer:
            payload = None
            try:
                payload = json.loads(msg.value.decode())
                ts = int(payload['ts'])
                o = float(payload['o'])
                h = float(payload['h'])
                l = float(payload['l'])
                c = float(payload['c'])
                v = float(payload['v'])
                item = {
                    'symbol': payload.get('symbol', 'BTC/USDT'),
                    'timeframe': payload.get('timeframe', '1m'),
                    'ts': ts,
                    'o': o,
                    'h': h,
                    'l': l,
                    'c': c,
                    'v': v,
                    'source': payload.get('source', 'kafka')
                }
                batch.append(item)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                print(f'Malformed or incomplete message, skipping: {payload} | Error: {e}')
                continue

            if len(batch) >= batch_size or (time.time() - last_flush) >= batch_timeout:
                await flush_batch()

            if stop_event.is_set():
                await flush_batch()
                break
    finally:
        try:
            await flush_batch()
        except Exception as e:
            print("Error during final flush:", e)
        await consumer.stop()
        await dm.close()
        print("Consumer stopped cleanly.")
