import os
import json
import time
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from kafka import KafkaProducer
import ccxt
import ccxt.pro as ccxtpro   # para websockets

# ---------------- CONFIG ----------------
BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC', 'prices')
EXCHANGES = os.getenv('EXCHANGES', 'binance').split(',')
SYMBOLS = os.getenv('SYMBOLS', 'BTC/USDT').split(',')
TIMEFRAMES = os.getenv('TIMEFRAMES', '1m,30m,4h,1d').split(',')
FETCH_INTERVAL = int(os.getenv('FETCH_INTERVAL', 60))  # cada minuto
MODE = os.getenv('PRODUCER_MODE', 'ALL').upper()       # HISTORICAL, REST, WS, ALL
# ----------------------------------------

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# REST clients
exchange_instances = {
    name.strip(): getattr(ccxt, name.strip())({'enableRateLimit': True})
    for name in EXCHANGES
}

# WS clients
ws_exchange_instances = {
    name.strip(): getattr(ccxtpro, name.strip())({'enableRateLimit': True})
    for name in EXCHANGES if hasattr(ccxtpro, name.strip())
}

# Guarda el último timestamp histórico bajado para cada tupla exch-symbol-tf
last_ts = {}

def publish_msg(msg):
    """Publica en Kafka y loggea en consola"""
    producer.send(TOPIC, msg)
    print(f"Produced: {msg}")

# ---------------- REST ----------------
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
                        ts_ms, open_, high, low, close, volume = candle
                        if ts_ms >= now_ms:
                            break
                        msg = {
                            'exchange': exch_name,
                            'symbol': symbol.replace('/', ''),
                            'timeframe': tf,
                            'ts_candle': datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + 'Z',
                            'ts_ingest': datetime.now(timezone.utc).isoformat(),
                            'open': open_,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume
                        }
                        publish_msg(msg)

                    last_ts[key] = ohlcv[-1][0] + 1

                except Exception as e:
                    print(f"Error fetching historical {symbol} {tf} from {exch_name}: {e}")

    producer.flush()


def fetch_and_publish():
    for exch_name, exch in exchange_instances.items():
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                try:
                    ohlcv = exch.fetch_ohlcv(symbol, timeframe=tf, limit=1)
                    if not ohlcv:
                        continue
                    ts_ms, open_, high, low, close, volume, *_ = ohlcv[0]

                    msg = {
                        'exchange': exch_name,
                        'symbol': symbol.replace('/', ''),
                        'timeframe': tf,
                        'ts_candle': datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + 'Z',
                        'ts_ingest': datetime.now(timezone.utc).isoformat(),
                        'open': open_,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume
                    }
                    publish_msg(msg)

                except Exception as e:
                    print(f"Error fetching live {symbol} {tf} from {exch_name}: {e}")

    producer.flush()

# ---------------- WS ----------------
async def stream_websocket(exchange_name, exch):
    """Escucha velas en tiempo real vía WS"""
    while True:
        try:
            for symbol in SYMBOLS:
                for tf in TIMEFRAMES:
                    ohlcv = await exch.watch_ohlcv(symbol, timeframe=tf)
                    ts_ms, open_, high, low, close, volume = ohlcv[-1]
                    msg = {
                        'exchange': exchange_name,
                        'symbol': symbol.replace('/', ''),
                        'timeframe': tf,
                        'ts_candle': datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + 'Z',
                        'ts_ingest': datetime.now(timezone.utc).isoformat(),
                        'open': open_,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume
                    }
                    publish_msg(msg)
        except Exception as e:
            print(f"WebSocket error on {exchange_name}: {e}")
            await asyncio.sleep(5)

async def run_websockets():
    tasks = []
    for exch_name, exch in ws_exchange_instances.items():
        tasks.append(asyncio.create_task(stream_websocket(exch_name, exch)))
    await asyncio.gather(*tasks)

# ---------------- MAIN ----------------
if __name__ == '__main__':
    try:
        threads = []

        # Histórico (si aplica)
        if MODE in ('HISTORICAL', 'REST', 'WS', 'ALL'):
            t_hist = threading.Thread(
                target=lambda: [fetch_historical_and_publish() or time.sleep(FETCH_INTERVAL) for _ in iter(int, 1)]
            )
            t_hist.daemon = True
            t_hist.start()
            threads.append(t_hist)

        # Polling REST (si aplica)
        if MODE in ('REST', 'ALL'):
            t_rest = threading.Thread(
                target=lambda: [fetch_and_publish() or time.sleep(FETCH_INTERVAL) for _ in iter(int, 1)]
            )
            t_rest.daemon = True
            t_rest.start()
            threads.append(t_rest)

        # Websockets (si aplica)
        if MODE in ('WS', 'ALL'):
            asyncio.run(run_websockets())

        # Si solo histórico o REST, esperar hilos
        if MODE in ('HISTORICAL', 'REST'):
            for t in threads:
                t.join()

    except KeyboardInterrupt:
        print("Producer stopped")
