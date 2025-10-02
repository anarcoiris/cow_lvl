import os
import json
import time
from datetime import datetime
from kafka import KafkaProducer
import ccxt

BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC', 'prices')
EXCHANGES = os.getenv('EXCHANGES', 'binance').split(',')
SYMBOLS = os.getenv('SYMBOLS', 'BTC/USDT').split(',')
FETCH_INTERVAL = int(os.getenv('FETCH_INTERVAL', 10))

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Crear instancias CCXT
exchange_instances = {
    name.strip(): getattr(ccxt, name.strip())({'enableRateLimit': True})
    for name in EXCHANGES
}

def fetch_and_publish():
    for exch_name, exch in exchange_instances.items():
        for symbol in SYMBOLS:
            try:
                ohlcv = exch.fetch_ohlcv(symbol, timeframe='1m', limit=1)
                if not ohlcv:
                    continue
                ts_ms, open_, high, low, close, *_ = ohlcv[0]
                msg = {
                    'exchange': exch_name,
                    'symbol': symbol.replace('/', ''),
                    'ts': datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + 'Z',
                    'open': open_,
                    'high': high,
                    'low': low,
                    'close': close
                }
                producer.send(TOPIC, msg)
                print(f"Produced: {msg}")
            except Exception as e:
                print(f"Error fetching {symbol} from {exch_name}: {e}")
    producer.flush()

if __name__ == '__main__':
    try:
        while True:
            fetch_and_publish()
            time.sleep(FETCH_INTERVAL)
    except KeyboardInterrupt:
        print("Producer stopped")
