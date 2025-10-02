import os
import json
import time
from datetime import datetime, timezone, timedelta
from kafka import KafkaProducer
import ccxt

BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC', 'prices')
EXCHANGES = os.getenv('EXCHANGES', 'binance').split(',')
SYMBOLS = os.getenv('SYMBOLS', 'BTC/USDT').split(',')
TIMEFRAMES = os.getenv('TIMEFRAMES', '1m,30m,4h,1d').split(',')
FETCH_INTERVAL = int(os.getenv('FETCH_INTERVAL', 60))  # cada minuto

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

exchange_instances = {
    name.strip(): getattr(ccxt, name.strip())({'enableRateLimit': True})
    for name in EXCHANGES
}

# Guarda el último timestamp histórico bajado para cada tupla exch-symbol-tf
last_ts = {}

def fetch_historical_and_publish():
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    limit = 500  # máximo permitido por la mayoría de APIs
    for exch_name, exch in exchange_instances.items():
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                key = f"{exch_name}_{symbol}_{tf}"
                start_ts = last_ts.get(key, None)

                # Si no hay start_ts, inicializa a hace 30 días (puedes cambiar)
                if start_ts is None:
                    start_ts = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)

                try:
                    # Descarga OHLCV desde start_ts en adelante
                    ohlcv = exch.fetch_ohlcv(symbol, timeframe=tf, since=start_ts, limit=limit)
                    if not ohlcv:
                        print(f"No more historical data for {key}")
                        continue

                    for candle in ohlcv:
                        ts_ms, open_, high, low, close, volume = candle
                        if ts_ms >= now_ms:
                            # Ya llegamos al tiempo actual, paramos histórico
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
                        producer.send(TOPIC, msg)
                        print(f"Produced historical: {msg}")

                    last_ts[key] = ohlcv[-1][0] + 1  # guarda el último timestamp + 1ms para la próxima

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
                    producer.send(TOPIC, msg)
                    print(f"Produced live: {msg}")
                except Exception as e:
                    print(f"Error fetching live {symbol} {tf} from {exch_name}: {e}")
    producer.flush()


if __name__ == '__main__':
    try:
        # Primero descarga histórico (poco a poco en cada iteración)
        while True:
            fetch_historical_and_publish()
            time.sleep(FETCH_INTERVAL)
            # Luego, una vez que termine histórico, puedes ejecutar fetch_and_publish()
            # o incluir una lógica para alternar entre histórico y en vivo.
            # Aquí simple ejemplo: solo histórico.
    except KeyboardInterrupt:
        print("Producer stopped")
