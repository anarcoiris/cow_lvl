"""Ejemplo: ejecutar websocket_to_kafka como módulo."""

import asyncio
import argparse

# Import relativo al paquete
from ..data_sources.websocket_to_kafka import binance_ws_to_kafka

def run_producer(exchange: str ='binance', symbol: str = 'btcusdt', timeframe: str = '30m', bootstrap: str = 'localhost:9092', topic: str = 'prices'):
    """Synchronous entrypoint for the producer (calls asyncio.run)."""
    asyncio.run(binance_ws_to_kafka(exchange=exchange, symbol=symbol, timeframe=timeframe, bootstrap_servers=bootstrap, topic=topic))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='btcusdt')
    parser.add_argument('--exchange', default='binance')
    parser.add_argument('--timeframe', default='1m')
    parser.add_argument('--bootstrap', default='localhost:9092')
    parser.add_argument('--topic', default='prices')
    args = parser.parse_args()
    run_producer(exchange=args.exchange,symbol=args.symbol, timeframe=args.timeframe, bootstrap=args.bootstrap, topic=args.topic)
