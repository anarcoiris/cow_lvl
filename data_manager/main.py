"""CLI demo para trading_data_manager v3."""
import argparse
import sys

# Import relativo dentro del paquete para evitar colisiones con paquetes instalados llamados `examples`
from .examples.example_kafka_producer import run_producer
from .examples.example_kafka_consumer import run_consumer

def main():
    parser = argparse.ArgumentParser(prog="data_manager.main")
    sub = parser.add_subparsers(dest='cmd')

    p_prod = sub.add_parser('prod', help='Run Kafka producer (websocket -> kafka)')
    p_prod.add_argument('--symbol', default='btcusdt', help='symbol for producer (e.g. btcusdt)')
    p_prod.add_argument('--timeframe', default='1m', help='timeframe for kline stream (e.g. 1m)')
    p_prod.add_argument('--bootstrap', default='localhost:9092', help='kafka bootstrap servers')
    p_prod.add_argument('--topic', default='prices', help='kafka topic')

    p_cons = sub.add_parser('cons', help='Run Kafka consumer (kafka -> sqlite)')
    p_cons.add_argument('--db', default='data_manager/exports/marketdata.db', help='sqlite db path')
    p_cons.add_argument('--replica', default=None, help='replica sqlite path (optional)')
    p_cons.add_argument('--bootstrap', default='localhost:9092', help='kafka bootstrap servers')
    p_cons.add_argument('--topic', default='prices', help='kafka topic')
    p_cons.add_argument('--group', default='sqlite_saver', help='kafka consumer group id')
    p_cons.add_argument('--nan-strategy', default='drop', choices=['drop','null','fill_forward','zero'])

    args = parser.parse_args()

    if args.cmd == 'prod':
        # run_producer is synchronous entrypoint (starts asyncio internally)
        run_producer(symbol=args.symbol, timeframe=args.timeframe, bootstrap=args.bootstrap, topic=args.topic)
    elif args.cmd == 'cons':
    	run_consumer(topic=args.topic, bootstrap=args.bootstrap, group=args.group,
                    db=args.db, replica=args.replica, nan_strategy=args.nan_strategy)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
