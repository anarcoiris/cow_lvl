"""Ejemplo: ejecutar kafka_consumer_sqlite como módulo."""

import argparse
import asyncio

from ..data_sources.kafka_consumer_sqlite import consume_and_store

def run_consumer(topic: str = 'prices', bootstrap: str = 'localhost:9092', group: str = 'sqlite_saver',
                 db: str = 'market_data_v3.db', replica: str = None, nan_strategy: str = 'drop'):
    """Synchronous entrypoint for the consumer (calls asyncio.run)."""
    asyncio.run(consume_and_store(topic=topic, bootstrap_servers=bootstrap, group_id=group, db=db, replica=replica, nan_strategy=nan_strategy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='prices')
    parser.add_argument('--bootstrap', default='localhost:9092')
    parser.add_argument('--group', default='sqlite_saver')
    parser.add_argument('--db', default='market_data_v3.db')
    parser.add_argument('--replica', default=None)
    parser.add_argument('--nan-strategy', default='drop')
    args = parser.parse_args()
    run_consumer(topic=args.topic, bootstrap=args.bootstrap, group=args.group, db=args.db, replica=args.replica, nan_strategy=args.nan_strategy)
