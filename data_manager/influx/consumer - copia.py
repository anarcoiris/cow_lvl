import os
import json
from kafka import KafkaConsumer
from influxdb_client import InfluxDBClient, Point, WritePrecision
from datetime import datetime

BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TOPIC = os.getenv('KAFKA_TOPIC', 'prices')

INFLUX_URL = os.getenv('INFLUXDB_URL', 'http://influxdb:8086')
INFLUX_TOKEN = os.getenv('INFLUXDB_TOKEN', '')
INFLUX_ORG = os.getenv('INFLUXDB_ORG', '')
INFLUX_BUCKET = os.getenv('INFLUXDB_BUCKET', '')

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='market-group'
)

with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
    write_api = client.write_api()
    print("Consumer started. Waiting for messages...")
for msg in consumer:
    data = msg.value
    try:
        ts = data.get("ts")  # puede venir como string ISO o epoch ms
        if isinstance(ts, str):
            # parsea ISO a datetime
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, (int, float)):
            # viene como epoch ms
            ts = datetime.utcfromtimestamp(ts / 1000.0)

        point = (
            Point("prices")
                .tag("exchange", data['exchange'])
                .tag("symbol", data['symbol'])
                .tag("timeframe", data.get("timeframe", "unknown"))
                .field("open", float(data['open']))
                .field("high", float(data['high']))
                .field("low", float(data['low']))
                .field("close", float(data['close']))
                .field("volume", float(data['volume']))
                .time(ts, WritePrecision.NS)
        )

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        print(f"Written to InfluxDB: {data}")
    except Exception as e:
        print(f"Error writing to InfluxDB: {e}")
