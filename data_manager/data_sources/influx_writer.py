
"""Escritor a InfluxDB 2.x (síncrono) para ser ejecutado en threadpool si es necesario."""
from influxdb_client import InfluxDBClient, Point, WritePrecision
from typing import Sequence, Any

def write_to_influx(symbol: str, timeframe: str, ohlcv_data: Sequence[Sequence[Any]], influx_token: str, org: str, bucket: str, url: str = "http://localhost:8086") -> None:
    client = InfluxDBClient(url=url, token=influx_token, org=org)
    write_api = client.write_api(write_options=WritePrecision.MS)
    points = []
    for row in ohlcv_data:
        ts, o, h, l, c, v = row
        p = Point("ohlcv")             .tag("symbol", symbol)             .tag("timeframe", timeframe)             .field("open", float(o) if o is not None else None)             .field("high", float(h) if h is not None else None)             .field("low", float(l) if l is not None else None)             .field("close", float(c) if c is not None else None)             .field("volume", float(v) if v is not None else None)             .time(int(ts), WritePrecision.MS)
        points.append(p)
    write_api.write(bucket=bucket, org=org, record=points)
    client.close()
