import os
import time
from influxdb_client import InfluxDBClient, DeletePredicateRequest
from datetime import datetime, timezone, timedelta

# Configuration (set these as environment variables or hardcode)
INFLUX_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", "J4twWiQSH6QZF33ZyAB9NJLoNMyrHjOlvY6UJGgczJfk-_DC3d5BFEiZzQOYC39ObPYwxF5kZTAZtzIX-Xr40Q==")
INFLUX_ORG = os.getenv("INFLUXDB_ORG", "BreOrganization")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "prices")
DAYS_TO_KEEP = int(os.getenv("DAYS_TO_KEEP", "3600"))  # Delete data older than this

def delete_old_data():
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        delete_api = client.delete_api()

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=DAYS_TO_KEEP)

        start = "1970-01-01T00:00:00Z"  # Desde el inicio epoch
        stop = cutoff.isoformat(timespec='seconds').replace('+00:00', 'Z')  # formato ISO 8601 UTC

        predicate = '_measurement="prices"'

        print(f"Deleting data older than {DAYS_TO_KEEP} days from bucket '{INFLUX_BUCKET}'...")
        delete_api.delete(start=start, stop=stop, predicate=predicate, bucket=INFLUX_BUCKET, org=INFLUX_ORG)
        print("Deletion completed.")

if __name__ == "__main__":
    delete_old_data()
