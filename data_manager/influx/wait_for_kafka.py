import socket
import time
import os

host, port = os.getenv("KAFKA_BOOTSTRAP_SERVERS").split(":")

print(f"Esperando a Kafka en {host}:{port}...")

while True:
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            print("Kafka está disponible.")
            break
    except Exception:
        print("Kafka no está disponible. Reintentando en 2s...")
        time.sleep(2)
