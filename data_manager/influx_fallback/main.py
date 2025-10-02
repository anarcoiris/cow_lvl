import subprocess
import time
import socket
import os

def wait_for_service(host, port, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"{host}:{port} está listo.")
                return True
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"No se pudo conectar a {host}:{port} en {timeout}s")

if __name__ == "__main__":
    # Espera Kafka e InfluxDB
    wait_for_service("kafka", 9092)
    wait_for_service("influxdb", 8086)

    # Ejecuta procesos en paralelo
    procs = [
        subprocess.Popen(["python", "producer.py"]),
        subprocess.Popen(["python", "consumer.py"])
        # subprocess.Popen(["python", "evolves3.py"])
    ]

    # Esperar que terminen (o que uno falle)
    for p in procs:
        p.wait()
