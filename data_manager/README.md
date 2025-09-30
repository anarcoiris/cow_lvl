
# trading_data_manager (v3) — Kafka + SQLite + WebSocket integration

Esta versión integra Kafka como bus de eventos para desacoplar la ingestión (websocket) y el almacenamiento (SQLite/Inlfux).
Incluye:
- Producer: websocket -> Kafka (Binance kline stream -> topic `ohlcv_raw`).
- Consumer: Kafka -> SQLite (DataManager) con commit manual de offsets tras persistencia exitosa.
- Opcional: consumer adicional para escribir a InfluxDB.
- Docker Compose para levantar ZooKeeper + Kafka (entorno dev/testing).
- Estrategias robustas de NaN, replicación SQLite (primary + replica), retries y backoff.

## Arquitectura recomendada

WebSocket -> Kafka (topic: ohlcv_raw) -> Consumers:
 - consumer_sqlite (grupo: sqlite_saver) -> escribe en SQLite primary (+replica)
 - consumer_influx (grupo: influx_writer) -> escribe en Influx (opcional)

Kafka mantiene durabilidad y permite reprocesado si cambias la lógica.

## Requisitos / Instalación

Recomendado usar Docker Compose para Kafka en desarrollo:
```bash
docker-compose up -d
```

Instala dependencias Python:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución ejemplo local

1. Levanta Kafka (docker-compose).
2. Ejecuta el producer (websocket -> kafka):
   ```bash
   python -m data_sources.websocket_to_kafka --symbol btcusdt --timeframe 1m
   ```
3. Ejecuta el consumer que persiste en SQLite:
   ```bash
   python -m data_sources.kafka_consumer_sqlite --topic ohlcv_raw --group sqlite_saver --db market_data_v3.db
   ```

## Notas sobre consistencia y sincronicidad
- El consumer usa `enable_auto_commit=False` y hace `commit()` solo **tras** confirmar que la inserción en SQLite (y replica si está configurada) concluyó con éxito. Esto proporciona *at-least-once* con control manual de offsets.
- Para exactamente-once necesitarías transacciones distribuidas o un sistema de almacenamiento que soporte idempotencia por diseño (por ejemplo, aplicar deduplicado por `(symbol, timeframe, ts)` que ya está implementado con `UNIQUE` y `INSERT OR IGNORE`).
- SQLite puede ser el cuello de botella si Kafka produce a alta velocidad; en ese caso se recomienda un destino más escalable (Postgres/TimescaleDB).
