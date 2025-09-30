"""DataManager, pensado para integrarse con Kafka consumer o standalone como modulo.

Archivo: proyecto/data_manager/manager.py

Notas principales del rediseño:
- Se mantiene la compatibilidad con la API existente (insert_ohlcv, fetch_ohlcv, to_dataframe, get_last_ts, init_db, close).
- Se añade un nuevo método `upsert_ohlcv` pensado para flujos en tiempo real (websockets). Mantiene la semántica de "actualizar si ya existe".
- Para permitir `ON CONFLICT ... DO UPDATE` se crea (si no existe) un índice único `idx_ohlcv_unique` sobre (symbol, timeframe, ts). Esto es no destructivo y no altera datos existentes.
- `_exec_many` se hace robusto usando `await conn.executemany(...)` y commit explícito.
- `insert_ohlcv` conserva el comportamiento original (INSERT OR IGNORE) para no cambiar la lógica de otros scripts que dependan de él.
- Se mantiene la lógica de replicación a `replica_db_path` y el _push_to_influx en hilo.

Asegúrate de tener backup de tu .db antes de probar (siempre recomendable).
"""

import os
import asyncio
import time
import math
from typing import List, Optional, Sequence, Tuple, Any, Dict, Callable
import aiosqlite
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from .utils import ts_to_datetime

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.5  # seconds


class DataManager:
    def __init__(self, db_path: str = './exports/marketdata_main.db', replica_db_path: Optional[str] = 'exports/marketdata_replica.db',
                 nan_strategy: str = 'drop', influx_config: Optional[dict] = None,
                 max_workers: int = 2):
        self.db_path = db_path
        self.replica_db_path = replica_db_path
        self.nan_strategy = nan_strategy
        self.influx_config = influx_config or None
        self._conn: Optional[aiosqlite.Connection] = None
        self._replica_conn: Optional[aiosqlite.Connection] = None
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def init_db(self):
        """Inicializa conexiones y, si existe, aplica schema.sql.
        Además asegura la existencia de un índice único (symbol, timeframe, ts)
        para poder usar upserts sin reescribir la tabla existente.
        """
        os.makedirs(os.path.dirname(self.db_path) or '.', exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        here = os.path.dirname(__file__)
        schema_path = os.path.join(here, 'schema.sql')
        if os.path.exists(schema_path):
            # aplicar schema si existe
            with open(schema_path, 'r', encoding='utf-8') as f:
                await self._conn.executescript(f.read())
            await self._conn.commit()

        # Crear indice único no destructivo para soportar upserts
        try:
            await self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique ON ohlcv(symbol, timeframe, ts);"
            )
            await self._conn.commit()
        except Exception:
            # Si la tabla no existe o no tiene columnas esperadas, no fallamos aquí,
            # porque no queremos romper la inicialización si el proyecto usa un schema distinto.
            pass

        if self.replica_db_path:
            os.makedirs(os.path.dirname(self.replica_db_path) or '.', exist_ok=True)
            self._replica_conn = await aiosqlite.connect(self.replica_db_path)
            self._replica_conn.row_factory = aiosqlite.Row
            if os.path.exists(schema_path):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    await self._replica_conn.executescript(f.read())
                await self._replica_conn.commit()
            try:
                await self._replica_conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique ON ohlcv(symbol, timeframe, ts);"
                )
                await self._replica_conn.commit()
            except Exception:
                pass

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
        if self._replica_conn:
            await self._replica_conn.close()
            self._replica_conn = None
        self._executor.shutdown(wait=False)

    async def _exec_many(self, conn: aiosqlite.Connection, sql: str, params: Sequence[Sequence[Any]]):
        """Ejecuta many + commit de forma robusta."""
        if not params:
            return
        await conn.executemany(sql, params)
        await conn.commit()

    async def insert_ohlcv(self, symbol: str, timeframe: str, rows: Sequence[Sequence[Any]], source: str = 'unknown'):
        """Inserción masiva conservadora: mantiene comportamiento preexistente (INSERT OR IGNORE).
        No actualiza filas existentes: si quieres actualización usa upsert_ohlcv.
        """
        if not self._conn:
            await self.init_db()

        params = []
        for r in rows:
            if len(r) < 6:
                continue
            ts = int(r[0])
            o = None if r[1] is None else float(r[1])
            h = None if r[2] is None else float(r[2])
            l = None if r[3] is None else float(r[3])
            c = None if r[4] is None else float(r[4])
            v = None if r[5] is None else float(r[5])
            # conservación de la semántica previa: si nan_strategy == 'drop' se saltan filas
            if self.nan_strategy == 'drop' and (o is None or h is None or l is None or c is None or v is None):
                continue
            params.append((symbol, timeframe, ts, o, h, l, c, v, source))

        if not params:
            return 0

        sql = """
        INSERT OR IGNORE INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self._exec_many(self._conn, sql, params)
        if self._replica_conn:
            try:
                await self._exec_many(self._replica_conn, sql, params)
            except Exception:
                # registramos meta_sync pero no rompemos el flujo principal
                try:
                    await self._conn.execute("INSERT INTO meta_sync (symbol,timeframe,last_ts,last_sync) VALUES (?,?,?,?)", (symbol, timeframe, None, int(time.time()*1000)))
                    await self._conn.commit()
                except Exception:
                    pass

        # Empujar a Influx de forma best-effort en un hilo
        if self.influx_config:
            asyncio.get_running_loop().run_in_executor(self._executor, self._push_to_influx_sync, symbol, timeframe, rows, self.influx_config)

        return len(params)

    async def upsert_ohlcv(self, symbol: str, timeframe: str, rows: Sequence[Sequence[Any]], source: str = 'unknown'):
        """Upsert (insert or update) de OHLCV. Pensado para flujos en tiempo real (websockets).

        rows expected: sequence of [ts, o, h, l, c, v]
        """
        if not self._conn:
            await self.init_db()

        params = []
        for r in rows:
            if len(r) < 6:
                continue
            ts = int(r[0])
            o = None if r[1] is None else float(r[1])
            h = None if r[2] is None else float(r[2])
            l = None if r[3] is None else float(r[3])
            c = None if r[4] is None else float(r[4])
            v = None if r[5] is None else float(r[5])
            if self.nan_strategy == 'drop' and (o is None or h is None or l is None or c is None):
                # se permite volumen None si procede
                continue
            params.append((symbol, timeframe, ts, o, h, l, c, v, source))

        if not params:
            return 0

        # Usamos ON CONFLICT ... DO UPDATE asumiendo que el índice único idx_ohlcv_unique existe
        sql = """
        INSERT INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            source=excluded.source
        ;
        """
        # Ejecutar primero en DB primaria
        await self._exec_many(self._conn, sql, params)

        # Intentar replicar en replica si está configurada (no crítico)
        if self._replica_conn:
            try:
                await self._exec_many(self._replica_conn, sql, params)
            except Exception:
                try:
                    await self._conn.execute("INSERT INTO meta_sync (symbol,timeframe,last_ts,last_sync) VALUES (?,?,?,?)", (symbol, timeframe, None, int(time.time()*1000)))
                    await self._conn.commit()
                except Exception:
                    pass

        # push to influx best-effort
        if self.influx_config:
            asyncio.get_running_loop().run_in_executor(self._executor, self._push_to_influx_sync, symbol, timeframe, rows, self.influx_config)

        return len(params)

    def _push_to_influx_sync(self, symbol: str, timeframe: str, rows: Sequence[Sequence[Any]], influx_config: dict):
        try:
            from data_sources.influx_writer import write_to_influx
            write_to_influx(symbol, timeframe, rows, influx_config['token'], influx_config['org'], influx_config['bucket'], url=influx_config.get('url','http://localhost:8086'))
        except Exception:
            pass

    async def fetch_ohlcv(self, symbol: str, timeframe: str, start_ts: Optional[int] = None,
                         end_ts: Optional[int] = None, limit: Optional[int] = None):
        if not self._conn:
            await self.init_db()
        params: List[Any] = [symbol, timeframe]
        where = "WHERE symbol = ? AND timeframe = ?"
        if start_ts is not None:
            where += " AND ts >= ?"
            params.append(int(start_ts))
        if end_ts is not None:
            where += " AND ts <= ?"
            params.append(int(end_ts))
        order_limit = " ORDER BY ts ASC"
        if limit is not None:
            order_limit += f" LIMIT {int(limit)}"
        sql = f"SELECT ts, open, high, low, close, volume FROM ohlcv {where} {order_limit};"
        cur = await self._conn.execute(sql, params)
        rows = await cur.fetchall()
        await cur.close()
        return [tuple(row) for row in rows]

    async def to_dataframe(self, symbol: str, timeframe: str, start_ts: Optional[int] = None,
                           end_ts: Optional[int] = None, limit: Optional[int] = None):
        rows = await self.fetch_ohlcv(symbol, timeframe, start_ts, end_ts, limit)
        if not rows:
            return pd.DataFrame(columns=['open','high','low','close','volume'])
        df = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume'])
        df['datetime'] = df['ts'].apply(ts_to_datetime)
        df = df.set_index('datetime')
        df = df.drop(columns=['ts'])
        return df

    async def get_last_ts(self, symbol: str, timeframe: str):
        if not self._conn:
            await self.init_db()
        cur = await self._conn.execute("SELECT ts FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT 1", (symbol, timeframe))
        row = await cur.fetchone()
        await cur.close()
        if row:
            return int(row[0])
        return None
