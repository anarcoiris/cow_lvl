#!/usr/bin/env python3
"""
Archivo: proyecto/data_sources/influx_to_sql.py

Extrae OHLCV desde InfluxDB (Flux) y lo inserta en una base SQLite mediante DataManager.
Añade además un streamer WebSocket a Binance que hace upsert en la misma DB.

Uso:
    # sólo importar histórico desde Influx
    python influx_to_sql.py --mode import --symbol BTCUSDT --exchange binance --timeframe 30m --start '-90d' --bucket prices --sqlite '../data_manager/exports/influx29sept25.db'

    # sólo stream en tiempo real (kline)
    python influx_to_sql.py --mode stream --symbol BTCUSDT --timeframe 30m --sqlite '../data_manager/exports/influx29sept25.db'

    # ambas cosas: import histórico y luego stream
    python influx_to_sql.py --mode both --symbol BTCUSDT --timeframe 30m --start '-90d' --bucket prices --sqlite '../data_manager/exports/influx29sept25.db'
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Sequence, List
import re
from datetime import datetime

# librerías externas requeridas
import aiosqlite
import websockets
from influxdb_client import InfluxDBClient
import influxdb_client.client.write_api as wa

# Monkeypatch para evitar error en __del__
def safe_del(self):
    try:
        subj = getattr(self, "_subject", None)
        if subj:
            subj.dispose()
    except Exception:
        pass

wa.WriteApi.__del__ = safe_del

# ----------------------
# Config y helpers
# ----------------------
DEFAULT_INFLUX_CONFIG = {
    "url": "http://localhost:8086",
    "token": "X5AtE8ugteP26xo0lkNco0AXklsmNcTs9q49B4eWXl4x-3o74wo6_FJIyTjo5wWNXoc6gYXluFzblcKDGYKXeQ==",
    "org": "BreOrganization",
    "bucket": "prices"
}

import importlib, traceback

def locate_and_import_datamanager_verbose() -> Optional[type]:
    # 1) intento directo paquete
    try:
        from data_manager.manager import DataManager  # type: ignore
        print("[INFO] Import directo 'data_manager.manager' OK")
        return DataManager
    except Exception as e:
        print("[DEBUG] Import directo data_manager.manager falló:", e)

    # 2) añadir ruta raíz del repo (dos niveles arriba del script) al sys.path y reintentar
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[2] if len(this_file.parents) >= 3 else this_file.parents[-1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        print("[DEBUG] Añadido repo_root a sys.path:", repo_root)
    try:
        mod = importlib.import_module("data_manager.manager")
        if hasattr(mod, "DataManager"):
            print("[INFO] Import tras añadir repo_root OK")
            return getattr(mod, "DataManager")
    except Exception as e:
        print("[DEBUG] Import tras añadir repo_root falló:")
        traceback.print_exc()

    # 3) búsqueda exhaustiva hacia arriba hasta root y carga por ruta si existe manager.py
    for p in this_file.parents:
        candidate = p / "data_manager" / "manager.py"
        if candidate.exists():
            print("[DEBUG] Encontrado candidate:", candidate)
            try:
                spec = importlib.util.spec_from_file_location("dm_local", str(candidate))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                if hasattr(module, "DataManager"):
                    print("[INFO] Cargado DataManager desde:", candidate)
                    return getattr(module, "DataManager")
            except Exception:
                print("[DEBUG] Error cargando candidate:")
                traceback.print_exc()
    print("[WARN] No se encontró DataManager en busqueda exhaustiva.")
    return None

# Y luego asigna:
DataManager = locate_and_import_datamanager_verbose()

def _format_start_for_flux(start: str) -> str:
    """
    Normaliza el valor start para inyectarlo en range(start: ...).
    - Si es relativo como -90d o -24h o empieza por now() lo devuelve tal cual.
    - Si parece una marca ISO (contiene 'T' o ':'), lo envuelve en time(v: "...").
    - Elimina comillas exteriores si vienen.
    """
    if start is None:
        return "-90d"
    s = str(start).strip()
    # quitar comillas exteriores si las hay
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1]
    # relativo tipo -90d, -24h, now(), now() etc.
    if re.match(r"^-?\d+[smhdwMy]$", s) or s.startswith("-") and re.match(r"^-\d", s) or s.startswith("now") or s.endswith("()"):
        return s
    # si parece un timestamp ISO (contiene T o espacio y :)
    if ("T" in s) or (":" in s and "-" in s):
        # usar time(v: "2025-09-29T...") para evitar ambigüedades
        return f'time(v: "{s}")'
    # default: envolver en comillas
    return f'"{s}"'


def build_flux_query(bucket: str, start: str = "-90d", measurement: str = "prices",
                     symbol: Optional[str] = None, exchange: Optional[str] = None,
                     timeframe: Optional[str] = None) -> str:
    """
    Construye una query Flux que:
     - filtra por measurement,
     - filtra por fields (open,high,low,close,volume),
     - filtra por tags symbol/exchange/timeframe si se dan,
     - pivotea rows por _field -> columnas (open,high,low,close,volume),
     - devuelve columnas con _time, open, high, low, close, volume, symbol, exchange, timeframe.

    Esto evita errores de compilación por expresiones irregularmente formadas y facilita parsing.
    """
    start_expr = _format_start_for_flux(start)

    # construir condiciones OR si timeframe tiene comas
    def or_join(tag: str, value: str) -> str:
        vals = [v.strip() for v in value.split(",") if v.strip()]
        if len(vals) == 1:
            return f'r["{tag}"] == "{vals[0]}"'
        return " or ".join([f'r[\"{tag}\"] == \"{v}\"' for v in vals])

    filters: List[str] = []
    filters.append(f'r["_measurement"] == "{measurement}"')
    filters.append('(r["_field"] == "open" or r["_field"] == "high" or r["_field"] == "low" or r["_field"] == "close" or r["_field"] == "volume")')

    if exchange:
        filters.append(or_join("exchange", exchange))
    if symbol:
        filters.append(or_join("symbol", symbol))
    if timeframe:
        filters.append(or_join("timeframe", timeframe))

    filter_lines = "\n  |> ".join([f'filter(fn: (r) => {f})' for f in filters])

    # pivot para tener columnas por field
    query = f'''from(bucket: "{bucket}")
  |> range(start: {start_expr})
  |> {filter_lines}
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time","open","high","low","close","volume","symbol","exchange","timeframe"])
  |> sort(columns: ["_time"])
'''
    return query

# ----------------------
# Import desde Influx
# ----------------------
async def import_from_influx(influx_cfg: dict, dm: DataManager, start: str, bucket: str,
                             symbol: Optional[str], exchange: Optional[str], timeframe: Optional[str]):
    """
    Consulta Influx con la query construida por build_flux_query,
    parsea tables y hace upsert en la base (idempotente).
    """
    url = influx_cfg.get("url")
    token = influx_cfg.get("token")
    org = influx_cfg.get("org")
    if not (url and token and org):
        raise RuntimeError("Influx config missing token/org/url. Revisa config/influx.json o pásalo por args.")

    inserted = 0
    # usar context manager para limpiar recursos
    with InfluxDBClient(url=url, token=token, org=org) as client:
        query_api = client.query_api()
        query = build_flux_query(bucket=bucket, start=start, symbol=symbol, exchange=exchange, timeframe=timeframe)
        print(f"[INFO] Ejecutando query en bucket={bucket} start={start} ...")
        tables = query_api.query(query, org=org)

        rows_list = []
        count = 0
        for table in tables:
            for record in table.records:
                count += 1
                t = record.get_time()
                if t is None:
                    continue
                ts = int(t.timestamp())
                vals = record.values
                rows_list.append({
                    "ts": ts,
                    "symbol": vals.get("symbol") or (symbol or ""),
                    "timeframe": vals.get("timeframe") or (timeframe or ""),
                    "o": vals.get("open"),
                    "h": vals.get("high"),
                    "l": vals.get("low"),
                    "c": vals.get("close"),
                    "v": vals.get("volume"),
                })

        print(f"[INFO] Registros Flux leídos: {count}. Filas pivotadas: {len(rows_list)}")
        if not rows_list:
            return 0

        rows_list.sort(key=lambda r: (r["symbol"], r["timeframe"], r["ts"]))

        batch = []
        for r in rows_list:
            batch.append([int(r["ts"]), r["o"], r["h"], r["l"], r["c"], r["v"]])
            if len(batch) >= 500:
                inserted += await dm.upsert_ohlcv(r["symbol"], r["timeframe"], batch, source="influx")
                batch = []
        if batch:
            inserted += await dm.upsert_ohlcv(rows_list[-1]["symbol"], rows_list[-1]["timeframe"], batch, source="influx")

    print(f"[INFO] Import completado. Upsertadas aprox. {inserted} filas en {dm.db_path}")
    return inserted


# ----------------------
# Binance WS streamer (kline)
# ----------------------
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"

def _binance_kline_stream_name(symbol: str, interval: str) -> str:
    return f"{symbol.lower()}@kline_{interval}"

async def binance_ws_kline_loop(symbol: str, timeframe: str, dm: DataManager, partial_updates: bool = False):
    """
    Conecta al stream kline de Binance y hace upsert_ohlcv por cada vela finalizada.
    Si partial_updates=True hace upsert también para velas no-finalizadas (intrabar).
    """
    stream = _binance_kline_stream_name(symbol, timeframe)
    url = f"{BINANCE_WS_BASE}/{stream}"
    backoff = 1.0
    while True:
        try:
            print(f"[WS] Connecting to {url}")
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                backoff = 1.0
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    k = msg.get("k")
                    if not k:
                        continue
                    is_final = k.get("x", False)
                    ts_start_ms = int(k.get("t", 0))
                    ts = int(ts_start_ms / 1000)
                    try:
                        o = float(k.get("o", 0))
                        h = float(k.get("h", 0))
                        l = float(k.get("l", 0))
                        c = float(k.get("c", 0))
                        v = float(k.get("v", 0))
                    except Exception:
                        continue
                    if is_final:
                        # vela finalizada -> escribir/upsert
                        await dm.upsert_ohlcv(symbol.upper(), timeframe, [[ts, o, h, l, c, v]], source="binance_ws")
                    else:
                        # vela parcial -> opcional (puede producir muchos writes)
                        if partial_updates:
                            await dm.upsert_ohlcv(symbol.upper(), timeframe, [[ts, o, h, l, c, v]], source="binance_ws_partial")
        except Exception as e:
            print(f"[WS] Connection error: {e}. Reconnect in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

# ----------------------
# CLI / Orquestación
# ----------------------
def load_influx_config(path: str = "config/influx.json") -> dict:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_INFLUX_CONFIG, f, indent=2)
        print(f"[INFO] Archivo de configuración creado en {p}. Rellena token/org/url si procede.")
        return DEFAULT_INFLUX_CONFIG
    else:
        with open(p, "r", encoding="utf-8") as f:
            try:
                cfg = json.load(f)
                for k in ("url", "token", "org", "bucket"):
                    cfg.setdefault(k, DEFAULT_INFLUX_CONFIG.get(k, ""))
                return cfg
            except json.JSONDecodeError:
                p.unlink(missing_ok=True)
                with open(p, "w", encoding="utf-8") as fw:
                    json.dump(DEFAULT_INFLUX_CONFIG, fw, indent=2)
                print(f"[WARN] {path} corrupto: reescrito con valores por defecto.")
                return DEFAULT_INFLUX_CONFIG

def parse_args():
    p = argparse.ArgumentParser(description="Extrae OHLCV desde InfluxDB y/o stream de Binance hacia SQLite (DataManager).")
    p.add_argument("--mode", choices=["import","stream","both"], default="import", help="Modo: import (influx), stream (ws), both")
    p.add_argument("--config", type=str, default="config/influx.json", help="Path al config JSON de Influx")
    p.add_argument("--bucket", type=str, default="prices")
    p.add_argument("--start", type=str, default="-90d")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--exchange", type=str, default="binance")
    p.add_argument("--timeframe", type=str, default="1m", help="Binance interval (1m,3m,5m,15m,30m,1h, etc.)")
    p.add_argument("--sqlite", type=str, default="./data_manager/exports/marketdata_base.db")
    p.add_argument("--partial_updates", action="store_true", help="Si se activa, el streamer escribirá también velas parciales (intrabar)")
    return p.parse_args()


async def main_async():
    args = parse_args()
    influx_cfg = load_influx_config(args.config)
    if args.bucket:
        influx_cfg["bucket"] = args.bucket

    if DataManager is None:
        raise RuntimeError("DataManager no disponible: revisa imports/paths.")

    dm = DataManager(db_path=args.sqlite, influx_config=influx_cfg)
    await dm.init_db()

    tasks = []
    if args.mode in ("import","both"):
        tasks.append(import_from_influx(influx_cfg=influx_cfg, dm=dm, start=args.start,
                                        bucket=args.bucket, symbol=args.symbol, exchange=args.exchange, timeframe=args.timeframe))
    if args.mode in ("stream","both"):
        tasks.append(binance_ws_kline_loop(symbol=args.symbol, timeframe=args.timeframe, dm=dm, partial_updates=args.partial_updates))

    if not tasks:
        print("No hay tareas. Saliendo.")
        return

    await asyncio.gather(*tasks)


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Exit by user")
    except Exception as e:
        print(f"[ERROR] Ejecución fallida: {e}")


if __name__ == "__main__":
    main()
