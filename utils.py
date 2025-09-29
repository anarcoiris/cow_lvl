"""
utils.py - Utilidades unificadas y limpias para Env.Model

Objetivo:
 - Proveer funciones robustas y defensivas para I/O (sqlite/CCXT), preparación de datos y utilidades de backtest.
 - Forzar consistencia de dtypes (np.float32 / torch.float32 cuando aplique).
 - Mantener compatibilidad hacia atrás en la API pública mínima.
 - Depender opcionalmente de ccxt/joblib/scikit-learn/torch y caer con mensajes claros si faltan.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional deps
try:
    import joblib
except Exception:
    joblib = None

try:
    import ccxt
except Exception:
    ccxt = None

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
except Exception:
    torch = None
    TensorDataset = None
    DataLoader = None

logger = logging.getLogger("utils")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


# ---------------------------
# Helpers and small utilities
# ---------------------------
def _ensure_dir(p: Union[str, Path]):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)


def safe_load_json(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load json from %s", p)
        return None


def safe_dump_json(obj: Any, path: Union[str, Path]) -> bool:
    p = Path(path)
    _ensure_dir(p)
    try:
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    except Exception:
        logger.exception("Failed to write json to %s", p)
        return False


def _normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure there's a timestamp column in datetime64[ns, UTC] if ts exists in seconds
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        return df
    if "ts" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
            return df
        except Exception:
            logger.debug("Failed to convert 'ts' to datetime")
    # fallback: try to parse if 'time' exists
    for candidate in ("time", "datetime"):
        if candidate in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df[candidate], utc=True)
                return df
            except Exception:
                pass
    return df


# ---------------------------
# IO: SQLite helpers
# ---------------------------
def load_ohlcv_from_sqlite(db_path: Union[str, Path], table: str, symbol: str, timeframe: str, limit: int = 10000) -> pd.DataFrame:
    """
    Load OHLCV rows from sqlite table. Returns DataFrame sorted ascending by timestamp.
    Expected columns at least: ts (unix seconds) or timestamp, open, high, low, close, volume, symbol, timeframe.
    """
    p = Path(db_path)
    if not p.exists():
        logger.error("SQLite DB not found: %s", p)
        return pd.DataFrame()
    try:
        con = sqlite3.connect(str(p))
        q = f"SELECT * FROM {table} WHERE symbol = ? AND timeframe = ? ORDER BY ts DESC LIMIT ?"
        df = pd.read_sql_query(q, con, params=[symbol, timeframe, int(limit)])
        con.close()
        if df.empty:
            return df
        # normalize times and sort ascending
        df = _normalize_time_column(df)
        sort_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else df.columns[0])
        df = df.sort_values(sort_col).reset_index(drop=True)
        return df
    except Exception:
        logger.exception("Failed to read sqlite %s table %s", db_path, table)
        return pd.DataFrame()


def fetch_ccxt_ohlcv(exchange_id: str, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 500):
    """
    Fetch OHLCV via ccxt. Returns list of candles OR empty on failure.
    Non-fatal: logs nicely if ccxt missing.
    """
    if ccxt is None:
        logger.warning("ccxt not installed; cannot fetch OHLCV.")
        return []
    try:
        ex_cls = getattr(ccxt, exchange_id)
        ex = ex_cls()
        k = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        return k
    except Exception:
        logger.exception("Failed to fetch OHLCV via ccxt")
        return []


# ---------------------------
# Scaler helpers (joblib)
# ---------------------------
def safe_fit_scaler(scaler, X: np.ndarray) -> Optional[Any]:
    if scaler is None:
        return None
    if joblib is None:
        logger.warning("joblib not available; cannot fit scaler.")
        return None
    try:
        scaler.fit(X)
        return scaler
    except Exception:
        logger.exception("Scaler fit failed")
        return None


def safe_transform_scaler(scaler, X: np.ndarray) -> np.ndarray:
    if scaler is None:
        return X
    if joblib is None:
        logger.warning("joblib not available; passing through data unscaled")
        return X
    try:
        return scaler.transform(X)
    except Exception:
        logger.exception("Scaler transform failed; returning original array")
        return X


def save_scaler(scaler, path: Union[str, Path]) -> bool:
    if joblib is None:
        logger.warning("joblib not available; cannot save scaler.")
        return False
    p = Path(path)
    _ensure_dir(p)
    try:
        joblib.dump(scaler, str(p))
        return True
    except Exception:
        logger.exception("Failed to dump scaler to %s", p)
        return False


def load_scaler(path: Union[str, Path]):
    p = Path(path)
    if not p.exists():
        logger.warning("Scaler file not found: %s", p)
        return None
    if joblib is None:
        logger.warning("joblib not available; cannot load scaler.")
        return None
    try:
        return joblib.load(str(p))
    except Exception:
        logger.exception("Failed to load scaler %s", p)
        return None


# ---------------------------
# Price quantization helpers
# ---------------------------
def quantize_price(price: float, tick: Optional[float] = None) -> float:
    if tick is None or tick <= 0:
        return float(price)
    return float(round(price / tick) * tick)


def quantize_amount(amount: float, step: Optional[float] = None) -> float:
    if step is None or step <= 0:
        return float(amount)
    return float(math.floor(amount / step) * step)


# ---------------------------
# Simulators / Backtest helpers
# ---------------------------
def simulate_ou(T: int = 1000, theta: float = 0.15, mu: float = 0.0, sigma: float = 0.3, x0: float = 0.0, seed: Optional[int] = None):
    """
    Simulate Ornstein-Uhlenbeck process; returns numpy array of floats length T.
    """
    rng = np.random.RandomState(seed)
    x = np.zeros(T, dtype=np.float64)
    x[0] = x0
    dt = 1.0
    for t in range(1, T):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.randn()
    return x


def simulate_fill(order_price: float, side: str, future_segment: pd.DataFrame, slippage: float = 0.0, fail_on_no_fill: bool = False) -> Tuple[bool, float, Optional[int]]:
    """
    Simulate fill on a future price segment.

    Parameters
    ----------
    order_price : float
        Price at which the order was issued (market/limit semantics up to caller).
    side : str
        'buy' or 'sell'
    future_segment : pd.DataFrame
        Future rows (must contain 'high' and 'low' and index that corresponds to original df labels).
    slippage : float
        Fractional slippage to apply to executed price (e.g. 0.001 => 0.1% worse)
    fail_on_no_fill : bool
        If True, return (False, 0.0, None) when no fill occurs.

    Returns
    -------
    (filled: bool, executed_price: float, fill_pos: Optional[int])
        fill_pos is the positional index inside `future_segment` (0-based) of the row that executed the order.
        If cannot determine pos, returns None.
    """
    if future_segment is None or future_segment.empty:
        return False, 0.0, None
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    # We assume marketable: buy fills if low <= order_price (i.e. price sank to or below maker price)
    # For sell, fills if high >= order_price (i.e. price rose to or above)
    try:
        if side == "buy":
            hits = future_segment[future_segment["low"] <= order_price]
            executed_price = order_price * (1.0 + slippage)
        else:
            hits = future_segment[future_segment["high"] >= order_price]
            executed_price = order_price * (1.0 - slippage)
    except Exception:
        # if the future_segment lacks expected columns, fallback
        logger.exception("simulate_fill: future_segment missing expected columns")
        return False, 0.0, None

    if hits.empty:
        # no fill in the segment
        if fail_on_no_fill:
            return False, 0.0, None
        return False, 0.0, None

    # Determine label of the first matching hit and convert to positional index within future_segment
    label = hits.index[0]
    try:
        pos = int(future_segment.index.get_loc(label))
    except Exception:
        # fallback: if index is integer-like, try to coerce
        try:
            pos = int(hits.index[0]) if isinstance(hits.index[0], (int,)) else None
        except Exception:
            pos = None

    return True, float(executed_price), pos


# ---------------------------
# Small utilities that other modules expect
# ---------------------------
def ensure_iterable(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]


def to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None


# ---------------------------
# Public API listing
# ---------------------------
__all__ = [
    "safe_load_json",
    "safe_dump_json",
    "load_ohlcv_from_sqlite",
    "fetch_ccxt_ohlcv",
    "create_history_buffer",
    "prepare_model_input",
    "safe_fit_scaler",
    "safe_transform_scaler",
    "save_scaler",
    "load_scaler",
    "quantize_price",
    "quantize_amount",
    "simulate_fill",
    "simulate_ou",
    "ensure_iterable",
    "to_float_or_none",
]

# The module intentionally leaves placeholders (e.g. create_history_buffer, prepare_model_input)
# to be provided by the codebase's other modules / higher-level wrappers. The functions present
# are defensive and intended to be stable infra utilities.
