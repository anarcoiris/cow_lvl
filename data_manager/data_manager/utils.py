
"""Funciones utilitarias ampliadas."""
import datetime, math
from typing import Any, Optional

def ts_to_datetime(ts: int) -> datetime.datetime:
    return datetime.datetime.utcfromtimestamp(ts / 1000)

def datetime_to_ts(dt: datetime.datetime) -> int:
    return int(dt.timestamp() * 1000)

def safe_symbol(symbol: str) -> str:
    return symbol.replace('/', '_')

def is_finite_number(x: Any) -> bool:
    try:
        if x is None:
            return False
        return math.isfinite(float(x))
    except Exception:
        return False

def is_nan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return False
