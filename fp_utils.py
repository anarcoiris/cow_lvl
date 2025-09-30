"""
fp_utils.py

Funciones (helpers) comunes para fibopkg3:
- Normalización de symbol/timeframe
- Conversiones seguras numpy <-> torch (float32)
- Save/load model con meta, inferencia de shapes y carga parcial segura
- Quantize price/amount y simulate_fill básico
- Simulate OU (generador sintético) – versión robusta
- Otros helpers (scalar_from_tensor_or_array)
"""

from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Sequence, Union
import numpy as np
import json

LOGGER = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    _HAS_TORCH = False

# Timeframe mappings (agrégalas según convenga)
TIMEFRAME_MAP = {
    "1m": "1m", "1min": "1m", "1M": "1m",
    "3m": "3m", "3min": "3m",
    "5m": "5m", "5min": "5m",
    "15m": "15m", "15min": "15m",
    "30m": "30m", "30min": "30m", "30M": "30m",
    "1h": "1h", "60m": "1h", "1H": "1h",
    "4h": "4h", "1d": "1d", "1D": "1d", "daily": "1d"
}

COMMON_QUOTE_ASSETS = ["USDT", "USD", "BTC", "ETH", "EUR"]


def normalize_timeframe(tf: str) -> str:
    if tf is None:
        raise ValueError("timeframe is None")
    t = str(tf).strip()
    tl = t.lower()
    if tl in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[tl]
    # heuristica flexible: '30 min' , '30m', '30M'
    m = re.match(r"^(\d+)\s*(m|min|minutes?)$", tl)
    if m:
        return f"{int(m.group(1))}m"
    m = re.match(r"^(\d+)\s*(h|hours?)$", tl)
    if m:
        return f"{int(m.group(1))}h"
    m = re.match(r"^(\d+)\s*(d|days?)$", tl)
    if m:
        return f"{int(m.group(1))}d"
    raise ValueError(f"Unknown timeframe: {tf}")


def normalize_symbol(sym: str, default_quote_candidates: Optional[Sequence[str]] = None) -> str:
    if sym is None:
        raise ValueError("symbol is None")
    s = str(sym).strip().upper()
    s = s.replace("-", "/").replace("_", "/")
    if "/" in s:
        base, quote = [p.strip() for p in s.split("/", 1)]
        return f"{base}/{quote}"
    if default_quote_candidates is None:
        default_quote_candidates = COMMON_QUOTE_ASSETS
    for q in default_quote_candidates:
        if s.endswith(q):
            base = s[: -len(q)]
            if base:
                return f"{base}/{q}"
    return s  # fallback: return uppercased raw


# ---- numpy <-> torch safe conversions ----

def to_numpy_float32(x: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
    """
    Convierte tensor o array a numpy float32.
    """
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    raise TypeError("Unsupported type for to_numpy_float32")


def to_tensor_float32(arr: Union[np.ndarray, Sequence[float], "torch.Tensor"], device: Optional[Union[str, "torch.device"]] = "cpu"):
    """
    Devuelve torch.Tensor en dtype float32. Si torch no disponible, lanza RuntimeError.
    device puede ser 'cpu', 'cuda' o torch.device.
    """
    if not _HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch not available (pip install torch)")

    if isinstance(arr, torch.Tensor):
        return arr.float().to(device)
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr.astype(np.float32)).to(device)
    # try generic sequence
    a = np.array(arr, dtype=np.float32)
    return torch.from_numpy(a).to(device)


def scalar_from_tensor_or_array(x: Union["torch.Tensor", np.ndarray, float, int]) -> float:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray):
        return float(np.asarray(x).item())
    if isinstance(x, (int, float)):
        return float(x)
    raise TypeError("Unsupported type for scalar extraction")


# ---- model save / load helpers (state_dict + meta) ----

def save_model_with_meta(model: Any, path_model: Union[str, Path], meta: Optional[Dict[str, Any]] = None) -> None:
    """
    Guarda payload = {'state_dict': model.state_dict(), 'meta': meta} en path_model (torch.save).
    Si torch no disponible -> lanza.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required to save model")

    p = Path(path_model)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, str(p))
    LOGGER.info("Saved model to %s", p)


def _infer_config_from_state_dict(state_dict: dict) -> Dict[str, Any]:
    """
    Intenta inferir input_size y hidden para LSTM2Head basado en peso 'lstm.weight_ih_l0'
    """
    config: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if "lstm.weight_ih_l0" in k:
            try:
                rows, cols = v.shape
                if rows % 4 == 0:
                    config["hidden"] = rows // 4
                    config["input_size"] = cols
            except Exception:
                pass
            break
    return config


def load_model_with_meta(path_model: Union[str, Path], model_factory=None, device: Optional[Union[str, "torch.device"]] = "cpu", allow_partial: bool = True):
    """
    Carga un checkpoint. Si model_factory es callable(model_kwargs) -> model_instance,
    lo usará para crear el modelo. Si model_factory es None intenta inferir input_size/hidden
    y construye un modelo LSTM2Head si está definido en el entorno (no incluido aquí).
    Devuelve tuple (model, meta).
    - path_model puede apuntar a un object que contenga {'state_dict', 'meta'} o a un state_dict puro.
    - allow_partial: intenta copia parcial de pesos cuando shapes no coinciden.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required to load model")

    p = Path(path_model)
    if not p.exists():
        raise FileNotFoundError(p)

    checkpoint = torch.load(str(p), map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        meta = checkpoint.get("meta", None)
    else:
        state_dict = checkpoint
        meta = None

    inferred = _infer_config_from_state_dict(state_dict)
    # prefer meta info if present
    input_size = None
    hidden = None
    if meta and isinstance(meta, dict):
        input_size = meta.get("input_size") or meta.get("inputsize")
        hidden = meta.get("hidden")
    # use inferred if necessary
    if input_size is None:
        input_size = inferred.get("input_size")
    if hidden is None:
        hidden = inferred.get("hidden", 64)

    if model_factory is not None:
        model = model_factory({"input_size": int(input_size), "hidden": int(hidden)})
    else:
        # try to import LSTM2Head from fiboevo if available
        try:
            from fibopkg2.fiboevo import LSTM2Head  # type: ignore

            model = LSTM2Head(input_size=int(input_size), hidden_size=int(hidden)).to(device)
        except Exception:
            raise RuntimeError("No model_factory provided and LSTM2Head not importable. Provide model_factory to build model.")

    # strict load
    try:
        model.load_state_dict(state_dict)
        LOGGER.info("Model loaded strictly from %s", p)
    except RuntimeError as e:
        LOGGER.warning("Strict load failed: %s", e)
        if allow_partial:
            LOGGER.info("Attempting partial load of compatible parameter slices.")
            _safe_partial_load_state_dict(model, state_dict)
        else:
            raise
    return model, meta


def _safe_partial_load_state_dict(model: Any, checkpoint_state: dict):
    """
    Copia porciones compatibles entre checkpoint_state y model.state_dict().
    Para tensores 2D copia el sub-bloque superior-left que encaje; para vectores copia la porción inicial.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required")
    ms = model.state_dict()
    new_state = {}
    for k, v in checkpoint_state.items():
        if k not in ms:
            LOGGER.debug("Skipping unexpected key in checkpoint: %s", k)
            continue
        target = ms[k]
        try:
            if v.shape == target.shape:
                new_state[k] = v
                continue
            # 2D tensors
            if v.ndim == 2 and target.ndim == 2:
                rows = min(v.shape[0], target.shape[0])
                cols = min(v.shape[1], target.shape[1])
                tmp = target.clone()
                tmp[:rows, :cols] = v[:rows, :cols]
                new_state[k] = tmp
                LOGGER.info("Partially copied 2D tensor %s (%s -> %s)", k, v.shape, target.shape)
            elif v.ndim == 1 and target.ndim == 1:
                l = min(v.shape[0], target.shape[0])
                tmp = target.clone()
                tmp[:l] = v[:l]
                new_state[k] = tmp
                LOGGER.info("Partially copied 1D tensor %s (%s -> %s)", k, v.shape, target.shape)
            else:
                LOGGER.debug("Cannot handle different ndim for key %s: %d vs %d", k, v.ndim, target.ndim)
        except Exception as ee:
            LOGGER.exception("Error copying %s: %s", k, ee)
    ms.update(new_state)
    model.load_state_dict(ms)
    LOGGER.info("Loaded model with partial state_dict (copied overlapping slices).")


# ---- quantize & simulate ----

def quantize_price(price: float, tick: Optional[float]) -> float:
    try:
        p = float(price)
    except Exception:
        raise TypeError("price must be numeric")
    if tick is None or tick == 0:
        return p
    tick = float(tick)
    rounded = round(p / tick) * tick
    return float(rounded)


def quantize_amount(amount: float, lot: Optional[float]) -> float:
    try:
        a = float(amount)
    except Exception:
        raise TypeError("amount must be numeric")
    if lot is None or lot == 0:
        return a
    lot = float(lot)
    rounded = max(0.0, (round(a / lot) * lot))
    return float(rounded)


def simulate_fill(order_price: float, side: str, future_segment: Sequence[float] = None,
                  slippage_tolerance_pct: float = 0.5, tick: Optional[float] = None,
                  lot: Optional[float] = None, method: str = "slippage") -> Tuple[bool, Optional[float], Optional[int]]:
    """
    Simula si una orden a order_price se llena dentro del segmento futuro (serie de precios).
    - side: 'long' o 'short'
    - future_segment: array-like de precios (p.ej. lows para long, highs para short)
    Devuelve (filled, executed_price, idx) donde idx es offset relativo del fill.
    """
    if future_segment is None or len(future_segment) == 0:
        return False, None, None
    arr = np.asarray(future_segment, dtype=np.float32)
    if side.lower() == "long":
        # if any future low <= order_price
        idxs = np.where(arr <= order_price)[0]
        if idxs.size == 0:
            return False, None, None
        idx = int(idxs[0])
        exec_price = arr[idx] * (1.0 + slippage_tolerance_pct / 100.0)
    else:
        idxs = np.where(arr >= order_price)[0]
        if idxs.size == 0:
            return False, None, None
        idx = int(idxs[0])
        exec_price = arr[idx] * (1.0 - slippage_tolerance_pct / 100.0)
    exec_price = quantize_price(exec_price, tick)
    exec_price = float(exec_price)
    if lot is not None:
        # optionally quantize amount externally
        pass
    return True, exec_price, idx


# ---- Simulador OU robusto ----

def simulate_ou(theta: float = 1.0, sigma: float = 0.5, dt: float = 0.01, steps: int = 5000, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simula un proceso Ornstein-Uhlenbeck y devuelve (close, high, low) arrays en float32.
    Garantiza high >= low y evita valores negativos patológicos.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(steps + 1, dtype=np.float32)
    dw = rng.normal(loc=0.0, scale=np.sqrt(dt), size=steps).astype(np.float32)
    for t in range(steps):
        x[t + 1] = x[t] - theta * x[t] * dt + sigma * dw[t]
    price = 100.0 + np.cumsum(x).astype(np.float32)
    noise = 0.001 * np.abs(rng.standard_normal(len(price))).astype(np.float32)
    high = price * (1.0 + noise)
    low = price * (1.0 - noise)
    low = np.minimum(low, high * 0.9999)
    return price.astype(np.float32), high.astype(np.float32), low.astype(np.float32)
