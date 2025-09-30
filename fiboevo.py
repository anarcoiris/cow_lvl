#!/usr/bin/env python3
"""
fiboevo_unified.py — Módulo unificado y pulido.

Contiene:
 - helpers de normalización: normalize_symbol, normalize_timeframe
 - conversiones dtype/tensor: to_tensor_float32, ensure_float32_tensor
 - feature engineering: add_technical_features(...)
 - creación de secuencias: create_sequences_from_df(...)
 - modelo LSTM: LSTM2Head (si torch disponible)
 - train/eval helpers: train_epoch, eval_epoch (dtype/shape-robustos)
 - persistencia: save_model, load_model, save_scaler, load_scaler
 - dataset builder: build_dataset_for_training
 - predict helpers: prepare_model_input, predict_with_model
 - backtest helpers: simulate_fill, quantize_price/amount, simulate_ou
 - I/O helpers: load_data_from_sqlite, load_ohlcv_from_sqlite, fetch_ccxt_ohlcv

Diseñado para ser importado por prepare_dataset.py, trading_daemon.py, trading_gui*, etc.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Optional ML stack imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

try:
    import joblib
except Exception:
    joblib = None

# Optional ccxt
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

LOGGER = logging.getLogger("fiboevo_unified")
if not LOGGER.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    LOGGER.addHandler(ch)
    LOGGER.setLevel(logging.INFO)


# ----------------------------
# Normalization helpers
# ----------------------------
TIMEFRAME_MAP = {
    "1m": "1m", "1min": "1m", "60s": "1m",
    "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "60m": "1h", "4h": "4h", "1d": "1d", "daily": "1d"
}
COMMON_QUOTE_ASSETS = ["USDT", "USD", "BTC", "ETH", "EUR"]


def normalize_timeframe(tf: str) -> str:
    if tf is None:
        raise ValueError("timeframe is None")
    t = str(tf).strip()
    tl = t.lower()
    if tl in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[tl]
    m = re.match(r"^(\d+)\s*(m|min|minutes?)$", t, flags=re.I)
    if m:
        return f"{int(m.group(1))}m"
    m = re.match(r"^(\d+)\s*(h|hours?)$", t, flags=re.I)
    if m:
        return f"{int(m.group(1))}h"
    # fallback: return trimmed lower
    return tl


def normalize_symbol(sym: str, default_quotes: Optional[Sequence[str]] = None) -> str:
    if sym is None:
        raise ValueError("symbol is None")
    s = str(sym).strip().upper()
    s = s.replace("-", "/").replace("_", "/")
    if "/" in s:
        base, quote = [p.strip() for p in s.split("/", 1)]
        return f"{base}/{quote}"
    candidates = default_quotes or COMMON_QUOTE_ASSETS
    for q in candidates:
        if s.endswith(q):
            base = s[:-len(q)]
            if base:
                return f"{base}/{q}"
    # unknown: return as-is (uppercase)
    return s


# ----------------------------
# Torch / dtype helpers
# ----------------------------
def to_numpy_float32(arr: Any) -> np.ndarray:
    a = np.asarray(arr)
    return a.astype(np.float32)


def to_tensor_float32(arr: Any, device: Optional[str] = None) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch no disponible")
    a = np.asarray(arr, dtype=np.float32)
    t = torch.from_numpy(a)
    if device:
        try:
            return t.to(device)
        except Exception:
            dev = torch.device(device)
            return t.to(dev)
    return t


def ensure_float32_tensor(t: "torch.Tensor") -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch no disponible")
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    if not torch.is_floating_point(t):
        raise TypeError("tensor must be floating")
    if t.dtype != torch.float32:
        t = t.to(torch.float32)
    return t


# ----------------------------
# Determinism
# ----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass


# ----------------------------
# Small indicators / fallbacks
# ----------------------------
def simple_rsi(close: Sequence[float], length: int = 14) -> np.ndarray:
    s = pd.Series(close, dtype=float)
    diff = s.diff(1)
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    ma_up = up.rolling(length, min_periods=length).mean()
    ma_down = down.rolling(length, min_periods=length).mean()
    rs = ma_up / (ma_down + 1e-12)
    return (100 - (100 / (1 + rs))).to_numpy()


def atr_simple(high: Sequence[float], low: Sequence[float], close: Sequence[float], length: int = 14) -> np.ndarray:
    h = pd.Series(high, dtype=float)
    l = pd.Series(low, dtype=float)
    c = pd.Series(close, dtype=float)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean().to_numpy()


# ----------------------------
# Feature engineering
# ----------------------------
def add_technical_features(
    close: Sequence[float],
    high: Optional[Sequence[float]] = None,
    low: Optional[Sequence[float]] = None,
    volume: Optional[Sequence[float]] = None,
    window_long: int = 50,
    vp_window: int = 100,
    include_vp: bool = True,
    dropna_after: bool = False,
) -> pd.DataFrame:
    """
    Produce DataFrame alineado con close.
    NO hace dropna() por defecto: caller debe call df.dropna() cuando corresponda.
    """
    close_arr = np.asarray(close, dtype=np.float64)
    df = pd.DataFrame({"close": close_arr})
    if high is None:
        high = close_arr
    if low is None:
        low = close_arr
    df["high"] = np.asarray(high, dtype=np.float64)
    df["low"] = np.asarray(low, dtype=np.float64)

    # safe log close
    df["log_close"] = np.log(df["close"].replace(0, np.nan))

    # returns
    df["ret_1"] = df["close"].pct_change(1)
    df["log_ret_1"] = df["log_close"].diff(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["log_ret_5"] = df["log_close"].diff(5)

    # moving averages and EMAs
    for L in [5, 10, 20, 50, window_long]:
        df[f"sma_{L}"] = df["close"].rolling(L, min_periods=L).mean()
        df[f"ema_{L}"] = df["close"].ewm(span=L, adjust=False).mean()
        df[f"ma_diff_{L}"] = df["close"] - df[f"sma_{L}"]

    # bollinger
    bb_m = df["close"].rolling(20, min_periods=20).mean()
    bb_std = df["close"].rolling(20, min_periods=20).std()
    df["bb_m"] = bb_m
    df["bb_std"] = bb_std
    df["bb_up"] = bb_m + 2 * bb_std
    df["bb_dn"] = bb_m - 2 * bb_std
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / bb_m.replace(0, np.nan)

    # RSI (pandas_ta optional)
    try:
        import pandas_ta as pta  # type: ignore
        df["rsi_14"] = pta.rsi(df["close"], length=14)
    except Exception:
        df["rsi_14"] = simple_rsi(df["close"].values, 14)

    # ATR
    df["atr_14"] = atr_simple(df["high"], df["low"], df["close"], 14)

    # volatility
    df["raw_vol_10"] = df["log_ret_1"].rolling(10, min_periods=10).std()
    df["raw_vol_30"] = df["log_ret_1"].rolling(30, min_periods=30).std()

    # fibonacci levels
    lookback = 50
    highs = df["high"].rolling(lookback, min_periods=lookback).max()
    lows = df["low"].rolling(lookback, min_periods=lookback).min()
    fibs = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    for f in fibs:
        level = lows + (highs - lows) * f
        name = f"fib_{int(f*1000)}"
        df[name] = level
        df[f"dist_{name}"] = (df["close"] - level) / df["close"].replace(0, np.nan)

    # TD-like setups
    close_v = df["close"].values
    buy_setup = np.zeros(len(close_v), dtype=int)
    sell_setup = np.zeros(len(close_v), dtype=int)
    for i in range(4, len(close_v)):
        buy_setup[i] = buy_setup[i - 1] + 1 if close_v[i] < close_v[i - 4] else 0
        sell_setup[i] = sell_setup[i - 1] + 1 if close_v[i] > close_v[i - 4] else 0
    df["td_buy_setup"] = buy_setup
    df["td_sell_setup"] = sell_setup

    # volume profile approx
    if include_vp and volume is not None:
        vol = np.asarray(volume, dtype=np.float64)
        df["volume"] = vol
        pocs = [np.nan] * len(df)
        if vp_window <= 0:
            vp_window = 100
        for i in range(vp_window, len(df)):
            seg = df["close"].iloc[i - vp_window : i]
            seg_vol = df["volume"].iloc[i - vp_window : i].values
            if np.nansum(seg_vol) <= 0:
                pocs[i] = np.nan
                continue
            hist, edges = np.histogram(seg.values, bins=20, weights=seg_vol)
            max_bin = int(np.argmax(hist))
            pocs[i] = 0.5 * (edges[max_bin] + edges[max_bin + 1])
        df["vp_poc"] = pocs
        df["dist_vp"] = (df["close"] - df["vp_poc"]) / df["close"].replace(0, np.nan)
    else:
        df["volume"] = df.get("volume", np.nan)
        df["vp_poc"] = np.nan
        df["dist_vp"] = np.nan

    if dropna_after:
        df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------
# Sequences for LSTM
# ----------------------------
def create_sequences_from_df(df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int = 32, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences (X) and targets (y_ret as log-return, y_vol as std of log-returns).
    Returns:
      X: (N, seq_len, F) float32
      y_ret: (N,) float32  -- log_return = log(close_{t+h}) - log(close_t)
      y_vol: (N,) float32  -- rolling std of log-returns over horizon
    Caller: df must be sorted asc and preferably dropna() applied (function will copy and assert).
    """
    if "close" not in df.columns:
        raise RuntimeError("DataFrame must contain 'close' column.")
    df_loc = df.reset_index(drop=True).copy()
    present = [c for c in feature_cols if c in df_loc.columns]
    if len(present) == 0:
        raise RuntimeError("No requested feature_cols present in dataframe.")
    arr = df_loc[present].astype(np.float32).values
    close = df_loc["close"].astype(np.float64).values  # use float64 for log stability
    if (close <= 0).any():
        raise ValueError("Non-positive close price detected; cannot compute log-returns.")
    log_close = np.log(close)
    M = len(df_loc)
    N = M - seq_len - horizon + 1
    if N <= 0:
        raise ValueError(f"Not enough rows for seq_len+horizon: need {seq_len + horizon}, got {M}")
    X = np.zeros((N, seq_len, arr.shape[1]), dtype=np.float32)
    y_ret = np.zeros((N,), dtype=np.float32)
    y_vol = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        X[i] = arr[i : i + seq_len]
        t_end = i + seq_len - 1
        t_target = t_end + horizon
        y_ret[i] = float(log_close[t_target] - log_close[t_end])
        # y_vol: std of log-returns between t_end+1 .. t_target inclusive (h samples)
        if horizon >= 1:
            segment = log_close[t_end + 1 : t_target + 1] - log_close[t_end : t_target]
            y_vol[i] = float(np.std(segment)) if len(segment) > 0 else 0.0
        else:
            y_vol[i] = 0.0
    return X, y_ret, y_vol


# ----------------------------
# Model: LSTM2Head
# ----------------------------
if torch is not None:
    class LSTM2Head(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            assert input_size >= 1
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
            self.head_ret = nn.Sequential(nn.Linear(hidden_size, max(4, hidden_size // 2)), nn.ReLU(), nn.Linear(max(4, hidden_size // 2), 1))
            self.head_vol = nn.Sequential(nn.Linear(hidden_size, max(4, hidden_size // 2)), nn.ReLU(), nn.Linear(max(4, hidden_size // 2), 1), nn.Softplus())

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            ret = self.head_ret(last).squeeze(-1)
            vol = self.head_vol(last).squeeze(-1)
            return ret, vol
else:
    LSTM2Head = None


# ----------------------------
# Training / Eval helpers
# ----------------------------
def _ensure_shape_1d(t: "torch.Tensor") -> "torch.Tensor":
    if t is None:
        return t
    if hasattr(t, "dim") and t.dim() == 2 and t.size(1) == 1:
        return t.view(-1)
    # squeeze(-1) may produce scalar for shape (), guard that
    if hasattr(t, "dim") and t.dim() >= 1:
        return t.squeeze(-1)
    return t


def train_epoch(model: Any, loader: DataLoader, optimizer: Any, device: "torch.device", alpha_vol_loss: float = 0.5, grad_clip: float = 1.0) -> float:
    if torch is None:
        raise RuntimeError("torch no disponible")
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    n = 0
    for batch in loader:
        if len(batch) == 3:
            xb, yret, yvol = batch
        else:
            xb = batch[0]; yret = batch[1]; yvol = batch[2]
        xb = xb.to(device=device)
        yret = yret.to(device=device)
        yvol = yvol.to(device=device) if yvol is not None else None
        # normalize shape to 1d
        yret = _ensure_shape_1d(yret)
        if yvol is not None:
            yvol = _ensure_shape_1d(yvol)
        optimizer.zero_grad(set_to_none=True)
        pred_ret, pred_vol = model(xb)
        loss_ret = mse(pred_ret, yret)
        loss_vol = mse(pred_vol, yvol) if yvol is not None else torch.tensor(0.0, device=device)
        loss = loss_ret + alpha_vol_loss * loss_vol
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(1, n)


def eval_epoch(model: Any, loader: DataLoader, device: "torch.device", alpha_vol_loss: float = 0.5) -> float:
    if torch is None:
        raise RuntimeError("torch no disponible")
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                xb, yret, yvol = batch
            else:
                xb = batch[0]; yret = batch[1]; yvol = None
            xb = xb.to(device=device)
            yret = yret.to(device=device)
            yret = _ensure_shape_1d(yret)
            if yvol is not None:
                yvol = yvol.to(device=device)
                yvol = _ensure_shape_1d(yvol)
            pred_ret, pred_vol = model(xb)
            loss_ret = mse(pred_ret, yret)
            loss_vol = mse(pred_vol, yvol) if yvol is not None else torch.tensor(0.0, device=device)
            loss = loss_ret + alpha_vol_loss * loss_vol
            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
    return total_loss / max(1, n)


# ----------------------------
# Quantize / simulate helpers
# ----------------------------
def quantize_price(price: float, tick: Optional[float]) -> float:
    if tick is None or tick == 0:
        return float(price)
    q = float(price) / float(tick)
    # round to nearest tick (safer for price)
    return float(round(q) * float(tick))


def quantize_amount(amount: float, lot: Optional[float]) -> float:
    if lot is None or lot == 0:
        return float(amount)
    q = float(amount) / float(lot)
    return float(math.floor(q) * float(lot))


def simulate_fill(order_price: float, side: str, future_segment: pd.Series, slippage_tolerance_pct: float = 0.5, tick: Optional[float] = None, lot: Optional[float] = None, method: str = "slippage") -> Tuple[bool, Optional[float], Optional[int]]:
    """
    Simula fill en future_segment (serie ordenada). Devuelve (filled, executed_price, pos_index)
    pos_index es índice posicional relativo a future_segment (0..len-1). Caller puede usar .index[pos_index] si necesita label.
    - side: 'buy' or 'sell'
    - future_segment: pd.Series of prices (ordered forward)
    """
    if future_segment is None or len(future_segment) == 0:
        return False, None, None
    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if side == "buy":
        cond = future_segment <= order_price
    else:
        cond = future_segment >= order_price
    hits = future_segment[cond]
    if hits.empty:
        return False, None, None
    # pick first hit label
    label = hits.index[0]
    # positional index (works with RangeIndex and other Index types)
    try:
        pos = int(future_segment.index.get_loc(label))
    except Exception:
        # fallback: try to compute by comparing positions
        pos = int(np.where(future_segment.index == label)[0][0]) if label in future_segment.index else 0
    executed_price = float(order_price)
    slippage = executed_price * (float(slippage_tolerance_pct) / 100.0)
    if method == "slippage":
        executed_price = executed_price + slippage if side == "buy" else executed_price - slippage
    if tick:
        executed_price = quantize_price(executed_price, tick)
    # lot handled externally for amount quantization
    return True, float(executed_price), pos


# ----------------------------
# Save / load model & scaler
# ----------------------------
def save_model(model: Any, path_model: Path, meta: Optional[Dict] = None, path_meta: Optional[Path] = None) -> None:
    if torch is None:
        raise RuntimeError("torch no disponible")
    path_model = Path(path_model)
    meta = meta or {}
    if path_model.is_dir():
        os.makedirs(path_model, exist_ok=True)
        target = path_model / "model_best.pt"
    else:
        os.makedirs(path_model.parent, exist_ok=True)
        target = path_model
    try:
        payload = {"state": model.state_dict(), "meta": meta}
        torch.save(payload, str(target))
        LOGGER.info("Saved model+meta -> %s", target)
    except Exception:
        torch.save(model.state_dict(), str(target))
        if meta and path_meta:
            try:
                with open(path_meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                LOGGER.exception("Failed to save meta sidecar")
        LOGGER.info("Saved state_dict -> %s (meta -> %s)", target, path_meta)


def load_model(path_model: Path, device: Optional["torch.device"] = None) -> Tuple[Any, Dict]:
    if torch is None:
        raise RuntimeError("torch no disponible")
    p = Path(path_model)
    if p.is_dir():
        for cand in ("model_best.pt", "final_model.pt", "model.pt", "model.pth"):
            cp = p / cand
            if cp.exists():
                p = cp
                break
        else:
            raise RuntimeError(f"Directory provided and no default model found in {p}")
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    obj = torch.load(str(p), map_location=device)
    state = obj
    meta: Dict = {}
    if isinstance(obj, dict):
        if "state" in obj and isinstance(obj["state"], dict):
            state = obj["state"]
            meta = obj.get("meta", {}) or {}
        elif "state_dict" in obj:
            state = obj["state_dict"]
        elif "model_state_dict" in obj:
            state = obj["model_state_dict"]
        else:
            # try sidecar meta files
            sidecar = p.parent / "meta.json"
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
    # normalize DataParallel keys
    normalized = {}
    for k, v in state.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        normalized[nk] = v
    state = normalized

    # try infer input_size/hidden if not in meta
    inferred_input = None
    inferred_hidden = None
    for k in ("lstm.weight_ih_l0", "head_ret.0.weight", "head_ret.2.weight"):
        if k in state:
            w = state[k]
            try:
                shape = getattr(w, "shape", None)
                if shape and len(shape) == 2:
                    if k.startswith("lstm"):
                        inferred_hidden = int(shape[0] // 4)
                        inferred_input = int(shape[1])
                    else:
                        inferred_hidden = int(shape[1])
            except Exception:
                pass

    input_size = meta.get("input_size", None)
    hidden = meta.get("hidden", None)
    num_layers = meta.get("num_layers", 2)
    if input_size is None and inferred_input is not None:
        input_size = int(inferred_input)
    if hidden is None and inferred_hidden is not None:
        hidden = int(inferred_hidden)
    if input_size is None:
        raise RuntimeError("No se pudo inferir input_size. Guarda meta con 'input_size' o pásalo al guardar.")
    if LSTM2Head is None:
        raise RuntimeError("LSTM2Head no disponible en este entorno (torch faltante).")
    model = LSTM2Head(input_size=int(input_size), hidden_size=int(hidden or 64), num_layers=int(num_layers)).to(device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    return model, meta


def save_scaler(scaler: Any, path: Path) -> None:
    if joblib is None:
        raise RuntimeError("joblib no disponible")
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    joblib.dump(scaler, str(path))


def load_scaler(path: Path) -> Any:
    if joblib is None:
        raise RuntimeError("joblib no disponible")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(str(path))


# ----------------------------
# Build dataset helper
# ----------------------------
def build_dataset_for_training(df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int, horizon: int, val_frac: float = 0.2, scaler: Optional[Any] = None) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Any]:
    """
    Fit scaler on training portion (rows used by training sequences) and return train/val sequences + scaler.
    Returns: (Xtr,ytr,voltr), (Xv,yv,volv), fitted_scaler
    """
    if StandardScaler is None:
        raise RuntimeError("scikit-learn no disponible")
    df_local = df.copy().reset_index(drop=True)
    df_local = df_local.dropna().reset_index(drop=True)
    scaler = scaler or StandardScaler()
    n_total = len(df_local)
    min_required = seq_len + horizon + 1
    if n_total < min_required:
        raise ValueError("No hay suficientes filas para seq_len+horizon")
    n_sequences = n_total - seq_len - horizon + 1
    n_val_seq = max(1, int(round(val_frac * n_sequences)))
    n_train_seq = n_sequences - n_val_seq
    if n_train_seq <= 0:
        raise ValueError("val_frac demasiado grande")
    # rows used by training sequences: we choose to fit on rows up to the last row used by training sequences
    train_rows_end = seq_len + n_train_seq - 1
    # fit scaler on raw features up to train_rows_end (inclusive)
    scaler.fit(df_local[list(feature_cols)].iloc[: train_rows_end + 1].values.astype(np.float64))
    # store feature names if possible (useful later)
    try:
        scaler.feature_names_in_ = np.array(list(feature_cols), dtype=object)
    except Exception:
        pass
    df_local[list(feature_cols)] = scaler.transform(df_local[list(feature_cols)].values.astype(np.float64)).astype(np.float32)
    X_all, y_all, v_all = create_sequences_from_df(df_local, feature_cols, seq_len=seq_len, horizon=horizon)
    Xtr = X_all[:n_train_seq]
    ytr = y_all[:n_train_seq]
    voltr = v_all[:n_train_seq]
    Xv = X_all[n_train_seq:]
    yv = y_all[n_train_seq:]
    volv = v_all[n_train_seq:]
    return (Xtr, ytr, voltr), (Xv, yv, volv), scaler


# ----------------------------
# Predict helpers
# ----------------------------
def prepare_model_input(df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int, scaler: Optional[Any] = None, device: Optional[str] = "cpu"):
    """
    Return torch tensor shape (1, seq_len, F) dtype float32 on given device.
    """
    if torch is None:
        raise RuntimeError("torch no disponible")
    if len(df) < seq_len:
        raise ValueError("df must have at least seq_len rows")
    df_local = df.copy().reset_index(drop=True)
    last = df_local.iloc[-seq_len:][list(feature_cols)].values.astype(np.float32)
    if scaler is not None:
        try:
            last = scaler.transform(last.astype(np.float64)).astype(np.float32)
        except Exception:
            last = last.astype(np.float32)
    t = torch.from_numpy(last).unsqueeze(0)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return t.to(device)


def predict_with_model(model: Any, input_tensor: Any) -> Tuple[float, float]:
    if torch is None:
        raise RuntimeError("torch no disponible")
    model.eval()
    with torch.no_grad():
        out_ret, out_vol = model(input_tensor)
    return float(out_ret.cpu().numpy().ravel()[0]), float(out_vol.cpu().numpy().ravel()[0])


# ----------------------------
# Simulators / IO
# ----------------------------
def simulate_ou(theta: float = 1.0, sigma: float = 0.5, dt: float = 0.01, steps: int = 5000, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)
    x = np.zeros(steps + 1, dtype=np.float32)
    dw = rng.normal(loc=0.0, scale=np.sqrt(dt), size=steps).astype(np.float32)
    for t in range(steps):
        x[t + 1] = x[t] - theta * x[t] * dt + sigma * dw[t]
    price = 100.0 + np.cumsum(x)
    noise = 0.001 * np.abs(rng.standard_normal(len(price))).astype(np.float32)
    high = price * (1.0 + noise)
    low = price * (1.0 - noise)
    low = np.minimum(low, high * 0.9999)
    return price.astype(np.float32), high.astype(np.float32), low.astype(np.float32)


def load_data_from_sqlite(path: str, table: str, symbol: str, timeframe: str, start_ts: Optional[str] = None, limit: int = 100000) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SQLite file not found: {path}")
    conn = sqlite3.connect(str(p), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    q = f"SELECT timestamp, open, high, low, close, volume FROM {table} WHERE symbol = ? AND timeframe = ?"
    params = [symbol, timeframe]
    if start_ts:
        q += " AND timestamp >= ?"
        params.append(start_ts)
    q += " ORDER BY timestamp ASC LIMIT ?"
    params.append(int(limit))
    df = pd.read_sql_query(q, conn, params=params, parse_dates=["timestamp"])
    conn.close()
    return df


def load_ohlcv_from_sqlite(sqlite_path: str, table: str = "ohlcv", symbol: Optional[str] = None, limit: int = 2000, time_col: str = "timestamp") -> pd.DataFrame:
    p = Path(sqlite_path)
    if not p.exists():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")
    conn = sqlite3.connect(str(p), detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    params = []
    q = f"SELECT * FROM {table}"
    if symbol:
        q += " WHERE symbol = ?"
        params.append(symbol)
    # try to use provided time_col; if not present DB will raise -> caller sees explicit error
    q += " ORDER BY " + time_col + " ASC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(q, conn, params=params, parse_dates=[time_col])
    conn.close()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)
        if "timestamp" not in df.columns:
            df = df.rename(columns={time_col: "timestamp"})
    return df


def fetch_ccxt_ohlcv(symbol: str, timeframe: str = "1m", limit: int = 1000, exchange_id: str = "binance", api_key: Optional[str] = None, api_secret: Optional[str] = None, sandbox: bool = True) -> pd.DataFrame:
    if ccxt is None:
        raise RuntimeError("ccxt no instalado")
    cls = getattr(ccxt, exchange_id, None)
    if cls is None:
        raise RuntimeError(f"Exchange {exchange_id} not found")
    params = {"enableRateLimit": True}
    if api_key and api_secret:
        params.update({"apiKey": api_key, "secret": api_secret})
    ex = cls(params)
    if sandbox:
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            pass
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


# ----------------------------
# Exports
# ----------------------------
__all__ = [
    "normalize_timeframe",
    "normalize_symbol",
    "to_tensor_float32",
    "ensure_float32_tensor",
    "seed_everything",
    "add_technical_features",
    "create_sequences_from_df",
    "LSTM2Head",
    "train_epoch",
    "eval_epoch",
    "quantize_price",
    "quantize_amount",
    "simulate_fill",
    "save_model",
    "load_model",
    "save_scaler",
    "load_scaler",
    "build_dataset_for_training",
    "prepare_model_input",
    "predict_with_model",
    "simulate_ou",
    "load_data_from_sqlite",
    "load_ohlcv_from_sqlite",
    "fetch_ccxt_ohlcv",
]
