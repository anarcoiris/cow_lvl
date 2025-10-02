#!/usr/bin/env python3
"""
Evolve & Trade (Fixed & Hardened) - MODIFIED

Incluye mejoras:
 - prepare_input_for_model: pipeline de input robusto (evita doble-scaling)
 - ensure_scaler_feature_names: fallback para assignar feature_names_in_ desde meta
 - simulate_fill: unificada (modo simple + modo dependiente de liquidez/size)
 - deterministic ordering en add_technical_features
 - load_model devuelve (model, meta) y hace comprobaciones post-load
 - seed_everything protegido si torch == None
 - backtest_market_maker acepta optional scaler para preparar entradas
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import pandas as pd

# matplotlib only used when --plot is set
import matplotlib.pyplot as plt  # noqa: F401
from tqdm import trange

# Optional torch/scikit imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

# Optional DEAP
try:
    from deap import base, creator, tools, algorithms  # type: ignore
    _HAS_DEAP = True
except Exception:  # pragma: no cover
    _HAS_DEAP = False

# Influx optional
try:
    from influxdb_client import InfluxDBClient  # type: ignore
    _HAS_INFLUX = True
except Exception:  # pragma: no cover
    _HAS_INFLUX = False

# pandas_ta optional (fallbacks provided)
try:
    import pandas_ta as pta  # type: ignore
    _HAS_PT = True
except Exception:  # pragma: no cover
    _HAS_PT = False

# joblib optional
try:
    import joblib
except Exception:
    joblib = None

LOGGER = logging.getLogger("evolve_trade")


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ==========================================
# Seeding / determinism (guarded)
# ==========================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # set deterministic cuDNN flags if available
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
        except Exception:
            # torch might fail in some minimal python builds; ignore
            pass


# ==========================================
# Simple indicator fallbacks
# ==========================================
def simple_rsi(close: Sequence[float], length: int = 14) -> np.ndarray:
    c = pd.Series(close, dtype=float)
    diff = c.diff(1)
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


# ==========================================
# Data loading (Influx or simulated OU)
# ==========================================
def load_data_from_influx(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    timeframe: str = "30m",
    measurement: str = "prices",
    field: str = "close",
    start: str = "-70d",
    limit: int = 100000,
    influx_cfg: Optional[dict] = None,
) -> np.ndarray:
    if not _HAS_INFLUX:
        raise RuntimeError("Influx client no instalado. pip install influxdb-client")

    INFLUX_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN", None)
    INFLUX_ORG = os.getenv("INFLUXDB_ORG", None)
    INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "prices")

    if influx_cfg:
        INFLUX_URL = influx_cfg.get("url", INFLUX_URL)
        INFLUX_TOKEN = influx_cfg.get("token", INFLUX_TOKEN)
        INFLUX_ORG = influx_cfg.get("org", INFLUX_ORG)
        INFLUX_BUCKET = influx_cfg.get("bucket", INFLUX_BUCKET)

    if INFLUX_TOKEN is None or INFLUX_ORG is None:
        raise RuntimeError("Influx token/org no configurado. Exporta INFLUXDB_TOKEN e INFLUXDB_ORG.")

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start})
      |> filter(fn: (r) => r["_measurement"] == "{measurement}")
      |> filter(fn: (r) => r["symbol"] == "{symbol}")
      |> filter(fn: (r) => r["exchange"] == "{exchange}")
      |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
      |> filter(fn: (r) => r["_field"] == "{field}")
      |> sort(columns: ["_time"], desc: false)
      |> limit(n: {limit})
    '''

    LOGGER.info("Querying Influx %s %s %s %s", symbol, exchange, timeframe, field)
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        tables = client.query_api().query(query)
        values: List[float] = []
        for table in tables:
            for record in table.records:
                values.append(float(record.get_value()))
    if not values:
        raise ValueError("No se obtuvieron datos desde Influx.")
    return np.asarray(values, dtype=np.float32)


def simulate_ou(theta: float = 1.0, sigma: float = 0.5, dt: float = 0.01, steps: int = 5000, seed: Optional[int] = None):
    """
    Simula un proceso OU y devuelve arrays (close, high, low) tipo np.float32.
    Usa numpy.random.Generator para reproducibilidad.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    x = np.zeros(steps + 1, dtype=np.float32)

    # Genera increments dW como vector — evita scalars y .astype sobre float
    dw = rng.normal(loc=0.0, scale=np.sqrt(dt), size=steps).astype(np.float32)

    for t in range(steps):
        x[t + 1] = x[t] - theta * x[t] * dt + sigma * dw[t]

    price = 100 + np.cumsum(x)  # price-like series

    # Genera OHLC plausibles
    noise = 0.001 * np.abs(rng.standard_normal(len(price))).astype(np.float32)
    high = price * (1.0 + noise)
    low = price * (1.0 - noise)
    # Garantiza low <= high
    low = np.minimum(low, high * 0.9999)

    return price.astype(np.float32), high.astype(np.float32), low.astype(np.float32)

def add_technical_features(
    close: Union[np.ndarray, pd.Series],
    high: Optional[Union[np.ndarray, pd.Series]] = None,
    low: Optional[Union[np.ndarray, pd.Series]] = None,
    volume: Optional[Union[np.ndarray, pd.Series]] = None,
    window_long: int = 50,
    include_vp: bool = False,
    fib_retracements: Optional[Sequence[float]] = None,
    fib_extensions: Optional[Sequence[float]] = None,
    include_fib_dist_cols: bool = True,
    fib_composite: Optional[Dict] = None,
    fib_lookback: int = 50,
    dropna_after: bool = False,
    out_dtype: str = "float32",
) -> pd.DataFrame:
    # --- defaults
    if fib_retracements is None:
        fib_retracements = [0.236, 0.382, 0.5, 0.618, 0.786]
    if fib_extensions is None:
        fib_extensions = [1.272, 1.618, 2.0]

    # Convert inputs to Series to preserve index if present
    if isinstance(close, np.ndarray):
        idx = None
        close_s = pd.Series(close)
    else:
        close_s = pd.Series(close)
        idx = close_s.index

    df = pd.DataFrame({"close": close_s})
    if high is None:
        high_s = close_s.copy()
    else:
        high_s = pd.Series(high, index=idx) if not isinstance(high, pd.Series) else high
    if low is None:
        low_s = close_s.copy()
    else:
        low_s = pd.Series(low, index=idx) if not isinstance(low, pd.Series) else low
    df["high"] = high_s
    df["low"] = low_s

    # safe log_close
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_close"] = np.log(df["close"].replace(0, np.nan))

    # returns (keep both real and log for compatibility; prefer log for modeling)
    df["ret_1"] = df["close"].pct_change(1)
    df["log_ret_1"] = df["log_close"].diff(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["log_ret_5"] = df["log_close"].diff(5)

    # moving averages and EMAs (short/medium/long + window_long)
    MA_WINDOWS = [5, 20, 50]
    EMA_WINDOWS = [5, 20]
    if window_long not in MA_WINDOWS:
        MA_WINDOWS.append(window_long)
    if window_long not in EMA_WINDOWS:
        EMA_WINDOWS.append(window_long)

    for L in sorted(set(MA_WINDOWS)):
        df[f"sma_{L}"] = df["close"].rolling(L, min_periods=L).mean()
    for L in sorted(set(EMA_WINDOWS)):
        df[f"ema_{L}"] = df["close"].ewm(span=L, adjust=False).mean()

    # representative MA diff (prefer 20 if present)
    if "sma_20" in df.columns:
        df["ma_diff_20"] = df["close"] - df["sma_20"]
    else:
        L = sorted(set(MA_WINDOWS))[0]
        df[f"ma_diff_{L}"] = df["close"] - df[f"sma_{L}"]

    # Bollinger 20
    bb_m = df["close"].rolling(20, min_periods=20).mean()
    bb_std = df["close"].rolling(20, min_periods=20).std()
    df["bb_m"] = bb_m
    df["bb_std"] = bb_std
    df["bb_up"] = bb_m + 2.0 * bb_std
    df["bb_dn"] = bb_m - 2.0 * bb_std
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / bb_m.replace(0, np.nan)

    # RSI (pandas_ta fallback else simple)
    try:
        import pandas_ta as pta  # type: ignore
        df["rsi_14"] = pta.rsi(df["close"], length=14)
    except Exception:
        s = df["close"]
        diff = s.diff(1)
        up = diff.clip(lower=0)
        down = -diff.clip(upper=0)
        ma_up = up.rolling(14, min_periods=14).mean()
        ma_down = down.rolling(14, min_periods=14).mean()
        rs = ma_up / (ma_down + 1e-12)
        df["rsi_14"] = (100 - (100 / (1 + rs))).to_numpy()

    # ATR
    try:
        df["atr_14"] = atr_simple(df["high"].values, df["low"].values, df["close"].values, length=14)
    except Exception:
        df["atr_14"] = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()

    # volatility (log returns preferred)
    df["raw_vol_10"] = df["log_ret_1"].rolling(10, min_periods=10).std()
    df["raw_vol_30"] = df["log_ret_1"].rolling(30, min_periods=30).std()

    # Fibonacci (retracements + extensions) using fib_lookback
    highs = df["high"].rolling(fib_lookback, min_periods=fib_lookback).max()
    lows = df["low"].rolling(fib_lookback, min_periods=fib_lookback).min()
    range_ = (highs - lows).replace(0, np.nan)

    all_f_levels = []
    # retracements (0..1)
    for f in fib_retracements:
        lvl = lows + (highs - lows) * float(f)
        tag = f"fib_r_{int(f*1000)}"
        df[tag] = lvl
        if include_fib_dist_cols:
            df[f"dist_{tag}"] = (df["close"] - lvl) / range_
        all_f_levels.append(("r", f, tag))

    # extensions (>1)
    for f in fib_extensions:
        lvl = lows + (highs - lows) * float(f)
        tag = f"fibext_{int(f*1000)}"
        df[tag] = lvl
        if include_fib_dist_cols:
            df[f"dist_{tag}"] = (df["close"] - lvl) / range_
        all_f_levels.append(("ext", f, tag))

    # TD-like setups (simple)
    close_v = df["close"].values
    buy_setup = np.zeros(len(close_v), dtype=int)
    sell_setup = np.zeros(len(close_v), dtype=int)
    for i in range(4, len(close_v)):
        buy_setup[i] = buy_setup[i - 1] + 1 if close_v[i] < close_v[i - 4] else 0
        sell_setup[i] = sell_setup[i - 1] + 1 if close_v[i] > close_v[i - 4] else 0
    df["td_buy_setup"] = buy_setup
    df["td_sell_setup"] = sell_setup

    # Volume profile approximation (optional, expensive)
    if include_vp and (volume is not None):
        # si volume es array/series, respetar índice
        vol_s = pd.Series(volume, index=idx) if not isinstance(volume, pd.Series) else volume
        df["volume"] = vol_s
        window = 100
        pocs = [np.nan] * len(df)
        for i in range(window, len(df)):
            seg = df["close"].iloc[i - window : i]
            seg_vol = df["volume"].iloc[i - window : i].values
            if np.nansum(seg_vol) <= 0:
                pocs[i] = np.nan
                continue
            hist, edges = np.histogram(seg.values, bins=20, weights=seg_vol)
            max_bin = int(np.argmax(hist))
            pocs[i] = 0.5 * (edges[max_bin] + edges[max_bin + 1])
        df["vp_poc"] = pocs
        df["dist_vp"] = (df["close"] - df["vp_poc"]) / df["close"].replace(0, np.nan)
    else:
        # conservar `volume` sólo si fue pasado por el caller y no existe ya
        if volume is not None and "volume" not in df.columns:
            df["volume"] = pd.Series(volume, index=idx)

    # Optional: compute fib composite scalar (compute dist internally if needed)
    if fib_composite:
        method = fib_composite.get("method", "cubic_sum")
        weights = fib_composite.get("weights", None)
        normalize = bool(fib_composite.get("normalize", True))

        # collect dist columns if created
        dist_cols = [f"dist_{tag}" for (_t, _fval, tag) in all_f_levels if f"dist_{tag}" in df.columns]

        if len(dist_cols) == 0:
            # compute dist array on the fly (no column creation)
            dist_list = []
            for (_t, fval, tag) in all_f_levels:
                lvl = df[tag].values
                dist_list.append((df["close"].values - lvl) / range_.values)
            if dist_list:
                dist_arr = np.vstack(dist_list).T
            else:
                dist_arr = None
        else:
            dist_arr = df[dist_cols].values if len(dist_cols) > 0 else None

        if dist_arr is None or dist_arr.shape[1] == 0:
            df["fib_composite"] = np.nan
        else:
            K = dist_arr.shape[1]
            if weights is None:
                w = np.ones(K, dtype=float)
            else:
                w = np.asarray(weights, dtype=float)
                if w.size != K:
                    if w.size == 1:
                        w = np.repeat(w, K)
                    else:
                        w = np.resize(w, K)
            if method == "cubic_sum":
                with np.errstate(invalid="ignore"):
                    comp = np.nansum((dist_arr ** 3) * w.reshape(1, -1), axis=1)
                if normalize:
                    denom = np.sum(np.abs(w)) or 1.0
                    comp = comp / denom
                df["fib_composite"] = comp
            else:
                comp = np.nansum(dist_arr * w.reshape(1, -1), axis=1)
                if normalize:
                    denom = np.sum(np.abs(w)) or 1.0
                    comp = comp / denom
                df["fib_composite"] = comp

    # Deterministic ordering
    ordered = [
        "close", "high", "low", "log_close",
        "ret_1", "log_ret_1", "ret_5", "log_ret_5",
    ]
    for L in sorted(set(MA_WINDOWS)):
        ordered.append(f"sma_{L}")
    for L in sorted(set(EMA_WINDOWS)):
        ordered.append(f"ema_{L}")
    ordered += ["ma_diff_20", "bb_m", "bb_std", "bb_up", "bb_dn", "bb_width", "rsi_14", "atr_14", "raw_vol_10", "raw_vol_30"]
    for (_t, fval, tag) in all_f_levels:
        if tag in df.columns:
            ordered.append(tag)
            if include_fib_dist_cols:
                ordered.append(f"dist_{tag}")
    ordered += ["td_buy_setup", "td_sell_setup"]
    if "volume" in df.columns:
        ordered += ["volume", "vp_poc", "dist_vp"]
    if "fib_composite" in df.columns:
        ordered.append("fib_composite")

    remaining = [c for c in df.columns if c not in ordered]
    final_cols = [c for c in ordered if c in df.columns] + remaining
    df = df.loc[:, final_cols]

    if dropna_after:
        df = df.dropna().reset_index(drop=True)

    # ---- final cleanup antes de devolver ----
    # coercer objetos a numérico donde proceda, reemplazar infinitos por NaN
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # cast final de float64 -> out_dtype si procede
    if out_dtype == "float32":
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype(np.float32)

    return df



# ------------------------------
# Multi-timeframe helpers
# ------------------------------
# (kept unchanged from original with some defensive typing)
def build_ohlc_df_from_close(close: np.ndarray, freq_alias: str = "1T", start_ts: Optional[pd.Timestamp]=None) -> pd.DataFrame:
    n = len(close)
    if start_ts is None:
        start_ts = pd.Timestamp("2020-01-01 00:00:00")
    idx = pd.date_range(start=start_ts, periods=n, freq="1T")
    s = pd.Series(close, index=idx)
    ohlc = s.resample(freq_alias).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    ohlc = ohlc.ffill().bfill()
    ohlc["volume"] = np.nan
    return ohlc


def create_multi_tf_merged_df_from_base(
    close_base: np.ndarray,
    base_freq: str = "1T",
    tfs: Sequence[str] = ("1T", "4H", "1D"),
    start_ts: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if start_ts is None:
        start_ts = pd.Timestamp("2020-01-01 00:00:00")
    idx_base = pd.date_range(start=start_ts, periods=len(close_base), freq=base_freq)
    df_base = pd.DataFrame({"close": close_base}, index=idx_base)

    tf_dfs: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        if tf == base_freq:
            df_tf = df_base.copy()
            df_tf["high"] = df_tf["close"] * (1.0 + 0.0005 * np.abs(np.random.randn(len(df_tf))))
            df_tf["low"] = df_tf["close"] * (1.0 - 0.0005 * np.abs(np.random.randn(len(df_tf))))
            df_tf["volume"] = np.nan
        else:
            s = df_base["close"]
            ohlc = s.resample(tf).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            ohlc = ohlc.ffill().bfill()
            df_tf = pd.DataFrame({"close": ohlc["close"].values, "high": ohlc["high"].values, "low": ohlc["low"].values}, index=ohlc.index)
            df_tf["volume"] = np.nan

        df_tf_feats = add_technical_features(df_tf["close"].values, high=df_tf["high"].values, low=df_tf["low"].values, volume=None)
        tag = tf.replace("T", "m").replace("H", "h").replace("D", "d")  # crude mapping
        df_tf_feats = df_tf_feats.add_suffix(f"_{tag}")
        tf_dfs[tf] = df_tf_feats

    merged = df_base.copy()
    for tf, df_feats in tf_dfs.items():
        df_up = df_feats.reindex(idx_base, method="ffill")
        cols_keep = [c for c in df_up.columns if not c.startswith("close_")]
        merged = pd.concat([merged, df_up[cols_keep]], axis=1)

    merged = merged.ffill().bfill()
    return merged, tf_dfs


# ==========================================
# Sequences for LSTM
# ==========================================
def create_sequences_from_df(
    df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int = 32, horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert seq_len >= 1 and horizon >= 1
    dfc = df.reset_index(drop=True).copy()

    if "close" not in dfc.columns:
        raise RuntimeError("create_sequences_from_df expects column 'close' in df")

    present = [c for c in feature_cols if c in dfc.columns]
    if len(present) == 0:
        raise RuntimeError("No feature columns present after filtering.")

    arr = dfc[present].astype(np.float32).values
    close = dfc["close"].astype(np.float64).values  # use float64 for log stability

    M = len(dfc)
    N = M - seq_len - horizon + 1
    if N <= 0:
        raise ValueError(f"Not enough rows: need >= seq_len + horizon ({seq_len + horizon}), got {M}")

    F = arr.shape[1]
    X = np.zeros((N, seq_len, F), dtype=np.float32)
    y_ret = np.zeros((N,), dtype=np.float32)
    y_vol = np.zeros((N,), dtype=np.float32)

    if np.any(close <= 0):
        raise RuntimeError("Non-positive close values present; cannot compute log-returns. Clean data first.")

    logc = np.log(close)

    for i in range(N):
        X[i] = arr[i : i + seq_len]
        t0 = i + seq_len - 1
        t_h = t0 + horizon
        y_ret[i] = float(logc[t_h] - logc[t0])
        if horizon >= 1:
            rets = logc[t0 + 1 : t_h + 1] - logc[t0 : t_h]
            y_vol[i] = float(np.std(rets, ddof=0)) if rets.size > 0 else 0.0
        else:
            y_vol[i] = 0.0

    return X.astype(np.float32), y_ret.astype(np.float32), y_vol.astype(np.float32)


# ==========================================
# LSTM model with two heads (ret, vol)
# ==========================================
if torch is not None:
    class LSTM2Head(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0)
            )
            self.head_ret = nn.Sequential(
                nn.Linear(hidden_size, max(4, hidden_size // 2)), nn.ReLU(), nn.Linear(max(4, hidden_size // 2), 1)
            )
            self.head_vol = nn.Sequential(
                nn.Linear(hidden_size, max(4, hidden_size // 2)), nn.ReLU(), nn.Linear(max(4, hidden_size // 2), 1), nn.Softplus()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            ret = self.head_ret(last).squeeze(-1)
            vol = self.head_vol(last).squeeze(-1)
            return ret, vol
else:
    LSTM2Head = None


# ==========================================
# Training / evaluation
# ==========================================
def _ensure_shape_1d(t: "torch.Tensor") -> "torch.Tensor":
    if t is None:
        return t
    if hasattr(t, "dim") and t.dim() == 2 and t.size(1) == 1:
        return t.view(-1)
    if hasattr(t, "dim") and t.dim() >= 1:
        return t.squeeze(-1)
    return t


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha_vol_loss: float = 0.5,
    grad_clip: float = 1.0,
) -> float:
    if torch is None:
        raise RuntimeError("torch no disponible")
    model.train()
    total_loss = 0.0
    n = 0
    mse = nn.MSELoss()
    for batch in loader:
        if len(batch) == 3:
            xb, yret, yvol = batch
        else:
            xb = batch[0]; yret = batch[1]; yvol = batch[2]
        xb = xb.to(device)
        yret = yret.to(device)
        yvol = yvol.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred_ret, pred_vol = model(xb)
        loss_ret = mse(pred_ret, yret)
        loss_vol = mse(pred_vol, yvol)
        loss = loss_ret + alpha_vol_loss * loss_vol
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(1, n)


def eval_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device, alpha_vol_loss: float = 0.5
) -> float:
    if torch is None:
        raise RuntimeError("torch no disponible")
    model.eval()
    total_loss = 0.0
    n = 0
    mse = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                xb, yret, yvol = batch
            else:
                xb = batch[0]; yret = batch[1]; yvol = None
            xb = xb.to(device)
            yret = yret.to(device)
            yret = _ensure_shape_1d(yret)
            if yvol is not None:
                yvol = yvol.to(device)
                yvol = _ensure_shape_1d(yvol)
            pred_ret, pred_vol = model(xb)
            loss_ret = mse(pred_ret, yret)
            loss_vol = mse(pred_vol, yvol) if yvol is not None else torch.tensor(0.0, device=device)
            loss = loss_ret + alpha_vol_loss * loss_vol
            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
    return total_loss / max(1, n)


# --- Fill simulation utilities (combined + liquidity-aware) ---
def quantize_price(price: float, tick: Optional[float]) -> float:
    if tick is None or tick == 0:
        return float(price)
    q = float(price) / float(tick)
    q_round = round(q)
    return float(q_round * float(tick))


def quantize_amount(amount: float, lot: Optional[float]) -> float:
    if lot is None or lot == 0:
        return float(amount)
    q = float(amount) / float(lot)
    q_floor = math.floor(q)
    return float(q_floor * float(lot))


def simulate_fill(
    order_price: float,
    side: str,
    future_segment: pd.Series = None,
    slippage_tolerance_pct: float = 0.5,
    tick: Optional[float] = None,
    lot: Optional[float] = None,
    method: str = "slippage",
    *,
    # optional liquidity-aware args
    use_liquidity_model: bool = False,
    order_notional: Optional[float] = None,
    future_volumes: Optional[Sequence[float]] = None,
    avg_vol_window: int = 10,
    impact_k: float = 2.0,
    impact_alpha: float = 0.8,
    max_slippage_frac: float = 0.5,
):
    """
    Simula el fill de una orden limit en el backtester.

    - future_segment: pd.Series, con índice igual al df original (puede ser RangeIndex o DatetimeIndex).
                      Para BUY debe pasarse la serie de LOWs (porque la orden de compra se llena si low <= order_price).
                      Para SELL debe pasarse la serie de HIGHs (porque la orden de venta se llena si high >= order_price).
    - Basic behavior (compatibilidad): si use_liquidity_model==False, se usa el primer-hit simple
      como en la implementación original (slippage simple en porcentaje).
    - Liquidity model: si use_liquidity_model==True y se provee order_notional y future_volumes (o avg_vol_window),
      se calculará slippage en función del tamaño relativo al volumen y se devolverá fraction filled estimada.
    - Returns: (filled: bool, executed_price: float|None, fill_label_or_fraction: index label | filled_fraction)
      - In simple mode, third return is fill_label (index label)
      - In liquidity mode, third return is filled_fraction (0..1)
    """
    if future_segment is None or len(future_segment) == 0:
        return False, None, None

    side = side.lower()
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    # SIMPLE first-hit mode (backwards compatible)
    if not use_liquidity_model:
        if side == "buy":
            cond = future_segment <= order_price
        else:
            cond = future_segment >= order_price
        hits = future_segment[cond]
        if hits.empty:
            return False, None, None
        fill_label = hits.index[0]
        executed_price = float(order_price)
        slippage = executed_price * (float(slippage_tolerance_pct) / 100.0)
        if method == "slippage":
            if side == "buy":
                executed_price = executed_price + slippage
            else:
                executed_price = executed_price - slippage
        if tick:
            executed_price = quantize_price(executed_price, tick)
        return True, float(executed_price), fill_label

    # LIQUIDITY-AWARE MODE
    # If future_volumes provided we attempt to compute avg volume
    avg_vol = None
    try:
        if future_volumes is not None:
            fv = np.asarray(future_volumes, dtype=float)
            if len(fv) >= 1:
                avg_vol = float(np.nanmean(fv[:avg_vol_window]))
    except Exception:
        avg_vol = None

    # fallback estimate if vol unknown
    if avg_vol is None or avg_vol <= 0:
        # if unknown, degrade gracefully using simple mode but smaller slippage
        # attempt simple first-hit but with reduced slippage because we don't know liquidity
        if side == "buy":
            cond = future_segment <= order_price
        else:
            cond = future_segment >= order_price
        hits = future_segment[cond]
        if hits.empty:
            return False, None, 0.0
        fill_label = hits.index[0]
        executed_price = float(order_price)
        slippage = executed_price * (float(slippage_tolerance_pct) / 100.0) * 0.5
        if method == "slippage":
            if side == "buy":
                executed_price = executed_price + slippage
            else:
                executed_price = executed_price - slippage
        if tick:
            executed_price = quantize_price(executed_price, tick)
        # estimate filled_fraction conservatively
        return True, float(executed_price), 0.5

    # compute relative size measure 'rel' (order_notional in same units as avg_vol)
    if order_notional is None:
        # if not provided, attempt to estimate notional from price * arbitrary amount => fallback
        order_notional = float(order_price)

    rel = order_notional / (avg_vol + 1e-12)
    slippage_frac = min(max_slippage_frac, impact_k * (rel ** impact_alpha))
    if side == "buy":
        executed_price = order_price * (1.0 + slippage_frac)
    else:
        executed_price = order_price * (1.0 - slippage_frac)
    if tick:
        executed_price = quantize_price(executed_price, tick)

    # estimate filled fraction (simple heuristic)
    filled_fraction = float(np.clip(1.0 / (1.0 + rel), 0.0, 1.0))
    return True, float(executed_price), float(filled_fraction)


# ==========================================
# Save / load model & scaler (improved)
# ==========================================
def save_model(model: Any, path_model: Path, meta: Optional[Dict] = None, path_meta: Optional[Path] = None) -> None:
    if torch is None:
        raise RuntimeError("torch no disponible")
    path_model = Path(path_model)
    meta = {} if meta is None else meta
    os.makedirs(path_model.parent, exist_ok=True)
    if path_meta is None and meta:
        payload = {"state": model.state_dict(), "meta": meta}
        torch.save(payload, str(path_model))
        LOGGER.info("Saved combined model+meta to %s", path_model)
    else:
        torch.save(model.state_dict(), str(path_model))
        if meta is not None and path_meta is not None:
            with open(path_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        LOGGER.info("Saved model to %s and meta to %s", path_model, path_meta)


def load_model(path_model: Path, input_size: Optional[int] = None, hidden: Optional[int] = None, device: Optional[torch.device] = None) -> Tuple[Any, Dict]:
    """
    Carga un state_dict o un payload combinado y reconstruye un LSTM2Head.
    Devuelve (model, meta).
    """
    if torch is None:
        raise RuntimeError("torch no disponible")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_model = Path(path_model)
    if not path_model.exists():
        raise FileNotFoundError(f"Model not found: {path_model}")

    obj = torch.load(str(path_model), map_location=device)

    state = obj
    meta: Dict = {}
    if isinstance(obj, dict):
        if "state" in obj and isinstance(obj["state"], dict):
            state = obj["state"]
            meta = obj.get("meta", {}) or {}
        elif "model_state_dict" in obj:
            state = obj["model_state_dict"]
        elif "state_dict" in obj:
            state = obj["state_dict"]
        else:
            sidecar = path_model.parent / "meta.json"
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

    if input_size is None and inferred_input is not None:
        input_size = int(inferred_input)
    if hidden is None and inferred_hidden is not None:
        hidden = int(inferred_hidden)

    # prefer meta input_size if present
    try:
        if meta and "input_size" in meta and meta.get("input_size") is not None:
            input_size = int(meta.get("input_size"))
    except Exception:
        pass

    if input_size is None:
        raise RuntimeError("Could not infer input_size. Provide input_size in meta when saving or pass it to load_model.")

    # construct model
    if LSTM2Head is None:
        raise RuntimeError("LSTM2Head no disponible en este entorno (torch faltante).")
    model = LSTM2Head(input_size=int(input_size), hidden_size=int(hidden or 64)).to(device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)

    # post-load checks and warnings
    try:
        meta_out = dict(meta or {})
        if "feature_cols" in meta_out:
            try:
                declared_input = int(meta_out.get("input_size", len(meta_out.get("feature_cols"))))
                if declared_input != int(input_size):
                    LOGGER.warning("meta.input_size (%s) != inferred input_size (%s). Using inferred input_size.", declared_input, input_size)
            except Exception:
                pass
    except Exception:
        pass

    model.eval()
    return model, meta


def save_scaler(scaler: Any, path: Path) -> None:
    if joblib is None:
        raise RuntimeError("joblib no disponible")
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    joblib.dump(scaler, str(path))


def load_scaler(path: Path, meta: Optional[Dict] = None) -> Any:
    """
    Load scaler and optionally ensure feature_names_in_ using meta['feature_cols'].
    """
    if joblib is None:
        raise RuntimeError("joblib no disponible")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    sc = joblib.load(str(path))
    sc = ensure_scaler_feature_names(sc, meta)
    return sc


def ensure_scaler_feature_names(scaler: Any, meta: Optional[Dict] = None) -> Any:
    """
    If scaler is missing feature_names_in_, try to set it from meta['feature_cols'].
    This avoids silent column-order mismatches at inference time.
    """
    if scaler is None:
        return scaler
    if hasattr(scaler, "feature_names_in_"):
        return scaler
    if meta is not None and isinstance(meta.get("feature_cols"), (list, tuple)):
        try:
            scaler.feature_names_in_ = np.array(meta["feature_cols"], dtype=object)
        except Exception:
            pass
    return scaler


# ----------------------------
# Build dataset helper
# ----------------------------
def build_dataset_for_training(df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int, horizon: int, val_frac: float = 0.2, scaler: Optional[Any] = None) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray], Any]:
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
    train_rows_end = seq_len + n_train_seq - 1
    scaler.fit(df_local[list(feature_cols)].iloc[: train_rows_end + 1].values.astype(np.float64))
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
    Compat: older helper kept for simple usage. Returns torch tensor (1,seq_len,F) on given device.
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


def prepare_input_for_model(df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int, scaler: Optional[Any] = None, method: str = "per_row"):
    """
    Prepara un tensor (1, seq_len, F) listo para pasar al modelo.
    - df: DataFrame con features (raw or already scaled if scaler is None)
    - feature_cols: order expected by model
    - scaler: sklearn-like scaler or None
    - method: 'per_row' (transform each row) or 'flat' (flatten then transform)
    Returns torch.Tensor dtype float32 (CPU); caller should .to(device).
    Raises if scaler.feature_names_in_ mismatches columns (prevents silent misalignment).
    """
    if torch is None:
        raise RuntimeError("torch no disponible")
    if len(df) < seq_len:
        raise ValueError("Not enough rows for seq_len")

    last = df.iloc[-seq_len:][list(feature_cols)].astype(np.float64).values  # shape (seq_len, F)

    if scaler is not None:
        # Check feature_names_in_ alignment
        if hasattr(scaler, "feature_names_in_"):
            expected = np.asarray(scaler.feature_names_in_, dtype=object)
            got = np.asarray(feature_cols, dtype=object)
            if set(expected) != set(got):
                raise RuntimeError("Scaler.feature_names_in_ and feature_cols mismatch (different sets). Aborting to avoid misaligned transforms.")
            if not np.array_equal(expected, got):
                # reorder columns to match scaler
                order_idx = [list(got).index(name) for name in expected]
                last = last[:, order_idx]

        if method == "per_row":
            try:
                scaled = scaler.transform(last.astype(np.float64)).astype(np.float32)
            except Exception:
                scaled = last.astype(np.float32)
            arr = scaled
        else:
            flat = last.astype(np.float64).reshape(-1, last.shape[1])
            try:
                flat_s = scaler.transform(flat).astype(np.float32)
            except Exception:
                flat_s = flat.astype(np.float32)
            arr = flat_s.reshape(seq_len, last.shape[1]).astype(np.float32)
    else:
        arr = last.astype(np.float32)

    t = torch.from_numpy(arr).unsqueeze(0)  # shape (1, seq_len, F)
    return t


def predict_with_model(model: Any, input_tensor: Any) -> Tuple[float, float]:
    if torch is None:
        raise RuntimeError("torch no disponible")
    model.eval()
    with torch.no_grad():
        out_ret, out_vol = model(input_tensor)
    return float(out_ret.cpu().numpy().ravel()[0]), float(out_vol.cpu().numpy().ravel()[0])


# ==========================================
# Backtester (market making / scalp) - updated to use prepare_input_for_model
# ==========================================
def backtest_market_maker(
    df: pd.DataFrame,
    model: nn.Module,
    feature_cols: Sequence[str],
    seq_len: int,
    horizon: int,
    slippage_tolerance_pct: float = 0.5,
    tick: float = None,
    lot: float = None,
    fill_method: str = "slippage",
    capital: float = 1_000_000.0,
    intraday_frac: float = 0.10,
    sigma_levels: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    tp_sigma: float = 0.5,
    sl_sigma: float = 1.0,
    trend_weighting: bool = True,
    swing_window: int = 100,
    scaler: Optional[Any] = None,  # NEW optional arg: scaler used to transform feature rows
) -> Tuple[dict, List[dict]]:
    device = next(model.parameters()).device
    df_loc = df.copy().reset_index(drop=True)
    N = len(df_loc)

    df_loc["_ret_1"] = df_loc["close"].pct_change(1)
    df_loc["_raw_vol_30"] = df_loc["_ret_1"].rolling(30, min_periods=30).std()
    df_loc["sma_50"] = df_loc["close"].rolling(50, min_periods=50).mean()
    df_loc["sma_200"] = df_loc["close"].rolling(200, min_periods=200).mean()
    df_loc["swing_delta"] = df_loc["close"].pct_change(swing_window)

    intraday_cap = capital * intraday_frac
    cash_intraday = intraday_cap
    trades: List[dict] = []

    intraday_exposure = 1.0
    last_vol_spike_idx = -9999
    vol_spike_cooldown = max(10, int(1.0 / max(1, horizon)))
    max_pct_per_order = 0.02
    risk_per_trade_pct = 0.005
    single_level_cap = 0.35

    start_idx = max(seq_len, 200, swing_window + 1)
    if N - start_idx <= horizon + 1:
        raise ValueError("Not enough history for backtest given seq_len/horizon")

    LOGGER.info("Backtest start=%d end=%d N=%d", start_idx, N, N)

    raw_fracs = np.array([0.4 * (0.7 ** i) for i in range(len(sigma_levels))], dtype=float)
    raw_fracs = raw_fracs / raw_fracs.sum()
    level_fracs = np.minimum(raw_fracs, single_level_cap)
    if level_fracs.sum() <= 0:
        level_fracs = raw_fracs
    else:
        level_fracs = level_fracs / level_fracs.sum()

    for t in range(start_idx, N - horizon - 1):
        # Build model input window using prepare_input_for_model (handles scaler/order)
        seq_df = df_loc.iloc[t - seq_len : t].reset_index(drop=True)
        try:
            x_t = prepare_input_for_model(seq_df, feature_cols, seq_len, scaler=scaler, method="per_row").to(device)
        except Exception as e:
            # fallback to previous method if prepare fails (but warn)
            LOGGER.debug("prepare_input_for_model failed at t=%d: %s. Falling back to raw np array.", t, e)
            seq_x = df_loc[feature_cols].iloc[t - seq_len : t].values.astype(np.float32)
            x_t = torch.from_numpy(seq_x).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_ret, pred_vol = model(x_t)
        try:
            pred_ret = float(pred_ret.cpu().numpy().ravel()[0])
        except Exception:
            pred_ret = float(pred_ret.item()) if hasattr(pred_ret, "item") else float(pred_ret)
        try:
            pred_vol = float(pred_vol.cpu().numpy().ravel()[0])
        except Exception:
            pred_vol = float(pred_vol.item()) if hasattr(pred_vol, "item") else float(pred_vol)

        if not math.isfinite(pred_vol) or pred_vol <= 1e-6:
            pred_vol = float(df_loc["_raw_vol_30"].iloc[t]) if pd.notna(df_loc["_raw_vol_30"].iloc[t]) else 0.001
            pred_vol = max(pred_vol, 1e-4)

        trend_bias = 0.0
        if trend_weighting:
            sma50 = df_loc["sma_50"].iloc[t]
            sma200 = df_loc["sma_200"].iloc[t]
            if pd.notna(sma50) and pd.notna(sma200):
                direction = 1.0 if sma50 > sma200 else -1.0
                strength = abs(df_loc["swing_delta"].iloc[t]) if pd.notna(df_loc["swing_delta"].iloc[t]) else 0.0
                trend_bias = direction * min(1.0, strength * 10.0)

        base_long_share = 0.5 + 0.5 * math.tanh(pred_ret * 50.0)
        long_share = base_long_share * 0.7 + (0.3 * (0.5 * (trend_bias + 1)))
        long_share = float(np.clip(long_share, 0.0, 1.0))

        funds_long = intraday_cap * intraday_exposure * long_share
        funds_short = intraday_cap * intraday_exposure * (1.0 - long_share)

        base_price = float(df_loc["close"].iloc[t])

        # LONG side: buy dips
        for i, s in enumerate(sigma_levels):
            size_value = funds_long * float(level_fracs[i])
            size_cap = intraday_cap * max_pct_per_order
            if size_value > size_cap:
                size_value = size_cap

            sl_pct = sl_sigma * pred_vol if pred_vol > 0 else 1e-6
            max_loss_allowed = intraday_cap * risk_per_trade_pct
            max_size_from_risk = max_loss_allowed / (sl_pct + 1e-12)
            if size_value > max_size_from_risk:
                size_value = max_size_from_risk

            if size_value < 1.0:
                continue

            order_price = base_price * (1.0 - s * pred_vol)
            future_segment = df_loc["low"].iloc[t + 1 : t + 1 + horizon]
            future_vols = df_loc["volume"].iloc[t + 1 : t + 1 + horizon].values if "volume" in df_loc.columns else None

            # simulate fill: use liquidity-aware model if volumes present and order is meaningful
            use_liq = future_vols is not None and np.nansum(np.asarray(future_vols, dtype=float)) > 0.0
            filled, entry_price, fill_info = simulate_fill(order_price, "buy", future_segment=future_segment,
                                                            slippage_tolerance_pct=slippage_tolerance_pct,
                                                            tick=tick, lot=lot, method=fill_method,
                                                            use_liquidity_model=use_liq,
                                                            order_notional=size_value, future_volumes=future_vols)
            LOGGER.debug("t=%d LONG level=%s order_price=%.6f filled=%s entry=%s info=%s", t, s, order_price, filled, entry_price, repr(fill_info))
            if filled:
                if use_liq:
                    filled_fraction = float(fill_info)
                    exec_size = size_value * filled_fraction
                    # use executed price
                    exec_price = entry_price
                    # compute pnl over horizon using exec_price and future_after_fill window
                    try:
                        # approximate fill position as t (worst-case)
                        pos_fill = t
                    except Exception:
                        pos_fill = t
                else:
                    # non-liq path: fill_info is index label
                    try:
                        pos_fill = int(df_loc.index.get_loc(fill_info))
                    except Exception:
                        pos_fill = t
                    exec_price = entry_price
                    exec_size = size_value

                tp = exec_price * (1.0 + tp_sigma * pred_vol)
                sl = exec_price * (1.0 - sl_sigma * pred_vol)
                start_pos = pos_fill + 1
                end_pos = min(pos_fill + 1 + horizon, N)
                future_after_fill = df_loc.iloc[start_pos:end_pos]

                if future_after_fill.empty:
                    close_h = float(df_loc["close"].iloc[min(pos_fill + horizon, N - 1)])
                    pnl = (close_h - exec_price) / exec_price * exec_size
                else:
                    hit_tp = bool((future_after_fill["high"] >= tp).any())
                    hit_sl = bool((future_after_fill["low"] <= sl).any())
                    if hit_tp and not hit_sl:
                        pnl = (tp - exec_price) / exec_price * exec_size
                    elif hit_sl and not hit_tp:
                        pnl = (sl - exec_price) / exec_price * exec_size
                    elif hit_tp and hit_sl:
                        idx_tp = int(future_after_fill[future_after_fill["high"] >= tp].index[0])
                        idx_sl = int(future_after_fill[future_after_fill["low"] <= sl].index[0])
                        pnl = ((tp if idx_tp < idx_sl else sl) - exec_price) / exec_price * exec_size
                    else:
                        close_h = float(future_after_fill["close"].iloc[-1])
                        pnl = (close_h - exec_price) / exec_price * exec_size

                cash_intraday += pnl
                trades.append({"t": t, "side": "long", "entry": exec_price, "size": exec_size, "pnl": pnl, "fill_pos": pos_fill})

        # SHORT side: sell spikes (symmetrical)
        for i, s in enumerate(sigma_levels):
            size_value = funds_short * float(level_fracs[i])
            size_cap = intraday_cap * max_pct_per_order
            if size_value > size_cap:
                size_value = size_cap

            sl_pct = sl_sigma * pred_vol if pred_vol > 0 else 1e-6
            max_loss_allowed = intraday_cap * risk_per_trade_pct
            max_size_from_risk = max_loss_allowed / (sl_pct + 1e-12)
            if size_value > max_size_from_risk:
                size_value = max_size_from_risk

            if size_value < 1.0:
                continue

            order_price = base_price * (1.0 + s * pred_vol)
            future_segment = df_loc["high"].iloc[t + 1 : t + 1 + horizon]
            future_vols = df_loc["volume"].iloc[t + 1 : t + 1 + horizon].values if "volume" in df_loc.columns else None
            use_liq = future_vols is not None and np.nansum(np.asarray(future_vols, dtype=float)) > 0.0

            filled, entry_price, fill_info = simulate_fill(order_price, "sell", future_segment=future_segment,
                                                            slippage_tolerance_pct=slippage_tolerance_pct,
                                                            tick=tick, lot=lot, method=fill_method,
                                                            use_liquidity_model=use_liq,
                                                            order_notional=size_value, future_volumes=future_vols)
            LOGGER.debug("t=%d SHORT level=%s order_price=%.6f filled=%s entry=%s info=%s", t, s, order_price, filled, entry_price, repr(fill_info))
            if filled:
                if use_liq:
                    filled_fraction = float(fill_info)
                    exec_size = size_value * filled_fraction
                    exec_price = entry_price
                    pos_fill = t
                else:
                    try:
                        pos_fill = int(df_loc.index.get_loc(fill_info))
                    except Exception:
                        pos_fill = t
                    exec_price = entry_price
                    exec_size = size_value

                tp = exec_price * (1.0 - tp_sigma * pred_vol)
                sl = exec_price * (1.0 + sl_sigma * pred_vol)
                start_pos = pos_fill + 1
                end_pos = min(pos_fill + 1 + horizon, N)
                future_after_fill = df_loc.iloc[start_pos:end_pos]

                if future_after_fill.empty:
                    close_h = float(df_loc["close"].iloc[min(pos_fill + horizon, N - 1)])
                    pnl = (exec_price - close_h) / exec_price * exec_size
                else:
                    hit_tp = bool((future_after_fill["low"] <= tp).any())
                    hit_sl = bool((future_after_fill["high"] >= sl).any())
                    if hit_tp and not hit_sl:
                        pnl = (exec_price - tp) / exec_price * exec_size
                    elif hit_sl and not hit_tp:
                        pnl = (exec_price - sl) / exec_price * exec_size
                    elif hit_tp and hit_sl:
                        idx_tp = int(future_after_fill[future_after_fill["low"] <= tp].index[0])
                        idx_sl = int(future_after_fill[future_after_fill["high"] >= sl].index[0])
                        pnl = (exec_price - (tp if idx_tp < idx_sl else sl)) / exec_price * exec_size
                    else:
                        close_h = float(future_after_fill["close"].iloc[-1])
                        pnl = (exec_price - close_h) / exec_price * exec_size

                cash_intraday += pnl
                trades.append({"t": t, "side": "short", "entry": exec_price, "size": exec_size, "pnl": pnl, "fill_pos": pos_fill})

        # Volatility gating
        vol_now = float(df_loc["_raw_vol_30"].iloc[t]) if pd.notna(df_loc["_raw_vol_30"].iloc[t]) else 0.0
        if vol_now > 0 and pred_vol > 3.0 * vol_now:
            if (t - last_vol_spike_idx) > vol_spike_cooldown:
                prev_exposure = intraday_exposure
                intraday_exposure = max(0.1, intraday_exposure * 0.8)
                last_vol_spike_idx = t
                LOGGER.info(
                    "Vol spike at t=%d: vol_now=%.6f pred_vol=%.6f exposure %.3f->%.3f",
                    t,
                    vol_now,
                    pred_vol,
                    prev_exposure,
                    intraday_exposure,
                )

    total_pnl = cash_intraday - intraday_cap
    returns = [tr["pnl"] / (intraday_cap + 1e-12) for tr in trades]
    avg_trade = float(np.mean(returns)) if returns else 0.0
    winrate = float(np.mean([1 if tr["pnl"] > 0 else 0 for tr in trades])) if trades else 0.0
    summary = {
        "intraday_cap": intraday_cap,
        "final_intraday_cash": cash_intraday,
        "total_pnl": total_pnl,
        "n_trades": len(trades),
        "avg_trade_return": avg_trade,
        "winrate": winrate,
    }
    return summary, trades


# ==========================================
# DEAP evolution (with repair)
# ==========================================
@dataclass
class EvoBounds:
    hidden: Tuple[int, int] = (16, 256)
    log_lr: Tuple[float, float] = (-5.0, -2.0)
    seq_len: Tuple[int, int] = (8, 96)
    epochs: Tuple[int, int] = (5, 50)
    tp_sigma: Tuple[float, float] = (0.1, 1.0)
    sl_sigma: Tuple[float, float] = (0.5, 2.0)
    intraday_multiplier: Tuple[float, float] = (0.5, 1.5)


BOUNDS = EvoBounds()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def repair_individual(ind: List[float]) -> List[float]:
    hidden, log_lr, seq_len, epochs, tp_sigma, sl_sigma, intraday_mul = ind
    hidden = int(clamp(round(hidden), *BOUNDS.hidden))
    log_lr = float(clamp(log_lr, *BOUNDS.log_lr))
    seq_len = int(clamp(round(seq_len), *BOUNDS.seq_len))
    epochs = int(clamp(round(epochs), *BOUNDS.epochs))
    tp_sigma = float(clamp(tp_sigma, *BOUNDS.tp_sigma))
    sl_sigma = float(clamp(sl_sigma, *BOUNDS.sl_sigma))
    intraday_mul = float(clamp(intraday_mul, *BOUNDS.intraday_multiplier))
    return [hidden, log_lr, seq_len, epochs, tp_sigma, sl_sigma, intraday_mul]


if _HAS_DEAP:
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception:
        pass

    toolbox = base.Toolbox()
    toolbox.register("hidden", random.randint, *BOUNDS.hidden)
    toolbox.register("log_lr", lambda: random.uniform(*BOUNDS.log_lr))
    toolbox.register("seq_len", random.randint, *BOUNDS.seq_len)
    toolbox.register("epochs", random.randint, *BOUNDS.epochs)
    toolbox.register("tp_sigma", lambda: random.uniform(*BOUNDS.tp_sigma))
    toolbox.register("sl_sigma", lambda: random.uniform(*BOUNDS.sl_sigma))
    toolbox.register("intraday_multiplier", lambda: random.uniform(*BOUNDS.intraday_multiplier))

    def init_individual():
        return creator.Individual(
            [
                toolbox.hidden(),
                toolbox.log_lr(),
                toolbox.seq_len(),
                toolbox.epochs(),
                toolbox.tp_sigma(),
                toolbox.sl_sigma(),
                toolbox.intraday_multiplier(),
            ]
        )

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def mutate_custom(individual, mu=0.0, sigma=1.0, indpb=0.2):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = individual[i] + random.gauss(mu, sigma)
        repaired = repair_individual(individual)
        individual[:] = repaired
        return (individual,)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_custom)
    toolbox.register("select", tools.selTournament, tournsize=3)


def evaluate_individual(
    ind: Sequence[float], df: pd.DataFrame, feature_cols: Sequence[str], device: torch.device, horizon: int = 5
) -> Tuple[float]:
    ind = repair_individual(list(ind))
    hidden, log_lr, seq_len, epochs, tp_sigma, sl_sigma, intraday_mul = ind
    lr = 10 ** float(log_lr)

    X, y_ret, y_vol = create_sequences_from_df(df, feature_cols, seq_len=int(seq_len), horizon=horizon)
    n = len(X)
    split = int(0.7 * n)
    if split <= 0 or n - split <= 10:
        return (-1e9,)  # invalid split

    Xtr, Xv = X[:split], X[split:]
    ytr, yv = y_ret[:split], y_ret[split:]
    voltr, volv = y_vol[:split], y_vol[split:]

    batch = 128
    tr_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(voltr))
    val_ds = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv), torch.from_numpy(volv))
    tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

    model = LSTM2Head(input_size=X.shape[2], hidden_size=int(hidden), num_layers=2).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for _ in range(int(epochs)):
        train_epoch(model, tr_loader, opt, device, alpha_vol_loss=0.5)

    val_start_idx = int(seq_len) + split
    val_end_idx = val_start_idx + len(Xv) + horizon
    if val_end_idx > len(df):
        val_end_idx = len(df)
    df_val = df.iloc[val_start_idx:val_end_idx].reset_index(drop=True)

    try:
        summary, trades = backtest_market_maker(
            df_val,
            model,
            feature_cols,
            seq_len=int(seq_len),
            horizon=horizon,
            capital=100_000.0,
            intraday_frac=0.10 * float(intraday_mul),
            sigma_levels=(0.5, 1.0, 1.5, 2.0),
            tp_sigma=float(tp_sigma),
            sl_sigma=float(sl_sigma),
            trend_weighting=True,
            swing_window=50,
        )
        pnl = float(summary.get("total_pnl", -1e9))
        n_trades = int(summary.get("n_trades", 0))
        score = pnl - 0.01 * max(0, 50 - n_trades)
    except Exception as e:
        LOGGER.debug("Backtest failed in evaluation: %s", e)
        score = -1e9
    return (score,)


def run_evolution(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    pop_size: int = 20,
    gens: int = 15,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
    horizon: int = 5,
):
    assert _HAS_DEAP, "DEAP no instalado (pip install deap)"
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    for g in range(gens):
        for ind in pop:
            if not ind.fitness.valid:
                ind[:] = repair_individual(ind)
                ind.fitness.values = evaluate_individual(ind, df, feature_cols, device, horizon)
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        for ind in offspring:
            if not ind.fitness.valid:
                ind[:] = repair_individual(ind)
                ind.fitness.values = evaluate_individual(ind, df, feature_cols, device, horizon)
        pop = toolbox.select(pop + offspring, k=len(pop))
        hof.update(pop)
        record = stats.compile(pop)
        LOGGER.info("Gen %d: avg=%.3f max=%.3f", g, record["avg"], record["max"])
    return pop, hof


# ==========================================
# CLI / main
# ==========================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--use_influx", action="store_true", help="Load data from InfluxDB instead of OU simulator")
    p.add_argument("--do_evolve", action="store_true", help="Run DEAP evolution")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_model", type=str, default="final_model.pt")
    p.add_argument("--save_meta", type=str, default="final_meta.json")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--export_trades", type=str, default="", help="CSV path to export trades from quick backtest")
    p.add_argument("--verbose", "-v", action="count", default=1, help="-v INFO, -vv DEBUG, none WARNING")
    p.add_argument("--quiet", action="store_true", help="Silence logs (overrides -v)")
    p.add_argument("--config", type=str, default="", help="Optional JSON config to override CLI")
    return p


def main_cli() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    verbosity = 0 if args.quiet else args.verbose
    setup_logging(verbosity)

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch is not None else None
    LOGGER.info("Device: %s", device)

    # Load data
    try:
        if args.use_influx:
            close = load_data_from_influx(limit=20000)
            noise = 0.001 * np.abs(np.random.randn(len(close)))
            high = close * (1.0 + noise)
            low = close * (1.0 - noise)
        else:
            raise RuntimeError("fallback to OU")
    except Exception as e:
        LOGGER.warning("Falling back to OU simulator: %s", e)
        close, high, low = simulate_ou(steps=5000, seed=args.seed)

    df = add_technical_features(close, high=high, low=low, volume=None)

    # Feature selection
    feature_cols = [c for c in df.columns if c not in ["close", "high", "low", "volume"]]
    feature_cols = [
        c
        for c in feature_cols
        if c.startswith("ret_")
        or c.startswith("sma")
        or c.startswith("ema")
        or ("rsi" in c)
        or ("atr" in c)
        or ("raw_vol" in c)
        or ("bb" in c)
        or ("fib" in c)
        or ("td" in c)
        or ("vp" in c)
    ]
    if not feature_cols:
        raise RuntimeError("No feature columns selected")

    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    LOGGER.info("Dropped %d rows due to NaNs", rows_before - len(df))

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    try:
        if hasattr(scaler, "feature_names_in_"):
            scaler.feature_names_in_ = np.array(list(feature_cols), dtype=object)
    except Exception:
        pass

    seq_len = int(args.seq_len)
    epochs = int(args.epochs)
    hidden = int(args.hidden)
    horizon = int(args.horizon)

    X, y_ret, y_vol = create_sequences_from_df(df, feature_cols, seq_len=seq_len, horizon=horizon)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_ret), torch.from_numpy(y_vol))
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    model = LSTM2Head(input_size=X.shape[2], hidden_size=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=float(args.lr))

    LOGGER.info("Training model for %d epochs", epochs)
    for e in trange(epochs, desc="Final training"):
        loss = train_epoch(model, loader, opt, device, alpha_vol_loss=0.5)
        if e % 5 == 0:
            LOGGER.info("Epoch %d/%d loss=%.6f", e + 1, epochs, loss)

    meta = {"feature_cols": feature_cols, "seq_len": seq_len, "horizon": horizon, "hidden": hidden, "lr": args.lr, "input_size": X.shape[2]}
    save_model(model, Path(args.save_model), meta, Path(args.save_meta))

    try:
        if joblib is not None:
            joblib.dump(scaler, "scaler.pkl")
            LOGGER.info("Saved scaler to scaler.pkl")
    except Exception as e:
        LOGGER.warning("Could not save scaler via joblib: %s", e)

    df_test = df.iloc[-2000:].reset_index(drop=True)
    summary, trades = backtest_market_maker(
        df_test,
        model,
        feature_cols,
        seq_len=seq_len,
        horizon=horizon,
        capital=1_000_000,
        intraday_frac=0.10,
        sigma_levels=(0.5, 1.0, 1.5, 2.0),
        tp_sigma=0.5,
        sl_sigma=1.0,
        trend_weighting=True,
        scaler=scaler,
    )
    LOGGER.info("Backtest summary: %s", summary)
    print("Backtest summary:", json.dumps(summary, indent=2))

    if args.export_trades:
        out = Path(args.export_trades)
        pd.DataFrame(trades).to_csv(out, index=False)
        LOGGER.info("Exported %d trades to %s", len(trades), out)

    if args.plot:
        prices = df_test["close"].values
        plt.figure(figsize=(12, 5))
        plt.plot(prices, label="price")
        plt.title("Price (test segment)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main_cli()


# Exports
__all__ = [
    "seed_everything",
    "simulate_ou",
    "add_technical_features",
    "create_multi_tf_merged_df_from_base",
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
    "ensure_scaler_feature_names",
    "build_dataset_for_training",
    "prepare_input_for_model",
    "prepare_model_input",
    "predict_with_model",
    "backtest_market_maker",
    "build_ohlc_df_from_close",
    "load_data_from_influx",
]

#--------#
# NOTAS: #
#--------#
"""

Primero analiza el conjunto, sin los fiboevo.py anteriores al que acabas de actualizar, y revisa:
1: Qué cambios en el resto de la logica requiere el nuevo fiboevo.py
2: Qué conflictos puede haber y cómo resolverlos.

Analiza esta informacion y elabora, razonadamente, un plan o lista de cambios que habría que aplicar al resto de scripts involucrados (principalmente trading_daemon, trading_gui_extended y prepare_dataset)
Si hay conflictos, analiza los pros y los contras de cada uno, evalua cual debería ser el merge más indicado para corregir problemas, expandir funcionalidad y no romper otras partes que dependan de estas salvo que sea completamente necesario (en cuyo caso indicarás qué modificaciones requiere en los callers)
Haz todos los preparativos necesarios para una transición fina a la actualización de trading_gui_extended.py (y tomando las notas que sean necesarias para el TradingDaemon u otros)
ChatGPT Plus
Pensó durante 13s

Perfecto — voy a analizar razonada y planificadamente el conjunto sin los fiboevo.py anteriores y centrarme en las implicaciones de sustituirlos por la versión modificada que te entregué. Te doy (1) un diagnóstico de impacto, (2) conflictos potenciales y cómo resolverlos, y (3) un plan de cambios concreto y priorizado con snippets de parche para trading_daemon, trading_gui_extended, y prepare_dataset + pasos de verificación.

Lo entrego en español, con secciones claras para que puedas aplicarlo paso a paso.

Resumen ejecutivo (1 línea)

La nueva fiboevo.py introduce: prepare_input_for_model, ensure_scaler_feature_names, load_model → (model, meta), load_scaler(..., meta), y una simulate_fill unificada; todo ello mejora robustez (alineamiento columnas, evitar doble-scaling, modo de liquidez). Esto requiere cambios localizados en los callers que cargan modelos/scalers y en los puntos de inferencia/backtest para usar prepare_input_for_model y validar meta.

1) Cambios en el resto de la lógica que requiere la nueva fiboevo.py

A continuación enumero las áreas del código que deben adaptarse y por qué:

A. Carga de modelos

Qué cambió: load_model ahora devuelve (model, meta) (antes devolvía sólo model o model + sidecar implícito).
Impacto: Todos los callers que hacen model = load_model(path) fallarán o ignorarán meta.
Acción: cambiar por model, model_meta = load_model(path) y usar model_meta para validar feature_cols, seq_len, horizon, input_size.

B. Carga de scaler

Qué cambió: load_scaler(path, meta) inyecta feature_names_in_ cuando falta; existe además ensure_scaler_feature_names.
Impacto: Callers que hacían scaler = joblib.load("scaler.pkl") deberían usar ahora scaler = load_scaler("scaler.pkl", meta) o ensure_scaler_feature_names(scaler, meta) para prevenir desalineamiento de columnas.
Acción: actualizar cargas de scaler en CLI, GUI y daemon.

C. Inferencia — preparación de la entrada

Qué cambió: introducción de prepare_input_for_model(df, feature_cols, seq_len, scaler, method) y recomendaciones de usarlo en vez de construir arrays manualmente (.values.astype(np.float32)).
Impacto: backtest_market_maker, trading_daemon.iteration_once() (o similar) y GUI predictores deben sustituir la construcción manual de seq_x por prepare_input_for_model(...).to(device) para evitar:

doble scaling (si df ya estaba escalado)

reordenamientos de columnas que provoquen features mal alineadas
Acción: pasar scaler a backtest_market_maker, y usar prepare_input_for_model siempre.

D. Backtester simulate_fill

Qué cambió: simulate_fill ahora soporta modo simple (compatible) y modo liquidity-aware que puede devolver filled_fraction en tercer valor.
Impacto: Callers que interpreten siempre el tercer retorno como índice label deben comprobar comportamiento. En el backtester que te entregué he adaptado la lógica pero otros callers externos deben revisar/ajustar.
Acción: revisar usos de simulate_fill y adaptar (o forzar use_liquidity_model=False si se espera label).

E. Metadata y validaciones

Qué hay ahora: meta puede contener feature_cols, input_size, seq_len, horizon. prepare_input_for_model valida scaler.feature_names_in_ vs feature_cols.
Impacto: scripts que suponen auto-detección libre de features ahora deben preferir meta['feature_cols'].
Acción: añadir rutina de validación en daemon/gui que:

si meta.feature_cols existe, usarlo y avisar si columnas faltan;

si no existe, auto-detectar pero loggear advertencia.

F. Guardar artefactos en prepare_dataset

Qué hay que hacer: garantizar que prepare_dataset/training guarden meta completo y que scaler se guarde con feature_names_in_.
Impacto: si no se guardan, load_scaler(..., meta) no podrá inyectar nombres y puede producir desalineamientos.
Acción: añadir scaler.feature_names_in_ = np.array(feature_cols) al guardar y guardar meta.json con feature_cols, seq_len, horizon, input_size.

G. Indices / timestamps / IDs

Qué hay: create_sequences_from_df espera df.dropna().reset_index(drop=True) — por tanto utiliza posiciones. simulate_fill usa index label para obtener posición mediante df.index.get_loc(label).
Impacto: si el código del daemon o del GUI mezcla DataFrames con índices temporales y luego resetea índices de manera inconsistente con el uso en simulate_fill, los fill_label y get_loc pueden devolver errores.
Acción: definir una convención clara:

para backtest: usar DataFrame con índice temporal (DatetimeIndex) y no resetear indices hasta después de simular fills; cuando sea necesario, usar df.reset_index() localmente en create_sequences_from_df (ya hace reset_index internamente).

trading_daemon: al recibir datos por websocket, conservar timestamps como índice y sólo pasar subsecciones con reset_index(drop=True) a prepare_input_for_model.

2) Conflictos posibles y recomendaciones de resolución

A continuación describo conflictos probables, pros/cons y la recomendación para el merge.

Conflicto 1 — load_model firma antigua vs nueva

Problema: muchos sitios hacen model = fiboevo.load_model(path). Nueva firma (model, meta) romperá esos callers.

Opciones:

Cambiar todos los callers para usar la nueva firma (recomendado).

Pro: usa meta de modelo, mejor seguridad/diagnóstico.

Contra: trabajo de refactor.

Crear wrapper retrocompatible: def load_model_compat(path): m = load_model(path); if isinstance(m, tuple): return m[0]; return m

Pro: mínimo cambio visible.

Contra: oculta meta y perpetúa malas prácticas.

Recomendación: aplicar cambio en callers (opción 1) + añadir en fiboevo.py una función load_model_legacy(path) que devuelva sólo model para compatibilidad temporal. Documentar y planificar remover legacy en siguiente release.

Conflicto 2 — escala doble / mezcla de datos escalados y no escalados

Problema: algunos módulos quizá pre-escalaban df antes de llamar a backtester o al daemon. Si además se pasa scaler a prepare_input_for_model, se corre el riesgo de doble-scaling.

Opciones:

Estandarizar: definir que nunca se escale a nivel global; el escalado se hace justo antes de crear secuencias (training) y prepare_input_for_model siempre aplica scaler si se le pasa.

Pro: un único lugar de escalado, evita duplicados.

Contra: hay que refactorizar callers que actualmente pasan df escalado.

Detección automática: introducir meta flag scaler_applied o un campo meta['scaled_by'] = "scaler.pkl"; prepare_input_for_model puede comprobar consistencia mediante la media/var en meta y decidir no volver a escalar.

Pro: más robusto frente a diferentes pipelines.

Contra: requiere guardar más metadata y aplica heurísticas.

Recomendación: combinar 1 y 2: normalizar el pipeline (preferible) y además añadir defensas heurísticas en prepare_input_for_model (p.ej. comprobar si los valores están ya estandarizados usando percentiles y comparar con scaler.mean_ si está disponible). Para transición, aceptar ambos y emitir warning si detecta posible doble-scaling.

Conflicto 3 — retorno de simulate_fill variable (label vs fraction)

Problema: algunos callers esperan el tercer elemento sea fill_label; la nueva función puede devolver filled_fraction.

Opciones:

Mantener retrocompatibilidad por defecto use_liquidity_model=False (ya hecho).

Forzar contrato consistente: devolver siempre (filled, executed_price, info_dict) donde info_dict tiene keys label o filled_fraction.

Recomendación: cambiar el contrato hacia (filled, executed_price, info) donde info es dict (más explícito). Pero esto rompe compatibilidad. Mientras tanto, no romper: mantener comportamiento actual (simple: label) y documentar que callers deben revisar type(fill_info) (datetime/int vs float). En una segunda versión, migrar a dict.

Conflicto 4 — orden de columnas / feature_cols ausentes

Problema: modelo meta indica feature_cols que pueden faltar en runtime (por ejemplo si websocket envía menos features).

Opciones:

Abort inference si faltan columnas (estricto).

Fallback: rellenar columnas faltantes con NaN/0 y loggear (más flexible).

Recomendación: por defecto abort con advertencia (evita inferencias con inputs inválidos). Ofrecer una opción allow_missing_features=True que rellene con np.nan/impute y proseguir (para escenarios exploratorios).

Conflicto 5 — indices y posiciones vs timestamps

Problema: tras reset_index y operaciones, simulate_fill puede devolver indices que get_loc no encuentre.

Recomendación: adoptar convención y documentarla: mantener DatetimeIndex para backtests y usar reset_index(drop=True) sólo en funciones que lo requieran; simulate_fill ya documenta que devuelve label compatible con df.index.get_loc. Añadir defensiva: si get_loc falla, usar fallback a nearest integer index.

3) Plan de cambios (priorizado, con snippets y verificación)

Voy a listar los cambios concretos a aplicar en trading_daemon.py, trading_gui_extended.py y prepare_dataset.py — en orden de prioridad. Para cada cambio incluyo un snippet propuesto.

Prioridad Alta — mantener compatibilidad e integridad (urgente)
1. Actualizar cargas de modelo y scaler (todo caller)

Objetivo: usar model, meta = load_model(path) y scaler = load_scaler(path, meta).

Snippet a aplicar (ejemplo en trading_daemon.py)

Reemplazar:

model = fiboevo.load_model(model_path)
scaler = joblib.load("scaler.pkl")


por:

model, model_meta = fiboevo.load_model(model_path)
# Prefer fiboevo.load_scaler which inyecta feature_names_in_ desde meta
try:
    scaler = fiboevo.load_scaler("scaler.pkl", meta=model_meta)
except Exception:
    # fallback: try joblib directly and ensure feature names
    scaler = joblib.load("scaler.pkl")
    scaler = fiboevo.ensure_scaler_feature_names(scaler, model_meta)


Por qué: garantiza que scaler.feature_names_in_ está presente y que disponemos de model_meta para validaciones posteriores.

Verificación:

Reiniciar daemon, comprueba logs: meta cargado y scaler.feature_names_in_ inyectado o existente.

Prueba: llamar prepare_input_for_model con últimas filas y comprobar que no lanza error de mismatch.

2. Sustituir inferencia manual por prepare_input_for_model

Objetivo: eliminar construcción manual de arrays (evita doble-scaling y desalineamiento).

En backtest_market_maker (ya modificado en la versión entregada) — si tienes una versión antigua en tu repo, sustituir:

seq_x = df_loc[feature_cols].iloc[t - seq_len : t].values.astype(np.float32)
x_t = torch.from_numpy(seq_x).unsqueeze(0).to(device)
pred_ret, pred_vol = model(x_t)


por:

seq_df = df_loc.iloc[t - seq_len : t].reset_index(drop=True)
x_t = fiboevo.prepare_input_for_model(seq_df, feature_cols, seq_len, scaler=scaler, method="per_row").to(device)
pred_ret, pred_vol = model(x_t)


En trading_daemon.iteration_once() (o similar):

Reemplazar el bloque que construye X manualmente por:

# antes de predecir, verifica meta
if hasattr(self, "model_meta") and self.model_meta:
    feature_cols = self.model_meta.get("feature_cols", feature_cols_auto)
# prepare df segment (last seq_len rows)
seq_df = feats_df.iloc[-seq_len:].reset_index(drop=True)
x_t = fiboevo.prepare_input_for_model(seq_df, feature_cols, seq_len, scaler=self.model_scaler).to(self.model_device)
pred_ret, pred_vol = self.model(x_t)


Verificación:

Ejecutar un ciclo de predicción con datos reales; comparar outputs con runs anteriores (espera que la diferencia provenga sólo de corrección de alineamiento y no por bugs).

3. Validación post-load del modelo (en daemon/gui)

Objetivo: evitar inferencias con features faltantes o input_size inconsistent.

Snippet ejemplo:

def validate_model_loaded(model_meta, scaler, current_feats):
    if model_meta and isinstance(model_meta.get("feature_cols"), (list,tuple)):
        feat_meta = model_meta["feature_cols"]
        missing = [c for c in feat_meta if c not in current_feats.columns]
        if missing:
            logger.warning("Missing features: %s", missing)
            # decide: raise or continue with fallback
            raise RuntimeError("Cannot infer: required features missing.")
    # check input_size
    declared = int(model_meta.get("input_size", len(feat_meta)))
    if declared != len(feat_meta):
        logger.warning("meta.input_size (%s) != len(feature_cols) (%s)", declared, len(feat_meta))


Dónde: llamar justo después de load_model.

Verificación: cargar modelos con meta incompleta y comprobar que la función lanza advertencias/errores controlados.

Prioridad Media — coherencia de datos y preservación de indices
4. Normalizar convención de índices (timestamps)

Objetivo: definir y aplicar una convención para índices en backtest y daemon (DatetimeIndex preferido).

Regla sugerida:

Los DataFrame de OHLCV que provienen de DB (Influx/SQLite) deben conservar DatetimeIndex.

create_sequences_from_df seguirá reset_index(drop=True) internamente (ya lo hace), así que no requiere cambio.

simulate_fill espera etiquetas compatibles con df.index.get_loc(label) — no resetear índices antes de llamar a simulate_fill.

Cambios concretos: revisar puntos donde se hace df.reset_index() antes de pasar segmentos a simulate_fill y evitar hacerlo.

Verificación: backtest que utilice timestamps; verificar que fill_label devuelto por simulate_fill se transforma correctamente a position con df.index.get_loc(fill_label).

5. Preparar transición del daemon: pasar scaler y model_meta a estado del daemon

Objetivo: almacenar self.model, self.model_meta, self.model_scaler, self.model_device en TradingDaemon para uso en cada iteración.
Por qué: evita recargas repetidas y facilita validación.

Snippet (inicialización tras cargar modelos):

self.model, self.model_meta = fiboevo.load_model(model_path)
self.model_scaler = fiboevo.load_scaler(scaler_path, meta=self.model_meta)  # or ensure_scaler_feature_names
self.model_device = torch.device("cuda"

"""
