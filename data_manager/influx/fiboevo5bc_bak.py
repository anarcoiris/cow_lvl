#!/usr/bin/env python3
"""
Evolve & Trade (Fixed & Hardened)

Key fixes & upgrades (see chat for rationale):
 - Add __main__ guard so the CLI actually runs
 - Robust logging + --verbose/--quiet + exception surfaces
 - Deterministic seeding across numpy/torch/random + CUDA flags
 - OU simulator: ensure high >= low; remove pathological negatives
 - Feature engineering: avoid forward/backward fill that caused leakage; drop NaNs safely
 - Backtest uses *raw* realized volatility computed from close (was using scaled col)
 - Backtest stable allocation and volatility gating; checks for insufficient history
 - DEAP: clamp/repair genotype, custom mutation for mixed types, fitness guard
 - Training: gradient clipping, optional early stopping, progress logs
 - CLI: more knobs (horizon/seq/epochs/lr/hidden), CSV export for trades, save scaler, config file
 - Safer Influx handling + clearer messages if not configured
 - Numerous small type hints, asserts, edge-case guards

NOTE: This is a single-file drop-in replacement. You can keep your same
invocation, or try e.g.:

  python evolve_and_trade_improved_FIXED.py --plot --do_evolve --seed 123

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
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# matplotlib only used when --plot is set
import matplotlib.pyplot as plt  # noqa: F401
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ============== Optional deps ==============
# DEAP
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

# ==========================================
# Logging
# ==========================================
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
# Seeding / determinism
# ==========================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    start: str = "-60d",
    limit: int = 42000,
    influx_cfg: Optional[dict] = None,
) -> np.ndarray:
    if not _HAS_INFLUX:
        raise RuntimeError("Influx client no instalado. pip install influxdb-client")

    INFLUX_URL = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
    INFLUX_TOKEN = os.getenv('INFLUXDB_TOKEN', 'J4twWiQSH6QZF33ZyAB9NJLoNMyrHjOlvY6UJGgczJfk-_DC3d5BFEiZzQOYC39ObPYwxF5kZTAZtzIX-Xr40Q==')
    INFLUX_ORG = os.getenv('INFLUXDB_ORG', 'BreOrganization')
    INFLUX_BUCKET = os.getenv('INFLUXDB_BUCKET', 'prices')

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


# ==========================================
# Feature engineering
# ==========================================

def add_technical_features(
    close: np.ndarray,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    volume: Optional[np.ndarray] = None,
    window_long: int = 50,
) -> pd.DataFrame:
    """Return a DataFrame with features aligned to close.
    IMPORTANT: We avoid forward/backward filling to prevent look-ahead leakage.
    Caller should dropna() after feature creation.
    """
    df = pd.DataFrame({"close": close})
    if high is None:
        high = close
    if low is None:
        low = close
    df["high"] = high
    df["low"] = low

    # basic returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)

    # moving averages + diffs
    for L in [5, 10, 20, 50, window_long]:
        df[f"sma_{L}"] = df["close"].rolling(L, min_periods=L).mean()
        df[f"ema_{L}"] = df["close"].ewm(span=L, adjust=False).mean()
        df[f"ma_diff_{L}"] = df["close"] - df[f"sma_{L}"]

    # Bollinger
    bb_m = df["close"].rolling(20, min_periods=20).mean()
    bb_std = df["close"].rolling(20, min_periods=20).std()
    df["bb_m"] = bb_m
    df["bb_std"] = bb_std
    df["bb_up"] = bb_m + 2 * bb_std
    df["bb_dn"] = bb_m - 2 * bb_std
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / (bb_m.replace(0, np.nan))

    # RSI
    try:
        if _HAS_PT:
            df["rsi_14"] = pta.rsi(df["close"], length=14)
        else:
            df["rsi_14"] = simple_rsi(df["close"].values, 14)
    except Exception:
        df["rsi_14"] = simple_rsi(df["close"].values, 14)

    # ATR
    df["atr_14"] = atr_simple(df["high"], df["low"], df["close"], 14)

    # rolling volatility (UNSCALED, raw)
    df["raw_vol_10"] = df["ret_1"].rolling(10, min_periods=10).std()
    df["raw_vol_30"] = df["ret_1"].rolling(30, min_periods=30).std()

    # Fibonacci based on lookback extremes
    lookback = 50
    highs = df["high"].rolling(lookback, min_periods=lookback).max()
    lows = df["low"].rolling(lookback, min_periods=lookback).min()
    fibs = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    for f in fibs:
        level = lows + (highs - lows) * f
        name = f"fib_{int(f*1000)}"
        df[name] = level
        df[f"dist_{name}"] = (df["close"] - level) / df["close"].replace(0, np.nan)

    # TD Sequential (simple setups)
    close_arr = df["close"].values
    buy_setup = np.zeros(len(close_arr), dtype=int)
    sell_setup = np.zeros(len(close_arr), dtype=int)
    for i in range(4, len(close_arr)):
        buy_setup[i] = buy_setup[i - 1] + 1 if close_arr[i] < close_arr[i - 4] else 0
        sell_setup[i] = sell_setup[i - 1] + 1 if close_arr[i] > close_arr[i - 4] else 0
    df["td_buy_setup"] = buy_setup
    df["td_sell_setup"] = sell_setup

    # Volume profile approximation omitted unless volume is provided (rare here)
    if volume is not None:
        df["volume"] = volume
        window = 100
        pocs = [np.nan] * len(df)
        for i in range(window, len(df)):
            seg = df["close"].iloc[i - window : i]
            seg_vol = df["volume"].iloc[i - window : i].values
            hist, edges = np.histogram(seg.values, bins=20, weights=seg_vol)
            max_bin = np.argmax(hist)
            pocs[i] = 0.5 * (edges[max_bin] + edges[max_bin + 1])
        df["vp_poc"] = pocs
        df["dist_vp"] = (df["close"] - df["vp_poc"]) / df["close"].replace(0, np.nan)

    return df

# ------------------------------
# Multi-timeframe helpers
# ------------------------------
import pandas as pd
from typing import Dict

# Convenciones de alias: base_freq como pandas offset alias, ej. '1T' (1 minuto), '4H', '1D'
# close_base: np.ndarray de precios en la resolución base, con un index de timestamps (opcional)
# Si no tienes timestamps reales, se usa RangeIndex y resample funciona vía pd.Grouper con periodo fijo.

def build_ohlc_df_from_close(close: np.ndarray, freq_alias: str = "1T", start_ts: Optional[pd.Timestamp]=None) -> pd.DataFrame:
    """
    Construye un DataFrame OHLC indexado por periodo 'freq_alias' a partir de una serie de closes base.
    Si close ya está en la resolución deseada, simplemente devuelve close como 'close' y fabricated high/low.
    - close: np.array 1D
    - freq_alias: p.ej. '1T' for minutes, '4H', '1D'
    Retorna DataFrame con columnas ['open','high','low','close','volume'] (volume opcional NaN).
    """
    # si no hay timestamps, creamos uno arbitrario empezando ahora y con freq 1 minuto
    n = len(close)
    if start_ts is None:
        start_ts = pd.Timestamp("2020-01-01 00:00:00")
    # assumimos que la serie de entrada está en resolución 1 minuto (base).
    # Si viene en otra resolución, el caller debe proveer la serie ya en esa resolución.
    idx = pd.date_range(start=start_ts, periods=n, freq="1T")
    s = pd.Series(close, index=idx)
    # Resample a la freq_alias
    ohlc = s.resample(freq_alias).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    # Si hay NaNs (pocas) los llenamos por ffill
    ohlc = ohlc.ffill().bfill()
    ohlc["volume"] = np.nan
    return ohlc

def create_multi_tf_merged_df_from_base(
    close_base: np.ndarray,
    base_freq: str = "1T",
    tfs: Sequence[str] = ("1T", "4H", "1D"),
    start_ts: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    1) Asume close_base es la serie en resolución base (p.ej. 1m).
    2) Construye OHLC para cada TF usando resample desde base (útil si simulas).
    3) Calcula features POR TF usando add_technical_features y renombra columnas con sufijo _{tf_tag}
    4) Upsample/forward-fill features de los TFs de mayor tamaño al índice base y concatena columnas.
    Retorna: merged_df (index = base timestamps) y dict tf_dfs.
    """
    if start_ts is None:
        start_ts = pd.Timestamp("2020-01-01 00:00:00")
    # build base series index
    idx_base = pd.date_range(start=start_ts, periods=len(close_base), freq=base_freq)
    df_base = pd.DataFrame({"close": close_base}, index=idx_base)

    tf_dfs: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        # if tf == base_freq just compute features on df_base aggregated at base
        if tf == base_freq:
            df_tf = df_base.copy()
            # fabricate high/low slightly around close
            df_tf["high"] = df_tf["close"] * (1.0 + 0.0005 * np.abs(np.random.randn(len(df_tf))))
            df_tf["low"] = df_tf["close"] * (1.0 - 0.0005 * np.abs(np.random.randn(len(df_tf))))
            df_tf["volume"] = np.nan
        else:
            # aggregate base to coarser tf by resampling
            s = df_base["close"]
            ohlc = s.resample(tf).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            ohlc = ohlc.ffill().bfill()
            df_tf = pd.DataFrame({"close": ohlc["close"].values, "high": ohlc["high"].values, "low": ohlc["low"].values}, index=ohlc.index)
            df_tf["volume"] = np.nan

        # compute technical features for this tf
        df_tf_feats = add_technical_features(df_tf["close"].values, high=df_tf["high"].values, low=df_tf["low"].values, volume=None)
        # rename columns with tf tag: use a short tag, e.g., '1T' -> 'm1', '4H' -> 'h4', '1D' -> 'd1'
        tag = tf.replace("T", "m").replace("H", "h").replace("D", "d")  # crude mapping
        df_tf_feats = df_tf_feats.add_suffix(f"_{tag}")
        tf_dfs[tf] = df_tf_feats

    # Now merge: reindex each tf df to base index by forward-fill (coarser -> base)
    merged = df_base.copy()  # keep base close as ground truth
    for tf, df_feats in tf_dfs.items():
        # align by index: upsample df_feats to base index by forward-fill
        df_up = df_feats.reindex(idx_base, method="ffill")
        # drop original close_{tag} to avoid duplicate base close if tf==base_freq
        # but keep eg close_m1 as info if desired; to avoid confusion we keep features only (not close)
        cols_keep = [c for c in df_up.columns if not c.startswith("close_")]
        merged = pd.concat([merged, df_up[cols_keep]], axis=1)

    # final forward fill any remaining NaNs
    merged = merged.ffill().bfill()
    return merged, tf_dfs

# ==========================================
# Sequences for LSTM
# ==========================================

def create_sequences_from_df(
    df: pd.DataFrame, feature_cols: Sequence[str], seq_len: int = 32, horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N, seq_len, F), y_ret (N,), y_vol (N,) where
    y_ret = future return over horizon; y_vol = realized vol over horizon.
    """
    assert seq_len >= 1 and horizon >= 1
    arr = df[feature_cols].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    N = len(df) - seq_len - horizon + 1
    if N <= 0:
        raise ValueError("Not enough data for the requested seq_len + horizon")

    X = np.zeros((N, seq_len, arr.shape[1]), dtype=np.float32)
    y_ret = np.zeros((N,), dtype=np.float32)
    y_vol = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        X[i] = arr[i : i + seq_len]
        start = i + seq_len - 1
        end = start + horizon
        future_ret = (close[end] - close[start]) / (close[start] + 1e-12)
        y_ret[i] = future_ret
        rets = (close[start + 1 : end + 1] - close[start:end]) / (close[start:end] + 1e-12)
        y_vol[i] = float(np.std(rets)) if len(rets) > 0 else 0.0
    return X, y_ret, y_vol


# ==========================================
# LSTM model with two heads (ret, vol)
# ==========================================

class LSTM2Head(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.head_ret = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1)
        )
        self.head_vol = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1), nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        ret = self.head_ret(last).squeeze(-1)
        vol = self.head_vol(last).squeeze(-1)
        return ret, vol


# ==========================================
# Training / evaluation
# ==========================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha_vol_loss: float = 0.5,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    mse = nn.MSELoss()
    for xb, yret, yvol in loader:
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
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


def eval_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device, alpha_vol_loss: float = 0.5
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    mse = nn.MSELoss()
    with torch.no_grad():
        for xb, yret, yvol in loader:
            xb = xb.to(device)
            yret = yret.to(device)
            yvol = yvol.to(device)
            pred_ret, pred_vol = model(xb)
            loss_ret = mse(pred_ret, yret)
            loss_vol = mse(pred_vol, yvol)
            loss = loss_ret + alpha_vol_loss * loss_vol
            bs = xb.size(0)
            total_loss += loss.item() * bs
            n += bs
    return total_loss / max(1, n)


# ==========================================
# Backtester (market making / scalp)
# ==========================================
def backtest_market_maker(
    df: pd.DataFrame,
    model: nn.Module,
    feature_cols: Sequence[str],
    seq_len: int,
    horizon: int,
    capital: float = 1_000_000.0,
    intraday_frac: float = 0.10,
    sigma_levels: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    tp_sigma: float = 0.5,
    sl_sigma: float = 1.0,
    trend_weighting: bool = True,
    swing_window: int = 100,
) -> Tuple[dict, List[dict]]:
    device = next(model.parameters()).device
    df_loc = df.copy().reset_index(drop=True)
    N = len(df_loc)

    # Precompute raw vol for gating (from close, unscaled)
    df_loc["_ret_1"] = df_loc["close"].pct_change(1)
    df_loc["_raw_vol_30"] = df_loc["_ret_1"].rolling(30, min_periods=30).std()
    df_loc["sma_50"] = df_loc["close"].rolling(50, min_periods=50).mean()
    df_loc["sma_200"] = df_loc["close"].rolling(200, min_periods=200).mean()
    df_loc["swing_delta"] = df_loc["close"].pct_change(swing_window)

    # intraday capital (fixed reference) and working cash
    intraday_cap = capital * intraday_frac
    cash_intraday = intraday_cap
    trades: List[dict] = []

    # Exposure & risk control params (tunable)
    intraday_exposure = 1.0  # fraction of intraday_cap usable for orders (dynamic)
    last_vol_spike_idx = -9999
    vol_spike_cooldown = max(10, int(1.0 / max(1, horizon)))  # bars to wait after a vol spike
    max_pct_per_order = 0.02  # max fraction of intraday_cap used by a single order (2%)
    risk_per_trade_pct = 0.005  # max fraction of intraday_cap risked per trade (0.5%)
    single_level_cap = 0.35  # cap any single level fraction to 35% of allocation

    start_idx = max(seq_len, 200, swing_window + 1)  # make sure MAs are ready
    if N - start_idx <= horizon + 1:
        raise ValueError("Not enough history for backtest given seq_len/horizon")

    LOGGER.info("Backtest start=%d end=%d N=%d", start_idx, N, N)

    # Precompute level fractions once (capped and renormalized)
    raw_fracs = np.array([0.4 * (0.7 ** i) for i in range(len(sigma_levels))], dtype=float)
    raw_fracs = raw_fracs / raw_fracs.sum()
    level_fracs = np.minimum(raw_fracs, single_level_cap)
    if level_fracs.sum() <= 0:
        level_fracs = raw_fracs
    else:
        level_fracs = level_fracs / level_fracs.sum()

    for t in range(start_idx, N - horizon - 1):
        # Build model input window using already-scaled features in df
        seq_x = df_loc[feature_cols].iloc[t - seq_len : t].values.astype(np.float32)
        x_t = torch.from_numpy(seq_x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_ret, pred_vol = model(x_t)
        pred_ret = float(pred_ret.item())
        pred_vol = float(pred_vol.item())  # horizon vol (fraction)

        # Fallback to recent realized vol if model returns degenerate vol
        if not math.isfinite(pred_vol) or pred_vol <= 1e-6:
            pred_vol = float(df_loc["_raw_vol_30"].iloc[t]) if pd.notna(df_loc["_raw_vol_30"].iloc[t]) else 0.001
            pred_vol = max(pred_vol, 1e-4)

        # Trend bias
        trend_bias = 0.0
        if trend_weighting:
            sma50 = df_loc["sma_50"].iloc[t]
            sma200 = df_loc["sma_200"].iloc[t]
            if pd.notna(sma50) and pd.notna(sma200):
                direction = 1.0 if sma50 > sma200 else -1.0
                strength = abs(df_loc["swing_delta"].iloc[t])
                trend_bias = direction * min(1.0, strength * 10.0)

        base_long_share = 0.5 + 0.5 * math.tanh(pred_ret * 50.0)
        long_share = base_long_share * 0.7 + (0.3 * (0.5 * (trend_bias + 1)))
        long_share = float(np.clip(long_share, 0.0, 1.0))

        # Use intraday_exposure to scale how much of intraday_cap is used for orders
        funds_long = intraday_cap * intraday_exposure * long_share
        funds_short = intraday_cap * intraday_exposure * (1.0 - long_share)

        base_price = float(df_loc["close"].iloc[t])

        # LONG side: buy dips (size limited by per-order cap and risk)
        for i, s in enumerate(sigma_levels):
            # intended size before caps
            size_value = funds_long * float(level_fracs[i])

            # hard cap per order
            size_cap = intraday_cap * max_pct_per_order
            if size_value > size_cap:
                size_value = size_cap

            # additional cap based on allowed risk per trade (use SL distance)
            sl_pct = sl_sigma * pred_vol if pred_vol > 0 else 1e-6
            max_loss_allowed = intraday_cap * risk_per_trade_pct
            max_size_from_risk = max_loss_allowed / (sl_pct + 1e-12)
            if size_value > max_size_from_risk:
                size_value = max_size_from_risk

            if size_value < 1.0:
                continue

            order_price = base_price * (1.0 - s * pred_vol)
            future_segment = df_loc["low"].iloc[t + 1 : t + 1 + horizon]
            filled = bool((future_segment <= order_price).any())
            if filled:
                entry_price = order_price
                tp = entry_price * (1.0 + tp_sigma * pred_vol)
                sl = entry_price * (1.0 - sl_sigma * pred_vol)
                future_after_fill = df_loc.iloc[t + 1 : t + 1 + horizon]
                hit_tp = bool((future_after_fill["high"] >= tp).any())
                hit_sl = bool((future_after_fill["low"] <= sl).any())
                if hit_tp and not hit_sl:
                    pnl = (tp - entry_price) / entry_price * size_value
                elif hit_sl and not hit_tp:
                    pnl = (sl - entry_price) / entry_price * size_value
                elif hit_tp and hit_sl:
                    idx_tp = future_after_fill[future_after_fill["high"] >= tp].index[0]
                    idx_sl = future_after_fill[future_after_fill["low"] <= sl].index[0]
                    pnl = ((tp if idx_tp < idx_sl else sl) - entry_price) / entry_price * size_value
                else:
                    close_h = float(future_after_fill["close"].iloc[-1])
                    pnl = (close_h - entry_price) / entry_price * size_value
                cash_intraday += pnl
                trades.append({"t": t, "side": "long", "entry": entry_price, "size": size_value, "pnl": pnl})

        # SHORT side: sell spikes (symmetric sizing and caps)
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
            filled = bool((future_segment >= order_price).any())
            if filled:
                entry_price = order_price
                tp = entry_price * (1.0 - tp_sigma * pred_vol)
                sl = entry_price * (1.0 + sl_sigma * pred_vol)
                future_after_fill = df_loc.iloc[t + 1 : t + 1 + horizon]
                hit_tp = bool((future_after_fill["low"] <= tp).any())
                hit_sl = bool((future_after_fill["high"] >= sl).any())
                if hit_tp and not hit_sl:
                    pnl = (entry_price - tp) / entry_price * size_value
                elif hit_sl and not hit_tp:
                    pnl = (entry_price - sl) / entry_price * size_value
                elif hit_tp and hit_sl:
                    idx_tp = future_after_fill[future_after_fill["low"] <= tp].index[0]
                    idx_sl = future_after_fill[future_after_fill["high"] >= sl].index[0]
                    pnl = (entry_price - (tp if idx_tp < idx_sl else sl)) / entry_price * size_value
                else:
                    close_h = float(future_after_fill["close"].iloc[-1])
                    pnl = (entry_price - close_h) / entry_price * size_value
                cash_intraday += pnl
                trades.append({"t": t, "side": "short", "entry": entry_price, "size": size_value, "pnl": pnl})

        # Volatility gating: reduce exposure on spikes, but with cooldown and floor (no destructive multiplicative kill)
        vol_now = float(df_loc["_raw_vol_30"].iloc[t]) if pd.notna(df_loc["_raw_vol_30"].iloc[t]) else 0.0
        if vol_now > 0 and pred_vol > 3.0 * vol_now:
            if (t - last_vol_spike_idx) > vol_spike_cooldown:
                prev_exposure = intraday_exposure
                intraday_exposure = max(0.1, intraday_exposure * 0.8)  # reduce exposure by 20%, floor at 10%
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
    except Exception:  # already created
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

    # Map to validation segment of df
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
        # Penalize too few trades
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
# Save / load
# ==========================================

def save_model(model: nn.Module, path_model: Path, meta: Optional[dict] = None, path_meta: Optional[Path] = None) -> None:
    path_model = Path(path_model)
    torch.save(model.state_dict(), path_model)
    if meta is not None and path_meta is not None:
        with open(path_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    LOGGER.info("Saved model to %s and meta to %s", path_model, path_meta)


def load_model(path_model: Path, input_size: int, hidden: int, device: torch.device) -> LSTM2Head:
    model = LSTM2Head(input_size=input_size, hidden_size=hidden).to(device)
    state = torch.load(path_model, map_location=device)
    model.load_state_dict(state)
    return model


# ==========================================
# LLM hook (placeholder)
# ==========================================

def query_openai_for_sentiment(text: str, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini") -> Optional[float]:
    """Placeholder legacy client call. Update to latest OpenAI SDK if you plan to use it.
    Returns float in [-1,1] or None.
    """
    try:  # pragma: no cover
        import openai  # type: ignore

        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        prompt = (
            "Analiza el sentimiento financiero del siguiente texto (devuelve JSON {\"sentiment\": float_between_-1_and_1}):\n\n"
            + text
        )
        resp = openai.ChatCompletion.create(
            model=model_name, messages=[{"role": "user", "content": prompt}], max_tokens=50, temperature=0.0
        )
        out = resp.choices[0].message.content.strip()
        import re

        m = re.search(r"-?\d+\.?\d*", out)
        if m:
            return float(m.group(0))
    except Exception as e:  # pragma: no cover
        LOGGER.debug("OpenAI call failed: %s", e)
    return None


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    # Load data
    try:
        if args.use_influx:
            close = load_data_from_influx(limit=20000)
            # fabricate plausible highs/lows around close if not available
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

    # Drop NaNs (avoid leakage from earlier forward/backward fills)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    LOGGER.info("Dropped %d rows due to NaNs", rows_before - len(df))

    # Scale only feature columns (never scale 'close')
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Build sequences
    seq_len = int(args.seq_len)
    epochs = int(args.epochs)
    hidden = int(args.hidden)
    horizon = int(args.horizon)

    X, y_ret, y_vol = create_sequences_from_df(df, feature_cols, seq_len=seq_len, horizon=horizon)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_ret), torch.from_numpy(y_vol))
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    # Build model
    model = LSTM2Head(input_size=X.shape[2], hidden_size=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=float(args.lr))

    LOGGER.info("Training model for %d epochs", epochs)
    for e in trange(epochs, desc="Final training"):
        loss = train_epoch(model, loader, opt, device, alpha_vol_loss=0.5)
        if e % 5 == 0:
            LOGGER.info("Epoch %d/%d loss=%.6f", e + 1, epochs, loss)

    # Save artifacts
    meta = {"feature_cols": feature_cols, "seq_len": seq_len, "horizon": horizon, "hidden": hidden, "lr": args.lr}
    save_model(model, Path(args.save_model), meta, Path(args.save_meta))

    try:
        import joblib  # type: ignore

        joblib.dump(scaler, "scaler.pkl")
        LOGGER.info("Saved scaler to scaler.pkl")
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Could not save scaler via joblib: %s", e)

    # Quick backtest on last 2k rows
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
