#!/usr/bin/env python3
"""
prepare_dataset.py

Carga OHLCV desde SQLite, genera features con fiboevo (unificado), crea dataset (train/val/test)
listo para PyTorch (X: [N, seq_len, F], y: [N,]).

Características:
 - manejo robusto de symbol/timeframe (normalización + heurísticas de consulta a SQLite)
 - conversiones numpy->torch forzando dtype=float32
 - comprobaciones explícitas de shapes y dtypes
 - logging consistente en lugar de prints
 - uso de funciones helper (to_tensor_float32 / ensure_float32_tensor) si están disponibles en fiboevo,
   con fallback a implementaciones locales
 - escala (StandardScaler) ajustada solo sobre filas de train (evitar data leakage)
 - guardado de artifacts: scaler.pkl y meta.json
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional deps
try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

# Torch optional
try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
except Exception:
    torch = None
    TensorDataset = None
    DataLoader = None

# Try to import fiboevo unified module (may provide helpers)
try:
    import fiboevo as fiboevo
except Exception:
    fiboevo = None

# Attempt to import helpers from fiboevo; fallback to local implementations
try:
    from fiboevo import normalize_timeframe as fb_normalize_timeframe, normalize_symbol as fb_normalize_symbol
except Exception:
    fb_normalize_timeframe = None
    fb_normalize_symbol = None

try:
    from fiboevo import to_tensor_float32 as fb_to_tensor_float32, ensure_float32_tensor as fb_ensure_float32_tensor
except Exception:
    fb_to_tensor_float32 = None
    fb_ensure_float32_tensor = None

# Local fallback implementations
TIMEFRAME_MAP = {
    '1m': '1m', '1min': '1m', '1M': '1m',
    '3m': '3m', '3min': '3m',
    '5m': '5m', '5min': '5m',
    '15m': '15m', '15min': '15m',
    '30m': '30m', '30min': '30m', '30M': '30m',
    '1h': '1h', '60m': '1h', '1H': '1h',
    '4h': '4h', '1d': '1d', '1D': '1d', 'daily': '1d'
}
COMMON_QUOTE_ASSETS = ['USDT', 'USD', 'BTC', 'ETH', 'EUR', 'USDC']


def normalize_timeframe(tf: str) -> str:
    """Normalize various user timeframe representations to canonical like '30m', '1h', '1d'."""
    if fb_normalize_timeframe:
        try:
            return fb_normalize_timeframe(tf)
        except Exception:
            pass
    if tf is None:
        raise ValueError("timeframe is None")
    t = str(tf).strip()
    t_low = t.lower()
    if t_low in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[t_low]
    m = re.match(r"^(\d+)\s*(m|min|minutes?)$", t_low)
    if m:
        return f"{int(m.group(1))}m"
    m = re.match(r"^(\d+)\s*(h|hours?)$", t_low)
    if m:
        return f"{int(m.group(1))}h"
    # fallback: return original trimmed
    return t


def normalize_symbol(sym: str, default_quote_candidates: Optional[List[str]] = None) -> str:
    """
    Normalize symbol to a canonical form (prefer base/quote with '/').
    Example: 'BTCUSDT' -> 'BTC/USDT' or 'BTC/USDT' stays same.
    """
    if fb_normalize_symbol:
        try:
            return fb_normalize_symbol(sym)
        except Exception:
            pass
    if sym is None:
        raise ValueError("symbol is None")
    s = str(sym).strip().upper()
    # unify separators
    s = s.replace('-', '/').replace('_', '/')
    if '/' in s:
        base, quote = [p.strip() for p in s.split('/', 1)]
        return f"{base}/{quote}"
    if default_quote_candidates is None:
        default_quote_candidates = COMMON_QUOTE_ASSETS
    for q in default_quote_candidates:
        if s.endswith(q):
            base = s[:-len(q)]
            if base:
                return f"{base}/{q}"
    # fallback to uppercase original
    return s


def _ensure_dir(p: Union[str, Path]):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)


def _try_alternate_symbol_variants(con, table: str, base_query: str, params: List[Any], symbol: str, timeframe: Optional[str], limit: Optional[int]) -> pd.DataFrame:
    """
    When initial SQL returns no rows, attempt symbol/timeframe variants heuristics.
    """
    import pandas as _pd

    def _exec(q, p):
        try:
            return _pd.read_sql_query(q, con, params=p)
        except Exception:
            return _pd.DataFrame()

    alt_symbols = []
    if '/' in symbol:
        alt_symbols.append(symbol.replace('/', ''))
        alt_symbols.append(symbol.replace('/', '').replace('-', ''))
    else:
        # try add common quotes
        for q in COMMON_QUOTE_ASSETS:
            if symbol.endswith(q):
                base = symbol[:-len(q)]
                if base:
                    alt_symbols.append(f"{base}/{q}")
        # also try insert slash every 3..5 chars (heuristic)
        if len(symbol) > 6:
            alt_symbols.append('/'.join([symbol[:-4], symbol[-4:]]))

    for sym in alt_symbols:
        q = base_query + " AND symbol = ?"
        p = params + [sym]
        if limit:
            q = q + f" LIMIT {int(limit)}"
        df = _exec(q, p)
        if not df.empty:
            logger.info("SQLite: found rows with alternate symbol variant '%s'", sym)
            return df

    # timeframe alternate
    if timeframe:
        try:
            tf_norm = normalize_timeframe(timeframe)
            if tf_norm != timeframe:
                q = base_query + " AND timeframe = ?"
                p2 = params + [tf_norm]
                if limit:
                    q = q + f" LIMIT {int(limit)}"
                df = _exec(q, p2)
                if not df.empty:
                    logger.info("SQLite: found rows with alternate timeframe '%s'", tf_norm)
                    return df
        except Exception:
            pass

    return pd.DataFrame()


def load_ohlcv_sqlite(sqlite_path: str, table: str = "ohlcv", symbol: Optional[str] = None,
                      timeframe: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Read OHLCV rows from sqlite. Returns ascending-sorted DataFrame with at least:
    ts (int epoch seconds), open, high, low, close, volume, timestamp (datetime).
    """
    p = Path(sqlite_path)
    if not p.exists():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

    import sqlite3
    con = sqlite3.connect(str(p))

    base_q = f"SELECT * FROM {table}"
    conds: List[str] = []
    params: List[Any] = []
    if symbol:
        conds.append("symbol = ?")
        params.append(symbol)
    if timeframe:
        conds.append("timeframe = ?")
        params.append(timeframe)
    q = base_q
    if conds:
        q = q + " WHERE " + " AND ".join(conds)

    def _exec_query(order_col: str):
        q_with_order = q + f" ORDER BY {order_col} ASC"
        if limit:
            q_with_order += f" LIMIT {int(limit)}"
        return pd.read_sql_query(q_with_order, con, params=params)

    df = pd.DataFrame()
    tried = []
    try:
        df = _exec_query("ts")
    except Exception:
        tried.append("ts")
        try:
            df = _exec_query("timestamp")
        except Exception:
            tried.append("timestamp")
            con.close()
            raise RuntimeError(f"Failed query ordering by {tried}")

    if df.empty and (symbol or timeframe):
        df_alt = _try_alternate_symbol_variants(con, table, base_q, [], symbol or "", timeframe, limit)
        if not df_alt.empty:
            df = df_alt

    con.close()

    if df.empty:
        logger.warning("No rows found in sqlite for table=%s symbol=%s timeframe=%s", table, symbol, timeframe)
        return df

    # normalize ts/timestamp
    if "ts" not in df.columns and "timestamp" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10 ** 9
        except Exception:
            df["ts"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(method="ffill").astype(int)

    required = ["ts", "open", "high", "low", "close"]
    for r in required:
        if r not in df.columns:
            raise RuntimeError(f"Required column '{r}' not found in sqlite table '{table}'. Found columns: {list(df.columns)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["ts"] = df["ts"].astype(int)
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_features_with_fiboevo(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features via fiboevo.add_technical_features.
    Ensures numeric columns are float32 for compactness.
    """
    if fiboevo is None or not hasattr(fiboevo, "add_technical_features"):
        raise RuntimeError("fiboevo.add_technical_features not available in environment.")

    df = df_ohlcv.copy().reset_index(drop=True)
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values if "high" in df.columns else None
    low = df["low"].astype(float).values if "low" in df.columns else None
    vol = df["volume"].astype(float).values if "volume" in df.columns else None

    feats = fiboevo.add_technical_features(close, high=high, low=low, volume=vol)
    if not isinstance(feats, pd.DataFrame):
        feats = pd.DataFrame(feats)

    feats = feats.reset_index(drop=True)

    # attach original ohlcv/time cols if missing
    for col in ("timestamp", "open", "high", "low", "close", "volume"):
        if col in df.columns and col not in feats.columns:
            feats[col] = df[col].values

    for tag in ("symbol", "timeframe", "exchange"):
        if tag in df.columns:
            feats[tag] = df[tag].values

    # Ensure numeric columns float32
    for c in feats.select_dtypes(include=[np.number]).columns:
        feats[c] = feats[c].astype(np.float32)

    return feats


def create_sequences_from_df(df: pd.DataFrame, feature_cols: List[str], seq_len: int = 32, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences from features dataframe.

    Returns:
      X: np.ndarray shape (N, seq_len, F) dtype float32
      y_ret: np.ndarray shape (N,) dtype float32  # log returns
      y_vol: np.ndarray shape (N,) dtype float32  # volatility proxy (std of returns in window)
    """
    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be DataFrame")
    if "close" not in df.columns:
        raise ValueError("close column required in df")
    df = df.reset_index(drop=True).copy()

    present = [c for c in feature_cols if c in df.columns]
    if len(present) == 0:
        return np.zeros((0, seq_len, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    close = df["close"].astype(np.float64).values
    with np.errstate(divide='ignore', invalid='ignore'):
        logc = np.log(close + 1e-12)

    mat = df[present].astype(np.float32).values
    M = len(df)
    min_rows = seq_len + horizon
    if M < min_rows:
        return np.zeros((0, seq_len, mat.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    N = M - seq_len - horizon + 1
    F = mat.shape[1]
    X = np.zeros((N, seq_len, F), dtype=np.float32)
    y_ret = np.zeros((N,), dtype=np.float32)
    y_vol = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        X[i] = mat[i:i+seq_len]
        # target: log return between end of window and horizon ahead
        idx_end = i + seq_len - 1
        idx_target = idx_end + horizon
        y_ret[i] = float(logc[idx_target] - logc[idx_end])
        # vol proxy: std of log-returns within the window
        win_close = close[i:i+seq_len+1]
        if len(win_close) >= 2:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.diff(np.log(win_close + 1e-12))
            y_vol[i] = float(np.nanstd(r.astype(np.float64)))
        else:
            y_vol[i] = 0.0

    return X, y_ret, y_vol


def train_val_test_split_indices(N: int, val_frac: float = 0.2, test_frac: float = 0.1) -> Tuple[slice, slice, slice]:
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError("val_frac and test_frac must be non-negative and sum < 1.0")
    n_test = int(np.floor(N * test_frac))
    n_val = int(np.floor(N * val_frac))
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise RuntimeError("Split yields no training samples; reduce val/test fractions or increase data.")
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, N)


def fit_scaler_on_train(X_train: np.ndarray, scaler: Optional[Any] = None) -> Any:
    """
    Ajusta StandardScaler sobre X_train (N, seq_len, F) -> (N*seq_len, F)
    """
    if StandardScaler is None:
        raise RuntimeError("scikit-learn no disponible. Instala scikit-learn para usar el scaler.")
    if scaler is None:
        scaler = StandardScaler()
    N, S, F = X_train.shape
    flat = X_train.reshape(N * S, F)
    scaler.fit(flat)
    return scaler


def transform_X_with_scaler(X: np.ndarray, scaler: Any) -> np.ndarray:
    N, S, F = X.shape
    flat = X.reshape(N * S, F)
    flat_t = scaler.transform(flat)
    return flat_t.reshape(N, S, F).astype(np.float32)


def to_torch_dataset(X: np.ndarray, y: np.ndarray, y_vol: Optional[np.ndarray] = None, batch_size: int = 64, shuffle_train: bool = False, device: Optional[Any] = None):
    """
    Convert numpy arrays to PyTorch DataLoader. If y_vol provided, TensorDataset contains 3 tensors (X,y,yvol).
    Targets are forced to view(-1) i.e. shape (N,).
    """
    if torch is None:
        raise RuntimeError("PyTorch no disponible. Instala torch si quieres DataLoader/TensorDataset.")
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device).view(-1)
    if y_vol is not None:
        yvt = torch.tensor(y_vol, dtype=torch.float32, device=device).view(-1)
        ds = TensorDataset(Xt, yt, yvt)
    else:
        ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle_train)
    return loader


def build_and_save_dataset(sqlite_path: str,
                           table: str,
                           symbol: str,
                           timeframe: str,
                           seq_len: int,
                           horizon: int,
                           val_frac: float = 0.2,
                           test_frac: float = 0.1,
                           save_dir: Optional[str] = None,
                           scaler_filename: Optional[str] = None,
                           batch_size: int = 64,
                           device: Optional[Any] = None) -> Dict[str, Any]:
    """
    End-to-end pipeline:
     - load OHLCV from sqlite
     - compute features with fiboevo.add_technical_features
     - dropna and remove warmup rows
     - create sequences X/y (log returns)
     - split train/val/test (indices)
     - fit scaler on train rows only and transform
     - optionally save scaler and meta.json
     - return dict with arrays and dataloaders (if torch available)
    """
    tf_norm = normalize_timeframe(timeframe) if timeframe else timeframe
    sym_norm = normalize_symbol(symbol) if symbol else symbol

    df_ohlcv = load_ohlcv_sqlite(sqlite_path, table=table, symbol=sym_norm, timeframe=tf_norm)
    if df_ohlcv is None or df_ohlcv.empty:
        raise RuntimeError(f"No data loaded from sqlite (table={table}, symbol={symbol}, timeframe={timeframe})")

    df_feats = compute_features_with_fiboevo(df_ohlcv)

    before = len(df_feats)
    df_clean = df_feats.dropna().reset_index(drop=True)
    after = len(df_clean)
    removed = before - after
    logger.info("Filas totales: %d; Filas tras dropna: %d; Eliminadas: %d", before, after, removed)

    # select feature columns
    exclude = {"timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "exchange", "created_at", "updated_at", "ts"}
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    if len(feature_cols) == 0:
        # attempt coercion
        coerced = []
        for c in df_clean.columns:
            if c in exclude:
                continue
            coerced_series = pd.to_numeric(df_clean[c], errors="coerce")
            n_valid = int(coerced_series.notna().sum())
            if n_valid > 0 and n_valid >= max(2, len(df_clean) // 10):
                coerced.append(c)
                df_clean[c] = coerced_series.astype(np.float32)
        feature_cols = [c for c in coerced if c not in exclude]

    if len(feature_cols) == 0:
        feature_cols = ["close"]
        logger.warning("No numeric feature columns detected after cleaning; falling back to ['close'] as single feature.")
    else:
        logger.info("Selected %d numeric feature columns for modeling", len(feature_cols))

    X_all, y_all, vol_all = create_sequences_from_df(df_clean, feature_cols, seq_len=seq_len, horizon=horizon)
    if X_all is None or X_all.shape[0] == 0:
        raise RuntimeError("No sequences produced by create_sequences_from_df")

    N = X_all.shape[0]
    train_slice, val_slice, test_slice = train_val_test_split_indices(N, val_frac=val_frac, test_frac=test_frac)

    Xtr, ytr, voltr = X_all[train_slice], y_all[train_slice], vol_all[train_slice]
    Xv, yv, volv = X_all[val_slice], y_all[val_slice], vol_all[val_slice]
    Xtst, ytst, voltst = X_all[test_slice], y_all[test_slice], vol_all[test_slice]

    # Fit scaler only on training rows (train_rows_end logic equivalently applied)
    if StandardScaler is None:
        raise RuntimeError("scikit-learn no instalado; instala scikit-learn para usar scaler.")
    scaler = fit_scaler_on_train(Xtr)

    Xtr_s = transform_X_with_scaler(Xtr, scaler)
    Xv_s = transform_X_with_scaler(Xv, scaler) if Xv.size else Xv
    Xtst_s = transform_X_with_scaler(Xtst, scaler) if Xtst.size else Xtst

    artifacts: Dict[str, Any] = {}
    if save_dir:
        sd = Path(save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        if joblib is None:
            logger.warning("joblib no disponible: no se guardará el scaler (instala joblib).")
        else:
            sp = sd / (scaler_filename if scaler_filename else "scaler.pkl")
            try:
                joblib.dump(scaler, str(sp))
                artifacts["scaler_path"] = str(sp)
                logger.info("Scaler guardado en %s", sp)
            except Exception:
                logger.exception("Failed saving scaler")
        meta = {
            "feature_cols": feature_cols,
            "seq_len": seq_len,
            "horizon": horizon,
            "train_samples": int(Xtr.shape[0]),
            "val_samples": int(Xv.shape[0]),
            "test_samples": int(Xtst.shape[0]),
            "rows_before_dropna": int(before),
            "rows_after_dropna": int(after),
            "removed_warmup": int(removed),
            "sqlite_path": str(sqlite_path),
            "table": table, "symbol": symbol, "timeframe": timeframe
        }
        try:
            (sd / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            artifacts["meta_path"] = str(sd / "meta.json")
            logger.info("Meta guardada en %s", sd / "meta.json")
        except Exception:
            logger.exception("Failed saving meta.json")

    dataloaders: Dict[str, Any] = {}
    if torch is not None:
        device_obj = device if device is not None else None
        dataloaders["train"] = to_torch_dataset(Xtr_s, ytr, y_vol=voltr, batch_size=batch_size, shuffle_train=True, device=device_obj)
        dataloaders["val"] = to_torch_dataset(Xv_s, yv, y_vol=volv, batch_size=batch_size, shuffle_train=False, device=device_obj)
        dataloaders["test"] = to_torch_dataset(Xtst_s, ytst, y_vol=voltst, batch_size=batch_size, shuffle_train=False, device=device_obj)
    else:
        logger.info("torch no disponible: retornando numpy arrays en lugar de DataLoaders")

    out = {
        "Xtr": Xtr_s, "ytr": ytr, "voltr": voltr,
        "Xv": Xv_s, "yv": yv, "volv": volv,
        "Xtst": Xtst_s, "ytst": ytst, "voltst": voltst,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "dataloaders": dataloaders,
        "artifacts": artifacts,
        "df_clean": df_clean,
    }
    return out


# ----------------------
# CLI
# ----------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Build dataset from sqlite using fiboevo features and produce PyTorch-ready sets.")
    p.add_argument("--sqlite", type=str, required=True, help="Path to sqlite db")
    p.add_argument("--table", type=str, default="ohlcv")
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--timeframe", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--save_dir", type=str, default="artifacts")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default=None, help="Torch device (e.g. 'cpu' or 'cuda:0')")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device is not None and torch is not None:
        device = torch.device(device)
    res = build_and_save_dataset(
        sqlite_path=args.sqlite,
        table=args.table,
        symbol=args.symbol,
        timeframe=args.timeframe,
        seq_len=args.seq_len,
        horizon=args.horizon,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        device=device
    )
    logger.info("[DONE] Dataset preparado. Shapes:")
    logger.info("Xtr %s ytr %s", getattr(res['Xtr'], 'shape', type(res['Xtr'])), getattr(res['ytr'], 'shape', type(res['ytr'])))
    logger.info("Xv %s yv %s", getattr(res['Xv'], 'shape', type(res['Xv'])), getattr(res['yv'], 'shape', type(res['yv'])))
    logger.info("Xtst %s ytst %s", getattr(res['Xtst'], 'shape', type(res['Xtst'])), getattr(res['ytst'], 'shape', type(res['ytst'])))
    logger.info("feature_cols: %s", res["feature_cols"])
    if "artifacts" in res:
        logger.info("artifacts: %s", res["artifacts"])


if __name__ == "__main__":
    main()
