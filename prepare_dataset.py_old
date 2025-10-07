#!/usr/bin/env python3
"""
prepare_dataset.py

Carga OHLCV desde SQLite, genera features con fiboevo, crea dataset (train/val/test)
listo para PyTorch (X: [N, seq_len, F], y: [N,]).

Guarda scaler.pkl y meta.json si save_dir es provisto.

Ejemplo de uso (desde CLI):
    python prepare_dataset.py \
      --sqlite data_manager/exports/marketdata_base.db \
      --table ohlcv --symbol BTCUSDT --timeframe 30m \
      --seq_len 32 --horizon 10 --val_frac 0.2 --test_frac 0.1 --save_dir artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# local helper: ensure local project dir is first in sys.path so "import fiboevo" finds local module
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Intentos de dependencias (fallbacks con mensajes claros)
try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

try:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
except Exception:
    torch = None
    TensorDataset = None
    DataLoader = None

# Import fiboevo (debe estar en el mismo dir o en PYTHONPATH)
# We'll check availability below
try:
    import importlib
    import fiboevo  # type: ignore
except Exception:
    fiboevo = None  # we'll handle this later

# ----------------------
# Logging init (file + console)
# ----------------------
def init_logging(log_dir: str = "logs", level: int = logging.INFO) -> Path:
    ld = Path(log_dir)
    ld.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logfile = ld / f"log_{ts}.txt"
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.handlers.RotatingFileHandler(str(logfile), maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    root.info("init_logging: logfile=%s", logfile)
    root.debug("CWD: %s", os.getcwd())
    root.debug("sys.path: %s", sys.path)
    return logfile

# Initialize logging early
LOGFILE = init_logging(level=logging.DEBUG)
LOGGER = logging.getLogger("prepare_dataset")

# ----------------------
# fiboevo availability check
# ----------------------
def check_fiboevo() -> Optional[object]:
    global fiboevo
    try:
        if "fiboevo" in sys.modules:
            importlib.reload(sys.modules["fiboevo"])
            fib = sys.modules["fiboevo"]
        else:
            import fiboevo as fib
        LOGGER.info("fiboevo loaded from: %s", getattr(fib, "__file__", "<builtin/module>"))
        # required function
        if not hasattr(fib, "add_technical_features"):
            LOGGER.error("fiboevo.add_technical_features not present in module loaded from %s", getattr(fib, "__file__", None))
            return None
        # optionally check create_sequences_from_df presence
        if not hasattr(fib, "create_sequences_from_df"):
            LOGGER.warning("fiboevo.create_sequences_from_df not present; prepare_dataset uses its local build_sequences_targets.")
        return fib
    except Exception as e:
        LOGGER.exception("Failed to import fiboevo: %s", e)
        return None

# perform check
FIB = check_fiboevo()
if FIB is None:
    LOGGER.error("fiboevo module missing or incomplete. Aborting dataset preparation.")
    # Do not raise here to allow inspecting logs; caller can decide. But most flows should abort.
    # raise RuntimeError("fiboevo not available")

# ----------------------
# Helpers DB / IO
# ----------------------
def load_ohlcv_sqlite(sqlite_path: str, table: str = "ohlcv", symbol: Optional[str] = None,
                      timeframe: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Lee la tabla 'table' de sqlite y filtra por symbol/timeframe si se entregan.
    Devuelve DataFrame ordenado asc con columnas mínimas: ts (int), open, high, low, close, volume.
    """
    p = Path(sqlite_path)
    if not p.exists():
        raise FileNotFoundError(f"SQLite file not found: {sqlite_path}")

    import sqlite3
    con = sqlite3.connect(str(p))
    # construir query
    q = f"SELECT * FROM {table}"
    conds = []
    params = []
    if symbol:
        conds.append("symbol = ?")
        params.append(symbol)
    if timeframe:
        conds.append("timeframe = ?")
        params.append(timeframe)
    if conds:
        q += " WHERE " + " AND ".join(conds)

    # Intentamos ordenar por 'ts' si existe; si no, hacemos fallback a 'timestamp'
    try:
        q_with_order = q + " ORDER BY ts ASC"
        if limit:
            q_with_order += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(q_with_order, con, params=params)
    except Exception:
        # fallback: ordenar por 'timestamp'
        q_with_order = q + " ORDER BY timestamp ASC"
        if limit:
            q_with_order += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(q_with_order, con, params=params)
    con.close()

    # normalizar columnas: si 'timestamp' existe y 'ts' no, crear ts
    if "ts" not in df.columns and "timestamp" in df.columns:
        try:
            df["ts"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
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

# ----------------------
# Feature pipeline
# ----------------------
def compute_features_with_fiboevo(df_ohlcv: pd.DataFrame, dropna_after: bool = False) -> pd.DataFrame:
    """
    Recibe DataFrame con columns open, high, low, close, volume, timestamp.
    Llama a fiboevo.add_technical_features sobre arrays numericos y devuelve DataFrame de features,
    adjuntando timestamp, close, high, low, volume, symbol/timeframe (if present).
    NO muta df_ohlcv original (trabaja sobre copia).
    """
    if FIB is None:
        raise RuntimeError("fiboevo module not available (compute_features_with_fiboevo)")

    df = df_ohlcv.copy().reset_index(drop=True)
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    vol = df["volume"].astype(float).values if "volume" in df.columns else None

    # call fiboevo.add_technical_features (should return DataFrame)
    feats = FIB.add_technical_features(close, high=high, low=low, volume=vol)
    if not isinstance(feats, pd.DataFrame):
        # be defensive
        feats = pd.DataFrame(feats)

    feats = feats.reset_index(drop=True)
    # attach original OHLCV and timestamp columns to preserve context (we'll exclude them from features)
    feats["timestamp"] = df["timestamp"].values
    feats["open"] = df["open"].values
    feats["high"] = df["high"].values
    feats["low"] = df["low"].values
    feats["close"] = df["close"].values
    feats["volume"] = df["volume"].values if "volume" in df.columns else 0.0

    for tag in ("symbol", "timeframe", "exchange"):
        if tag in df.columns:
            feats[tag] = df[tag].values

    if dropna_after:
        feats = feats.dropna().reset_index(drop=True)

    return feats

# ----------------------
# Build sequences and targets (log-returns target)
# ----------------------
def build_sequences_targets(df_feats: pd.DataFrame, feature_cols: List[str], seq_len: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye X, y desde df_feats ordenado asc. Excluye timestamp y OHLC cols de feature set.
    - X shape: (N, seq_len, F)
    - y shape: (N,)  where y = log(close_{t+h}) - log(close_t)
    Raises informative errors if not enough rows.
    """
    df = df_feats.copy().reset_index(drop=True)

    # ensure feature_cols present (intersection)
    present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        LOGGER.warning("Some requested feature_cols are not present and will be ignored: %s", missing)
    if len(present) == 0:
        raise RuntimeError("No feature columns present after filtering.")

    # convert any datetime features to epoch floats (defensive)
    for c in present:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c]).astype("int64") / 1e9

    M = len(df)
    min_rows = seq_len + horizon
    if M < min_rows:
        raise RuntimeError(f"Not enough rows after cleaning: {M} < seq_len+horizon ({min_rows})")

    # log close (guard replace 0 -> NaN)
    df["__log_close"] = np.log(df["close"].replace(0, np.nan))

    N = M - seq_len - horizon + 1
    F = len(present)
    X = np.zeros((N, seq_len, F), dtype=np.float32)
    y = np.zeros((N,), dtype=np.float32)

    feat_matrix = df[present].astype(np.float32).values
    log_close = df["__log_close"].values

    idx = 0
    for i in range(0, N):
        t_end = i + seq_len - 1
        t_target = t_end + horizon
        X[idx] = feat_matrix[i: i + seq_len, :]
        # defensive: if any log_close involved is NaN, set y to 0 and warn
        if not np.isfinite(log_close[t_end]) or not np.isfinite(log_close[t_target]):
            y_val = 0.0
        else:
            y_val = log_close[t_target] - log_close[t_end]
        y[idx] = float(y_val)
        idx += 1

    return X, y

# ----------------------
# Split, scaler, dataloaders
# ----------------------
def train_val_test_split_indices(N: int, val_frac: float = 0.2, test_frac: float = 0.1) -> Tuple[slice, slice, slice]:
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError("val_frac and test_frac must be non-negative and sum < 1.0")
    n_test = int(np.floor(N * test_frac))
    n_val = int(np.floor(N * val_frac))
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise RuntimeError("Split yields no training samples; reduce val/test fractions or increase data.")
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, N)


def fit_scaler_on_train(X_train: np.ndarray, scaler: Optional[object] = None) -> object:
    if StandardScaler is None:
        raise RuntimeError("scikit-learn no disponible. Instala scikit-learn para usar el scaler.")
    if scaler is None:
        scaler = StandardScaler()
    N, S, F = X_train.shape
    flat = X_train.reshape(N * S, F)
    scaler.fit(flat)
    return scaler


def transform_X_with_scaler(X: np.ndarray, scaler: object) -> np.ndarray:
    N, S, F = X.shape
    flat = X.reshape(N * S, F)
    flat_t = scaler.transform(flat)
    return flat_t.reshape(N, S, F).astype(np.float32)


def to_torch_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle_train: bool = False):
    if torch is None:
        raise RuntimeError("PyTorch no disponible. Instala torch si quieres DataLoader/TensorDataset.")
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float().view(-1)  # 1-D labels for squeeze(-1) models
    ds = TensorDataset(Xt, yt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle_train)
    return loader


# ----------------------
# End-to-end pipeline
# ----------------------
def build_and_save_dataset(sqlite_path: str,
                           table: str,
                           symbol: str,
                           timeframe: str,
                           seq_len: int,
                           horizon: int,
                           val_frac: float = 0.2,
                           test_frac: float = 0.1,
                           save_dir: Optional[str] = None,
                           scaler_path: Optional[str] = None,
                           force_feature_cols: Optional[List[str]] = None) -> Dict:
    LOGGER.info("Starting dataset build: sqlite=%s table=%s symbol=%s timeframe=%s", sqlite_path, table, symbol, timeframe)
    df_ohlcv = load_ohlcv_sqlite(sqlite_path, table=table, symbol=symbol, timeframe=timeframe)
    LOGGER.info("Loaded %d rows from sqlite.", len(df_ohlcv))
    if len(df_ohlcv) == 0:
        raise RuntimeError("No data loaded from sqlite")

    # compute features
    try:
        df_feats = compute_features_with_fiboevo(df_ohlcv)
    except Exception as e:
        LOGGER.exception("Feature computation failed: %s", e)
        raise

    before = len(df_feats)
    df_clean = df_feats.dropna().reset_index(drop=True)
    after = len(df_clean)
    removed = before - after
    LOGGER.info("dropna: before=%d after=%d removed=%d", before, after, removed)

    # Feature selection
    exclude = set(["timestamp", "open", "high", "low", "close", "volume", "symbol", "timeframe", "exchange", "created_at", "updated_at", "ts"])
    if force_feature_cols:
        feature_cols = [c for c in force_feature_cols if c in df_clean.columns and c not in exclude]
        LOGGER.info("Using forced feature_cols (%d): %s", len(feature_cols), feature_cols[:10])
    else:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude]
        if len(feature_cols) == 0:
            coerced = []
            for c in df_clean.columns:
                if c in exclude:
                    continue
                coerced_series = pd.to_numeric(df_clean[c], errors="coerce")
                n_valid = int(coerced_series.notna().sum())
                if n_valid > 0 and n_valid >= max(2, len(df_clean) // 10):
                    coerced.append(c)
                    df_clean[c] = coerced_series
            feature_cols = [c for c in coerced if c not in exclude]

    if len(feature_cols) == 0:
        feature_cols = ["close"]
        LOGGER.warning("No numeric feature columns detected after cleaning; falling back to ['close'] as single feature.")
    LOGGER.info("Feature columns used (%d): %s", len(feature_cols), feature_cols[:20])

    # build sequences/targets
    X_all, y_all = build_sequences_targets(df_clean, feature_cols, seq_len=seq_len, horizon=horizon)
    N, S, F = X_all.shape
    LOGGER.info("Created sequences: N=%d seq_len=%d features=%d", N, S, F)

    # split
    train_slice, val_slice, test_slice = train_val_test_split_indices(N, val_frac=val_frac, test_frac=test_frac)
    Xtr, ytr = X_all[train_slice], y_all[train_slice]
    Xv, yv = X_all[val_slice], y_all[val_slice]
    Xtst, ytst = X_all[test_slice], y_all[test_slice]
    LOGGER.info("Split sizes -> train %d, val %d, test %d", Xtr.shape[0], Xv.shape[0], Xtst.shape[0])

    # fit scaler on train only
    if StandardScaler is None:
        raise RuntimeError("scikit-learn required for scaler")
    scaler = fit_scaler_on_train(Xtr)
    Xtr_s = transform_X_with_scaler(Xtr, scaler)
    Xv_s = transform_X_with_scaler(Xv, scaler)
    Xtst_s = transform_X_with_scaler(Xtst, scaler)

    artifacts: Dict[str, str] = {}
    if save_dir:
        sd = Path(save_dir)
        sd.mkdir(parents=True, exist_ok=True)
        if joblib is None:
            LOGGER.warning("joblib not available: scaler won't be saved.")
        else:
            sp = sd / (scaler_path if scaler_path else "scaler.pkl")
            try:
                joblib.dump(scaler, str(sp))
                artifacts["scaler_path"] = str(sp)
                LOGGER.info("Saved scaler to %s", sp)
            except Exception:
                LOGGER.exception("Failed to save scaler to %s", sp)
        meta = {
            "feature_cols": feature_cols,
            "seq_len": seq_len,
            "horizon": horizon,
            "train_slice_len": int(Xtr.shape[0]),
            "val_slice_len": int(Xv.shape[0]),
            "test_slice_len": int(Xtst.shape[0]),
            "rows_before_dropna": int(before),
            "rows_after_dropna": int(after),
            "removed_warmup": int(removed)
        }
        try:
            with open(sd / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            artifacts["meta_path"] = str(sd / "meta.json")
            LOGGER.info("Saved meta to %s", sd / "meta.json")
        except Exception:
            LOGGER.exception("Failed to save meta.json to %s", sd)

    # dataloaders
    dataloaders = {}
    if torch is not None:
        batch = 64
        Xtr_t = torch.from_numpy(Xtr_s).float()
        ytr_t = torch.from_numpy(ytr).float().view(-1)  # 1-D labels
        ds_tr = TensorDataset(Xtr_t, ytr_t)
        loader_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True)

        Xv_t = torch.from_numpy(Xv_s).float()
        yv_t = torch.from_numpy(yv).float().view(-1)
        loader_val = DataLoader(TensorDataset(Xv_t, yv_t), batch_size=batch, shuffle=False)

        Xtst_t = torch.from_numpy(Xtst_s).float()
        ytst_t = torch.from_numpy(ytst).float().view(-1)
        loader_test = DataLoader(TensorDataset(Xtst_t, ytst_t), batch_size=batch, shuffle=False)

        dataloaders = {"train": loader_tr, "val": loader_val, "test": loader_test}
    else:
        LOGGER.warning("torch not available: returning numpy arrays (X,y) instead of DataLoaders.")

    out = {
        "Xtr": Xtr_s, "ytr": ytr,
        "Xv": Xv_s, "yv": yv,
        "Xtst": Xtst_s, "ytst": ytst,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "dataloaders": dataloaders,
        "artifacts": artifacts,
        "df_clean": df_clean
    }
    LOGGER.info("Dataset build finished.")
    return out

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build dataset from sqlite using fiboevo features and produce PyTorch-ready sets.")
    p.add_argument("--sqlite", type=str, required=True, help="Path to sqlite db")
    p.add_argument("--table", type=str, default="ohlcv")
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--timeframe", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--save_dir", type=str, default="artifacts")
    p.add_argument("--force_features", type=str, default="", help="Comma-separated feature columns to force use (override auto selection)")
    return p.parse_args()


def main():
    args = parse_args()
    force_features = args.force_features.split(",") if args.force_features else None
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
        force_feature_cols=force_features
    )
    LOGGER.info("[DONE] Dataset prepared. Shapes: Xtr=%s ytr=%s Xv=%s yv=%s Xtst=%s ytst=%s", res["Xtr"].shape, res["ytr"].shape, res["Xv"].shape, res["yv"].shape, res["Xtst"].shape, res["ytst"].shape)
    print("[DONE] Dataset preparado. Shapes:")
    print("Xtr", res["Xtr"].shape, "ytr", res["ytr"].shape)
    print("Xv", res["Xv"].shape, "yv", res["yv"].shape)
    print("Xtst", res["Xtst"].shape, "ytst", res["ytst"].shape)
    print("feature_cols:", res["feature_cols"])
    if "artifacts" in res:
        print("artifacts:", res["artifacts"])


if __name__ == "__main__":
    main()
