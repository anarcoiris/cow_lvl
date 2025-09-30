#!/usr/bin/env python3
# check_env.py -- smoke tests for fiboevo + prepare_dataset + model + scaler
from __future__ import annotations
import logging, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import argparse
import importlib

def init_logging(level=logging.INFO):
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = logs / f"log_{ts}.txt"
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    root.setLevel(level)
    # clear handlers
    for h in list(root.handlers):
        root.removeHandler(h)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
    fh = logging.FileHandler(fname, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
    root.addHandler(ch); root.addHandler(fh)
    return fname

LOGFILE = init_logging()
LOGGER = logging.getLogger("check_env")

def main(cfg_path=None):
    LOGGER.info("Starting environment checks. Logfile: %s", LOGFILE)
    # dynamic import of fiboevo
    try:
        import fiboevo
    except Exception as e:
        LOGGER.exception("Failed to import fiboevo: %s", e)
        raise
    # config / defaults
    cfg = {}
    if cfg_path:
        cfg = json.loads(Path(cfg_path).read_text())
    sqlite = cfg.get("sqlite_path", "data.db")
    table = cfg.get("table", "ohlcv")
    symbol = cfg.get("symbol", None)
    timeframe = cfg.get("timeframe", None)
    seq_len = int(cfg.get("seq_len", 32))
    horizon = int(cfg.get("horizon", 10))
    model_dir = Path(cfg.get("model_dir", "artifacts"))
    scaler_path = cfg.get("scaler_path", "artifacts/scaler.pkl")
    meta_path = Path(cfg.get("meta_path", "artifacts/meta.json"))

    # resolve model
    try:
        # reuse daemon-style resolution if available
        from trading_daemon import _resolve_model_path
        model_file = _resolve_model_path(model_dir)
    except Exception:
        # fallback: try common names
        model_file = None
        for cand in ("model_best.pt","final_model.pt","model.pt","model.pth"):
            p = Path(model_dir) / cand
            if p.exists():
                model_file = p
                break
        if model_file is None:
            LOGGER.warning("No model file found in %s", model_dir)
            model_file = None
    LOGGER.info("Resolved model directory %s -> %s", model_dir, model_file)

    # Try load model via fiboevo.load_model if available
    model = None; meta = {}
    try:
        if model_file:
            model, meta = fiboevo.load_model(Path(model_file))
            LOGGER.info("Model loaded OK. meta keys: %s", list(meta.keys()))
        else:
            LOGGER.warning("Skipping model load; no file")
    except Exception as e:
        LOGGER.exception("Model load failed: %s", e)

    # Try load scaler via joblib if path exists
    scaler = None
    try:
        import joblib
        p = Path(scaler_path)
        if p.exists():
            scaler = joblib.load(str(p))
            LOGGER.info("Scaler loaded from %s", p)
        else:
            LOGGER.warning("Scaler not found at %s", p)
    except Exception as e:
        LOGGER.exception("Scaler load error: %s", e)

    # Load sample ohlcv from sqlite using prepare_dataset loader if present
    try:
        from prepare_dataset import load_ohlcv_sqlite, compute_features_with_fiboevo
        df = load_ohlcv_sqlite(sqlite, table=table, symbol=symbol, timeframe=timeframe, limit=5000)
        LOGGER.info("Loaded %d rows from sqlite", len(df))
        feats = compute_features_with_fiboevo(df)
        LOGGER.info("Computed features. Shape: %s", feats.shape)
        before = len(feats)
        feats_clean = feats.dropna().reset_index(drop=True)
        after = len(feats_clean)
        LOGGER.info("After dropna: before=%d after=%d removed=%d", before, after, before-after)
    except Exception as e:
        LOGGER.exception("Failed to load/compute features: %s", e)
        raise

    # Build sequences quick smoke
    try:
        from prepare_dataset import build_sequences_targets
        X,y = build_sequences_targets(feats_clean, feature_cols=meta.get("feature_cols", [c for c in feats_clean.columns if c not in ['timestamp','open','close','high','low','volume']]) , seq_len=seq_len, horizon=horizon)
        LOGGER.info("build_sequences_targets success: X.shape=%s y.shape=%s", X.shape, y.shape)
    except Exception as e:
        LOGGER.exception("build_sequences_targets failed: %s", e)

    # Quick inference smoke
    if model is not None:
        try:
            # prepare last seq
            last_X = X[-1].astype(np.float32)
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            xt = torch.from_numpy(last_X).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                out = model(xt)
            LOGGER.info("Inference ok. output types: %s", type(out))
        except Exception as e:
            LOGGER.exception("Inference smoke test failed: %s", e)

    LOGGER.info("Environment checks finished. See logfile: %s", LOGFILE)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="")
    args = ap.parse_args()
    cfgp = args.config if args.config else None
    main(cfgp)
