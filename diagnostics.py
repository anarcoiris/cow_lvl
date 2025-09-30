# diagnostics.py (snippet)
import json, numpy as np, pandas as pd, math
def check_model_vs_features(model_meta: dict, scaler, feats_df: pd.DataFrame, verbose=True):
    fc_meta = model_meta.get("feature_cols", [])
    if verbose: print("meta feature_cols len:", len(fc_meta))
    # 1) Check presence
    missing = [c for c in fc_meta if c not in feats_df.columns]
    extra = [c for c in feats_df.columns if c not in fc_meta]
    print("missing in feats:", missing)
    print("extra in feats (not in meta):", extra[:10], "..." if len(extra)>10 else "")
    # 2) Scaler names
    if hasattr(scaler, "feature_names_in_"):
        sn = list(getattr(scaler, "feature_names_in_"))
        print("scaler.feature_names_in_ len:", len(sn))
        print("scaler first 10:", sn[:10])
        print("meta first 10:", fc_meta[:10])
        if sn != fc_meta:
            print("MISMATCH: scaler.feature_names_in_ != meta['feature_cols']")
    else:
        print("Scaler has no feature_names_in_")
    # 3) Basic value checks
    if len(fc_meta)>0 and fc_meta[0] in feats_df.columns:
        print("sample raw values (first row):")
        print(feats_df[fc_meta].iloc[:3].to_string())
