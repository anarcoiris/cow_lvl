
# ---------------------------
# GUI: pestaña Audit + helpers
# ---------------------------
from pathlib import Path
import numpy as np
import pandas as pd
import math
import json
import time
import threading

def _build_audit_tab(self):
    """Añadir pestaña 'Audit' al Notebook. Llamar desde _build_ui()"""
    tab = Frame(self.nb)
    self.nb.add(tab, text="Audit")

    top = Frame(tab)
    top.pack(fill=X, padx=6, pady=6)

    Label(top, text="Dataset Auditing Tools", font=("Arial", 11, "bold")).pack(anchor="w")

    Button(top, text="Run Audit", command=self._run_audit).pack(side=LEFT, padx=6)
    Button(top, text="Export features CSV", command=lambda: self._export_df(self.df_features, "features.csv")).pack(side=LEFT, padx=6)
    Button(top, text="Export scaled CSV", command=lambda: self._export_df(self.df_scaled, "scaled.csv")).pack(side=LEFT, padx=6)
    Button(top, text="Save Audit Report", command=lambda: self._save_audit_report()).pack(side=LEFT, padx=6)

    # Audit display
    self.audit_text = Text(tab, height=20)
    self.audit_text.pack(fill=BOTH, expand=True, padx=6, pady=6)

    # quick Treeview for a few checks results (optional)
    cols = ("check", "result")
    self.audit_tree = ttk.Treeview(tab, columns=cols, show="headings", height=6)
    for c in cols:
        self.audit_tree.heading(c, text=c)
        self.audit_tree.column(c, width=300, anchor=W)
    self.audit_tree.pack(fill=X, padx=6, pady=(0,6))

# ---------------------------
# Export helper
# ---------------------------
def _export_df(self, df: pd.DataFrame, filename: str):
    if df is None:
        self._append_audit_log(f"No hay dataframe para exportar: {filename}")
        return
    out_path = Path("exports")
    out_path.mkdir(parents=True, exist_ok=True)
    full = out_path / filename
    try:
        df.to_csv(full, index=False)
        self._append_audit_log(f"Exportado {filename} → {full}")
    except Exception as e:
        self._append_audit_log(f"Error exportando {filename}: {e}")

# ---------------------------
# Small logger helper for the audit area
# ---------------------------
def _append_audit_log(self, msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        self.audit_text.insert("end", line)
        self.audit_text.see("end")
    except Exception:
        # fallback if audit_text not ready
        print(line)

# ---------------------------
# Save the last audit report to JSON
# ---------------------------
def _save_audit_report(self, outname: str = None):
    outp = outname or f"audit_report_{int(time.time())}.json"
    outdir = Path("exports")
    outdir.mkdir(exist_ok=True)
    full = outdir / outp
    if not hasattr(self, "_last_audit_report") or self._last_audit_report is None:
        self._append_audit_log("No hay informe de auditoría para guardar.")
        return
    try:
        with open(full, "w", encoding="utf-8") as f:
            json.dump(self._last_audit_report, f, indent=2, default=str)
        self._append_audit_log(f"Audit report guardado en {full}")
    except Exception as e:
        self._append_audit_log(f"Error guardando informe: {e}")

# ---------------------------
# Rutina de auditoría principal
# ---------------------------
def _run_audit(self):
    """
    Ejecuta comprobaciones automáticas sobre:
      - self.df_loaded (raw)
      - self.df_features (technical features, pre-scaled)
      - self.df_scaled  (features escaladas)
      - self.feature_cols_used, self.scaler_used

    Guarda resultados en self._last_audit_report (dict) y los escribe en audit_text / audit_tree.
    """
    # run in background thread to avoid bloquear UI
    def _job():
        report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": [], "summary": {}}
        self._last_audit_report = None

        # Basic availability checks
        if self.df_features is None:
            report["checks"].append({"name": "df_features_present", "ok": False, "msg": "df_features is None"})
            self._append_audit_log("df_features no cargado.")
            self._last_audit_report = report
            return
        else:
            report["checks"].append({"name": "df_features_present", "ok": True, "msg": f"shape={self.df_features.shape}"})

        dff = self.df_features.copy()
        dfs = self.df_scaled.copy() if self.df_scaled is not None else None

        # 1) Metadata columns in features
        meta_cols = [c for c in dff.columns if c.lower() in ("timestamp", "symbol", "created_at", "updated_at", "ts")]
        if meta_cols:
            report["checks"].append({"name": "metadata_columns", "ok": False, "msg": f"Metadata columns present: {meta_cols}"})
        else:
            report["checks"].append({"name": "metadata_columns", "ok": True, "msg": "no metadata columns detected"})

        # 2) NaN / rows dropped detection (requires original df_loaded context)
        if self.df_loaded is not None:
            before = len(self.df_loaded)
            after = len(dff)
            dropped = before - after
            report["checks"].append({"name": "dropna_rows", "ok": True, "msg": f"Rows before={before}, after features dropna={after}, dropped={dropped}"})
        else:
            report["checks"].append({"name": "dropna_rows", "ok": None, "msg": "df_loaded not available"})

        # 3) Scaler checks
        scaler = getattr(self, "scaler_used", None) or getattr(self, "model_scaler", None)
        if scaler is None and dfs is None:
            report["checks"].append({"name": "scaler_presence", "ok": False, "msg": "No scaler y no df_scaled disponibles"})
        else:
            if dfs is not None:
                # compute per-column means/std of scaled df
                means = dfs.mean(numeric_only=True).to_dict()
                stds = dfs.std(numeric_only=True, ddof=0).to_dict()
                # quick heuristic: if means ~ 0 and stds ~1 for many columns -> scaler likely fit on whole df
                near_zero = [c for c, v in means.items() if abs(v) < 1e-2]
                near_one = [c for c, v in stds.items() if abs(v - 1.0) < 0.05]
                pct_zero = len(near_zero) / max(1, len(means))
                pct_one = len(near_one) / max(1, len(stds))
                msg = f"{len(means)} cols: {len(near_zero)}≈0-mean ({pct_zero:.2%}), {len(near_one)}≈1-std ({pct_one:.2%})"
                suspect_full_scaler = pct_zero > 0.8 and pct_one > 0.8
                report["checks"].append({"name": "scaled_stats", "ok": not suspect_full_scaler, "msg": msg})
                if suspect_full_scaler:
                    report["checks"].append({"name": "possible_scaler_fitted_on_all", "ok": False, "msg": "Scaled data statistics suggest scaler was fit on full dataset (means≈0,std≈1). Fit scaler only on train!"})
                else:
                    report["checks"].append({"name": "scaler_stats_ok", "ok": True, "msg": "Scaled data not globally standardized OR scaler applied only to subset"})

            if scaler is not None:
                # check feature_names_in_ alignment if available
                try:
                    if hasattr(scaler, "feature_names_in_"):
                        featnames = list(map(str, scaler.feature_names_in_))
                        present = [c for c in featnames if c in dff.columns]
                        missing = [c for c in featnames if c not in dff.columns]
                        msg = f"scaler.feature_names_in_ count={len(featnames)}, missing in df_features={len(missing)}"
                        ok = len(missing) == 0
                        report["checks"].append({"name": "scaler_feature_names_alignment", "ok": ok, "msg": msg, "missing": missing})
                    else:
                        report["checks"].append({"name": "scaler_feature_names", "ok": None, "msg": "scaler has no feature_names_in_ attribute"})
                except Exception as e:
                    report["checks"].append({"name": "scaler_feature_names_err", "ok": False, "msg": str(e)})

        # 4) Exact-match leakage: features that equal future close shifted (close[t+k])
        # We'll test k in 1..min(5, horizon)
        if "close" in dff.columns:
            M = min(5, int(getattr(self, "horizon", 5)))
            matches = []
            for k in range(1, M + 1):
                shifted = dff["close"].shift(-k)
                # For each feature, test if values equal shifted close (on the overlapping slice)
                for col in dff.columns:
                    if col == "close": 
                        continue
                    a = dff[col].values
                    b = shifted.values
                    # align non-nans indices
                    valid = ~np.isnan(b) & ~np.isnan(a)
                    if valid.sum() < 10:
                        continue
                    # normalize floats with relatively tolerant check
                    if np.allclose(a[valid], b[valid], atol=1e-6, rtol=1e-6):
                        matches.append({"feature": col, "shift": k})
            if matches:
                report["checks"].append({"name": "exact_shift_matches", "ok": False, "msg": f"Found {len(matches)} features that equal close.shift(-k)", "matches": matches})
            else:
                report["checks"].append({"name": "exact_shift_matches", "ok": True, "msg": "No exact matches to future close detected"})

        # 5) High correlation with future close (heuristic)
        corr_flags = []
        if "close" in dff.columns:
            horizon = int(getattr(self, "horizon", 1))
            future_close = dff["close"].shift(-horizon)
            for col in dff.select_dtypes(include=[np.number]).columns:
                if col == "close":
                    continue
                valid = (~dff[col].isna()) & (~future_close.isna())
                if valid.sum() < 20:
                    continue
                c = np.corrcoef(dff[col].values[valid], future_close.values[valid])[0, 1]
                if not np.isfinite(c):
                    continue
                if abs(c) > 0.95:  # very high correlation -> suspicious
                    corr_flags.append({"feature": col, "corr_with_future_close": float(c)})
            if corr_flags:
                report["checks"].append({"name": "very_high_corr_with_future_close", "ok": False, "msg": f"{len(corr_flags)} features highly correlated (>0.95) with future close", "details": corr_flags})
            else:
                report["checks"].append({"name": "corr_with_future_close", "ok": True, "msg": "No features with extremely high corr (>0.95) with future close"})

        # 6) Feature naming heuristics (suspicious patterns)
        suspicious_names = [c for c in dff.columns if any(x in c.lower() for x in ["shift", "t+1", "next", "future", "target", "_lead", "_lag"]) ]
        if suspicious_names:
            report["checks"].append({"name": "suspicious_feature_names", "ok": False, "msg": f"Found suspicious names: {suspicious_names}"})
        else:
            report["checks"].append({"name": "suspicious_feature_names", "ok": True, "msg": "No suspicious names found"})

        # 7) Recommend split indices for correct scaler fitting (if possible)
        # Heuristic: if df length > seq_len + horizon, show recommended train_rows_end for val_frac if present
        n_total = len(dff)
        seq_len = int(getattr(self, "seq_len", 32))
        val_frac = float(getattr(self, "val_frac", 0.2))
        n_sequences = n_total - seq_len - int(getattr(self, "horizon", 1)) + 1
        n_val_seq = max(1, int(round(val_frac * n_sequences)))
        n_train_seq = n_sequences - n_val_seq
        train_rows_end = seq_len + n_train_seq - 1
        report["checks"].append({"name": "recommended_train_end", "ok": True, "msg": f"n_total={n_total}, n_seq={n_sequences}, recommended train_rows_end_idx={train_rows_end} (use df.iloc[:{train_rows_end+1}] to fit scaler)"})

        # 8) Quick summary
        report["summary"] = {
            "n_features": len(dff.columns),
            "n_rows": n_total,
            "seq_len": seq_len,
            "horizon": int(getattr(self, "horizon", 1)),
            "scaler_present": scaler is not None,
        }

        # Save report to object and print to UI
        self._last_audit_report = report

        # Update GUI: textual and table
        try:
            self.audit_text.delete("1.0", "end")
            for chk in report["checks"]:
                ok = chk.get("ok", None)
                tag = "OK" if ok else ("WARN" if ok is False else "N/A")
                line = f"{tag:4} {chk['name']:35} - {chk.get('msg','')}\n"
                self.audit_text.insert("end", line)
            # populate tree with main flags (first 6)
            for i in self.audit_tree.get_children():
                self.audit_tree.delete(i)
            for chk in report["checks"][:8]:
                self.audit_tree.insert("", "end", values=(chk["name"], chk.get("msg","")))
            self.audit_text.see("end")
        except Exception:
            print("Audit report:", report)

    # run job in thread
    t = threading.Thread(target=_job, daemon=True)
    t.start()
