#!/usr/bin/env python3
# trading_gui_extended.py
"""
 Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№
  Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№
   Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№
    Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№
     Duplicado !! --  Y cuidado, quitar esto de aqui pronto...!!!! /№№;№;№;;№;№

    def _connect_websocket(self):
        if websocket is None:
            self._enqueue_log("websocket-client library missing. Run: pip install websocket-client")
            return
        url = self.websocket_url_var.get().strip()
        if not url:
            self._enqueue_log("Websocket URL empty.")
            self.websocket_status_var.set("Disconnected")
            return

        # If already connected, ignore
        if getattr(self, "ws_connected", False):
            self._enqueue_log("Already connected.")
            return

        # Create WebSocketApp with callbacks
        def on_open(ws):
            self.ws_queue.put({"type":"_meta","event":"open"})
            self._enqueue_log("WS open (callback)")

        def on_close(ws, close_status_code, close_msg):
            self.ws_queue.put({"type":"_meta","event":"close", "code": close_status_code, "msg": close_msg})
            self._enqueue_log(f"WS closed: {close_status_code} {close_msg}")

        def on_error(ws, err):
            self.ws_queue.put({"type":"_meta","event":"error", "error": str(err)})
            self._enqueue_log(f"WS error: {err}")

        def on_message(ws, message):
            # Put raw message in queue; actual parsing done on main thread to avoid race conditions with Tk.
            self.ws_queue.put({"type":"message","raw": message})

        self.ws_app = websocket.WebSocketApp(url,
                                             on_open=on_open,
                                             on_message=on_message,
                                             on_error=on_error,
                                             on_close=on_close)
        # Run in thread
        self.ws_thread = threading.Thread(target=lambda: self.ws_app.run_forever(ping_interval=30, ping_timeout=10), daemon=True)
        self.ws_thread.start()
        self.websocket_status_var.set("Connecting...")
        # schedule queue processor
        try:
            self.nb.after(150, self._process_ws_queue)
        except Exception:
            # fallback to top-level window if nb missing
            self.after(150, self._process_ws_queue)


- Uses fiboevo for features and model (expects fiboevo.add_technical_features and LSTM2Head)
- Thread-safe logging via queue
- Separate Prepare(Data) and Train(Model) workers; also combined Prepare+Train
- Saves artifacts in ./artifacts/

 Asuntos a comprobar:

Implementar websocket real (streaming ticker/push) — necesito URL y protocolo; se puede usar websocket-client o asyncio websockets.

Guardado seguro de api_key/api_secret (usar keyring o cifrado).

Indicador visual de carga/estado del modelo (por ejemplo, icono o texto "Model loaded" en Status).

Posibilidad de reiniciar/recargar el daemon loop tras un load_model_and_scaler si lo deseas.

Añadir opción para que Get Latest Prediction use exactamente la misma preprocessing que el daemon (si cambian feature_cols) — ahora intento preferir daemon.model_meta['feature_cols'] si existe.

Mostrar histórico de predicciones / mini-gráfica (requiere matplotlib embedding).

Mejor manejo de timeframes complejos (semana/mes) — ampliar timeframe_to_seconds.

Automatizar validación de modelo/compatibilidad antes de cargar (uso de meta.json).

Soporte para múltiples tickers/exchanges simultáneos en la pestaña Status (si necesitas monitorizar varios).
"""

from __future__ import annotations
import threading
import time
import json
import traceback

# --------------- logging setup
import logging
import logging.handlers
import os
from pathlib import Path
from datetime import datetime
import sys
import traceback


try:
    import websocket  # websocket-client
except Exception:
    websocket = None

def init_logging(log_dir: str = "logs", app_name: str = "trading_gui", level: int = logging.INFO, max_bytes: int = 10_000_000, backup_count: int = 5):
    """
    Inicializa logging con:
      - fichero logs/log_YYYYMMDD_HHMMSS.txt (rotating handler)
      - salida a consola
      - captura de excepciones no manejadas
    Llamar lo antes posible en el script principal.
    """
    # ensure dir
    ld = Path(log_dir)
    ld.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logfile = ld / f"log_{ts}.txt"

    root = logging.getLogger()
    root.setLevel(level)

    # formatter
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    # rotating file handler
    fh = logging.handlers.RotatingFileHandler(str(logfile), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # uncaught exceptions -> log
    def excepthook(exc_type, exc_value, exc_tb):
        root.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        # also print to stderr
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = excepthook

    # log startup info
    root.info("init_logging: logfile=%s", logfile)
    root.debug("Python executable: %s", sys.executable)
    root.debug("CWD: %s", os.getcwd())
    root.debug("sys.path: %s", sys.path)
    return logfile

# Example of use (call at very top, before heavy imports):
LOGFILE = init_logging(log_dir="logs", app_name="trading_gui", level=logging.INFO)
logging.getLogger(__name__).info("Logging initialized")

import queue
from typing import Optional, List, Dict, Any
import math

import numpy as np
import pandas as pd
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# optional deps
try:
    import torch
except Exception:
    torch = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

# local modules (robust import)
try:
    from trading_daemon import TradingDaemon
except Exception:
    TradingDaemon = None

# Normalize fiboevo import so 'fibo' variable is always present (fix inconsistency)
try:
    import fiboevo as fibo
except Exception:
    try:
        import importlib
        fibo = importlib.import_module("fiboevo")
    except Exception:
        fibo = None

try:
    import utils_clean as utils
except Exception:
    try:
        import utils as utils
    except Exception:
        utils = None

APP_TITLE = "Trading GUI Extended (Prepare · Train · Backtest)"

def safe_now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def detect_suspicious_feature_names(feature_cols: List[str]) -> List[str]:
    import re
    return [c for c in feature_cols if re.match(r"^lr_\d+$", c)]

# Utility: parse timeframe string to seconds (supports m, h, d). Returns seconds or None.
def timeframe_to_seconds(tf: str) -> Optional[int]:
    try:
        tf = tf.strip().lower()
        if tf.endswith("m"):
            return int(float(tf[:-1]) * 60)
        if tf.endswith("h"):
            return int(float(tf[:-1]) * 3600)
        if tf.endswith("d"):
            return int(float(tf[:-1]) * 86400)
        if tf.endswith("s"):
            return int(float(tf[:-1]))
        # fallback: if plain number, assume seconds
        if tf.isdigit():
            return int(tf)
    except Exception:
        pass
    return None


class _QueueLoggingHandler(logging.Handler):
    """
    Logging handler que envía líneas de log formateadas a una queue (thread-safe).
    Mantener este handler es bueno para separar producción de logs y UI.
    """
    def __init__(self, q: "queue.Queue", max_msg_len: int = 2000):
        super().__init__()
        self.q = q
        self.max_msg_len = max_msg_len

    def emit(self, record: logging.LogRecord):
        try:
            # Formatea el registro (usando el Formatter asignado al handler)
            msg = self.format(record)

            # Si hay excepción en el record, añade stacktrace
            if record.exc_info:
                exc_text = "".join(traceback.format_exception(*record.exc_info))
                msg = f"{msg}\n{exc_text}"

            # Acortar si es excesivamente largo (evita llenar la UI)
            if self.max_msg_len and len(msg) > self.max_msg_len:
                msg = msg[:self.max_msg_len] + "...[truncated]"

            # Intentar poner en la cola sin bloquear
            try:
                self.q.put_nowait(msg)
            except queue.Full:
                # Si la cola está llena, descartar el mensaje pero registra una nota local
                # (no usamos logging aquí para evitar loops)
                try:
                    # Mantén una notita mínima para no perder por completo la info
                    self.q.get_nowait()  # descartar el más viejo y reintentar
                    self.q.put_nowait(msg)
                except Exception:
                    # si no se puede, simplemente saltamos el log para no bloquear
                    pass
        except Exception:
            # evitamos que logging falle por completo
            try:
                self.handleError(record)
            except Exception:
                pass


# class GUIHandler(logging.Handler):
#     """Logging handler that writes to a ScrolledText widget in a thread-safe way (via after)."""
#     def __init__(self, text_widget_getter):
#         """
#         text_widget_getter: callable that returns the ScrolledText widget (or None until ready).
#         Use callable so handler can be constructed before UI exists.
#         """
#         super().__init__()
#         self._get_widget = text_widget_getter
#
#     def emit(self, record):
#         try:
#             msg = self.format(record)
#             widget = self._get_widget()
#             if not widget:
#                 return
#             # append on the GUI thread
#             def append():
#                 try:
#                     widget.configure(state="normal")
#                     widget.insert("end", msg + "\n")
#                     widget.see("end")
#                     widget.configure(state="disabled")
#                 except Exception:
#                     pass
#             # Use after to ensure thread-safety
#             try:
#                 widget.after(1, append)
#             except Exception:
#                 # fallback: direct call (if we're on main thread)
#                 append()
#         except Exception:
#             self.handleError(record)

# --------------------------
# Main GUI
# --------------------------
class TradingAppExtended:
    POLL_MS = 250

    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("1200x800")

        # logging widget
        self.ui_log_queue = queue.Queue(maxsize=2000)
        self.ui_log_level = logging.DEBUG
        self.log_widget = None  # se creará con _build_log_pane
        self.log_flush_interval_ms = 200  # cada 200ms volcamos cola

        self.ws_app = None
        self.ws_thread = None
        self.ws_queue = queue.Queue()
        self.ws_connected = False
        self.ws_orderbook = {"bids": {}, "asks": {}}
        self.show_orderbook_var = BooleanVar(value=True)
        self.show_trades_var = BooleanVar(value=True)
        self.ws_verbose_var = IntVar(value=1)  # 0=none,1=info,2=debug
        self.big_trade_threshold_var = DoubleVar(value=10000.0)
        self.trades_tree = None
        self.big_trades_tree = None
        self.orderbook_tree = None

        # Config vars
        self.sqlite_path = StringVar(value="data_manager/exports/marketdata_base.db")
        self.table = StringVar(value="ohlcv")
        self.symbol = StringVar(value="BTCUSDT")
        self.timeframe = StringVar(value="30m")
        self.seq_len = IntVar(value=32)
        self.horizon = IntVar(value=10)
        self.batch_size = IntVar(value=64)
        self.epochs = IntVar(value=10)
        self.hidden = IntVar(value=64)
        self.lr = DoubleVar(value=1e-3)
        self.val_frac = DoubleVar(value=0.1)
        self.dtype_choice = StringVar(value="float32")
        self.feature_cols_manual = StringVar(value="")
        # NEW: model path variable (user can specify a checkpoint file)
        self.model_path_var = StringVar(value="")

        # ------------------------------
        # NEW: Status-related settings
        # ------------------------------
        self.inference_interval_var = DoubleVar(value=5.0)   # seconds between inference attempts (UI control only)
        self.trade_interval_var = DoubleVar(value=30.0)      # seconds between trade execution (UI control only)
        self.refresh_interval_var = DoubleVar(value=5.0)     # seconds for status refresh (UI control)
        self.websocket_url_var = StringVar(value="")         # websocket url (placeholder)
        self.api_key_var = StringVar(value="")               # displayed, not persisted
        self.api_secret_var = StringVar(value="")            # displayed, not persisted
        self.model_symbol_var = StringVar(value="")          # model metadata (symbol)
        self.model_exchange_var = StringVar(value="")        # model metadata (exchange)
        self.model_timeframe_var = StringVar(value="")       # model metadata (timeframe)

        # status display vars (read-only labels)
        self.last_price_var = StringVar(value="N/A")
        self.last_ts_var = StringVar(value="N/A")
        self.pred_log_var = StringVar(value="N/A")
        self.pred_pct_var = StringVar(value="N/A")
        self.pred_vol_var = StringVar(value="N/A")
        self.pred_ts_var = StringVar(value="N/A")
        self.ticker_var = StringVar(value="N/A")
        self.websocket_status_var = StringVar(value="Disconnected")

        # internals
        self.daemon: Optional[TradingDaemon] = None
        self.training_thread: Optional[threading.Thread] = None
        self.prepare_thread: Optional[threading.Thread] = None
        self.combine_thread: Optional[threading.Thread] = None
        self.backtest_thread: Optional[threading.Thread] = None

        # thread-safe logging queue
        self.log_queue: "queue.Queue[str]" = queue.Queue()

        # dataframes for preview
        self.df_loaded: Optional[pd.DataFrame] = None
        self.df_features: Optional[pd.DataFrame] = None
        self.df_scaled: Optional[pd.DataFrame] = None

        # arrays for training
        self.X_full: Optional[np.ndarray] = None
        self.y_full: Optional[np.ndarray] = None
        self.feature_cols_used: Optional[List[str]] = None
        self.scaler_used: Optional[Any] = None

        # model artifacts in memory
        self.model = None
        self.model_meta = None
        self.model_scaler = None

        # config path
        self.config_path = Path("config/gui_config.json")

        # build UI
        self._build_ui()
        self._init_ui_logging(self.root)    # crea widget y pone handler
        self._schedule_log_flush()          # arranca loop de vaciado centralizado (una sola vez)
        # si _init_ui_logging crea self.log_widget dentro de un frame, podemos añadir un clear button:
        btn_clear = Button(self.log_widget.master, text="Clear logs", command=self._clear_logs)
        btn_clear.pack(side=LEFT, padx=6)
        # load saved config if any
        self._load_config_on_start()

        # schedule log flush
        self._schedule_log_flush()

    # --------------------------
    # UI building
    # --------------------------
    # --- Websocket connect / callbacks ---

    def _init_ui_logging(self, root_parent):
        """
        Inicializa el panel de logs (metodo de instancia).
        Crea self.log_widget y añade un _QueueLoggingHandler (si no existe).
        NO programará su propio after loop aparte; el vaciado lo hace _schedule_log_flush centralizado.
        """
        import queue as _qmod
        if getattr(self, "log_widget", None):
            return

        # Controls frame + widget
        frm = Frame(root_parent)
        frm.pack(fill=BOTH, side=BOTTOM, padx=6, pady=4)

        ctrl_row = Frame(frm)
        ctrl_row.pack(fill=X)
        Label(ctrl_row, text="Logs:").pack(side=LEFT)

        # autoscroll
        self._log_autoscroll = BooleanVar(value=True)
        Checkbutton(ctrl_row, text="Autoscroll", variable=self._log_autoscroll).pack(side=LEFT, padx=4)
        # clear / save / open external window buttons
        Button(ctrl_row, text="Clear", command=self._clear_logs).pack(side=LEFT, padx=4)
        Button(ctrl_row, text="Save", command=self._save_logs_to_file).pack(side=LEFT, padx=4)
        Button(ctrl_row, text="Open window", command=self._open_logs_window).pack(side=LEFT, padx=4)

        # text widget
        self.log_widget = ScrolledText(frm, height=8, state="disabled", wrap="none")
        self.log_widget.pack(fill=BOTH, expand=True)

        # Add Queue handler to root logger only once
        root_logger = logging.getLogger()
        existing_handlers = [h for h in root_logger.handlers if isinstance(h, _QueueLoggingHandler)]
        # If existing handler uses same queue, reuse; otherwise create one and attach
        reuse = False
        for h in existing_handlers:
            try:
                if getattr(h, "q", None) is getattr(self, "ui_log_queue", None):
                    self._ui_log_handler = h
                    reuse = True
                    break
            except Exception:
                continue
        if not reuse:
            self._ui_log_handler = _QueueLoggingHandler(getattr(self, "ui_log_queue", None))
            self._ui_log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                                               "%Y-%m-%d %H:%M:%S"))
            self._ui_log_handler.setLevel(getattr(self, "ui_log_level", logging.DEBUG))
            root_logger.addHandler(self._ui_log_handler)
    # helper methods recommended to add in the class:

    def _clear_logs(self):
        if getattr(self, "log_widget", None):
            try:
                self.log_widget.configure(state="normal")
                self.log_widget.delete("1.0", "end")
                self.log_widget.configure(state="disabled")
            except Exception:
                pass

    def _save_logs_to_file(self):
        if not getattr(self, "log_widget", None):
            return
        try:
            from tkinter.filedialog import asksaveasfilename
            fname = asksaveasfilename(defaultextension=".txt",
                                      filetypes=[("Text files","*.txt"),("All files","*.*")])
            if not fname:
                return
            content = self.log_widget.get("1.0", "end-1c")
            with open(fname, "w", encoding="utf-8") as fh:
                fh.write(content)
        except Exception:
            logging.getLogger(__name__).exception("Failed saving logs")

    def _open_logs_window(self):
        """Opens snapshot window with current log content (no live mirroring for simplicity)."""
        if not getattr(self, "log_widget", None):
            return
        try:
            top = Toplevel(self.root)
            top.title("Logs")
            top.minsize(400, 200)
            sw = ScrolledText(top, height=20, width=100, state="normal", wrap="none")
            sw.pack(fill=BOTH, expand=True)
            sw.insert("end", self.log_widget.get("1.0", "end-1c"))
            sw.configure(state="disabled")
        except Exception:
            logging.getLogger(__name__).exception("Failed opening logs window")

    def _teardown_ui_logging(self):
        """
        Remove handler from root logger to avoid duplicates when reloading UI or exiting.
        """
        import logging
        try:
            root_logger = logging.getLogger()
            if getattr(self, "_ui_log_handler", None):
                try:
                    root_logger.removeHandler(self._ui_log_handler)
                except Exception:
                    pass
                self._ui_log_handler = None
        except Exception:
            logging.getLogger(__name__).exception("Error during teardown UI logging")


    def _disconnect_websocket(self):
        if getattr(self, "ws_app", None):
            try:
                self.ws_app.close()
            except Exception:
                pass
        self.ws_connected = False
        self.websocket_status_var.set("Disconnected")
        self._enqueue_log("Websocket closed by user.")

    # --- Queue processor: parse and update UI ---
    def _process_ws_queue(self):
        import queue as _queue  # import local para crear cola si es necesario
        q = getattr(self, "ws_queue", None)
        if q is None:
            # nothing to process; reschedule and return
            try:
                self.nb.after(150, self._process_ws_queue)
                return
            except Exception:
                try:
                    self.after(150, self._process_ws_queue)
                    return
                except Exception:
                    return

        # validar que q actúe como queue
        if not (hasattr(q, "empty") and callable(getattr(q, "empty")) and hasattr(q, "get_nowait") and callable(getattr(q, "get_nowait"))):
            # posible sobrescritura accidental; log y recrear cola para recuperar
            try:
                self._enqueue_log(f"Warning: ws_queue is not queue (type={type(q).__name__}). Recreating internal queue.")
            except Exception:
                pass
            try:
                self.ws_queue = _queue.Queue()
                q = self.ws_queue
            except Exception as e:
                try:
                    self._enqueue_log(f"Could not recreate ws_queue: {e}")
                except Exception:
                    pass
                # reprogramar y salir
                try:
                    self.nb.after(150, self._process_ws_queue)
                    return
                except Exception:
                    try:
                        self.after(150, self._process_ws_queue)
                        return
                    except Exception:
                        return

        processed = 0
        MAX_PER_CYCLE = 200

        while processed < MAX_PER_CYCLE:
            try:
                if q.empty():
                    break
            except Exception:
                # si empty falla por algun motivo, intentar get_nowait con try/except
                pass

            try:
                evt = q.get_nowait()
            except Exception:
                break
            processed += 1

            # asegurarnos de que evt sea dict-likish
            if not isinstance(evt, dict):
                # log no-dict item y seguir
                try:
                    self._enqueue_log(f"WS queue received non-dict item (type={type(evt).__name__}): {repr(evt)[:200]}")
                except Exception:
                    pass
                continue

            # meta events
            try:
                if evt.get("type") == "_meta":
                    if evt.get("event") == "open":
                        self.ws_connected = True
                        try:
                            self.websocket_status_var.set("Connected")
                        except Exception:
                            pass
                    elif evt.get("event") == "close":
                        self.ws_connected = False
                        try:
                            self.websocket_status_var.set("Disconnected")
                        except Exception:
                            pass
                    elif evt.get("event") == "error":
                        self._enqueue_log("WS error: " + evt.get("error", ""))
                    continue
            except Exception:
                # proteger de metadatos mal formados
                try:
                    self._enqueue_log("Malformed _meta event in ws queue.")
                except Exception:
                    pass
                continue

            # message events
            try:
                if evt.get("type") == "message":
                    raw = evt.get("raw")
                    data = None
                    if isinstance(raw, (str, bytes)):
                        try:
                            # si bytes -> decodificar
                            if isinstance(raw, bytes):
                                raw_dec = raw.decode("utf-8", errors="ignore")
                            else:
                                raw_dec = raw
                            data = json.loads(raw_dec)
                        except Exception:
                            data = None
                    else:
                        # raw could be already parsed object
                        if isinstance(raw, dict):
                            data = raw
                        else:
                            # fallback: maybe evt had 'json' field pre-parsed
                            data = evt.get("json")

                    # Verbose logging
                    try:
                        v = int(self.ws_verbose_var.get())
                    except Exception:
                        v = 0
                    if v >= 2:
                        try:
                            self._enqueue_log(f"RAW WS MSG: {str(raw)[:1000]}")
                        except Exception:
                            pass

                    parsed = None
                    if isinstance(data, dict):
                        # Binance-style trade
                        try:
                            if data.get("e") in ("trade", "aggTrade"):
                                parsed = {
                                    "kind": "trade",
                                    "ts": data.get("E") or int(time.time() * 1000),
                                    "symbol": data.get("s"),
                                    "price": float(data.get("p") or data.get("price") or 0),
                                    "qty": float(data.get("q") or data.get("qty") or 0),
                                    "side": ("sell" if data.get("m") else "buy") if "m" in data else data.get("side", "unknown"),
                                }
                            elif data.get("e") in ("depthUpdate", "depth"):
                                parsed = {"kind": "orderbook_update", "bids": data.get("b", []), "asks": data.get("a", [])}
                            elif "bids" in data and "asks" in data:
                                parsed = {"kind": "orderbook_update", "bids": data["bids"], "asks": data["asks"]}
                            elif data.get("type") in ("l2update", "snapshot", "update", "orderBook"):
                                parsed = {"kind": "orderbook_generic", "data": data}
                            elif data.get("e") in ("24hrTicker", "ticker") or data.get("type") == "ticker":
                                parsed = {"kind": "ticker", "price": float(data.get("c") or data.get("price") or 0)}
                            else:
                                # fallback: detect keys price/qty
                                keys = set(data.keys())
                                if {"price", "qty"} <= keys or {"p", "q"} <= keys:
                                    try:
                                        price = float(data.get("price") or data.get("p") or 0)
                                        qty = float(data.get("qty") or data.get("q") or 0)
                                        parsed = {"kind": "trade", "ts": int(time.time() * 1000), "price": price, "qty": qty, "side": data.get("side", "unknown")}
                                    except Exception:
                                        parsed = None
                        except Exception:
                            parsed = None
                    else:
                        parsed = None

                    # Process parsed payload
                    if parsed:
                        kind = parsed.get("kind")
                        if kind == "trade":
                            try:
                                price = parsed.get("price")
                                ts = parsed.get("ts")
                                qty = parsed.get("qty", 0)
                                side = parsed.get("side", "unknown")
                                # Update UI vars
                                try:
                                    self.last_price_var.set(str(price))
                                except Exception:
                                    pass
                                try:
                                    self.last_ts_var.set(str(ts))
                                except Exception:
                                    pass
                                if parsed.get("symbol"):
                                    try:
                                        self.ticker_var.set(parsed.get("symbol"))
                                    except Exception:
                                        pass

                                # Insert to trades_tree (keep only last N rows)
                                if getattr(self, "trades_tree", None) and getattr(self, "show_trades_var", None) and self.show_trades_var.get():
                                    try:
                                        tstr = time.strftime("%H:%M:%S", time.localtime(ts / 1000)) if isinstance(ts, (int, float)) else str(ts)
                                        self.trades_tree.insert("", 0, values=(tstr, side, f"{price:.8g}", f"{qty:.8g}"))
                                        # trim to 200 rows
                                        children = self.trades_tree.get_children()
                                        if len(children) > 200:
                                            for i in children[200:]:
                                                try:
                                                    self.trades_tree.delete(i)
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass

                                # Big trades
                                try:
                                    big_threshold = float(self.big_trade_threshold_var.get())
                                except Exception:
                                    big_threshold = None
                                try:
                                    if big_threshold is not None and qty >= big_threshold:
                                        if getattr(self, "big_trades_tree", None):
                                            tstr = time.strftime("%H:%M:%S", time.localtime(ts / 1000)) if isinstance(ts, (int, float)) else str(ts)
                                            self.big_trades_tree.insert("", 0, values=(tstr, side, f"{price:.8g}", f"{qty:.8g}"))
                                            # trim
                                            children_b = self.big_trades_tree.get_children()
                                            if len(children_b) > 100:
                                                for i in children_b[100:]:
                                                    try:
                                                        self.big_trades_tree.delete(i)
                                                    except Exception:
                                                        pass
                                            if getattr(self, "ws_verbose_var", None) and self.ws_verbose_var.get() >= 1:
                                                self._enqueue_log(f"Big trade {side} {qty}@{price}")
                                except Exception:
                                    pass
                            except Exception:
                                # proteger para que trade mal formado no rompa loop
                                try:
                                    self._enqueue_log("Malformed trade event in ws queue.")
                                except Exception:
                                    pass

                        elif kind in ("orderbook_update", "orderbook_generic"):
                            try:
                                bids = parsed.get("bids", [])
                                asks = parsed.get("asks", [])
                                # ensure orderbook dicts exist
                                if not hasattr(self, "ws_orderbook") or not isinstance(self.ws_orderbook, dict):
                                    self.ws_orderbook = {"bids": {}, "asks": {}}
                                # apply bid updates
                                for entry in bids:
                                    try:
                                        pstr, qstr = entry[0], entry[1]
                                        price = float(pstr)
                                        qf = float(qstr)
                                        if qf == 0:
                                            if price in self.ws_orderbook["bids"]:
                                                del self.ws_orderbook["bids"][price]
                                        else:
                                            self.ws_orderbook["bids"][price] = qf
                                    except Exception:
                                        continue
                                # apply ask updates
                                for entry in asks:
                                    try:
                                        pstr, qstr = entry[0], entry[1]
                                        price = float(pstr)
                                        qf = float(qstr)
                                        if qf == 0:
                                            if price in self.ws_orderbook["asks"]:
                                                del self.ws_orderbook["asks"][price]
                                        else:
                                            self.ws_orderbook["asks"][price] = qf
                                    except Exception:
                                        continue

                                # Rebuild orderbook tree display (top 20 each)
                                if getattr(self, "orderbook_tree", None) and getattr(self, "show_orderbook_var", None) and self.show_orderbook_var.get():
                                    try:
                                        # clear
                                        self.orderbook_tree.delete(*self.orderbook_tree.get_children())
                                    except Exception:
                                        pass

                                    try:
                                        # Asks: sorted ascending (best asks first)
                                        top_asks = sorted(self.ws_orderbook["asks"].items(), key=lambda x: x[0])[:20]
                                        for price, qty in top_asks:
                                            try:
                                                self.orderbook_tree.insert("", "end", values=("ASK", f"{price:.8g}", f"{qty:.8g}"))
                                            except Exception:
                                                pass
                                        # Bids: sorted descending (best bids first)
                                        top_bids = sorted(self.ws_orderbook["bids"].items(), key=lambda x: -x[0])[:20]
                                        for price, qty in top_bids:
                                            try:
                                                self.orderbook_tree.insert("", "end", values=("BID", f"{price:.8g}", f"{qty:.8g}"))
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass
                            except Exception:
                                try:
                                    self._enqueue_log("Malformed orderbook event in ws queue.")
                                except Exception:
                                    pass

                        elif kind == "ticker":
                            try:
                                p = parsed.get("price")
                                self.last_price_var.set(str(p))
                            except Exception:
                                pass

                    # end parsed handling
            except Exception:
                # proteger procesamiento de cada item
                try:
                    self._enqueue_log("Exception while processing ws queue item.")
                except Exception:
                    pass
                continue  # seguir con siguiente item

        # reschedule
        try:
            self.nb.after(150, self._process_ws_queue)
        except Exception:
            try:
                self.after(150, self._process_ws_queue)
            except Exception:
                # no more scheduling possible
                pass

    # --- Optional: on app exit ensure websocket closed ---
    def _cleanup_ws_on_exit(self):
        try:
            if getattr(self, "ws_app", None):
                try:
                    self.ws_app.close()
                except Exception:
                    pass
        except Exception:
            pass

    # --- UI: función para extender _build_status_tab (llamar desde _build_status_tab después del 'Bottom: tree showing last data rows') ---
    def _extend_status_tab_ui(self, parent_tab):
        # Controls row: connect/disconnect, verbose, threshold
        frm_ws_ctrl = Frame(parent_tab)
        frm_ws_ctrl.pack(fill=X, padx=6, pady=2)

        Label(frm_ws_ctrl, text="Websocket URL:").pack(side=LEFT, padx=(0,4))
        Entry(frm_ws_ctrl, textvariable=self.websocket_url_var, width=50).pack(side=LEFT, padx=(0,6))
        Button(frm_ws_ctrl, text="Connect", command=self._connect_websocket).pack(side=LEFT, padx=4)
        Button(frm_ws_ctrl, text="Disconnect", command=self._disconnect_websocket).pack(side=LEFT, padx=4)

        Label(frm_ws_ctrl, text="Verbose:").pack(side=LEFT, padx=(12,4))
        OptionMenu(frm_ws_ctrl, self.ws_verbose_var, 0,1,2).pack(side=LEFT)
        Label(frm_ws_ctrl, text="Big trade threshold:").pack(side=LEFT, padx=(12,4))
        Entry(frm_ws_ctrl, textvariable=self.big_trade_threshold_var, width=8).pack(side=LEFT)

        Checkbutton(frm_ws_ctrl, text="Show Orderbook", variable=self.show_orderbook_var).pack(side=LEFT, padx=8)
        Checkbutton(frm_ws_ctrl, text="Show Trades", variable=self.show_trades_var).pack(side=LEFT, padx=4)

        # Frames for orderbook and trades
        frm_ws_display = Frame(parent_tab)
        frm_ws_display.pack(fill=BOTH, expand=True, padx=6, pady=6)

        # Left: trades
        frm_trades = Frame(frm_ws_display, relief=RIDGE, bd=1)
        frm_trades.pack(side=LEFT, fill=BOTH, expand=True, padx=4, pady=4)
        Label(frm_trades, text="Recent Trades").pack(anchor="w")
        self.trades_tree = ttk.Treeview(frm_trades, columns=("ts","side","price","qty"), show="headings", height=8)
        for c, t in [("ts","TS"),("side","Side"),("price","Price"),("qty","Qty")]:
            self.trades_tree.heading(c, text=t)
            self.trades_tree.column(c, width=80, anchor=W)
        self.trades_tree.pack(fill=BOTH, expand=True)

        Label(frm_trades, text="Big Trades").pack(anchor="w", pady=(6,0))
        self.big_trades_tree = ttk.Treeview(frm_trades, columns=("ts","side","price","qty"), show="headings", height=4)
        for c, t in [("ts","TS"),("side","Side"),("price","Price"),("qty","Qty")]:
            self.big_trades_tree.heading(c, text=t)
            self.big_trades_tree.column(c, width=80, anchor=W)
        self.big_trades_tree.pack(fill=X, expand=False)

        # Right: orderbook
        frm_ob = Frame(frm_ws_display, relief=RIDGE, bd=1)
        frm_ob.pack(side=LEFT, fill=BOTH, expand=True, padx=4, pady=4)
        Label(frm_ob, text="Orderbook (top bids/asks)").pack(anchor="w")
        self.orderbook_tree = ttk.Treeview(frm_ob, columns=("side","price","qty"), show="headings", height=18)
        for c, t in [("side","Side"),("price","Price"),("qty","Qty")]:
            self.orderbook_tree.heading(c, text=t)
            self.orderbook_tree.column(c, width=100, anchor=W)
        self.orderbook_tree.pack(fill=BOTH, expand=True)

    def _build_ui(self):
        top = Frame(self.root)
        top.pack(side=TOP, fill=X, padx=6, pady=6)

        Label(top, text="SQLite:").pack(side=LEFT)
        Entry(top, textvariable=self.sqlite_path, width=40).pack(side=LEFT, padx=4)
        Button(top, text="Browse", command=self._browse_sqlite).pack(side=LEFT, padx=4)
        Label(top, text="Table:").pack(side=LEFT, padx=(8,0))
        Entry(top, textvariable=self.table, width=10).pack(side=LEFT, padx=4)
        Label(top, text="Symbol:").pack(side=LEFT, padx=(8,0))
        Entry(top, textvariable=self.symbol, width=12).pack(side=LEFT, padx=4)
        Label(top, text="TF:").pack(side=LEFT, padx=(8,0))
        Entry(top, textvariable=self.timeframe, width=6).pack(side=LEFT, padx=4)

        # NEW: model path entry + browse
        Label(top, text="Model:").pack(side=LEFT, padx=(8,0))
        Entry(top, textvariable=self.model_path_var, width=30).pack(side=LEFT, padx=(4,0))
        Button(top, text="Browse Model", command=self._browse_model).pack(side=LEFT, padx=4)

        frame_ctl = Frame(self.root)
        frame_ctl.pack(side=TOP, fill=X, padx=6, pady=6)
        self.btn_start = Button(frame_ctl, text="Start Daemon", bg="#4CAF50", fg="white", command=self._start_daemon)
        self.btn_start.pack(side=LEFT, padx=6)
        self.btn_stop = Button(frame_ctl, text="Stop Daemon", bg="#f44336", fg="white", command=self._stop_daemon, state=DISABLED)
        self.btn_stop.pack(side=LEFT, padx=6)

        # Notebook
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(side=TOP, fill=BOTH, expand=True, padx=6, pady=6)


        # Tabs
        self._build_preview_tab()
        self._build_train_tab()
        self._build_backtest_tab()
        self._build_status_tab()

        # Logs area se puso en _init_ui_logging
        # frame_logs = Frame(self.root)
        # frame_logs.pack(side=BOTTOM, fill=BOTH, padx=6, pady=6)
        # Label(frame_logs, text="Logs").pack(anchor="w")
        # self.txt_logs = Text(frame_logs, height=8)
        # self.txt_logs.pack(fill=BOTH, expand=True)
        # Button(frame_logs, text="Clear logs", command=self._clear_logs).pack(side=LEFT, padx=6)

    def _build_preview_tab(self):
        tab = Frame(self.nb); self.nb.add(tab, text="Preview")
        top = Frame(tab); top.pack(fill=X, padx=6, pady=6)
        Label(top, text="Preview latest rows from SQLite:").pack(side=LEFT)
        Button(top, text="Refresh Preview", command=self._refresh_preview).pack(side=LEFT, padx=6)
        Button(top, text="Preview loaded (df_loaded)", command=lambda: self._show_preview("loaded")).pack(side=LEFT, padx=6)
        Button(top, text="Preview features", command=lambda: self._show_preview("features")).pack(side=LEFT, padx=6)
        Button(top, text="Preview scaled", command=lambda: self._show_preview("scaled")).pack(side=LEFT, padx=6)

        self.preview_tree = ttk.Treeview(tab)
        self.preview_tree.pack(fill=BOTH, expand=True, padx=6, pady=6)

    def _build_train_tab(self):
        tab = Frame(self.nb); self.nb.add(tab, text="Training")
        left = Frame(tab); left.pack(side=LEFT, fill=Y, padx=6, pady=6)
        Label(left, text="Training config", font=("Arial",11,"bold")).pack(anchor="w")
        self._add_labeled_entry(left, "seq_len", self.seq_len)
        self._add_labeled_entry(left, "horizon", self.horizon)
        self._add_labeled_entry(left, "hidden", self.hidden)
        self._add_labeled_entry(left, "epochs", self.epochs)
        self._add_labeled_entry(left, "batch_size", self.batch_size)
        self._add_labeled_entry(left, "learning rate", self.lr)
        self._add_labeled_entry(left, "val fraction", self.val_frac)
        Label(left, text="dtype:").pack(anchor="w", pady=(6,0))
        OptionMenu(left, self.dtype_choice, "float32", "float64").pack(anchor="w")
        Label(left, text="feature_cols (comma optional)").pack(anchor="w", pady=(6,0))
        Entry(left, textvariable=self.feature_cols_manual, width=30).pack(anchor="w")
        Button(left, text="Save Config", command=self._save_config).pack(anchor="w", pady=(6,2))
        Button(left, text="Prepare Data (background)", command=self._start_prepare).pack(anchor="w", pady=(2,2))
        Button(left, text="Train Model (background)", command=self._start_train_model).pack(anchor="w", pady=(2,2))
        Button(left, text="Prepare + Train (background)", command=self._start_prepare_and_train).pack(anchor="w", pady=(2,2))
        Button(left, text="Load artifacts model", command=self._load_model_from_artifacts).pack(anchor="w", pady=(6,2))
        # NEW: load model file (background) - will call daemon.load_model_and_scaler if daemon exists
        Button(left, text="Load model file (background)", command=self._load_model_file_background).pack(anchor="w", pady=(2,2))

        right = Frame(tab); right.pack(side=LEFT, fill=BOTH, expand=True, padx=6, pady=6)
        Label(right, text="Prepare / Train log & previews", font=("Arial",11,"bold")).pack(anchor="w")
        self.train_info_text = Text(right)
        self.train_info_text.pack(fill=BOTH, expand=True)

    def _build_backtest_tab(self):
        tab = Frame(self.nb); self.nb.add(tab, text="Backtest")
        top = Frame(tab); top.pack(fill=X, padx=6, pady=6)
        Label(top, text="enter_th").pack(side=LEFT)
        Entry(top, textvariable=DoubleVar(value=0.0005), width=8).pack(side=LEFT, padx=4)  # placeholder local
        Button(top, text="Run simple backtest", command=self._start_backtest).pack(side=LEFT, padx=6)
        self.bt_text = Text(tab, height=20)
        self.bt_text.pack(fill=BOTH, expand=True, padx=6, pady=6)

    def _build_status_tab(self):
        tab = Frame(self.nb); self.nb.add(tab, text="Status")
        Label(tab, text="Daemon status & ledger").pack(anchor="w", padx=6, pady=4)

        # Top: settings form (intervals, websocket, api keys, model meta)
        frm_settings = Frame(tab)
        frm_settings.pack(fill=X, padx=6, pady=4)

        # Intervals
        Label(frm_settings, text="Inference interval (s):").grid(row=0, column=0, sticky=W, padx=2, pady=2)
        Entry(frm_settings, textvariable=self.inference_interval_var, width=8).grid(row=0, column=1, sticky=W, padx=2, pady=2)
        Label(frm_settings, text="Trade interval (s):").grid(row=0, column=2, sticky=W, padx=8, pady=2)
        Entry(frm_settings, textvariable=self.trade_interval_var, width=8).grid(row=0, column=3, sticky=W, padx=2, pady=2)
        Label(frm_settings, text="Refresh interval (s):").grid(row=0, column=4, sticky=W, padx=8, pady=2)
        Entry(frm_settings, textvariable=self.refresh_interval_var, width=8).grid(row=0, column=5, sticky=W, padx=2, pady=2)
        Button(frm_settings, text="Apply Intervals", command=self._apply_status_settings).grid(row=0, column=6, sticky=W, padx=8, pady=2)

        # Websocket + API keys (placeholders; not persisted)
        Label(frm_settings, text="Websocket URL:").grid(row=1, column=0, sticky=W, padx=2, pady=2)
        Entry(frm_settings, textvariable=self.websocket_url_var, width=40).grid(row=1, column=1, columnspan=3, sticky=W, padx=2, pady=2)
        Button(frm_settings, text="Connect Websocket", command=self._connect_websocket).grid(row=1, column=4, sticky=W, padx=8, pady=2)
        Button(frm_settings, text="Disconnect", command=self._disconnect_websocket).grid(row=1, column=5, sticky=W, padx=2, pady=2)
        Label(frm_settings, textvariable=self.websocket_status_var).grid(row=1, column=6, sticky=W, padx=8, pady=2)

        Label(frm_settings, text="API Key:").grid(row=2, column=0, sticky=W, padx=2, pady=2)
        Entry(frm_settings, textvariable=self.api_key_var, width=30, show="*").grid(row=2, column=1, sticky=W, padx=2, pady=2)
        Label(frm_settings, text="API Secret:").grid(row=2, column=2, sticky=W, padx=2, pady=2)
        Entry(frm_settings, textvariable=self.api_secret_var, width=30, show="*").grid(row=2, column=3, sticky=W, padx=2, pady=2)

        # Model metadata display (symbol/exchange/timeframe)
        Label(frm_settings, text="Model symbol:").grid(row=3, column=0, sticky=W, padx=2, pady=2)
        Entry(frm_settings, textvariable=self.model_symbol_var, width=12).grid(row=3, column=1, sticky=W, padx=2, pady=2)
        Label(frm_settings, text="Model exchange:").grid(row=3, column=2, sticky=W, padx=2, pady=2)
        Entry(frm_settings, textvariable=self.model_exchange_var, width=12).grid(row=3, column=3, sticky=W, padx=2, pady=2)
        Label(frm_settings, text="Model timeframe:").grid(row=3, column=4, sticky=W, padx=2, pady=2)
        Entry(frm_settings, textvariable=self.model_timeframe_var, width=8).grid(row=3, column=5, sticky=W, padx=2, pady=2)

        # Middle: status controls
        frm_controls = Frame(tab)
        frm_controls.pack(fill=X, padx=6, pady=4)
        Button(frm_controls, text="Refresh Status", command=self._refresh_status).pack(side=LEFT, padx=6)
        Button(frm_controls, text="Fetch Last Data", command=self._fetch_last_rows_status).pack(side=LEFT, padx=6)
        Button(frm_controls, text="Get Latest Prediction", command=self._get_latest_prediction_thread).pack(side=LEFT, padx=6)

        # Right side: small summary card for latest data / prediction
        frm_summary = Frame(tab, relief=RIDGE, bd=1)
        frm_summary.pack(fill=X, padx=6, pady=6)

        Label(frm_summary, text="Ticker:").grid(row=0, column=0, sticky=W, padx=4, pady=2)
        Label(frm_summary, textvariable=self.ticker_var).grid(row=0, column=1, sticky=W, padx=4, pady=2)
        Label(frm_summary, text="Last price:").grid(row=1, column=0, sticky=W, padx=4, pady=2)
        Label(frm_summary, textvariable=self.last_price_var).grid(row=1, column=1, sticky=W, padx=4, pady=2)
        Label(frm_summary, text="Last ts:").grid(row=2, column=0, sticky=W, padx=4, pady=2)
        Label(frm_summary, textvariable=self.last_ts_var).grid(row=2, column=1, sticky=W, padx=4, pady=2)

        Label(frm_summary, text="Prediction (log):").grid(row=0, column=2, sticky=W, padx=12, pady=2)
        Label(frm_summary, textvariable=self.pred_log_var).grid(row=0, column=3, sticky=W, padx=4, pady=2)
        Label(frm_summary, text="Prediction (%):").grid(row=1, column=2, sticky=W, padx=12, pady=2)
        Label(frm_summary, textvariable=self.pred_pct_var).grid(row=1, column=3, sticky=W, padx=4, pady=2)
        Label(frm_summary, text="Volatility:").grid(row=2, column=2, sticky=W, padx=12, pady=2)
        Label(frm_summary, textvariable=self.pred_vol_var).grid(row=2, column=3, sticky=W, padx=4, pady=2)
        Label(frm_summary, text="Pred timestamp:").grid(row=3, column=2, sticky=W, padx=12, pady=2)
        Label(frm_summary, textvariable=self.pred_ts_var).grid(row=3, column=3, sticky=W, padx=4, pady=2)

        # Extra websocket controls & visual options
        frm_ws_ctrl = Frame(tab)
        frm_ws_ctrl.pack(fill=X, padx=6, pady=(2,6))

        Label(frm_ws_ctrl, text="Verbose:").pack(side=LEFT, padx=(0,4))
        OptionMenu(frm_ws_ctrl, self.ws_verbose_var, 0,1,2).pack(side=LEFT)
        Label(frm_ws_ctrl, text="Big trade threshold:").pack(side=LEFT, padx=(12,4))
        Entry(frm_ws_ctrl, textvariable=self.big_trade_threshold_var, width=10).pack(side=LEFT)
        Checkbutton(frm_ws_ctrl, text="Show Orderbook", variable=self.show_orderbook_var).pack(side=LEFT, padx=8)
        Checkbutton(frm_ws_ctrl, text="Show Trades", variable=self.show_trades_var).pack(side=LEFT, padx=4)

        # Bottom: tree showing last data rows (existing)
        self.status_data_tree = ttk.Treeview(tab)
        self.status_data_tree.pack(fill=BOTH, expand=False, padx=6, pady=6, ipady=20)

        # Websocket displays: trades / big trades / orderbook
        frm_ws_display = Frame(tab)
        frm_ws_display.pack(fill=BOTH, expand=True, padx=6, pady=4)

        # Left: trades + big trades
        frm_trades = Frame(frm_ws_display, relief=RIDGE, bd=1)
        frm_trades.pack(side=LEFT, fill=BOTH, expand=True, padx=4, pady=4)
        Label(frm_trades, text="Recent Trades").pack(anchor="w")
        self.trades_tree = ttk.Treeview(frm_trades, columns=("ts","side","price","qty"), show="headings", height=8)
        for c, t in [("ts","TS"),("side","Side"),("price","Price"),("qty","Qty")]:
            self.trades_tree.heading(c, text=t)
            self.trades_tree.column(c, width=80, anchor=W)
        self.trades_tree.pack(fill=BOTH, expand=True)

        Label(frm_trades, text="Big Trades").pack(anchor="w", pady=(6,0))
        self.big_trades_tree = ttk.Treeview(frm_trades, columns=("ts","side","price","qty"), show="headings", height=4)
        for c, t in [("ts","TS"),("side","Side"),("price","Price"),("qty","Qty")]:
            self.big_trades_tree.heading(c, text=t)
            self.big_trades_tree.column(c, width=80, anchor=W)
        self.big_trades_tree.pack(fill=X, expand=False)

        # Right: orderbook
        frm_ob = Frame(frm_ws_display, relief=RIDGE, bd=1)
        frm_ob.pack(side=LEFT, fill=BOTH, expand=True, padx=4, pady=4)
        Label(frm_ob, text="Orderbook (top bids/asks)").pack(anchor="w")
        self.orderbook_tree = ttk.Treeview(frm_ob, columns=("side","price","qty"), show="headings", height=18)
        for c, t in [("side","Side"),("price","Price"),("qty","Qty")]:
            self.orderbook_tree.heading(c, text=t)
            self.orderbook_tree.column(c, width=100, anchor=W)
        self.orderbook_tree.pack(fill=BOTH, expand=True)

        # ensure queue processing is active (safe to call multiple times)
        try:
            self.nb.after(150, self._process_ws_queue)
        except Exception:
            # fallback to instance root if needed
            try:
                self.after(150, self._process_ws_queue)
            except Exception:
                pass


    def _add_labeled_entry(self, parent, label, var):
        Label(parent, text=label).pack(anchor="w", pady=(6,0))
        Entry(parent, textvariable=var).pack(anchor="w", pady=(0,6))

    # --------------------------
    # Logging queue flush & helpers
    # --------------------------

    def _setup_logging_internals(self):
        """Llamar en __init__ antes de build_ui o justo después: crea la cola y parámetros por defecto."""
        import queue as _qmod
        self.ui_log_queue: "queue.Queue[str]" = getattr(self, "ui_log_queue", None) or _qmod.Queue(maxsize=2000)
        # Legacy queue (si código antiguo usa .log_queue)
        self.log_queue = getattr(self, "log_queue", None) or _qmod.Queue(maxsize=2000)
        self.log_flush_interval_ms = getattr(self, "log_flush_interval_ms", 200)
        self._log_max_lines = getattr(self, "_log_max_lines", 2000)
        self._log_autoscroll = getattr(self, "_log_autoscroll", None)  # si lo crea _init_ui_logging será BooleanVar
        # Keep a reference to the handler we attach (to avoid duplicates)
        self._ui_log_handler = None


    def _enqueue_log(self, msg: str, level=logging.INFO):
        """Unifica envío de logs: cola UI, legacy queue y logging.getLogger()."""
        ts = safe_now_str()
        line = f"[{ts}] {msg}"
        # 1) legacy queue (no bloquear)
        try:
            if getattr(self, "log_queue", None) is not None:
                try:
                    self.log_queue.put_nowait(line)
                except Exception:
                    try:
                        self.log_queue.put(line, block=False)
                    except Exception:
                        pass
        except Exception:
            pass

        # 2) ui queue
        try:
            if getattr(self, "ui_log_queue", None) is not None:
                try:
                    self.ui_log_queue.put_nowait(line)
                except Exception:
                    try:
                        self.ui_log_queue.put(line, block=False)
                    except Exception:
                        pass
        except Exception:
            pass

        # 3) emit to python logging so module loggers also flow to file/console
        try:
            logging.getLogger().log(level, msg)
        except Exception:
            try:
                print(line)
            except Exception:
                pass

    def _schedule_log_flush(self):
        """
        Bucle de vaciado centralizado (debe llamarse UNA vez desde __init__ después de construir la UI).
        Procesa un máximo N mensajes por ciclo y reprograma con after para evitar congelación.
        """
        import time
        cap_per_cycle = 250  # limitar por ciclo (ajusta si quieres más/menos)
        t_start = time.monotonic()
        processed = 0

        # Prefer: ui_log_queue -> log_widget
        q = getattr(self, "ui_log_queue", None)
        widget = getattr(self, "log_widget", None)
        if q is not None and widget is not None:
            try:
                while processed < cap_per_cycle:
                    try:
                        msg = q.get_nowait()
                    except Exception:
                        break
                    processed += 1
                    try:
                        widget.configure(state="normal")
                        widget.insert("end", msg + "\n")
                        # rotate if too many lines
                        try:
                            nlines = int(widget.index("end-1c").split(".")[0])
                            if nlines > self._log_max_lines:
                                delete_to = f"{nlines - self._log_max_lines + 1}.0"
                                widget.delete("1.0", delete_to)
                        except Exception:
                            pass
                        if getattr(self, "_log_autoscroll", None) and self._log_autoscroll.get():
                            widget.see("end")
                        widget.configure(state="disabled")
                    except Exception:
                        try:
                            widget.configure(state="disabled")
                        except Exception:
                            pass
            finally:
                # Reprogramar (no bloqueante)
                try:
                    self.root.after(self.log_flush_interval_ms, self._schedule_log_flush)
                except Exception:
                    pass
            return

        # Fallback: legacy queue -> txt_logs if present
        try:
            fallback_q = getattr(self, "log_queue", None)
            for _ in range(200):
                try:
                    line = fallback_q.get_nowait()
                except Exception:
                    break
                try:
                    if getattr(self, "txt_logs", None):
                        self.txt_logs.insert(END, line + "\n")
                        self.txt_logs.see(END)
                except Exception:
                    pass
        except Exception:
            try:
                if getattr(self, "txt_logs", None):
                    self.txt_logs.insert(END, f"[{safe_now_str()}] ERROR flushing logs\n")
            except Exception:
                pass
        finally:
            try:
                self.root.after(self.log_flush_interval_ms, self._schedule_log_flush)
            except Exception:
                pass


    def _append_log(self, msg: str):
        """Safe to call from main thread (puts in queue too)."""
        self._enqueue_log(msg)


    def _clear_logs(self):
        if getattr(self, "log_widget", None):
            try:
                self.log_widget.configure(state="normal")
                self.log_widget.delete("1.0", "end")
                self.log_widget.configure(state="disabled")
            except Exception:
                pass
        elif getattr(self, "txt_logs", None):
            try:
                self.txt_logs.delete("1.0", END)
            except Exception:
                pass

    # --------------------------
    # Status-related helpers (NEW)
    # --------------------------
    def _apply_status_settings(self):
        """Apply intervals/settings - currently only logs and stores into vars.
        In future could push into daemon via update_from_dict."""
        try:
            inf = float(self.inference_interval_var.get())
            trade = float(self.trade_interval_var.get())
            ref = float(self.refresh_interval_var.get())
            self._enqueue_log(f"Applied status intervals: inference={inf}s trade={trade}s refresh={ref}s")
            # If daemon exists, we could push these params into it via update_from_dict (optionally)
            if self.daemon:
                try:
                    cfg = {"poll_interval": ref}
                    # we intentionally do not force changes to model runtime params here,
                    # but you may add more mappings (e.g. inference interval) if your daemon uses them.
                    self.daemon.update_from_dict(cfg, save=False, reload_artifacts=False)
                    self._enqueue_log("Pushed refresh interval to daemon.poll_interval.")
                except Exception:
                    self._enqueue_log("Failed to push settings to daemon (update_from_dict missing or error).")
        except Exception as e:
            self._enqueue_log(f"Apply intervals failed: {e}")

    def _connect_websocket(self):
        if websocket is None:
            self._enqueue_log("websocket-client library missing. Run: pip install websocket-client")
            return
        url = self.websocket_url_var.get().strip()
        if not url:
            self._enqueue_log("Websocket URL empty.")
            self.websocket_status_var.set("Disconnected")
            return

        # If already connected, ignore
        if getattr(self, "ws_connected", False):
            self._enqueue_log("Already connected.")
            return

        # Create WebSocketApp with callbacks
        def on_open(ws):
            self.ws_queue.put({"type":"_meta","event":"open"})
            self._enqueue_log("WS open (callback)")

        def on_close(ws, close_status_code, close_msg):
            self.ws_queue.put({"type":"_meta","event":"close", "code": close_status_code, "msg": close_msg})
            self._enqueue_log(f"WS closed: {close_status_code} {close_msg}")

        def on_error(ws, err):
            self.ws_queue.put({"type":"_meta","event":"error", "error": str(err)})
            self._enqueue_log(f"WS error: {err}")

        def on_message(ws, message):
            # Put raw message in queue; actual parsing done on main thread to avoid race conditions with Tk.
            self.ws_queue.put({"type":"message","raw": message})

        self.ws_app = websocket.WebSocketApp(url,
                                             on_open=on_open,
                                             on_message=on_message,
                                             on_error=on_error,
                                             on_close=on_close)
        # Run in thread
        self.ws_thread = threading.Thread(target=lambda: self.ws_app.run_forever(ping_interval=30, ping_timeout=10), daemon=True)
        self.ws_thread.start()
        self.websocket_status_var.set("Connecting...")
        # schedule queue processor
        try:
            self.nb.after(150, self._process_ws_queue)
        except Exception:
            # fallback to top-level window if nb missing
            self.after(150, self._process_ws_queue)

    def _refresh_status(self):
        """Refresh small metadata panel: tries to read model metadata from daemon or GUI memory."""
        try:
            # prefer daemon metadata if present
            if self.daemon and getattr(self.daemon, "model_meta", None):
                mm = self.daemon.model_meta
                self.model_symbol_var.set(mm.get("symbol", self.symbol.get()))
                self.model_exchange_var.set(mm.get("exchange", self.daemon.exchange_id or ""))
                self.model_timeframe_var.set(mm.get("timeframe", self.timeframe.get()))
                self._enqueue_log("Refreshed model metadata from daemon.")
            else:
                # fallback to GUI values or current form fields
                self.model_symbol_var.set(self.symbol.get())
                self.model_exchange_var.set("")
                self.model_timeframe_var.set(self.timeframe.get())
                self._enqueue_log("Refreshed model metadata from GUI state.")
        except Exception as e:
            self._enqueue_log(f"Refresh status failed: {e}")

    def _fetch_last_rows_status(self, limit: int = 50):
        """Load last rows for the selected symbol/timeframe and populate the status_data_tree.
        Runs in background to avoid UI freeze."""
        def worker():
            try:
                # Prefer daemon SQLite if present
                df = None
                if self.daemon:
                    try:
                        df = self.daemon._load_recent_rows(limit=limit)
                    except Exception:
                        # fallback to local sqlite read
                        df = None
                if df is None:
                    # local read
                    p = Path(self.sqlite_path.get())
                    if not p.exists():
                        self._enqueue_log(f"SQLite not found: {p}")
                        return
                    import sqlite3
                    con = sqlite3.connect(str(p))
                    q = f"SELECT * FROM {self.table.get()} WHERE symbol = ? AND timeframe = ? ORDER BY ts DESC LIMIT ?"
                    df = pd.read_sql_query(q, con, params=[self.symbol.get(), self.timeframe.get(), int(limit)])
                    con.close()
                    if df.empty:
                        self._enqueue_log("No rows returned for status fetch.")
                        # schedule UI clear
                        self.root.after(0, self._clear_status_data_tree)
                        return
                # ensure chronological ascending order for display
                if "ts" in df.columns and "timestamp" not in df.columns:
                    try:
                        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
                    except Exception:
                        pass
                sort_col = "timestamp" if "timestamp" in df.columns else ("ts" if "ts" in df.columns else df.columns[0])
                df = df.sort_values(sort_col).reset_index(drop=True)

                # update last price/timestamp/ticker for summary
                last_row = df.iloc[-1] if len(df) > 0 else None
                lp = ""
                lts = ""
                ticker = self.symbol.get()
                if last_row is not None:
                    if "close" in last_row:
                        lp = str(float(last_row["close"]))
                    elif "price" in last_row:
                        lp = str(float(last_row["price"]))
                    else:
                        # pick first numeric
                        for c in df.columns:
                            try:
                                lp = str(float(last_row[c])); break
                            except Exception:
                                continue
                    if "timestamp" in df.columns:
                        try:
                            # format to readable
                            tval = last_row.get("timestamp", None)
                            if pd.isna(tval):
                                lts = str(last_row.get("ts", ""))
                            else:
                                lts = str(pd.to_datetime(tval))
                        except Exception:
                            lts = str(last_row.get("ts", ""))
                # schedule UI update
                def update_ui():
                    # populate tree
                    self._populate_status_data_tree(df)
                    self.last_price_var.set(lp or "N/A")
                    self.last_ts_var.set(lts or "N/A")
                    self.ticker_var.set(ticker or "N/A")
                self.root.after(0, update_ui)
                self._enqueue_log(f"Status data fetched: {len(df)} rows.")
            except Exception as e:
                self._enqueue_log(f"Fetch last rows failed: {e}")
                self._enqueue_log(traceback.format_exc())
        threading.Thread(target=worker, daemon=True).start()

    def _clear_status_data_tree(self):
        for iid in self.status_data_tree.get_children():
            self.status_data_tree.delete(iid)
        self.status_data_tree["columns"] = ()

    def _populate_status_data_tree(self, df: pd.DataFrame):
        try:
            # Clear existing
            for iid in self.status_data_tree.get_children():
                self.status_data_tree.delete(iid)
            # Setup columns
            cols = list(df.columns)
            self.status_data_tree["columns"] = cols
            # heading config
            for c in cols:
                self.status_data_tree.heading(c, text=c)
                self.status_data_tree.column(c, width=120)
            # insert rows (stringified)
            # limit to last 500 rows to avoid UI overload
            max_rows = min(len(df), 500)
            start = max(0, len(df) - max_rows)
            for idx in range(start, len(df)):
                row = df.iloc[idx]
                values = []
                for c in cols:
                    try:
                        v = row.get(c, "")
                        values.append(str(v))
                    except Exception:
                        values.append("")
                self.status_data_tree.insert("", "end", values=values)
        except Exception:
            self._enqueue_log("Failed to populate status data tree")
            self._enqueue_log(traceback.format_exc())

    # --------------------------
    # Prediction helper (NEW)
    # --------------------------
    def _get_latest_prediction_thread(self):
        """Start prediction worker in background to avoid UI freeze."""
        t = threading.Thread(target=self._get_latest_prediction, daemon=True)
        t.start()
        self._enqueue_log("Prediction worker started (background).")

    def _get_latest_prediction(self):
        """
        Compute latest prediction using:
         - last data from sqlite (via daemon._load_recent_rows if available)
         - features via fibo.add_technical_features
         - sequence builder (fibo or internal)
         - scaler self.daemon.model_scaler or self.model_scaler
         - model inference using self.daemon.model or self.model

        This function **does not execute trades**. It only runs inference.
        """
        try:
            # load recent data
            df = None
            if self.daemon:
                try:
                    df = self.daemon._load_recent_rows(limit=1000)
                except Exception:
                    df = None
            if df is None:
                # local read
                p = Path(self.sqlite_path.get())
                if not p.exists():
                    self._enqueue_log(f"SQLite not found for prediction: {p}")
                    return
                import sqlite3
                con = sqlite3.connect(str(p))
                q = f"SELECT * FROM {self.table.get()} WHERE symbol = ? AND timeframe = ? ORDER BY ts DESC LIMIT ?"
                df = pd.read_sql_query(q, con, params=[self.symbol.get(), self.timeframe.get(), 1000])
                con.close()
            if df is None or df.empty:
                self._enqueue_log("No data available for prediction.")
                return

            # compute features (use fibo if available)
            if fibo is None or not hasattr(fibo, "add_technical_features"):
                self._enqueue_log("fiboevo.add_technical_features not available; cannot compute features for prediction.")
                return
            try:
                close = df["close"].astype(float).values
                high = df["high"].astype(float).values if "high" in df.columns else None
                low = df["low"].astype(float).values if "low" in df.columns else None
                vol = df["volume"].astype(float).values if "volume" in df.columns else None
                feats = fibo.add_technical_features(close, high=high, low=low, volume=vol)
                if not isinstance(feats, pd.DataFrame):
                    feats = pd.DataFrame(np.asarray(feats))
                # attach columns from df if missing
                for col in ("timestamp","open","high","low","close","volume","symbol","timeframe"):
                    if col in df.columns and col not in feats.columns:
                        feats[col] = df[col].values
                feats = feats.dropna().reset_index(drop=True)
            except Exception as e:
                self._enqueue_log(f"Feature computation for prediction failed: {e}")
                self._enqueue_log(traceback.format_exc())
                return

            # detect feature columns (use daemon.model_meta['feature_cols'] if available)
            feature_cols = None
            if self.daemon and getattr(self.daemon, "model_meta", None):
                mc = self.daemon.model_meta.get("feature_cols", None)
                if isinstance(mc, (list, tuple)):
                    # ensure present in feats
                    feature_cols = [c for c in mc if c in feats.columns]
            if feature_cols is None:
                # fallback to auto-detect numeric features
                exclude = {"timestamp","open","high","low","close","volume","symbol","timeframe","exchange"}
                feature_cols = [c for c in feats.columns if c not in exclude and pd.api.types.is_numeric_dtype(feats[c])]
            if not feature_cols:
                self._enqueue_log("No feature columns available for prediction.")
                return

            seq_len = int(self.seq_len.get())
            horizon = int(self.horizon.get())
            # build sequences
            try:
                if hasattr(fibo, "create_sequences_from_df"):
                    X_all, y_ret, y_vol = fibo.create_sequences_from_df(feats, feature_cols, seq_len=seq_len, horizon=horizon)
                    X_all = np.asarray(X_all)
                else:
                    X_all, y = self._build_sequences_internal(feats, feature_cols, seq_len, horizon, dtype=np.float32)
            except Exception as e:
                self._enqueue_log(f"Sequence builder failed for prediction: {e}")
                self._enqueue_log(traceback.format_exc())
                return

            if X_all is None or X_all.shape[0] == 0:
                self._enqueue_log("No sequences produced for prediction (maybe not enough rows).")
                return

            # choose last sequence
            X_last = X_all[-1:]  # shape (1, seq_len, F)

            # scaler: prefer daemon.model_scaler then GUI.model_scaler
            scaler = None
            if self.daemon and getattr(self.daemon, "model_scaler", None) is not None:
                scaler = self.daemon.model_scaler
            elif getattr(self, "model_scaler", None) is not None:
                scaler = self.model_scaler

            Xp = X_last
            if scaler is not None:
                try:
                    flat = X_last.reshape(-1, X_last.shape[2])
                    flat_s = scaler.transform(flat)
                    Xp = flat_s.reshape(X_last.shape)
                except Exception as e:
                    self._enqueue_log(f"Scaler transform failed for prediction: {e}")

            # model: prefer daemon.model then gui self.model
            model_ref = None
            artifact_lock = None
            if self.daemon and getattr(self.daemon, "model", None) is not None:
                artifact_lock = getattr(self.daemon, "_artifact_lock", None)
                # safe copy under lock if available
                if artifact_lock is not None:
                    artifact_lock.acquire()
                    try:
                        model_ref = self.daemon.model
                    finally:
                        artifact_lock.release()
                else:
                    model_ref = self.daemon.model
            elif getattr(self, "model", None) is not None:
                model_ref = self.model

            if model_ref is None:
                self._enqueue_log("No model loaded for inference.")
                return

            # run inference (use torch if available)
            if torch is None:
                self._enqueue_log("PyTorch not available; cannot run inference.")
                return
            import torch as _torch
            try:
                model_ref.to("cpu")
                model_ref.eval()
                with _torch.no_grad():
                    xb = _torch.from_numpy(Xp).float()
                    out = model_ref(xb)
                    # interpret output
                    if isinstance(out, (tuple, list)):
                        out_r = out[0]
                        out_v = out[1] if len(out) > 1 else None
                    else:
                        out_r = out
                        out_v = None
                    pred_val = float(out_r.cpu().numpy().ravel()[0])
                    pred_vol = float(out_v.cpu().numpy().ravel()[0]) if out_v is not None else None
            except Exception as e:
                self._enqueue_log(f"Inference error: {e}")
                self._enqueue_log(traceback.format_exc())
                return

            # convert log-return to pct if plausible (since training used log diffs)
            pred_log = pred_val
            try:
                pred_pct = math.exp(pred_log) - 1.0
            except Exception:
                pred_pct = None

            # estimate prediction timestamp: last timestamp + horizon * timeframe
            pred_ts_str = "N/A"
            try:
                last_ts = None
                if "timestamp" in feats.columns:
                    last_ts = pd.to_datetime(feats["timestamp"].iloc[-1])
                elif "ts" in feats.columns:
                    last_ts = pd.to_datetime(feats["ts"].iloc[-1], unit="s", utc=True)
                if last_ts is not None:
                    tf_sec = timeframe_to_seconds(self.model_timeframe_var.get() or self.timeframe.get())
                    if tf_sec is None:
                        # try to infer from self.timeframe (GUI)
                        tf_sec = timeframe_to_seconds(self.timeframe.get())
                    if tf_sec:
                        pred_ts = last_ts + pd.Timedelta(seconds=int(tf_sec * horizon))
                        pred_ts_str = str(pred_ts)
                    else:
                        pred_ts_str = str(last_ts)
            except Exception:
                pred_ts_str = "N/A"

            # Schedule UI update on main thread
            def _update_ui():
                self.pred_log_var.set(f"{pred_log:.6f}")
                self.pred_pct_var.set(f"{(pred_pct*100):.3f}%" if pred_pct is not None else "N/A")
                self.pred_vol_var.set(f"{pred_vol:.6f}" if pred_vol is not None else "N/A")
                self.pred_ts_var.set(pred_ts_str)
                # also update last price/timestamp to current values
                try:
                    if "close" in feats.columns:
                        self.last_price_var.set(str(float(feats["close"].iloc[-1])))
                    if "timestamp" in feats.columns:
                        self.last_ts_var.set(str(pd.to_datetime(feats["timestamp"].iloc[-1])))
                except Exception:
                    pass
            self.root.after(0, _update_ui)

            self._enqueue_log(f"Prediction computed: log={pred_log:.6f} pct={(pred_pct*100) if pred_pct is not None else 'N/A'} vol={pred_vol}")
        except Exception as e:
            self._enqueue_log(f"Prediction worker failed: {e}")
            self._enqueue_log(traceback.format_exc())


    # --------------------------
    # Config persistence
    # --------------------------
    def _save_config(self):
        cfg = {
            "sqlite_path": self.sqlite_path.get(),
            "table": self.table.get(),
            "symbol": self.symbol.get(),
            "timeframe": self.timeframe.get(),
            "seq_len": int(self.seq_len.get()),
            "horizon": int(self.horizon.get()),
            "hidden": int(self.hidden.get()),
            "epochs": int(self.epochs.get()),
            "batch_size": int(self.batch_size.get()),
            "lr": float(self.lr.get()),
            "val_frac": float(self.val_frac.get()),
            "dtype": self.dtype_choice.get(),
            "feature_cols_manual": self.feature_cols_manual.get()
        }
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            self._enqueue_log(f"Config saved to {self.config_path}")
        except Exception as e:
            self._enqueue_log(f"Save config failed: {e}")

    def _load_config_on_start(self):
        if self.config_path.exists():
            try:
                cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
                self.sqlite_path.set(cfg.get("sqlite_path", self.sqlite_path.get()))
                self.table.set(cfg.get("table", self.table.get()))
                self.symbol.set(cfg.get("symbol", self.symbol.get()))
                self.timeframe.set(cfg.get("timeframe", self.timeframe.get()))
                self.seq_len.set(cfg.get("seq_len", self.seq_len.get()))
                self.horizon.set(cfg.get("horizon", self.horizon.get()))
                self.hidden.set(cfg.get("hidden", self.hidden.get()))
                self.epochs.set(cfg.get("epochs", self.epochs.get()))
                self.batch_size.set(cfg.get("batch_size", self.batch_size.get()))
                self.lr.set(cfg.get("lr", self.lr.get()))
                self.val_frac.set(cfg.get("val_frac", self.val_frac.get()))
                self.dtype_choice.set(cfg.get("dtype", self.dtype_choice.get()))
                self.feature_cols_manual.set(cfg.get("feature_cols_manual", self.feature_cols_manual.get()))
                self._enqueue_log(f"Config loaded from {self.config_path}")
            except Exception as e:
                self._enqueue_log(f"Failed loading config: {e}")

    # --------------------------
    # SQLite preview and load helpers
    # --------------------------
    def _browse_sqlite(self):
        p = filedialog.askopenfilename(title="Select SQLite", filetypes=[("SQLite","*.db *.sqlite *.sqlite3"),("All","*.*")])
        if p:
            self.sqlite_path.set(p)

    # NEW: browse model file
    def _browse_model(self):
        p = filedialog.askopenfilename(title="Select model checkpoint", filetypes=[("PyTorch checkpoint","*.pt *.pth"),("All","*.*")])
        if p:
            self.model_path_var.set(p)

    def _refresh_preview(self):
        p = Path(self.sqlite_path.get())
        if not p.exists():
            self._enqueue_log(f"SQLite not found: {p}")
            return
        try:
            import sqlite3
            con = sqlite3.connect(str(p))
            q = f"SELECT * FROM {self.table.get()} WHERE symbol = ? AND timeframe = ? ORDER BY ts DESC LIMIT 200"
            df = pd.read_sql_query(q, con, params=[self.symbol.get(), self.timeframe.get()])
            con.close()
            if df.empty:
                self._enqueue_log("No rows found for selection.")
                # clear tree
                for iid in self.preview_tree.get_children():
                    self.preview_tree.delete(iid)
                return
            df = df.astype(str)
            cols = list(df.columns)
            self.preview_tree["columns"] = cols
            for c in cols:
                self.preview_tree.heading(c, text=c)
                self.preview_tree.column(c, width=120)
            # clear and insert
            for iid in self.preview_tree.get_children():
                self.preview_tree.delete(iid)
            for _, row in df.iterrows():
                self.preview_tree.insert("", "end", values=list(row.values))
            self._enqueue_log(f"Preview loaded: {len(df)} rows.")
        except Exception as e:
            self._enqueue_log(f"Preview failed: {e}")
            self._enqueue_log(traceback.format_exc())

    def _show_preview(self, which: str):
        if which == "features":
            df = self.df_features
        elif which == "scaled":
            df = self.df_scaled
        else:
            df = self.df_loaded
        if df is None:
            self._enqueue_log(f"No dataframe available for preview: {which}")
            return
        try:
            s = df.head(200).to_string()
            self.train_info_text.delete("1.0", END)
            self.train_info_text.insert(END, s)
        except Exception as e:
            self._enqueue_log(f"Preview show failed: {e}")

    # --------------------------
    # Prepare Data worker (background)
    # --------------------------
    def _start_prepare(self):
        if self.prepare_thread and self.prepare_thread.is_alive():
            self._enqueue_log("Prepare already running.")
            return
        t = threading.Thread(target=self._training_worker_prepare_only, daemon=True)
        self.prepare_thread = t
        t.start()
        self._enqueue_log("Prepare thread started.")

    def _training_worker_prepare_only(self):
        """Prepare: load sqlite, compute features, dropna, build sequences, fit scaler (only preview), store X/y/scaler."""
        try:
            self._enqueue_log("Prepare: loading data from sqlite...")
            df = self._load_df_for_training()
            if df is None:
                self._enqueue_log("No data; abort prepare.")
                return
            self.df_loaded = df.copy()
            self._enqueue_log(f"Loaded {len(df)} rows from sqlite.")

            # features
            if fibo is None or not hasattr(fibo, "add_technical_features"):
                self._enqueue_log("fiboevo.add_technical_features not available; abort.")
                return
            try:
                close = df["close"].astype(float).values
                high = df["high"].astype(float).values
                low = df["low"].astype(float).values
                vol = df["volume"].astype(float).values if "volume" in df.columns else None
                feats = fibo.add_technical_features(close, high=high, low=low, volume=vol)
                if not isinstance(feats, pd.DataFrame):
                    # try to coerce
                    arr = np.asarray(feats)
                    if arr.ndim == 2:
                        cols = [f"f{i}" for i in range(arr.shape[1])]
                        feats = pd.DataFrame(arr, columns=cols)
                    else:
                        self._enqueue_log("Features from fiboevo not in expected shape. Abort.")
                        return
                # attach OHLCV and timestamp if not present
                for col in ("timestamp","open","high","low","close","volume","symbol","timeframe"):
                    if col in df.columns and col not in feats.columns:
                        feats[col] = df[col].values
                if "timestamp" in feats.columns:
                    feats = feats.sort_values("timestamp").reset_index(drop=True)
                self.df_features = feats.copy()
                self._enqueue_log(f"Computed features. Shape: {self.df_features.shape}")
            except Exception as e:
                self._enqueue_log(f"Feature computation error: {e}")
                self._enqueue_log(traceback.format_exc())
                return

            # dropna
            before = len(self.df_features)
            self.df_features = self.df_features.dropna().reset_index(drop=True)
            after = len(self.df_features)
            self._enqueue_log(f"dropna: before={before}, after={after}, removed={before-after}")

            # detect feature cols
            if self.feature_cols_manual.get().strip():
                feature_cols = [c.strip() for c in self.feature_cols_manual.get().split(",") if c.strip() and c in self.df_features.columns]
            else:
                exclude = {"timestamp","open","high","low","close","volume","symbol","timeframe","exchange"}
                feature_cols = [c for c in self.df_features.columns if c not in exclude and pd.api.types.is_numeric_dtype(self.df_features[c])]
            if not feature_cols:
                self._enqueue_log("No numeric feature columns detected after cleaning. Abort.")
                return
            self.feature_cols_used = feature_cols
            self._enqueue_log(f"Feature cols selected ({len(feature_cols)}): {feature_cols[:20]}{'...' if len(feature_cols)>20 else ''}")

            # build sequences
            seq_len = int(self.seq_len.get()); horizon = int(self.horizon.get())
            dtype = np.float32 if self.dtype_choice.get() == "float32" else np.float64
            self._enqueue_log("Building sequences...")
            try:
                if hasattr(fibo, "create_sequences_from_df"):
                    X, y_ret, y_vol = fibo.create_sequences_from_df(self.df_features, feature_cols, seq_len=seq_len, horizon=horizon)
                    X = np.asarray(X).astype(dtype)
                    y = np.asarray(y_ret).astype(dtype).reshape(-1,)
                else:
                    X, y = self._build_sequences_internal(self.df_features, feature_cols, seq_len, horizon, dtype=dtype)
            except Exception as e:
                self._enqueue_log(f"Sequence building failed: {e}")
                self._enqueue_log(traceback.format_exc())
                return

            if X.size == 0:
                self._enqueue_log("No sequences produced. Check seq_len/horizon/warmup.")
                return
            self.X_full = X
            self.y_full = y
            self._enqueue_log(f"Sequences built: N={X.shape[0]}, seq_len={X.shape[1]}, features={X.shape[2]}")

            # fit scaler on training portion only (temporal)
            N = X.shape[0]; val_frac = float(self.val_frac.get())
            n_val = int(np.floor(N * val_frac)); n_train = N - n_val
            if n_train <= 0:
                self._enqueue_log("Train split <= 0; adjust val_frac or get more data.")
                return
            Xtr = X[:n_train]
            if StandardScaler is not None:
                try:
                    scaler = StandardScaler()
                    flat = Xtr.reshape(-1, Xtr.shape[2])
                    scaler.fit(flat)
                    # prepare df_scaled for preview (transform original features matrix)
                    if self.feature_cols_used is not None:
                        flat_all = self.df_features[self.feature_cols_used].astype(float).values
                        flat_scaled = scaler.transform(flat_all)
                        df_scaled = self.df_features.copy()
                        for i, c in enumerate(self.feature_cols_used):
                            df_scaled[c] = flat_scaled[:, i]
                        self.df_scaled = df_scaled
                        self._enqueue_log("Scaler fitted and df_scaled prepared.")
                    self.scaler_used = scaler
                except Exception as e:
                    self._enqueue_log(f"Scaler fit failed: {e}")
                    self.scaler_used = None
            else:
                self._enqueue_log("scikit-learn not available: skipping scaler.")
                self.scaler_used = None

            # previews into train_info_text (main thread safe since we only enqueue)
            try:
                self._enqueue_log("Preview raw (first 5 rows):\n" + self.df_loaded.head(5).to_string())
                self._enqueue_log("Preview features (first 5 rows):\n" + self.df_features.head(5).to_string())
                if self.df_scaled is not None:
                    self._enqueue_log("Preview scaled (first 5 rows):\n" + self.df_scaled.head(5).to_string())
            except Exception:
                pass

            self._enqueue_log("Prepare stage finished successfully.")
        except Exception as e:
            self._enqueue_log(f"Prepare worker unexpected error: {e}")
            self._enqueue_log(traceback.format_exc())

    # --------------------------
    # Train Model worker (A)
    # --------------------------
    def _start_train_model(self):
        if self.training_thread and self.training_thread.is_alive():
            self._enqueue_log("Training already running.")
            return
        # require X_full loaded
        if self.X_full is None or self.y_full is None:
            self._enqueue_log("No prepared dataset found. Run Prepare Data first or use Prepare+Train.")
            return
        t = threading.Thread(target=self._train_model_worker, daemon=True)
        self.training_thread = t
        t.start()
        self._enqueue_log("Training thread started.")

    def _train_model_worker(self):
        """Train model using self.X_full, self.y_full, self.feature_cols_used and scaler_used.
           Saves best model to artifacts/model_best.pt, scaler to artifacts/scaler.pkl and meta.json.
        """
        try:
            if torch is None:
                self._enqueue_log("PyTorch is not available; cannot train.")
                return
            X = self.X_full; y = self.y_full
            if X is None or y is None:
                self._enqueue_log("No data found to train.")
                return
            dtype_np = np.float32 if self.dtype_choice.get()=="float32" else np.float64
            # temporal split
            N = X.shape[0]; val_frac = float(self.val_frac.get())
            n_val = int(np.floor(N * val_frac)); n_train = N - n_val
            if n_train <= 0:
                self._enqueue_log("Train split <= 0; abort.")
                return
            Xtr = X[:n_train]; ytr = y[:n_train]
            Xv = X[n_train:] if n_val>0 else None; yv = y[n_train:] if n_val>0 else None

            # apply scaler if present
            scaler = self.scaler_used
            if scaler is not None:
                try:
                    flat_tr = Xtr.reshape(-1, Xtr.shape[2])
                    flat_tr_t = scaler.transform(flat_tr).reshape(Xtr.shape)
                    Xtr = flat_tr_t.astype(dtype_np)
                    if Xv is not None:
                        flat_v = Xv.reshape(-1, Xv.shape[2])
                        Xv = scaler.transform(flat_v).reshape(Xv.shape).astype(dtype_np)
                    self._enqueue_log("Applied scaler to sequences for training.")
                except Exception as e:
                    self._enqueue_log(f"Scaler.transform failed: {e}. Proceeding without scaling for training.")

            # Torch datasets
            device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
            self._enqueue_log(f"Using device: {device}")
            Xtr_t = torch.from_numpy(Xtr).float().to(device) if dtype_np==np.float32 else torch.from_numpy(Xtr).double().to(device)
            ytr_t = torch.from_numpy(ytr).float().view(-1,1).to(device)
            train_ds = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(self.batch_size.get()), shuffle=True)

            val_loader = None
            if Xv is not None:
                Xv_t = torch.from_numpy(Xv).float().to(device) if dtype_np==np.float32 else torch.from_numpy(Xv).double().to(device)
                yv_t = torch.from_numpy(yv).float().view(-1,1).to(device)
                val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xv_t, yv_t), batch_size=int(self.batch_size.get()), shuffle=False)

            # instantiate model
            F = X.shape[2]
            hidden = int(self.hidden.get())
            try:
                model = fibo.LSTM2Head(input_size=F, hidden_size=hidden)
            except Exception as e:
                self._enqueue_log(f"Could not instantiate model LSTM2Head: {e}")
                return
            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=float(self.lr.get()))
            loss_fn = torch.nn.MSELoss()

            # training loop
            best_val = float("inf")
            artifacts_dir = Path("artifacts"); artifacts_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifacts_dir / "model_best.pt"
            scaler_path = artifacts_dir / "scaler.pkl"
            meta_path = artifacts_dir / "meta.json"

            epochs = int(self.epochs.get())
            for ep in range(epochs):
                model.train()
                running_loss = 0.0; batches = 0
                for xb, yb in train_loader:
                    opt.zero_grad()
                    out_r, out_v = model(xb)
                    # ensure out_r shape is (batch,1)
                    if out_r.dim() == 1:
                        out_r = out_r.view(-1,1)
                    loss = loss_fn(out_r, yb)
                    loss.backward()
                    opt.step()
                    running_loss += float(loss.item()); batches += 1
                train_loss = running_loss / max(1, batches)
                val_loss = None
                if val_loader is not None:
                    model.eval()
                    vl = 0.0; vc = 0
                    with torch.no_grad():
                        for xv, yv_b in val_loader:
                            out_r, _ = model(xv)
                            if out_r.dim() == 1:
                                out_r = out_r.view(-1,1)
                            l = loss_fn(out_r, yv_b)
                            vl += float(l.item()); vc += 1
                    val_loss = vl / max(1, vc)
                # log
                self._enqueue_log(f"[Train] Epoch {ep+1}/{epochs} train_loss={train_loss:.6f}" + (f" val_loss={val_loss:.6f}" if val_loss is not None else ""))
                # save best
                if val_loss is not None and val_loss < best_val:
                    best_val = val_loss
                    try:
                        torch.save(model.state_dict(), str(model_path))
                        meta = {"feature_cols": self.feature_cols_used, "seq_len": int(self.seq_len.get()), "horizon": int(self.horizon.get()), "hidden": hidden}
                        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                        if scaler is not None and joblib is not None:
                            joblib.dump(scaler, str(scaler_path))
                        self._enqueue_log(f"Saved best model & artifacts to {artifacts_dir}")
                    except Exception as e:
                        self._enqueue_log(f"Saving artifacts failed: {e}")

            # final save if best never updated
            if not model_path.exists():
                try:
                    torch.save(model.state_dict(), str(model_path))
                    meta = {"feature_cols": self.feature_cols_used, "seq_len": int(self.seq_len.get()), "horizon": int(self.horizon.get()), "hidden": hidden}
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    if scaler is not None and joblib is not None:
                        joblib.dump(scaler, str(scaler_path))
                    self._enqueue_log(f"Saved final model to {model_path}")
                except Exception as e:
                    self._enqueue_log(f"Final save failed: {e}")

            self.model = model
            self.model_meta = {"feature_cols": self.feature_cols_used}
            self.model_scaler = scaler
            self._enqueue_log("Training completed.")
        except Exception as e:
            self._enqueue_log(f"Training worker failed: {e}")
            self._enqueue_log(traceback.format_exc())

    # --------------------------
    # Combined Prepare + Train (C)
    # --------------------------
    def _start_prepare_and_train(self):
        if self.combine_thread and self.combine_thread.is_alive():
            self._enqueue_log("Prepare+Train already running.")
            return
        t = threading.Thread(target=self._prepare_and_train_worker, daemon=True)
        self.combine_thread = t
        t.start()
        self._enqueue_log("Prepare+Train worker started.")

    def _prepare_and_train_worker(self):
        # run prepare then train sequentially
        self._training_worker_prepare_only()
        # small pause to ensure arrays are set
        time.sleep(0.5)
        if self.X_full is None:
            self._enqueue_log("After prepare, no X_full available. Aborting train.")
            return
        self._train_model_worker()

    # --------------------------
    # Load artifacts helper (ADDED METHOD)
    # --------------------------
    def _load_model_from_artifacts(self):
        """
        Carga modelo, meta y scaler desde ./artifacts:
          - model_best.pt (estado guardado con torch.save(...))
          - meta.json
          - scaler.pkl (joblib)
        Guarda en self.model, self.model_meta, self.model_scaler.
        Usa self._enqueue_log() para mensajes (thread-safe).
        """
        try:
            artifacts_dir = Path("artifacts")
            model_path = artifacts_dir / "model_best.pt"
            meta_path = artifacts_dir / "meta.json"
            scaler_path = artifacts_dir / "scaler.pkl"

            if not model_path.exists():
                self._enqueue_log(f"No se encontró {model_path}. Ejecuta entrenamiento o coloca el artefacto en ./artifacts/")
                return

            if fibo is None:
                self._enqueue_log("fiboevo no disponible; no puedo reconstruir la arquitectura del modelo.")
                return

            # leer meta si existe
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    self._enqueue_log(f"meta.json leído: {meta_path}")
                except Exception as e:
                    self._enqueue_log(f"Advertencia: no se pudo leer meta.json: {e}")

            # require torch to load state dict
            if torch is None:
                self._enqueue_log("torch no está instalado; no se puede cargar el state_dict del modelo (instala torch).")
                return

            # Cargar checkpoint en CPU (no asignamos aún al dispositivo final)
            try:
                ckpt = torch.load(str(model_path), map_location="cpu")
            except Exception as e:
                self._enqueue_log(f"Fallo al cargar checkpoint {model_path}: {e}")
                self._enqueue_log(traceback.format_exc())
                return

            # Extraer state_dict y meta embebida si existe
            state = ckpt
            ckpt_meta = {}
            if isinstance(ckpt, dict):
                # casos comunes: {'state': state_dict, 'meta': meta}, {'model_state_dict': ...}, plain state_dict
                if "state" in ckpt and isinstance(ckpt["state"], dict):
                    state = ckpt["state"]
                    ckpt_meta = ckpt.get("meta", {})
                elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
                    state = ckpt["model_state_dict"]
                    ckpt_meta = ckpt.get("meta", {})
                elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                    state = ckpt["state_dict"]
                    ckpt_meta = ckpt.get("meta", {})
                else:
                    # top-level might already be a state_dict, but there may also be a meta key
                    if "meta" in ckpt and isinstance(ckpt["meta"], dict):
                        ckpt_meta = ckpt["meta"]

            # merge meta priorities: explicit meta.json > checkpoint meta > {}
            combined_meta = dict(ckpt_meta)
            combined_meta.update(meta or {})
            meta = combined_meta

            # normalize keys 'module.' si vinieron de DataParallel
            normalized = {}
            if isinstance(state, dict):
                for k, v in state.items():
                    nk = k[len("module.") :] if k.startswith("module.") else k
                    normalized[nk] = v
                state = normalized
            else:
                # state may be a serialized model object; we will try to use it directly later
                state = state

            # Inferir input_size, hidden, num_layers desde state si es posible
            inferred_input = None
            inferred_hidden = None
            inferred_layers = 1

            if isinstance(state, dict):
                ih_keys = sorted([k for k in state.keys() if k.startswith("lstm.weight_ih_l")])
                if ih_keys:
                    try:
                        # deducir num_layers por la máxima lX encontrada
                        max_idx = max(int(k.split("l")[-1]) for k in ih_keys)
                        inferred_layers = max_idx + 1
                    except Exception:
                        inferred_layers = len(ih_keys)

                if "lstm.weight_ih_l0" in state:
                    w = state["lstm.weight_ih_l0"]
                    shape = getattr(w, "shape", None)
                    if shape and len(shape) == 2:
                        inferred_input = int(shape[1])
                        inferred_hidden = int(shape[0] // 4)

                # fallback: tratar de inferir hidden desde heads si no hay lstm
                if inferred_hidden is None:
                    for cand in ("head_ret.0.weight", "head_ret.2.weight", "head_ret.weight"):
                        if cand in state:
                            w = state[cand]
                            shape = getattr(w, "shape", None)
                            if shape and len(shape) == 2:
                                inferred_hidden = int(shape[1])
                                break

            # metadata override / fallback a self.feature_cols_used
            input_size = None
            if "input_size" in meta and meta.get("input_size") is not None:
                try:
                    input_size = int(meta.get("input_size"))
                except Exception:
                    input_size = None

            if input_size is None:
                feature_cols_meta = meta.get("feature_cols", None)
                if isinstance(feature_cols_meta, (list, tuple)):
                    input_size = len(feature_cols_meta)
                elif getattr(self, "feature_cols_used", None):
                    try:
                        input_size = len(self.feature_cols_used)
                    except Exception:
                        input_size = None

            if input_size is None and inferred_input is not None:
                input_size = int(inferred_input)

            hidden = None
            if "hidden" in meta and meta.get("hidden") is not None:
                try:
                    hidden = int(meta.get("hidden"))
                except Exception:
                    hidden = None
            if hidden is None and inferred_hidden is not None:
                hidden = int(inferred_hidden)
            if hidden is None:
                # fallback razonable
                hidden = int(getattr(self, "hidden", self.hidden.get() if hasattr(self, "hidden") else 64))

            num_layers = None
            if "num_layers" in meta and meta.get("num_layers") is not None:
                try:
                    num_layers = int(meta.get("num_layers"))
                except Exception:
                    num_layers = None
            if num_layers is None and inferred_layers is not None:
                num_layers = int(inferred_layers)
            if num_layers is None:
                num_layers = 2

            # si no pudimos inferir input_size, avisar y usar 1 (intentar cargar strict=False después)
            if input_size is None or input_size <= 0:
                self._enqueue_log("Warning: no pude inferir input_size del checkpoint o meta. Intentaré reconstruir modelo con input_size=1 y cargar de forma no estricta.")
                input_size = 1

            # Construir instancia del modelo con fibo.LSTM2Head
            try:
                try:
                    model = fibo.LSTM2Head(input_size=int(input_size), hidden_size=int(hidden), num_layers=int(num_layers))
                except Exception as e_build:
                    # intento fallback con num_layers=2 / input_size=1 si la primera falla
                    self._enqueue_log(f"Advertencia: fallo al instanciar LSTM2Head con input={input_size}, hidden={hidden}, layers={num_layers}: {e_build}")
                    try:
                        model = fibo.LSTM2Head(input_size=int(input_size or 1), hidden_size=int(hidden or 64), num_layers=2)
                        self._enqueue_log("Instanciado LSTM2Head con fallback params.")
                    except Exception as e2:
                        self._enqueue_log(f"No se pudo instanciar fibo.LSTM2Head con ningun fallback: {e2}")
                        self._enqueue_log(traceback.format_exc())
                        return
            except Exception:
                self._enqueue_log("Error inesperado al instanciar LSTM2Head.")
                self._enqueue_log(traceback.format_exc())
                return

            # Intentar cargar state_dict en el modelo construido
            try:
                if isinstance(state, dict):
                    try:
                        model.load_state_dict(state)  # intento estricto primero
                        self._enqueue_log("state_dict cargado con strict=True.")
                    except Exception as e_strict:
                        # buscar posibles contenedores alternativos en el checkpoint
                        alt_keys = []
                        for alt in ("state", "model_state", "model_state_dict", "state_dict"):
                            if isinstance(ckpt, dict) and alt in ckpt and isinstance(ckpt[alt], dict):
                                alt_keys.append(alt)
                        loaded = False
                        for k in alt_keys:
                            try:
                                model.load_state_dict(ckpt[k])
                                self._enqueue_log(f"state_dict cargado desde checkpoint['{k}'].")
                                loaded = True
                                break
                            except Exception:
                                pass

                        if not loaded:
                            # cargar non-strict para tolerar cambios de arquitectura (se notificarán keys faltantes/inesperadas)
                            try:
                                res = model.load_state_dict(state, strict=False)
                                # load_state_dict devuelve namedtuple con missing/unexpected en PyTorch
                                missing = getattr(res, "missing_keys", None)
                                unexpected = getattr(res, "unexpected_keys", None)
                                self._enqueue_log("state_dict cargado con strict=False (posible incompatibilidades).")
                                if missing:
                                    self._enqueue_log(f"Missing keys: {missing}")
                                if unexpected:
                                    self._enqueue_log(f"Unexpected keys: {unexpected}")
                            except Exception as e_nonstrict:
                                self._enqueue_log(f"No se pudo cargar state_dict ni en strict=True ni en strict=False: {e_nonstrict}")
                                self._enqueue_log(traceback.format_exc())
                                return
                else:
                    # ckpt no es dict -> puede ser objeto serializado (modelo completo)
                    try:
                        model = state
                        self._enqueue_log("Checkpoint contiene objeto modelo serializado; se usó tal cual.")
                    except Exception:
                        self._enqueue_log("Checkpoint no es state_dict ni objeto serializado reconocido.")
                        return
            except Exception as e_load:
                self._enqueue_log(f"Error cargando state_dict: {e_load}")
                self._enqueue_log(traceback.format_exc())
                return

            # mover modelo a dispositivo disponible
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
                self._enqueue_log(f"Modelo preparado en device {device} (input_size={input_size}, hidden={hidden}, num_layers={num_layers}).")
            except Exception as e_dev:
                self._enqueue_log(f"Fallo al mover modelo a device: {e_dev}")
                self._enqueue_log(traceback.format_exc())

            # Assign to self
            self.model = model
            # enriquecer meta retornado
            meta_out = dict(meta or {})
            meta_out.setdefault("input_size", int(input_size))
            meta_out.setdefault("hidden", int(hidden))
            meta_out.setdefault("num_layers", int(num_layers))
            if "feature_cols" not in meta_out and getattr(self, "feature_cols_used", None):
                meta_out["feature_cols"] = list(self.feature_cols_used)
            self.model_meta = meta_out

            self._enqueue_log(f"Modelo cargado desde {model_path} y metadata aplicada: {meta_out}")

            # cargar scaler si existe
            if scaler_path.exists():
                if joblib is not None:
                    try:
                        scaler = joblib.load(str(scaler_path))
                        self.model_scaler = scaler
                        self._enqueue_log(f"Scaler cargado desde {scaler_path}.")
                    except Exception as e:
                        self._enqueue_log(f"Fallo al cargar scaler.pkl: {e}")
                        self._enqueue_log(traceback.format_exc())
                        self.model_scaler = None
                else:
                    self._enqueue_log("joblib no disponible; no se puede cargar scaler.pkl.")
                    self.model_scaler = None
            else:
                self.model_scaler = None
                self._enqueue_log("No se encontró scaler.pkl en artifacts.")

        except Exception as e_outer:
            self._enqueue_log(f"_load_model_from_artifacts fallo inesperado: {e_outer}")
            try:
                self._enqueue_log(traceback.format_exc())
            except Exception:
                pass


    # --------------------------
    # Internal helpers
    # --------------------------
    def _load_df_for_training(self) -> Optional[pd.DataFrame]:
        p = Path(self.sqlite_path.get())
        if not p.exists():
            self._enqueue_log(f"SQLite not found: {p}")
            return None
        try:
            import sqlite3
            con = sqlite3.connect(str(p))
            q = f"SELECT * FROM {self.table.get()} WHERE symbol = ? AND timeframe = ? ORDER BY ts ASC"
            df = pd.read_sql_query(q, con, params=[self.symbol.get(), self.timeframe.get()])
            con.close()
            if df.empty:
                return df
            # normalize timestamp if necessary
            if "timestamp" not in df.columns and "ts" in df.columns:
                df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
            return df
        except Exception as e:
            self._enqueue_log(f"Load sqlite failed: {e}")
            self._enqueue_log(traceback.format_exc())
            return None

    def _build_sequences_internal(self, df_feats: pd.DataFrame, feature_cols: List[str], seq_len: int, horizon: int, dtype=np.float32):
        df = df_feats.copy().reset_index(drop=True)
        present = [c for c in feature_cols if c in df.columns]
        if len(present) == 0:
            raise RuntimeError("No feature cols present.")
        for c in present:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c]).astype("int64") / 1e9
        # log close
        df["__logc"] = np.log(df["close"].replace(0, np.nan))
        df = df.dropna().reset_index(drop=True)
        M = len(df)
        min_rows = seq_len + horizon
        if M < min_rows:
            return np.zeros((0, seq_len, len(present)), dtype=dtype), np.zeros((0,), dtype=dtype)
        N = M - seq_len - horizon + 1
        F = len(present)
        X = np.zeros((N, seq_len, F), dtype=dtype)
        y = np.zeros((N,), dtype=dtype)
        mat = df[present].astype(dtype).values
        logc = df["__logc"].values
        idx = 0
        for i in range(N):
            X[idx] = mat[i:i+seq_len, :]
            y[idx] = logc[i+seq_len+horizon-1] - logc[i+seq_len-1]
            idx += 1
        return X, y

    # --------------------------
    # Backtest (simple) - placeholder
    # --------------------------
    def _start_backtest(self):
        if self.backtest_thread and self.backtest_thread.is_alive():
            self._enqueue_log("Backtest already running.")
            return
        t = threading.Thread(target=self._backtest_worker, daemon=True)
        self.backtest_thread = t
        t.start()
        self._enqueue_log("Backtest thread started.")

    def _backtest_worker(self):
        try:
            # very simple: run model over last portion and compute simple PnL as in earlier code
            self._enqueue_log("Backtest: loading data/features...")
            df = self._load_df_for_training()
            if df is None or df.empty:
                self._enqueue_log("No data for backtest.")
                return
            if fibo is None or not hasattr(fibo, "add_technical_features"):
                self._enqueue_log("fiboevo not available for features.")
                return
            close = df["close"].astype(float).values
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            vol = df["volume"].astype(float).values if "volume" in df.columns else None
            feats = fibo.add_technical_features(close, high=high, low=low, volume=vol)
            for col in ("timestamp","close"):
                if col in df.columns and col not in feats.columns:
                    feats[col] = df[col].values
            feats = feats.dropna().reset_index(drop=True)
            # use internal builder and then model
            seq_len = int(self.seq_len.get()); horizon = int(self.horizon.get())
            feature_cols = [c for c in feats.columns if c not in ("timestamp","open","high","low","close","volume","symbol","timeframe","exchange")]
            X, y = self._build_sequences_internal(feats, feature_cols, seq_len, horizon, dtype=np.float32)
            if X.shape[0] == 0:
                self._enqueue_log("No sequences for backtest.")
                return
            # load model from artifacts if not in memory
            if self.model is None:
                artifacts_dir = Path("artifacts")
                model_path = artifacts_dir / "model_best.pt"
                if model_path.exists() and fibo is not None:
                    try:
                        model = fibo.LSTM2Head(input_size=len(feature_cols), hidden_size=int(self.hidden.get()))
                        import torch
                        st = torch.load(str(model_path), map_location="cpu")
                        model.load_state_dict(st)
                        model.eval()
                        self.model = model
                        self._enqueue_log("Using artifacts model for backtest.")
                    except Exception as e:
                        self._enqueue_log(f"Could not load artifacts model: {e}")
            model = self.model
            if model is None:
                self._enqueue_log("No model available for backtest.")
                return
            # simple inference (CPU)
            import torch
            device = torch.device("cpu")
            model.to(device)
            model.eval()
            preds = []
            with torch.no_grad():
                for i in range(0, X.shape[0], 256):
                    xb = torch.from_numpy(X[i:i+256]).float().to(device)
                    out_r, out_v = model(xb)
                    preds.extend(out_r.cpu().numpy().ravel().tolist())
            preds = np.array(preds, dtype=np.float32)
            # compute simple PnL using close series from feats
            closes = feats["close"].values
            N = len(preds)
            pos_pct = 0.01
            equity = 10000.0
            pnl_list = []
            trades = 0
            wins = 0
            for i in range(N):
                pr = preds[i]; c0 = closes[i + seq_len - 1]
                if pr > 0.0005:
                    # buy
                    future_idx = i + seq_len - 1 + horizon
                    if future_idx < len(closes):
                        c_future = closes[future_idx]
                        ret = math.log(c_future) - math.log(c0)
                        usd_pnl = equity * pos_pct * (math.exp(ret) - 1.0)
                        pnl_list.append(usd_pnl); trades += 1
                        if usd_pnl > 0: wins += 1
            total_pnl = sum(pnl_list)
            winrate = (wins / trades) if trades>0 else 0.0
            self._enqueue_log(f"Backtest finished: trades={trades}, total_pnl={total_pnl:.2f}, winrate={winrate:.2%}")
            self.bt_text.insert(END, f"Backtest results: trades={trades}, total_pnl={total_pnl:.2f}, winrate={winrate:.2%}\n")
        except Exception as e:
            self._enqueue_log(f"Backtest failed: {e}")
            self._enqueue_log(traceback.format_exc())

    # --------------------------
    # Daemon control (UPDATED)
    # --------------------------
    def _start_daemon(self):
        if TradingDaemon is None:
            self._enqueue_log("TradingDaemon not available.")
            return
        if self.daemon:
            self._enqueue_log("Daemon already running.")
            return
        try:
            feat_cols = None
            if self.feature_cols_manual.get().strip():
                feat_cols = [c.strip() for c in self.feature_cols_manual.get().split(",") if c.strip()]

            # Normalize model_path input: empty string -> None (so TradingDaemon uses its defaults)
            mp_raw = self.model_path_var.get().strip() if getattr(self, "model_path_var", None) is not None else ""
            model_path_arg = mp_raw if mp_raw else None

            # Warn if torch missing but user specified a model file
            if model_path_arg and torch is None:
                self._enqueue_log("Warning: Torch not available in environment. Model load will fail unless torch is installed.")

            # Instantiate daemon WITHOUT auto-loading artifacts to avoid blocking GUI
            self.daemon = TradingDaemon(
                sqlite_path=self.sqlite_path.get(),
                sqlite_table=self.table.get(),
                symbol=self.symbol.get(),
                timeframe=self.timeframe.get(),
                model_path=model_path_arg,   # None or explicit path
                scaler_path=None,
                meta_path=None,
                ledger_path=None,
                exchange_id=None,
                api_key=None,
                api_secret=None,
                paper=True,
                seq_len=self.seq_len.get(),
                feature_cols=feat_cols,
                auto_load_artifacts=False,   # important: avoid blocking constructor
            )
            self.daemon.start_loop()
            self.btn_start.config(state=DISABLED); self.btn_stop.config(state=NORMAL)
            self._enqueue_log("Daemon started.")

            # If model_path was provided, load it in background to avoid UI freeze
            if model_path_arg:
                def _bg_load():
                    self._enqueue_log(f"Background: loading model from {model_path_arg} ...")
                    try:
                        # prefer daemon API for central behavior
                        self.daemon.load_model_and_scaler(model_path=model_path_arg)
                        if self.daemon.model is not None:
                            self._enqueue_log("Model loaded (background).")
                        else:
                            self._enqueue_log("Background load finished: model still None (check logs).")
                    except Exception as e:
                        self._enqueue_log(f"Background model load failed: {e}")
                        self._enqueue_log(traceback.format_exc())
                threading.Thread(target=_bg_load, daemon=True).start()

        except Exception as e:
            self._enqueue_log(f"Start daemon failed: {e}")
            self._enqueue_log(traceback.format_exc())

    def _stop_daemon(self):
        if self.daemon:
            try:
                self.daemon.stop()
            except Exception:
                pass
            self.daemon = None
            self.btn_start.config(state=NORMAL); self.btn_stop.config(state=DISABLED)
            self._enqueue_log("Daemon stopped.")

    # --------------------------
    # Background model load from model_path UI (NEW)
    # --------------------------
    def _load_model_file_background(self):
        """
        Called from Training tab button: loads the model file specified in self.model_path_var
        into the daemon (if running) using daemon.load_model_and_scaler in background; if no daemon,
        calls the local _load_model_from_artifacts (which expects artifacts/model_best.pt).
        """
        mp_raw = self.model_path_var.get().strip() if getattr(self, "model_path_var", None) is not None else ""
        model_path_arg = mp_raw if mp_raw else None

        if model_path_arg is None:
            # No explicit file chosen; fallback to artifacts
            self._enqueue_log("No model file selected in UI. Use 'Load artifacts model' to load artifacts/model_best.pt.")
            return

        # If daemon exists, call its loader in background
        if self.daemon:
            def _bg_load():
                self._enqueue_log(f"Background: loading model {model_path_arg} into daemon...")
                try:
                    self.daemon.load_model_and_scaler(model_path=model_path_arg)
                    if self.daemon.model is not None:
                        self._enqueue_log("Model loaded into daemon (background).")
                    else:
                        self._enqueue_log("Background load finished: model still None (check logs).")
                except Exception as e:
                    self._enqueue_log(f"Background daemon load failed: {e}")
                    self._enqueue_log(traceback.format_exc())
            threading.Thread(target=_bg_load, daemon=True).start()
            return

        # If no daemon, we can attempt to load into GUI memory via existing function (but that function targets artifacts/)
        self._enqueue_log("No daemon running: use 'Load artifacts model' to load artifacts/model_best.pt into the GUI.")
        return

# --------------------------
# Run
# --------------------------
def main():
    root = Tk()
    app = TradingAppExtended(root)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as _e:
        # capture traceback
        _tb = traceback.format_exc()
        # detect tkinter/display related failure (common message contains 'no display name and no $DISPLAY')
        if "no display name and no $display" in _tb.lower() or "tkinter" in _tb.lower() or "tclerror" in _tb.lower():
            # Headless fallback: report availability of fibo and run smoke tests (add_technical_features)
            try:
                # prefer existing 'fibo' variable, fallback to 'fiboevo' name or import
                fibo_mod = globals().get("fibo", None)
                if fibo_mod is None and "fiboevo" in globals():
                    fibo_mod = globals().get("fiboevo")
                if fibo_mod is None:
                    try:
                        import fiboevo as fibo_mod  # type: ignore
                    except Exception:
                        fibo_mod = None

                log = logging.getLogger("trading_gui_extended")
                log.warning("Tkinter / DISPLAY error detected. Entering headless mode.")
                if fibo_mod is None:
                    log.warning("fiboevo module is not available in this environment. Many features will be inactive.")
                else:
                    funcs = [name for name in ("add_technical_features", "create_sequences_from_df", "LSTM2Head") if hasattr(fibo_mod, name)]
                    log.info("fiboevo: detected functions: %s", funcs or "(none)")
                    if hasattr(fibo_mod, "add_technical_features"):
                        try:
                            close = pd.Series([1.0, 1.1, 1.2, 1.25, 1.3])
                            feats = fibo_mod.add_technical_features(close, dropna_after=True)
                            try:
                                s = getattr(feats, "shape", None)
                                log.info("add_technical_features smoke test OK, result shape=%s", s)
                            except Exception:
                                log.info("add_technical_features smoke test OK (no .shape attribute).")
                        except Exception:
                            log.exception("Error during add_technical_features smoke test.")
                log.info("Headless mode completed. To use the GUI run in an environment with a display (set $DISPLAY).")
            except Exception:
                traceback.print_exc()
                raise
        else:
            # Not a display/Tkinter issue — re-raise
            raise
