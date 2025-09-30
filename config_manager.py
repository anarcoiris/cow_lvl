"""
config_manager.py

Gestión centralizada de configuración JSON para fibopkg3.

Características:
- Carga/guarda atómico con backup (.bak)
- Validación opcional por jsonschema (si está instalado) y validación básica en su ausencia
- Soporta placeholders tipo ${ENV:VAR_NAME} para leer secretos desde variables de entorno
- merge_config: combinación profunda (útil para overrides desde CLI o GUI)
- Utilities: init_default_config, watch/save helpers
"""

from __future__ import annotations
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import threading
import datetime
import logging

LOGGER = logging.getLogger(__name__)
_LOCK = threading.RLock()

# Esquema mínimo; es intencionalmente liberal para no romper si quieres ampliar
_MINIMAL_SCHEMA = {
    "type": "object",
    "properties": {
        "daemon": {"type": "object"},
        "db": {"type": "object"},
        "model": {"type": "object"},
        "exchanges": {"type": "object"},
    },
    "required": ["daemon", "db"],
}

try:
    from jsonschema import validate, ValidationError  # type: ignore

    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False
    ValidationError = Exception  # pragma: no cover


def _atomic_write(path: Path, data: str) -> None:
    """
    Escritura atómica: escribe a temp y luego renombra.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        # rename es atómico en la misma FS
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def _apply_env_placeholders(obj: Any) -> Any:
    """
    Sustituye strings del tipo "${ENV:VARNAME}" por el valor de la variable de entorno.
    Funciona recursivamente sobre dicts/listas.
    """
    if isinstance(obj, dict):
        return {k: _apply_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply_env_placeholders(v) for v in obj]
    if isinstance(obj, str):
        # placeholder exacto: ${ENV:NAME}
        if obj.startswith("${ENV:") and obj.endswith("}"):
            var = obj[6:-1]
            return os.environ.get(var, "")
        # allow inline: "foo-${ENV:NAME}"
        import re

        def repl(m):
            name = m.group(1)
            return os.environ.get(name, "")

        return re.sub(r"\$\{ENV:([A-Za-z0-9_]+)\}", repl, obj)
    return obj


def load_config(path: str | Path = "config.json", use_env_placeholders: bool = True) -> Dict[str, Any]:
    """
    Carga config.json, valida mínimamente y sustituye placeholders de entorno si use_env_placeholders=True.
    Lanza FileNotFoundError si no existe.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    cfg = json.loads(text)
    if use_env_placeholders:
        cfg = _apply_env_placeholders(cfg)
    # validación básica
    if _HAS_JSONSCHEMA:
        try:
            validate(cfg, _MINIMAL_SCHEMA)
        except ValidationError as e:
            raise RuntimeError(f"config.json failed schema validation: {e}")
    else:
        # minimal checks
        if not isinstance(cfg, dict):
            raise RuntimeError("config.json must be a JSON object")
        for k in ("daemon", "db"):
            if k not in cfg:
                raise RuntimeError(f"config.json missing required key: {k}")
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path = "config.json", *, backup: bool = True) -> None:
    """
    Guarda config de forma atómica. Si backup=True, hace copia config.json.YYYYMMDD_HHMMSS.bak antes.
    """
    p = Path(path)
    with _LOCK:
        if p.exists() and backup:
            ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            bak = p.with_name(p.name + f".bak.{ts}")
            try:
                shutil.copy2(p, bak)
            except Exception as e:
                LOGGER.debug("Could not backup config (%s): %s", bak, e)
        data = json.dumps(cfg, indent=2, ensure_ascii=False)
        _atomic_write(p, data)
        LOGGER.info("Saved config to %s", p)


def init_default_config(path: str | Path = "config.json", overwrite: bool = False) -> None:
    """
    Crea un config.json de ejemplo si no existe (o sobrescribe si overwrite=True).
    """
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"{p} already exists")
    default = {
        "daemon": {
            "symbol": "BTC/USDT",
            "timeframe": "30m",
            "paper": True,
            "ledger_path": "artifacts/ledger.csv",
            "ws_enabled": True,
            "poll_interval_s": 30,
            "max_position_pct": 0.02
        },
        "db": {
            "sqlite_path": "data/market.sqlite",
            "ohlcv_table": "ohlcv"
        },
        "model": {
            "artifacts_dir": "artifacts",
            "default_model": None,
            "train": {"seq_len": 32, "horizon": 10, "epochs": 20}
        },
        "exchanges": {
            "binance": {"enabled": False, "api_key": "${ENV:BINANCE_KEY}", "api_secret": "${ENV:BINANCE_SECRET}"}
        }
    }
    save_config(default, p, backup=False)


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge profundo: override sobreescribe el base. No modifica objetos de entrada.
    """
    import copy

    out = copy.deepcopy(base)

    def _merge(a, b):
        if not isinstance(b, dict):
            return b
        if not isinstance(a, dict):
            return copy.deepcopy(b)
        res = dict(a)
        for k, v in b.items():
            if k in res:
                res[k] = _merge(res[k], v)
            else:
                res[k] = copy.deepcopy(v)
        return res

    return _merge(out, override)


# Helper rápido para mostrar config path en GUI/CLI
def config_info(path: str | Path = "config.json") -> Dict[str, Any]:
    p = Path(path)
    return {"path": str(p.resolve()), "exists": p.exists(), "modified": p.stat().st_mtime if p.exists() else None}
