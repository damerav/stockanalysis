"""4C. Dashboard Data Bridge — Atomic JSON state files for dashboard consumption."""

import os
import json
import tempfile
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DATA_DIR = "./data"


def _atomic_write(filepath: str, data: dict):
    """Write JSON atomically: write to temp → rename to prevent partial reads."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(filepath), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, default=str)
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def write_spy_state(prediction: dict = None, indicators: dict = None,
                    flow_alerts: list = None):
    """Write SPY predictor state for dashboard consumption."""
    state = {
        "updated_at": datetime.now().isoformat(),
        "prediction": prediction or {},
        "indicators": indicators or {},
        "flow_alerts": flow_alerts or [],
    }
    _atomic_write(os.path.join(DATA_DIR, "spy_state.json"), state)


def write_es_state(position: dict = None, signals: list = None,
                   regime: str = "Med", pnl: dict = None, chart_data: dict = None):
    """Write ES strategy state for dashboard consumption."""
    state = {
        "updated_at": datetime.now().isoformat(),
        "position": position or {"status": "FLAT", "lots": 0},
        "signals": signals or [],
        "regime": regime,
        "pnl": pnl or {"daily": 0.0, "unrealized": 0.0},
        "chart_data": chart_data or {},
    }
    _atomic_write(os.path.join(DATA_DIR, "es_state.json"), state)


def read_state(filename: str) -> dict:
    """Read a state JSON file. Returns empty dict if missing."""
    path = os.path.join(DATA_DIR, filename)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
