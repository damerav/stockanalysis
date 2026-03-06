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
                    flow_alerts: list = None, enhanced_prediction: dict = None):
    """Write SPY predictor state for dashboard consumption."""
    state = {
        "updated_at": datetime.now().isoformat(),
        "prediction": prediction or {},
        "indicators": indicators or {},
        "flow_alerts": flow_alerts or [],
        "enhanced_prediction": enhanced_prediction or {},
    }
    _atomic_write(os.path.join(DATA_DIR, "spy_state.json"), state)


def compute_enhanced_prediction(prediction: dict, flow_alerts: list) -> dict:
    """Compute Enhanced Prediction by fusing model prediction with institutional flow.

    Scoring:
    - Model Score = confidence × direction_sign (+1 bull, -1 bear, 0 neutral)
    - Flow Score = (CALL - PUT notional) / total × 100 from last 20 alerts
    - Enhanced Score = (Model × 0.65) + (Flow × 0.35)
    """
    if not prediction:
        return {"enhanced_direction": "NEUTRAL", "enhanced_score": 0.0,
                "flow_score": 0.0, "flow_alert_count": 0,
                "model_score": 0.0, "alignment": "NO_DATA"}

    scale_label = prediction.get("scale_label", "NEUTRAL")
    confidence = float(prediction.get("confidence", 0))
    if "BULLISH" in scale_label:
        direction_sign = 1.0
    elif "BEARISH" in scale_label:
        direction_sign = -1.0
    else:
        direction_sign = 0.0
    model_score = confidence * direction_sign

    recent_alerts = (flow_alerts or [])[-20:]
    call_notional = sum(float(a.get("notional", 0)) for a in recent_alerts
                        if a.get("direction", "").upper() == "CALL")
    put_notional = sum(float(a.get("notional", 0)) for a in recent_alerts
                       if a.get("direction", "").upper() == "PUT")
    total_notional = call_notional + put_notional
    flow_score = ((call_notional - put_notional) / total_notional) * 100.0 if total_notional > 0 else 0.0

    enhanced_score = (model_score * 0.65) + (flow_score * 0.35)

    model_bull = direction_sign > 0
    model_bear = direction_sign < 0
    flow_bull = flow_score > 10
    flow_bear = flow_score < -10
    signals_conflict = (model_bull and flow_bear) or (model_bear and flow_bull)

    if signals_conflict and abs(enhanced_score) > 10:
        enhanced_direction = "CONFLICTED"
        alignment = "CONFLICTED"
    elif enhanced_score > 40:
        enhanced_direction = "BULLISH"
        alignment = "ALIGNED_BULLISH"
    elif enhanced_score > 10:
        enhanced_direction = "LEAN BULLISH"
        alignment = "LEAN_BULLISH"
    elif enhanced_score < -40:
        enhanced_direction = "BEARISH"
        alignment = "ALIGNED_BEARISH"
    elif enhanced_score < -10:
        enhanced_direction = "LEAN BEARISH"
        alignment = "LEAN_BEARISH"
    else:
        enhanced_direction = "NEUTRAL"
        alignment = "NEUTRAL"

    return {"enhanced_direction": enhanced_direction,
            "enhanced_score": round(enhanced_score, 2),
            "flow_score": round(flow_score, 2),
            "flow_alert_count": len(recent_alerts),
            "model_score": round(model_score, 2),
            "alignment": alignment}


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
