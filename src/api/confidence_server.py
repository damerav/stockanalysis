"""GAP 2: Real-Time AI Confidence API — /confidence and /exit endpoints.

FastAPI server on DGX Spark providing sub-50ms inference for MT5/FxDreema.

Usage:
    uvicorn src.api.confidence_server:app --host 0.0.0.0 --port 8100
"""

import os
import time
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="ES AI Confidence Engine", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Models loaded once at startup
# ---------------------------------------------------------------------------

_entry_gate = None
_exit_ctrl = None
_regime_detector = None
_audit_log_path = "./logs/trade_audit.jsonl"


def _load_models():
    """Load AI models into memory at startup."""
    global _entry_gate, _exit_ctrl, _regime_detector
    from src.es_strategy.ai_models import ESEntryGate, ESExitController
    from src.es_strategy.indicators import RegimeDetector

    _entry_gate = ESEntryGate()
    if not _entry_gate.load():
        logger.warning("Entry gate model not found — /confidence will return block")
        _entry_gate = None

    _exit_ctrl = ESExitController()
    if not _exit_ctrl.load():
        logger.info("Exit CNN not found — /exit will return defaults")

    _regime_detector = RegimeDetector()
    os.makedirs(os.path.dirname(_audit_log_path), exist_ok=True)
    logger.info("AI models loaded for confidence server")


# ---------------------------------------------------------------------------
# Request / Response schemas (Section 7)
# ---------------------------------------------------------------------------

class ConfidenceRequest(BaseModel):
    """Input for /confidence endpoint."""
    snapshot_1m: dict = Field(..., description="1-minute bar: open, high, low, close, volume")
    l2_metrics: dict = Field(default_factory=dict, description="100ms L2 aggregation metrics")
    strategy_context: dict = Field(default_factory=dict, description="Current strategy state")


class ConfidenceResponse(BaseModel):
    entry_conf: float = Field(..., description="Entry confidence 0-1")
    vol_regime: str = Field(..., description="Low|Mid|High")
    advice: str = Field(..., description="allow|block")
    quantity: int = Field(0, description="Suggested lot count 0-3")
    latency_ms: float = Field(0, description="Inference latency in ms")
    model_version: str = Field("", description="Model file used")
    feature_hash: str = Field("", description="SHA256 of feature vector")


class ExitRequest(BaseModel):
    bar_window: list[list[float]] = Field(..., description="Last 20 bars × N features")
    regime: str = Field("Med", description="Current volatility regime")
    strategy_context: dict = Field(default_factory=dict)


class ExitResponse(BaseModel):
    exit_conf_reversal: float = Field(..., description="Reversal probability 0-1")
    tp2_trail_atr: float = Field(..., description="TP2 trailing stop multiplier × ATR")
    runner_trail_atr: float = Field(..., description="Runner trailing stop multiplier × ATR")
    latency_ms: float = Field(0)
    model_version: str = Field("")


class HealthResponse(BaseModel):
    status: str
    entry_gate_loaded: bool
    exit_ctrl_loaded: bool
    uptime_seconds: float


class SpreadUpdate(BaseModel):
    """GAP 19: Dynamic spread input from broker or manual entry."""
    strike_K: float = Field(..., description="Sold strike price")
    credit_C: float = Field(..., description="Credit width")


# ---------------------------------------------------------------------------
# Audit logging (GAP 15)
# ---------------------------------------------------------------------------

def _audit_log(endpoint: str, request_data: dict, response_data: dict,
               feature_hash: str = "", latency_ms: float = 0):
    """Append structured audit record to JSONL file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "entry_conf": response_data.get("entry_conf"),
        "exit_conf": response_data.get("exit_conf_reversal"),
        "advice": response_data.get("advice"),
        "vol_regime": response_data.get("vol_regime"),
        "model_version": response_data.get("model_version", ""),
        "feature_hash": feature_hash,
        "latency_ms": latency_ms,
    }
    try:
        with open(_audit_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Audit log write failed: {e}")


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def _build_features_from_snapshot(snapshot: dict, l2: dict, ctx: dict) -> np.ndarray:
    """Build the 17-feature vector from API input."""
    price = snapshot.get("close", 0)
    high = snapshot.get("high", price)
    low = snapshot.get("low", price)
    volume = snapshot.get("volume", 0)

    kc_mid = ctx.get("kc_mid", price)
    kc_upper = ctx.get("kc_upper", price + 5)
    kc_lower = ctx.get("kc_lower", price - 5)
    vwap = ctx.get("vwap", price)
    atr = ctx.get("atr_14", 1.0) or 1.0
    rsi = ctx.get("rsi_14", 50)
    roc_3 = ctx.get("roc_3", 0)
    daily_pnl = ctx.get("daily_pnl", 0)

    # L2 features (use defaults if not provided)
    imbalance = l2.get("imbalance", 0)
    depth_ratio = l2.get("depth_ratio", 1.0)

    features = np.array([
        (price - kc_mid) / atr,                          # price_vs_kc_mid
        (price - vwap) / atr,                             # price_vs_vwap
        rsi,                                               # rsi
        roc_3,                                             # roc_3
        ctx.get("atr_regime_pct", 0.5),                   # atr_regime_pct
        volume / 1000.0,                                   # volume_ratio
        (kc_upper - kc_lower) / atr if atr else 0,       # kc_width
        ctx.get("ema9_slope", 0),                         # ema9_slope
        ctx.get("macd_hist", 0),                          # macd_hist
        ctx.get("bb_width", 0),                           # bb_width
        ctx.get("momentum_3bar", 0),                      # momentum_3bar
        ctx.get("momentum_5bar", 0),                      # momentum_5bar
        ctx.get("bars_since_trade", 0),                   # bars_since_trade
        daily_pnl / 1000.0,                               # daily_pnl normalised
        ctx.get("time_sin", 0),                           # time_sin
        ctx.get("time_cos", 0),                           # time_cos
        imbalance,                                         # spread_vs_atr / imbalance
    ], dtype=np.float64)
    return features


def _feature_hash(features: np.ndarray) -> str:
    """SHA256 hash of feature vector for audit trail."""
    return hashlib.sha256(features.tobytes()).hexdigest()[:16]


def _get_model_version() -> str:
    model_dir = "./models"
    try:
        files = sorted([f for f in os.listdir(model_dir) if f.startswith("es_entry")], reverse=True)
        return files[0] if files else "none"
    except Exception:
        return "none"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_start_time = time.time()


@app.on_event("startup")
def startup():
    _load_models()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        entry_gate_loaded=_entry_gate is not None and _entry_gate.model is not None,
        exit_ctrl_loaded=_exit_ctrl is not None and _exit_ctrl.model is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/confidence", response_model=ConfidenceResponse)
def confidence(req: ConfidenceRequest):
    """Entry gate inference. Returns entry_conf, vol_regime, advice."""
    t0 = time.perf_counter()

    features = _build_features_from_snapshot(req.snapshot_1m, req.l2_metrics, req.strategy_context)
    fhash = _feature_hash(features)

    # Determine regime
    regime = req.strategy_context.get("vol_regime", "Med")

    # Get configurable threshold (GAP 3: default 0.70 per spec)
    threshold = req.strategy_context.get("entry_threshold", 0.70)

    if _entry_gate is None or _entry_gate.model is None:
        # Fail-closed (GAP 16): block if AI unavailable
        latency = (time.perf_counter() - t0) * 1000
        resp = ConfidenceResponse(
            entry_conf=0.0, vol_regime=regime, advice="block",
            quantity=0, latency_ms=round(latency, 2),
            model_version="none", feature_hash=fhash,
        )
        _audit_log("/confidence", req.dict(), resp.dict(), fhash, latency)
        return resp

    result = _entry_gate.predict(features, regime)
    p_enter = result.get("p_enter", 0)
    advice = "allow" if p_enter >= threshold else "block"
    quantity = result.get("quantity", 0) if advice == "allow" else 0

    latency = (time.perf_counter() - t0) * 1000
    model_ver = _get_model_version()

    resp = ConfidenceResponse(
        entry_conf=round(p_enter, 4),
        vol_regime=regime,
        advice=advice,
        quantity=quantity,
        latency_ms=round(latency, 2),
        model_version=model_ver,
        feature_hash=fhash,
    )
    _audit_log("/confidence", req.dict(), resp.dict(), fhash, latency)
    return resp


@app.post("/exit", response_model=ExitResponse)
def exit_signal(req: ExitRequest):
    """Exit controller inference. Returns trail multipliers."""
    t0 = time.perf_counter()

    regime = req.regime
    bar_window = np.array(req.bar_window, dtype=np.float64)

    if _exit_ctrl is None:
        # Return defaults
        from src.es_strategy.ai_models import ESExitController
        defaults = ESExitController().predict(np.zeros((20, 19)), regime)
        latency = (time.perf_counter() - t0) * 1000
        resp = ExitResponse(
            exit_conf_reversal=defaults.get("p_cont_5", 0.5),
            tp2_trail_atr=defaults.get("tp2_trail", 1.25),
            runner_trail_atr=defaults.get("runner_trail", 2.0),
            latency_ms=round(latency, 2), model_version="defaults",
        )
        _audit_log("/exit", req.dict(), resp.dict(), latency_ms=latency)
        return resp

    result = _exit_ctrl.predict(bar_window, regime)
    latency = (time.perf_counter() - t0) * 1000

    resp = ExitResponse(
        exit_conf_reversal=round(1.0 - result.get("p_cont_5", 0.5), 4),
        tp2_trail_atr=result.get("tp2_trail", 1.25),
        runner_trail_atr=result.get("runner_trail", 2.0),
        latency_ms=round(latency, 2),
        model_version="es_exit_cnn.pt" if _exit_ctrl.model else "defaults",
    )
    _audit_log("/exit", req.dict(), resp.dict(), latency_ms=latency)
    return resp


@app.post("/spread")
def update_spread(req: SpreadUpdate):
    """GAP 19: Update spread inputs dynamically from broker or manual entry."""
    if not hasattr(app, "_engine"):
        app._engine = None
    if app._engine is None:
        # Store for later use — engine will pick up on next restart
        app._spread_K = req.strike_K
        app._spread_C = req.credit_C
        logger.info(f"Spread stored: K={req.strike_K}, C={req.credit_C}")
        return {"status": "stored", "strike_K": req.strike_K, "credit_C": req.credit_C}

    ok = app._engine.update_spread(req.strike_K, req.credit_C)
    if not ok:
        raise HTTPException(status_code=409, detail="Cannot update spread while position is open")
    return {"status": "updated", "strike_K": req.strike_K, "credit_C": req.credit_C}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8100)
