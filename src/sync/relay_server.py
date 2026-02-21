"""5B. FastAPI Relay Server — In-memory state relay for cloud dashboards."""

import os
import time
import asyncio
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

logger = logging.getLogger(__name__)

app = FastAPI(title="SPY/ES Relay Server", version="1.0.0")

# CORS for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory state ---
state = {
    "prediction": {},
    "indicators": {},
    "flow_alerts": [],
    "premarket": {},
    "es_state": {},
    "last_heartbeat": 0,
    "source_status": "UNKNOWN",
}

API_KEY = os.environ.get("RELAY_API_KEY", "changeme")
ADMIN_KEY = os.environ.get("RELAY_ADMIN_KEY", "admin-changeme")
STALE_TIMEOUT = 90  # seconds


def verify_api_key(request: Request):
    """Verify X-API-Key header."""
    key = request.headers.get("X-API-Key", "")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def verify_admin_key(request: Request):
    key = request.headers.get("X-API-Key", "")
    if key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")


def _check_stale():
    """Update source status based on heartbeat."""
    if state["last_heartbeat"] == 0:
        state["source_status"] = "UNKNOWN"
    elif time.time() - state["last_heartbeat"] > STALE_TIMEOUT:
        state["source_status"] = "OFFLINE"
    else:
        state["source_status"] = "ONLINE"


# --- Push endpoints (DGX → Relay) ---

@app.post("/push/prediction")
async def push_prediction(data: dict, _=Depends(verify_api_key)):
    state["prediction"] = data
    state["prediction"]["received_at"] = datetime.now().isoformat()
    return {"status": "ok"}


@app.post("/push/flow_alert")
async def push_flow_alert(data: dict, _=Depends(verify_api_key)):
    state["flow_alerts"].append(data)
    state["flow_alerts"] = state["flow_alerts"][-100:]  # keep last 100
    return {"status": "ok"}


@app.post("/push/premarket")
async def push_premarket(data: dict, _=Depends(verify_api_key)):
    state["premarket"] = data
    return {"status": "ok"}


@app.post("/push/es_state")
async def push_es_state(data: dict, _=Depends(verify_api_key)):
    state["es_state"] = data
    state["es_state"]["received_at"] = datetime.now().isoformat()
    return {"status": "ok"}


@app.post("/push/heartbeat")
async def push_heartbeat(data: dict, _=Depends(verify_api_key)):
    state["last_heartbeat"] = time.time()
    state["source_status"] = "ONLINE"
    return {"status": "ok"}


# --- GET endpoints (Cloud dashboards → Relay) ---

@app.get("/state")
async def get_state():
    """Full SPY state for cloud dashboard."""
    _check_stale()
    return {
        "prediction": state["prediction"],
        "indicators": state.get("indicators", {}),
        "flow_alerts": state["flow_alerts"][-20:],
        "premarket": state["premarket"],
        "source_status": state["source_status"],
        "updated_at": state["prediction"].get("received_at", ""),
    }


@app.get("/state/es")
async def get_es_state():
    """ES strategy state for cloud dashboard."""
    _check_stale()
    result = dict(state["es_state"])
    result["source_status"] = state["source_status"]
    return result


@app.get("/stream")
async def stream_events():
    """Server-Sent Events for real-time push."""
    async def event_generator():
        last_pred = ""
        last_es = ""
        while True:
            _check_stale()
            pred_str = str(state["prediction"])
            es_str = str(state["es_state"])

            if pred_str != last_pred:
                import json
                yield f"event: prediction\ndata: {json.dumps(state['prediction'])}\n\n"
                last_pred = pred_str

            if es_str != last_es:
                import json
                yield f"event: es_state\ndata: {json.dumps(state['es_state'])}\n\n"
                last_es = es_str

            yield f"event: heartbeat\ndata: {json.dumps({'source': state['source_status']})}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    _check_stale()
    return {
        "status": "healthy",
        "source_status": state["source_status"],
        "last_heartbeat": state["last_heartbeat"],
        "uptime_source": time.time() - state["last_heartbeat"] if state["last_heartbeat"] else None,
    }


@app.post("/admin/reset")
async def admin_reset(_=Depends(verify_admin_key)):
    state["prediction"] = {}
    state["indicators"] = {}
    state["flow_alerts"] = []
    state["premarket"] = {}
    state["es_state"] = {}
    state["last_heartbeat"] = 0
    state["source_status"] = "UNKNOWN"
    return {"status": "reset"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
