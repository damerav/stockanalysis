"""5A. Cloud Publisher — Pushes state from DGX to AWS relay."""

import time
import json
import logging
import threading
import requests
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds
HEARTBEAT_INTERVAL = 30  # seconds


class CloudPublisher:
    """Publishes DGX state to AWS relay server via HTTPS POST."""

    def __init__(self, config: dict = None):
        config = config or {}
        sync_cfg = config.get("sync", {})
        self.enabled = sync_cfg.get("enabled", False)
        self.relay_url = sync_cfg.get("relay_url", "").rstrip("/")
        self.api_key = sync_cfg.get("api_key", "")
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def _post(self, endpoint: str, data: dict) -> bool:
        """POST with retry logic. Returns True on success."""
        if not self.enabled or not self.relay_url:
            return True  # no-op if disabled

        url = f"{self.relay_url}{endpoint}"
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(url, json=data, headers=self._headers(), timeout=10)
                if resp.status_code in (200, 201):
                    return True
                logger.warning(f"Relay POST {endpoint} returned {resp.status_code}")
            except requests.RequestException as e:
                wait = RETRY_BACKOFF * (attempt + 1)
                logger.warning(f"Relay POST {endpoint} failed (attempt {attempt+1}): {e}, retry in {wait}s")
                time.sleep(wait)
        logger.error(f"All {MAX_RETRIES} attempts failed for {endpoint}")
        return False

    def push_prediction(self, prediction: dict):
        self._post("/push/prediction", prediction)

    def push_flow_alert(self, alert: dict):
        self._post("/push/flow_alert", alert)

    def push_premarket(self, summary: dict):
        self._post("/push/premarket", summary)

    def push_es_state(self, state: dict):
        self._post("/push/es_state", state)

    def push_heartbeat(self):
        self._post("/push/heartbeat", {"status": "alive", "timestamp": time.time()})

    def start_heartbeat(self):
        """Start background heartbeat thread."""
        if not self.enabled:
            return
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        logger.info("Cloud publisher heartbeat started")

    def stop(self):
        self._running = False

    def _heartbeat_loop(self):
        while self._running:
            self.push_heartbeat()
            time.sleep(HEARTBEAT_INTERVAL)
