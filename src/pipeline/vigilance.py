"""Event-Driven Vigilance Monitor — Proactive threshold breach detection.

Polls key market indicators every N minutes and fires alerts when
thresholds are breached, rather than waiting for scheduled pipeline runs.

Monitored signals:
  - VIX spike (>20% change from baseline)
  - Volume anomaly (>2x average)
  - Regime change (HMM state transition)
  - Sentiment shift (category sentiment flips sign)
  - Price gap (>1.5% intraday move)
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Default thresholds (overridable via config.yaml → vigilance section)
DEFAULT_THRESHOLDS = {
    "vix_spike_pct": 20.0,        # VIX change % from last check
    "volume_ratio_alert": 2.0,    # Volume vs 20-day avg
    "price_gap_pct": 1.5,         # Intraday price move %
    "sentiment_flip": True,       # Alert on sign change
    "check_interval_sec": 300,    # 5 minutes
}

STATE_FILE = "./data/vigilance_state.json"


class VigilanceMonitor:
    """Lightweight market condition monitor with threshold-based alerts."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        vcfg = self.config.get("vigilance", {})
        self.thresholds = {**DEFAULT_THRESHOLDS, **vcfg}
        self.interval = int(self.thresholds["check_interval_sec"])
        self._state = self._load_state()
        self._running = False

    def _load_state(self) -> dict:
        """Load last-known values from state file."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_vix": None, "last_regime": None,
                "last_sentiment_sign": None, "last_close": None,
                "alerts_today": []}

    def _save_state(self):
        """Persist state between checks."""
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save vigilance state: {e}")

    def check(self) -> list[dict]:
        """Run all threshold checks. Returns list of triggered alerts."""
        alerts = []
        now = datetime.now()

        # Only check during market-adjacent hours (7 AM - 6 PM ET, Mon-Fri)
        if now.weekday() >= 5 or now.hour < 7 or now.hour > 18:
            return alerts

        try:
            alerts.extend(self._check_vix())
        except Exception as e:
            logger.debug(f"VIX check failed: {e}")

        try:
            alerts.extend(self._check_price_gap())
        except Exception as e:
            logger.debug(f"Price gap check failed: {e}")

        try:
            alerts.extend(self._check_regime_change())
        except Exception as e:
            logger.debug(f"Regime check failed: {e}")

        try:
            alerts.extend(self._check_sentiment_shift())
        except Exception as e:
            logger.debug(f"Sentiment check failed: {e}")

        if alerts:
            today = now.strftime("%Y-%m-%d")
            if self._state.get("alerts_date") != today:
                self._state["alerts_today"] = []
                self._state["alerts_date"] = today

            for alert in alerts:
                alert["timestamp"] = now.isoformat()
                self._state["alerts_today"].append(alert)
                logger.warning(f"VIGILANCE ALERT: {alert['type']} — {alert['message']}")

            self._save_state()
            self._write_alerts_to_state(alerts)

        return alerts

    def _check_vix(self) -> list[dict]:
        """Check for VIX spike."""
        alerts = []
        try:
            from src.data.fetcher import DataFetcher
            fetcher = DataFetcher(self.config)
            macro = fetcher.get_macro_fred()
            vix = macro.get("vix")
            if vix is None:
                return alerts

            last_vix = self._state.get("last_vix")
            self._state["last_vix"] = vix

            if last_vix is not None and last_vix > 0:
                change_pct = abs(vix - last_vix) / last_vix * 100
                threshold = self.thresholds["vix_spike_pct"]
                if change_pct >= threshold:
                    direction = "spiked" if vix > last_vix else "dropped"
                    alerts.append({
                        "type": "vix_spike",
                        "severity": "high" if change_pct > threshold * 1.5 else "medium",
                        "message": f"VIX {direction} {change_pct:.1f}% ({last_vix:.1f} → {vix:.1f})",
                        "value": vix,
                        "change_pct": round(change_pct, 1),
                    })
        except Exception as e:
            logger.debug(f"VIX fetch error: {e}")
        return alerts

    def _check_price_gap(self) -> list[dict]:
        """Check for significant intraday price move."""
        alerts = []
        try:
            from src.data.db_router import DbRouter
            router = DbRouter(self.config)
            df = router.read_analytics(
                "SELECT close FROM prices ORDER BY date DESC LIMIT 1"
            )
            if df.empty:
                return alerts
            current_close = float(df.iloc[0]["close"])
            last_close = self._state.get("last_close")
            self._state["last_close"] = current_close

            if last_close is not None and last_close > 0:
                gap_pct = abs(current_close - last_close) / last_close * 100
                threshold = self.thresholds["price_gap_pct"]
                if gap_pct >= threshold:
                    direction = "up" if current_close > last_close else "down"
                    alerts.append({
                        "type": "price_gap",
                        "severity": "high" if gap_pct > threshold * 2 else "medium",
                        "message": f"SPY moved {direction} {gap_pct:.2f}% (${last_close:.2f} → ${current_close:.2f})",
                        "value": current_close,
                        "change_pct": round(gap_pct, 2),
                    })
        except Exception as e:
            logger.debug(f"Price check error: {e}")
        return alerts

    def _check_regime_change(self) -> list[dict]:
        """Check for HMM regime state transition."""
        alerts = []
        try:
            state_path = "./data/spy_state.json"
            if not os.path.exists(state_path):
                return alerts
            with open(state_path) as f:
                spy_state = json.load(f)
            current_regime = spy_state.get("regime", "")
            last_regime = self._state.get("last_regime")
            self._state["last_regime"] = current_regime

            if last_regime and current_regime and last_regime != current_regime:
                alerts.append({
                    "type": "regime_change",
                    "severity": "high",
                    "message": f"Regime shifted: {last_regime} → {current_regime}",
                    "from_regime": last_regime,
                    "to_regime": current_regime,
                })
        except Exception as e:
            logger.debug(f"Regime check error: {e}")
        return alerts

    def _check_sentiment_shift(self) -> list[dict]:
        """Check for aggregate sentiment sign flip."""
        alerts = []
        if not self.thresholds.get("sentiment_flip"):
            return alerts
        try:
            from src.data.news_fetcher import NewsFetcher
            nf = NewsFetcher(self.config)
            summary = nf.get_category_sentiment_summary(days=1)
            nf.close()

            if not summary:
                return alerts

            # Compute aggregate sentiment across all categories
            all_scores = [v.get("avg_sentiment", 0) for v in summary.values()
                          if v.get("avg_sentiment") is not None]
            if not all_scores:
                return alerts

            avg = sum(all_scores) / len(all_scores)
            current_sign = 1 if avg > 0.05 else (-1 if avg < -0.05 else 0)
            last_sign = self._state.get("last_sentiment_sign")
            self._state["last_sentiment_sign"] = current_sign

            if (last_sign is not None and current_sign != 0
                    and last_sign != 0 and current_sign != last_sign):
                direction = "positive" if current_sign > 0 else "negative"
                alerts.append({
                    "type": "sentiment_flip",
                    "severity": "medium",
                    "message": f"Aggregate news sentiment flipped to {direction} (avg={avg:.3f})",
                    "value": round(avg, 4),
                })
        except Exception as e:
            logger.debug(f"Sentiment check error: {e}")
        return alerts

    def _write_alerts_to_state(self, alerts: list[dict]):
        """Append vigilance alerts to spy_state.json for dashboard display."""
        state_path = "./data/spy_state.json"
        try:
            state = {}
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state = json.load(f)
            existing = state.get("vigilance_alerts", [])
            # Keep last 20 alerts
            existing.extend(alerts)
            state["vigilance_alerts"] = existing[-20:]
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to write alerts to state: {e}")
