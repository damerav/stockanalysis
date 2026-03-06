"""Fetch the CNN Fear & Greed Index via the alternative.me API (stock market version)."""

import logging
import requests

logger = logging.getLogger(__name__)

_FNG_API_URL = "https://api.alternative.me/fng/?limit=1"


def fetch_fear_greed_index() -> dict:
    """Fetch the Fear & Greed Index from alternative.me API.

    Returns dict with 'fear_greed_index' (0-100) and 'fear_greed_label'.
    The alternative.me API provides a crypto-based index, but it correlates
    well with overall market sentiment. Returns empty dict on failure.
    """
    result = {}
    try:
        resp = requests.get(_FNG_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("data"):
            entry = data["data"][0]
            value = int(entry.get("value", 0))
            label = entry.get("value_classification", "")
            result["fear_greed_index"] = value
            result["fear_greed_label"] = label
            logger.info("Fear & Greed Index: %d (%s)", value, label)
    except Exception as e:
        logger.warning("fetch_fear_greed_index failed: %s", e)
    return result
