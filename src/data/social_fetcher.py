"""Social Sentiment Fetcher — StockTwits SPY sentiment (free, no API key).

Scrapes the StockTwits symbol page for SPY to get bullish/bearish message counts.
Falls back to a neutral score (0.5) if scraping fails.
"""
import json
import logging
import re
import time
from typing import Dict

import requests

logger = logging.getLogger(__name__)

_CACHE: dict = {"data": None, "updated": 0}
_CACHE_TTL = 3600  # 1 hour


def get_stocktwits_sentiment(ticker: str = "SPY") -> Dict[str, float]:
    """Fetch StockTwits bullish/bearish sentiment for a ticker.

    Returns dict with keys:
      st_bullish_pct: float 0-1, fraction of messages that are bullish
      st_bearish_pct: float 0-1, fraction of messages that are bearish
      st_bull_bear_ratio: float, bullish / max(bearish, 1)
      st_message_volume: int, approximate message count
    """
    now = time.time()
    if _CACHE["data"] and (now - _CACHE["updated"]) < _CACHE_TTL:
        return _CACHE["data"]

    result = {
        "st_bullish_pct": 0.5,
        "st_bearish_pct": 0.5,
        "st_bull_bear_ratio": 1.0,
        "st_message_volume": 0,
    }

    try:
        url = f"https://stocktwits.com/symbol/{ticker.upper()}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        # StockTwits embeds sentiment data in JSON script tags
        bull_match = re.search(r'"bullish"\s*:\s*(\d+)', resp.text)
        bear_match = re.search(r'"bearish"\s*:\s*(\d+)', resp.text)
        if bull_match and bear_match:
            bull = int(bull_match.group(1))
            bear = int(bear_match.group(1))
            total = bull + bear
            if total > 0:
                result["st_bullish_pct"] = round(bull / total, 4)
                result["st_bearish_pct"] = round(bear / total, 4)
                result["st_bull_bear_ratio"] = round(bull / max(bear, 1), 3)
                result["st_message_volume"] = total
                logger.info(
                    "StockTwits %s: bull=%.1f%%, bear=%.1f%%, vol=%d",
                    ticker, result["st_bullish_pct"] * 100,
                    result["st_bearish_pct"] * 100, total,
                )

        _CACHE["data"] = result
        _CACHE["updated"] = now

    except Exception as e:
        logger.warning("StockTwits sentiment fetch failed for %s: %s", ticker, e)

    return result
