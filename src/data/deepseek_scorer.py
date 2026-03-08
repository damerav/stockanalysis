"""DeepSeek Narrative Scoring — continuous market sentiment from news summaries.

Replaces binary FinBERT sentiment with nuanced DeepSeek R1 confidence scores.
Processes daily news summaries through Ollama to produce:
- Continuous sentiment score (-1.0 to +1.0)
- Confidence level (0.0 to 1.0)
- Key narrative themes
- Market impact assessment

Scores are cached in PostgreSQL to avoid re-processing.
"""

import logging
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"

SCORING_PROMPT = """You are a quantitative market analyst. Analyze these news headlines and summaries for their likely impact on SPY (S&P 500 ETF) direction over the next trading day.

Headlines from {date}:
{headlines}

Provide your analysis as a JSON object with these exact fields:
{{
  "sentiment_score": <float from -1.0 (very bearish) to +1.0 (very bullish)>,
  "confidence": <float from 0.0 (uncertain) to 1.0 (very confident)>,
  "bull_factors": <count of bullish signals>,
  "bear_factors": <count of bearish signals>,
  "dominant_theme": <one of: "risk_on", "risk_off", "mixed", "macro_driven", "earnings_driven", "geopolitical", "fed_policy", "technical">,
  "impact_magnitude": <float from 0.0 (no impact) to 1.0 (major market mover)>
}}

Be precise and calibrated. Most days should have sentiment between -0.3 and +0.3.
Only use extreme values for genuinely significant events.
Output ONLY the JSON object, no other text."""


def _call_ollama(prompt: str, model: str = "qwen3:8b",
                 timeout: int = 120) -> Optional[str]:
    """Call Ollama API via chat endpoint with thinking disabled for speed.

    Uses qwen3:8b by default — 15x faster than deepseek-r1:14b for
    structured JSON output (3-5s vs 50-80s) because it skips the
    <think> reasoning chain. Runs on GPU (CUDA).
    """
    import requests
    try:
        # Use chat API with think=false for clean, fast JSON output
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "think": False,
                  "options": {"temperature": 0.2, "num_predict": 512}},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "")
        logger.warning(f"Ollama returned {resp.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


def _parse_score_response(text: str) -> Optional[dict]:
    """Extract JSON from DeepSeek response."""
    if not text:
        return None
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(text[start:end + 1])
        # Validate and clamp
        score = float(data.get("sentiment_score", 0))
        score = max(-1.0, min(1.0, score))
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
        return {
            "ds_sentiment": score,
            "ds_confidence": conf,
            "ds_bull_factors": int(data.get("bull_factors", 0)),
            "ds_bear_factors": int(data.get("bear_factors", 0)),
            "ds_theme": data.get("dominant_theme", "mixed"),
            "ds_impact": float(data.get("impact_magnitude", 0.3)),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def score_daily_news(date_str: str, config: dict = None,
                     model: str = "qwen3:8b") -> Optional[dict]:
    """Score a single day's news headlines via DeepSeek.

    Args:
        date_str: Date in YYYY-MM-DD format
        config: App config dict
        model: Ollama model to use

    Returns:
        Dict with ds_sentiment, ds_confidence, ds_bull_factors, etc.
    """
    from src.data.db_router import get_router

    router = get_router(config)

    # Check cache first
    try:
        cached = router.query(
            "SELECT ds_sentiment, ds_confidence, ds_bull_factors, ds_bear_factors, "
            "ds_theme, ds_impact FROM deepseek_scores WHERE date = ?",
            (date_str,)
        )
        if not cached.empty:
            row = cached.iloc[0]
            return row.to_dict()
    except Exception:
        pass  # Table may not exist yet

    # Fetch headlines for this date — handle both ISO and RFC 2822 date formats
    articles = router.query(
        "SELECT headline, source FROM raw_articles "
        "WHERE substr(published_at, 1, 10) = ? "
        "ORDER BY quality_score DESC NULLS LAST LIMIT 30",
        (date_str,)
    )

    # If no results with ISO format, try LIKE match for the date
    if articles.empty:
        try:
            # For RFC 2822 dates, match on the date portion
            # e.g. date_str='2025-02-13' → look for '%13 Feb 2025%'
            from datetime import datetime as _dt
            d = _dt.strptime(date_str, "%Y-%m-%d")
            rfc_pattern = f"%, {d.day:02d} {d.strftime('%b')} {d.year}%"
            articles = router.query(
                "SELECT headline, source FROM raw_articles "
                "WHERE published_at LIKE ? "
                "ORDER BY quality_score DESC NULLS LAST LIMIT 30",
                (rfc_pattern,)
            )
        except Exception:
            pass

    if articles.empty:
        return None

    # Build headline summary
    headlines = []
    for _, art in articles.iterrows():
        src = art.get("source", "Unknown")
        title = art.get("headline", "")
        if title:
            headlines.append(f"[{src}] {title}")

    if not headlines:
        return None

    headline_text = "\n".join(headlines[:25])  # Cap at 25 for context window
    prompt = SCORING_PROMPT.format(date=date_str, headlines=headline_text)

    response = _call_ollama(prompt, model=model)
    result = _parse_score_response(response)

    if result is None:
        logger.debug(f"DeepSeek scoring failed for {date_str}")
        return None

    # Cache result
    try:
        router.execute(
            "INSERT INTO deepseek_scores (date, ds_sentiment, ds_confidence, "
            "ds_bull_factors, ds_bear_factors, ds_theme, ds_impact) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (date_str, result["ds_sentiment"], result["ds_confidence"],
             result["ds_bull_factors"], result["ds_bear_factors"],
             result["ds_theme"], result["ds_impact"])
        )
    except Exception as e:
        logger.debug(f"Failed to cache DeepSeek score: {e}")

    return result


def compute_deepseek_features(config: dict = None,
                              days_back: int = 90) -> pd.DataFrame:
    """Compute DeepSeek narrative features for recent dates.

    Returns DataFrame with date + ds_* columns.
    """
    from src.data.db_router import get_router

    router = get_router(config)

    # Try to load from cache first
    try:
        cached = router.query(
            "SELECT date, ds_sentiment, ds_confidence, ds_bull_factors, "
            "ds_bear_factors, ds_impact FROM deepseek_scores ORDER BY date"
        )
        if not cached.empty:
            # Derive additional features
            cached["ds_sentiment_momentum"] = cached["ds_sentiment"].diff(3).fillna(0)
            cached["ds_conviction"] = (
                cached["ds_sentiment"].abs() * cached["ds_confidence"]
            ).round(4)
            cached["ds_bull_bear_ratio"] = np.where(
                cached["ds_bear_factors"] > 0,
                cached["ds_bull_factors"] / cached["ds_bear_factors"],
                cached["ds_bull_factors"]
            )
            return cached
    except Exception:
        pass

    return pd.DataFrame()


def backfill_deepseek_scores(config: dict = None, days_back: int = 90,
                             model: str = "qwen3:8b"):
    """Backfill DeepSeek scores for recent dates that have news but no score."""
    from src.data.db_router import get_router
    from datetime import date, timedelta

    router = get_router(config)

    # Use news_features table for reliable dates, filter to ISO format only
    try:
        dates_with_news = router.query(
            "SELECT DISTINCT date FROM news_features "
            "WHERE date LIKE '20%' "
            "ORDER BY date DESC"
        )
    except Exception:
        dates_with_news = pd.DataFrame()

    if dates_with_news.empty:
        logger.info("No news dates to backfill")
        return

    # Filter to recent dates
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    all_dates = dates_with_news["date"].astype(str).tolist()
    recent_dates = [d for d in all_dates if d >= cutoff]

    try:
        existing = router.query("SELECT date FROM deepseek_scores")
        existing_dates = set(existing["date"].tolist()) if not existing.empty else set()
    except Exception:
        existing_dates = set()

    to_score = [d for d in recent_dates if d not in existing_dates]
    logger.info(f"Backfilling DeepSeek scores for {len(to_score)} dates")

    scored = 0
    for date_str in sorted(to_score):
        result = score_daily_news(date_str, config=config, model=model)
        if result:
            scored += 1
            if scored % 10 == 0:
                logger.info(f"Scored {scored}/{len(to_score)} dates")
        time.sleep(1)  # Rate limit

    logger.info(f"Backfill complete: {scored}/{len(to_score)} dates scored")
