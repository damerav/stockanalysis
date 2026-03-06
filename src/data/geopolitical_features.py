"""Geopolitical & Macro-Shock Feature Engineering.

Computes daily features from news headlines that capture:
1. Geopolitical risk score (war, sanctions, conflict keywords)
2. Oil shock indicator (crude price spike detection)
3. Flight-to-safety signal (gold + bond momentum)
4. FinBERT financial sentiment (more accurate than VADER for finance)
"""

import logging
import os
import re
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_date_str(pub_at: str) -> str:
    """Parse published_at to 'YYYY-MM-DD' from either ISO or RFC 2822 format."""
    if not pub_at:
        return ""
    # ISO format: starts with 4-digit year
    if pub_at[:4].isdigit():
        return pub_at[:10]
    # RFC 2822 format: "Fri, 01 Aug 2025 07:00:00 GMT"
    try:
        dt = parsedate_to_datetime(pub_at)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

# --- Geopolitical risk keyword lexicon (weighted) ---
# Higher weight = stronger geopolitical signal
GEO_KEYWORDS = {
    # War / military
    "war": 3.0, "military strike": 3.0, "airstrike": 3.0, "missile": 2.5,
    "invasion": 3.0, "troops deployed": 2.5, "bombing": 2.5, "attack": 2.0,
    "conflict": 2.0, "escalation": 2.0, "retaliation": 2.0, "casualties": 2.0,
    # Geopolitical hotspots
    "iran": 1.5, "strait of hormuz": 3.0, "middle east": 1.0, "taiwan": 1.5,
    "north korea": 1.5, "ukraine": 1.0, "russia": 1.0, "china": 0.5,
    # Sanctions / trade
    "sanctions": 2.0, "embargo": 2.5, "trade war": 2.0, "tariff": 1.5,
    "export ban": 2.0, "blockade": 2.5,
    # Energy disruption
    "oil supply disruption": 3.0, "pipeline": 1.0, "opec": 1.0,
    "energy crisis": 2.5, "oil surge": 2.0, "crude spike": 2.0,
    # Nuclear
    "nuclear": 2.0, "enrichment": 1.5, "weapons": 1.5,
    # Financial contagion
    "default": 1.5, "debt crisis": 2.0, "bank run": 2.5, "contagion": 2.0,
    "systemic risk": 2.0, "credit crunch": 2.0,
}

# Fear/panic keywords (market-specific)
FEAR_KEYWORDS = {
    "crash": 2.5, "plunge": 2.0, "tumble": 1.5, "selloff": 1.5, "sell-off": 1.5,
    "panic": 2.0, "fear": 1.0, "risk-off": 1.5, "flight to safety": 2.0,
    "bear market": 2.0, "correction": 1.0, "recession": 1.5, "stagflation": 2.0,
    "black swan": 2.5, "circuit breaker": 3.0, "margin call": 2.0,
}

# Positive/recovery keywords
RECOVERY_KEYWORDS = {
    "ceasefire": 2.0, "peace talks": 1.5, "de-escalation": 2.0, "truce": 2.0,
    "agreement": 1.0, "deal": 0.5, "rally": 1.0, "recovery": 1.0,
    "stimulus": 1.5, "rate cut": 1.5, "easing": 1.0, "dovish": 1.0,
}


def compute_geopolitical_score(text: str) -> dict:
    """Score a single text for geopolitical risk, fear, and recovery signals.

    Returns dict with geo_risk, fear_score, recovery_score (all 0-10 scale).
    """
    text_lower = text.lower()

    geo_risk = 0.0
    for kw, weight in GEO_KEYWORDS.items():
        if kw in text_lower:
            geo_risk += weight

    fear = 0.0
    for kw, weight in FEAR_KEYWORDS.items():
        if kw in text_lower:
            fear += weight

    recovery = 0.0
    for kw, weight in RECOVERY_KEYWORDS.items():
        if kw in text_lower:
            recovery += weight

    # Normalize to 0-10 scale (cap at 10)
    return {
        "geo_risk": min(geo_risk, 10.0),
        "fear_score": min(fear, 10.0),
        "recovery_score": min(recovery, 10.0),
    }


def compute_daily_geopolitical_features(config: dict = None) -> pd.DataFrame:
    """Compute daily geopolitical features from news articles.

    Routes through PostgreSQL (primary) with SQLite news.db fallback.
    Returns DataFrame with columns: date, geo_risk_score, geo_fear_score,
    geo_recovery_score, geo_net_risk, geo_article_ratio, geo_max_risk
    """
    # Query articles from PostgreSQL via router
    try:
        from src.data.db_router import get_router
        router = get_router(config)
        cutoff = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        df_articles = router.query(
            f"SELECT published_at, headline, summary FROM raw_articles "
            f"WHERE published_at >= '{cutoff}'"
        )
        if not df_articles.empty:
            rows = list(df_articles.itertuples(index=False, name=None))
        else:
            rows = []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame()

    records = []
    for pub_at, headline, summary in rows:
        text = f"{headline or ''} {summary or ''}"
        date_str = _parse_date_str(pub_at or "")
        if not date_str:
            continue
        scores = compute_geopolitical_score(text)
        records.append({"date": date_str, **scores})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Aggregate per day
    daily = df.groupby("date").agg(
        geo_risk_score=("geo_risk", "mean"),
        geo_fear_score=("fear_score", "mean"),
        geo_recovery_score=("recovery_score", "mean"),
        geo_max_risk=("geo_risk", "max"),
        _total=("geo_risk", "count"),
        _geo_articles=("geo_risk", lambda x: (x > 0).sum()),
    ).reset_index()

    # Ratio of articles with any geopolitical content
    daily["geo_article_ratio"] = daily["_geo_articles"] / daily["_total"].replace(0, 1)
    # Net risk = risk - recovery (positive = more risk)
    daily["geo_net_risk"] = daily["geo_risk_score"] - daily["geo_recovery_score"]

    daily = daily.drop(columns=["_total", "_geo_articles"])
    return daily


def compute_finbert_sentiment(texts: list[str], batch_size: int = 32) -> list[dict]:
    """Compute FinBERT sentiment for a list of texts.

    Returns list of dicts with keys: label, score, positive, negative, neutral.
    Falls back to empty results if FinBERT unavailable.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        logger.warning("transformers not installed — skipping FinBERT")
        return [{"label": "neutral", "score": 0.0, "positive": 0.0,
                 "negative": 0.0, "neutral": 1.0} for _ in texts]

    try:
        pipe = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU (DGX can use 0 for GPU)
            truncation=True,
            max_length=512,
        )
    except Exception as e:
        logger.warning(f"FinBERT load failed: {e}")
        return [{"label": "neutral", "score": 0.0, "positive": 0.0,
                 "negative": 0.0, "neutral": 1.0} for _ in texts]

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Truncate long texts
        batch = [t[:512] if t else "neutral" for t in batch]
        try:
            preds = pipe(batch)
            for p in preds:
                label = p["label"].lower()
                score = p["score"]
                results.append({
                    "label": label,
                    "score": score,
                    "positive": score if label == "positive" else 0.0,
                    "negative": score if label == "negative" else 0.0,
                    "neutral": score if label == "neutral" else 0.0,
                })
        except Exception as e:
            logger.warning(f"FinBERT batch failed: {e}")
            results.extend([{"label": "neutral", "score": 0.0, "positive": 0.0,
                             "negative": 0.0, "neutral": 1.0} for _ in batch])

    return results


def compute_daily_finbert_features(config: dict = None, days_back: int = 7) -> pd.DataFrame:
    """Compute daily FinBERT sentiment features from recent news articles.

    Uses the shared finbert_cache_utils for cache-aware scoring.
    Cache key is url_hash (SHA-256 of headline+summary[:300]) — works for
    both raw_articles and news tables.

    Returns DataFrame with: date, finbert_positive, finbert_negative,
    finbert_neutral, finbert_score (positive - negative)
    """
    cutoff_dt = datetime.now() - timedelta(days=days_back)
    cutoff_iso = cutoff_dt.strftime("%Y-%m-%d")

    from src.data.db_router import get_router
    from src.data.finbert_cache_utils import score_articles_with_cache
    router = get_router(config)

    logger.info(f"Computing daily FinBERT features for {cutoff_iso} …")

    # Fetch recent articles — use ISO-format filter + ORDER BY id DESC with LIMIT
    # to avoid pulling the entire table. The published_at column has mixed formats
    # (ISO + RFC 2822), so we also filter in Python after parsing.
    try:
        articles_df = router.query(
            "SELECT id, published_at, headline, summary "
            "FROM raw_articles "
            f"WHERE published_at >= '{cutoff_iso}' "
            "ORDER BY id DESC "
            "LIMIT 5000"
        )
    except Exception as e:
        logger.warning(f"raw_articles query failed: {e}")
        return pd.DataFrame()

    if articles_df.empty:
        return pd.DataFrame()

    # Python-side date filter to handle RFC 2822 dates that slipped through SQL
    articles = []
    for _, row in articles_df.iterrows():
        date_str = _parse_date_str(row["published_at"] or "")
        if not date_str:
            continue
        # Only keep articles within the actual date range
        if date_str < cutoff_iso:
            continue
        articles.append({
            "id": row["id"],
            "headline": row["headline"] or "",
            "summary": row["summary"] or "",
            "published_at": row["published_at"],
            "_date": date_str,
        })

    if not articles:
        return pd.DataFrame()

    logger.info(f"FinBERT geopolitical: {len(articles)} articles in last {days_back} days")

    # Load FinBERT pipeline
    finbert_pipe = None
    try:
        from transformers import pipeline as hf_pipeline
        finbert_pipe = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU; change to 0 for GPU
            truncation=True,
            max_length=512,
        )
    except Exception as e:
        logger.warning(f"FinBERT load failed in geopolitical_features: {e}")

    # Score with cache — only uncached articles will hit the model
    scored = score_articles_with_cache(
        router=router,
        articles=articles,
        finbert_pipeline=finbert_pipe,
        batch_size=32,
    )

    # Reconstruct DataFrame with dates
    records = []
    for article, score in zip(articles, scored):
        records.append({
            "date": article["_date"],
            "fb_positive": score["fb_positive"],
            "fb_negative": score["fb_negative"],
            "fb_neutral": score["fb_neutral"],
        })

    if not records:
        return pd.DataFrame()

    result_df = pd.DataFrame(records)
    daily = result_df.groupby("date").agg(
        finbert_positive=("fb_positive", "mean"),
        finbert_negative=("fb_negative", "mean"),
        finbert_neutral=("fb_neutral", "mean"),
    ).reset_index()
    daily["finbert_score"] = daily["finbert_positive"] - daily["finbert_negative"]
    return daily



def compute_oil_shock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute oil price shock features from the main feature DataFrame.

    Expects df to have 'crude' column. Adds:
    - crude_pct_change: daily % change in crude
    - crude_vs_ma20: crude deviation from 20-day MA
    - crude_shock: binary flag for >3% daily move
    - crude_momentum_5d: 5-day crude momentum
    """
    if "crude" not in df.columns:
        df["crude_pct_change"] = 0.0
        df["crude_vs_ma20"] = 0.0
        df["crude_shock"] = 0
        df["crude_momentum_5d"] = 0.0
        return df

    crude = df["crude"].ffill()
    df["crude_pct_change"] = crude.pct_change()
    crude_ma20 = crude.rolling(20, min_periods=5).mean()
    df["crude_vs_ma20"] = (crude - crude_ma20) / crude_ma20.replace(0, np.nan)
    df["crude_shock"] = (df["crude_pct_change"].abs() > 0.03).astype(int)
    df["crude_momentum_5d"] = crude.pct_change(5)

    return df


def compute_flight_to_safety_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute flight-to-safety features from gold and bond data.

    Expects df to have 'gold' and 'us10y_yield' columns. Adds:
    - gold_momentum_5d: 5-day gold price momentum
    - gold_vs_ma20: gold deviation from 20-day MA
    - yield_change_5d: 5-day change in 10Y yield
    - safety_signal: composite flight-to-safety indicator
    """
    if "gold" in df.columns:
        gold = df["gold"].ffill()
        df["gold_momentum_5d"] = gold.pct_change(5)
        gold_ma20 = gold.rolling(20, min_periods=5).mean()
        df["gold_vs_ma20"] = (gold - gold_ma20) / gold_ma20.replace(0, np.nan)
    else:
        df["gold_momentum_5d"] = 0.0
        df["gold_vs_ma20"] = 0.0

    if "us10y_yield" in df.columns:
        df["yield_change_5d"] = df["us10y_yield"].diff(5)
    else:
        df["yield_change_5d"] = 0.0

    # Composite safety signal: gold up + yields down = flight to safety
    df["safety_signal"] = df["gold_momentum_5d"].fillna(0) - df["yield_change_5d"].fillna(0) * 0.1

    return df
