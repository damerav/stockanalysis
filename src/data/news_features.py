"""News Feature Processor — NLP pipeline for news-driven prediction.

Computes TF-IDF vectors, VADER sentiment, and n-gram features from raw articles.
Inspired by Finance-And-ML/US-Stock-Prediction-Using-ML-And-Spark.
"""

import os
import re
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# VADER sentiment — use nltk if available, else simple keyword fallback
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False
    _vader = None


def _simple_sentiment(text: str) -> float:
    """Keyword-based sentiment fallback when VADER unavailable."""
    pos = ["surge", "rally", "gain", "bull", "up", "rise", "profit", "beat", "strong"]
    neg = ["crash", "drop", "fall", "bear", "down", "loss", "miss", "weak", "fear"]
    text_lower = text.lower()
    p = sum(1 for w in pos if w in text_lower)
    n = sum(1 for w in neg if w in text_lower)
    total = p + n
    if total == 0:
        return 0.0
    return (p - n) / total


class NewsFeatureProcessor:
    """Processes raw news articles into ML-ready features."""

    def __init__(self, config: dict = None, max_features: int = 5000):
        import yaml
        if config is None:
            with open("config.yaml") as f:
                config = yaml.safe_load(f) or {}
        self.config = config
        self.max_features = config.get("news_pipeline", {}).get("tfidf_max_features", max_features)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        self._fitted = False
        from src.data.db_router import get_router
        self.router = get_router(config)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize article text."""
        if not text:
            return ""
        text = re.sub(r"<[^>]+>", "", text)  # strip HTML
        text = re.sub(r"http\S+", "", text)   # strip URLs
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def _get_sentiment(self, text: str) -> dict:
        """Get VADER sentiment scores (or fallback)."""
        if _HAS_VADER and _vader:
            scores = _vader.polarity_scores(text)
            return {
                "compound": scores["compound"],
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
            }
        compound = _simple_sentiment(text)
        return {
            "compound": compound,
            "positive": max(0, compound),
            "negative": abs(min(0, compound)),
            "neutral": 1.0 - abs(compound),
        }

    def process_articles(self, articles: list[dict] = None, limit: int = 10000) -> pd.DataFrame:
        """Process raw articles into feature DataFrame.

        Uses FinBERT scores from cache if available, falling back to VADER.

        Args:
            articles: Pre-fetched article dicts. If None, queries raw_articles.
            limit: Max articles to load when querying DB (default 10000).

        Returns DataFrame with columns: date, ticker, headline, sentiment_compound,
        sentiment_positive, sentiment_negative, clean_text
        """
        if articles is None:
            df = self.router.query(
                f"SELECT * FROM raw_articles ORDER BY id DESC LIMIT {limit}"
            )
            if df.empty:
                articles = []
            else:
                articles = df.to_dict("records")

        if not articles:
            return pd.DataFrame()

        # Check FinBERT cache for all articles
        from src.data.finbert_cache_utils import _make_url_hash, get_cached_scores
        hashes = [_make_url_hash(a.get("headline", ""), a.get("summary", "")) for a in articles]
        cached_scores = get_cached_scores(self.router, hashes)
        cache_hits = sum(1 for h in hashes if h in cached_scores)
        cache_misses = len(hashes) - cache_hits
        logger.info(f"Processing {len(articles)} articles for news features …")
        if cache_misses > 0:
            logger.info(f"FinBERT cache: {cache_hits} hits, {cache_misses} misses — "
                         f"running FinBERT on {cache_misses} articles")
            # Score uncached articles with FinBERT and write to cache
            try:
                from src.data.finbert_cache_utils import score_articles_with_cache
                from transformers import pipeline as hf_pipeline
                finbert_pipe = hf_pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=-1,
                    truncation=True,
                    max_length=512,
                )
                scored = score_articles_with_cache(
                    router=self.router,
                    articles=articles,
                    finbert_pipeline=finbert_pipe,
                    batch_size=32,
                )
                # Update cached_scores with newly scored articles
                for i, s in enumerate(scored):
                    h = hashes[i]
                    if h not in cached_scores:
                        cached_scores[h] = {
                            "fb_positive": s["fb_positive"],
                            "fb_negative": s["fb_negative"],
                            "fb_neutral": s["fb_neutral"],
                            "fb_score": s["fb_score"],
                        }
                logger.info(f"FinBERT scoring complete: {cache_misses} articles scored")
            except Exception as e:
                logger.warning(f"FinBERT scoring failed, using VADER fallback: {e}")
        else:
            logger.info(f"FinBERT cache: {cache_hits} hits, 0 misses — all cached!")

        records = []
        for i, a in enumerate(articles):
            text = (a.get("headline", "") + " " + a.get("summary", "")).strip()
            clean = self._clean_text(text)
            pub = a.get("published_at", "")
            date_str = pub[:10] if pub else datetime.now().strftime("%Y-%m-%d")

            # Use FinBERT score if cached, else VADER
            h = hashes[i]
            if h in cached_scores:
                fb = cached_scores[h]
                sent = {
                    "compound": fb["fb_score"],
                    "positive": fb["fb_positive"],
                    "negative": fb["fb_negative"],
                    "neutral": fb["fb_neutral"],
                }
            else:
                sent = self._get_sentiment(text)  # VADER fallback

            records.append({
                "date": date_str,
                "ticker": a.get("ticker", "MARKET"),
                "headline": a.get("headline", ""),
                "clean_text": clean,
                "sentiment_compound": sent["compound"],
                "sentiment_positive": sent["positive"],
                "sentiment_negative": sent["negative"],
                "source": a.get("source", "unknown"),
            })
        return pd.DataFrame(records)

    def fit_tfidf(self, texts: list[str]):
        """Fit TF-IDF vectorizer on corpus."""
        clean = [self._clean_text(t) for t in texts]
        self.vectorizer.fit(clean)
        self._fitted = True

    def transform_tfidf(self, texts: list[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors. Fits if not already fitted."""
        clean = [self._clean_text(t) for t in texts]
        if not self._fitted:
            self.fit_tfidf(texts)
        return self.vectorizer.transform(clean).toarray()

    def build_daily_features(self) -> pd.DataFrame:
        """Aggregate article-level features into daily features.

        Returns DataFrame with: date, article_count, avg_sentiment, max_sentiment,
        min_sentiment, sentiment_std, positive_ratio, negative_ratio
        """
        df = self.process_articles()
        if df.empty:
            return pd.DataFrame()

        daily = df.groupby("date").agg(
            article_count=("sentiment_compound", "count"),
            avg_sentiment=("sentiment_compound", "mean"),
            max_sentiment=("sentiment_compound", "max"),
            min_sentiment=("sentiment_compound", "min"),
            sentiment_std=("sentiment_compound", "std"),
            positive_ratio=("sentiment_positive", "mean"),
            negative_ratio=("sentiment_negative", "mean"),
        ).reset_index()
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
        return daily

    def save_vectorizer(self, path: str = "./models/news_tfidf.pkl"):
        """Save fitted TF-IDF vectorizer."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load_vectorizer(self, path: str = "./models/news_tfidf.pkl"):
        """Load a previously fitted TF-IDF vectorizer."""
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
            self._fitted = True

    def store_features(self, daily_df: pd.DataFrame):
        """Store daily news features via DbRouter."""
        if daily_df.empty:
            return
        try:
            # Ensure table exists
            self.router.execute("""
                CREATE TABLE IF NOT EXISTS news_features (
                    date TEXT PRIMARY KEY,
                    article_count INTEGER,
                    avg_sentiment REAL, max_sentiment REAL, min_sentiment REAL,
                    sentiment_std REAL, positive_ratio REAL, negative_ratio REAL
                )
            """)
            for _, row in daily_df.iterrows():
                self.router.execute(
                    """INSERT OR REPLACE INTO news_features
                       (date, article_count, avg_sentiment, max_sentiment,
                        min_sentiment, sentiment_std, positive_ratio, negative_ratio)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (row["date"], int(row["article_count"]),
                     row["avg_sentiment"], row["max_sentiment"], row["min_sentiment"],
                     row["sentiment_std"], row["positive_ratio"], row["negative_ratio"]),
                )
            logger.info(f"Stored {len(daily_df)} daily news feature rows")
        except Exception as e:
            logger.warning(f"Failed to store news features: {e}")

    def close(self):
        pass  # Router manages connections
