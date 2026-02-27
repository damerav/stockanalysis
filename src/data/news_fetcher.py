"""News Fetcher — Collects news from Finnhub and RSS feeds into news.db.

Inspired by Finance-And-ML/US-Stock-Prediction-Using-ML-And-Spark.
"""

import os
import json
import sqlite3
import logging
import time
from datetime import datetime, timedelta

import requests
import feedparser
import yaml

logger = logging.getLogger(__name__)

NEWS_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    ticker TEXT,
    headline TEXT,
    summary TEXT,
    url TEXT UNIQUE,
    published_at TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_articles_ticker ON raw_articles(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_articles_published ON raw_articles(published_at);
"""


class NewsFetcher:
    """Fetches news from multiple sources and stores in a dedicated news.db."""

    # Ticker aliases for matching headlines to tickers
    TICKER_ALIASES = {
        "SPY": ["SPY", "S&P 500", "S&P500", "SP500"],
        "AAPL": ["Apple", "AAPL"],
        "MSFT": ["Microsoft", "MSFT"],
        "NVDA": ["Nvidia", "NVDA", "NVIDIA"],
        "AMZN": ["Amazon", "AMZN"],
        "GOOGL": ["Google", "Alphabet", "GOOGL", "GOOG"],
        "META": ["Meta", "Facebook", "META"],
        "TSLA": ["Tesla", "TSLA"],
    }

    RSS_FEEDS = [
        ("reuters", "https://feeds.reuters.com/reuters/businessNews"),
        ("cnbc", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147"),
        ("marketwatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ]

    def __init__(self, config: dict = None):
        if config is None:
            with open("config.yaml") as f:
                config = yaml.safe_load(f) or {}
        self.config = config
        self.finnhub_key = config.get("finnhub", {}).get("api_key", "")
        db_path = config.get("news_pipeline", {}).get("db_path", "./data/news.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(NEWS_DB_SCHEMA)

    def fetch_finnhub(self, category: str = "general", days_back: int = 3) -> int:
        """Fetch market news from Finnhub API. Returns count of new articles."""
        if not self.finnhub_key:
            logger.warning("No Finnhub API key configured")
            return 0
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        url = (f"https://finnhub.io/api/v1/news?category={category}"
               f"&from={from_date}&to={to_date}&token={self.finnhub_key}")
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            articles = resp.json()
        except Exception as e:
            logger.error(f"Finnhub fetch failed: {e}")
            return 0

        count = 0
        now = datetime.now().isoformat()
        for a in articles:
            headline = a.get("headline", "")
            summary = a.get("summary", "")
            article_url = a.get("url", "")
            pub_ts = a.get("datetime", 0)
            pub_at = datetime.fromtimestamp(pub_ts).isoformat() if pub_ts else now
            ticker = self._match_ticker(headline + " " + summary)
            try:
                self.conn.execute(
                    "INSERT OR IGNORE INTO raw_articles "
                    "(source, ticker, headline, summary, url, published_at, fetched_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    ("finnhub", ticker, headline, summary, article_url, pub_at, now),
                )
                count += 1
            except sqlite3.IntegrityError:
                pass
        self.conn.commit()
        logger.info(f"Finnhub: fetched {count} new articles")
        return count

    def fetch_rss(self) -> int:
        """Fetch news from RSS feeds. Returns count of new articles."""
        count = 0
        now = datetime.now().isoformat()
        for source, feed_url in self.RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:50]:
                    headline = entry.get("title", "")
                    summary = entry.get("summary", "")[:500]
                    article_url = entry.get("link", "")
                    pub_at = entry.get("published", now)
                    ticker = self._match_ticker(headline + " " + summary)
                    try:
                        self.conn.execute(
                            "INSERT OR IGNORE INTO raw_articles "
                            "(source, ticker, headline, summary, url, published_at, fetched_at) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (source, ticker, headline, summary, article_url, pub_at, now),
                        )
                        count += 1
                    except sqlite3.IntegrityError:
                        pass
            except Exception as e:
                logger.warning(f"RSS fetch failed for {source}: {e}")
        self.conn.commit()
        logger.info(f"RSS: fetched {count} new articles")
        return count

    def fetch_all(self) -> int:
        """Run all fetchers. Returns total new articles."""
        total = self.fetch_finnhub()
        total += self.fetch_rss()
        return total

    def _match_ticker(self, text: str) -> str:
        """Match text to a ticker using alias mapping. Returns 'MARKET' if no match."""
        text_upper = text.upper()
        for ticker, aliases in self.TICKER_ALIASES.items():
            for alias in aliases:
                if alias.upper() in text_upper:
                    return ticker
        return "MARKET"

    def get_recent(self, days: int = 3, ticker: str = None) -> list[dict]:
        """Get recent articles from news.db."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        if ticker:
            rows = self.conn.execute(
                "SELECT * FROM raw_articles WHERE published_at >= ? AND ticker = ? "
                "ORDER BY published_at DESC", (cutoff, ticker)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM raw_articles WHERE published_at >= ? "
                "ORDER BY published_at DESC", (cutoff,)
            ).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM raw_articles LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    def close(self):
        self.conn.close()
