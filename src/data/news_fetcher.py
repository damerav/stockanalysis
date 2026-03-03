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
    fetched_at TEXT,
    sentiment_compound REAL,
    sentiment_pos REAL,
    sentiment_neg REAL,
    sentiment_neu REAL
);
CREATE INDEX IF NOT EXISTS idx_raw_articles_ticker ON raw_articles(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_articles_published ON raw_articles(published_at);
"""

# Migration: add category column to existing databases
NEWS_DB_MIGRATION = """
ALTER TABLE raw_articles ADD COLUMN category TEXT DEFAULT 'markets';
"""

# Migration: add sentiment columns to existing databases
NEWS_DB_SENTIMENT_MIGRATION = [
    "ALTER TABLE raw_articles ADD COLUMN sentiment_compound REAL",
    "ALTER TABLE raw_articles ADD COLUMN sentiment_pos REAL",
    "ALTER TABLE raw_articles ADD COLUMN sentiment_neg REAL",
    "ALTER TABLE raw_articles ADD COLUMN sentiment_neu REAL",
]

# Migration: add quality_score column
NEWS_DB_QUALITY_MIGRATION = "ALTER TABLE raw_articles ADD COLUMN quality_score REAL DEFAULT 0.5"

# Source credibility tiers (0.0 - 1.0)
# Tier 1: Major financial news outlets with editorial standards
# Tier 2: Aggregators and secondary sources
# Tier 3: Google News proxied / general
SOURCE_CREDIBILITY = {
    # Tier 1 — high credibility
    "bloomberg_markets": 0.95, "fed_press": 0.95,
    "cnbc_top": 0.85, "cnbc_economy": 0.85, "cnbc_earnings": 0.85, "cnbc_finance": 0.85,
    "marketwatch_top": 0.80, "marketwatch_markets": 0.80, "marketwatch_stocks": 0.80,
    "yahoo_finance": 0.75, "seekingalpha_market": 0.75, "seekingalpha_news": 0.75,
    "investing_news": 0.70, "investing_analysis": 0.75,
    # Tier 2 — moderate credibility
    "finnhub": 0.70, "alpha_vantage": 0.65,
    # Tier 3 — Google News aggregated (variable quality)
    "google_business": 0.55, "google_markets": 0.55, "google_economy": 0.55,
}
# Default for unknown sources
_DEFAULT_CREDIBILITY = 0.50


# VADER sentiment — compute at fetch time
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None


def _compute_vader(text: str) -> tuple:
    """Compute VADER sentiment. Returns (compound, pos, neg, neu)."""
    if _vader and text:
        scores = _vader.polarity_scores(text)
        return (scores["compound"], scores["pos"], scores["neg"], scores["neu"])
    # Simple keyword fallback
    if not text:
        return (0.0, 0.0, 0.0, 1.0)
    text_lower = text.lower()
    pos_words = ["surge", "rally", "gain", "bull", "rise", "profit", "beat", "strong", "soar", "jump"]
    neg_words = ["crash", "drop", "fall", "bear", "loss", "miss", "weak", "fear", "plunge", "sink"]
    p = sum(1 for w in pos_words if w in text_lower)
    n = sum(1 for w in neg_words if w in text_lower)
    total = p + n
    if total == 0:
        return (0.0, 0.0, 0.0, 1.0)
    compound = (p - n) / total
    return (compound, max(0, compound), abs(min(0, compound)), 1.0 - abs(compound))


def _compute_quality_score(source: str, headline: str, summary: str,
                           sentiment_compound: float) -> float:
    """Compute article quality score (0.0 - 1.0).

    Factors:
      - Source credibility (40%): editorial standards of the outlet
      - Content depth (30%): headline + summary length as proxy for substance
      - Sentiment confidence (15%): strong VADER signal = more informative
      - Specificity (15%): contains numbers, tickers, or named entities
    """
    # Source credibility (0.0 - 1.0)
    cred = SOURCE_CREDIBILITY.get(source, _DEFAULT_CREDIBILITY)

    # Content depth: longer = more substantive (capped at 1.0)
    text_len = len(headline or "") + len(summary or "")
    depth = min(text_len / 400.0, 1.0)  # 400 chars = full score

    # Sentiment confidence: strong signal = more informative
    sent_conf = min(abs(sentiment_compound) / 0.5, 1.0)

    # Specificity: numbers, $ signs, % signs, ticker-like patterns
    combined = (headline or "") + " " + (summary or "")
    has_numbers = 1.0 if any(c.isdigit() for c in combined) else 0.0
    has_dollar = 1.0 if "$" in combined else 0.0
    has_pct = 1.0 if "%" in combined else 0.0
    specificity = (has_numbers * 0.4 + has_dollar * 0.3 + has_pct * 0.3)

    score = (cred * 0.40 + depth * 0.30 + sent_conf * 0.15 + specificity * 0.15)
    return round(min(max(score, 0.0), 1.0), 3)


class NewsFetcher:
    """Fetches news from multiple sources and stores in a dedicated news.db."""

    # Ticker aliases for matching headlines to tickers
    TICKER_ALIASES = {
        "SPY": ["SPY", "S&P 500", "S&P500", "SP500", "S&P", "SPX"],
        "AAPL": ["Apple", "AAPL", "iPhone", "iPad", "Tim Cook"],
        "MSFT": ["Microsoft", "MSFT", "Azure", "Satya Nadella"],
        "NVDA": ["Nvidia", "NVDA", "NVIDIA", "Jensen Huang"],
        "AMZN": ["Amazon", "AMZN", "AWS", "Andy Jassy"],
        "GOOGL": ["Google", "Alphabet", "GOOGL", "GOOG", "Sundar Pichai"],
        "META": ["Meta", "Facebook", "META", "Zuckerberg", "Instagram", "WhatsApp"],
        "TSLA": ["Tesla", "TSLA", "Elon Musk"],
        "JPM": ["JPMorgan", "JP Morgan", "JPM", "Jamie Dimon"],
        "V": ["Visa"],
        "MA": ["Mastercard"],
        "BRK": ["Berkshire", "Buffett", "Warren Buffett"],
        "UNH": ["UnitedHealth", "UNH"],
        "XOM": ["Exxon", "ExxonMobil", "XOM"],
        "JNJ": ["Johnson & Johnson", "JNJ", "J&J"],
        "WMT": ["Walmart", "WMT"],
        "PG": ["Procter & Gamble", "P&G"],
        "HD": ["Home Depot"],
        "DIS": ["Disney", "DIS"],
        "NFLX": ["Netflix", "NFLX"],
        "CRM": ["Salesforce", "CRM"],
        "AMD": ["AMD", "Advanced Micro"],
        "INTC": ["Intel", "INTC"],
        "BA": ["Boeing", "BA"],
        # Macro / market-wide keywords
        "MACRO": ["Federal Reserve", "Fed ", "FOMC", "interest rate", "inflation",
                   "CPI", "jobs report", "unemployment", "GDP", "Treasury",
                   "yield curve", "recession", "tariff", "trade war"],
        "VIX": ["VIX", "volatility", "fear index", "CBOE"],
    }

    # Feed categories for sentiment decomposition
    FEED_CATEGORIES = [
        "markets", "forex", "bonds", "commodities", "crypto",
        "centralbanks", "economic", "ipo", "derivatives",
        "fintech", "regulation", "institutional", "analysis",
    ]

    # (source_name, url, category) — 3-tuple format
    RSS_FEEDS = [
        # ===== MARKETS (existing + worldmonitor) =====
        ("cnbc_top", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147", "markets"),
        ("cnbc_economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258", "markets"),
        ("cnbc_earnings", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839135", "markets"),
        ("cnbc_finance", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664", "markets"),
        ("marketwatch_top", "https://feeds.marketwatch.com/marketwatch/topstories/", "markets"),
        ("marketwatch_markets", "https://feeds.marketwatch.com/marketwatch/marketpulse/", "markets"),
        ("marketwatch_stocks", "https://feeds.marketwatch.com/marketwatch/StockstoWatch/", "markets"),
        ("google_business", "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en", "markets"),
        ("google_markets", "https://news.google.com/rss/search?q=stock+market+SPY+S%26P+500&hl=en-US&gl=US&ceid=US:en", "markets"),
        ("google_economy", "https://news.google.com/rss/search?q=US+economy+federal+reserve+inflation&hl=en-US&gl=US&ceid=US:en", "markets"),
        ("yahoo_finance", "https://finance.yahoo.com/news/rssindex", "markets"),
        ("investing_news", "https://www.investing.com/rss/news.rss", "markets"),
        ("investing_analysis", "https://www.investing.com/rss/news_301.rss", "markets"),
        ("seekingalpha_market", "https://seekingalpha.com/market_currents.xml", "markets"),
        ("seekingalpha_news", "https://seekingalpha.com/feed.xml", "markets"),
        ("bloomberg_markets", "https://feeds.bloomberg.com/markets/news.rss", "markets"),
        # ===== CENTRAL BANKS =====
        ("fed_press", "https://www.federalreserve.gov/feeds/press_all.xml", "centralbanks"),
        ("google_ecb", "https://news.google.com/rss/search?q=ECB+European+Central+Bank+interest+rate&hl=en-US&gl=US&ceid=US:en", "centralbanks"),
        ("google_boj", "https://news.google.com/rss/search?q=Bank+of+Japan+BOJ+monetary+policy&hl=en-US&gl=US&ceid=US:en", "centralbanks"),
        ("google_boe", "https://news.google.com/rss/search?q=Bank+of+England+BOE+interest+rate&hl=en-US&gl=US&ceid=US:en", "centralbanks"),
        ("google_global_cb", "https://news.google.com/rss/search?q=central+bank+monetary+policy+rate+decision&hl=en-US&gl=US&ceid=US:en", "centralbanks"),
        # ===== FOREX =====
        ("google_forex", "https://news.google.com/rss/search?q=forex+currency+exchange+rate+USD&hl=en-US&gl=US&ceid=US:en", "forex"),
        ("google_dollar", "https://news.google.com/rss/search?q=US+dollar+DXY+currency+strength&hl=en-US&gl=US&ceid=US:en", "forex"),
        # ===== BONDS =====
        ("google_bonds", "https://news.google.com/rss/search?q=bond+market+treasury+yield+fixed+income&hl=en-US&gl=US&ceid=US:en", "bonds"),
        ("google_treasury", "https://news.google.com/rss/search?q=US+Treasury+bond+yield+curve&hl=en-US&gl=US&ceid=US:en", "bonds"),
        # ===== COMMODITIES =====
        ("google_oil", "https://news.google.com/rss/search?q=oil+price+crude+WTI+Brent+OPEC&hl=en-US&gl=US&ceid=US:en", "commodities"),
        ("google_gold", "https://news.google.com/rss/search?q=gold+price+precious+metals+silver&hl=en-US&gl=US&ceid=US:en", "commodities"),
        ("google_agriculture", "https://news.google.com/rss/search?q=agriculture+commodity+wheat+corn+soybean&hl=en-US&gl=US&ceid=US:en", "commodities"),
        # ===== ECONOMIC DATA =====
        ("google_econ_data", "https://news.google.com/rss/search?q=CPI+GDP+NFP+PMI+economic+data+report&hl=en-US&gl=US&ceid=US:en", "economic"),
        ("google_trade", "https://news.google.com/rss/search?q=trade+tariff+import+export+trade+war&hl=en-US&gl=US&ceid=US:en", "economic"),
        ("google_housing", "https://news.google.com/rss/search?q=housing+market+real+estate+mortgage+rates&hl=en-US&gl=US&ceid=US:en", "economic"),
        # ===== IPO / EARNINGS / M&A =====
        ("google_ipo", "https://news.google.com/rss/search?q=IPO+initial+public+offering+stock+listing&hl=en-US&gl=US&ceid=US:en", "ipo"),
        ("google_earnings", "https://news.google.com/rss/search?q=earnings+report+quarterly+results+EPS&hl=en-US&gl=US&ceid=US:en", "ipo"),
        ("google_ma", "https://news.google.com/rss/search?q=merger+acquisition+M%26A+deal+buyout&hl=en-US&gl=US&ceid=US:en", "ipo"),
        # ===== DERIVATIVES =====
        ("google_options", "https://news.google.com/rss/search?q=options+market+call+put+derivatives+trading&hl=en-US&gl=US&ceid=US:en", "derivatives"),
        ("google_futures", "https://news.google.com/rss/search?q=futures+trading+ES+NQ+commodities+futures&hl=en-US&gl=US&ceid=US:en", "derivatives"),
        # ===== REGULATION =====
        ("google_sec", "https://news.google.com/rss/search?q=SEC+securities+regulation+enforcement&hl=en-US&gl=US&ceid=US:en", "regulation"),
        ("google_finreg", "https://news.google.com/rss/search?q=financial+regulation+banking+rules+compliance&hl=en-US&gl=US&ceid=US:en", "regulation"),
        # ===== INSTITUTIONAL =====
        ("google_hedgefund", "https://news.google.com/rss/search?q=hedge+fund+institutional+investor+13F&hl=en-US&gl=US&ceid=US:en", "institutional"),
        ("google_pe", "https://news.google.com/rss/search?q=private+equity+venture+capital+investment&hl=en-US&gl=US&ceid=US:en", "institutional"),
        # ===== ANALYSIS =====
        ("google_outlook", "https://news.google.com/rss/search?q=market+outlook+forecast+prediction+analyst&hl=en-US&gl=US&ceid=US:en", "analysis"),
        ("google_risk", "https://news.google.com/rss/search?q=market+risk+volatility+VIX+fear+greed&hl=en-US&gl=US&ceid=US:en", "analysis"),
        # ===== CRYPTO (useful for risk-on/risk-off signal) =====
        ("google_crypto", "https://news.google.com/rss/search?q=bitcoin+cryptocurrency+crypto+market&hl=en-US&gl=US&ceid=US:en", "crypto"),
        # ===== FINTECH =====
        ("google_fintech", "https://news.google.com/rss/search?q=fintech+trading+technology+algorithmic&hl=en-US&gl=US&ceid=US:en", "fintech"),
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
        # Create base schema (without category — handles existing DBs)
        self.conn.executescript(NEWS_DB_SCHEMA)
        # Migrate: add category column if missing
        try:
            self.conn.execute("SELECT category FROM raw_articles LIMIT 1")
        except sqlite3.OperationalError:
            try:
                self.conn.execute(NEWS_DB_MIGRATION)
                self.conn.commit()
                logger.info("Migrated news.db: added category column")
            except sqlite3.OperationalError:
                pass
        # Create category index (safe after migration)
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_articles_category ON raw_articles(category)")
            self.conn.commit()
        except sqlite3.OperationalError:
            pass
        # Migrate: add sentiment columns if missing
        try:
            self.conn.execute("SELECT sentiment_compound FROM raw_articles LIMIT 1")
        except sqlite3.OperationalError:
            for sql in NEWS_DB_SENTIMENT_MIGRATION:
                try:
                    self.conn.execute(sql)
                except sqlite3.OperationalError:
                    pass
            self.conn.commit()
            logger.info("Migrated news.db: added sentiment columns")
        # Migrate: add quality_score column if missing
        try:
            self.conn.execute("SELECT quality_score FROM raw_articles LIMIT 1")
        except sqlite3.OperationalError:
            try:
                self.conn.execute(NEWS_DB_QUALITY_MIGRATION)
                self.conn.commit()
                logger.info("Migrated news.db: added quality_score column")
            except sqlite3.OperationalError:
                pass

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
            sent = _compute_vader(headline + " " + summary)
            qscore = _compute_quality_score("finnhub", headline, summary, sent[0])
            try:
                self.conn.execute(
                    "INSERT OR IGNORE INTO raw_articles "
                    "(source, ticker, headline, summary, url, published_at, fetched_at, "
                    "sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu, quality_score) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    ("finnhub", ticker, headline, summary, article_url, pub_at, now,
                     sent[0], sent[1], sent[2], sent[3], qscore),
                )
                count += 1
            except sqlite3.IntegrityError:
                pass
        self.conn.commit()
        logger.info(f"Finnhub: fetched {count} new articles")
        return count

    def fetch_rss(self) -> int:
        """Fetch news from RSS feeds (45+ categorized feeds) in parallel. Returns count of new articles."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        now = datetime.now().isoformat()
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; StockAnalysis/2.0; +https://github.com/damerav/stockanalysis)"
        }

        def _fetch_one_feed(feed_tuple):
            source, feed_url, category = feed_tuple
            articles = []
            try:
                resp = requests.get(feed_url, headers=headers, timeout=15)
                feed = feedparser.parse(resp.content)
                for entry in feed.entries[:100]:
                    headline = entry.get("title", "")
                    summary = entry.get("summary", "")[:500]
                    article_url = entry.get("link", "")
                    pub_at = now
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            from time import mktime
                            pub_at = datetime.fromtimestamp(mktime(entry.published_parsed)).isoformat()
                        except Exception:
                            pub_at = entry.get("published", now)
                    elif entry.get("published"):
                        raw_pub = entry["published"]
                        for fmt in ("%a, %d %b %Y %H:%M:%S %Z",
                                    "%a, %d %b %Y %H:%M:%S %z",
                                    "%Y-%m-%dT%H:%M:%S%z",
                                    "%Y-%m-%dT%H:%M:%SZ",
                                    "%Y-%m-%d %H:%M:%S"):
                            try:
                                pub_at = datetime.strptime(raw_pub.strip(), fmt).isoformat()
                                break
                            except ValueError:
                                continue
                        else:
                            pub_at = raw_pub
                    ticker = self._match_ticker(headline + " " + summary)
                    sent = _compute_vader(headline + " " + summary)
                    quality = _compute_quality_score(source, headline, summary, sent[0])
                    articles.append((source, ticker, headline, summary, article_url, pub_at, now, category,
                                     sent[0], sent[1], sent[2], sent[3], quality))
            except Exception as e:
                logger.warning(f"RSS fetch failed for {source}: {e}")
            return source, category, articles

        # Fetch all feeds in parallel (8 threads — polite but fast)
        all_articles = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_one_feed, ft): ft for ft in self.RSS_FEEDS}
            for future in as_completed(futures):
                source, category, articles = future.result()
                if articles:
                    all_articles.extend(articles)
                    logger.info(f"RSS {source} [{category}]: {len(articles)} articles")

        # Batch insert (single transaction)
        count = 0
        for row in all_articles:
            try:
                self.conn.execute(
                    "INSERT OR IGNORE INTO raw_articles "
                    "(source, ticker, headline, summary, url, published_at, fetched_at, category, "
                    "sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu, quality_score) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", row,
                )
                count += 1
            except sqlite3.IntegrityError:
                pass
        self.conn.commit()
        logger.info(f"RSS total: {count} new articles from {len(self.RSS_FEEDS)} feeds")
        return count

    def fetch_all(self) -> int:
        """Run all fetchers. Returns total new articles."""
        total = self.fetch_finnhub()
        total += self.fetch_finnhub_company_news()
        total += self.fetch_alpha_vantage_news()
        total += self.fetch_rss()
        return total

    def fetch_finnhub_company_news(self, days_back: int = 3) -> int:
        """Fetch company-specific news from Finnhub for top tickers."""
        if not self.finnhub_key:
            return 0
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        count = 0
        now = datetime.now().isoformat()
        for ticker in tickers:
            url = (f"https://finnhub.io/api/v1/company-news?symbol={ticker}"
                   f"&from={from_date}&to={to_date}&token={self.finnhub_key}")
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                articles = resp.json()
                for a in articles[:50]:  # cap per ticker
                    headline = a.get("headline", "")
                    summary = a.get("summary", "")
                    article_url = a.get("url", "")
                    pub_ts = a.get("datetime", 0)
                    pub_at = datetime.fromtimestamp(pub_ts).isoformat() if pub_ts else now
                    try:
                        sent = _compute_vader(headline + " " + summary)
                        qscore = _compute_quality_score("finnhub", headline, summary, sent[0])
                        self.conn.execute(
                            "INSERT OR IGNORE INTO raw_articles "
                            "(source, ticker, headline, summary, url, published_at, fetched_at, "
                            "sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu, quality_score) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            ("finnhub_company", ticker, headline, summary, article_url, pub_at, now,
                             sent[0], sent[1], sent[2], sent[3], qscore),
                        )
                        count += 1
                    except sqlite3.IntegrityError:
                        pass
                time.sleep(0.3)  # rate limit
            except Exception as e:
                logger.warning(f"Finnhub company news for {ticker} failed: {e}")
        self.conn.commit()
        logger.info(f"Finnhub company news: {count} new articles")
        return count

    def fetch_alpha_vantage_news(self) -> int:
        """Fetch news sentiment from Alpha Vantage (free: 25 req/day)."""
        av_key = self.config.get("alpha_vantage", {}).get("api_key", "")
        if not av_key:
            logger.debug("No Alpha Vantage API key — skipping news sentiment")
            return 0
        topics = ["financial_markets", "economy_macro", "technology"]
        count = 0
        now = datetime.now().isoformat()
        for topic in topics:
            url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
                   f"&topics={topic}&limit=50&apikey={av_key}")
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("feed", []):
                    headline = item.get("title", "")
                    summary = item.get("summary", "")[:500]
                    article_url = item.get("url", "")
                    pub_at = item.get("time_published", now)
                    # Convert YYYYMMDDTHHMMSS to ISO
                    if "T" in pub_at and len(pub_at) >= 15:
                        try:
                            pub_at = datetime.strptime(pub_at[:15], "%Y%m%dT%H%M%S").isoformat()
                        except ValueError:
                            pass
                    ticker = self._match_ticker(headline + " " + summary)
                    # Also check AV's ticker_sentiment
                    for ts in item.get("ticker_sentiment", []):
                        t = ts.get("ticker", "")
                        if t in self.TICKER_ALIASES:
                            ticker = t
                            break
                    try:
                        sent = _compute_vader(headline + " " + summary)
                        qscore = _compute_quality_score("alpha_vantage", headline, summary, sent[0])
                        self.conn.execute(
                            "INSERT OR IGNORE INTO raw_articles "
                            "(source, ticker, headline, summary, url, published_at, fetched_at, "
                            "sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu, quality_score) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            ("alpha_vantage", ticker, headline, summary, article_url, pub_at, now,
                             sent[0], sent[1], sent[2], sent[3], qscore),
                        )
                        count += 1
                    except sqlite3.IntegrityError:
                        pass
                time.sleep(1)  # conservative rate limit
            except Exception as e:
                logger.warning(f"Alpha Vantage news ({topic}) failed: {e}")
        self.conn.commit()
        logger.info(f"Alpha Vantage news: {count} new articles")
        return count

    def _match_ticker(self, text: str) -> str:
        """Match text to a ticker using alias mapping. Returns 'MARKET' if no match."""
        text_upper = text.upper()
        for ticker, aliases in self.TICKER_ALIASES.items():
            for alias in aliases:
                if alias.upper() in text_upper:
                    return ticker
        return "MARKET"

    def get_recent(self, days: int = 3, ticker: str = None) -> list[dict]:
        """Get recent articles from news.db, sorted newest-first."""
        # Use fetched_at for filtering (always ISO format) since published_at
        # has mixed formats (ISO vs RFC 2822) that break SQL string comparison
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        if ticker:
            rows = self.conn.execute(
                "SELECT * FROM raw_articles WHERE fetched_at >= ? AND ticker = ? "
                "ORDER BY fetched_at DESC", (cutoff, ticker)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM raw_articles WHERE fetched_at >= ? "
                "ORDER BY fetched_at DESC", (cutoff,)
            ).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM raw_articles LIMIT 0").description]
        results = [dict(zip(cols, r)) for r in rows]
        # Re-sort by published_at in Python to handle mixed date formats
        def _parse_date(art):
            raw = art.get("published_at", "")
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                return dt.replace(tzinfo=None)
            except (ValueError, TypeError):
                pass
            for fmt in ("%a, %d %b %Y %H:%M:%S %Z",
                        "%a, %d %b %Y %H:%M:%S %z",
                        "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(raw.strip(), fmt)
                    return dt.replace(tzinfo=None)
                except (ValueError, TypeError):
                    continue
            return datetime.min
        results.sort(key=_parse_date, reverse=True)
        return results

    def get_recent_by_category(self, days: int = 3, category: str = None) -> dict[str, list[dict]]:
        """Get recent articles grouped by category. Returns {category: [articles]}."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        if category:
            rows = self.conn.execute(
                "SELECT * FROM raw_articles WHERE fetched_at >= ? AND category = ? "
                "ORDER BY fetched_at DESC", (cutoff, category)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM raw_articles WHERE fetched_at >= ? "
                "ORDER BY fetched_at DESC", (cutoff,)
            ).fetchall()
        cols = [d[0] for d in self.conn.execute("SELECT * FROM raw_articles LIMIT 0").description]
        grouped: dict[str, list[dict]] = {}
        for r in rows:
            art = dict(zip(cols, r))
            cat = art.get("category", "markets")
            grouped.setdefault(cat, []).append(art)
        return grouped

    def get_category_sentiment_summary(self, days: int = 1) -> dict[str, dict]:
        """Get article counts, sentiment stats, and quality-weighted sentiment per category.

        Returns {category: {count, sources, avg_sentiment, weighted_sentiment}} for pipeline feature computation.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            "SELECT category, COUNT(*) as cnt, COUNT(DISTINCT source) as src_cnt, "
            "AVG(sentiment_compound) as avg_sent, "
            "CASE WHEN SUM(COALESCE(quality_score, 0.5)) > 0 "
            "  THEN SUM(sentiment_compound * COALESCE(quality_score, 0.5)) / SUM(COALESCE(quality_score, 0.5)) "
            "  ELSE AVG(sentiment_compound) END as weighted_sent "
            "FROM raw_articles WHERE fetched_at >= ? GROUP BY category",
            (cutoff,),
        ).fetchall()
        return {
            r[0] or "markets": {
                "count": r[1], "source_count": r[2],
                "avg_sentiment": r[3], "weighted_sentiment": r[4],
            }
            for r in rows
        }

    def backfill_sentiment(self) -> int:
        """Backfill VADER sentiment for articles that have NULL sentiment_compound."""
        rows = self.conn.execute(
            "SELECT id, headline, summary FROM raw_articles "
            "WHERE sentiment_compound IS NULL"
        ).fetchall()
        if not rows:
            logger.info("No articles need sentiment backfill")
            return 0
        count = 0
        for row_id, headline, summary in rows:
            text = ((headline or "") + " " + (summary or "")).strip()
            sent = _compute_vader(text)
            self.conn.execute(
                "UPDATE raw_articles SET sentiment_compound=?, sentiment_pos=?, "
                "sentiment_neg=?, sentiment_neu=? WHERE id=?",
                (sent[0], sent[1], sent[2], sent[3], row_id),
            )
            count += 1
        self.conn.commit()
        logger.info(f"Backfilled sentiment for {count} articles in news.db")
        return count

    def sync_sentiment_to_postgres(self) -> int:
        """Sync sentiment values from news.db to PostgreSQL raw_articles."""
        try:
            from src.data.db_router import get_router
            router = get_router(self.config)
            if not router.using_postgres:
                return 0
            pg = router.get_pg()
            cur = pg.cursor()
            # Get articles with sentiment from news.db
            rows = self.conn.execute(
                "SELECT url, sentiment_compound, sentiment_pos, sentiment_neg, sentiment_neu "
                "FROM raw_articles WHERE sentiment_compound IS NOT NULL"
            ).fetchall()
            updated = 0
            for url, compound, pos, neg, neu in rows:
                cur.execute(
                    "UPDATE raw_articles SET sentiment_compound=%s, sentiment_pos=%s, "
                    "sentiment_neg=%s, sentiment_neu=%s "
                    "WHERE url=%s AND (sentiment_compound IS NULL OR sentiment_compound = 0)",
                    (compound, pos, neg, neu, url),
                )
                if cur.rowcount > 0:
                    updated += 1
            cur.close()
            logger.info(f"Synced sentiment to PostgreSQL: {updated} rows updated")
            return updated
        except Exception as e:
            logger.warning(f"PostgreSQL sentiment sync failed: {e}")
            return 0

    def close(self):
        self.conn.close()
