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

# Migration: add category column to existing databases
NEWS_DB_MIGRATION = """
ALTER TABLE raw_articles ADD COLUMN category TEXT DEFAULT 'markets';
"""


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
        """Fetch news from RSS feeds (45+ categorized feeds). Returns count of new articles."""
        count = 0
        now = datetime.now().isoformat()
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; StockAnalysis/2.0; +https://github.com/damerav/stockanalysis)"
        }
        for feed_tuple in self.RSS_FEEDS:
            source, feed_url, category = feed_tuple
            try:
                resp = requests.get(feed_url, headers=headers, timeout=15)
                feed = feedparser.parse(resp.content)
                entries = feed.entries[:100]  # up to 100 per feed
                src_count = 0
                for entry in entries:
                    headline = entry.get("title", "")
                    summary = entry.get("summary", "")[:500]
                    article_url = entry.get("link", "")
                    # Normalize published date to ISO format for consistent sorting
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
                    try:
                        self.conn.execute(
                            "INSERT OR IGNORE INTO raw_articles "
                            "(source, ticker, headline, summary, url, published_at, fetched_at, category) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            (source, ticker, headline, summary, article_url, pub_at, now, category),
                        )
                        src_count += 1
                    except sqlite3.IntegrityError:
                        pass
                count += src_count
                if src_count > 0:
                    logger.info(f"RSS {source} [{category}]: {src_count} new articles")
            except Exception as e:
                logger.warning(f"RSS fetch failed for {source}: {e}")
            time.sleep(0.3)  # be polite between feeds
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
                        self.conn.execute(
                            "INSERT OR IGNORE INTO raw_articles "
                            "(source, ticker, headline, summary, url, published_at, fetched_at) "
                            "VALUES (?,?,?,?,?,?,?)",
                            ("finnhub_company", ticker, headline, summary, article_url, pub_at, now),
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
                        self.conn.execute(
                            "INSERT OR IGNORE INTO raw_articles "
                            "(source, ticker, headline, summary, url, published_at, fetched_at) "
                            "VALUES (?,?,?,?,?,?,?)",
                            ("alpha_vantage", ticker, headline, summary, article_url, pub_at, now),
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
        """Get article counts and basic stats per category for recent articles.

        Returns {category: {count, sources}} for pipeline feature computation.
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            "SELECT category, COUNT(*) as cnt, COUNT(DISTINCT source) as src_cnt "
            "FROM raw_articles WHERE fetched_at >= ? GROUP BY category",
            (cutoff,),
        ).fetchall()
        return {
            r[0] or "markets": {"count": r[1], "source_count": r[2]}
            for r in rows
        }

    def close(self):
        self.conn.close()
