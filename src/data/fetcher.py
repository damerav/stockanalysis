"""1C. Fallback Data Sources — yfinance, Finnhub, RSS feeds, FRED."""

import logging
import requests
import pandas as pd
import feedparser
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class FallbackFetcher:
    """Fallback data sources when Polygon is unavailable."""

    def __init__(self, finnhub_key: str = ""):
        self.finnhub_key = finnhub_key

    # --- Price data fallback (yfinance) ---

    def get_daily_bars_yf(self, ticker: str, days: int = 252) -> pd.DataFrame:
        """Fetch daily OHLCV from yfinance as Polygon fallback."""
        try:
            import yfinance as yf
            end = datetime.now()
            start = end - timedelta(days=int(days * 1.5))  # buffer for weekends
            data = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"), progress=False)
            if data.empty:
                logger.warning(f"yfinance returned no data for {ticker}")
                return pd.DataFrame()
            # Handle multi-level columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            df = data.reset_index()
            df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                    "Close": "close", "Volume": "volume"})
            return df[["date", "open", "high", "low", "close", "volume"]].tail(days)
        except Exception as e:
            logger.error(f"yfinance fetch failed: {e}")
            return pd.DataFrame()

    # --- News (Finnhub + RSS) ---

    def get_news_finnhub(self, category: str = "general", days: int = 1) -> list[dict]:
        """Fetch market news from Finnhub free tier."""
        if not self.finnhub_key:
            return []
        try:
            url = "https://finnhub.io/api/v1/news"
            params = {"category": category, "token": self.finnhub_key}
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Finnhub returned {resp.status_code}")
                return []
            articles = resp.json()
            cutoff = datetime.now() - timedelta(days=days)
            results = []
            for a in articles:
                dt = datetime.fromtimestamp(a.get("datetime", 0))
                if dt >= cutoff:
                    results.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "source": a.get("source", "finnhub"),
                        "headline": a.get("headline", ""),
                        "summary": a.get("summary", "")[:500],
                        "url": a.get("url", ""),
                        "fetched_at": datetime.now().isoformat(),
                    })
            return results
        except Exception as e:
            logger.error(f"Finnhub fetch failed: {e}")
            return []

    def get_news_rss(self) -> list[dict]:
        """Scrape news from Yahoo Finance, CNBC, MarketWatch RSS feeds."""
        feeds = [
            ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US", "yahoo"),
            ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "cnbc"),
            ("https://feeds.marketwatch.com/marketwatch/topstories/", "marketwatch"),
        ]
        articles = []
        for url, source in feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    pub_date = ""
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
                    articles.append({
                        "date": pub_date or datetime.now().strftime("%Y-%m-%d"),
                        "source": source,
                        "headline": entry.get("title", ""),
                        "summary": entry.get("summary", "")[:500],
                        "url": entry.get("link", ""),
                        "fetched_at": datetime.now().isoformat(),
                    })
            except Exception as e:
                logger.warning(f"RSS feed {source} failed: {e}")
        return articles

    # --- Macro data (FRED) ---

    def get_macro_fred(self) -> dict:
        """Fetch macro indicators from FRED (free, no key needed for some series)."""
        series = {
            "vix": "VIXCLS",
            "us10y_yield": "DGS10",
            "dxy": "DTWEXBGS",
            "fed_funds": "FEDFUNDS",
            "gold": None,  # fetched via yfinance below
            "crude": "DCOILWTICO",
        }
        result = {}
        for name, fred_id in series.items():
            if fred_id is None:
                continue
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
                df = pd.read_csv(url)
                # FRED CSV columns vary; find the date and value columns
                if df.shape[1] >= 2:
                    df.columns = ["date", "value"]
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.dropna(subset=["value"])
                    if not df.empty:
                        result[name] = float(df.iloc[-1]["value"])
                    else:
                        result[name] = None
                else:
                    result[name] = None
            except Exception as e:
                logger.warning(f"FRED {name} ({fred_id}) failed: {e}")
                result[name] = None

        # Gold price via yfinance (FRED series unreliable)
        try:
            import yfinance as yf
            gold_data = yf.download("GC=F", period="5d", progress=False)
            if not gold_data.empty:
                if isinstance(gold_data.columns, pd.MultiIndex):
                    gold_data.columns = gold_data.columns.get_level_values(0)
                result["gold"] = float(gold_data["Close"].iloc[-1])
            else:
                result["gold"] = None
        except Exception as e:
            logger.warning(f"Gold price fetch failed: {e}")
            result["gold"] = None

        # Compute VIX change if we have current VIX
        if result.get("vix") is not None:
            try:
                url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
                df = pd.read_csv(url)
                df.columns = ["date", "value"]
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["value"])
                if len(df) >= 2:
                    result["vix_change"] = float(df.iloc[-1]["value"]) - float(df.iloc[-2]["value"])
                else:
                    result["vix_change"] = 0.0
            except Exception:
                result["vix_change"] = 0.0
        else:
            result["vix_change"] = None

        return result
