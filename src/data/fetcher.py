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

    def __init__(self, finnhub_key: str = "", fred_key: str = "", config: dict = None):
        self.config = config or {}
        if config:
            self.finnhub_key = finnhub_key or config.get("finnhub", {}).get("api_key", "")
            self.fred_key = fred_key or config.get("fred", {}).get("api_key", "")
        else:
            self.finnhub_key = finnhub_key
            self.fred_key = fred_key
        # Resolve from encrypted DB if config values are placeholders
        try:
            from src.data.secrets_manager import get_secret
            if not self.finnhub_key or self.finnhub_key == "FROM_ENCRYPTED_DB":
                self.finnhub_key = get_secret("finnhub_api_key", fallback=self.finnhub_key or "")
            if not self.fred_key or self.fred_key == "FROM_ENCRYPTED_DB":
                self.fred_key = get_secret("fred_api_key", fallback=self.fred_key or "")
        except Exception:
            pass

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
        """Fetch macro indicators from FRED. Uses official API if key is set, else CSV fallback."""
        series = {
            # Core Market & Rates
            "vix": "VIXCLS",
            "us10y_yield": "DGS10",
            "us3m_yield": "DTB3",              # 3-Month Treasury Bill rate
            "yield_curve_10y3m": "T10Y3M",     # 10Y-3M spread (recession signal)
            "sahm_rule": "SAHMREALTIME",       # Sahm Rule recession indicator
            "consumer_conf": "UMCSENT",        # U. of Michigan Consumer Sentiment
            "ism_pmi": "INDPRO",               # Industrial Production Index (manufacturing activity proxy)
            "dxy": "DTWEXBGS",
            "fed_funds": "FEDFUNDS",
            "gold": None,  # fetched via yfinance below
            "crude": "DCOILWTICO",
            # Inflation
            "cpi": "CPIAUCSL",                 # CPI All Urban Consumers, SA
            "core_cpi": "CPILFESL",            # Core CPI (less food & energy)
            "pce": "PCEPI",                    # Personal Consumption Expenditures Price Index
            "core_pce": "PCEPILFE",            # Core PCE
            "ppi": "PPIACO",                   # Producer Price Index, All Commodities
            # Growth & Employment
            "gdp": "GDP",                      # GDP, SA Annual Rate
            "nfp": "PAYEMS",                   # Non-Farm Payrolls
            "unemployment_rate": "UNRATE",
            "initial_claims": "ICSA",          # Initial Jobless Claims
            "continuing_claims": "CCSA",       # Continuing Jobless Claims
            # Activity
            "retail_sales": "RSXFS",           # Retail Sales, ex. Food Services
            "industrial_production": "INDPRO", # Industrial Production Index
            # Housing
            "housing_starts": "HOUST",
            "building_permits": "PERMIT",
            "case_shiller_hpi": "CSUSHPINSA",  # S&P/Case-Shiller National HPI
        }
        result = {}
        for name, fred_id in series.items():
            if fred_id is None:
                continue
            try:
                if self.fred_key:
                    # Official FRED API (more reliable, higher rate limits)
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": fred_id,
                        "api_key": self.fred_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 5,
                    }
                    resp = requests.get(url, params=params, timeout=15)
                    if resp.status_code == 200:
                        obs = resp.json().get("observations", [])
                        for o in obs:
                            val = o.get("value", ".")
                            if val != ".":
                                result[name] = float(val)
                                break
                        else:
                            result[name] = None
                    else:
                        logger.warning(f"FRED API {fred_id} returned {resp.status_code}")
                        result[name] = None
                else:
                    # CSV fallback (no key needed)
                    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
                    df = pd.read_csv(url)
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

        # Gold price — Polygon primary, yfinance fallback
        gold_fetched = False
        try:
            import os as _os
            _poly_key = ""
            try:
                from src.data.secrets_manager import get_secret
                _poly_key = get_secret("polygon_api_key", fallback="")
            except Exception:
                pass
            if not _poly_key:
                _poly_key = (self.config.get("polygon", {}) or {}).get("api_key", "")
            if not _poly_key or _poly_key == "FROM_ENCRYPTED_DB":
                _poly_key = _os.environ.get("POLYGON_API_KEY", "")
            if _poly_key:
                from src.data.polygon_fetcher import PolygonFetcher
                _poly = PolygonFetcher(_poly_key)
                from datetime import datetime, timedelta
                _today = datetime.now().strftime("%Y-%m-%d")
                _ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                _gld = _poly.get_daily_bars("GLD", _ago, _today)
                if not _gld.empty:
                    result["gold"] = float(_gld["close"].iloc[-1])
                    gold_fetched = True
                    logger.debug("Gold from Polygon (GLD)")
        except Exception as e:
            logger.debug(f"Polygon gold failed: {e}")

        if not gold_fetched:
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
                logger.warning(f"Gold price fetch failed (both sources): {e}")
                result["gold"] = None

        # Compute VIX change if we have current VIX
        if result.get("vix") is not None:
            try:
                if self.fred_key:
                    url = "https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        "series_id": "VIXCLS",
                        "api_key": self.fred_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 5,
                    }
                    resp = requests.get(url, params=params, timeout=15)
                    if resp.status_code == 200:
                        obs = resp.json().get("observations", [])
                        vals = [float(o["value"]) for o in obs if o.get("value", ".") != "."]
                        if len(vals) >= 2:
                            result["vix_change"] = vals[0] - vals[1]
                        else:
                            result["vix_change"] = 0.0
                    else:
                        result["vix_change"] = 0.0
                else:
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

    def get_news(self) -> list[dict]:
        """Fetch news from all available sources."""
        articles = self.get_news_finnhub()
        articles.extend(self.get_news_rss())
        return articles

    # --- VIX Term Structure (yfinance, free) ---

    def get_vix_term_structure(self) -> dict:
        """Fetch VIX term structure: VIX9D, VIX3M, VIX6M, VVIX, SKEW."""
        tickers = {
            "vix9d": "^VIX9D",
            "vix3m": "^VIX3M",
            "vix6m": "^VIX6M",
            "vvix": "^VVIX",
            "skew": "^SKEW",
        }
        result = {}
        try:
            import yfinance as yf
            for name, ticker in tickers.items():
                try:
                    data = yf.download(ticker, period="5d", progress=False)
                    if not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                        result[name] = float(data["Close"].iloc[-1])
                    else:
                        result[name] = None
                except Exception as e:
                    logger.warning(f"VIX term {name} ({ticker}) failed: {e}")
                    result[name] = None
        except ImportError:
            logger.error("yfinance not installed")
        return result

    # --- Cross-Asset Signals (yfinance, free) ---

    def get_cross_asset_signals(self) -> dict:
        """Fetch cross-asset and breadth signals: HYG, LQD, TLT, EEM, copper/gold."""
        tickers = {
            "hyg": "HYG",       # High-yield corporate bonds
            "lqd": "LQD",       # Investment-grade corporate bonds
            "tlt": "TLT",       # 20+ year Treasury bonds
            "eem": "EEM",       # Emerging markets
            "spy": "SPY",       # For ratio computation
            "xlk": "XLK",       # Tech sector
            "xlf": "XLF",       # Financial sector
            "xle": "XLE",       # Energy sector
            "cper": "CPER",     # Copper ETF
            "gld": "GLD",       # Gold ETF
            "xlv": "XLV",       # Healthcare
            "xli": "XLI",       # Industrials
            "xlu": "XLU",       # Utilities (defensive)
            "xlb": "XLB",       # Materials
            "xlp": "XLP",       # Consumer Staples (defensive)
            "xly": "XLY",       # Consumer Discretionary
            "xlre": "XLRE",     # Real Estate
            "qqq": "QQQ",       # Nasdaq-100 (growth vs. value)
            "iwm": "IWM",       # Russell 2000 (small cap risk appetite)
            "dia": "DIA",       # Dow Jones (value vs. growth)
        }
        prices = {}
        try:
            import yfinance as yf
            for name, ticker in tickers.items():
                try:
                    data = yf.download(ticker, period="10d", progress=False)
                    if not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                        prices[name] = float(data["Close"].iloc[-1])
                        # Also get 5-day change for momentum
                        if len(data) >= 5:
                            prices[f"{name}_5d_chg"] = float(
                                (data["Close"].iloc[-1] / data["Close"].iloc[-5] - 1) * 100
                            )
                    else:
                        prices[name] = None
                except Exception as e:
                    logger.warning(f"Cross-asset {name} ({ticker}) failed: {e}")
                    prices[name] = None
        except ImportError:
            logger.error("yfinance not installed")
            return {}

        # Compute derived ratios
        result = {}
        spy_p = prices.get("spy")
        hyg_p = prices.get("hyg")
        lqd_p = prices.get("lqd")
        tlt_p = prices.get("tlt")
        eem_p = prices.get("eem")
        cper_p = prices.get("cper")
        gld_p = prices.get("gld")

        result["hy_spread"] = (hyg_p / lqd_p) if hyg_p and lqd_p else None
        result["tlt_spy_ratio"] = (tlt_p / spy_p) if tlt_p and spy_p else None
        result["eem_spy_ratio"] = (eem_p / spy_p) if eem_p and spy_p else None
        result["copper_gold_ratio"] = (cper_p / gld_p) if cper_p and gld_p else None

        # Sector rotation: tech vs financials relative strength
        xlk_p = prices.get("xlk")
        xlf_p = prices.get("xlf")
        xle_p = prices.get("xle")
        result["xlk_xlf_ratio"] = (xlk_p / xlf_p) if xlk_p and xlf_p else None
        result["xlk_xle_ratio"] = (xlk_p / xle_p) if xlk_p and xle_p else None

        # Pass through sector ETF prices for rotation features
        for etf in ["xlk", "xlf", "xle", "xlv", "xli", "xlu", "xlb", "xlp", "xly", "xlre", "qqq", "iwm", "dia"]:
            result[etf] = prices.get(etf)

        return result
