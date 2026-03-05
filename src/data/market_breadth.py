"""Market Breadth & Index Fundamentals — S&P 500 specific features.

Computes:
  1. Index Fundamentals: P/E ratio, earnings yield, dividend yield from SPY ETF info
  2. Market Breadth: % stocks above 50/200-day MA, advance/decline ratio, new highs/lows

Data sources (all free, no API keys):
  - yfinance: SPY info for P/E, dividend yield; S&P 500 constituent prices
  - Wikipedia: S&P 500 constituent list
  - Shiller dataset (datahub.io): Historical CAPE ratio (monthly, backfill)

Usage:
    from src.data.market_breadth import fetch_index_fundamentals, fetch_market_breadth

    fundamentals = fetch_index_fundamentals()
    # {'sp500_pe': 28.5, 'sp500_earnings_yield': 0.035, 'sp500_dividend_yield': 0.013, 'sp500_cape': 38.2}

    breadth = fetch_market_breadth()
    # {'pct_above_sma50': 0.65, 'pct_above_sma200': 0.72, 'advance_decline_ratio': 1.3, ...}
"""

import logging
import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Cache S&P 500 tickers (refreshed once per day)
_sp500_tickers_cache = {"tickers": None, "updated": 0}
_TICKER_CACHE_TTL = 86400  # 24 hours


def _get_sp500_tickers() -> list[str]:
    """Get current S&P 500 constituent tickers from Wikipedia."""
    now = time.time()
    if _sp500_tickers_cache["tickers"] and (now - _sp500_tickers_cache["updated"]) < _TICKER_CACHE_TTL:
        return _sp500_tickers_cache["tickers"]

    # Method 1: Wikipedia with proper headers
    try:
        import requests as req
        resp = req.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "StockAnalysis/2.7 (research; Python/3.12)"},
            timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        _sp500_tickers_cache["tickers"] = tickers
        _sp500_tickers_cache["updated"] = now
        logger.info("Loaded %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as e:
        logger.warning("Wikipedia S&P 500 list failed: %s — trying datahub fallback", e)

    # Method 2: Fallback — use a hardcoded top-100 subset for breadth approximation
    # These are the largest S&P 500 components by market cap (covers ~65% of index weight)
    try:
        import yfinance as yf
        # Get tickers from the SPY ETF holdings (top holdings)
        spy = yf.Ticker("SPY")
        # yfinance doesn't reliably expose all holdings, so use sector ETFs as proxy
        # Instead, use a curated list of major S&P 500 components
        pass
    except Exception:
        pass

    # Method 3: Static fallback — top 100 S&P 500 by weight (updated periodically)
    _FALLBACK_TOP100 = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "XOM",
        "JNJ", "JPM", "V", "PG", "MA", "HD", "AVGO", "LLY", "MRK", "COST",
        "ABBV", "PEP", "KO", "ADBE", "WMT", "CRM", "TMO", "MCD", "CSCO", "ACN",
        "ABT", "NFLX", "DHR", "LIN", "TXN", "AMD", "CMCSA", "NEE", "PM", "ORCL",
        "INTC", "UPS", "RTX", "HON", "AMGN", "LOW", "UNP", "QCOM", "BA", "SPGI",
        "GS", "CAT", "ELV", "SBUX", "INTU", "BLK", "ISRG", "MDLZ", "GILD", "ADP",
        "DE", "BKNG", "PLD", "VRTX", "SYK", "MMC", "REGN", "ADI", "LRCX", "CI",
        "ZTS", "PANW", "SNPS", "CDNS", "CB", "ETN", "BSX", "KLAC", "MO", "SO",
        "DUK", "CME", "SHW", "BDX", "CL", "ICE", "FI", "EOG", "SLB", "PNC",
        "NOC", "APD", "TGT", "PYPL", "USB", "WM", "MCK", "ANET", "MSI", "GD",
    ]
    logger.info("Using fallback top-100 S&P 500 tickers for breadth approximation")
    _sp500_tickers_cache["tickers"] = _FALLBACK_TOP100
    _sp500_tickers_cache["updated"] = now
    return _FALLBACK_TOP100


def fetch_index_fundamentals() -> dict:
    """Fetch S&P 500 index-level fundamental data from yfinance.

    Returns dict with:
      sp500_pe: trailing P/E ratio
      sp500_forward_pe: forward P/E ratio
      sp500_earnings_yield: 1/PE (inverse of trailing PE)
      sp500_dividend_yield: trailing 12-month dividend yield
    """
    result = {
        "sp500_pe": None,
        "sp500_forward_pe": None,
        "sp500_earnings_yield": None,
        "sp500_dividend_yield": None,
        "sp500_cape": None,
        "buffett_indicator": None,
    }
    try:
        import yfinance as yf
        spy = yf.Ticker("SPY")
        info = spy.info or {}

        pe = info.get("trailingPE")
        fwd_pe = info.get("forwardPE")
        div_yield = info.get("trailingAnnualDividendYield") or info.get("dividendYield")

        if pe and pe > 0:
            result["sp500_pe"] = round(float(pe), 2)
            result["sp500_earnings_yield"] = round(1.0 / float(pe), 4)
        if fwd_pe and fwd_pe > 0:
            result["sp500_forward_pe"] = round(float(fwd_pe), 2)
        if div_yield is not None:
            result["sp500_dividend_yield"] = round(float(div_yield), 4)

        logger.info("Index fundamentals: PE=%.1f, EY=%.3f, DY=%.3f",
                     result["sp500_pe"] or 0,
                     result["sp500_earnings_yield"] or 0,
                     result["sp500_dividend_yield"] or 0)
    except Exception as e:
        logger.warning("fetch_index_fundamentals failed: %s", e)

    # --- Shiller CAPE Ratio (from Shiller's dataset via datahub.io) ---
    try:
        cape_url = "https://datahub.io/core/s-and-p-500/r/data.csv"
        cape_df = pd.read_csv(cape_url, timeout=15)
        cape_col = next((c for c in cape_df.columns if "PE10" in c.upper() or "CAPE" in c.upper()), None)
        if cape_col:
            cape_df[cape_col] = pd.to_numeric(cape_df[cape_col], errors="coerce")
            cape_df = cape_df.dropna(subset=[cape_col])
            if not cape_df.empty:
                result["sp500_cape"] = round(float(cape_df[cape_col].iloc[-1]), 2)
                logger.info("Shiller CAPE: %.2f", result["sp500_cape"])
    except Exception as e:
        logger.warning("CAPE fetch failed: %s", e)

    # --- Buffett Indicator: Wilshire 5000 / GDP (from FRED CSV, free) ---
    try:
        w5000_df = pd.read_csv(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WILL5000PR",
            timeout=15
        )
        w5000_df.columns = ["date", "value"]
        w5000_df["value"] = pd.to_numeric(w5000_df["value"], errors="coerce")
        w5000_df = w5000_df.dropna(subset=["value"])

        gdp_df = pd.read_csv(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP",
            timeout=15
        )
        gdp_df.columns = ["date", "gdp"]
        gdp_df["gdp"] = pd.to_numeric(gdp_df["gdp"], errors="coerce")
        gdp_df = gdp_df.dropna(subset=["gdp"])

        if not w5000_df.empty and not gdp_df.empty:
            latest_w5000 = float(w5000_df["value"].iloc[-1])
            latest_gdp = float(gdp_df["gdp"].iloc[-1])
            if latest_gdp > 0:
                result["buffett_indicator"] = round((latest_w5000 / latest_gdp) * 100, 2)
                logger.info("Buffett Indicator: %.1f%%", result["buffett_indicator"])
    except Exception as e:
        logger.warning("Buffett Indicator fetch failed: %s", e)

    return result


def fetch_market_breadth(lookback_days: int = 250) -> dict:
    """Compute S&P 500 market breadth indicators.

    Downloads recent price history for all S&P 500 constituents and computes:
      pct_above_sma50: fraction of stocks with close > 50-day SMA
      pct_above_sma200: fraction of stocks with close > 200-day SMA
      advance_decline_ratio: advancing / declining stocks (today vs yesterday)
      new_highs_52w: count of stocks at 52-week high
      new_lows_52w: count of stocks at 52-week low
      breadth_thrust: net advances / total stocks (McClellan-style)

    This is computationally expensive (~2-3 min for 500 tickers).
    Should be called once per day, results cached in DB.
    """
    result = {
        "pct_above_sma50": None,
        "pct_above_sma200": None,
        "advance_decline_ratio": None,
        "new_highs_52w": None,
        "new_lows_52w": None,
        "breadth_thrust": None,
    }

    tickers = _get_sp500_tickers()
    if not tickers:
        logger.warning("No S&P 500 tickers available for breadth calculation")
        return result

    try:
        import yfinance as yf
        # Download all tickers in one batch (much faster than individual)
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 50)  # extra buffer for SMA calc

        logger.info("Downloading %d S&P 500 tickers for breadth calculation...", len(tickers))
        data = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"),
                           progress=False, threads=True)

        if data.empty:
            logger.warning("No data returned for S&P 500 breadth")
            return result

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data[["Close"]]

        if closes.empty or len(closes) < 50:
            logger.warning("Insufficient data for breadth calculation")
            return result

        # Drop tickers with too many NaNs
        valid_mask = closes.notna().sum() > 50
        closes = closes.loc[:, valid_mask]
        n_stocks = closes.shape[1]

        if n_stocks < 100:
            logger.warning("Only %d valid tickers, breadth may be unreliable", n_stocks)

        # Latest close
        latest = closes.iloc[-1]
        prev = closes.iloc[-2] if len(closes) > 1 else latest

        # SMA calculations
        sma50 = closes.rolling(50).mean().iloc[-1]
        sma200 = closes.rolling(200).mean().iloc[-1]

        # % above SMA
        above_50 = (latest > sma50).sum()
        above_200 = (latest > sma200).sum()
        valid_50 = sma50.notna().sum()
        valid_200 = sma200.notna().sum()

        if valid_50 > 0:
            result["pct_above_sma50"] = round(above_50 / valid_50, 4)
        if valid_200 > 0:
            result["pct_above_sma200"] = round(above_200 / valid_200, 4)

        # Advance/Decline
        advances = (latest > prev).sum()
        declines = (latest < prev).sum()
        unchanged = n_stocks - advances - declines
        if declines > 0:
            result["advance_decline_ratio"] = round(advances / declines, 3)
        elif advances > 0:
            result["advance_decline_ratio"] = float(advances)  # all advancing

        # Breadth thrust = net advances / total
        result["breadth_thrust"] = round((advances - declines) / max(n_stocks, 1), 4)

        # 52-week highs/lows
        if len(closes) >= 252:
            high_52w = closes.iloc[-252:].max()
            low_52w = closes.iloc[-252:].min()
            result["new_highs_52w"] = int((latest >= high_52w * 0.99).sum())  # within 1% of high
            result["new_lows_52w"] = int((latest <= low_52w * 1.01).sum())    # within 1% of low
        else:
            high_all = closes.max()
            low_all = closes.min()
            result["new_highs_52w"] = int((latest >= high_all * 0.99).sum())
            result["new_lows_52w"] = int((latest <= low_all * 1.01).sum())

        logger.info("Market breadth: %%>50SMA=%.1f%%, %%>200SMA=%.1f%%, A/D=%.2f, "
                     "highs=%d, lows=%d",
                     (result["pct_above_sma50"] or 0) * 100,
                     (result["pct_above_sma200"] or 0) * 100,
                     result["advance_decline_ratio"] or 0,
                     result["new_highs_52w"] or 0,
                     result["new_lows_52w"] or 0)

    except Exception as e:
        logger.warning("fetch_market_breadth failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def store_breadth_fundamentals(router, date: str, fundamentals: dict, breadth: dict):
    """Store breadth and fundamental data in the market_breadth table."""
    merged = {**fundamentals, **breadth}
    cols = ["sp500_pe", "sp500_forward_pe", "sp500_earnings_yield", "sp500_dividend_yield",
            "pct_above_sma50", "pct_above_sma200", "advance_decline_ratio",
            "new_highs_52w", "new_lows_52w", "breadth_thrust",
            "sp500_cape", "buffett_indicator"]
    # Cast numpy types to native Python to avoid PostgreSQL serialization errors
    def _native(v):
        if v is None:
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        return v
    values = [_native(merged.get(c)) for c in cols]

    sql = (
        "INSERT OR REPLACE INTO market_breadth "
        "(date, " + ", ".join(cols) + ") "
        "VALUES (?, " + ", ".join(["?"] * len(cols)) + ")"
    )
    try:
        router.execute(sql, tuple([date] + values))
        logger.info("Stored market breadth for %s", date)
    except Exception as e:
        logger.error("store_breadth_fundamentals failed: %s", e)


def get_breadth_history(router, days: int = 504) -> pd.DataFrame:
    """Load historical breadth data from DB."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return router.query(
        "SELECT * FROM market_breadth WHERE date >= ? ORDER BY date", (cutoff,)
    )
