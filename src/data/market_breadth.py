"""Market Breadth & Index Fundamentals — S&P 500 specific features.

Computes:
  1. Index Fundamentals: P/E ratio, earnings yield, dividend yield from SPY ETF info
  2. Market Breadth: % stocks above 50/200-day MA, advance/decline ratio, new highs/lows
  3. Market Concentration: top-N contribution, sector participation, HHI

Data sources:
  - Polygon.io (primary): Grouped daily bars for all US stocks in one API call
  - yfinance (fallback): S&P 500 constituent prices if Polygon unavailable
  - Wikipedia: S&P 500 constituent list + GICS sector mapping
  - Shiller dataset (datahub.io): Historical CAPE ratio (monthly, backfill)
"""

import logging
import time
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_polygon_fetcher():
    """Get a PolygonFetcher instance using the encrypted API key. Returns None if unavailable."""
    try:
        from src.data.secrets_manager import get_secret
        key = get_secret("polygon_api_key")
        if not key:
            return None
        from src.data.polygon_fetcher import PolygonFetcher
        return PolygonFetcher(key)
    except Exception as e:
        logger.debug("Polygon fetcher unavailable: %s", e)
        return None

# Cache S&P 500 tickers + sector mapping (refreshed once per day)
_sp500_tickers_cache = {"tickers": None, "sectors": None, "updated": 0}
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
        # Also cache GICS sector mapping for concentration analysis
        if "GICS Sector" in df.columns:
            sectors = dict(zip(
                df["Symbol"].str.replace(".", "-", regex=False),
                df["GICS Sector"]
            ))
            _sp500_tickers_cache["sectors"] = sectors
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
        import requests as _req
        from io import StringIO
        cape_resp = _req.get("https://datahub.io/core/s-and-p-500/r/data.csv", timeout=15)
        cape_resp.raise_for_status()
        cape_df = pd.read_csv(StringIO(cape_resp.text))
        cape_col = next((c for c in cape_df.columns if "PE10" in c.upper() or "CAPE" in c.upper()), None)
        if cape_col:
            cape_df[cape_col] = pd.to_numeric(cape_df[cape_col], errors="coerce")
            # Skip zero/null values (dataset lags — recent months may be 0)
            cape_valid = cape_df[cape_df[cape_col] > 0]
            if not cape_valid.empty:
                result["sp500_cape"] = round(float(cape_valid[cape_col].iloc[-1]), 2)
                logger.info("Shiller CAPE: %.2f", result["sp500_cape"])
    except Exception as e:
        logger.warning("CAPE fetch failed: %s", e)

    # --- Buffett Indicator: Total Market Cap / GDP ---
    # FRED WILL5000PR/WILL5000IND series are retired (404).
    # Use yfinance ^W5000 (Wilshire 5000 Full Cap Index) instead.
    # 1 index point ≈ $1 billion in total US market cap.
    # Formula: Buffett Indicator = (^W5000 value / FRED GDP in billions) × 100
    try:
        import yfinance as yf
        import requests as _req

        # Fetch Wilshire 5000 via yfinance
        w5k = yf.Ticker("^W5000")
        w5k_hist = w5k.history(period="5d")
        if not w5k_hist.empty:
            w5000_val = float(w5k_hist["Close"].dropna().iloc[-1])

            # Fetch GDP from FRED API
            gdp_val = None
            fred_key = None
            try:
                from src.data.secrets_manager import get_secret
                fred_key = get_secret("fred_api_key")
            except Exception:
                pass

            if fred_key:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": "GDP", "api_key": fred_key,
                    "file_type": "json", "sort_order": "desc", "limit": 5,
                }
                resp = _req.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    for o in resp.json().get("observations", []):
                        v = o.get("value", ".")
                        if v != ".":
                            gdp_val = float(v)
                            break

            if w5000_val > 0 and gdp_val and gdp_val > 0:
                result["buffett_indicator"] = round((w5000_val / gdp_val) * 100, 2)
                logger.info("Buffett Indicator: %.1f%% (W5000=%.0f, GDP=%.0f)",
                            result["buffett_indicator"], w5000_val, gdp_val)
            else:
                logger.warning("Buffett Indicator: W5000=%.0f, GDP=%s — cannot compute",
                               w5000_val, gdp_val)
        else:
            logger.warning("Buffett Indicator: ^W5000 returned no data")
    except Exception as e:
        logger.warning("Buffett Indicator fetch failed: %s", e)

    return result


def fetch_market_breadth(lookback_days: int = 250) -> dict:
    """Compute S&P 500 market breadth indicators.

    Uses Polygon.io grouped daily bars as primary source (2 API calls for
    today + yesterday), falling back to yfinance batch download.

    For SMA-based metrics (pct_above_sma50/200), uses yfinance since those
    need 200+ days of history which would require too many Polygon calls.

    Returns dict with breadth + concentration features.
    """
    result = {
        "pct_above_sma50": None,
        "pct_above_sma200": None,
        "advance_decline_ratio": None,
        "new_highs_52w": None,
        "new_lows_52w": None,
        "breadth_thrust": None,
        "trin": None,
    }

    tickers = _get_sp500_tickers()
    if not tickers:
        logger.warning("No S&P 500 tickers available for breadth calculation")
        return result

    ticker_set = set(tickers)
    closes = None  # Will hold full lookback DataFrame for SMA calcs
    polygon_today = None  # Single-day DataFrame from Polygon
    polygon_prev = None

    # --- Strategy 1: Polygon grouped daily for today + yesterday ---
    # Gets advance/decline, concentration, TRIN from 2 API calls
    pf = _get_polygon_fetcher()
    if pf:
        try:
            today = datetime.now()
            # Try last 5 business days to find 2 valid trading days
            dates_to_try = pd.bdate_range(
                end=today.strftime("%Y-%m-%d"),
                periods=5
            ).strftime("%Y-%m-%d").tolist()[::-1]  # most recent first

            found_days = []
            for d in dates_to_try:
                if len(found_days) >= 2:
                    break
                df = pf.get_grouped_daily(d)
                if not df.empty:
                    df = df[df["ticker"].isin(ticker_set)]
                    if len(df) >= 100:
                        found_days.append((d, df))

            if len(found_days) >= 2:
                polygon_today = found_days[0][1].set_index("ticker")
                polygon_prev = found_days[1][1].set_index("ticker")
                common = polygon_today.index.intersection(polygon_prev.index)
                logger.info("Polygon grouped daily: %d tickers for %s, %d for %s, %d common",
                            len(polygon_today), found_days[0][0],
                            len(polygon_prev), found_days[1][0], len(common))

                latest = polygon_today.loc[common, "close"]
                prev = polygon_prev.loc[common, "close"]
                n_stocks = len(common)

                # Advance/Decline
                advances = (latest > prev).sum()
                declines = (latest < prev).sum()
                if declines > 0:
                    result["advance_decline_ratio"] = round(advances / declines, 3)
                elif advances > 0:
                    result["advance_decline_ratio"] = float(advances)

                # Breadth thrust
                result["breadth_thrust"] = round((advances - declines) / max(n_stocks, 1), 4)

                # TRIN
                if "volume" in polygon_today.columns and "volume" in polygon_prev.columns:
                    try:
                        vol_today = polygon_today.loc[common, "volume"]
                        adv_mask = latest > prev
                        decl_mask = latest < prev
                        adv_vol = vol_today[adv_mask].sum()
                        decl_vol = vol_today[decl_mask].sum()
                        if decl_vol > 0 and advances > 0 and declines > 0:
                            trin = (advances / declines) / (adv_vol / decl_vol)
                            result["trin"] = round(float(trin), 4)
                    except Exception as e:
                        logger.debug("TRIN from Polygon failed: %s", e)

                # Concentration analysis from Polygon data
                try:
                    # Build a 2-row closes DataFrame for concentration calc
                    poly_closes = pd.DataFrame({
                        t: [polygon_prev.loc[t, "close"], polygon_today.loc[t, "close"]]
                        for t in common
                    }, index=[found_days[1][0], found_days[0][0]])
                    conc = fetch_market_concentration(closes=poly_closes)
                    result.update(conc)
                except Exception as e:
                    logger.warning("Concentration from Polygon failed: %s", e)

                logger.info("Polygon breadth: A/D=%.2f, thrust=%.4f, TRIN=%.4f",
                            result["advance_decline_ratio"] or 0,
                            result["breadth_thrust"] or 0,
                            result["trin"] or 0)
            elif len(found_days) == 1:
                logger.warning("Polygon: only found 1 trading day, need 2 for breadth")
            else:
                logger.warning("Polygon grouped daily returned no valid data")
        except Exception as e:
            logger.warning("Polygon breadth fetch failed: %s — falling back to yfinance", e)

    # --- Strategy 2: yfinance for SMA-based metrics + fallback ---
    # SMA calculations need 200+ days of history
    try:
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 50)

        logger.info("Downloading %d S&P 500 tickers via yfinance for SMA breadth...", len(tickers))
        data = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"),
                           progress=False, threads=True)

        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"]
                volumes = data["Volume"] if "Volume" in data.columns.get_level_values(0) else None
            else:
                closes = data[["Close"]]
                volumes = None

            if closes is not None and not closes.empty and len(closes) >= 50:
                valid_mask = closes.notna().sum() > 50
                closes = closes.loc[:, valid_mask]
                if volumes is not None:
                    common_cols = closes.columns.intersection(volumes.columns)
                    volumes = volumes[common_cols]
                n_stocks = closes.shape[1]

                latest = closes.iloc[-1]
                prev_close = closes.iloc[-2] if len(closes) > 1 else latest

                # SMA calculations
                sma50 = closes.rolling(50).mean().iloc[-1]
                sma200 = closes.rolling(200).mean().iloc[-1]

                above_50 = (latest > sma50).sum()
                above_200 = (latest > sma200).sum()
                valid_50 = sma50.notna().sum()
                valid_200 = sma200.notna().sum()

                if valid_50 > 0:
                    result["pct_above_sma50"] = round(above_50 / valid_50, 4)
                if valid_200 > 0:
                    result["pct_above_sma200"] = round(above_200 / valid_200, 4)

                # 52-week highs/lows
                if len(closes) >= 252:
                    high_52w = closes.iloc[-252:].max()
                    low_52w = closes.iloc[-252:].min()
                    result["new_highs_52w"] = int((latest >= high_52w * 0.99).sum())
                    result["new_lows_52w"] = int((latest <= low_52w * 1.01).sum())
                else:
                    high_all = closes.max()
                    low_all = closes.min()
                    result["new_highs_52w"] = int((latest >= high_all * 0.99).sum())
                    result["new_lows_52w"] = int((latest <= low_all * 1.01).sum())

                # If Polygon didn't provide advance/decline, compute from yfinance
                if result["advance_decline_ratio"] is None:
                    advances = (latest > prev_close).sum()
                    declines = (latest < prev_close).sum()
                    if declines > 0:
                        result["advance_decline_ratio"] = round(advances / declines, 3)
                    elif advances > 0:
                        result["advance_decline_ratio"] = float(advances)
                    result["breadth_thrust"] = round(
                        (advances - declines) / max(n_stocks, 1), 4)

                # If Polygon didn't provide TRIN, compute from yfinance
                if result["trin"] is None and volumes is not None and len(closes) > 1:
                    try:
                        vol_latest = volumes.iloc[-1]
                        adv_mask = latest > prev_close
                        decl_mask = latest < prev_close
                        vol_aligned = vol_latest.reindex(adv_mask.index)
                        adv_vol = vol_aligned[adv_mask].sum()
                        decl_vol = vol_aligned[decl_mask].sum()
                        advances = adv_mask.sum()
                        declines = decl_mask.sum()
                        if decl_vol > 0 and advances > 0 and declines > 0:
                            trin = (advances / declines) / (adv_vol / decl_vol)
                            result["trin"] = round(float(trin), 4)
                    except Exception as e:
                        logger.debug("TRIN from yfinance failed: %s", e)

                # If Polygon didn't provide concentration, compute from yfinance
                if result.get("top5_contribution") is None:
                    try:
                        conc = fetch_market_concentration(closes=closes)
                        result.update(conc)
                    except Exception as e:
                        logger.warning("Concentration from yfinance failed: %s", e)

                logger.info("Market breadth: %%>50SMA=%.1f%%, %%>200SMA=%.1f%%, A/D=%.2f, "
                            "highs=%d, lows=%d",
                            (result["pct_above_sma50"] or 0) * 100,
                            (result["pct_above_sma200"] or 0) * 100,
                            result["advance_decline_ratio"] or 0,
                            result["new_highs_52w"] or 0,
                            result["new_lows_52w"] or 0)
    except Exception as e:
        logger.warning("yfinance breadth fetch failed: %s", e)

    return result


def _get_sp500_sectors() -> dict:
    """Return ticker→GICS sector mapping. Populated by _get_sp500_tickers()."""
    if not _sp500_tickers_cache.get("sectors"):
        _get_sp500_tickers()  # triggers Wikipedia fetch which populates sectors
    return _sp500_tickers_cache.get("sectors") or {}


def fetch_market_concentration(closes: pd.DataFrame = None,
                               lookback_days: int = 250) -> dict:
    """Compute market concentration and sector participation features.

    Measures whether SPY's move is driven by a few mega-caps (narrow leadership)
    or broad-based participation. Narrow rallies are fragile and more likely to
    reverse; broad moves are more durable.

    Args:
        closes: DataFrame of S&P 500 constituent close prices (tickers as columns).
                If None, downloads fresh data via yfinance.
        lookback_days: Days of history for download (only used if closes is None).

    Returns dict with:
        top5_contribution: % of SPY's daily return explained by top 5 stocks
        top10_contribution: % of SPY's daily return explained by top 10 stocks
        sector_participation: fraction of 11 GICS sectors moving same direction as SPY
        breadth_divergence: sign(SPY return) vs sign(median stock return) — 1=aligned, -1=divergent
        herfindahl_return: Herfindahl index of squared return contributions (concentration measure)
        pct_stocks_same_dir: % of stocks moving in same direction as SPY
    """
    result = {
        "top5_contribution": None,
        "top10_contribution": None,
        "sector_participation": None,
        "breadth_divergence": None,
        "herfindahl_return": None,
        "pct_stocks_same_dir": None,
    }

    try:
        if closes is None:
            import yfinance as yf
            tickers = _get_sp500_tickers()
            if not tickers:
                return result
            end = datetime.now()
            start = end - timedelta(days=lookback_days + 50)
            data = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                               end=end.strftime("%Y-%m-%d"),
                               progress=False, threads=True)
            if data.empty:
                return result
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"]
            else:
                closes = data[["Close"]]

        if closes.empty or len(closes) < 2:
            return result

        # Drop tickers with too many NaNs (adaptive threshold based on DataFrame length)
        min_valid = min(50, max(2, len(closes) - 1))
        valid_mask = closes.notna().sum() >= min_valid
        closes = closes.loc[:, valid_mask]
        n_stocks = closes.shape[1]
        if n_stocks < 50:
            logger.warning("Only %d valid tickers for concentration analysis", n_stocks)
            return result

        # Daily returns for all stocks
        daily_rets = closes.pct_change(fill_method=None)
        latest_rets = daily_rets.iloc[-1].dropna()
        if len(latest_rets) < 50:
            return result

        # SPY return approximation: equal-weight average of all constituents
        # (cap-weighted would be better but we don't have weights)
        spy_ret = latest_rets.mean()

        # --- Top-N contribution ---
        # Sort stocks by absolute return contribution (magnitude)
        abs_rets = latest_rets.abs().sort_values(ascending=False)
        total_abs = abs_rets.sum()
        if total_abs > 0:
            result["top5_contribution"] = round(
                float(abs_rets.iloc[:5].sum() / total_abs), 4)
            result["top10_contribution"] = round(
                float(abs_rets.iloc[:10].sum() / total_abs), 4)

        # --- Herfindahl index of return concentration ---
        # HHI of squared return shares — higher = more concentrated
        if total_abs > 0:
            shares = (abs_rets / total_abs) ** 2
            result["herfindahl_return"] = round(float(shares.sum()), 6)

        # --- % stocks moving same direction as SPY ---
        if spy_ret > 0:
            same_dir = (latest_rets > 0).sum()
        elif spy_ret < 0:
            same_dir = (latest_rets < 0).sum()
        else:
            same_dir = len(latest_rets)
        result["pct_stocks_same_dir"] = round(same_dir / len(latest_rets), 4)

        # --- Breadth divergence ---
        # +1 if SPY direction matches median stock direction, -1 if divergent
        median_ret = latest_rets.median()
        if spy_ret == 0 or median_ret == 0:
            result["breadth_divergence"] = 0.0
        elif np.sign(spy_ret) == np.sign(median_ret):
            result["breadth_divergence"] = 1.0
        else:
            result["breadth_divergence"] = -1.0

        # --- Sector participation ---
        # What fraction of GICS sectors are moving in the same direction as SPY?
        sectors = _get_sp500_sectors()
        if sectors:
            # Build sector→mean return
            sector_rets = {}
            for ticker, ret in latest_rets.items():
                sec = sectors.get(ticker)
                if sec:
                    sector_rets.setdefault(sec, []).append(ret)
            sector_means = {s: np.mean(rs) for s, rs in sector_rets.items() if rs}
            if sector_means and spy_ret != 0:
                same_dir_sectors = sum(
                    1 for s_ret in sector_means.values()
                    if np.sign(s_ret) == np.sign(spy_ret)
                )
                result["sector_participation"] = round(
                    same_dir_sectors / len(sector_means), 4)
            elif sector_means:
                result["sector_participation"] = 0.5  # flat day

        logger.info("Market concentration: top5=%.1f%%, top10=%.1f%%, "
                     "sector_part=%.1f%%, same_dir=%.1f%%, HHI=%.4f",
                     (result["top5_contribution"] or 0) * 100,
                     (result["top10_contribution"] or 0) * 100,
                     (result["sector_participation"] or 0) * 100,
                     (result["pct_stocks_same_dir"] or 0) * 100,
                     result["herfindahl_return"] or 0)

    except Exception as e:
        logger.warning("fetch_market_concentration failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def store_breadth_fundamentals(router, date: str, fundamentals: dict, breadth: dict):
    """Store breadth and fundamental data in the market_breadth table.
    
    Uses upsert logic that preserves existing non-null values when new values
    are null (e.g., if a data source is temporarily unavailable).
    """
    merged = {**fundamentals, **breadth}
    cols = ["sp500_pe", "sp500_forward_pe", "sp500_earnings_yield", "sp500_dividend_yield",
            "pct_above_sma50", "pct_above_sma200", "advance_decline_ratio",
            "new_highs_52w", "new_lows_52w", "breadth_thrust",
            "sp500_cape", "buffett_indicator", "fear_greed_index", "trin",
            "top5_contribution", "top10_contribution", "sector_participation",
            "breadth_divergence", "herfindahl_return", "pct_stocks_same_dir"]
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

    # First try INSERT, then UPDATE only non-null values to preserve existing data
    insert_sql = (
        "INSERT OR REPLACE INTO market_breadth "
        "(date, " + ", ".join(cols) + ") "
        "VALUES (?, " + ", ".join(["?"] * len(cols)) + ")"
    )
    try:
        # Check if row exists
        existing = router.read_analytics(
            "SELECT date FROM market_breadth WHERE date = ?", (date,)
        )
        if existing.empty:
            # No existing row — insert all values
            router.execute(insert_sql, tuple([date] + values))
        else:
            # Row exists — only update columns where new value is not None
            updates = []
            update_vals = []
            for col, val in zip(cols, values):
                if val is not None:
                    updates.append(f"{col} = ?")
                    update_vals.append(val)
            if updates:
                update_sql = "UPDATE market_breadth SET " + ", ".join(updates) + " WHERE date = ?"
                update_vals.append(date)
                router.execute(update_sql, tuple(update_vals))
        logger.info("Stored market breadth for %s", date)
    except Exception as e:
        logger.error("store_breadth_fundamentals failed: %s", e)


def get_breadth_history(router, days: int = 504) -> pd.DataFrame:
    """Load historical breadth data from DB."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return router.query(
        "SELECT * FROM market_breadth WHERE date >= ? ORDER BY date", (cutoff,)
    )
