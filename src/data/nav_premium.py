"""ETF NAV Premium/Discount Features — SPY vs underlying index.

SPY's market price can deviate from its Net Asset Value (the S&P 500 index
scaled by the trust's divisor). These deviations reveal supply/demand
imbalances in the ETF wrapper itself:

  - Premium (SPY > NAV): excess demand for SPY shares, often from
    institutional hedging or retail inflows → bullish pressure
  - Discount (SPY < NAV): redemption pressure or risk-off selling
    hitting the ETF before the underlying → bearish signal

Since real-time iNAV requires expensive feeds (NSCC/GIF), we compute
daily proxies using freely available data:

  1. SPY close vs S&P 500 index (^GSPC) — the core premium/discount
  2. SPY close vs ES futures (front month) — the basis spread
  3. Derived signals: z-score, momentum, mean-reversion, regime

Data sources: yfinance (free), with Polygon fallback for SPY/ES.

Features produced (12 total):
  - nav_premium_pct: (SPY_close / SPX_scaled - 1) * 100
  - nav_premium_zscore: 20-day z-score of premium
  - nav_premium_ma5: 5-day moving average of premium
  - nav_premium_momentum: premium change over 5 days
  - nav_premium_mean_rev: premium minus its 20-day mean (mean-reversion signal)
  - nav_premium_extreme: 1 if |z-score| > 2 (unusual dislocation)
  - spy_es_basis_pct: (SPY - ES_front) / SPY * 100
  - spy_es_basis_zscore: 20-day z-score of basis
  - nav_premium_vol: 20-day rolling std of premium (regime indicator)
  - nav_premium_skew: 20-day rolling skewness of premium
  - nav_premium_regime: 1=persistent premium, -1=persistent discount, 0=normal
  - nav_creation_pressure: premium * volume_ratio (premium weighted by volume)
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# SPY trust divisor: SPY ≈ SPX / 10 (approximate, drifts slightly over time)
# We compute the actual ratio dynamically from recent data.
_SPY_SPX_RATIO_APPROX = 10.0


def _fetch_yfinance(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV from yfinance for multiple tickers."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed")
        return {}

    result = {}
    try:
        data = yf.download(tickers, start=start, end=end,
                           progress=False, group_by="ticker")
        if data.empty:
            return {}
        for ticker in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex) and len(tickers) > 1:
                    t_data = data[ticker].dropna(how="all")
                else:
                    t_data = data.dropna(how="all")
                if t_data.empty:
                    continue
                t_data = t_data.reset_index()
                date_col = "Date" if "Date" in t_data.columns else "date"
                t_data["date"] = pd.to_datetime(t_data[date_col]).dt.strftime("%Y-%m-%d")
                rename_map = {}
                for col in t_data.columns:
                    if col.lower() == "close":
                        rename_map[col] = "close"
                    elif col.lower() == "volume":
                        rename_map[col] = "volume"
                    elif col.lower() == "open":
                        rename_map[col] = "open"
                    elif col.lower() == "high":
                        rename_map[col] = "high"
                    elif col.lower() == "low":
                        rename_map[col] = "low"
                t_data = t_data.rename(columns=rename_map)
                result[ticker] = t_data
            except Exception as e:
                logger.debug(f"yfinance parse {ticker}: {e}")
    except Exception as e:
        logger.warning(f"yfinance download failed: {e}")
    return result


def fetch_nav_premium_data(days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch SPY, S&P 500 index, and ES futures data to compute NAV premium.

    Args:
        days: Calendar days of history to fetch.

    Returns:
        DataFrame with date + all NAV premium features, or None.
    """
    end = datetime.now()
    start = end - timedelta(days=days + 10)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    # ^GSPC = S&P 500 index, ES=F = E-mini S&P 500 front month futures
    tickers = ["SPY", "^GSPC", "ES=F"]
    bars = _fetch_yfinance(tickers, start_str, end_str)

    if "SPY" not in bars or "^GSPC" not in bars:
        logger.warning("Missing SPY or ^GSPC data for NAV premium computation")
        return None

    spy_df = bars["SPY"][["date", "close", "volume"]].copy()
    spy_df = spy_df.rename(columns={"close": "spy_close", "volume": "spy_volume"})

    spx_df = bars["^GSPC"][["date", "close"]].copy()
    spx_df = spx_df.rename(columns={"close": "spx_close"})

    # Merge on date
    df = spy_df.merge(spx_df, on="date", how="inner")

    # ES futures (optional — may not always be available via yfinance)
    if "ES=F" in bars:
        es_df = bars["ES=F"][["date", "close"]].copy()
        es_df = es_df.rename(columns={"close": "es_close"})
        df = df.merge(es_df, on="date", how="left")
    else:
        df["es_close"] = np.nan

    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 5:
        logger.warning(f"Only {len(df)} rows for NAV premium — too few")
        return None

    # Convert to float
    for col in ["spy_close", "spx_close", "spy_volume", "es_close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Core: NAV premium/discount ---
    # Compute dynamic SPY/SPX ratio from the data (more accurate than fixed 10.0)
    # SPY ≈ SPX / divisor. The divisor drifts due to dividends/expenses.
    # We use the median ratio over the window as the "fair" divisor.
    ratio = df["spx_close"] / df["spy_close"]
    fair_divisor = ratio.rolling(20, min_periods=5).median()
    # Fill early rows with overall median
    fair_divisor = fair_divisor.fillna(ratio.median())

    # NAV proxy = SPX / fair_divisor
    nav_proxy = df["spx_close"] / fair_divisor

    # Premium = (SPY - NAV) / NAV * 100 (in percentage points)
    df["nav_premium_pct"] = ((df["spy_close"] - nav_proxy) / nav_proxy * 100)

    # --- Derived features ---
    premium = df["nav_premium_pct"]

    # Z-score (20-day)
    prem_mean = premium.rolling(20, min_periods=5).mean()
    prem_std = premium.rolling(20, min_periods=5).std()
    df["nav_premium_zscore"] = ((premium - prem_mean) / prem_std.replace(0, np.nan)).fillna(0)

    # 5-day moving average (smoothed signal)
    df["nav_premium_ma5"] = premium.rolling(5, min_periods=1).mean()

    # Momentum (5-day change in premium)
    df["nav_premium_momentum"] = premium.diff(5)

    # Mean-reversion signal (current premium minus 20-day mean)
    df["nav_premium_mean_rev"] = premium - prem_mean

    # Extreme dislocation flag
    df["nav_premium_extreme"] = (df["nav_premium_zscore"].abs() > 2).astype(int)

    # --- SPY vs ES basis ---
    if df["es_close"].notna().sum() > 5:
        # ES futures trade at index level (~5900) while SPY ≈ SPX/10 (~590)
        # Scale ES to SPY-equivalent using the same dynamic divisor
        es_scaled = df["es_close"] / fair_divisor
        df["spy_es_basis_pct"] = ((df["spy_close"] - es_scaled) /
                                   df["spy_close"].replace(0, np.nan) * 100)
        basis = df["spy_es_basis_pct"]
        basis_mean = basis.rolling(20, min_periods=5).mean()
        basis_std = basis.rolling(20, min_periods=5).std()
        df["spy_es_basis_zscore"] = ((basis - basis_mean) /
                                      basis_std.replace(0, np.nan)).fillna(0)
    else:
        df["spy_es_basis_pct"] = 0.0
        df["spy_es_basis_zscore"] = 0.0

    # --- Volatility of premium (regime indicator) ---
    df["nav_premium_vol"] = premium.rolling(20, min_periods=5).std().fillna(0)

    # --- Skewness of premium (asymmetric demand) ---
    df["nav_premium_skew"] = premium.rolling(20, min_periods=10).skew().fillna(0)

    # --- Premium regime: persistent premium vs discount ---
    # If 5-day MA > +0.01% → creation pressure (premium regime)
    # If 5-day MA < -0.01% → redemption pressure (discount regime)
    ma5 = df["nav_premium_ma5"]
    df["nav_premium_regime"] = np.where(ma5 > 0.01, 1,
                                np.where(ma5 < -0.01, -1, 0))

    # --- Creation/redemption pressure: premium weighted by relative volume ---
    vol_ma20 = df["spy_volume"].rolling(20, min_periods=5).mean()
    vol_ratio = (df["spy_volume"] / vol_ma20.replace(0, np.nan)).fillna(1)
    df["nav_creation_pressure"] = premium * vol_ratio

    # Select output columns
    feature_cols = [
        "date",
        "nav_premium_pct", "nav_premium_zscore", "nav_premium_ma5",
        "nav_premium_momentum", "nav_premium_mean_rev", "nav_premium_extreme",
        "spy_es_basis_pct", "spy_es_basis_zscore",
        "nav_premium_vol", "nav_premium_skew",
        "nav_premium_regime", "nav_creation_pressure",
    ]
    result = df[feature_cols].copy()

    # Clean infinities
    result = result.replace([np.inf, -np.inf], 0)
    result = result.fillna(0)

    logger.info(f"NAV premium features computed: {len(result)} rows, "
                f"latest premium={result['nav_premium_pct'].iloc[-1]:.4f}%")
    return result


def store_nav_premium(router, nav_df: pd.DataFrame):
    """Store NAV premium data to the nav_premium table."""
    if nav_df is None or nav_df.empty:
        return

    cols = [c for c in nav_df.columns if c != "date"]
    col_str = ", ".join(cols)
    ph_str = ", ".join(["?"] * len(cols))

    inserted = 0
    for _, row in nav_df.iterrows():
        vals = [row["date"]]
        for c in cols:
            v = row.get(c)
            vals.append(float(v) if pd.notna(v) else None)
        try:
            router.execute(
                f"INSERT OR REPLACE INTO nav_premium (date, {col_str}) VALUES (?, {ph_str})",
                tuple(vals)
            )
            inserted += 1
        except Exception as e:
            logger.debug(f"NAV premium insert failed for {row['date']}: {e}")

    logger.info(f"Stored {inserted} NAV premium rows")


def backfill_nav_premium(router, years: int = 10):
    """Backfill NAV premium data for historical training.

    Uses yfinance which provides ~10 years of daily data for SPY and ^GSPC.
    """
    days = int(years * 365)
    logger.info(f"Backfilling {years} years of NAV premium data...")
    nav_df = fetch_nav_premium_data(days=days)
    if nav_df is not None:
        store_nav_premium(router, nav_df)
        logger.info(f"NAV premium backfill complete: {len(nav_df)} rows")
    else:
        logger.warning("NAV premium backfill returned no data")
