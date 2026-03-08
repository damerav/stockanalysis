"""ETF Fund Flow Fetcher — captures institutional capital movement.

Tracks daily estimated fund flows for major ETFs (SPY, QQQ, IWM, TLT, HYG,
GLD, XLK, XLF, XLE, EEM) using volume × price delta as a proxy for actual
flow data (which requires expensive Bloomberg/Refinitiv subscriptions).

The proxy formula: flow_proxy = volume * (close - open) * sign(close - prev_close)
This captures the directional conviction of volume — large volume on up days
signals inflows, large volume on down days signals outflows.

Derived features:
  - equity_bond_flow_ratio: SPY+QQQ flows vs TLT+HYG flows (risk appetite)
  - growth_value_flow_ratio: QQQ+XLK flows vs XLF+XLE flows (style rotation)
  - em_dm_flow_ratio: EEM flows vs SPY flows (global risk appetite)
  - flow_momentum_5d: 5-day rolling sum of SPY flow proxy
  - flow_breadth: % of tracked ETFs with positive flow proxy
  - safe_haven_flow: GLD + TLT combined flow proxy (flight to safety)

Data source priority: Polygon.io (primary) → yfinance (fallback).
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRACKED_ETFS = [
    "SPY", "QQQ", "IWM",
    "TLT", "HYG",
    "GLD",
    "XLK", "XLF", "XLE",
    "EEM",
]


def _get_polygon_key() -> str:
    """Resolve Polygon API key from encrypted secrets, config, or env."""
    try:
        from src.data.secrets_manager import get_secret
        key = get_secret("polygon_api_key", fallback="")
        if key:
            return key
    except Exception:
        pass
    try:
        from src.data.init_db import load_config
        cfg = load_config()
        key = (cfg.get("polygon", {}) or {}).get("api_key", "")
        if key and key != "FROM_ENCRYPTED_DB":
            return key
    except Exception:
        pass
    return os.environ.get("POLYGON_API_KEY", "")


def _fetch_bars_polygon(tickers: list[str], from_date: str, to_date: str) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for multiple tickers via Polygon.io.

    Returns dict mapping ticker -> DataFrame with date/open/high/low/close/volume.
    """
    api_key = _get_polygon_key()
    if not api_key:
        return {}

    from src.data.polygon_fetcher import PolygonFetcher
    polygon = PolygonFetcher(api_key)

    result = {}
    for ticker in tickers:
        try:
            df = polygon.get_daily_bars(ticker, from_date, to_date)
            if not df.empty:
                result[ticker] = df
                logger.debug(f"Polygon: {ticker} → {len(df)} bars")
            else:
                logger.debug(f"Polygon: {ticker} → empty")
        except Exception as e:
            logger.debug(f"Polygon: {ticker} failed: {e}")
    return result


def _fetch_bars_yfinance(tickers: list[str], from_date: str, to_date: str) -> dict[str, pd.DataFrame]:
    """Fallback: fetch daily OHLCV via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — cannot fall back")
        return {}

    result = {}
    try:
        data = yf.download(tickers, start=from_date, end=to_date,
                           progress=False, group_by="ticker")
        if data.empty:
            return {}
        for ticker in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    t_data = data[ticker].dropna(how="all")
                else:
                    t_data = data.dropna(how="all")
                if t_data.empty:
                    continue
                t_data = t_data.reset_index()
                t_data["date"] = pd.to_datetime(t_data["Date"]).dt.strftime("%Y-%m-%d")
                t_data = t_data.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })
                result[ticker] = t_data[["date", "open", "high", "low", "close", "volume"]]
            except Exception as e:
                logger.debug(f"yfinance parse {ticker}: {e}")
    except Exception as e:
        logger.warning(f"yfinance download failed: {e}")
    return result


def _fetch_all_bars(tickers: list[str], from_date: str, to_date: str) -> dict[str, pd.DataFrame]:
    """Fetch bars for all tickers: Polygon first, yfinance fallback for missing."""
    bars = _fetch_bars_polygon(tickers, from_date, to_date)
    missing = [t for t in tickers if t not in bars]
    if missing:
        logger.info(f"Polygon returned {len(bars)}/{len(tickers)} tickers, "
                     f"falling back to yfinance for {missing}")
        yf_bars = _fetch_bars_yfinance(missing, from_date, to_date)
        bars.update(yf_bars)
    else:
        logger.info(f"Polygon: all {len(tickers)} tickers fetched")
    return bars


def _compute_flow_proxy(df: pd.DataFrame) -> np.ndarray:
    """Compute flow proxy from OHLCV: volume * intraday_move * direction."""
    close = df["close"].values.astype(float)
    opn = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    direction = np.sign(close - prev_close)
    intraday_move = close - opn
    return (volume * intraday_move * direction) / 1e9  # billions scale


def fetch_etf_flows(days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch ETF price/volume data and compute flow proxies.

    Args:
        days: Number of calendar days to fetch.

    Returns:
        DataFrame with date + flow proxy columns per ETF + derived ratios.
    """
    end = datetime.now()
    start = end - timedelta(days=days + 10)
    from_date = start.strftime("%Y-%m-%d")
    to_date = end.strftime("%Y-%m-%d")

    bars = _fetch_all_bars(TRACKED_ETFS, from_date, to_date)
    if not bars:
        logger.warning("No ETF bar data from any source")
        return None

    # Use SPY dates as the reference index (most liquid, always has data)
    ref_ticker = "SPY" if "SPY" in bars else next(iter(bars))
    ref_dates = bars[ref_ticker]["date"].tolist()

    flows = {}
    for ticker in TRACKED_ETFS:
        if ticker not in bars:
            continue
        df = bars[ticker].copy()
        df = df.set_index("date").reindex(ref_dates).reset_index()
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        # Forward-fill gaps then drop any remaining NaN
        df = df.ffill().bfill()
        flows[f"flow_{ticker.lower()}"] = _compute_flow_proxy(df)

    if not flows:
        return None

    flow_df = pd.DataFrame(flows)
    flow_df["date"] = ref_dates

    # --- Derived features ---
    spy_flow = flow_df.get("flow_spy", pd.Series(0, index=flow_df.index))
    qqq_flow = flow_df.get("flow_qqq", pd.Series(0, index=flow_df.index))
    tlt_flow = flow_df.get("flow_tlt", pd.Series(0, index=flow_df.index))
    hyg_flow = flow_df.get("flow_hyg", pd.Series(0, index=flow_df.index))
    gld_flow = flow_df.get("flow_gld", pd.Series(0, index=flow_df.index))
    xlk_flow = flow_df.get("flow_xlk", pd.Series(0, index=flow_df.index))
    xlf_flow = flow_df.get("flow_xlf", pd.Series(0, index=flow_df.index))
    xle_flow = flow_df.get("flow_xle", pd.Series(0, index=flow_df.index))
    eem_flow = flow_df.get("flow_eem", pd.Series(0, index=flow_df.index))

    equity_flow = spy_flow + qqq_flow
    bond_flow = tlt_flow + hyg_flow
    flow_df["equity_bond_flow_ratio"] = (
        equity_flow / bond_flow.replace(0, np.nan)
    ).fillna(0).clip(-10, 10)

    growth_flow = qqq_flow + xlk_flow
    value_flow = xlf_flow + xle_flow
    flow_df["growth_value_flow_ratio"] = (
        growth_flow / value_flow.replace(0, np.nan)
    ).fillna(0).clip(-10, 10)

    flow_df["em_dm_flow_ratio"] = (
        eem_flow / spy_flow.replace(0, np.nan)
    ).fillna(0).clip(-10, 10)

    flow_df["flow_momentum_5d"] = spy_flow.rolling(5, min_periods=1).sum()

    flow_cols = [c for c in flow_df.columns if c.startswith("flow_") and c != "flow_momentum_5d"]
    if flow_cols:
        flow_df["flow_breadth"] = (flow_df[flow_cols] > 0).sum(axis=1) / len(flow_cols)

    flow_df["safe_haven_flow"] = gld_flow + tlt_flow

    return flow_df


def store_etf_flows(router, flow_df: pd.DataFrame):
    """Store ETF flow data to the etf_flows table."""
    if flow_df is None or flow_df.empty:
        return

    cols = [c for c in flow_df.columns if c != "date"]
    col_str = ", ".join(cols)
    ph_str = ", ".join(["?"] * len(cols))

    inserted = 0
    for _, row in flow_df.iterrows():
        vals = [row["date"]]
        for c in cols:
            v = row.get(c)
            vals.append(float(v) if pd.notna(v) else None)
        try:
            router.execute(
                f"INSERT OR REPLACE INTO etf_flows (date, {col_str}) VALUES (?, {ph_str})",
                tuple(vals)
            )
            inserted += 1
        except Exception as e:
            logger.debug(f"ETF flow insert failed for {row['date']}: {e}")

    logger.info(f"Stored {inserted} ETF flow rows")


def backfill_etf_flows(router, years: int = 5):
    """Backfill ETF flow data for historical training."""
    days = int(years * 365)
    logger.info(f"Backfilling {years} years of ETF flow data (Polygon primary)...")
    flow_df = fetch_etf_flows(days=days)
    if flow_df is not None:
        store_etf_flows(router, flow_df)
        logger.info(f"ETF flow backfill complete: {len(flow_df)} rows")
    else:
        logger.warning("ETF flow backfill returned no data")
