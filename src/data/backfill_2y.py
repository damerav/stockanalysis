"""Backfill 2+ years of historical data from yfinance + FRED.

Loads SPY prices, macro indicators (VIX, yields, DXY, fed funds, gold, crude),
and recomputes technicals for all dates. This gives the model enough data
to train properly (~500+ trading days).

Usage:
    python -m src.data.backfill_2y [--years 3]
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.init_db import init_db, get_connection, load_config
from src.data.db_router import get_router
from src.data.features import compute_all_technicals, store_technicals

logger = logging.getLogger(__name__)


def _fetch_fred_series(series_id: str, fred_key: str,
                       start: str, end: str) -> pd.DataFrame:
    """Fetch a full FRED time series as DataFrame with date + value columns."""
    import requests
    if fred_key:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": fred_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "asc",
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            obs = resp.json().get("observations", [])
            rows = []
            for o in obs:
                val = o.get("value", ".")
                if val != ".":
                    rows.append({"date": o["date"], "value": float(val)})
            return pd.DataFrame(rows)
    # CSV fallback
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    if df.shape[1] >= 2:
        df.columns = ["date", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df
    return pd.DataFrame()


def backfill_historical(years: int = 3, config: dict = None):
    """Load years of historical SPY + macro data and recompute technicals."""
    if config is None:
        config = load_config()

    db_path = init_db(config)
    conn = get_connection(config)

    fred_key = config.get("fred", {}).get("api_key", "")
    # Resolve from encrypted DB if placeholder
    try:
        from src.data.secrets_manager import get_secret
        if not fred_key or fred_key == "FROM_ENCRYPTED_DB":
            fred_key = get_secret("fred_api_key", fallback=fred_key or "")
    except Exception:
        pass

    try:
        router = get_router(config)
    except Exception:
        logger.warning("DbRouter unavailable, using SQLite only")
        router = None

    def _db_exec(sql, params=None):
        """Execute via router (handles INSERT OR REPLACE → ON CONFLICT)."""
        if router:
            router.execute(sql, params)
        else:
            conn.execute(sql, params)
            conn.commit()

    def _db_query(sql, params=None):
        """Query via router."""
        if router:
            return router.query(sql, params)
        return pd.read_sql_query(sql, conn, params=params)

    days = int(years * 365)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # ── 1. SPY Prices from yfinance ──
    logger.info(f"Fetching {years} years of SPY data ({start_date} to {end_date})...")
    try:
        import yfinance as yf
        data = yf.download("SPY", start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        df = data.reset_index()
        df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        prices = df[["date", "open", "high", "low", "close", "volume"]].copy()
        logger.info(f"Got {len(prices)} trading days of SPY prices")
    except Exception as e:
        logger.error(f"yfinance failed: {e}")
        conn.close()
        return

    # Insert prices
    for _, row in prices.iterrows():
        sql = """INSERT OR REPLACE INTO prices (date, open, high, low, close, volume)
                 VALUES (?, ?, ?, ?, ?, ?)"""
        params = (row["date"], float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]), int(row["volume"]))
        _db_exec(sql, params)
    logger.info(f"Inserted {len(prices)} price rows")

    # ── 2. Macro data from FRED ──
    fred_series = {
        # Core Market & Rates
        "vix": "VIXCLS",
        "us10y_yield": "DGS10",
        "us3m_yield": "DTB3",
        "yield_curve_10y3m": "T10Y3M",
        "sahm_rule": "SAHMREALTIME",
        "consumer_conf": "UMCSENT",
        "ism_pmi": "INDPRO",
        "dxy": "DTWEXBGS",
        "fed_funds": "FEDFUNDS",
        "crude": "DCOILWTICO",
        # Inflation
        "cpi": "CPIAUCSL",
        "core_cpi": "CPILFESL",
        "pce": "PCEPI",
        "core_pce": "PCEPILFE",
        "ppi": "PPIACO",
        # Growth & Employment
        "gdp": "GDP",
        "nfp": "PAYEMS",
        "unemployment_rate": "UNRATE",
        "initial_claims": "ICSA",
        "continuing_claims": "CCSA",
        # Activity
        "retail_sales": "RSXFS",
        "industrial_production": "INDPRO",
        # Housing
        "housing_starts": "HOUST",
        "building_permits": "PERMIT",
        "case_shiller_hpi": "CSUSHPINSA",
    }

    macro_frames = {}
    for name, series_id in fred_series.items():
        logger.info(f"Fetching FRED {name} ({series_id})...")
        try:
            df = _fetch_fred_series(series_id, fred_key, start_date, end_date)
            if not df.empty:
                df = df.rename(columns={"value": name})
                macro_frames[name] = df.set_index("date")[name]
                logger.info(f"  {name}: {len(df)} observations")
            else:
                logger.warning(f"  {name}: no data returned")
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    # Gold from yfinance
    logger.info("Fetching gold prices from yfinance...")
    try:
        import yfinance as yf
        gold = yf.download("GC=F", start=start_date, end=end_date, progress=False)
        if isinstance(gold.columns, pd.MultiIndex):
            gold.columns = gold.columns.get_level_values(0)
        gold = gold.reset_index()
        gold["date"] = pd.to_datetime(gold["Date"]).dt.strftime("%Y-%m-%d")
        gold = gold.rename(columns={"Close": "gold"})
        macro_frames["gold"] = gold.set_index("date")["gold"]
        logger.info(f"  gold: {len(gold)} observations")
    except Exception as e:
        logger.warning(f"  gold failed: {e}")

    # Merge all macro into one DataFrame aligned to trading dates
    trading_dates = prices["date"].tolist()
    macro_df = pd.DataFrame({"date": trading_dates})
    macro_df = macro_df.set_index("date")

    for name, series in macro_frames.items():
        series.index = pd.to_datetime(series.index).strftime("%Y-%m-%d")
        # Reindex to trading dates with forward-fill so that weekend/non-trading
        # day observations (e.g. ICSA on Saturdays, GDP quarterly) propagate
        # to the next available trading day.
        all_dates = sorted(set(series.index.tolist() + trading_dates))
        reindexed = series.reindex(all_dates).ffill()
        reindexed = reindexed.reindex(trading_dates)
        macro_df[name] = reindexed.values

    # Forward-fill macro data (FRED has gaps on weekends/holidays)
    macro_df = macro_df.ffill()
    macro_df = macro_df.reset_index()

    # Compute vix_change
    if "vix" in macro_df.columns:
        macro_df["vix_change"] = macro_df["vix"].diff()
    else:
        macro_df["vix_change"] = None

    # Insert macro rows
    inserted = 0
    # Build column list dynamically from all available macro columns
    _macro_cols = [
        "vix", "vix_change", "us10y_yield", "us3m_yield", "yield_curve_10y3m",
        "dxy", "fed_funds", "gold", "crude",
        "sahm_rule", "consumer_conf", "ism_pmi",
        "cpi", "core_cpi", "pce", "core_pce", "ppi",
        "gdp", "nfp", "unemployment_rate", "initial_claims", "continuing_claims",
        "retail_sales", "industrial_production",
        "housing_starts", "building_permits", "case_shiller_hpi",
    ]
    _col_str = ", ".join(_macro_cols)
    _ph_str = ", ".join(["?"] * len(_macro_cols))
    for _, row in macro_df.iterrows():
        vals = [row["date"]]
        for c in _macro_cols:
            v = row.get(c)
            vals.append(float(v) if pd.notna(v) else None)
        _db_exec(
            f"INSERT OR REPLACE INTO macro (date, {_col_str}) VALUES (?, {_ph_str})",
            tuple(vals)
        )
        inserted += 1
    logger.info(f"Inserted {inserted} macro rows")

    # ── 3. Recompute technicals ──
    logger.info("Recomputing technicals for all dates...")
    price_df = _db_query(
        "SELECT date, open, high, low, close, volume FROM prices ORDER BY date"
    )
    if len(price_df) > 0:
        tech_df = compute_all_technicals(price_df, config=config)
        store_technicals(conn, tech_df, config=config)
        logger.info(f"Computed and stored technicals for {len(tech_df)} rows")

    # ── Summary ──
    cnt_prices = _db_query("SELECT COUNT(*) as cnt FROM prices")
    cnt_macro = _db_query("SELECT COUNT(*) as cnt FROM macro")
    cnt_tech = _db_query("SELECT COUNT(*) as cnt FROM technicals")
    logger.info(f"\nBackfill complete:")
    logger.info(f"  Prices: {cnt_prices.iloc[0]['cnt']} rows")
    logger.info(f"  Macro:  {cnt_macro.iloc[0]['cnt']} rows")
    logger.info(f"  Technicals: {cnt_tech.iloc[0]['cnt']} rows")

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backfill 2+ years of historical data")
    parser.add_argument("--years", type=int, default=5,
                        help="Years of history to load (default: 5)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    backfill_historical(years=args.years, config=config)
