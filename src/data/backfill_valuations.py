"""Backfill historical CAPE ratio and Buffett Indicator into market_breadth table.

CAPE: Monthly Shiller PE10 from datahub.io, forward-filled to daily trading dates.
Buffett: Wilshire 5000 (yfinance ^W5000) / GDP (FRED quarterly), forward-filled.

Usage:
    python -m src.data.backfill_valuations [--years 5]
"""

import argparse
import logging
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd

from src.data.init_db import load_config
from src.data.db_router import get_router

logger = logging.getLogger(__name__)


def backfill_valuations(years: int = 5, config: dict = None):
    """Backfill historical CAPE and Buffett Indicator."""
    if config is None:
        config = load_config()

    router = get_router(config)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(years * 365))
    start_str = start_date.strftime("%Y-%m-%d")

    # Get trading dates from prices table
    dates_df = router.query(
        "SELECT date FROM prices WHERE date >= ? ORDER BY date", (start_str,)
    )
    if dates_df.empty:
        logger.error("No price data found. Run backfill_historical first.")
        router.close()
        return
    trading_dates = dates_df["date"].astype(str).tolist()
    logger.info(f"Found {len(trading_dates)} trading dates to backfill valuations for.")

    # --- 1. Shiller CAPE Ratio ---
    cape_map = {}
    logger.info("Fetching historical Shiller CAPE ratio from datahub.io...")
    try:
        import requests
        resp = requests.get("https://datahub.io/core/s-and-p-500/r/data.csv", timeout=30)
        resp.raise_for_status()
        cape_df = pd.read_csv(StringIO(resp.text))
        cape_col = next((c for c in cape_df.columns if "PE10" in c.upper() or "CAPE" in c.upper()), None)
        if cape_col:
            cape_df["date"] = pd.to_datetime(cape_df["Date"])
            cape_df[cape_col] = pd.to_numeric(cape_df[cape_col], errors="coerce")
            cape_df = cape_df[cape_df[cape_col] > 0].sort_values("date")
            # Monthly data — create a date→value series, then forward-fill to daily
            cape_series = cape_df.set_index("date")[cape_col]
            # Reindex to daily and forward-fill
            daily_idx = pd.date_range(start=cape_series.index.min(), end=end_date, freq="D")
            cape_daily = cape_series.reindex(daily_idx).ffill()
            cape_map = {d.strftime("%Y-%m-%d"): round(float(v), 2)
                        for d, v in cape_daily.items() if pd.notna(v)}
            logger.info(f"  CAPE: {len(cape_df)} monthly points → {len(cape_map)} daily values")
        else:
            logger.warning("  Could not find CAPE/PE10 column in Shiller dataset")
    except Exception as e:
        logger.error(f"  CAPE backfill failed: {e}")

    # --- 2. Buffett Indicator ---
    buffett_map = {}
    logger.info("Fetching Wilshire 5000 + GDP for Buffett Indicator...")
    try:
        import yfinance as yf
        import requests

        # Wilshire 5000 daily from yfinance
        w5k = yf.download("^W5000", start=start_str, end=end_date.strftime("%Y-%m-%d"), progress=False)
        if isinstance(w5k.columns, pd.MultiIndex):
            w5k.columns = w5k.columns.get_level_values(0)
        if not w5k.empty:
            w5k = w5k.reset_index()
            w5k["date"] = pd.to_datetime(w5k["Date"]).dt.strftime("%Y-%m-%d")
            w5k_series = w5k.set_index("date")["Close"]
            logger.info(f"  Wilshire 5000: {len(w5k_series)} daily values")

            # GDP quarterly from FRED
            fred_key = ""
            try:
                from src.data.secrets_manager import get_secret
                fred_key = get_secret("fred_api_key", fallback="")
            except Exception:
                fred_key = config.get("fred", {}).get("api_key", "")
                if fred_key == "FROM_ENCRYPTED_DB":
                    fred_key = ""

            gdp_series = pd.Series(dtype=float)
            if fred_key:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": "GDP", "api_key": fred_key,
                    "file_type": "json", "sort_order": "asc",
                    "observation_start": start_str,
                }
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    obs = resp.json().get("observations", [])
                    rows = {o["date"]: float(o["value"])
                            for o in obs if o.get("value", ".") != "."}
                    gdp_series = pd.Series(rows)
                    logger.info(f"  GDP: {len(gdp_series)} quarterly values")

            if not gdp_series.empty:
                # Forward-fill GDP to daily
                gdp_series.index = pd.to_datetime(gdp_series.index)
                daily_idx = pd.date_range(start=gdp_series.index.min(), end=end_date, freq="D")
                gdp_daily = gdp_series.reindex(daily_idx).ffill()

                for date_str in trading_dates:
                    w5k_val = w5k_series.get(date_str)
                    dt = pd.to_datetime(date_str)
                    gdp_val = gdp_daily.get(dt) if dt in gdp_daily.index else None
                    if w5k_val and gdp_val and float(gdp_val) > 0:
                        buffett_map[date_str] = round((float(w5k_val) / float(gdp_val)) * 100, 2)
                logger.info(f"  Buffett Indicator: computed for {len(buffett_map)} dates")
            else:
                logger.warning("  No GDP data from FRED — Buffett Indicator skipped")
        else:
            logger.warning("  Wilshire 5000 returned no data")
    except Exception as e:
        logger.error(f"  Buffett Indicator backfill failed: {e}")

    # --- 3. Upsert into market_breadth table ---
    logger.info("Writing valuations to market_breadth table...")
    inserted = 0
    updated = 0
    for date_str in trading_dates:
        cape_val = cape_map.get(date_str)
        buffett_val = buffett_map.get(date_str)
        if cape_val is None and buffett_val is None:
            continue

        # Check if row exists
        existing = router.query(
            "SELECT date FROM market_breadth WHERE date = ?", (date_str,)
        )
        if existing.empty:
            # Insert new row with just the valuation columns
            router.execute(
                "INSERT INTO market_breadth (date, sp500_cape, buffett_indicator) VALUES (?, ?, ?)",
                (date_str, cape_val, buffett_val)
            )
            inserted += 1
        else:
            # Update only non-null values
            sets, vals = [], []
            if cape_val is not None:
                sets.append("sp500_cape = ?")
                vals.append(cape_val)
            if buffett_val is not None:
                sets.append("buffett_indicator = ?")
                vals.append(buffett_val)
            if sets:
                vals.append(date_str)
                router.execute(
                    f"UPDATE market_breadth SET {', '.join(sets)} WHERE date = ?",
                    tuple(vals)
                )
                updated += 1

    logger.info(f"Valuation backfill complete: {inserted} inserted, {updated} updated")
    router.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backfill CAPE & Buffett Indicator")
    parser.add_argument("--years", type=int, default=5, help="Years of history (default: 5)")
    args = parser.parse_args()
    config = load_config()
    backfill_valuations(years=args.years, config=config)
