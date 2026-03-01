"""Backfill 5-minute intraday bars from yfinance (last 60 days).

yfinance provides free 5-min data for the last 60 trading days.
This populates the intraday_bars table so compute_intraday_microstructure
can generate the 8 microstructure features.

Note: The microstructure function was written for 5-second bars.
We adapt by also updating the function's bar-count assumptions,
OR we resample to match. Since 5-min is the finest free data,
we store 5-min bars and adjust the microstructure computation.

Usage:
    python -m src.data.backfill_intraday
"""

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.data.init_db import init_db, get_connection, load_config

logger = logging.getLogger(__name__)


def backfill_intraday(config: dict = None):
    """Download 60 days of 5-min SPY bars from yfinance and store them."""
    if config is None:
        config = load_config()

    conn = get_connection(config)

    logger.info("Fetching 60 days of 5-min SPY bars from yfinance...")
    try:
        import yfinance as yf
        # yfinance max for 5m interval is 60 days
        data = yf.download("SPY", period="60d", interval="5m", progress=False)
        if data.empty:
            logger.error("yfinance returned no intraday data")
            conn.close()
            return

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = data.reset_index()
        # Column is "Datetime" for intraday
        dt_col = "Datetime" if "Datetime" in df.columns else "Date"
        df["timestamp"] = pd.to_datetime(df[dt_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        df["date"] = pd.to_datetime(df[dt_col]).dt.strftime("%Y-%m-%d")
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})

        # Compute VWAP per day
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["tp_vol"] = df["typical_price"] * df["volume"]
        df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
        df["cum_vol"] = df.groupby("date")["volume"].cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)

        logger.info(f"Got {len(df)} 5-min bars across {df['date'].nunique()} days")

        # Insert into intraday_bars
        inserted = 0
        for _, row in df.iterrows():
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO intraday_bars
                       (timestamp, ticker, open, high, low, close, volume, vwap)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (row["timestamp"], "SPY",
                     float(row["open"]), float(row["high"]),
                     float(row["low"]), float(row["close"]),
                     int(row["volume"]), float(row["vwap"]) if pd.notna(row["vwap"]) else None)
                )
                inserted += 1
            except Exception as e:
                logger.debug(f"Insert failed for {row['timestamp']}: {e}")

        conn.commit()
        logger.info(f"Inserted {inserted} intraday bars")

        # Verify
        count = conn.execute("SELECT COUNT(*) FROM intraday_bars").fetchone()[0]
        dates = conn.execute(
            "SELECT COUNT(DISTINCT substr(timestamp, 1, 10)) FROM intraday_bars"
        ).fetchone()[0]
        logger.info(f"Total intraday_bars: {count} rows, {dates} unique days")

    except Exception as e:
        logger.error(f"Intraday backfill failed: {e}")
        import traceback
        traceback.print_exc()

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backfill 5-min intraday bars")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    backfill_intraday(config=config)
