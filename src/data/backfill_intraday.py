"""Backfill 5-second intraday bars from Polygon.io.

Replaces the legacy yfinance 5-minute backfill with high-fidelity
5-second bars from Polygon REST API. Uses DbRouter for PostgreSQL
primary with SQLite fallback.

Usage:
    python -m src.data.backfill_intraday
    python -m src.data.backfill_intraday --days 30
"""

import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd

from src.data.init_db import load_config
from src.data.db_router import DbRouter
from src.data.polygon_fetcher import PolygonFetcher

logger = logging.getLogger(__name__)


def backfill_intraday(config: dict = None, days_to_backfill: int = 60):
    """Download N days of 5-second SPY bars from Polygon and store them."""
    if config is None:
        config = load_config()

    router = DbRouter(config)

    # Resolve Polygon API key
    api_key = config.get("polygon", {}).get("api_key", "")
    if not api_key or api_key in ("YOUR_POLYGON_KEY", "FROM_ENCRYPTED_DB"):
        try:
            from src.data.secrets_manager import get_secret
            api_key = get_secret("polygon_api_key", fallback="")
        except Exception:
            api_key = ""

    if not api_key:
        logger.error("Polygon API key not found — cannot backfill intraday data")
        return

    polygon = PolygonFetcher(api_key)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_to_backfill)

    logger.info(f"Fetching {days_to_backfill} days of 5-second SPY bars from Polygon...")
    all_bars = []
    for day in pd.date_range(start_date, end_date):
        date_str = day.strftime("%Y-%m-%d")
        try:
            df = polygon.get_5s_bars("SPY", date_str)
            if not df.empty:
                all_bars.append(df)
                logger.info(f"  {date_str}: {len(df)} bars")
        except Exception as e:
            logger.warning(f"  {date_str}: failed — {e}")

    if not all_bars:
        logger.info("No new bars fetched")
        return

    full_df = pd.concat(all_bars, ignore_index=True)
    logger.info(f"Total: {len(full_df)} 5s bars across {len(all_bars)} days")

    inserted = 0
    for _, row in full_df.iterrows():
        try:
            router.execute(
                "INSERT INTO intraday_bars "
                "(timestamp, ticker, open, high, low, close, volume, vwap) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (timestamp, ticker) DO NOTHING",
                (row["timestamp"], "SPY",
                 float(row["open"]), float(row["high"]),
                 float(row["low"]), float(row["close"]),
                 int(row["volume"]),
                 float(row["vwap"]) if pd.notna(row.get("vwap")) else None)
            )
            inserted += 1
        except Exception as e:
            logger.debug(f"Insert failed for {row['timestamp']}: {e}")

    logger.info(f"Inserted {inserted} intraday bars")

    # Verify
    try:
        count_df = router.read_analytics("SELECT COUNT(*) as cnt FROM intraday_bars")
        total = int(count_df.iloc[0]["cnt"]) if not count_df.empty else 0
        logger.info(f"Total intraday_bars in DB: {total}")
    except Exception:
        pass

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Backfill 5-second intraday bars from Polygon")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--days", type=int, default=60, help="Number of past days to backfill")
    args = parser.parse_args()

    config = load_config(args.config)
    backfill_intraday(config=config, days_to_backfill=args.days)
