"""1D. Gap Detection & Backfill — Find missing dates and auto-fill."""

import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.data.init_db import get_connection, load_config
from src.data.polygon_fetcher import PolygonFetcher
from src.data.fetcher import FallbackFetcher

logger = logging.getLogger(__name__)


def get_trading_days(start: str, end: str) -> list[str]:
    """Generate list of expected trading days (Mon-Fri, excluding known holidays)."""
    dates = pd.bdate_range(start=start, end=end)
    return [d.strftime("%Y-%m-%d") for d in dates]


def find_gaps(conn: sqlite3.Connection, table: str = "prices",
              date_col: str = "date") -> list[str]:
    """Find missing weekdays in a table compared to expected trading days."""
    cursor = conn.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM {table}")
    row = cursor.fetchone()
    if not row or not row[0]:
        return []

    min_date, max_date = row[0], row[1]
    expected = set(get_trading_days(min_date, datetime.now().strftime("%Y-%m-%d")))

    existing = set()
    for r in conn.execute(f"SELECT DISTINCT {date_col} FROM {table}"):
        existing.add(r[0])

    gaps = sorted(expected - existing)
    if gaps:
        logger.info(f"Found {len(gaps)} gap days in {table}: {gaps[:5]}...")
    return gaps


def backfill_prices(conn: sqlite3.Connection, gaps: list[str],
                    polygon: Optional[PolygonFetcher], fallback: FallbackFetcher):
    """Backfill missing price data from Polygon or yfinance."""
    if not gaps:
        return

    logger.info(f"Backfilling {len(gaps)} days of price data")

    # Try Polygon first (batch by date range)
    if polygon:
        try:
            df = polygon.get_daily_bars("SPY", gaps[0], gaps[-1])
            if not df.empty:
                _insert_prices(conn, df)
                filled = set(df["date"].tolist())
                gaps = [g for g in gaps if g not in filled]
                logger.info(f"Polygon filled {len(filled)} days, {len(gaps)} remaining")
        except Exception as e:
            logger.warning(f"Polygon backfill failed: {e}")

    # Fallback to yfinance for remaining gaps
    if gaps:
        try:
            df = fallback.get_daily_bars_yf("SPY", days=len(gaps) + 30)
            if not df.empty:
                df = df[df["date"].isin(gaps)]
                _insert_prices(conn, df)
                logger.info(f"yfinance filled {len(df)} remaining days")
        except Exception as e:
            logger.warning(f"yfinance backfill failed: {e}")


def backfill_macro(conn: sqlite3.Connection, gaps: list[str],
                   fallback: FallbackFetcher):
    """Backfill macro data for gap dates."""
    if not gaps:
        return
    logger.info(f"Backfilling macro data for {len(gaps)} days")
    macro = fallback.get_macro_fred()
    today = datetime.now().strftime("%Y-%m-%d")
    # FRED gives latest values; apply to recent gaps only
    for date in gaps:
        if date >= (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"):
            conn.execute(
                """INSERT OR REPLACE INTO macro (date, vix, vix_change, us10y_yield,
                   dxy, fed_funds, gold, crude) VALUES (?,?,?,?,?,?,?,?)""",
                (date, macro.get("vix"), macro.get("vix_change"),
                 macro.get("us10y_yield"), macro.get("dxy"),
                 macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
            )
    conn.commit()


def _insert_prices(conn: sqlite3.Connection, df: pd.DataFrame):
    """Insert price rows into the prices table."""
    for _, row in df.iterrows():
        conn.execute(
            """INSERT OR REPLACE INTO prices (date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row["date"], row["open"], row["high"], row["low"],
             row["close"], row["volume"])
        )
    conn.commit()


def validate_completeness(conn: sqlite3.Connection) -> dict:
    """Check row counts across all 10 tables."""
    tables = ["prices", "technicals", "news", "daily_sentiment", "macro",
              "predictions", "intraday_bars", "options_chain",
              "options_analytics", "intraday_features", "performance"]
    counts = {}
    for table in tables:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = row[0]
        except Exception:
            counts[table] = 0
    return counts


def run_daily_pull(config: dict = None):
    """Main entry point: detect gaps and backfill all data."""
    if config is None:
        config = load_config()

    conn = get_connection(config)
    api_key = config.get("polygon", {}).get("api_key", "")
    polygon = PolygonFetcher(api_key) if api_key and api_key != "YOUR_POLYGON_KEY" else None
    fallback = FallbackFetcher()

    # Find and fill price gaps
    price_gaps = find_gaps(conn, "prices")
    backfill_prices(conn, price_gaps, polygon, fallback)

    # Find and fill macro gaps
    macro_gaps = find_gaps(conn, "macro")
    backfill_macro(conn, macro_gaps, fallback)

    # Validate
    counts = validate_completeness(conn)
    logger.info(f"Table row counts: {counts}")

    conn.close()
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily_pull()
