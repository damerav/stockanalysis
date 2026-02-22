"""1D. Gap Detection & Backfill — Find missing dates and auto-fill."""

import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.data.init_db import get_connection, load_config
from src.data.polygon_fetcher import PolygonFetcher
from src.data.fetcher import FallbackFetcher
from src.data.db_router import get_router

logger = logging.getLogger(__name__)


def get_trading_days(start: str, end: str) -> list[str]:
    """Generate list of expected trading days (Mon-Fri, excluding known holidays)."""
    dates = pd.bdate_range(start=start, end=end)
    return [d.strftime("%Y-%m-%d") for d in dates]


def find_gaps(conn: sqlite3.Connection, table: str = "prices",
              date_col: str = "date", config: dict = None) -> list[str]:
    """Find missing weekdays in a table compared to expected trading days.
    Enhancement 26: Checks DuckDB for analytics tables."""
    try:
        router = get_router(config)
        if table in {"prices", "technicals", "macro", "intraday_bars", "options_chain"}:
            df = router.read_analytics(f"SELECT MIN({date_col}) as mn, MAX({date_col}) as mx FROM {table}")
            if df.empty or df.iloc[0]["mn"] is None:
                return []
            min_date, max_date = df.iloc[0]["mn"], df.iloc[0]["mx"]
            expected = set(get_trading_days(min_date, datetime.now().strftime("%Y-%m-%d")))
            existing_df = router.read_analytics(f"SELECT DISTINCT {date_col} as d FROM {table}")
            existing = set(existing_df["d"].tolist()) if not existing_df.empty else set()
            gaps = sorted(expected - existing)
            if gaps:
                logger.info(f"Found {len(gaps)} gap days in {table} (DuckDB): {gaps[:5]}...")
            return gaps
    except Exception:
        pass

    # Fallback to SQLite
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
                    polygon: Optional[PolygonFetcher], fallback: FallbackFetcher,
                    config: dict = None):
    """Backfill missing price data. Enhancement 26: Writes to DuckDB."""
    if not gaps:
        return

    logger.info(f"Backfilling {len(gaps)} days of price data")

    # Try Polygon first (batch by date range)
    if polygon:
        try:
            df = polygon.get_daily_bars("SPY", gaps[0], gaps[-1])
            if not df.empty:
                _insert_prices(conn, df, config)
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
                _insert_prices(conn, df, config)
                logger.info(f"yfinance filled {len(df)} remaining days")
        except Exception as e:
            logger.warning(f"yfinance backfill failed: {e}")


def backfill_macro(conn: sqlite3.Connection, gaps: list[str],
                   fallback: FallbackFetcher, config: dict = None):
    """Backfill macro data for gap dates. Enhancement 26: Writes to DuckDB."""
    if not gaps:
        return
    logger.info(f"Backfilling macro data for {len(gaps)} days")
    macro = fallback.get_macro_fred()

    try:
        router = get_router(config)
        use_duck = True
    except Exception:
        use_duck = False

    for date in gaps:
        if date >= (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"):
            if use_duck:
                router.write_analytics(
                    """INSERT OR REPLACE INTO macro (date, vix, vix_change, us10y_yield,
                       dxy, fed_funds, gold, crude) VALUES (?,?,?,?,?,?,?,?)""",
                    (date, macro.get("vix"), macro.get("vix_change"),
                     macro.get("us10y_yield"), macro.get("dxy"),
                     macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
                )
            else:
                conn.execute(
                    """INSERT OR REPLACE INTO macro (date, vix, vix_change, us10y_yield,
                       dxy, fed_funds, gold, crude) VALUES (?,?,?,?,?,?,?,?)""",
                    (date, macro.get("vix"), macro.get("vix_change"),
                     macro.get("us10y_yield"), macro.get("dxy"),
                     macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
                )
    if not use_duck:
        conn.commit()


def _insert_prices(conn: sqlite3.Connection, df: pd.DataFrame, config: dict = None):
    """Insert price rows. Enhancement 26: Writes to DuckDB if available."""
    try:
        router = get_router(config)
        duck = router.get_analytics_conn()
        for _, row in df.iterrows():
            duck.execute(
                """INSERT OR REPLACE INTO prices (date, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (row["date"], row["open"], row["high"], row["low"],
                 row["close"], row["volume"])
            )
        return
    except Exception as e:
        logger.warning(f"DuckDB price insert failed, using SQLite: {e}")

    for _, row in df.iterrows():
        conn.execute(
            """INSERT OR REPLACE INTO prices (date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row["date"], row["open"], row["high"], row["low"],
             row["close"], row["volume"])
        )
    conn.commit()


def validate_completeness(conn: sqlite3.Connection, config: dict = None) -> dict:
    """Check row counts across all tables (DuckDB + SQLite)."""
    counts = {}
    # DuckDB analytics tables
    try:
        router = get_router(config)
        for table in ["prices", "technicals", "macro", "intraday_bars", "options_chain"]:
            df = router.read_analytics(f"SELECT COUNT(*) as cnt FROM {table}")
            counts[table] = int(df.iloc[0]["cnt"]) if not df.empty else 0
    except Exception:
        for table in ["prices", "technicals", "macro", "intraday_bars", "options_chain"]:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                counts[table] = row[0]
            except Exception:
                counts[table] = 0

    # SQLite operational tables
    sqlite_tables = ["news", "daily_sentiment", "predictions",
                     "options_analytics", "intraday_features", "performance"]
    for table in sqlite_tables:
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
    fallback = FallbackFetcher(config=config)
    # Find and fill price gaps
    price_gaps = find_gaps(conn, "prices", config=config)
    backfill_prices(conn, price_gaps, polygon, fallback, config=config)

    # Find and fill macro gaps
    macro_gaps = find_gaps(conn, "macro", config=config)
    backfill_macro(conn, macro_gaps, fallback, config=config)

    # Validate
    counts = validate_completeness(conn, config=config)
    logger.info(f"Table row counts: {counts}")

    conn.close()
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily_pull()
