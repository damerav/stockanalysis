"""1D. Gap Detection & Backfill — Find missing dates and auto-fill."""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from src.data.init_db import load_config
from src.data.polygon_fetcher import PolygonFetcher
from src.data.fetcher import FallbackFetcher
from src.data.db_router import get_router

logger = logging.getLogger(__name__)


def get_trading_days(start: str, end: str) -> list[str]:
    """Generate list of expected trading days using NYSE calendar."""
    from src.data.calendar import get_nyse_trading_days
    return get_nyse_trading_days(start, end)


def find_gaps(conn, table: str = "prices",
              date_col: str = "date", config: dict = None) -> list[str]:
    """Find missing weekdays in a table compared to expected trading days."""
    router = get_router(config)
    df = router.query(f"SELECT MIN({date_col}) as mn, MAX({date_col}) as mx FROM {table}")
    if df.empty or df.iloc[0]["mn"] is None:
        return []
    min_date, max_date = df.iloc[0]["mn"], df.iloc[0]["mx"]
    expected = set(get_trading_days(min_date, datetime.now().strftime("%Y-%m-%d")))
    existing_df = router.query(f"SELECT DISTINCT {date_col} as d FROM {table}")
    existing = set(existing_df["d"].tolist()) if not existing_df.empty else set()
    gaps = sorted(expected - existing)
    if gaps:
        logger.info(f"Found {len(gaps)} gap days in {table}: {gaps[:5]}...")
    return gaps


def backfill_prices(conn, gaps: list[str],
                    polygon: Optional[PolygonFetcher], fallback: FallbackFetcher,
                    config: dict = None):
    """Backfill missing price data via DbRouter."""
    if not gaps:
        return

    logger.info(f"Backfilling {len(gaps)} days of price data")

    if polygon:
        try:
            df = polygon.get_daily_bars("SPY", gaps[0], gaps[-1])
            if not df.empty:
                _insert_prices(df, config)
                filled = set(df["date"].tolist())
                gaps = [g for g in gaps if g not in filled]
                logger.info(f"Polygon filled {len(filled)} days, {len(gaps)} remaining")
        except Exception as e:
            logger.warning(f"Polygon backfill failed: {e}")

    if gaps:
        try:
            df = fallback.get_daily_bars_yf("SPY", days=len(gaps) + 30)
            if not df.empty:
                df = df[df["date"].isin(gaps)]
                _insert_prices(df, config)
                logger.info(f"yfinance filled {len(df)} remaining days")
        except Exception as e:
            logger.warning(f"yfinance backfill failed: {e}")


def backfill_macro(conn, gaps: list[str],
                   fallback: FallbackFetcher, config: dict = None):
    """Backfill macro data for gap dates via DbRouter."""
    if not gaps:
        return
    logger.info(f"Backfilling macro data for {len(gaps)} days")
    macro = fallback.get_macro_fred()
    router = get_router(config)

    for date in gaps:
        if date >= (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"):
            router.execute(
                """INSERT OR REPLACE INTO macro (date, vix, vix_change, us10y_yield,
                   dxy, fed_funds, gold, crude) VALUES (?,?,?,?,?,?,?,?)""",
                (date, macro.get("vix"), macro.get("vix_change"),
                 macro.get("us10y_yield"), macro.get("dxy"),
                 macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
            )


def _insert_prices(df: pd.DataFrame, config: dict = None):
    """Insert price rows via DbRouter."""
    router = get_router(config)
    for _, row in df.iterrows():
        router.execute(
            """INSERT OR REPLACE INTO prices (date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row["date"], row["open"], row["high"], row["low"],
             row["close"], row["volume"])
        )


def validate_completeness(conn=None, config: dict = None) -> dict:
    """Check row counts across all tables via DbRouter."""
    counts = {}
    router = get_router(config)
    all_tables = ["prices", "technicals", "macro", "intraday_bars", "options_chain",
                  "news", "daily_sentiment", "predictions",
                  "options_analytics", "intraday_features", "performance"]
    for table in all_tables:
        try:
            df = router.query(f"SELECT COUNT(*) as cnt FROM {table}")
            counts[table] = int(df.iloc[0]["cnt"]) if not df.empty else 0
        except Exception:
            counts[table] = 0
    return counts


def run_daily_pull(config: dict = None):
    """Main entry point: detect gaps and backfill all data."""
    if config is None:
        config = load_config()

    router = get_router(config)
    api_key = config.get("polygon", {}).get("api_key", "")
    if not api_key or api_key in ("YOUR_POLYGON_KEY", "FROM_ENCRYPTED_DB"):
        try:
            from src.data.secrets_manager import get_secret
            api_key = get_secret("polygon_api_key", fallback="")
        except Exception:
            pass
    polygon = PolygonFetcher(api_key) if api_key else None
    fallback = FallbackFetcher(config=config)
    # Find and fill price gaps
    price_gaps = find_gaps(None, "prices", config=config)
    backfill_prices(None, price_gaps, polygon, fallback, config=config)

    # Find and fill macro gaps
    macro_gaps = find_gaps(None, "macro", config=config)
    backfill_macro(None, macro_gaps, fallback, config=config)

    # Validate
    counts = validate_completeness(config=config)
    logger.info(f"Table row counts: {counts}")

    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_daily_pull()
