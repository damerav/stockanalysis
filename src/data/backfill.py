"""1E. Initial Bulk Load — First-time 252-day historical backfill."""

import argparse
import logging
from datetime import datetime, timedelta

from src.data.init_db import init_db, get_connection, load_config
from src.data.polygon_fetcher import PolygonFetcher
from src.data.fetcher import FallbackFetcher
from src.data.db_router import get_router

logger = logging.getLogger(__name__)


def _insert_prices_duck(router, df):
    """Insert price rows into the database via DbRouter."""
    for _, row in df.iterrows():
        router.execute(
            """INSERT OR REPLACE INTO prices (date, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row["date"], row["open"], row["high"], row["low"],
             row["close"], row["volume"])
        )


def bulk_load(days: int = 252, config: dict = None):
    """Load historical data for initial setup."""
    if config is None:
        config = load_config()

    # Initialize database
    db_path = init_db(config)
    conn = get_connection(config)
    logger.info(f"Database ready at {db_path}")

    # Get DB router
    try:
        router = get_router(config)
        use_duck = True
    except Exception as e:
        logger.warning(f"DbRouter unavailable, using SQLite only: {e}")
        use_duck = False

    api_key = config.get("polygon", {}).get("api_key", "")
    has_polygon = api_key and api_key != "YOUR_POLYGON_KEY"

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")

    # --- Load prices ---
    logger.info(f"Loading {days} days of SPY price data...")
    if has_polygon:
        polygon = PolygonFetcher(api_key)
        df = polygon.get_daily_bars("SPY", start_date, end_date)
        if not df.empty:
            df = df.tail(days)
            if use_duck:
                _insert_prices_duck(router, df)
            else:
                for _, row in df.iterrows():
                    conn.execute(
                        """INSERT OR REPLACE INTO prices (date, open, high, low, close, volume)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (row["date"], row["open"], row["high"], row["low"],
                         row["close"], row["volume"])
                    )
                conn.commit()
            logger.info(f"Loaded {len(df)} days from Polygon")
        else:
            logger.warning("Polygon returned no data, falling back to yfinance")
            has_polygon = False

    if not has_polygon:
        fallback = FallbackFetcher(config=config)
        df = fallback.get_daily_bars_yf("SPY", days=days)
        if not df.empty:
            if use_duck:
                _insert_prices_duck(router, df)
            else:
                for _, row in df.iterrows():
                    conn.execute(
                        """INSERT OR REPLACE INTO prices (date, open, high, low, close, volume)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (row["date"], row["open"], row["high"], row["low"],
                         row["close"], row["volume"])
                    )
                conn.commit()
            logger.info(f"Loaded {len(df)} days from yfinance")
        else:
            logger.error("No price data loaded from any source")

    # --- Load macro data ---
    logger.info("Loading macro data from FRED...")
    fallback = FallbackFetcher(config=config)
    macro = fallback.get_macro_fred()
    today = datetime.now().strftime("%Y-%m-%d")
    if use_duck:
        router.write_analytics(
            """INSERT OR REPLACE INTO macro (date, vix, vix_change, us10y_yield,
               dxy, fed_funds, gold, crude) VALUES (?,?,?,?,?,?,?,?)""",
            (today, macro.get("vix"), macro.get("vix_change"),
             macro.get("us10y_yield"), macro.get("dxy"),
             macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
        )
    else:
        conn.execute(
            """INSERT OR REPLACE INTO macro (date, vix, vix_change, us10y_yield,
               dxy, fed_funds, gold, crude) VALUES (?,?,?,?,?,?,?,?)""",
            (today, macro.get("vix"), macro.get("vix_change"),
             macro.get("us10y_yield"), macro.get("dxy"),
             macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
        )
        conn.commit()
    logger.info(f"Macro data loaded: {macro}")

    # --- Load news (operational — stays in SQLite) ---
    logger.info("Loading recent news...")
    news = fallback.get_news_rss()
    for article in news:
        conn.execute(
            """INSERT INTO news (date, source, headline, summary, url, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (article["date"], article["source"], article["headline"],
             article["summary"], article["url"], article["fetched_at"])
        )
    conn.commit()
    logger.info(f"Loaded {len(news)} news articles")

    # --- Summary ---
    # Check analytics tables
    if use_duck:
        for table in ["prices", "technicals", "macro", "intraday_bars", "options_chain"]:
            count_df = router.read_analytics(f"SELECT COUNT(*) as cnt FROM {table}")
            count = int(count_df.iloc[0]["cnt"]) if not count_df.empty else 0
            logger.info(f"  {table}: {count} rows")

    # Check SQLite operational tables
    sqlite_tables = ["news", "daily_sentiment", "predictions",
                     "options_analytics", "intraday_features", "performance"]
    for table in sqlite_tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"  {table} (SQLite): {count} rows")

    conn.close()
    logger.info("Bulk load complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Initial bulk data load")
    parser.add_argument("--days", type=int, default=252, help="Trading days to load (default: 252)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    bulk_load(days=args.days, config=config)
