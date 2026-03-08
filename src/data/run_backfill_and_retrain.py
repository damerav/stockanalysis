"""Run full backfill pipeline + retrain model.

This is the one-shot script to:
1. Migrate DB schema (add pandas-ta columns to technicals)
2. Recompute all technicals (including pandas-ta indicators)
3. Fill market breadth zeros for historical dates
4. Backfill options analytics from Polygon (rate-limited)
5. Backfill intraday bars + features from Polygon (rate-limited)
6. Retrain the model with all features populated

Usage:
    python -m src.data.run_backfill_and_retrain [--skip-polygon] [--skip-retrain]
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/backfill_retrain.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full backfill + retrain pipeline")
    parser.add_argument("--skip-polygon", action="store_true",
                        help="Skip Polygon API backfill (options + intraday)")
    parser.add_argument("--skip-retrain", action="store_true",
                        help="Skip model retrain after backfill")
    parser.add_argument("--skip-technicals", action="store_true",
                        help="Skip technicals recomputation")
    parser.add_argument("--skip-breadth", action="store_true",
                        help="Skip market breadth zero-fill")
    parser.add_argument("--days", type=int, default=504,
                        help="Trading days to backfill (default: 504)")
    args = parser.parse_args()

    from src.data.init_db import load_config
    from src.data.db_router import get_router

    config = load_config()
    router = get_router(config)

    # Step 1: Ensure DB schema is up to date
    logger.info("=" * 60)
    logger.info("STEP 1: Ensuring DB schema (pandas-ta columns)...")
    from src.data.backfill_polygon import _ensure_tables
    _ensure_tables(router)

    # Step 2: Recompute technicals
    if not args.skip_technicals:
        logger.info("=" * 60)
        logger.info("STEP 2: Recomputing all technicals (including pandas-ta)...")
        from src.data.backfill_polygon import recompute_technicals
        recompute_technicals(router, config)
    else:
        logger.info("STEP 2: Skipped (--skip-technicals)")

    # Step 3: Fill market breadth zeros
    if not args.skip_breadth:
        logger.info("=" * 60)
        logger.info("STEP 3: Filling market breadth zeros...")
        from src.data.backfill_polygon import (
            _get_trading_dates, backfill_market_breadth
        )
        trading_dates = _get_trading_dates(router, args.days)
        if trading_dates:
            backfill_market_breadth(router, trading_dates)
        else:
            logger.warning("No trading dates found!")
    else:
        logger.info("STEP 3: Skipped (--skip-breadth)")

    # Ensure trading_dates is available for later steps
    if not args.skip_polygon:
        from src.data.backfill_polygon import _get_trading_dates
        trading_dates = _get_trading_dates(router, args.days)

    # Step 4: Polygon backfill (slow — hours with rate limiting)
    if not args.skip_polygon:
        logger.info("=" * 60)
        logger.info("STEP 4: Polygon backfill (options + intraday)...")
        from src.data.secrets_manager import get_secret
        from src.data.polygon_fetcher import PolygonFetcher
        from src.data.backfill_polygon import backfill_options, backfill_intraday

        api_key = get_secret("polygon_api_key")
        if not api_key:
            api_key = (config.get("polygon", {}) or {}).get("api_key", "")
        if not api_key:
            api_key = os.environ.get("POLYGON_API_KEY", "")

        if api_key:
            polygon = PolygonFetcher(api_key)
            backfill_options(polygon, router, trading_dates)
            backfill_intraday(polygon, router, trading_dates)
        else:
            logger.warning("No Polygon API key — skipping options/intraday backfill")
    else:
        logger.info("STEP 4: Skipped (--skip-polygon)")

    # Step 4.5: Backfill ETF fund flows
    logger.info("=" * 60)
    logger.info("STEP 4.5: Backfilling ETF fund flows...")
    try:
        from src.data.etf_fetcher import backfill_etf_flows
        backfill_etf_flows(router, years=max(args.days // 252, 5))
    except Exception as e:
        logger.error(f"ETF flow backfill failed: {e}")

    # Step 5: Retrain model
    if not args.skip_retrain:
        logger.info("=" * 60)
        logger.info("STEP 5: Retraining model with all features...")
        try:
            from src.model.trainer import SPYPredictor
            predictor = SPYPredictor(config)
            result = predictor.train()
            logger.info(f"Retrain result: {result}")
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.info("STEP 5: Skipped (--skip-retrain)")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY:")
    for table in ["technicals", "options_analytics", "intraday_features",
                   "intraday_bars", "market_breadth", "etf_flows"]:
        try:
            cnt = router.query(f"SELECT COUNT(*) as cnt FROM {table}")
            logger.info(f"  {table}: {cnt.iloc[0]['cnt']} rows")
        except Exception:
            logger.info(f"  {table}: not found")

    router.close()
    logger.info("All done!")


if __name__ == "__main__":
    main()
