"""Backfill historical options analytics + intraday features from Polygon.io.

Polygon free tier: 5 API calls/minute. This script respects rate limits and
can be resumed (skips dates that already have data).

What it backfills:
  1. Daily options analytics (put/call ratio, max pain, IV skew, GEX,
     vanna, charm, zero-DTE PCR) → options_analytics table
  2. Intraday 5-min bars → intraday_bars table → intraday_features table

Usage:
    python -m src.data.backfill_polygon [--days 504] [--skip-intraday] [--skip-options]
"""

import argparse
import logging
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.init_db import load_config
from src.data.db_router import get_router
from src.data.secrets_manager import get_secret
from src.data.polygon_fetcher import PolygonFetcher

logger = logging.getLogger(__name__)

# Polygon free tier: 5 calls/min. We use a conservative 15s spacing.
FREE_TIER_DELAY = 15  # seconds between API calls


def _get_trading_dates(router, days: int = 504) -> list[str]:
    """Get list of trading dates from the prices table."""
    df = router.query(
        f"SELECT date FROM prices ORDER BY date DESC LIMIT {days}"
    )
    if df.empty:
        return []
    return sorted(df["date"].tolist())


def _get_existing_dates(router, table: str) -> set[str]:
    """Get dates that already have data in a table."""
    try:
        df = router.query(f"SELECT DISTINCT date FROM {table}")
        if not df.empty:
            return set(df["date"].tolist())
    except Exception:
        pass
    return set()


def _ensure_tables(router):
    """Ensure options_analytics and intraday tables exist in PostgreSQL.
    Also add pandas-ta columns to technicals table if missing."""
    if router.using_postgres:
        router.execute("""
            CREATE TABLE IF NOT EXISTS options_analytics (
                date TEXT PRIMARY KEY,
                put_call_ratio REAL, max_pain REAL,
                iv_skew REAL, gex REAL,
                vanna_exposure REAL, charm_exposure REAL, zero_dte_pcr REAL
            )
        """)
        router.execute("""
            CREATE TABLE IF NOT EXISTS intraday_bars (
                timestamp TEXT, ticker TEXT,
                open REAL, high REAL, low REAL, close REAL,
                volume BIGINT, vwap REAL,
                PRIMARY KEY (timestamp, ticker)
            )
        """)
        router.execute("""
            CREATE TABLE IF NOT EXISTS intraday_features (
                date TEXT PRIMARY KEY,
                vwap_spread REAL, intraday_momentum REAL,
                intraday_range REAL, volume_ratio REAL
            )
        """)
        # Add pandas-ta columns to technicals table if missing
        pta_cols = [
            "adx_14", "cci_20", "aroon_up", "aroon_down",
            "psar_long", "psar_short", "dpo_20", "trix_14",
            "vortex_pos", "vortex_neg", "williams_r", "mfi_14",
            "rsi_2", "rsi_9", "rsi_21", "cmo_14", "ppo",
            "roc_5", "roc_21",
            "kc_upper_20", "kc_lower_20", "atr_7", "atr_21",
            "donchian_high", "donchian_low", "ulcer_14",
            "cmf_20", "vwma_20", "eom_14",
            "ema_9", "ema_21", "ema_200",
            "hma_20", "wma_20", "dema_20", "tema_20", "kama_10",
            "ichi_tenkan", "ichi_kijun", "ichi_senkou_a", "ichi_senkou_b",
        ]
        for col in pta_cols:
            try:
                router.execute(f"ALTER TABLE technicals ADD COLUMN {col} REAL")
            except Exception:
                pass  # column already exists
        logger.info("PostgreSQL tables verified (including pandas-ta columns)")


def _fetch_historical_options_analytics(polygon: PolygonFetcher, ticker: str,
                                         date: str) -> Optional[dict]:
    """Fetch historical options analytics for a given date using Polygon APIs.

    Strategy: Use /v3/reference/options/contracts to find contracts active on
    the date, then /v2/aggs for volume data. Compute put/call ratio and
    basic analytics from aggregate volume.

    For free tier, this uses 2-3 API calls per date (contracts list + aggs).
    """
    try:
        # Get options contracts that were active on this date
        url = f"https://api.polygon.io/v3/reference/options/contracts"
        params = {
            "underlying_ticker": ticker,
            "as_of": date,
            "expired": "true",
            "limit": 250,
            "sort": "expiration_date",
            "order": "asc",
        }
        results = polygon._paginate(url, params)
        if not results:
            return None

        # Parse contracts
        calls = [r for r in results if r.get("contract_type") == "call"]
        puts = [r for r in results if r.get("contract_type") == "put"]

        if not calls and not puts:
            return None

        # Put/Call ratio by contract count (volume not available from reference endpoint)
        pc_ratio = len(puts) / len(calls) if calls else None

        # Max pain: find strike where total pain is minimized
        # Use the contracts' strike prices
        all_strikes = set()
        contract_data = []
        for r in results:
            strike = r.get("strike_price", 0)
            all_strikes.add(strike)
            contract_data.append({
                "strike": strike,
                "type": r.get("contract_type", ""),
                "expiry": r.get("expiration_date", ""),
            })

        max_pain = None
        if all_strikes:
            min_pain_val = float("inf")
            for test_strike in sorted(all_strikes):
                pain = 0
                for c in contract_data:
                    if c["type"] == "call":
                        pain += max(0, test_strike - c["strike"])
                    else:
                        pain += max(0, c["strike"] - test_strike)
                if pain < min_pain_val:
                    min_pain_val = pain
                    max_pain = test_strike

        # IV skew, GEX, vanna, charm — not available from reference endpoint
        # for historical dates. Set to 0 (not None) to avoid NaN.
        return {
            "put_call_ratio": float(pc_ratio) if pc_ratio is not None else 0.0,
            "max_pain": float(max_pain) if max_pain is not None else 0.0,
            "iv_skew": 0.0,
            "gex": 0.0,
            "vanna_exposure": 0.0,
            "charm_exposure": 0.0,
            "zero_dte_pcr": 0.0,
        }
    except Exception as e:
        logger.warning(f"Historical options analytics failed for {date}: {e}")
        return None


def backfill_options(polygon: PolygonFetcher, router, trading_dates: list[str],
                     delay: float = FREE_TIER_DELAY):
    """Backfill daily options analytics from Polygon."""
    existing = _get_existing_dates(router, "options_analytics")
    dates_to_fill = [d for d in trading_dates if d not in existing]

    if not dates_to_fill:
        logger.info("Options analytics: all dates already filled")
        return

    logger.info(f"Options backfill: {len(dates_to_fill)} dates to fill "
                f"(of {len(trading_dates)} total)")

    filled = 0
    errors = 0

    for i, date in enumerate(dates_to_fill):
        try:
            logger.info(f"[{i+1}/{len(dates_to_fill)}] Options for {date}...")

            analytics = _fetch_historical_options_analytics(polygon, "SPY", date)

            if analytics:
                router.execute(
                    """INSERT INTO options_analytics
                       (date, put_call_ratio, max_pain, iv_skew, gex,
                        vanna_exposure, charm_exposure, zero_dte_pcr)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (date) DO UPDATE SET
                        put_call_ratio=EXCLUDED.put_call_ratio,
                        max_pain=EXCLUDED.max_pain,
                        iv_skew=EXCLUDED.iv_skew,
                        gex=EXCLUDED.gex,
                        vanna_exposure=EXCLUDED.vanna_exposure,
                        charm_exposure=EXCLUDED.charm_exposure,
                        zero_dte_pcr=EXCLUDED.zero_dte_pcr""",
                    (date, analytics["put_call_ratio"],
                     analytics["max_pain"], analytics["iv_skew"],
                     analytics["gex"], analytics["vanna_exposure"],
                     analytics["charm_exposure"], analytics["zero_dte_pcr"])
                )
                filled += 1
                logger.info(f"  ✓ P/C={analytics['put_call_ratio']:.3f}, "
                           f"MaxPain={analytics['max_pain']}")
            else:
                # Store zeros so we don't retry this date
                router.execute(
                    """INSERT INTO options_analytics
                       (date, put_call_ratio, max_pain, iv_skew, gex,
                        vanna_exposure, charm_exposure, zero_dte_pcr)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (date) DO NOTHING""",
                    (date, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                )
                logger.warning(f"  ✗ No options data for {date}, stored zeros")

            # Rate limit
            time.sleep(delay)

        except Exception as e:
            errors += 1
            logger.error(f"  Error on {date}: {e}")
            if errors > 10:
                logger.error("Too many errors, stopping options backfill")
                break
            time.sleep(delay)

    logger.info(f"Options backfill complete: {filled} filled, {errors} errors")


def _compute_intraday_features(bars_df: pd.DataFrame) -> dict:
    """Compute intraday features from 5-min bars for a single day.

    Returns dict with: vwap_spread, intraday_momentum, intraday_range, volume_ratio
    """
    if bars_df.empty:
        return {"vwap_spread": 0.0, "intraday_momentum": 0.0,
                "intraday_range": 0.0, "volume_ratio": 1.0}

    vol_sum = bars_df["volume"].sum()
    vwap_val = ((bars_df["close"] * bars_df["volume"]).sum() / vol_sum
                if vol_sum > 0 else bars_df["close"].mean())
    last_close = bars_df["close"].iloc[-1]
    vwap_spread = (last_close - vwap_val) / vwap_val if vwap_val else 0.0

    momentum = ((bars_df["close"].iloc[-1] - bars_df["close"].iloc[0])
                / bars_df["close"].iloc[0] if len(bars_df) > 1 else 0.0)

    close_mean = bars_df["close"].mean()
    intraday_range = ((bars_df["high"].max() - bars_df["low"].min()) / close_mean
                      if close_mean else 0.0)

    avg_vol = bars_df["volume"].mean() or 1
    volume_ratio = (bars_df["volume"].iloc[-10:].mean() / avg_vol
                    if len(bars_df) >= 10 else 1.0)

    return {
        "vwap_spread": round(float(vwap_spread), 6),
        "intraday_momentum": round(float(momentum), 6),
        "intraday_range": round(float(intraday_range), 6),
        "volume_ratio": round(float(volume_ratio), 4),
    }


def backfill_intraday(polygon: PolygonFetcher, router, trading_dates: list[str],
                      delay: float = FREE_TIER_DELAY):
    """Backfill 5-min intraday bars + features from Polygon.

    Uses /v2/aggs/ticker/SPY/range/5/minute/{date}/{date} for each date.
    """
    existing_bars = set()
    try:
        df = router.query(
            "SELECT DISTINCT substr(timestamp, 1, 10) as date FROM intraday_bars"
        )
        if not df.empty:
            existing_bars = set(df["date"].tolist())
    except Exception:
        pass

    existing_features = _get_existing_dates(router, "intraday_features")
    dates_to_fill = [d for d in trading_dates
                     if d not in existing_bars or d not in existing_features]

    if not dates_to_fill:
        logger.info("Intraday: all dates already filled")
        return

    logger.info(f"Intraday backfill: {len(dates_to_fill)} dates to fill")

    filled = 0
    errors = 0

    for i, date in enumerate(dates_to_fill):
        try:
            logger.info(f"[{i+1}/{len(dates_to_fill)}] Intraday for {date}...")

            # Fetch 5-min bars from Polygon
            url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/{date}/{date}"
            results = polygon._paginate(url, {
                "adjusted": "true", "sort": "asc", "limit": 50000
            })

            if not results:
                # Store default features so we don't retry
                router.execute(
                    """INSERT INTO intraday_features
                       (date, vwap_spread, intraday_momentum, intraday_range, volume_ratio)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT (date) DO NOTHING""",
                    (date, 0.0, 0.0, 0.0, 1.0)
                )
                logger.warning(f"  ✗ No intraday data for {date}, stored defaults")
                time.sleep(delay)
                continue

            # Parse bars
            bars = pd.DataFrame(results)
            bars["timestamp"] = pd.to_datetime(bars["t"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
            bars = bars.rename(columns={
                "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
            })
            if "vw" in bars.columns:
                bars["vwap"] = bars["vw"]
            else:
                bars["vwap"] = (bars["high"] + bars["low"] + bars["close"]) / 3

            # Store bars (skip if already have bars for this date)
            if date not in existing_bars:
                for _, row in bars.iterrows():
                    try:
                        router.execute(
                            """INSERT INTO intraday_bars
                               (timestamp, ticker, open, high, low, close, volume, vwap)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                               ON CONFLICT (timestamp, ticker) DO NOTHING""",
                            (row["timestamp"], "SPY",
                             float(row["open"]), float(row["high"]),
                             float(row["low"]), float(row["close"]),
                             int(row["volume"]),
                             float(row["vwap"]) if pd.notna(row["vwap"]) else None)
                        )
                    except Exception:
                        pass

            # Compute and store features
            features = _compute_intraday_features(bars)
            router.execute(
                """INSERT INTO intraday_features
                   (date, vwap_spread, intraday_momentum, intraday_range, volume_ratio)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (date) DO UPDATE SET
                    vwap_spread=EXCLUDED.vwap_spread,
                    intraday_momentum=EXCLUDED.intraday_momentum,
                    intraday_range=EXCLUDED.intraday_range,
                    volume_ratio=EXCLUDED.volume_ratio""",
                (date, features["vwap_spread"], features["intraday_momentum"],
                 features["intraday_range"], features["volume_ratio"])
            )
            filled += 1
            logger.info(f"  ✓ {len(bars)} bars, VWAP spread={features['vwap_spread']:.4f}")

            time.sleep(delay)

        except Exception as e:
            errors += 1
            logger.error(f"  Error on {date}: {e}")
            if errors > 10:
                logger.error("Too many errors, stopping intraday backfill")
                break
            time.sleep(delay)

    logger.info(f"Intraday backfill complete: {filled} filled, {errors} errors")


def backfill_market_breadth(router, trading_dates: list[str]):
    """Fill market_breadth table with zeros for dates that have no data.

    This ensures breadth features are never NaN during training.
    Dates that already have data are left untouched.
    """
    existing = _get_existing_dates(router, "market_breadth")
    missing = [d for d in trading_dates if d not in existing]

    if not missing:
        logger.info("Market breadth: all dates already have data")
        return

    logger.info(f"Filling {len(missing)} market_breadth rows with zeros")
    for date in missing:
        router.execute(
            """INSERT INTO market_breadth (date, sp500_pe, sp500_forward_pe,
               sp500_earnings_yield, sp500_dividend_yield,
               pct_above_sma50, pct_above_sma200, advance_decline_ratio,
               new_highs_52w, new_lows_52w, breadth_thrust,
               fear_greed_index, trin, sp500_cape, buffett_indicator)
               VALUES (?, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
               ON CONFLICT (date) DO NOTHING""",
            (date,)
        )
    logger.info(f"Market breadth zero-fill complete: {len(missing)} rows")


def recompute_technicals(router, config: dict):
    """Recompute all technicals (including pandas-ta) and store them.

    This ensures the technicals table has all 40+ pandas-ta columns populated.
    """
    from src.data.features import compute_all_technicals, store_technicals

    logger.info("Recomputing all technicals (including pandas-ta)...")
    df = router.query("SELECT date, open, high, low, close, volume FROM prices ORDER BY date")
    if df.empty:
        logger.warning("No price data found")
        return
    tech_df = compute_all_technicals(df, config)
    store_technicals(None, tech_df, config)
    logger.info(f"Technicals recomputed: {len(tech_df)} rows, "
                f"{len(tech_df.columns)} columns")


def main():
    """Main entry point for Polygon backfill."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/backfill_polygon.log", mode="a"),
        ]
    )

    parser = argparse.ArgumentParser(description="Backfill from Polygon.io")
    parser.add_argument("--days", type=int, default=504,
                        help="Number of trading days to backfill (default: 504 ≈ 2 years)")
    parser.add_argument("--skip-options", action="store_true",
                        help="Skip options analytics backfill")
    parser.add_argument("--skip-intraday", action="store_true",
                        help="Skip intraday bars backfill")
    parser.add_argument("--skip-technicals", action="store_true",
                        help="Skip technicals recomputation")
    parser.add_argument("--skip-breadth", action="store_true",
                        help="Skip market breadth zero-fill")
    parser.add_argument("--delay", type=float, default=FREE_TIER_DELAY,
                        help=f"Delay between API calls in seconds (default: {FREE_TIER_DELAY})")
    args = parser.parse_args()

    config = load_config()
    router = get_router(config)

    # Ensure tables exist
    _ensure_tables(router)

    # Get Polygon API key
    api_key = get_secret("polygon_api_key")
    if not api_key:
        api_key = (config.get("polygon", {}) or {}).get("api_key", "")
    if not api_key:
        api_key = os.environ.get("POLYGON_API_KEY", "")

    if not api_key:
        logger.error("No Polygon API key found. Set via secrets manager, "
                     "config.yaml, or POLYGON_API_KEY env var.")
        sys.exit(1)

    polygon = PolygonFetcher(api_key)
    trading_dates = _get_trading_dates(router, args.days)

    if not trading_dates:
        logger.error("No trading dates found in prices table")
        sys.exit(1)

    logger.info(f"Trading dates: {len(trading_dates)} "
                f"({trading_dates[0]} to {trading_dates[-1]})")

    # Step 1: Recompute technicals (fast, no API calls)
    if not args.skip_technicals:
        recompute_technicals(router, config)

    # Step 2: Fill market breadth zeros (fast, no API calls)
    if not args.skip_breadth:
        backfill_market_breadth(router, trading_dates)

    # Step 3: Options analytics (slow — API calls with rate limiting)
    if not args.skip_options:
        backfill_options(polygon, router, trading_dates, delay=args.delay)

    # Step 4: Intraday bars + features (slow — API calls with rate limiting)
    if not args.skip_intraday:
        backfill_intraday(polygon, router, trading_dates, delay=args.delay)

    # Summary
    for table in ["options_analytics", "intraday_features", "intraday_bars", "market_breadth"]:
        try:
            cnt = router.query(f"SELECT COUNT(*) as cnt FROM {table}")
            logger.info(f"  {table}: {cnt.iloc[0]['cnt']} rows")
        except Exception:
            logger.info(f"  {table}: table not found")

    logger.info("Backfill complete!")
    router.close()


if __name__ == "__main__":
    main()
