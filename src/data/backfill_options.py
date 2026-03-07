"""Backfill historical options analytics from Polygon.io.

Uses Polygon REST API (paid plan) to reconstruct daily options metrics:
  - Put/Call ratio (by volume)
  - Max pain strike
  - IV skew (OTM put IV - OTM call IV)
  - GEX (gamma exposure)

Strategy: For each trading day, fetch the list of active SPY option contracts
via the reference endpoint (as_of=date), then fetch daily open/close for
a representative sample of near-the-money contracts to compute analytics.

To avoid excessive API calls, we sample ~50 contracts per day (25 puts + 25 calls)
near the money, which is sufficient for P/C ratio, max pain, IV skew, and GEX.

Usage:
    python -m src.data.backfill_options [--years 5] [--batch-size 20]
"""

import argparse
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.init_db import load_config
from src.data.db_router import get_router

logger = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"


class PolygonOptionsBackfiller:
    """Backfills historical options analytics using Polygon REST API."""

    def __init__(self, api_key: str):
        import requests
        self.api_key = api_key
        self.session = requests.Session()
        self.session.params = {"apiKey": self.api_key}
        self._req_count = 0
        self._window_start = time.time()

    def _rate_limit(self):
        """Respect Polygon rate limits."""
        self._req_count += 1
        elapsed = time.time() - self._window_start
        if elapsed < 60 and self._req_count >= 90:  # conservative limit
            sleep_time = 60 - elapsed + 1
            logger.debug(f"Rate limit pause: {sleep_time:.0f}s")
            time.sleep(sleep_time)
            self._req_count = 0
            self._window_start = time.time()
        elif elapsed >= 60:
            self._req_count = 0
            self._window_start = time.time()

    def _get(self, url: str, params: dict = None) -> dict | None:
        """GET with retry."""
        import requests
        for attempt in range(3):
            self._rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                elif resp.status_code == 403:
                    logger.warning("Polygon 403 — check API key and plan tier")
                    return None
                else:
                    logger.debug(f"Polygon {resp.status_code}: {resp.text[:200]}")
                    return None
            except requests.RequestException as e:
                time.sleep(2 ** attempt)
        return None

    def _paginate(self, url: str, params: dict = None) -> list:
        """Handle Polygon next_url pagination."""
        all_results = []
        while url:
            data = self._get(url, params)
            if not data:
                break
            all_results.extend(data.get("results", []))
            next_url = data.get("next_url")
            url = next_url if next_url else None
            params = None
        return all_results

    def get_contracts_for_date(self, underlying: str, date: str,
                               spot_price: float) -> pd.DataFrame:
        """Get active option contracts for a given date using reference endpoint.

        Filters to near-the-money contracts (±10% of spot) expiring within 60 days
        to keep API calls manageable.
        """
        strike_low = spot_price * 0.90
        strike_high = spot_price * 1.10
        exp_max = (pd.to_datetime(date) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")

        url = f"{BASE_URL}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "as_of": date,
            "strike_price.gte": f"{strike_low:.0f}",
            "strike_price.lte": f"{strike_high:.0f}",
            "expiration_date.lte": exp_max,
            "expiration_date.gte": date,
            "expired": "false",
            "limit": 250,
            "order": "asc",
            "sort": "strike_price",
        }
        results = self._paginate(url, params)
        if not results:
            return pd.DataFrame()

        rows = []
        for r in results:
            rows.append({
                "ticker": r.get("ticker", ""),
                "strike": r.get("strike_price", 0),
                "expiry": r.get("expiration_date", ""),
                "option_type": r.get("contract_type", "").lower(),
            })
        return pd.DataFrame(rows)

    def get_daily_open_close(self, option_ticker: str, date: str) -> dict | None:
        """Get daily OHLCV for a single option contract on a specific date.

        Uses /v1/open-close/{ticker}/{date} endpoint.
        """
        # Ensure O: prefix
        if not option_ticker.startswith("O:"):
            option_ticker = f"O:{option_ticker}"
        url = f"{BASE_URL}/v1/open-close/{option_ticker}/{date}"
        data = self._get(url, {"adjusted": "true"})
        if data and data.get("status") == "OK":
            return {
                "open": data.get("open", 0),
                "high": data.get("high", 0),
                "low": data.get("low", 0),
                "close": data.get("close", 0),
                "volume": data.get("volume", 0),
            }
        return None

    def compute_analytics_for_date(self, underlying: str, date: str,
                                    spot_price: float) -> dict | None:
        """Compute options analytics for a historical date.

        1. Get active contracts near the money
        2. Fetch daily volume for each contract
        3. Compute put/call ratio, max pain, IV skew proxy, GEX proxy
        """
        contracts = self.get_contracts_for_date(underlying, date, spot_price)
        if contracts.empty:
            return None

        # Fetch daily data for each contract
        chain_data = []
        for _, c in contracts.iterrows():
            ohlcv = self.get_daily_open_close(c["ticker"], date)
            if ohlcv and ohlcv["volume"] > 0:
                chain_data.append({
                    "strike": c["strike"],
                    "expiry": c["expiry"],
                    "option_type": c["option_type"],
                    "close": ohlcv["close"],
                    "volume": ohlcv["volume"],
                })

        if not chain_data:
            return None

        chain = pd.DataFrame(chain_data)
        calls = chain[chain["option_type"] == "call"]
        puts = chain[chain["option_type"] == "put"]

        # Put/Call ratio by volume
        call_vol = calls["volume"].sum()
        put_vol = puts["volume"].sum()
        pc_ratio = float(put_vol / call_vol) if call_vol > 0 else None

        # Max pain
        max_pain = self._calc_max_pain(chain, spot_price)

        # IV skew proxy: average OTM put price / average OTM call price
        # (not true IV, but a reasonable proxy from prices)
        otm_puts = puts[puts["strike"] < spot_price]
        otm_calls = calls[calls["strike"] > spot_price]
        iv_skew = None
        if not otm_puts.empty and not otm_calls.empty:
            avg_put_price = otm_puts["close"].mean()
            avg_call_price = otm_calls["close"].mean()
            if avg_call_price > 0:
                iv_skew = float(avg_put_price / avg_call_price - 1.0)

        # GEX proxy: not computable without Greeks, set to None
        gex = None

        return {
            "put_call_ratio": pc_ratio,
            "max_pain": max_pain,
            "iv_skew": iv_skew,
            "gex": gex,
            "vanna_exposure": None,
            "charm_exposure": None,
            "zero_dte_pcr": None,
        }

    def _calc_max_pain(self, chain: pd.DataFrame, spot: float) -> float | None:
        """Calculate max pain from volume-weighted chain."""
        strikes = chain["strike"].unique()
        if len(strikes) == 0:
            return None
        min_pain = float("inf")
        mp_strike = None
        for strike in strikes:
            call_pain = chain[chain["option_type"] == "call"].apply(
                lambda r: max(0, strike - r["strike"]) * r["volume"], axis=1).sum()
            put_pain = chain[chain["option_type"] == "put"].apply(
                lambda r: max(0, r["strike"] - strike) * r["volume"], axis=1).sum()
            total = call_pain + put_pain
            if total < min_pain:
                min_pain = total
                mp_strike = strike
        return float(mp_strike) if mp_strike else None


def backfill_options(years: int = 5, config: dict = None, batch_size: int = 20):
    """Backfill historical options analytics from Polygon.io.

    For each trading day in the window, fetches option contracts near the money,
    retrieves daily OHLCV for each, and computes put/call ratio, max pain, and
    IV skew proxy.  Results are stored in the ``options_analytics`` table.

    Args:
        years: Number of years of history to backfill.
        config: Config dict (loaded from config.yaml if None).
        batch_size: Number of dates to process before committing a progress log.
    """
    if config is None:
        config = load_config()

    router = get_router(config)

    # --- Resolve Polygon API key ---
    api_key = config.get("polygon", {}).get("api_key", "")
    if not api_key or api_key == "FROM_ENCRYPTED_DB":
        try:
            from src.data.secrets_manager import get_secret
            api_key = get_secret("polygon_api_key", fallback="")
        except Exception:
            pass
    if not api_key:
        logger.error("No Polygon API key found — cannot backfill options.")
        return

    backfiller = PolygonOptionsBackfiller(api_key)

    # --- Get trading dates with spot prices ---
    cutoff = (datetime.now() - timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")
    dates_df = router.query(
        "SELECT date, close FROM prices WHERE ticker = 'SPY' AND date >= ? ORDER BY date",
        (cutoff,),
    )
    if dates_df.empty:
        # Fallback: no ticker column
        dates_df = router.query(
            "SELECT date, close FROM prices WHERE date >= ? ORDER BY date",
            (cutoff,),
        )
    if dates_df.empty:
        logger.error("No price data found for backfill window.")
        return

    trading_dates = list(zip(dates_df["date"].tolist(), dates_df["close"].tolist()))
    logger.info(f"Found {len(trading_dates)} trading days in backfill window.")

    # --- Check which dates already have data ---
    existing_df = router.query(
        "SELECT date FROM options_analytics WHERE date >= ?", (cutoff,)
    )
    existing_dates = set(existing_df["date"].tolist()) if not existing_df.empty else set()
    to_process = [(d, p) for d, p in trading_dates if d not in existing_dates]
    logger.info(f"Skipping {len(existing_dates)} dates with existing data. "
                f"{len(to_process)} dates to backfill.")

    success = 0
    failed = 0
    for i, (date, spot) in enumerate(to_process):
        try:
            analytics = backfiller.compute_analytics_for_date("SPY", date, float(spot))
            if analytics and analytics.get("put_call_ratio") is not None:
                router.execute(
                    """INSERT OR REPLACE INTO options_analytics
                       (date, put_call_ratio, max_pain, iv_skew, gex,
                        vanna_exposure, charm_exposure, zero_dte_pcr)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (date,
                     analytics["put_call_ratio"],
                     analytics["max_pain"],
                     analytics["iv_skew"],
                     analytics["gex"],
                     analytics.get("vanna_exposure"),
                     analytics.get("charm_exposure"),
                     analytics.get("zero_dte_pcr")),
                )
                success += 1
                logger.info(f"[{i+1}/{len(to_process)}] {date}: P/C={analytics['put_call_ratio']:.3f}, "
                            f"MaxPain={analytics['max_pain']}")
            else:
                failed += 1
                logger.warning(f"[{i+1}/{len(to_process)}] {date}: no data returned")
        except Exception as e:
            failed += 1
            logger.error(f"[{i+1}/{len(to_process)}] {date}: {e}")

        if (i + 1) % batch_size == 0:
            logger.info(f"Progress: {i+1}/{len(to_process)} done ({success} ok, {failed} failed)")

    logger.info(f"Backfill complete: {success} days stored, {failed} failed out of {len(to_process)}.")
    router.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Backfill options analytics from Polygon.io")
    parser.add_argument("--years", type=int, default=5, help="Years of history (default: 5)")
    parser.add_argument("--batch-size", type=int, default=20, help="Log progress every N dates")
    args = parser.parse_args()

    config = load_config()
    backfill_options(years=args.years, config=config, batch_size=args.batch_size)
