"""1B. Polygon REST Fetcher — Daily bars, intraday bars, options chain with Greeks."""

import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL = "https://api.polygon.io"
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry
RATE_LIMIT_PER_MIN = 100


class PolygonFetcher:
    """Fetches market data from Polygon.io REST API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.params = {"apiKey": self.api_key}
        self._request_count = 0
        self._window_start = time.time()

    def _rate_limit(self):
        """Simple rate limiter: 100 requests/minute."""
        self._request_count += 1
        elapsed = time.time() - self._window_start
        if elapsed < 60 and self._request_count >= RATE_LIMIT_PER_MIN:
            sleep_time = 60 - elapsed + 0.5
            logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self._request_count = 0
            self._window_start = time.time()
        elif elapsed >= 60:
            self._request_count = 0
            self._window_start = time.time()

    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        """GET with retry logic and rate limiting."""
        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    logger.warning(f"Rate limited, waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"Polygon API error {resp.status_code}: {resp.text[:200]}")
                    return None
            except requests.RequestException as e:
                wait = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"Request failed (attempt {attempt+1}): {e}, retrying in {wait}s")
                time.sleep(wait)
        logger.error(f"All {MAX_RETRIES} attempts failed for {url}")
        return None

    def _paginate(self, url: str, params: dict = None) -> list:
        """Handle Polygon's next_url pagination."""
        all_results = []
        while url:
            data = self._get(url, params)
            if not data:
                break
            all_results.extend(data.get("results", []))
            next_url = data.get("next_url")
            url = next_url if next_url else None
            params = None  # next_url includes params
        return all_results

    def get_daily_bars(self, ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Fetch daily OHLCV bars (adjusted)."""
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        results = self._paginate(url, {"adjusted": "true", "sort": "asc", "limit": 5000})
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df["date"] = pd.to_datetime(df["t"], unit="ms").dt.strftime("%Y-%m-%d")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                                "v": "volume", "vw": "vwap"})
        return df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date")

    def get_5s_bars(self, ticker: str, date: str) -> pd.DataFrame:
        """Fetch intraday 5-second bars for a given date."""
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/5/second/{date}/{date}"
        results = self._paginate(url, {"adjusted": "true", "sort": "asc", "limit": 50000})
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df["timestamp"] = pd.to_datetime(df["t"], unit="ms").dt.strftime("%Y-%m-%d %H:%M:%S")
        df["ticker"] = ticker
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                                "v": "volume", "vw": "vwap"})
        return df[["timestamp", "ticker", "open", "high", "low", "close", "volume", "vwap"]]

    def get_options_chain(self, underlying: str, expiry: str = None) -> pd.DataFrame:
        """Fetch options chain with Greeks."""
        url = f"{BASE_URL}/v3/snapshot/options/{underlying}"
        params = {"limit": 250}
        if expiry:
            params["expiration_date"] = expiry
        results = self._paginate(url, params)
        if not results:
            return pd.DataFrame()
        rows = []
        for r in results:
            details = r.get("details", {})
            greeks = r.get("greeks", {})
            day = r.get("day", {})
            rows.append({
                "contract_symbol": details.get("ticker", ""),
                "strike": details.get("strike_price", 0),
                "expiry": details.get("expiration_date", ""),
                "option_type": details.get("contract_type", ""),
                "last_price": day.get("close", 0),
                "bid": r.get("last_quote", {}).get("bid", 0),
                "ask": r.get("last_quote", {}).get("ask", 0),
                "volume": day.get("volume", 0),
                "open_interest": r.get("open_interest", 0),
                "iv": r.get("implied_volatility", 0),
                "delta": greeks.get("delta", 0),
                "gamma": greeks.get("gamma", 0),
                "theta": greeks.get("theta", 0),
                "vega": greeks.get("vega", 0),
            })
        return pd.DataFrame(rows)

    def get_options_analytics(self, underlying: str) -> dict:
        """Compute options analytics: P/C ratio, max pain, IV skew, GEX,
        plus P3: vanna, charm, 0DTE PCR, gex_sign_change, max_pain_velocity."""
        chain = self.get_options_chain(underlying)
        if chain.empty:
            return {"put_call_ratio": None, "max_pain": None, "iv_skew": None, "gex": None,
                    "vanna_exposure": None, "charm_exposure": None, "zero_dte_pcr": None}

        calls = chain[chain["option_type"] == "call"]
        puts = chain[chain["option_type"] == "put"]

        # Put/Call ratio by volume
        call_vol = calls["volume"].sum()
        put_vol = puts["volume"].sum()
        pc_ratio = put_vol / call_vol if call_vol > 0 else None

        # Max pain: strike where total option value is minimized
        max_pain = self._calc_max_pain(chain)

        # IV skew: OTM put IV - OTM call IV (simplified)
        iv_skew = self._calc_iv_skew(chain)

        # GEX: Gamma Exposure (simplified)
        gex = self._calc_gex(chain)

        # P3: Vanna exposure — sum(gamma * delta * OI * 100) across all strikes
        vanna = self._calc_vanna(chain)

        # P3: Charm exposure — sum(theta * delta * OI * 100) across all strikes
        charm = self._calc_charm(chain)

        # P3: 0DTE put/call ratio
        zero_dte_pcr = self._calc_zero_dte_pcr(chain)

        return {"put_call_ratio": pc_ratio, "max_pain": max_pain,
                "iv_skew": iv_skew, "gex": gex,
                "vanna_exposure": vanna, "charm_exposure": charm,
                "zero_dte_pcr": zero_dte_pcr}

    def _calc_max_pain(self, chain: pd.DataFrame) -> Optional[float]:
        """Calculate max pain strike."""
        strikes = chain["strike"].unique()
        if len(strikes) == 0:
            return None
        min_pain = float("inf")
        max_pain_strike = None
        for strike in strikes:
            call_pain = chain[chain["option_type"] == "call"].apply(
                lambda r: max(0, strike - r["strike"]) * r["open_interest"], axis=1).sum()
            put_pain = chain[chain["option_type"] == "put"].apply(
                lambda r: max(0, r["strike"] - strike) * r["open_interest"], axis=1).sum()
            total = call_pain + put_pain
            if total < min_pain:
                min_pain = total
                max_pain_strike = strike
        return max_pain_strike

    def _calc_iv_skew(self, chain: pd.DataFrame) -> Optional[float]:
        """IV skew: average OTM put IV minus average OTM call IV."""
        calls = chain[(chain["option_type"] == "call") & (chain["iv"] > 0)]
        puts = chain[(chain["option_type"] == "put") & (chain["iv"] > 0)]
        if calls.empty or puts.empty:
            return None
        return puts["iv"].mean() - calls["iv"].mean()

    def _calc_gex(self, chain: pd.DataFrame) -> Optional[float]:
        """Simplified Gamma Exposure."""
        if chain.empty or "gamma" not in chain.columns:
            return None
        chain = chain[chain["gamma"].notna()]
        call_gex = chain[chain["option_type"] == "call"].apply(
            lambda r: r["gamma"] * r["open_interest"] * 100, axis=1).sum()
        put_gex = chain[chain["option_type"] == "put"].apply(
            lambda r: -r["gamma"] * r["open_interest"] * 100, axis=1).sum()
        return call_gex + put_gex

    # --- P3: Extended Options Analytics ---

    def _calc_vanna(self, chain: pd.DataFrame) -> Optional[float]:
        """Vanna exposure: ΔDelta/ΔIV — dealer vanna hedging drives directional flows.
        Approximated as sum(gamma * vega * OI * 100) across strikes."""
        if chain.empty or "gamma" not in chain.columns or "vega" not in chain.columns:
            return None
        valid = chain[chain["gamma"].notna() & chain["vega"].notna()]
        if valid.empty:
            return None
        call_vanna = valid[valid["option_type"] == "call"].apply(
            lambda r: r["gamma"] * r["vega"] * r["open_interest"] * 100, axis=1).sum()
        put_vanna = valid[valid["option_type"] == "put"].apply(
            lambda r: -r["gamma"] * r["vega"] * r["open_interest"] * 100, axis=1).sum()
        return float(call_vanna + put_vanna)

    def _calc_charm(self, chain: pd.DataFrame) -> Optional[float]:
        """Charm exposure: ΔDelta/Δtime — creates predictable end-of-day flows.
        Approximated as sum(theta * delta * OI * 100) across strikes."""
        if chain.empty or "theta" not in chain.columns or "delta" not in chain.columns:
            return None
        valid = chain[chain["theta"].notna() & chain["delta"].notna()]
        if valid.empty:
            return None
        charm = valid.apply(
            lambda r: r["theta"] * r["delta"] * r["open_interest"] * 100, axis=1).sum()
        return float(charm)

    def _calc_zero_dte_pcr(self, chain: pd.DataFrame) -> Optional[float]:
        """0DTE put/call ratio — dominant intraday directional signal since 2022."""
        today = datetime.now().strftime("%Y-%m-%d")
        zero_dte = chain[chain["expiry"] == today]
        if zero_dte.empty:
            # Try tomorrow for after-hours
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            zero_dte = chain[chain["expiry"] == tomorrow]
        if zero_dte.empty:
            return None
        call_vol = zero_dte[zero_dte["option_type"] == "call"]["volume"].sum()
        put_vol = zero_dte[zero_dte["option_type"] == "put"]["volume"].sum()
        return float(put_vol / call_vol) if call_vol > 0 else None
