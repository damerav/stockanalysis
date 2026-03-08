"""Enhanced Options Features — Risk-Neutral Density extraction.

Derives forward-looking signals from the SPY options chain:
- RND skewness (asymmetry of expected returns)
- RND kurtosis (tail risk / probability of extreme moves)
- IV surface slope (term structure of implied volatility)
- Volatility smile curvature
- Net dealer positioning (gamma imbalance direction)

Uses existing Polygon options chain data from polygon_fetcher.py.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional
from scipy import interpolate, stats

logger = logging.getLogger(__name__)


def compute_rnd_features(chain: pd.DataFrame,
                         spot_price: float) -> dict:
    """Extract Risk-Neutral Density features from an options chain.

    Args:
        chain: DataFrame from PolygonFetcher.get_options_chain()
               Must have: strike, option_type, iv, last_price, bid, ask,
               volume, open_interest, delta, gamma, expiry
        spot_price: Current underlying price

    Returns:
        Dict of RND-derived features
    """
    result = {
        "rnd_skewness": 0.0,
        "rnd_kurtosis": 0.0,
        "iv_smile_curvature": 0.0,
        "iv_term_slope": 0.0,
        "put_skew_25d": 0.0,
        "call_skew_25d": 0.0,
        "butterfly_spread": 0.0,
        "risk_reversal_25d": 0.0,
        "vol_of_vol": 0.0,
        "gamma_imbalance": 0.0,
        "oi_put_wall": 0.0,
        "oi_call_wall": 0.0,
    }

    if chain.empty or spot_price <= 0:
        return result

    try:
        # Filter to reasonable strikes (within 15% of spot)
        chain = chain.copy()
        chain["moneyness"] = chain["strike"] / spot_price
        mask = (chain["moneyness"] > 0.85) & (chain["moneyness"] < 1.15)
        chain = chain[mask]

        if len(chain) < 10:
            return result

        # Ensure IV is numeric and positive
        chain["iv"] = pd.to_numeric(chain["iv"], errors="coerce").fillna(0)
        chain = chain[chain["iv"] > 0.01]

        if len(chain) < 10:
            return result

        calls = chain[chain["option_type"] == "call"].copy()
        puts = chain[chain["option_type"] == "put"].copy()

        # --- 1. IV Smile Curvature ---
        # Fit quadratic to IV vs moneyness for calls
        if len(calls) >= 5:
            try:
                coeffs = np.polyfit(calls["moneyness"], calls["iv"], 2)
                result["iv_smile_curvature"] = float(coeffs[0])  # Quadratic term
            except Exception:
                pass

        # --- 2. Risk Reversal (25-delta) ---
        # Difference between 25-delta call IV and 25-delta put IV
        if len(calls) >= 3 and len(puts) >= 3:
            try:
                # Find options closest to 25-delta
                calls_sorted = calls.reindex(
                    (calls["delta"].abs() - 0.25).abs().sort_values().index
                )
                puts_sorted = puts.reindex(
                    (puts["delta"].abs() - 0.25).abs().sort_values().index
                )
                call_25d_iv = calls_sorted.iloc[0]["iv"]
                put_25d_iv = puts_sorted.iloc[0]["iv"]
                result["risk_reversal_25d"] = float(call_25d_iv - put_25d_iv)
                result["call_skew_25d"] = float(call_25d_iv)
                result["put_skew_25d"] = float(put_25d_iv)
            except Exception:
                pass

        # --- 3. Butterfly Spread (25d) ---
        # (25d call IV + 25d put IV) / 2 - ATM IV
        # Measures kurtosis of the risk-neutral distribution
        if len(calls) >= 3:
            try:
                atm_calls = calls.reindex(
                    (calls["moneyness"] - 1.0).abs().sort_values().index
                )
                atm_iv = atm_calls.iloc[0]["iv"]
                if result["call_skew_25d"] > 0 and result["put_skew_25d"] > 0:
                    result["butterfly_spread"] = float(
                        (result["call_skew_25d"] + result["put_skew_25d"]) / 2 - atm_iv
                    )
            except Exception:
                pass

        # --- 4. RND via Breeden-Litzenberger ---
        # Second derivative of call prices w.r.t. strike ≈ risk-neutral density
        if len(calls) >= 8:
            try:
                c = calls.sort_values("strike")
                strikes = c["strike"].values
                # Use midpoint of bid/ask for cleaner prices
                prices = ((c["bid"] + c["ask"]) / 2).values
                prices = np.where(prices > 0, prices, c["last_price"].values)

                if len(strikes) >= 8 and np.all(np.diff(strikes) > 0):
                    # Smooth with cubic spline
                    cs = interpolate.CubicSpline(strikes, prices)
                    fine_strikes = np.linspace(strikes.min(), strikes.max(), 200)
                    # Second derivative = RND (up to a discount factor)
                    rnd = cs(fine_strikes, 2)
                    rnd = np.maximum(rnd, 0)  # Density must be non-negative

                    total = np.trapz(rnd, fine_strikes)
                    if total > 0:
                        rnd_norm = rnd / total
                        # Compute moments
                        mean = np.trapz(fine_strikes * rnd_norm, fine_strikes)
                        var = np.trapz((fine_strikes - mean) ** 2 * rnd_norm, fine_strikes)
                        std = np.sqrt(max(var, 1e-10))

                        if std > 0:
                            skew = np.trapz(
                                ((fine_strikes - mean) / std) ** 3 * rnd_norm,
                                fine_strikes
                            )
                            kurt = np.trapz(
                                ((fine_strikes - mean) / std) ** 4 * rnd_norm,
                                fine_strikes
                            ) - 3  # Excess kurtosis

                            result["rnd_skewness"] = float(
                                np.clip(skew, -5, 5)
                            )
                            result["rnd_kurtosis"] = float(
                                np.clip(kurt, -10, 20)
                            )
            except Exception as e:
                logger.debug(f"RND extraction failed: {e}")

        # --- 5. Vol of Vol ---
        # Standard deviation of IV across strikes (measures uncertainty about vol)
        all_ivs = chain["iv"].dropna()
        if len(all_ivs) >= 5:
            result["vol_of_vol"] = float(all_ivs.std())

        # --- 6. Gamma Imbalance ---
        # Net gamma exposure: call gamma * OI - put gamma * OI
        # Positive = dealers short gamma (amplifies moves)
        if "gamma" in chain.columns and "open_interest" in chain.columns:
            try:
                call_gamma = (calls["gamma"] * calls["open_interest"] * 100).sum()
                put_gamma = (puts["gamma"] * puts["open_interest"] * 100).sum()
                total_gamma = abs(call_gamma) + abs(put_gamma)
                if total_gamma > 0:
                    result["gamma_imbalance"] = float(
                        (call_gamma - put_gamma) / total_gamma
                    )
            except Exception:
                pass

        # --- 7. OI Walls (support/resistance from options) ---
        # Strike with highest put OI = support, highest call OI = resistance
        if not puts.empty and "open_interest" in puts.columns:
            max_put_oi = puts.loc[puts["open_interest"].idxmax()]
            result["oi_put_wall"] = float(
                (spot_price - max_put_oi["strike"]) / spot_price
            )
        if not calls.empty and "open_interest" in calls.columns:
            max_call_oi = calls.loc[calls["open_interest"].idxmax()]
            result["oi_call_wall"] = float(
                (max_call_oi["strike"] - spot_price) / spot_price
            )

        # --- 8. IV Term Structure Slope ---
        # Compare near-term vs. far-term IV
        if "expiry" in chain.columns:
            try:
                chain["days_to_exp"] = (
                    pd.to_datetime(chain["expiry"]) - pd.Timestamp.now()
                ).dt.days
                near = chain[chain["days_to_exp"].between(1, 14)]
                far = chain[chain["days_to_exp"].between(25, 60)]
                if not near.empty and not far.empty:
                    near_iv = near["iv"].mean()
                    far_iv = far["iv"].mean()
                    if far_iv > 0:
                        result["iv_term_slope"] = float(
                            (far_iv - near_iv) / far_iv
                        )
            except Exception:
                pass

    except Exception as e:
        logger.warning(f"RND feature computation failed: {e}")

    # Cast all to native Python float
    return {k: round(float(v), 6) for k, v in result.items()}


def fetch_and_compute_rnd(config: dict = None,
                          underlying: str = "SPY") -> dict:
    """Fetch options chain from Polygon and compute RND features.

    Returns dict of feature name → value.
    """
    try:
        from src.data.polygon_fetcher import PolygonFetcher
        from src.data.secrets_manager import get_secret

        api_key = get_secret("polygon_api_key")
        if not api_key:
            # Try config.yaml fallback
            api_key = (config or {}).get("polygon", {}).get("api_key", "")
        if not api_key:
            logger.debug("No Polygon API key available for RND features")
            return _empty_rnd_features()

        fetcher = PolygonFetcher(api_key)
        chain = fetcher.get_options_chain(underlying)

        if chain.empty:
            logger.debug("Empty options chain from Polygon")
            return _empty_rnd_features()

        # Get spot price
        import yfinance as yf
        ticker = yf.Ticker(underlying)
        spot = ticker.info.get("regularMarketPrice") or ticker.info.get("previousClose", 0)
        if not spot:
            # Fallback: use last close from DB
            from src.data.db_router import get_router
            router = get_router(config)
            price_row = router.query(
                "SELECT close FROM prices ORDER BY date DESC LIMIT 1"
            )
            spot = float(price_row.iloc[0]["close"]) if not price_row.empty else 0

        if spot <= 0:
            return _empty_rnd_features()

        return compute_rnd_features(chain, spot)

    except Exception as e:
        logger.warning(f"RND feature fetch failed: {e}")
        return _empty_rnd_features()


def _empty_rnd_features() -> dict:
    """Return zero-valued RND features."""
    return {
        "rnd_skewness": 0.0,
        "rnd_kurtosis": 0.0,
        "iv_smile_curvature": 0.0,
        "iv_term_slope": 0.0,
        "put_skew_25d": 0.0,
        "call_skew_25d": 0.0,
        "butterfly_spread": 0.0,
        "risk_reversal_25d": 0.0,
        "vol_of_vol": 0.0,
        "gamma_imbalance": 0.0,
        "oi_put_wall": 0.0,
        "oi_call_wall": 0.0,
    }


RND_FEATURE_COLUMNS = list(_empty_rnd_features().keys())
