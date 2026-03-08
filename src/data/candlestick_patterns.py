"""Candlestick pattern detection — single and double candle patterns.

Detects 10 single-candle and 8 double-candle patterns from OHLCV data.
Returns binary flags (0/1) plus a composite bullish/bearish score.
All patterns use standard candlestick analysis rules with configurable
body/shadow thresholds.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# --- Helpers ---

def _body(o, c):
    """Absolute body size."""
    return np.abs(c - o)

def _upper_shadow(h, o, c):
    """Upper shadow (wick) length."""
    return h - np.maximum(o, c)

def _lower_shadow(l, o, c):
    """Lower shadow (tail) length."""
    return np.minimum(o, c) - l

def _is_bullish(o, c):
    """True if close > open (green candle)."""
    return c > o

def _total_range(h, l):
    """High - Low."""
    return (h - l).replace(0, np.nan)


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect single and double candlestick patterns from OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close (and optionally volume).
            Must be sorted by date ascending.

    Returns:
        DataFrame with same index, containing pattern columns (0/1 flags)
        plus composite scores.
    """
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    body = _body(o, c)
    rng = _total_range(h, l)
    upper = _upper_shadow(h, o, c)
    lower = _lower_shadow(l, o, c)
    bullish = _is_bullish(o, c)
    bearish = ~bullish

    # Average body over 14 days for relative sizing
    avg_body = body.rolling(14, min_periods=5).mean()
    small_body = body < (avg_body * 0.3)
    large_body = body > (avg_body * 1.2)

    # Previous candle values (for double patterns)
    o1 = o.shift(1)
    h1 = h.shift(1)
    l1 = l.shift(1)
    c1 = c.shift(1)
    body1 = _body(o1, c1)
    bullish1 = _is_bullish(o1, c1)
    bearish1 = ~bullish1

    result = pd.DataFrame(index=df.index)

    # =============================================
    # SINGLE CANDLE PATTERNS (10)
    # =============================================

    # 1. Hammer — small body at top, long lower shadow (>=2x body), tiny upper shadow
    #    Bullish reversal signal at bottom of downtrend
    result["cdl_hammer"] = (
        (lower >= 2 * body) &
        (upper <= body * 0.3) &
        (body > 0) &
        (rng > 0)
    ).astype(int)

    # 2. Inverted Hammer — small body at bottom, long upper shadow, tiny lower shadow
    #    Bullish reversal signal at bottom of downtrend
    result["cdl_inverted_hammer"] = (
        (upper >= 2 * body) &
        (lower <= body * 0.3) &
        (body > 0) &
        (rng > 0)
    ).astype(int)

    # 3. Hanging Man — same shape as hammer but at top of uptrend
    #    Bearish reversal signal (context-dependent, flagged same as hammer shape)
    result["cdl_hanging_man"] = (
        (lower >= 2 * body) &
        (upper <= body * 0.3) &
        (body > 0) &
        (c.rolling(5).mean().shift(1) < c.shift(1))  # prior uptrend
    ).astype(int)

    # 4. Shooting Star — inverted hammer shape at top of uptrend
    #    Bearish reversal signal
    result["cdl_shooting_star"] = (
        (upper >= 2 * body) &
        (lower <= body * 0.3) &
        (body > 0) &
        (c.rolling(5).mean().shift(1) < c.shift(1))  # prior uptrend
    ).astype(int)

    # 5. Doji — open ≈ close (very small body relative to range)
    #    Indecision signal
    result["cdl_doji"] = (
        (body <= rng * 0.1) &
        (rng > 0)
    ).astype(int)

    # 6. Dragonfly Doji — doji with long lower shadow, no upper shadow
    #    Bullish reversal
    result["cdl_dragonfly_doji"] = (
        (body <= rng * 0.1) &
        (lower >= rng * 0.6) &
        (upper <= rng * 0.1) &
        (rng > 0)
    ).astype(int)

    # 7. Gravestone Doji — doji with long upper shadow, no lower shadow
    #    Bearish reversal
    result["cdl_gravestone_doji"] = (
        (body <= rng * 0.1) &
        (upper >= rng * 0.6) &
        (lower <= rng * 0.1) &
        (rng > 0)
    ).astype(int)

    # 8. Marubozu — large body, no/tiny shadows (strong conviction)
    result["cdl_marubozu"] = (
        large_body &
        (upper <= body * 0.05) &
        (lower <= body * 0.05)
    ).astype(int)

    # 9. Spinning Top — small body, shadows on both sides roughly equal
    #    Indecision signal
    result["cdl_spinning_top"] = (
        small_body &
        (upper > body * 0.5) &
        (lower > body * 0.5) &
        (rng > 0)
    ).astype(int)

    # 10. High Wave — very small body, very long shadows both sides
    #     Extreme indecision
    result["cdl_high_wave"] = (
        small_body &
        (upper >= body * 2) &
        (lower >= body * 2) &
        (rng > 0)
    ).astype(int)

    # =============================================
    # DOUBLE CANDLE PATTERNS (8)
    # =============================================

    # 1. Bullish Engulfing — bearish candle followed by larger bullish candle
    #    that fully engulfs the prior body
    result["cdl_bullish_engulfing"] = (
        bearish1 &
        bullish &
        (o <= c1) &  # open at or below prior close
        (c >= o1) &  # close at or above prior open
        (body > body1)
    ).astype(int)

    # 2. Bearish Engulfing — bullish candle followed by larger bearish candle
    result["cdl_bearish_engulfing"] = (
        bullish1 &
        bearish &
        (o >= c1) &
        (c <= o1) &
        (body > body1)
    ).astype(int)

    # 3. Bullish Harami — large bearish candle followed by small bullish candle
    #    contained within prior body
    result["cdl_bullish_harami"] = (
        bearish1 &
        bullish &
        (o >= c1) &
        (c <= o1) &
        (body < body1 * 0.5)
    ).astype(int)

    # 4. Bearish Harami — large bullish candle followed by small bearish candle
    result["cdl_bearish_harami"] = (
        bullish1 &
        bearish &
        (o <= c1) &
        (c >= o1) &
        (body < body1 * 0.5)
    ).astype(int)

    # 5. Tweezer Bottom — two candles with matching lows (within 0.1%)
    #    Bullish reversal at support
    low_match = (np.abs(l - l1) / l1.replace(0, np.nan)) < 0.001
    result["cdl_tweezer_bottom"] = (
        low_match &
        bearish1 &
        bullish
    ).astype(int)

    # 6. Tweezer Top — two candles with matching highs
    #    Bearish reversal at resistance
    high_match = (np.abs(h - h1) / h1.replace(0, np.nan)) < 0.001
    result["cdl_tweezer_top"] = (
        high_match &
        bullish1 &
        bearish
    ).astype(int)

    # 7. Piercing Line — bearish candle, then bullish candle that opens below
    #    prior low and closes above midpoint of prior body
    prior_mid = (o1 + c1) / 2
    result["cdl_piercing_line"] = (
        bearish1 &
        bullish &
        (o < l1) &
        (c > prior_mid) &
        (c < o1)
    ).astype(int)

    # 8. Dark Cloud Cover — bullish candle, then bearish candle that opens above
    #    prior high and closes below midpoint of prior body
    result["cdl_dark_cloud"] = (
        bullish1 &
        bearish &
        (o > h1) &
        (c < prior_mid) &
        (c > o1)
    ).astype(int)

    # =============================================
    # COMPOSITE SCORES
    # =============================================

    # Bullish pattern score (sum of bullish signals)
    bullish_patterns = [
        "cdl_hammer", "cdl_inverted_hammer", "cdl_dragonfly_doji",
        "cdl_bullish_engulfing", "cdl_bullish_harami",
        "cdl_tweezer_bottom", "cdl_piercing_line",
    ]
    result["cdl_bullish_score"] = result[bullish_patterns].sum(axis=1)

    # Bearish pattern score (sum of bearish signals)
    bearish_patterns = [
        "cdl_hanging_man", "cdl_shooting_star", "cdl_gravestone_doji",
        "cdl_bearish_engulfing", "cdl_bearish_harami",
        "cdl_tweezer_top", "cdl_dark_cloud",
    ]
    result["cdl_bearish_score"] = result[bearish_patterns].sum(axis=1)

    # Net signal: positive = bullish, negative = bearish
    result["cdl_net_signal"] = result["cdl_bullish_score"] - result["cdl_bearish_score"]

    # Indecision score
    result["cdl_indecision"] = (
        result["cdl_doji"] + result["cdl_spinning_top"] + result["cdl_high_wave"]
    )

    # Fill NaN with 0 (first rows won't have shift data)
    result = result.fillna(0).astype(float)

    logger.info("Detected candlestick patterns: %d rows, %d pattern columns",
                len(result), len(result.columns))
    return result
