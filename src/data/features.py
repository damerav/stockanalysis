"""3A/3B. Feature Engineering — Build 35+ feature vector + compute technicals."""

import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Optional

pd.set_option('future.no_silent_downcasting', True)

from src.data.init_db import get_connection, load_config
from src.data.calendar import get_event_features, has_nearby_event

logger = logging.getLogger(__name__)


# --- 3B: Technical Indicator Computation ---

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, period: int = 20,
                      std: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = mid + std * rolling_std
    lower = mid - std * rolling_std
    return upper, mid, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_all_technicals(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Compute all technical indicators for a price DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
        config: Optional config dict with technicals section

    Returns:
        DataFrame with date + all technical columns
    """
    cfg = (config or {}).get("technicals", {})
    sma_periods = cfg.get("sma_periods", [20, 50])
    rsi_period = cfg.get("rsi_period", 14)
    macd_fast = cfg.get("macd_fast", 12)
    macd_slow = cfg.get("macd_slow", 26)
    macd_signal = cfg.get("macd_signal", 9)
    bb_period = cfg.get("bb_period", 20)
    bb_std = cfg.get("bb_std", 2)
    atr_period = cfg.get("atr_period", 14)

    df = df.sort_values("date").copy()
    close = df["close"]

    result = pd.DataFrame({"date": df["date"]})
    result["sma_20"] = compute_sma(close, sma_periods[0])
    result["sma_50"] = compute_sma(close, sma_periods[1]) if len(sma_periods) > 1 else np.nan
    result["rsi_14"] = compute_rsi(close, rsi_period)

    macd, macd_sig, macd_hist = compute_macd(close, macd_fast, macd_slow, macd_signal)
    result["macd"] = macd
    result["macd_signal"] = macd_sig
    result["macd_hist"] = macd_hist

    bb_upper, bb_mid, bb_lower = compute_bollinger(close, bb_period, bb_std)
    result["bb_upper"] = bb_upper
    result["bb_mid"] = bb_mid
    result["bb_lower"] = bb_lower

    result["atr_14"] = compute_atr(df["high"], df["low"], close, atr_period)

    return result


def store_technicals(conn: sqlite3.Connection, tech_df: pd.DataFrame):
    """Store computed technicals in the database."""
    for _, row in tech_df.iterrows():
        if pd.isna(row.get("sma_20")):
            continue
        conn.execute(
            """INSERT OR REPLACE INTO technicals
               (date, sma_20, sma_50, rsi_14, macd, macd_signal, macd_hist,
                bb_upper, bb_lower, bb_mid, atr_14)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (row["date"], row["sma_20"], row["sma_50"], row["rsi_14"],
             row["macd"], row["macd_signal"], row["macd_hist"],
             row["bb_upper"], row["bb_lower"], row["bb_mid"], row["atr_14"])
        )
    conn.commit()


# --- 3A: Feature Vector Construction ---

def build_feature_vector(conn: sqlite3.Connection, date: str = None) -> Optional[pd.DataFrame]:
    """Build the 35+ feature vector for model training/prediction.

    Joins data from: prices, technicals, macro, daily_sentiment,
    intraday_features, options_analytics.

    Returns DataFrame with one row per date, all features as columns.
    """
    query = """
    SELECT
        p.date,
        p.open, p.high, p.low, p.close, p.volume,
        -- Technical features
        t.sma_20, t.sma_50, t.rsi_14,
        t.macd, t.macd_signal, t.macd_hist,
        t.bb_upper, t.bb_lower, t.atr_14,
        -- Macro features
        m.vix, m.vix_change, m.us10y_yield, m.dxy, m.fed_funds, m.gold, m.crude,
        -- VIX term structure (P1)
        m.vix9d, m.vix3m, m.vix6m, m.vvix, m.skew_index,
        -- Cross-asset signals (P1)
        m.hy_spread, m.tlt_spy_ratio, m.eem_spy_ratio,
        m.copper_gold_ratio, m.xlk_xlf_ratio, m.xlk_xle_ratio,
        -- Sentiment features
        s.score as sentiment_score, s.confidence as sentiment_confidence,
        s.article_count, s.positive_ratio, s.negative_ratio,
        -- Decomposed sentiment (P2)
        s.macro_sentiment, s.earnings_sentiment,
        s.geopolitical_sentiment, s.technical_sentiment,
        s.sentiment_dispersion, s.sentiment_velocity,
        -- Intraday features
        i.vwap_spread, i.intraday_momentum, i.intraday_range, i.volume_ratio,
        -- Options features
        o.put_call_ratio, o.max_pain, o.iv_skew, o.gex
    FROM prices p
    LEFT JOIN technicals t ON p.date = t.date
    LEFT JOIN macro m ON p.date = m.date
    LEFT JOIN daily_sentiment s ON p.date = s.date
    LEFT JOIN intraday_features i ON p.date = i.date
    LEFT JOIN options_analytics o ON p.date = o.date
    """
    if date:
        query += f" WHERE p.date = '{date}'"
    query += " ORDER BY p.date"

    df = pd.read_sql_query(query, conn)
    if df.empty:
        return None

    # Derived features
    df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"].replace(0, np.nan)
    df["price_vs_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"].replace(0, np.nan)
    df["price_vs_sma20_pct"] = df["price_vs_sma20"] * 100
    df["price_vs_sma50_pct"] = df["price_vs_sma50"] * 100
    df["bb_upper_dist"] = (df["bb_upper"] - df["close"]) / df["close"]
    df["bb_lower_dist"] = (df["close"] - df["bb_lower"]) / df["close"]
    df["sma20_slope"] = df["sma_20"].diff(5) / df["sma_20"].shift(5)
    df["sma50_slope"] = df["sma_50"].diff(5) / df["sma_50"].shift(5)
    df["rsi_divergence"] = df["rsi_14"] - df["rsi_14"].shift(5)
    df["volume_trend"] = df["volume"] / df["volume"].rolling(20).mean()
    df["atr_percentile"] = df["atr_14"].rolling(252).rank(pct=True)
    df["momentum_5d"] = df["close"].pct_change(5)
    df["momentum_10d"] = df["close"].pct_change(10)

    # Max pain distance (if available)
    df["max_pain_distance"] = (df["close"] - df["max_pain"]) / df["close"]
    df["gex_normalized"] = df["gex"] / df["close"]

    # Fill forward macro data (reported less frequently)
    macro_cols = ["vix", "vix_change", "us10y_yield", "dxy", "fed_funds", "gold", "crude",
                  "vix9d", "vix3m", "vix6m", "vvix", "skew_index",
                  "hy_spread", "tlt_spy_ratio", "eem_spy_ratio",
                  "copper_gold_ratio", "xlk_xlf_ratio", "xlk_xle_ratio"]
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill().infer_objects(copy=False)

    # Fill NaN sentiment with neutral
    sentiment_cols = ["sentiment_score", "sentiment_confidence", "article_count",
                      "positive_ratio", "negative_ratio",
                      "macro_sentiment", "earnings_sentiment",
                      "geopolitical_sentiment", "technical_sentiment",
                      "sentiment_dispersion", "sentiment_velocity"]
    for col in sentiment_cols:
        df[col] = df[col].fillna(0)

    # --- VIX term structure derived features (P1) ---
    if "vix3m" in df.columns and "vix9d" in df.columns:
        df["vix_term_slope"] = (df["vix3m"] - df["vix9d"]) / df["vix9d"].replace(0, np.nan)
    else:
        df["vix_term_slope"] = 0.0
    if "vix6m" in df.columns and "vix3m" in df.columns and "vix9d" in df.columns:
        df["vix_term_curve"] = df["vix6m"] - 2 * df["vix3m"] + df["vix9d"]
    else:
        df["vix_term_curve"] = 0.0
    # VIX vs 20-day realised vol ratio
    realised_vol_20d = df["close"].pct_change().rolling(20).std() * np.sqrt(252) * 100
    df["vix_realised_ratio"] = df["vix"] / realised_vol_20d.replace(0, np.nan)

    # --- Calendar / event features (P1) ---
    cal_features = []
    for _, row in df.iterrows():
        try:
            d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            cal_features.append(get_event_features(d))
        except Exception:
            cal_features.append(get_event_features())
    cal_df = pd.DataFrame(cal_features, index=df.index)
    for col in cal_df.columns:
        df[col] = cal_df[col]

    # --- GAP 8: Context features ---
    # VIX percentile (90-day rolling)
    df["vix_percentile"] = df["vix"].rolling(90, min_periods=20).rank(pct=True)

    # SPY-ES z-score (placeholder — uses SPY close deviation from 20-day mean)
    spy_mean = df["close"].rolling(20).mean()
    spy_std = df["close"].rolling(20).std().replace(0, np.nan)
    df["spy_es_zscore"] = (df["close"] - spy_mean) / spy_std

    # RTH flag (always 1 for daily bars — intraday would check 9:30-16:00)
    df["rth_flag"] = 1

    # Minutes to close (0 for daily bars — populated in realtime)
    df["minutes_to_close"] = 0

    # Economic event proximity flag (now computed from calendar)
    df["event_proximity"] = cal_df.apply(
        lambda r: int(r.get("is_fomc_week", 0) or r.get("days_to_cpi", 999) <= 2
                      or r.get("days_to_nfp", 999) <= 2), axis=1
    )

    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used for model training."""
    return [
        # Technical
        "price_vs_sma20", "price_vs_sma50", "rsi_14", "macd", "macd_signal",
        "macd_hist", "bb_upper_dist", "bb_lower_dist", "atr_14",
        "sma20_slope", "sma50_slope",
        # Macro
        "vix", "vix_change", "us10y_yield", "dxy", "fed_funds", "gold", "crude",
        # VIX term structure (P1)
        "vix9d", "vix3m", "vix6m", "vvix", "skew_index",
        "vix_term_slope", "vix_term_curve", "vix_realised_ratio",
        # Cross-asset signals (P1)
        "hy_spread", "tlt_spy_ratio", "eem_spy_ratio",
        "copper_gold_ratio", "xlk_xlf_ratio", "xlk_xle_ratio",
        # Sentiment
        "sentiment_score", "article_count", "positive_ratio", "negative_ratio",
        # Decomposed sentiment (P2)
        "macro_sentiment", "earnings_sentiment",
        "geopolitical_sentiment", "technical_sentiment",
        "sentiment_dispersion", "sentiment_velocity",
        # Intraday
        "vwap_spread", "intraday_momentum", "intraday_range", "volume_ratio",
        # Options
        "put_call_ratio", "max_pain_distance", "iv_skew", "gex_normalized",
        # Derived
        "price_vs_sma20_pct", "price_vs_sma50_pct", "rsi_divergence",
        "volume_trend", "atr_percentile", "momentum_5d", "momentum_10d",
        # Calendar / event features (P1)
        "days_to_fomc", "is_fomc_week", "is_fomc_day",
        "days_to_cpi", "days_to_nfp", "days_to_opex",
        "is_triple_witching", "is_quarter_end",
        "day_of_week", "week_of_month",
        # Context (GAP 8)
        "vix_percentile", "spy_es_zscore", "rth_flag",
        "minutes_to_close", "event_proximity",
    ]


def get_adaptive_neutral_threshold(vix_level: float, base_threshold: float = 0.003) -> float:
    """Scale neutral zone with VIX to maintain consistent signal quality.

    When VIX is low (~12), threshold narrows to ~±0.2% (small moves are directional).
    When VIX is high (~35), threshold widens to ~±0.58% (moderate moves are noise).
    """
    vix_baseline = 18.0  # long-run VIX average
    if vix_level is None or vix_level <= 0:
        return base_threshold
    adaptive = base_threshold * (vix_level / vix_baseline)
    return max(0.001, min(0.008, adaptive))


def get_target(df: pd.DataFrame, threshold: float = 0.003,
               adaptive: bool = True) -> pd.Series:
    """Compute next-day direction target: UP(1), DOWN(-1), NEUTRAL(0).

    Args:
        df: DataFrame with 'close' column and optionally 'vix'
        threshold: Base ±0.3% daily return for neutral zone
        adaptive: If True and 'vix' column exists, scale threshold per regime
    """
    returns = df["close"].pct_change().shift(-1)  # next-day return
    target = pd.Series(0, index=df.index, dtype=int)

    if adaptive and "vix" in df.columns:
        # Per-row adaptive threshold based on VIX level
        for i in df.index:
            vix_val = df.loc[i, "vix"] if pd.notna(df.loc[i, "vix"]) else 18.0
            thresh = get_adaptive_neutral_threshold(vix_val, threshold)
            if pd.notna(returns.loc[i]):
                if returns.loc[i] > thresh:
                    target.loc[i] = 1
                elif returns.loc[i] < -thresh:
                    target.loc[i] = -1
    else:
        target[returns > threshold] = 1
        target[returns < -threshold] = -1

    return target
