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
from src.data.earnings_calendar import get_earnings_features
from src.data.fed_comms import get_fed_features
from src.data.db_router import get_router, ANALYTICS_TABLES

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


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative volume weighted by price direction."""
    direction = np.sign(close.diff()).fillna(0)
    return (volume * direction).cumsum()


def compute_garman_klass_volatility(high: pd.Series, low: pd.Series,
                                     open_: pd.Series, close: pd.Series,
                                     window: int = 20) -> pd.Series:
    """Garman-Klass volatility estimator — more efficient than close-to-close."""
    log_hl = (np.log(high / low)) ** 2
    log_co = (np.log(close / open_)) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return gk.rolling(window=window).mean().apply(np.sqrt)


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                       k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


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

    # Dynamic SMA computation for all configured periods
    for period in sma_periods:
        result[f"sma_{period}"] = compute_sma(close, period)
    # Ensure sma_20 and sma_50 always exist (backward compat)
    if "sma_20" not in result.columns:
        result["sma_20"] = compute_sma(close, 20)
    if "sma_50" not in result.columns:
        result["sma_50"] = compute_sma(close, 50)

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

    # New indicators
    result["obv"] = compute_obv(close, df["volume"].astype(float))
    result["garman_klass_vol"] = compute_garman_klass_volatility(
        df["high"], df["low"], df["open"], close
    )
    stoch_k, stoch_d = compute_stochastic(df["high"], df["low"], close)
    result["stoch_k"] = stoch_k
    result["stoch_d"] = stoch_d

    return result


def store_technicals(conn: sqlite3.Connection, tech_df: pd.DataFrame, config: dict = None):
    """Store computed technicals in the database.
    Enhancement 26: Writes to DuckDB analytics if available, falls back to SQLite."""
    cols = ("date", "sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_mid", "atr_14",
            "obv", "garman_klass_vol", "stoch_k", "stoch_d")
    # Also store dynamic SMAs beyond 20/50 (e.g. sma_200)
    extra_sma_cols = [c for c in tech_df.columns if c.startswith("sma_") and c not in ("sma_20", "sma_50")]
    all_cols = list(cols) + extra_sma_cols
    placeholders = ",".join(["?"] * len(all_cols))
    col_names = ",".join(all_cols)
    sql = f"INSERT OR REPLACE INTO technicals ({col_names}) VALUES ({placeholders})"

    def _row_vals(row):
        return tuple(float(row.get(c, 0) or 0) if c != "date" else row["date"] for c in all_cols)

    try:
        router = get_router(config)
        duck = router.get_analytics_conn()
        for _, row in tech_df.iterrows():
            if pd.isna(row.get("sma_20")):
                continue
            duck.execute(sql, _row_vals(row))
        logger.debug("Technicals stored in DuckDB")
    except Exception as e:
        logger.warning(f"DuckDB write failed, falling back to SQLite: {e}")
        for _, row in tech_df.iterrows():
            if pd.isna(row.get("sma_20")):
                continue
            conn.execute(sql, _row_vals(row))
            conn.commit()


# --- 3A: Feature Vector Construction ---

def compute_intraday_microstructure(conn: sqlite3.Connection, date: str, config: dict = None) -> dict:
    """Compute 8 intraday microstructure features from intraday_bars for a given date.

    Enhancement 26: Reads intraday_bars from DuckDB, prev close from DuckDB prices.
    Falls back to SQLite conn if DuckDB unavailable.

    Returns dict with keys: opening_gap_pct, opening_range_breakout,
    close_vs_high_pct, close_vs_low_pct, afternoon_reversal,
    institutional_hour_vol, tick_divergence, vwap_reclaim_count.
    All NaN if no intraday bars exist for the date.
    """
    fallback = {
        "opening_gap_pct": np.nan, "opening_range_breakout": np.nan,
        "close_vs_high_pct": np.nan, "close_vs_low_pct": np.nan,
        "afternoon_reversal": np.nan, "institutional_hour_vol": np.nan,
        "tick_divergence": np.nan, "vwap_reclaim_count": np.nan,
    }
    try:
        # Try DuckDB first for analytics tables
        try:
            router = get_router(config)
            bars = router.read_analytics(
                "SELECT timestamp, open, high, low, close, volume, vwap "
                "FROM intraday_bars WHERE ticker='SPY' AND timestamp LIKE ? ORDER BY timestamp",
                (f"{date}%",),
            )
            if bars.empty or len(bars) < 10:
                return fallback

            day_open = float(bars.iloc[0]["open"])
            day_high = float(bars["high"].max())
            day_low = float(bars["low"].min())
            day_close = float(bars.iloc[-1]["close"])

            # Previous day close from DuckDB prices
            prev_df = router.read_analytics(
                "SELECT close FROM prices WHERE date < ? ORDER BY date DESC LIMIT 1",
                (date,),
            )
            prev_close = float(prev_df.iloc[0]["close"]) if not prev_df.empty else day_open
        except Exception:
            # Fall back to SQLite
            bars = pd.read_sql_query(
                "SELECT timestamp, open, high, low, close, volume, vwap "
                "FROM intraday_bars WHERE ticker='SPY' AND timestamp LIKE ? ORDER BY timestamp",
                conn, params=(f"{date}%",),
            )
            if bars.empty or len(bars) < 10:
                return fallback

            day_open = float(bars.iloc[0]["open"])
            day_high = float(bars["high"].max())
            day_low = float(bars["low"].min())
            day_close = float(bars.iloc[-1]["close"])

            prev = conn.execute(
                "SELECT close FROM prices WHERE date < ? ORDER BY date DESC LIMIT 1",
                (date,),
            ).fetchone()
            prev_close = float(prev[0]) if prev else day_open

        # 1. opening_gap_pct
        opening_gap_pct = (day_open - prev_close) / prev_close if prev_close else 0.0

        # 2. opening_range_breakout — first 30 min
        # Detect bar interval: if <50 bars it's likely 5-min, else 5-sec
        bar_count = len(bars)
        if bar_count < 100:
            # 5-min bars: 6 bars = 30 min, last 18 bars = 90 min
            first_30 = bars.head(6)
            last_90_count = 18
        else:
            # 5-sec bars: 360 bars = 30 min, last 1080 bars = 90 min
            first_30 = bars.head(360)
            last_90_count = 1080
        or_high = float(first_30["high"].max())
        or_low = float(first_30["low"].min())
        if day_close > or_high:
            opening_range_breakout = 1
        elif day_close < or_low:
            opening_range_breakout = -1
        else:
            opening_range_breakout = 0

        # 3. close_vs_high_pct
        close_vs_high_pct = (day_close - day_high) / day_high if day_high else 0.0

        # 4. close_vs_low_pct
        close_vs_low_pct = (day_close - day_low) / day_low if day_low else 0.0

        # 5. afternoon_reversal — morning = first half, afternoon = last 90 min
        mid = len(bars) // 2
        morning_dir = float(bars.iloc[mid]["close"]) - day_open
        last_90_start = max(0, len(bars) - last_90_count)
        afternoon_dir = day_close - float(bars.iloc[last_90_start]["close"])
        afternoon_reversal = 1 if (morning_dir > 0 and afternoon_dir < 0) or \
                                   (morning_dir < 0 and afternoon_dir > 0) else 0

        # 6. institutional_hour_vol — 9:30-11:00 vs 14:00-16:00
        bars["ts_str"] = bars["timestamp"].astype(str)
        morning_vol = bars[bars["ts_str"].str.contains(
            r" (09:3|09:4|09:5|10:|11:0)", regex=True, na=False
        )]["volume"].sum()
        afternoon_vol = bars[bars["ts_str"].str.contains(
            r" (14:|15:)", regex=True, na=False
        )]["volume"].sum()
        institutional_hour_vol = float(morning_vol) / max(float(afternoon_vol), 1.0)

        # 7. tick_divergence — approximate with count of bars with large moves
        pct_moves = bars["close"].pct_change().abs()
        # Threshold depends on bar interval
        tick_threshold = 0.002 if bar_count < 100 else 0.001  # 0.2% for 5-min, 0.1% for 5-sec
        extreme_count = int((pct_moves > tick_threshold).sum())
        tick_divergence = extreme_count / max(len(bars), 1)

        # 8. vwap_reclaim_count — number of times close crossed VWAP
        if "vwap" in bars.columns and bars["vwap"].notna().any():
            above_vwap = (bars["close"] > bars["vwap"]).astype(int)
            vwap_reclaim_count = int(above_vwap.diff().abs().sum()) // 2
        else:
            vwap_reclaim_count = 0

        return {
            "opening_gap_pct": opening_gap_pct,
            "opening_range_breakout": opening_range_breakout,
            "close_vs_high_pct": close_vs_high_pct,
            "close_vs_low_pct": close_vs_low_pct,
            "afternoon_reversal": afternoon_reversal,
            "institutional_hour_vol": institutional_hour_vol,
            "tick_divergence": tick_divergence,
            "vwap_reclaim_count": vwap_reclaim_count,
        }
    except Exception:
        return fallback


def build_feature_vector(conn: sqlite3.Connection, date: str = None, config: dict = None) -> Optional[pd.DataFrame]:
    """Build the 35+ feature vector for model training/prediction.

    Enhancement 26: Reads analytics tables from DuckDB, operational from SQLite,
    merges in Python. Falls back to single SQLite JOIN if DuckDB unavailable.

    Returns DataFrame with one row per date, all features as columns.
    """
    # Try DuckDB router first
    try:
        router = get_router(config)
        df = router.read_feature_join(date)
    except Exception as e:
        logger.warning(f"DuckDB feature join failed, falling back to SQLite: {e}")
        df = pd.DataFrame()

    if df.empty:
        # Fallback: original single-DB JOIN on SQLite
        query = """
        SELECT
            p.date,
            p.open, p.high, p.low, p.close, p.volume,
            t.sma_20, t.sma_50, t.rsi_14,
            t.macd, t.macd_signal, t.macd_hist,
            t.bb_upper, t.bb_lower, t.atr_14,
            m.vix, m.vix_change, m.us10y_yield, m.dxy, m.fed_funds, m.gold, m.crude,
            m.vix9d, m.vix3m, m.vix6m, m.vvix, m.skew_index,
            m.hy_spread, m.tlt_spy_ratio, m.eem_spy_ratio,
            m.copper_gold_ratio, m.xlk_xlf_ratio, m.xlk_xle_ratio,
            s.score as sentiment_score, s.confidence as sentiment_confidence,
            s.article_count, s.positive_ratio, s.negative_ratio,
            s.macro_sentiment, s.earnings_sentiment,
            s.geopolitical_sentiment, s.technical_sentiment,
            s.sentiment_dispersion, s.sentiment_velocity,
            i.vwap_spread, i.intraday_momentum, i.intraday_range, i.volume_ratio,
            o.put_call_ratio, o.max_pain, o.iv_skew, o.gex,
            o.vanna_exposure, o.charm_exposure, o.zero_dte_pcr
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

    # --- P3: Extended options derived features ---
    # Fill None/NaN in options columns before computing derived features
    for oc in ["gex", "max_pain", "vanna_exposure", "charm_exposure", "zero_dte_pcr"]:
        if oc in df.columns:
            df[oc] = pd.to_numeric(df[oc], errors="coerce").fillna(0)
    # GEX sign change (1 if GEX flipped sign vs previous day)
    df["gex_sign_change"] = (np.sign(df["gex"]).diff().abs() > 0).astype(int)
    # Max pain velocity (rate of change of max pain over 5 days)
    df["max_pain_velocity"] = df["max_pain"].pct_change(5)
    # Vanna normalized by close price
    df["vanna_normalized"] = df["vanna_exposure"] / df["close"].replace(0, np.nan)
    # Charm normalized by close price
    df["charm_normalized"] = df["charm_exposure"] / df["close"].replace(0, np.nan)

    # --- P3: Earnings calendar features ---
    earn_features = []
    for _, row in df.iterrows():
        try:
            earn_features.append(get_earnings_features(conn, row["date"]))
        except Exception:
            earn_features.append({"earnings_density": 0, "days_to_next_mega": 30, "earnings_week": 0})
    earn_df = pd.DataFrame(earn_features, index=df.index)
    for col in earn_df.columns:
        df[col] = earn_df[col]

    # --- P3: Fed communication features ---
    fed_features = []
    for _, row in df.iterrows():
        try:
            fed_features.append(get_fed_features(conn, row["date"]))
        except Exception:
            fed_features.append({"fomc_hawkish_score": 0, "beige_book_score": 0, "fed_sentiment_avg": 0})
    fed_df = pd.DataFrame(fed_features, index=df.index)
    for col in fed_df.columns:
        df[col] = fed_df[col]

    # --- Intraday microstructure features (Enhancement 21) ---
    micro_features = []
    for _, row in df.iterrows():
        micro_features.append(compute_intraday_microstructure(conn, row["date"], config))
    micro_df = pd.DataFrame(micro_features, index=df.index)
    for col in micro_df.columns:
        df[col] = micro_df[col]

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

    # Fill NaN for P3 options derived features
    p3_options_cols = ["vanna_exposure", "charm_exposure", "zero_dte_pcr",
                       "gex_sign_change", "max_pain_velocity",
                       "vanna_normalized", "charm_normalized"]
    for col in p3_options_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill NaN for P3 earnings features
    p3_earnings_cols = ["earnings_density", "days_to_next_mega", "earnings_week"]
    for col in p3_earnings_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Fill NaN for P3 Fed features
    p3_fed_cols = ["fomc_hawkish_score", "beige_book_score", "fed_sentiment_avg"]
    for col in p3_fed_cols:
        if col in df.columns:
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
        lambda r: int(
            bool(r.get("is_fomc_week", 0))
            or (r.get("days_to_cpi") or 999) <= 2
            or (r.get("days_to_nfp") or 999) <= 2
        ), axis=1
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
        # Intraday microstructure (Enhancement 21)
        "opening_gap_pct", "opening_range_breakout", "close_vs_high_pct",
        "close_vs_low_pct", "afternoon_reversal", "institutional_hour_vol",
        "tick_divergence", "vwap_reclaim_count",
        # Options
        "put_call_ratio", "max_pain_distance", "iv_skew", "gex_normalized",
        # P3: Extended options
        "vanna_exposure", "charm_exposure", "zero_dte_pcr",
        "gex_sign_change", "max_pain_velocity", "vanna_normalized", "charm_normalized",
        # Derived
        "price_vs_sma20_pct", "price_vs_sma50_pct", "rsi_divergence",
        "volume_trend", "atr_percentile", "momentum_5d", "momentum_10d",
        # Calendar / event features (P1)
        "days_to_fomc", "is_fomc_week", "is_fomc_day",
        "days_to_cpi", "days_to_nfp", "days_to_opex",
        "is_triple_witching", "is_quarter_end",
        "day_of_week", "week_of_month",
        # P3: Earnings calendar
        "earnings_density", "days_to_next_mega", "earnings_week",
        # P3: Fed communication
        "fomc_hawkish_score", "beige_book_score", "fed_sentiment_avg",
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
