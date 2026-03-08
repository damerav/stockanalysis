"""3A/3B. Feature Engineering — Build 35+ feature vector + compute technicals."""

import logging
import os
import numpy as np
import pandas as pd
from datetime import date, datetime
from typing import Optional

pd.set_option('future.no_silent_downcasting', True)

from src.data.init_db import load_config
from src.data.calendar import get_event_features, has_nearby_event
from src.data.earnings_calendar import get_earnings_features
from src.data.fed_comms import get_fed_features
from src.data.db_router import get_router, DbRouter, ANALYTICS_TABLES
from src.data.geopolitical_features import (
    compute_daily_geopolitical_features,
    compute_daily_finbert_features,
    compute_oil_shock_features,
    compute_flight_to_safety_features,
)

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

    # --- Comprehensive technicals via pandas-ta ---
    try:
        import pandas_ta as ta
        pta_df = df[["open", "high", "low", "close", "volume"]].copy()
        pta_df.columns = ["Open", "High", "Low", "Close", "Volume"]
        H, L, C, V = pta_df["High"], pta_df["Low"], pta_df["Close"], pta_df["Volume"]

        # Trend
        _adx = ta.adx(H, L, C, length=14)
        if _adx is not None:
            result["adx_14"] = _adx.iloc[:, 0]  # ADX column
        _aroon = ta.aroon(H, L, length=14)
        if _aroon is not None:
            result["aroon_up"] = _aroon.iloc[:, 0]
            result["aroon_down"] = _aroon.iloc[:, 1]
        result["cci_20"] = ta.cci(H, L, C, length=20)
        _psar = ta.psar(H, L, C)
        if _psar is not None:
            result["psar_long"] = _psar.iloc[:, 0]
            result["psar_short"] = _psar.iloc[:, 1]
        result["dpo_20"] = ta.dpo(C, length=20)
        _trix = ta.trix(C, length=14)
        if _trix is not None:
            result["trix_14"] = _trix.iloc[:, 0]
        _vortex = ta.vortex(H, L, C, length=14)
        if _vortex is not None:
            result["vortex_pos"] = _vortex.iloc[:, 0]
            result["vortex_neg"] = _vortex.iloc[:, 1]

        # Momentum
        result["williams_r"] = ta.willr(H, L, C, length=14)
        result["mfi_14"] = ta.mfi(H, L, C, V, length=14)
        result["rsi_2"] = ta.rsi(C, length=2)
        result["rsi_9"] = ta.rsi(C, length=9)
        result["rsi_21"] = ta.rsi(C, length=21)
        result["cmo_14"] = ta.cmo(C, length=14)
        _ppo = ta.ppo(C)
        if _ppo is not None:
            result["ppo"] = _ppo.iloc[:, 0]
        result["roc_5"] = ta.roc(C, length=5)
        result["roc_21"] = ta.roc(C, length=21)

        # Volatility
        _kc = ta.kc(H, L, C, length=20)
        if _kc is not None:
            result["kc_upper_20"] = _kc.iloc[:, 0]
            result["kc_lower_20"] = _kc.iloc[:, 2]
        result["atr_7"] = ta.atr(H, L, C, length=7)
        result["atr_21"] = ta.atr(H, L, C, length=21)
        _dc = ta.donchian(H, L, length=20)
        if _dc is not None:
            result["donchian_high"] = _dc.iloc[:, 0]
            result["donchian_low"] = _dc.iloc[:, 2]
        result["ulcer_14"] = ta.ui(C, length=14)

        # Volume
        result["cmf_20"] = ta.cmf(H, L, C, V, length=20)
        result["vwma_20"] = ta.vwma(C, V, length=20)
        result["eom_14"] = ta.eom(H, L, C, V, length=14)

        # Moving Averages
        result["ema_9"] = ta.ema(C, length=9)
        result["ema_21"] = ta.ema(C, length=21)
        result["ema_200"] = ta.ema(C, length=200)
        result["hma_20"] = ta.hma(C, length=20)
        result["wma_20"] = ta.wma(C, length=20)
        result["dema_20"] = ta.dema(C, length=20)
        result["tema_20"] = ta.tema(C, length=20)
        result["kama_10"] = ta.kama(C, length=10)

        # Ichimoku Cloud
        _ichi = ta.ichimoku(H, L, C)
        if _ichi is not None and isinstance(_ichi, tuple) and len(_ichi) >= 1:
            ichi_df = _ichi[0]
            for src_col, dst_col in [("ITS_9", "ichi_tenkan"), ("IKS_26", "ichi_kijun"),
                                      ("ISA_9", "ichi_senkou_a"), ("ISB_26", "ichi_senkou_b")]:
                if src_col in ichi_df.columns:
                    result[dst_col] = ichi_df[src_col]

        logger.debug("pandas-ta indicators computed")
    except ImportError:
        logger.warning("pandas-ta not installed — skipping comprehensive technicals")
    except Exception as e:
        logger.warning("pandas-ta computation failed: %s", e)

    return result


def store_technicals(conn, tech_df: pd.DataFrame, config: dict = None):
    """Store computed technicals in the database.
    Routes to PostgreSQL via DbRouter. conn parameter kept for backward compat but ignored."""
    base_cols = ("date", "sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_hist",
                "bb_upper", "bb_lower", "bb_mid", "atr_14",
                "obv", "garman_klass_vol", "stoch_k", "stoch_d")
    # pandas-ta columns (v2.8+)
    pta_cols = (
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
    )
    # Also store dynamic SMAs beyond 20/50 (e.g. sma_200)
    extra_sma_cols = [c for c in tech_df.columns if c.startswith("sma_") and c not in ("sma_20", "sma_50")]
    # Only include pandas-ta cols that actually exist in the DataFrame
    available_pta = [c for c in pta_cols if c in tech_df.columns]
    all_cols = list(base_cols) + extra_sma_cols + available_pta
    placeholders = ",".join(["?"] * len(all_cols))
    col_names = ",".join(all_cols)
    sql = f"INSERT OR REPLACE INTO technicals ({col_names}) VALUES ({placeholders})"

    def _row_vals(row):
        return tuple(float(row.get(c, 0) or 0) if c != "date" else row["date"] for c in all_cols)

    try:
        router = get_router(config)
        if router.using_postgres:
            # Use PostgreSQL upsert
            pg_placeholders = ",".join(["%s"] * len(all_cols))
            update_cols = [c for c in all_cols if c != "date"]
            update_str = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
            pg_sql = (f"INSERT INTO technicals ({col_names}) VALUES ({pg_placeholders}) "
                      f"ON CONFLICT (date) DO UPDATE SET {update_str}")
            cur = router.get_pg().cursor()
            for _, row in tech_df.iterrows():
                if pd.isna(row.get("sma_20")):
                    continue
                cur.execute(pg_sql, _row_vals(row))
            cur.close()
            logger.debug("Technicals stored in PostgreSQL")
        else:
            for _, row in tech_df.iterrows():
                if pd.isna(row.get("sma_20")):
                    continue
                router.execute(sql, _row_vals(row))
            logger.debug("Technicals stored via router")
    except Exception as e:
        logger.error(f"Failed to store technicals: {e}")
        raise



# --- 3A: Feature Vector Construction ---

def compute_intraday_microstructure(conn, date: str, config: dict = None) -> dict:
    """Compute 8 intraday microstructure features from intraday_bars for a given date.

    Reads intraday_bars and prices via DbRouter (PostgreSQL primary).
    conn parameter kept for backward compat but ignored.

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
        router = get_router(config)
        bars = router.query(
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

        # Previous day close from prices table
        prev_df = router.query(
            "SELECT close FROM prices WHERE date < ? ORDER BY date DESC LIMIT 1",
            (date,),
        )
        prev_close = float(prev_df.iloc[0]["close"]) if not prev_df.empty else day_open

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
            r" (?:09:3|09:4|09:5|10:|11:0)", regex=True, na=False
        )]["volume"].sum()
        afternoon_vol = bars[bars["ts_str"].str.contains(
            r" (?:14:|15:)", regex=True, na=False
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


def build_feature_vector(conn, date: str = None, config: dict = None) -> Optional[pd.DataFrame]:
    """Build the 35+ feature vector for model training/prediction.

    Reads all tables via DbRouter (PostgreSQL primary).
    conn parameter kept for backward compat but ignored when router is available.

    Returns DataFrame with one row per date, all features as columns.
    """
    router = get_router(config)
    df = router.read_feature_join(date)

    if df.empty:
        return None

    # --- Market breadth & index fundamentals ---
    try:
        breadth_df = router.query(
            "SELECT date, sp500_pe, sp500_forward_pe, sp500_earnings_yield, "
            "sp500_dividend_yield, pct_above_sma50, pct_above_sma200, "
            "advance_decline_ratio, new_highs_52w, new_lows_52w, breadth_thrust, "
            "fear_greed_index, trin "
            "FROM market_breadth ORDER BY date"
        )
        if not breadth_df.empty:
            df = df.merge(breadth_df, on="date", how="left")
            # Forward-fill (breadth data may not exist for every date yet)
            breadth_cols = ["sp500_pe", "sp500_forward_pe", "sp500_earnings_yield",
                           "sp500_dividend_yield", "pct_above_sma50", "pct_above_sma200",
                           "advance_decline_ratio", "new_highs_52w", "new_lows_52w",
                           "breadth_thrust", "fear_greed_index", "trin"]
            for col in breadth_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(0)
        else:
            for col in ["sp500_pe", "sp500_forward_pe", "sp500_earnings_yield",
                        "sp500_dividend_yield", "pct_above_sma50", "pct_above_sma200",
                        "advance_decline_ratio", "new_highs_52w", "new_lows_52w",
                        "breadth_thrust", "fear_greed_index", "trin"]:
                df[col] = 0.0
    except Exception as e:
        logger.debug(f"Market breadth features failed: {e}")
        for col in ["sp500_pe", "sp500_forward_pe", "sp500_earnings_yield",
                    "sp500_dividend_yield", "pct_above_sma50", "pct_above_sma200",
                    "advance_decline_ratio", "new_highs_52w", "new_lows_52w",
                    "breadth_thrust", "fear_greed_index", "trin"]:
            df[col] = 0.0

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

    # --- Lagged return features (short-term momentum/mean-reversion signals) ---
    df["return_1d"] = df["close"].pct_change(1)
    df["return_2d"] = df["close"].pct_change(2)
    df["return_3d"] = df["close"].pct_change(3)
    df["momentum_20d"] = df["close"].pct_change(20)
    # Overnight gap (open vs previous close)
    df["overnight_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    # Intraday return (close vs open same day)
    df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
    # High-low range as % of close
    df["daily_range_pct"] = (df["high"] - df["low"]) / df["close"]
    # Close position within daily range (0=low, 1=high)
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    # RSI rate of change
    df["rsi_roc"] = df["rsi_14"].diff(1)
    # Volume spike (today vs yesterday)
    df["volume_spike"] = df["volume"] / df["volume"].shift(1).replace(0, np.nan)
    # VIX mean reversion signal (distance from 20-day mean)
    if "vix" in df.columns:
        vix_ma20 = df["vix"].rolling(20).mean()
        df["vix_mean_reversion"] = (df["vix"] - vix_ma20) / vix_ma20.replace(0, np.nan)

    # --- Price level features ---
    # 52-week high/low proximity (how far from yearly extremes)
    high_52w = df["high"].rolling(252, min_periods=100).max()
    low_52w = df["low"].rolling(252, min_periods=100).min()
    df["pct_from_52w_high"] = (df["close"] - high_52w) / high_52w.replace(0, np.nan)
    df["pct_from_52w_low"] = (df["close"] - low_52w) / low_52w.replace(0, np.nan)
    # Previous day's high/low proximity
    df["price_vs_prev_high"] = (df["close"] - df["high"].shift(1)) / df["high"].shift(1).replace(0, np.nan)
    df["price_vs_prev_low"] = (df["close"] - df["low"].shift(1)) / df["low"].shift(1).replace(0, np.nan)
    # Distance from nearest $50 round number (psychological level)
    round_50 = (df["close"] / 50).round() * 50
    df["dist_from_round_50"] = (df["close"] - round_50) / round_50.replace(0, np.nan)

    # --- Trend persistence & breakout features ---
    # Consecutive up/down days
    up_day = (df["close"] > df["close"].shift(1)).astype(int)
    down_day = (df["close"] < df["close"].shift(1)).astype(int)
    up_streak = up_day.groupby((up_day != up_day.shift()).cumsum()).cumcount() + 1
    df["consecutive_up_days"] = up_streak.where(up_day == 1, 0)
    down_streak = down_day.groupby((down_day != down_day.shift()).cumsum()).cumcount() + 1
    df["consecutive_down_days"] = down_streak.where(down_day == 1, 0)
    # 20-day Donchian breakout/breakdown (close vs prior day's channel)
    don_high_20 = df["high"].rolling(20, min_periods=20).max().shift(1)
    don_low_20 = df["low"].rolling(20, min_periods=20).min().shift(1)
    df["breakout_20d"] = (df["high"] > don_high_20).astype(int)
    df["breakdown_20d"] = (df["low"] < don_low_20).astype(int)

    # --- Candlestick pattern features ---
    try:
        from src.data.candlestick_patterns import detect_patterns
        cdl_df = detect_patterns(df)
        for col in cdl_df.columns:
            df[col] = cdl_df[col].values
    except Exception as e:
        logger.debug(f"Candlestick pattern detection failed: {e}")
        cdl_cols = [
            "cdl_hammer", "cdl_inverted_hammer", "cdl_hanging_man",
            "cdl_shooting_star", "cdl_doji", "cdl_dragonfly_doji",
            "cdl_gravestone_doji", "cdl_marubozu", "cdl_spinning_top",
            "cdl_high_wave", "cdl_bullish_engulfing", "cdl_bearish_engulfing",
            "cdl_bullish_harami", "cdl_bearish_harami", "cdl_tweezer_bottom",
            "cdl_tweezer_top", "cdl_piercing_line", "cdl_dark_cloud",
            "cdl_bullish_score", "cdl_bearish_score", "cdl_net_signal",
            "cdl_indecision",
        ]
        for col in cdl_cols:
            df[col] = 0

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

    # --- P3: Earnings calendar features (batched, vectorized) ---
    try:
        all_earnings = router.query(
            "SELECT date, ticker FROM earnings_calendar"
        )
        if not all_earnings.empty:
            all_earnings["date"] = pd.to_datetime(all_earnings["date"])
            earn_dates_sorted = np.sort(all_earnings["date"].values)

            df_dates = pd.to_datetime(df["date"])
            # Vectorized: density = count within ±3 days
            densities = []
            days_to_next_list = []
            earnings_week_list = []
            for d_ts in df_dates:
                mask = (earn_dates_sorted >= (d_ts - pd.Timedelta(days=3))) & \
                       (earn_dates_sorted <= (d_ts + pd.Timedelta(days=3)))
                densities.append(int(mask.sum()))
                future = earn_dates_sorted[earn_dates_sorted >= d_ts]
                days_to_next_list.append(min(int((pd.Timestamp(future[0]) - d_ts).days), 30) if len(future) > 0 else 30)
                week_start = d_ts - pd.Timedelta(days=d_ts.weekday())
                week_end = week_start + pd.Timedelta(days=4)
                week_mask = (earn_dates_sorted >= week_start) & (earn_dates_sorted <= week_end)
                earnings_week_list.append(1 if week_mask.any() else 0)
            df["earnings_density"] = densities
            df["days_to_next_mega"] = days_to_next_list
            df["earnings_week"] = earnings_week_list
        else:
            df["earnings_density"] = 0
            df["days_to_next_mega"] = 30
            df["earnings_week"] = 0
    except Exception:
        df["earnings_density"] = 0
        df["days_to_next_mega"] = 30
        df["earnings_week"] = 0

    # --- P3: Fed communication features (batched — vectorized merge+ffill) ---
    try:
        fed_all = router.query(
            "SELECT date, type, hawkish_score FROM fed_communications ORDER BY date"
        )
        if not fed_all.empty:
            # Pivot to get latest score per type per date, then ffill
            fomc_df = fed_all[fed_all["type"] == "fomc_statement"][["date", "hawkish_score"]].rename(
                columns={"hawkish_score": "fomc_hawkish_score"}).set_index("date")
            bb_df = fed_all[fed_all["type"] == "beige_book"][["date", "hawkish_score"]].rename(
                columns={"hawkish_score": "beige_book_score"}).set_index("date")
            # Merge onto df dates and forward-fill (carry last known score)
            df = df.set_index("date")
            df = df.join(fomc_df, how="left")
            df = df.join(bb_df, how="left")
            df["fomc_hawkish_score"] = df["fomc_hawkish_score"].ffill().fillna(0).round(3)
            df["beige_book_score"] = df["beige_book_score"].ffill().fillna(0).round(3)
            df["fed_sentiment_avg"] = ((df["fomc_hawkish_score"] + df["beige_book_score"]) / 2).round(3)
            df = df.reset_index()
        else:
            df["fomc_hawkish_score"] = 0
            df["beige_book_score"] = 0
            df["fed_sentiment_avg"] = 0
    except Exception:
        df["fomc_hawkish_score"] = 0
        df["beige_book_score"] = 0
        df["fed_sentiment_avg"] = 0

    # --- Intraday microstructure features (Enhancement 21) ---
    # Only compute for dates that have intraday bars (skip bulk of historical dates)
    try:
        try:
            router = get_router(config)
            bar_dates_df = router.read_analytics(
                "SELECT DISTINCT substr(timestamp, 1, 10) as bar_date FROM intraday_bars"
            )
        except Exception:
            bar_dates_df = router.query(
                "SELECT DISTINCT substr(timestamp, 1, 10) as bar_date FROM intraday_bars"
            )
        bar_dates = set(bar_dates_df["bar_date"].tolist()) if not bar_dates_df.empty else set()
    except Exception:
        bar_dates = set()

    micro_features = []
    fallback_micro = {
        "opening_gap_pct": 0.0, "opening_range_breakout": 0.0,
        "close_vs_high_pct": 0.0, "close_vs_low_pct": 0.0,
        "afternoon_reversal": 0.0, "institutional_hour_vol": 0.0,
        "tick_divergence": 0.0, "vwap_reclaim_count": 0.0,
    }
    for _, row in df.iterrows():
        if row["date"] in bar_dates:
            micro_features.append(compute_intraday_microstructure(conn, row["date"], config))
        else:
            micro_features.append(fallback_micro)
    micro_df = pd.DataFrame(micro_features, index=df.index)
    for col in micro_df.columns:
        df[col] = micro_df[col]

    # Fill forward macro data (reported less frequently)
    macro_cols = ["vix", "vix_change", "us10y_yield", "dxy", "fed_funds", "gold", "crude",
                  "us3m_yield", "yield_curve_10y3m", "sahm_rule", "consumer_conf", "ism_pmi",
                  "vix9d", "vix3m", "vix6m", "vvix", "skew_index",
                  "hy_spread", "tlt_spy_ratio", "eem_spy_ratio",
                  "copper_gold_ratio", "xlk_xlf_ratio", "xlk_xle_ratio",
                  "xlk", "xlf", "xle",
                  "xlv", "xli", "xlu", "xlb", "xlp", "xly", "xlre", "qqq", "iwm", "dia",
                  # v2.10: Comprehensive economic metrics
                  "cpi", "core_cpi", "pce", "core_pce", "ppi",
                  "gdp", "nfp", "unemployment_rate", "initial_claims", "continuing_claims",
                  "retail_sales", "industrial_production",
                  "housing_starts", "building_permits", "case_shiller_hpi"]
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill().infer_objects(copy=False)

    # --- v2.10: Derived economic features ---
    # Inflation momentum (YoY change — monthly data forward-filled to daily)
    if "cpi" in df.columns:
        df["cpi_mom"] = df["cpi"].pct_change(periods=252)  # ~12 months of trading days
    else:
        df["cpi_mom"] = 0.0
    if "ppi" in df.columns:
        df["ppi_mom"] = df["ppi"].pct_change(periods=252)
    else:
        df["ppi_mom"] = 0.0
    # Inflation surprise (acceleration: current YoY change minus prior YoY change)
    if "cpi" in df.columns:
        cpi_yoy = df["cpi"].pct_change(periods=252)
        df["inflation_surprise"] = cpi_yoy - cpi_yoy.shift(252)
    else:
        df["inflation_surprise"] = 0.0
    # Employment momentum (3-month smoothed NFP change)
    if "nfp" in df.columns:
        df["nfp_mom"] = df["nfp"].pct_change(periods=63).rolling(63).mean()  # ~3 months
    else:
        df["nfp_mom"] = 0.0
    # Claims trend (4-week MA diff — weekly data forward-filled)
    if "initial_claims" in df.columns:
        df["claims_trend"] = df["initial_claims"].rolling(20).mean().diff()  # ~4 weeks
    else:
        df["claims_trend"] = 0.0
    # Housing momentum (YoY HPI change)
    if "case_shiller_hpi" in df.columns:
        df["hpi_mom"] = df["case_shiller_hpi"].pct_change(periods=252)
    else:
        df["hpi_mom"] = 0.0

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

    # --- Calendar / event features (P1) — vectorized ---
    try:
        date_objs = pd.to_datetime(df["date"]).dt.date
        cal_results = [get_event_features(d) for d in date_objs]
    except Exception:
        cal_results = [get_event_features() for _ in range(len(df))]
    cal_df = pd.DataFrame(cal_results, index=df.index)
    df = pd.concat([df, cal_df], axis=1)

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

    # --- News-derived features from PostgreSQL (or news.db fallback) ---
    try:
        router = get_router(config)
        # Overall volume metrics
        news_daily = router.query(
            "SELECT substr(published_at, 1, 10) as date, "
            "COUNT(*) as news_volume, "
            "COUNT(DISTINCT source) as news_source_count "
            "FROM raw_articles GROUP BY substr(published_at, 1, 10)"
        )
        # Category-specific volume
        cat_volume = router.query(
            "SELECT substr(published_at, 1, 10) as date, category, COUNT(*) as cnt "
            "FROM raw_articles WHERE category IS NOT NULL "
            "GROUP BY substr(published_at, 1, 10), category"
        )

        if not news_daily.empty:
            news_daily = news_daily.set_index("date")
            df["news_volume"] = df["date"].map(news_daily["news_volume"]).fillna(0)
            df["news_source_count"] = df["date"].map(news_daily["news_source_count"]).fillna(0)
            nv = df["news_volume"].replace(0, np.nan)
            df["news_volume_spike"] = nv / nv.rolling(5, min_periods=1).mean()
            df["news_volume_spike"] = df["news_volume_spike"].fillna(1.0)
        else:
            df["news_volume"] = 0
            df["news_source_count"] = 0
            df["news_volume_spike"] = 1.0

        # Category-specific volume features
        cat_cols = {
            "centralbanks": "news_cb_volume",
            "commodities": "news_commodity_volume",
            "forex": "news_forex_volume",
            "bonds": "news_bond_volume",
            "economic": "news_econ_volume",
            "derivatives": "news_deriv_volume",
        }
        if not cat_volume.empty:
            for cat_name, col_name in cat_cols.items():
                cat_sub = cat_volume[cat_volume["category"] == cat_name].set_index("date")["cnt"]
                df[col_name] = df["date"].map(cat_sub).fillna(0)
        else:
            for col_name in cat_cols.values():
                df[col_name] = 0
    except Exception as e:
        logger.debug(f"News features failed: {e}")
        df["news_volume"] = 0
        df["news_source_count"] = 0
        df["news_volume_spike"] = 1.0
        for col_name in ["news_cb_volume", "news_commodity_volume", "news_forex_volume",
                         "news_bond_volume", "news_econ_volume", "news_deriv_volume"]:
            df[col_name] = 0

    # Sentiment momentum (3-day change in sentiment score)
    if "sentiment_score" in df.columns:
        df["sentiment_momentum"] = df["sentiment_score"].diff(3)
        df["sentiment_momentum"] = df["sentiment_momentum"].fillna(0)
    else:
        df["sentiment_momentum"] = 0

    # --- Geopolitical risk features from news.db ---
    try:
        geo_daily = compute_daily_geopolitical_features(config)
        if not geo_daily.empty:
            geo_daily = geo_daily.set_index("date")
            for col in ["geo_risk_score", "geo_fear_score", "geo_recovery_score",
                        "geo_net_risk", "geo_article_ratio", "geo_max_risk"]:
                if col in geo_daily.columns:
                    df[col] = df["date"].map(geo_daily[col]).fillna(0)
                else:
                    df[col] = 0.0
        else:
            for col in ["geo_risk_score", "geo_fear_score", "geo_recovery_score",
                        "geo_net_risk", "geo_article_ratio", "geo_max_risk"]:
                df[col] = 0.0
    except Exception as e:
        logger.debug(f"Geopolitical features failed: {e}")
        for col in ["geo_risk_score", "geo_fear_score", "geo_recovery_score",
                    "geo_net_risk", "geo_article_ratio", "geo_max_risk"]:
            df[col] = 0.0

    # --- Oil shock & flight-to-safety features ---
    try:
        df = compute_oil_shock_features(df)
        df = compute_flight_to_safety_features(df)
    except Exception as e:
        logger.debug(f"Oil/safety features failed: {e}")
        for col in ["crude_pct_change", "crude_vs_ma20", "crude_shock",
                     "crude_momentum_5d", "gold_momentum_5d", "gold_vs_ma20",
                     "yield_change_5d", "safety_signal"]:
            if col not in df.columns:
                df[col] = 0.0

    # --- FinBERT sentiment features from news.db ---
    # Use days_back matching actual news.db coverage (avoid scanning 730 days of empty data)
    try:
        fb_daily = compute_daily_finbert_features(config, days_back=90)
        if not fb_daily.empty:
            fb_daily = fb_daily.set_index("date")
            for col in ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_score"]:
                if col in fb_daily.columns:
                    df[col] = df["date"].map(fb_daily[col]).fillna(0)
                else:
                    df[col] = 0.0
        else:
            for col in ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_score"]:
                df[col] = 0.0
    except Exception as e:
        logger.debug(f"FinBERT features failed: {e}")
        for col in ["finbert_positive", "finbert_negative", "finbert_neutral", "finbert_score"]:
            df[col] = 0.0

    # Fill NaN for new features
    new_feat_cols = ["geo_risk_score", "geo_fear_score", "geo_recovery_score",
                     "geo_net_risk", "geo_article_ratio", "geo_max_risk",
                     "crude_pct_change", "crude_vs_ma20", "crude_shock",
                     "crude_momentum_5d", "gold_momentum_5d", "gold_vs_ma20",
                     "yield_change_5d", "safety_signal",
                     "finbert_positive", "finbert_negative", "finbert_neutral", "finbert_score",
                     "news_cb_volume", "news_commodity_volume", "news_forex_volume",
                     "news_bond_volume", "news_econ_volume", "news_deriv_volume",
                     # Market breadth & fundamentals
                     "sp500_pe", "sp500_forward_pe", "sp500_earnings_yield",
                     "sp500_dividend_yield", "pct_above_sma50", "pct_above_sma200",
                     "advance_decline_ratio", "new_highs_52w", "new_lows_52w",
                     "breadth_thrust", "fear_greed_index", "trin"]
    for col in new_feat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # --- Earnings Yield Gap (SPY earnings yield vs. 10Y Treasury) ---
    # "Fed Model" — positive gap means equities cheap vs. bonds
    try:
        if "sp500_earnings_yield" in df.columns and "us10y_yield" in df.columns:
            df["earnings_yield_gap"] = (
                df["sp500_earnings_yield"] - (df["us10y_yield"] / 100.0)
            ).round(4)
        else:
            df["earnings_yield_gap"] = 0.0
    except Exception as e:
        logger.warning("Earnings yield gap failed: %s", e)
        df["earnings_yield_gap"] = 0.0

    # --- Extended Sector Rotation Ratios ---
    try:
        # Defensive vs. Offensive: (XLU + XLP) / (XLY + XLK)
        if all(c in df.columns for c in ["xlu", "xlp", "xly", "xlk"]):
            df["defensive_offensive_ratio"] = (
                (df["xlu"] + df["xlp"]) / (df["xly"] + df["xlk"] + 1e-9)
            ).round(4)
        else:
            df["defensive_offensive_ratio"] = 0.0
        # QQQ vs. IWM (growth vs. small cap risk)
        if "qqq" in df.columns and "iwm" in df.columns:
            df["qqq_iwm_ratio"] = (df["qqq"] / df["iwm"].replace(0, np.nan)).round(4).fillna(1.0)
        else:
            df["qqq_iwm_ratio"] = 1.0
        # Healthcare vs. Energy (defensive vs. cyclical)
        if "xlv" in df.columns and "xle" in df.columns:
            df["xlv_xle_ratio"] = (df["xlv"] / df["xle"].replace(0, np.nan)).round(4).fillna(1.0)
        else:
            df["xlv_xle_ratio"] = 1.0
    except Exception as e:
        logger.warning("Sector rotation features failed: %s", e)
        df["defensive_offensive_ratio"] = 0.0
        df["qqq_iwm_ratio"] = 1.0
        df["xlv_xle_ratio"] = 1.0

    # --- Multi-Timeframe Technical Features ---
    try:
        daily_dates = pd.to_datetime(df["date"])
        price_idx = df.set_index(daily_dates)[["open", "high", "low", "close", "volume"]]
        weekly = price_idx.resample("W").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()
        monthly = price_idx.resample("ME").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"
        }).dropna()

        if len(weekly) >= 14:
            w_rsi = compute_rsi(weekly["close"], 14)
            # Reindex weekly onto daily dates with forward-fill (weekly value
            # carries forward to all trading days in that week)
            w_rsi_daily = w_rsi.reindex(daily_dates, method="ffill")
            df["weekly_rsi"] = w_rsi_daily.values
            w_mom = weekly["close"].pct_change(5)
            df["weekly_momentum_5w"] = w_mom.reindex(daily_dates, method="ffill").values
            _, _, w_macd_hist = compute_macd(weekly["close"])
            df["weekly_macd_hist"] = w_macd_hist.reindex(daily_dates, method="ffill").values

        if len(monthly) >= 12:
            m_rsi = compute_rsi(monthly["close"], 14)
            df["monthly_rsi"] = m_rsi.reindex(daily_dates, method="ffill").values
            m_mom = monthly["close"].pct_change(3)
            df["monthly_momentum_3m"] = m_mom.reindex(daily_dates, method="ffill").values

        for col in ["weekly_rsi", "weekly_momentum_5w", "weekly_macd_hist",
                     "monthly_rsi", "monthly_momentum_3m"]:
            if col in df.columns:
                df[col] = df[col].bfill().fillna(0)
            else:
                df[col] = 0.0
        logger.debug("Multi-timeframe features computed")
    except Exception as e:
        logger.warning("Multi-timeframe features failed: %s", e)
        for col in ["weekly_rsi", "weekly_momentum_5w", "weekly_macd_hist",
                     "monthly_rsi", "monthly_momentum_3m"]:
            df[col] = 0.0

    # --- StockTwits Social Sentiment ---
    try:
        from src.data.social_fetcher import get_stocktwits_sentiment
        st_data = get_stocktwits_sentiment("SPY")
        for col in ["st_bullish_pct", "st_bearish_pct", "st_bull_bear_ratio", "st_message_volume"]:
            df[col] = st_data.get(col, 0.0)
        logger.debug("StockTwits sentiment: bull=%.2f", st_data.get("st_bullish_pct", 0))
    except Exception as e:
        logger.warning("StockTwits features failed: %s", e)
        for col in ["st_bullish_pct", "st_bearish_pct", "st_bull_bear_ratio", "st_message_volume"]:
            df[col] = 0.0

    # --- CAPE and Buffett from market_breadth table ---
    try:
        cape_df = router.query(
            "SELECT date, sp500_cape, buffett_indicator FROM market_breadth ORDER BY date"
        )
        if not cape_df.empty:
            df = df.merge(cape_df, on="date", how="left")
            for col in ["sp500_cape", "buffett_indicator"]:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(0)
        else:
            df["sp500_cape"] = 0.0
            df["buffett_indicator"] = 0.0
    except Exception as e:
        logger.debug("CAPE/Buffett features failed: %s", e)
        df["sp500_cape"] = 0.0
        df["buffett_indicator"] = 0.0

    # --- ETF Fund Flow features ---
    etf_flow_cols = [
        "flow_spy", "flow_qqq", "flow_iwm", "flow_tlt", "flow_hyg",
        "flow_gld", "flow_xlk", "flow_xlf", "flow_xle", "flow_eem",
        "equity_bond_flow_ratio", "growth_value_flow_ratio",
        "em_dm_flow_ratio", "flow_momentum_5d", "flow_breadth",
        "safe_haven_flow",
    ]
    try:
        col_str = ", ".join(etf_flow_cols)
        etf_df = router.query(
            f"SELECT date, {col_str} FROM etf_flows ORDER BY date"
        )
        if not etf_df.empty:
            df = df.merge(etf_df, on="date", how="left")
            for col in etf_flow_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(0)
        else:
            for col in etf_flow_cols:
                df[col] = 0.0
    except Exception as e:
        logger.debug(f"ETF flow features failed: {e}")
        for col in etf_flow_cols:
            df[col] = 0.0

    # --- CFTC Commitment of Traders features (weekly, forward-filled to daily) ---
    cot_cols = [
        "cot_commercial_net", "cot_leveraged_net", "cot_asset_mgr_net",
        "cot_spec_ratio", "cot_commercial_change", "cot_leveraged_change",
    ]
    try:
        cot_col_str = ", ".join(cot_cols)
        cot_df = router.query(
            f"SELECT date, {cot_col_str} FROM cot_data ORDER BY date"
        )
        if not cot_df.empty:
            df = df.merge(cot_df, on="date", how="left")
            for col in cot_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(0)
        else:
            for col in cot_cols:
                df[col] = 0.0
    except Exception as e:
        logger.debug(f"COT features failed: {e}")
        for col in cot_cols:
            df[col] = 0.0

    # --- NAV Premium/Discount features (SPY vs S&P 500 index) ---
    nav_cols = [
        "nav_premium_pct", "nav_premium_zscore", "nav_premium_ma5",
        "nav_premium_momentum", "nav_premium_mean_rev", "nav_premium_extreme",
        "spy_es_basis_pct", "spy_es_basis_zscore",
        "nav_premium_vol", "nav_premium_skew",
        "nav_premium_regime", "nav_creation_pressure",
    ]
    try:
        nav_col_str = ", ".join(nav_cols)
        nav_df = router.query(
            f"SELECT date, {nav_col_str} FROM nav_premium ORDER BY date"
        )
        if not nav_df.empty:
            df = df.merge(nav_df, on="date", how="left")
            for col in nav_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().fillna(0)
        else:
            for col in nav_cols:
                df[col] = 0.0
    except Exception as e:
        logger.debug(f"NAV premium features failed: {e}")
        for col in nav_cols:
            df[col] = 0.0

    # Fill NaN for all v2.8+ features
    v28_cols = ["earnings_yield_gap", "defensive_offensive_ratio", "qqq_iwm_ratio",
                "xlv_xle_ratio", "weekly_rsi", "weekly_momentum_5w", "weekly_macd_hist",
                "monthly_rsi", "monthly_momentum_3m", "st_bullish_pct", "st_bearish_pct",
                "st_bull_bear_ratio", "st_message_volume", "sp500_cape", "buffett_indicator",
                "us3m_yield", "yield_curve_10y3m", "sahm_rule", "consumer_conf", "ism_pmi",
                # v2.10: Economic metrics
                "cpi", "core_cpi", "pce", "core_pce", "ppi",
                "gdp", "nfp", "unemployment_rate", "initial_claims", "continuing_claims",
                "retail_sales", "industrial_production",
                "housing_starts", "building_permits", "case_shiller_hpi",
                "cpi_mom", "ppi_mom", "inflation_surprise",
                "nfp_mom", "claims_trend", "hpi_mom"]
    for col in v28_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # ===== FINAL NaN KILLER — no NaN or inf should ever reach the model =====
    # --- v3.0: DeepSeek narrative scoring features ---
    ds_cols = ["ds_sentiment", "ds_confidence", "ds_bull_factors", "ds_bear_factors",
               "ds_impact", "ds_sentiment_momentum", "ds_conviction", "ds_bull_bear_ratio"]
    try:
        from src.data.deepseek_scorer import compute_deepseek_features
        ds_df = compute_deepseek_features(config)
        if not ds_df.empty:
            ds_df = ds_df.set_index("date")
            for col in ds_cols:
                if col in ds_df.columns:
                    df[col] = df["date"].map(ds_df[col]).fillna(0)
                else:
                    df[col] = 0.0
        else:
            for col in ds_cols:
                df[col] = 0.0
    except Exception as e:
        logger.debug(f"DeepSeek features failed: {e}")
        for col in ds_cols:
            df[col] = 0.0

    # --- v3.0: Enhanced options RND features ---
    rnd_cols = ["rnd_skewness", "rnd_kurtosis", "iv_smile_curvature",
                "put_skew_25d", "call_skew_25d", "butterfly_spread",
                "risk_reversal_25d", "vol_of_vol", "gamma_imbalance",
                "oi_put_wall", "oi_call_wall"]
    try:
        from src.data.options_rnd import fetch_and_compute_rnd
        rnd_data = fetch_and_compute_rnd(config)
        for col in rnd_cols:
            df[col] = rnd_data.get(col, 0.0)
    except Exception as e:
        logger.debug(f"RND features failed: {e}")
        for col in rnd_cols:
            df[col] = 0.0

    # --- v3.0: LLM Alpha features ---
    try:
        from src.data.alpha_generator import compute_alpha_features, load_alphas
        alphas = load_alphas()
        if alphas:
            df = compute_alpha_features(df, alphas)
    except Exception as e:
        logger.debug(f"Alpha features failed: {e}")

    # Replace inf/-inf with NaN, then forward-fill, back-fill, and fill with 0.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)

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
        # Lagged returns & short-term signals
        "return_1d", "return_2d", "return_3d", "momentum_20d",
        "overnight_gap", "intraday_return", "daily_range_pct", "close_position",
        "rsi_roc", "volume_spike", "vix_mean_reversion",
        # Price level features
        "pct_from_52w_high", "pct_from_52w_low",
        "price_vs_prev_high", "price_vs_prev_low", "dist_from_round_50",
        # Trend persistence & breakout
        "consecutive_up_days", "consecutive_down_days",
        "breakout_20d", "breakdown_20d",
        # Calendar / event features (P1)
        "days_to_fomc", "is_fomc_week", "is_fomc_day",
        "days_to_cpi", "days_to_nfp", "days_to_opex",
        "is_triple_witching", "is_quarter_end",
        "day_of_week", "week_of_month",
        # Day-of-week one-hot + expiry/rebalancing calendar
        "is_monday", "is_tuesday", "is_wednesday", "is_thursday", "is_friday",
        "is_0dte_day", "is_month_end", "is_quarter_end_week",
        # Holiday / long-weekend effects
        "is_pre_holiday", "is_post_holiday",
        "is_long_weekend_start", "is_long_weekend_end",
        # P3: Earnings calendar
        "earnings_density", "days_to_next_mega", "earnings_week",
        # P3: Fed communication
        "fomc_hawkish_score", "beige_book_score", "fed_sentiment_avg",
        # Context (GAP 8)
        "vix_percentile", "spy_es_zscore", "rth_flag",
        "minutes_to_close", "event_proximity",
        # News-derived (expanded news.db)
        "news_volume", "news_source_count", "news_volume_spike",
        "sentiment_momentum",
        # Geopolitical risk features
        "geo_risk_score", "geo_fear_score", "geo_recovery_score",
        "geo_net_risk", "geo_article_ratio", "geo_max_risk",
        # Oil shock features
        "crude_pct_change", "crude_vs_ma20", "crude_shock", "crude_momentum_5d",
        # Flight-to-safety features
        "gold_momentum_5d", "gold_vs_ma20", "yield_change_5d", "safety_signal",
        # FinBERT sentiment features
        "finbert_positive", "finbert_negative", "finbert_score",
        # Index fundamentals
        "sp500_pe", "sp500_forward_pe", "sp500_earnings_yield", "sp500_dividend_yield",
        # Market breadth
        "pct_above_sma50", "pct_above_sma200", "advance_decline_ratio",
        "new_highs_52w", "new_lows_52w", "breadth_thrust",
        "fear_greed_index", "trin",
        # v2.8: Macro / Valuation
        "sp500_cape", "buffett_indicator",
        "sahm_rule", "yield_curve_10y3m", "us3m_yield",
        "consumer_conf", "ism_pmi", "earnings_yield_gap",
        # v2.10: Comprehensive Economic Metrics (raw)
        "cpi", "core_cpi", "pce", "core_pce", "ppi",
        "gdp", "nfp", "unemployment_rate",
        "initial_claims", "continuing_claims",
        "retail_sales", "industrial_production",
        "housing_starts", "building_permits", "case_shiller_hpi",
        # v2.10: Derived economic features
        "cpi_mom", "ppi_mom", "inflation_surprise",
        "nfp_mom", "claims_trend", "hpi_mom",
        # v2.8: Comprehensive Technicals (pandas-ta)
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
        # v2.8: Multi-Timeframe
        "weekly_rsi", "weekly_momentum_5w", "weekly_macd_hist",
        "monthly_rsi", "monthly_momentum_3m",
        # v2.8: Social Sentiment
        "st_bullish_pct", "st_bearish_pct", "st_bull_bear_ratio", "st_message_volume",
        # v2.8: Sector Rotation
        "defensive_offensive_ratio", "qqq_iwm_ratio", "xlv_xle_ratio",
        "xlv", "xli", "xlu", "xlb", "xlp", "xly", "xlre", "qqq", "iwm", "dia",
        # Sector ETF raw prices (used in rotation ratios, also direct features)
        "xlk", "xlf", "xle",
        # ETF fund flow features (institutional capital movement)
        "flow_spy", "flow_qqq", "flow_iwm", "flow_tlt", "flow_hyg",
        "flow_gld", "flow_xlk", "flow_xlf", "flow_xle", "flow_eem",
        "equity_bond_flow_ratio", "growth_value_flow_ratio",
        "em_dm_flow_ratio", "flow_momentum_5d", "flow_breadth",
        "safe_haven_flow",
        # Raw SMA values (used in derived features, also direct signals)
        "sma_20", "sma_50",
        # Bollinger Band raw values
        "bb_upper", "bb_lower",
        # Raw options columns (used in derived features)
        "gex", "max_pain",
        # Sentiment confidence
        "sentiment_confidence",
        # FinBERT neutral (complement of positive/negative)
        "finbert_neutral",
        # Category-specific news volume (worldmonitor feeds)
        "news_cb_volume", "news_commodity_volume", "news_forex_volume",
        "news_bond_volume", "news_econ_volume", "news_deriv_volume",
        # CFTC Commitment of Traders (institutional futures positioning)
        "cot_commercial_net", "cot_leveraged_net", "cot_asset_mgr_net",
        "cot_spec_ratio", "cot_commercial_change", "cot_leveraged_change",
        # v3.0: DeepSeek narrative scoring (continuous LLM sentiment)
        "ds_sentiment", "ds_confidence", "ds_bull_factors", "ds_bear_factors",
        "ds_impact", "ds_sentiment_momentum", "ds_conviction", "ds_bull_bear_ratio",
        # v3.0: Enhanced options — Risk-Neutral Density features
        "rnd_skewness", "rnd_kurtosis", "iv_smile_curvature",
        "put_skew_25d", "call_skew_25d", "butterfly_spread",
        "risk_reversal_25d", "vol_of_vol", "gamma_imbalance",
        "oi_put_wall", "oi_call_wall",
        # v3.1: Candlestick composite scores (individual patterns too sparse for ML)
        "cdl_bullish_score", "cdl_bearish_score",
        "cdl_net_signal", "cdl_indecision",
        # v3.2: NAV premium/discount (SPY vs S&P 500 index)
        "nav_premium_pct", "nav_premium_zscore", "nav_premium_ma5",
        "nav_premium_momentum", "nav_premium_mean_rev", "nav_premium_extreme",
        "spy_es_basis_pct", "spy_es_basis_zscore",
        "nav_premium_vol", "nav_premium_skew",
        "nav_premium_regime", "nav_creation_pressure",
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


def get_target(df: pd.DataFrame, threshold: float = 0.004,
               adaptive: bool = True) -> pd.Series:
    """Compute next-day direction target: UP(1), DOWN(-1), NEUTRAL(0).

    Args:
        df: DataFrame with 'close' column and optionally 'vix'
        threshold: Base ±0.4% daily return for neutral zone
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
