"""6A. ES Strategy Indicators — ATR, Keltner Channel, EMA, VWAP, RSI, ROC, Regime."""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                    ema_period: int = 20, atr_period: int = 14,
                    multiplier: float = 1.5) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channel: EMA(20) ± 1.5 × ATR(14)."""
    mid = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper = mid + multiplier * atr_val
    lower = mid - multiplier * atr_val
    return upper, mid, lower


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """Session VWAP (resets daily — caller should pass single-session data)."""
    typical = (high + low + close) / 3
    cum_tp_vol = (typical * volume).cumsum()
    cum_vol = volume.cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def roc(close: pd.Series, period: int = 3) -> pd.Series:
    """Rate of Change."""
    return close.pct_change(period) * 100


class RegimeDetector:
    """ATR percentile-based volatility regime with 3-bar hysteresis."""

    def __init__(self, lookback: int = 10080, pct_low: int = 33, pct_high: int = 66):
        self.lookback = lookback
        self.pct_low = pct_low
        self.pct_high = pct_high
        self._current_regime = "Med"
        self._regime_bars = 0  # bars in current regime
        self.HYSTERESIS = 3

    def update(self, atr_value: float, atr_history: pd.Series) -> str:
        """Determine regime from ATR percentile with hysteresis.

        Returns: "Low", "Med", or "High"
        """
        if len(atr_history) < 20:
            return self._current_regime

        history = atr_history.tail(self.lookback).dropna()
        if len(history) == 0:
            return self._current_regime

        percentile = (history < atr_value).sum() / len(history) * 100

        # Determine raw regime
        if percentile < self.pct_low:
            raw = "Low"
        elif percentile > self.pct_high:
            raw = "High"
        else:
            raw = "Med"

        # Apply hysteresis: need 3 consecutive bars in new regime to switch
        if raw != self._current_regime:
            self._regime_bars += 1
            if self._regime_bars >= self.HYSTERESIS:
                self._current_regime = raw
                self._regime_bars = 0
                logger.info(f"Regime changed to {raw} (ATR pctile={percentile:.0f}%)")
        else:
            self._regime_bars = 0

        return self._current_regime

    @property
    def regime(self) -> str:
        return self._current_regime


def compute_bar_indicators(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """Compute all indicators for a bar DataFrame.

    Expects columns: timestamp, open, high, low, close, volume
    Returns DataFrame with added indicator columns.
    """
    df = df.copy()
    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)
    kc_upper, kc_mid, kc_lower = keltner_channel(df["high"], df["low"], df["close"])
    df["kc_upper"] = kc_upper
    df["kc_mid"] = kc_mid
    df["kc_lower"] = kc_lower
    df["ema_9"] = ema(df["close"], 9)
    df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])
    df["rsi_14"] = rsi(df["close"], 14)
    df["roc_3"] = roc(df["close"], 3)
    return df


class CuMLRegimeClassifier:
    """GAP 10: cuML LogisticRegression volatility regime classifier.

    Classes: Low / Mid / High based on ATR percentile.
    Falls back to sklearn if cuML is not available.
    """

    def __init__(self):
        self.model = None
        self._use_cuml = False

    def train(self, atr_percentiles: np.ndarray, labels: np.ndarray) -> dict:
        """Train regime classifier on ATR percentile features.

        Args:
            atr_percentiles: Array of ATR percentile values (0-1).
            labels: Array of regime labels (0=Low, 1=Med, 2=High).

        Returns:
            Training metrics dict.
        """
        X = atr_percentiles.reshape(-1, 1) if atr_percentiles.ndim == 1 else atr_percentiles

        try:
            from cuml.linear_model import LogisticRegression as cuLR
            self.model = cuLR(max_iter=1000)
            self._use_cuml = True
            logger.info("Using cuML LogisticRegression for regime classifier")
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000)
            self._use_cuml = False
            logger.info("cuML not available, using sklearn LogisticRegression")

        self.model.fit(X, labels)
        accuracy = float((self.model.predict(X) == labels).mean())
        logger.info(f"Regime classifier accuracy: {accuracy:.3f}")
        return {"accuracy": accuracy, "use_cuml": self._use_cuml, "samples": len(labels)}

    def predict(self, atr_percentile: float) -> str:
        """Predict regime from ATR percentile."""
        if self.model is None:
            # Fallback to simple thresholds
            if atr_percentile < 0.33:
                return "Low"
            elif atr_percentile > 0.66:
                return "High"
            return "Med"

        X = np.array([[atr_percentile]])
        pred = int(self.model.predict(X)[0])
        return {0: "Low", 1: "Med", 2: "High"}.get(pred, "Med")
