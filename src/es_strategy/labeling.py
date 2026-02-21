"""GAP 6/11: Triple-Barrier Labeling for Entry and Exit Models.

Entry labels: TP1 hit before stop within 60 bars → 1, else → 0.
Exit labels: Future adverse move ≥ 0.25×ATR within 5 bars → 1 (reversal), else → 0.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def label_entries_triple_barrier(df: pd.DataFrame, atr_col: str = "atr_14",
                                  tp_mult: float = 1.0, stop_mult: float = 0.20,
                                  credit_C: float = 10.0,
                                  timeout_bars: int = 60) -> pd.Series:
    """Triple-barrier entry labeling.

    For each bar, look forward up to timeout_bars:
      - If price hits TP1 (entry + tp_mult × ATR) before emergency stop → label = 1
      - If price hits stop (entry - stop_mult × credit) first → label = 0
      - If timeout reached → label = 0

    Args:
        df: DataFrame with 'close', 'high', 'low', and atr_col columns.
        tp_mult: TP1 target as multiple of ATR.
        stop_mult: Emergency stop as fraction of credit.
        credit_C: Credit width for stop calculation.
        timeout_bars: Maximum bars to look forward.

    Returns:
        Series of labels (1 = success, 0 = failure/timeout).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df[atr_col].values
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - 1):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        entry = close[i]
        tp_target = entry + tp_mult * atr[i]
        stop_level = entry - stop_mult * credit_C
        end = min(i + timeout_bars, n)

        for j in range(i + 1, end):
            # Check TP hit (long direction)
            if high[j] >= tp_target:
                labels[i] = 1
                break
            # Check stop hit
            if low[j] <= stop_level:
                labels[i] = 0
                break
        # If loop completes without break → timeout → label stays 0

    return pd.Series(labels, index=df.index)


def label_exits_reversal(df: pd.DataFrame, atr_col: str = "atr_14",
                          reversal_mult: float = 0.25,
                          horizon_bars: int = 5) -> pd.Series:
    """Exit labeling: adverse move ≥ reversal_mult × ATR within horizon_bars.

    For each bar, check if the maximum adverse move in the next horizon_bars
    exceeds the threshold. If so, label = 1 (reversal imminent).

    Args:
        df: DataFrame with 'close', 'low', and atr_col columns.
        reversal_mult: Threshold as multiple of ATR.
        horizon_bars: Number of bars to look forward.

    Returns:
        Series of labels (1 = reversal, 0 = continuation).
    """
    close = df["close"].values
    low = df["low"].values
    atr = df[atr_col].values
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - horizon_bars):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        threshold = reversal_mult * atr[i]
        # Max adverse move (assuming long position)
        future_low = np.min(low[i + 1:i + 1 + horizon_bars])
        adverse = close[i] - future_low

        if adverse >= threshold:
            labels[i] = 1

    return pd.Series(labels, index=df.index)


def generate_training_dataset(df: pd.DataFrame, credit_C: float = 10.0) -> dict:
    """Generate labeled datasets for both entry and exit models.

    Args:
        df: DataFrame with OHLCV + indicators (must have atr_14).

    Returns:
        Dict with 'entry_labels' and 'exit_labels' Series.
    """
    entry_labels = label_entries_triple_barrier(df, credit_C=credit_C)
    exit_labels = label_exits_reversal(df)

    logger.info(f"Entry labels: {entry_labels.sum()} positive / {len(entry_labels)} total "
                f"({entry_labels.mean():.1%} hit rate)")
    logger.info(f"Exit labels: {exit_labels.sum()} reversals / {len(exit_labels)} total "
                f"({exit_labels.mean():.1%} reversal rate)")

    return {
        "entry_labels": entry_labels,
        "exit_labels": exit_labels,
    }
