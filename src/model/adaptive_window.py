"""Adaptive Training Window Selection.

Tests multiple window lengths (63, 126, 252, 504 days) and selects
the one with best recent walk-forward accuracy. Shorter windows adapt
faster in choppy markets; longer windows capture persistent trends.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

CANDIDATE_WINDOWS = [63, 126, 252, 504]
VALIDATION_DAYS = 21  # Validate on most recent 21 trading days


def select_optimal_window(X: np.ndarray, y: np.ndarray,
                          model_cls=None, model_kwargs: dict = None,
                          candidate_windows: list[int] = None,
                          val_days: int = VALIDATION_DAYS) -> dict:
    """Test multiple training windows and select the best.

    Args:
        X: Full feature matrix (oldest to newest)
        y: Full target array
        model_cls: Model class with fit/predict (default: XGBClassifier)
        model_kwargs: Model constructor kwargs
        candidate_windows: Window sizes to test
        val_days: Number of recent days for validation

    Returns:
        Dict with optimal_window, scores per window, selected model
    """
    if model_cls is None:
        try:
            from xgboost import XGBClassifier
            model_cls = XGBClassifier
        except ImportError:
            logger.error("xgboost not available for adaptive window")
            return {"optimal_window": 252, "scores": {}, "error": "no xgboost"}

    if model_kwargs is None:
        model_kwargs = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "verbosity": 0,
            "objective": "multi:softprob",
            "num_class": 3,
        }

    windows = candidate_windows or CANDIDATE_WINDOWS
    n = len(X)
    embargo = 5

    if n < val_days + min(windows) + embargo:
        logger.warning(f"Insufficient data ({n} rows) for adaptive window selection")
        return {"optimal_window": min(n - val_days - embargo, 252), "scores": {}}

    # Validation set: last val_days
    X_val = X[n - val_days:]
    y_val = y[n - val_days:]

    scores = {}
    for w in windows:
        train_end = n - val_days - embargo
        train_start = max(0, train_end - w)
        if train_end - train_start < 30:
            continue

        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]

        try:
            model = model_cls(**model_kwargs)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            acc = float(np.mean(preds == y_val))
            scores[w] = acc
            logger.info(f"Window {w}d: train={len(X_train)}, val_acc={acc:.3f}")
        except Exception as e:
            logger.warning(f"Window {w}d failed: {e}")
            scores[w] = 0.0

    if not scores:
        return {"optimal_window": 252, "scores": scores}

    optimal = max(scores, key=scores.get)
    logger.info(f"Optimal window: {optimal}d (acc={scores[optimal]:.3f})")

    return {
        "optimal_window": optimal,
        "scores": scores,
        "best_accuracy": scores[optimal],
    }
