"""Purged Walk-Forward Cross-Validation for time-series models.

Implements embargo + purge to prevent information leakage through
autocorrelated features (e.g., 14-day RSI shares 13 days with next row).
Based on López de Prado's Advances in Financial Machine Learning.
"""

import logging
import numpy as np
from typing import Iterator

logger = logging.getLogger(__name__)


class PurgedWalkForwardCV:
    """Walk-forward CV with purge and embargo periods.

    Args:
        n_splits: Number of walk-forward splits
        embargo_days: Gap between train end and test start
        purge_days: Days to remove from train end (label overlap)
        min_train_size: Minimum training set size
        test_size: Fixed test size (None = auto)
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 5,
                 purge_days: int = 1, min_train_size: int = 60,
                 test_size: int = None):
        self.n_splits = n_splits
        self.embargo = embargo_days
        self.purge = purge_days
        self.min_train_size = min_train_size
        self.test_size = test_size

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Iterator[tuple]:
        """Generate purged walk-forward train/test splits.

        Yields:
            (train_indices, test_indices) tuples
        """
        n = len(X)
        if self.test_size:
            test_sz = self.test_size
        else:
            test_sz = max(21, n // (self.n_splits + 1))

        gap = self.embargo + self.purge
        splits_generated = 0

        for i in range(self.n_splits):
            test_end = n - i * test_sz
            test_start = test_end - test_sz
            if test_start < self.min_train_size + gap:
                break

            train_end = test_start - self.embargo
            train_start = max(0, train_end - self.purge - (train_end - self.purge))
            # Purge: remove last purge_days from training
            train_end_purged = train_end - self.purge

            if train_end_purged < self.min_train_size:
                break

            train_idx = np.arange(train_start, train_end_purged)
            test_idx = np.arange(test_start, min(test_end, n))

            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                splits_generated += 1
                yield train_idx, test_idx

        if splits_generated == 0:
            logger.warning("PurgedCV: no valid splits generated, falling back to simple split")
            split_point = int(n * 0.7)
            train_idx = np.arange(0, split_point - self.purge)
            test_idx = np.arange(split_point + self.embargo, n)
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


def purged_walk_forward_score(model_cls, X: np.ndarray, y: np.ndarray,
                              n_splits: int = 5, embargo: int = 5,
                              **model_kwargs) -> dict:
    """Run purged walk-forward CV and return aggregated metrics.

    Args:
        model_cls: Model class with fit/predict/predict_proba
        X: Feature matrix
        y: Target array
        n_splits: Number of CV splits
        embargo: Embargo days
        **model_kwargs: Passed to model constructor

    Returns:
        Dict with mean/std accuracy, per-fold results
    """
    cv = PurgedWalkForwardCV(n_splits=n_splits, embargo_days=embargo)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(np.mean(preds == y_test))

        fold_results.append({
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "accuracy": acc,
        })
        logger.info(f"Fold {fold_idx}: train={len(train_idx)}, "
                    f"test={len(test_idx)}, acc={acc:.3f}")

    accuracies = [f["accuracy"] for f in fold_results]
    return {
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0,
        "std_accuracy": float(np.std(accuracies)) if accuracies else 0,
        "n_folds": len(fold_results),
        "folds": fold_results,
    }
