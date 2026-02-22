"""Conformal Prediction — Uncertainty quantification for predictions.

Produces prediction sets (e.g., {UP} or {UP, NEUTRAL}) at a given
significance level. When the set contains multiple classes, the
prediction is flagged as LOW CONVICTION.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

DIRECTION_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


class ConformalPredictor:
    """Adaptive conformal prediction using nonconformity scores."""

    def __init__(self, significance: float = 0.10):
        """
        Args:
            significance: Error rate (0.10 = 90% coverage).
        """
        self.significance = significance
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None

    def calibrate(self, probs: np.ndarray, y_true: np.ndarray):
        """Calibrate using a held-out calibration set.

        Args:
            probs: (n, num_classes) predicted probabilities
            y_true: (n,) true class labels
        """
        # Nonconformity score = 1 - probability of true class
        scores = 1.0 - probs[np.arange(len(y_true)), y_true.astype(int)]
        self.calibration_scores = np.sort(scores)
        # Quantile at (1 - significance) * (1 + 1/n) level
        n = len(scores)
        q_level = min(1.0, (1 - self.significance) * (1 + 1 / n))
        self.quantile = float(np.quantile(self.calibration_scores, q_level))
        logger.info(f"Conformal calibrated: quantile={self.quantile:.4f}, "
                    f"n_cal={n}, significance={self.significance}")

    def predict_set(self, probs: np.ndarray) -> list[dict]:
        """Produce prediction sets for each sample.

        Args:
            probs: (n, num_classes) predicted probabilities

        Returns:
            List of dicts with 'prediction_set', 'point_prediction',
            'confidence', 'is_low_conviction'.
        """
        if self.quantile is None:
            # Not calibrated — return point predictions
            results = []
            for p in probs:
                pred = int(np.argmax(p))
                results.append({
                    "prediction_set": [DIRECTION_NAMES[pred]],
                    "point_prediction": DIRECTION_NAMES[pred],
                    "confidence": round(float(p[pred]) * 100, 1),
                    "is_low_conviction": False,
                    "set_size": 1,
                })
            return results

        results = []
        for p in probs:
            # Include class if 1 - p(class) <= quantile
            pred_set = []
            for c in range(len(p)):
                if 1.0 - p[c] <= self.quantile:
                    pred_set.append(c)

            if not pred_set:
                # Fallback: include highest probability class
                pred_set = [int(np.argmax(p))]

            point_pred = int(np.argmax(p))
            is_low = len(pred_set) > 1

            results.append({
                "prediction_set": [DIRECTION_NAMES[c] for c in sorted(pred_set)],
                "point_prediction": DIRECTION_NAMES[point_pred],
                "confidence": round(float(p[point_pred]) * 100, 1),
                "is_low_conviction": is_low,
                "set_size": len(pred_set),
            })

        return results

    def predict_single(self, probs: np.ndarray) -> dict:
        """Predict for a single sample (1D probability vector)."""
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        return self.predict_set(probs)[0]
