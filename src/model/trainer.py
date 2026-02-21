"""3C/3D. XGBoost SPY Direction Predictor — Train, predict, track accuracy."""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Direction labels
DIRECTION_MAP = {
    1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"
}
CONFIDENCE_MAP = {
    (0.7, 1.0): "STRONG",
    (0.5, 0.7): "",
    (0.0, 0.5): "WEAK"
}


def _label_to_class(label: int) -> int:
    """Map direction label (-1,0,1) to class index (0,1,2)."""
    return {-1: 0, 0: 1, 1: 2}[label]


def _class_to_label(cls: int) -> int:
    """Map class index (0,1,2) to direction label (-1,0,1)."""
    return {0: -1, 1: 0, 2: 1}[cls]


class SPYPredictor:
    """XGBoost-based SPY next-day direction predictor."""

    def __init__(self, config: dict = None):
        config = config or {}
        xgb_cfg = config.get("xgboost", {})
        self.lookback_days = xgb_cfg.get("lookback_days", 252)
        self.max_depth = xgb_cfg.get("max_depth", 6)
        self.learning_rate = xgb_cfg.get("learning_rate", 0.05)
        self.n_estimators = xgb_cfg.get("n_estimators", 500)
        self.neutral_threshold = xgb_cfg.get("neutral_threshold", 0.003)
        self.model = None
        self.feature_importances = None
        self.model_dir = "./models"
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, features_df: pd.DataFrame, target: pd.Series,
              use_gpu: bool = True) -> dict:
        """Train XGBoost with walk-forward time-series split.

        Args:
            features_df: Feature matrix (rows=dates, cols=features)
            target: Direction labels (-1, 0, 1)
            use_gpu: Use GPU acceleration if available

        Returns:
            Dict with training metrics
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("xgboost not installed. Run: pip install xgboost")
            return {"error": "xgboost not installed"}

        # Drop rows with NaN target (last row has no next-day return)
        valid_mask = target.notna()
        X = features_df[valid_mask].values
        y = target[valid_mask].values

        # Replace NaN features with 0 (XGBoost handles missing values natively)
        X = np.nan_to_num(X, nan=0.0)

        # Map labels to classes
        y_class = np.array([_label_to_class(int(v)) for v in y])

        if len(X) < 50:
            logger.warning(f"Only {len(X)} valid samples, need at least 50")
            return {"error": "insufficient data"}

        # Walk-forward split: 80% train, 20% validation (no shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_class[:split_idx], y_class[split_idx:]

        logger.info(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")

        # Determine device
        tree_method = "hist"
        device = "cpu"
        if use_gpu:
            try:
                # Try GPU
                test_model = xgb.XGBClassifier(tree_method="hist", device="cuda",
                                                n_estimators=1, verbosity=0)
                test_model.fit(X_train[:10], y_train[:10])
                tree_method = "hist"
                device = "cuda"
                logger.info("Using GPU (CUDA) for training")
            except Exception:
                logger.info("GPU not available, using CPU")

        # Train
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": tree_method,
            "device": device,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "mlogloss",
            "verbosity": 1,
            "early_stopping_rounds": 50,
        }

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Feature importances
        self.feature_importances = dict(zip(
            range(X.shape[1]),
            self.model.feature_importances_
        ))

        # Validation metrics
        val_pred = self.model.predict(X_val)
        accuracy = np.mean(val_pred == y_val)
        logger.info(f"Validation accuracy: {accuracy:.3f}")

        # Save model
        date_str = datetime.now().strftime("%Y%m%d")
        model_path = os.path.join(self.model_dir, f"xgb_spy_{date_str}.json")
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        return {
            "accuracy": float(accuracy),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "model_path": model_path,
            "best_iteration": self.model.best_iteration if hasattr(self.model, "best_iteration") else None,
        }

    def predict(self, features: np.ndarray) -> dict:
        """Generate prediction for a single feature vector.

        Returns:
            Dict with direction, confidence, probabilities, scale label
        """
        if self.model is None:
            logger.warning("No model loaded, returning neutral prediction")
            return self._neutral_prediction()

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Handle NaN features
        features = np.nan_to_num(features, nan=0.0)

        probs = self.model.predict_proba(features)[0]  # [p_down, p_neutral, p_up]
        pred_class = int(np.argmax(probs))
        pred_label = _class_to_label(pred_class)
        confidence = float(probs[pred_class]) * 100

        # Map to 5-level scale
        direction = DIRECTION_MAP[pred_label]
        strength = ""
        for (lo, hi), label in CONFIDENCE_MAP.items():
            if lo <= probs[pred_class] <= hi:
                strength = label
                break

        scale_label = f"{strength}_{direction}".strip("_") if strength else direction

        return {
            "direction": direction,
            "scale_label": scale_label,
            "confidence": round(confidence, 1),
            "probabilities": {
                "down": round(float(probs[0]) * 100, 1),
                "neutral": round(float(probs[1]) * 100, 1),
                "up": round(float(probs[2]) * 100, 1),
            },
            "predicted_class": pred_label,
        }

    def load_latest_model(self) -> bool:
        """Load the most recent saved model."""
        try:
            import xgboost as xgb
        except ImportError:
            return False

        if not os.path.exists(self.model_dir):
            return False

        models = sorted([
            f for f in os.listdir(self.model_dir)
            if f.startswith("xgb_spy_") and f.endswith(".json")
        ])
        if not models:
            logger.info("No saved models found")
            return False

        model_path = os.path.join(self.model_dir, models[-1])
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        logger.info(f"Loaded model: {model_path}")
        return True

    def _neutral_prediction(self) -> dict:
        return {
            "direction": "NEUTRAL",
            "scale_label": "NEUTRAL",
            "confidence": 0.0,
            "probabilities": {"down": 33.3, "neutral": 33.3, "up": 33.3},
            "predicted_class": 0,
        }

    def get_feature_importance_report(self, feature_names: list[str]) -> list[dict]:
        """Get sorted feature importances."""
        if not self.feature_importances:
            return []
        report = []
        for idx, importance in sorted(self.feature_importances.items(),
                                       key=lambda x: x[1], reverse=True):
            if idx < len(feature_names):
                report.append({
                    "feature": feature_names[idx],
                    "importance": round(float(importance), 4),
                })
        return report


def evaluate_past_prediction(conn, date: str) -> Optional[dict]:
    """Compare yesterday's prediction to actual outcome.

    Returns dict with evaluation results or None if no prediction exists.
    """
    row = conn.execute(
        "SELECT direction, confidence FROM predictions WHERE date = ?", (date,)
    ).fetchone()
    if not row:
        return None

    predicted = row[0]

    # Get actual return
    prices = conn.execute(
        "SELECT close FROM prices WHERE date >= ? ORDER BY date LIMIT 2", (date,)
    ).fetchall()
    if len(prices) < 2:
        return None

    actual_return = (prices[1][0] - prices[0][0]) / prices[0][0]
    if actual_return > 0.003:
        actual = "BULLISH"
    elif actual_return < -0.003:
        actual = "BEARISH"
    else:
        actual = "NEUTRAL"

    # Simplified: bullish/bearish match
    correct = 1 if (
        ("BULLISH" in predicted and actual == "BULLISH") or
        ("BEARISH" in predicted and actual == "BEARISH") or
        (predicted == "NEUTRAL" and actual == "NEUTRAL")
    ) else 0

    # Update cumulative accuracy
    perf_rows = conn.execute("SELECT COUNT(*), SUM(correct) FROM performance").fetchone()
    total = (perf_rows[0] or 0) + 1
    correct_total = (perf_rows[1] or 0) + correct
    cum_accuracy = correct_total / total

    conn.execute(
        """INSERT OR REPLACE INTO performance (date, predicted, actual, correct, cumulative_accuracy)
           VALUES (?, ?, ?, ?, ?)""",
        (date, predicted, actual, correct, cum_accuracy)
    )
    conn.commit()

    return {
        "date": date, "predicted": predicted, "actual": actual,
        "correct": bool(correct), "cumulative_accuracy": round(cum_accuracy, 3),
    }
