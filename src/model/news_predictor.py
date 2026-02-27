"""News-Based Price Movement Predictor.

Trains an XGBClassifier on TF-IDF vectors + sentiment scores to predict
short-term price movements at multiple horizons (15m, 60m, 4h).
Inspired by Finance-And-ML/US-Stock-Prediction-Using-ML-And-Spark.
"""

import os
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

# Price change buckets: strong_down, down, neutral, up, strong_up
LABELS = ["strong_down", "down", "neutral", "up", "strong_up"]
THRESHOLDS = [-0.01, -0.003, 0.003, 0.01]  # percentage boundaries


def _classify_change(pct_change: float) -> int:
    """Classify a percentage change into 5 buckets (0-4)."""
    if pct_change <= THRESHOLDS[0]:
        return 0  # strong_down
    elif pct_change <= THRESHOLDS[1]:
        return 1  # down
    elif pct_change <= THRESHOLDS[2]:
        return 2  # neutral
    elif pct_change <= THRESHOLDS[3]:
        return 3  # up
    else:
        return 4  # strong_up


class NewsPredictor:
    """XGBoost classifier for news-driven price movement prediction."""

    def __init__(self, horizon_minutes: int = 60):
        self.horizon = horizon_minutes
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.trained = False
        self.accuracy = 0.0
        self.feature_names = []

    def prepare_features(self, tfidf_matrix: np.ndarray,
                         sentiment_df: pd.DataFrame) -> np.ndarray:
        """Combine TF-IDF vectors with sentiment scores into feature matrix."""
        sent_cols = ["sentiment_compound", "sentiment_positive", "sentiment_negative"]
        available = [c for c in sent_cols if c in sentiment_df.columns]
        if available:
            sent_features = sentiment_df[available].values
            return np.hstack([tfidf_matrix, sent_features])
        return tfidf_matrix

    def create_targets(self, prices_df: pd.DataFrame,
                       dates: list[str]) -> np.ndarray:
        """Create classification targets from price changes.

        For daily data, uses next-day close change as proxy for the horizon.
        """
        if prices_df.empty:
            return np.array([])
        prices_df = prices_df.sort_values("date").copy()
        prices_df["pct_change"] = prices_df["close"].pct_change().shift(-1)
        date_to_change = dict(zip(prices_df["date"], prices_df["pct_change"]))
        targets = []
        for d in dates:
            change = date_to_change.get(d, 0)
            if pd.isna(change):
                change = 0
            targets.append(_classify_change(change))
        return np.array(targets)

    def train(self, X: np.ndarray, y: np.ndarray,
              test_size: float = 0.2) -> dict:
        """Train the model. Returns metrics dict."""
        if len(X) < 20:
            logger.warning(f"Too few samples ({len(X)}) for training")
            return {"accuracy": 0, "samples": len(X)}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
            if len(np.unique(y)) > 1 else None
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.trained = True

        logger.info(f"NewsPredictor trained: accuracy={self.accuracy:.3f}, "
                     f"samples={len(X)}, horizon={self.horizon}m")
        return {
            "accuracy": self.accuracy,
            "samples": len(X),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def predict(self, X: np.ndarray) -> list[dict]:
        """Predict price movement. Returns list of {label, probability} dicts."""
        if not self.trained:
            return [{"label": "neutral", "probability": 0.0}]
        proba = self.model.predict_proba(X)
        results = []
        for row in proba:
            idx = int(np.argmax(row))
            results.append({
                "label": LABELS[idx] if idx < len(LABELS) else "neutral",
                "probability": float(row[idx]),
                "all_probs": {LABELS[i]: float(row[i]) for i in range(len(row))
                              if i < len(LABELS)},
            })
        return results

    def save(self, path: str = "./models/news_predictor.pkl"):
        """Save trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "horizon": self.horizon,
                "accuracy": self.accuracy,
                "trained": self.trained,
                "saved_at": datetime.now().isoformat(),
            }, f)
        logger.info(f"NewsPredictor saved to {path}")

    def load(self, path: str = "./models/news_predictor.pkl"):
        """Load a previously trained model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.horizon = data.get("horizon", 60)
        self.accuracy = data.get("accuracy", 0)
        self.trained = data.get("trained", False)
        logger.info(f"NewsPredictor loaded from {path} (accuracy={self.accuracy:.3f})")
