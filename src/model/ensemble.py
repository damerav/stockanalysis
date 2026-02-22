"""Stacking Ensemble — XGBoost + BiLSTM + LightGBM with logistic meta-learner.

Two-layer architecture:
  Layer 1: Three diverse base learners produce probability vectors
  Layer 2: Logistic regression meta-learner combines them
"""

import logging
import numpy as np
from typing import Optional
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """Two-layer stacking ensemble for 3-class direction prediction."""

    def __init__(self, config: dict = None):
        cfg = (config or {}).get("ensemble", {})
        self.seq_len = cfg.get("bilstm_seq_len", 20)
        self.xgb_model = None
        self.lgbm_model = None
        self.bilstm_model = None
        self.meta_learner: Optional[LogisticRegression] = None
        self.num_classes = 3

    def fit(self, X: np.ndarray, y: np.ndarray,
            use_gpu: bool = True, embargo: int = 5) -> dict:
        """Train all base learners and meta-learner.

        Uses walk-forward: train base learners on first 70%, generate
        meta-features on next 20% (with embargo), train meta-learner on those.
        """
        n = len(X)
        train_end = int(n * 0.60)
        meta_start = train_end + embargo
        meta_end = int(n * 0.85)

        X_base_train = X[:train_end]
        y_base_train = y[:train_end]
        X_meta = X[meta_start:meta_end]
        y_meta = y[meta_start:meta_end]

        if len(X_base_train) < 50 or len(X_meta) < 20:
            logger.warning("Insufficient data for ensemble training")
            return {"error": "insufficient data"}

        metrics = {}

        # --- Base learner 1: XGBoost ---
        try:
            import xgboost as xgb
            device = "cuda" if use_gpu else "cpu"
            try:
                test = xgb.XGBClassifier(tree_method="hist", device="cuda",
                                         n_estimators=1, verbosity=0)
                test.fit(X_base_train[:10], y_base_train[:10])
            except Exception:
                device = "cpu"

            self.xgb_model = xgb.XGBClassifier(
                objective="multi:softprob", num_class=3,
                tree_method="hist", device=device,
                max_depth=6, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8, verbosity=0,
            )
            self.xgb_model.fit(X_base_train, y_base_train)
            xgb_meta_probs = self.xgb_model.predict_proba(X_meta)
            metrics["xgb_meta_acc"] = float(np.mean(
                self.xgb_model.predict(X_meta) == y_meta))
            logger.info(f"XGBoost base: meta_acc={metrics['xgb_meta_acc']:.3f}")
        except Exception as e:
            logger.warning(f"XGBoost base failed: {e}")
            xgb_meta_probs = np.full((len(X_meta), 3), 1/3)

        # --- Base learner 2: LightGBM ---
        try:
            import lightgbm as lgb
            self.lgbm_model = lgb.LGBMClassifier(
                objective="multiclass", num_class=3,
                num_leaves=63, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
            )
            self.lgbm_model.fit(X_base_train, y_base_train)
            lgbm_meta_probs = self.lgbm_model.predict_proba(X_meta)
            metrics["lgbm_meta_acc"] = float(np.mean(
                self.lgbm_model.predict(X_meta) == y_meta))
            logger.info(f"LightGBM base: meta_acc={metrics['lgbm_meta_acc']:.3f}")
        except ImportError:
            logger.warning("lightgbm not installed, skipping LightGBM base learner")
            lgbm_meta_probs = np.full((len(X_meta), 3), 1/3)
        except Exception as e:
            logger.warning(f"LightGBM base failed: {e}")
            lgbm_meta_probs = np.full((len(X_meta), 3), 1/3)

        # --- Base learner 3: BiLSTM ---
        try:
            from src.model.bilstm_model import BiLSTMClassifier
            self.bilstm_model = BiLSTMClassifier(
                input_dim=X.shape[1], seq_len=self.seq_len,
                hidden_dim=128, epochs=30,
            )
            self.bilstm_model.fit(X_base_train, y_base_train)
            bilstm_meta_probs = self.bilstm_model.predict_proba(X_meta)
            # Align length (BiLSTM may produce fewer outputs due to seq windowing)
            if len(bilstm_meta_probs) < len(X_meta):
                pad = np.full((len(X_meta) - len(bilstm_meta_probs), 3), 1/3)
                bilstm_meta_probs = np.vstack([pad, bilstm_meta_probs])
            elif len(bilstm_meta_probs) > len(X_meta):
                bilstm_meta_probs = bilstm_meta_probs[-len(X_meta):]
            metrics["bilstm_meta_acc"] = float(np.mean(
                np.argmax(bilstm_meta_probs, axis=1) == y_meta))
            logger.info(f"BiLSTM base: meta_acc={metrics['bilstm_meta_acc']:.3f}")
        except Exception as e:
            logger.warning(f"BiLSTM base failed: {e}")
            bilstm_meta_probs = np.full((len(X_meta), 3), 1/3)

        # --- Meta-learner: Logistic Regression on stacked probabilities ---
        meta_features = np.hstack([xgb_meta_probs, lgbm_meta_probs, bilstm_meta_probs])
        self.meta_learner = LogisticRegression(
            max_iter=1000, solver="lbfgs",
        )
        self.meta_learner.fit(meta_features, y_meta)
        meta_preds = self.meta_learner.predict(meta_features)
        metrics["meta_acc"] = float(np.mean(meta_preds == y_meta))
        logger.info(f"Meta-learner: acc={metrics['meta_acc']:.3f}")

        # --- Evaluate on holdout ---
        holdout_start = meta_end + embargo
        if holdout_start < n:
            X_hold = X[holdout_start:]
            y_hold = y[holdout_start:]
            hold_probs = self.predict_proba(X_hold)
            hold_preds = np.argmax(hold_probs, axis=1)
            # Align
            min_len = min(len(hold_preds), len(y_hold))
            metrics["holdout_acc"] = float(np.mean(
                hold_preds[-min_len:] == y_hold[-min_len:]))
            metrics["holdout_size"] = min_len
            logger.info(f"Ensemble holdout: acc={metrics['holdout_acc']:.3f}")

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using the full ensemble."""
        if self.meta_learner is None:
            return np.full((len(X), 3), 1/3)

        # Get base learner probabilities
        xgb_p = self.xgb_model.predict_proba(X) if self.xgb_model else np.full((len(X), 3), 1/3)

        if self.lgbm_model:
            lgbm_p = self.lgbm_model.predict_proba(X)
        else:
            lgbm_p = np.full((len(X), 3), 1/3)

        if self.bilstm_model:
            bilstm_p = self.bilstm_model.predict_proba(X)
            # Align
            if len(bilstm_p) < len(X):
                pad = np.full((len(X) - len(bilstm_p), 3), 1/3)
                bilstm_p = np.vstack([pad, bilstm_p])
            elif len(bilstm_p) > len(X):
                bilstm_p = bilstm_p[-len(X):]
        else:
            bilstm_p = np.full((len(X), 3), 1/3)

        meta_features = np.hstack([xgb_p, lgbm_p, bilstm_p])
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict_proba(X), axis=1)
