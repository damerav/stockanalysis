"""3C/3D. XGBoost SPY Direction Predictor — Train, predict, track accuracy.

P1 enhancements: isotonic calibration, SHAP explanations, performance gating,
stratified accuracy tracking.
P2 enhancements: purged walk-forward CV, adaptive training window, model registry,
stacking ensemble, conformal prediction.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional

from src.es_strategy.labeling import label_entries_triple_barrier, label_exits_reversal

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
        self.config = config
        self.lookback_days = xgb_cfg.get("lookback_days", 252)
        self.max_depth = xgb_cfg.get("max_depth", 6)
        self.learning_rate = xgb_cfg.get("learning_rate", 0.05)
        self.n_estimators = xgb_cfg.get("n_estimators", 500)
        self.neutral_threshold = xgb_cfg.get("neutral_threshold", 0.003)
        self.model = None
        self.calibrator = None  # P1: isotonic calibration
        self.feature_importances = None
        self.model_dir = "./models"
        self.prior_accuracy = None  # P1: performance gating
        # P2: Ensemble, conformal, registry
        self.ensemble = None
        self.conformal = None
        self.registry = None
        self.use_ensemble = config.get("ensemble", {}).get("enabled", False)
        self.use_conformal = config.get("conformal", {}).get("enabled", True)
        self.trained_feature_names = None  # loaded from _meta.json
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, features_df: pd.DataFrame, target: pd.Series,
              use_gpu: bool = True, feature_names: list[str] = None,
              force_save: bool = False) -> dict:
        """Train XGBoost with P2 enhancements: adaptive window, purged CV,
        ensemble, conformal prediction, model registry.

        Args:
            features_df: Feature matrix (rows=dates, cols=features)
            target: Direction labels (-1, 0, 1)
            use_gpu: Use GPU acceleration if available
            feature_names: Feature column names for registry
            force_save: Skip performance gating and always save model

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
        features_valid = features_df[valid_mask].apply(pd.to_numeric, errors="coerce")

        # --- Feature quality filtering: drop columns that are mostly NaN ---
        nan_rates = features_valid.isna().mean()
        good_cols = nan_rates[nan_rates < 0.5].index.tolist()
        dropped = len(features_valid.columns) - len(good_cols)
        if dropped > 0:
            logger.info(f"Dropped {dropped} features with >50% NaN "
                        f"({len(good_cols)} remaining)")
            features_valid = features_valid[good_cols]
            if feature_names:
                feature_names = [c for c in feature_names if c in good_cols]

        X = features_valid.values
        y = target[valid_mask].values

        # Replace NaN features with 0 (XGBoost handles missing values natively)
        X = np.nan_to_num(X, nan=0.0)

        # Map labels to classes
        y_class = np.array([_label_to_class(int(v)) for v in y])

        if len(X) < 50:
            logger.warning(f"Only {len(X)} valid samples, need at least 50")
            return {"error": "insufficient data"}

        # --- P2: Adaptive training window selection ---
        optimal_window = len(X)
        window_scores = {}
        try:
            from src.model.adaptive_window import select_optimal_window
            window_result = select_optimal_window(X, y_class)
            optimal_window = min(window_result["optimal_window"], len(X))
            window_scores = window_result.get("scores", {})
            logger.info(f"Adaptive window selected: {optimal_window}d")
            # Trim to optimal window
            if optimal_window < len(X):
                X = X[-optimal_window:]
                y_class = y_class[-optimal_window:]
        except Exception as e:
            logger.warning(f"Adaptive window selection failed (using full data): {e}")

        # Walk-forward split: 70/20/10 with 5-day embargo (GAP 12)
        train_end = int(len(X) * 0.70)
        val_end = int(len(X) * 0.90)
        embargo = 5  # 5-day gap to prevent look-ahead bias

        X_train = X[:train_end]
        y_train = y_class[:train_end]
        X_val = X[train_end + embargo:val_end]
        y_val = y_class[train_end + embargo:val_end]
        X_test = X[val_end + embargo:]
        y_test = y_class[val_end + embargo:]

        # Clip features at ±5σ (GAP 12)
        train_mean = np.nanmean(X_train, axis=0)
        train_std = np.nanstd(X_train, axis=0)
        train_std[train_std == 0] = 1.0
        clip_lo = train_mean - 5 * train_std
        clip_hi = train_mean + 5 * train_std
        X_train = np.clip(X_train, clip_lo, clip_hi)
        X_val = np.clip(X_val, clip_lo, clip_hi)
        X_test = np.clip(X_test, clip_lo, clip_hi)

        logger.info(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")

        # Determine device
        tree_method = "hist"
        device = "cpu"
        if use_gpu:
            try:
                test_model = xgb.XGBClassifier(tree_method="hist", device="cuda",
                                                n_estimators=1, verbosity=0)
                test_model.fit(X_train[:10], y_train[:10])
                tree_method = "hist"
                device = "cuda"
                logger.info("Using GPU (CUDA) for training")
            except Exception:
                logger.info("GPU not available, using CPU")

        # Train XGBoost — tune for dataset size
        n_samples = len(X_train)
        # Adaptive hyperparameters based on dataset size
        if n_samples < 300:
            # Small dataset: shallow trees, strong regularization, fewer estimators
            eff_depth = min(self.max_depth, 3)
            eff_lr = max(self.learning_rate, 0.03)
            eff_n = min(self.n_estimators, 300)
            eff_subsample = 0.7
            eff_colsample = 0.6
            eff_min_child = 5
            eff_gamma = 1.0
            eff_reg_alpha = 0.5
            eff_reg_lambda = 2.0
        elif n_samples < 1000:
            eff_depth = min(self.max_depth, 4)
            eff_lr = self.learning_rate
            eff_n = self.n_estimators
            eff_subsample = 0.8
            eff_colsample = 0.7
            eff_min_child = 3
            eff_gamma = 0.5
            eff_reg_alpha = 0.1
            eff_reg_lambda = 1.5
        else:
            eff_depth = self.max_depth
            eff_lr = self.learning_rate
            eff_n = self.n_estimators
            eff_subsample = 0.8
            eff_colsample = 0.8
            eff_min_child = 1
            eff_gamma = 0.0
            eff_reg_alpha = 0.0
            eff_reg_lambda = 1.0

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": tree_method,
            "device": device,
            "max_depth": eff_depth,
            "learning_rate": eff_lr,
            "n_estimators": eff_n,
            "subsample": eff_subsample,
            "colsample_bytree": eff_colsample,
            "min_child_weight": eff_min_child,
            "gamma": eff_gamma,
            "reg_alpha": eff_reg_alpha,
            "reg_lambda": eff_reg_lambda,
            "eval_metric": "mlogloss",
            "verbosity": 1,
            "early_stopping_rounds": 30,
        }
        logger.info(f"XGBoost params: depth={eff_depth}, lr={eff_lr}, "
                     f"n={eff_n}, samples={n_samples}")

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

        # Log low-gain features (GAP 12)
        total_imp = sum(self.model.feature_importances_)
        low_gain_features = []
        for idx, imp in enumerate(self.model.feature_importances_):
            pct = (imp / total_imp * 100) if total_imp > 0 else 0
            if pct < 1.0:
                low_gain_features.append(idx)
        if low_gain_features:
            logger.info(f"Low-gain features (<1%): {len(low_gain_features)} of {X.shape[1]}")

        # Validation metrics
        val_pred = self.model.predict(X_val)
        accuracy = np.mean(val_pred == y_val)
        logger.info(f"Validation accuracy: {accuracy:.3f}")

        # Test set metrics
        test_accuracy = None
        if len(X_test) > 0:
            test_pred = self.model.predict(X_test)
            test_accuracy = float(np.mean(test_pred == y_test))
            logger.info(f"Test accuracy: {test_accuracy:.3f}")

        # --- P2: Purged walk-forward CV score ---
        cv_result = {}
        try:
            from src.model.purged_cv import purged_walk_forward_score
            cv_result = purged_walk_forward_score(
                xgb.XGBClassifier, X, y_class, n_splits=5, embargo=5,
                objective="multi:softprob", num_class=3,
                tree_method=tree_method, device=device,
                max_depth=self.max_depth, learning_rate=self.learning_rate,
                n_estimators=200, verbosity=0,
            )
            logger.info(f"Purged CV: mean_acc={cv_result['mean_accuracy']:.3f} "
                        f"± {cv_result['std_accuracy']:.3f}")
        except Exception as e:
            logger.warning(f"Purged CV failed (non-fatal): {e}")

        # --- P1: Isotonic probability calibration ---
        try:
            from sklearn.calibration import CalibratedClassifierCV
            self.calibrator = CalibratedClassifierCV(
                self.model, method="isotonic", cv="prefit"
            )
            self.calibrator.fit(X_val, y_val)
            logger.info("Isotonic calibration fitted on validation set")
        except Exception as e:
            logger.warning(f"Isotonic calibration failed (non-fatal): {e}")
            self.calibrator = None

        # --- P2: Conformal prediction calibration ---
        if self.use_conformal and len(X_test) > 10:
            try:
                from src.model.conformal import ConformalPredictor
                cal_probs = self.model.predict_proba(X_test)
                self.conformal = ConformalPredictor(
                    significance=self.config.get("conformal", {}).get("significance", 0.10)
                )
                self.conformal.calibrate(cal_probs, y_test)
            except Exception as e:
                logger.warning(f"Conformal calibration failed (non-fatal): {e}")

        # --- P2: Stacking ensemble training ---
        ensemble_metrics = {}
        if self.use_ensemble:
            try:
                from src.model.ensemble import StackingEnsemble
                self.ensemble = StackingEnsemble(self.config)
                ensemble_metrics = self.ensemble.fit(X, y_class, use_gpu=use_gpu)
                logger.info(f"Ensemble trained: {ensemble_metrics}")
            except Exception as e:
                logger.warning(f"Ensemble training failed (non-fatal): {e}")
                self.ensemble = None

        # --- P1: Performance gating ---
        gated = False
        gate_reason = ""
        if not force_save:
            if accuracy < 0.52:
                gated = True
                gate_reason = f"val accuracy {accuracy:.3f} < 0.52 threshold"
            elif self.prior_accuracy is not None and accuracy < self.prior_accuracy - 0.02:
                gated = True
                gate_reason = (f"val accuracy {accuracy:.3f} degraded > 2% "
                               f"vs prior {self.prior_accuracy:.3f}")

        model_path = ""
        if gated:
            logger.warning(f"MODEL GATED — not saving: {gate_reason}")
            self.load_latest_model()
        else:
            date_str = datetime.now().strftime("%Y%m%d")
            model_path = os.path.join(self.model_dir, f"xgb_spy_{date_str}.json")
            self.model.save_model(model_path)
            self.prior_accuracy = float(accuracy)
            # Save feature names alongside model for inference alignment
            feat_names = (feature_names if feature_names
                          else list(features_valid.columns) if hasattr(features_valid, 'columns')
                          else [f"f{i}" for i in range(X.shape[1])])
            meta_path = model_path.replace(".json", "_meta.json")
            try:
                import json as _json
                meta_data = {"feature_names": feat_names}
                # Save conformal calibrator state for persistence across loads
                if self.conformal is not None and self.conformal.quantile is not None:
                    meta_data["conformal_quantile"] = self.conformal.quantile
                    meta_data["conformal_significance"] = self.conformal.significance
                with open(meta_path, "w") as mf:
                    _json.dump(meta_data, mf)
                logger.info(f"Feature metadata saved to {meta_path}")
            except Exception as e:
                logger.warning(f"Failed to save feature metadata: {e}")
            # Update in-memory feature names so callers don't need to reload
            self.trained_feature_names = feat_names
            # Save conformal calibrator if fitted
            if self.conformal is not None:
                conformal_path = model_path.replace(".json", "_conformal.pkl")
                try:
                    import pickle
                    with open(conformal_path, "wb") as cf:
                        pickle.dump(self.conformal, cf)
                    logger.info(f"Conformal calibrator saved to {conformal_path}")
                except Exception as e:
                    logger.warning(f"Failed to save conformal calibrator: {e}")
            logger.info(f"Model saved to {model_path}")

        metrics = {
            "accuracy": float(accuracy),
            "test_accuracy": test_accuracy,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "embargo_days": embargo,
            "low_gain_features": len(low_gain_features),
            "model_path": model_path if not gated else "gated",
            "gated": gated,
            "gate_reason": gate_reason,
            "calibrated": self.calibrator is not None,
            "best_iteration": self.model.best_iteration if hasattr(self.model, "best_iteration") else None,
            # P2 additions
            "adaptive_window": optimal_window,
            "window_scores": window_scores,
            "purged_cv": cv_result,
            "ensemble": ensemble_metrics,
            "conformal_calibrated": self.conformal is not None,
        }

        # --- P2: Register model in registry ---
        if not gated and model_path:
            try:
                from src.model.registry import ModelRegistry
                self.registry = ModelRegistry(self.config)
                self.registry.register(
                    metrics, model_path,
                    feature_names=feature_names or
                        [f"f{i}" for i in range(X.shape[1])],
                )
            except Exception as e:
                logger.warning(f"Model registry failed (non-fatal): {e}")

        return metrics

    def predict(self, features: np.ndarray,
                feature_names: list[str] = None) -> dict:
        """Generate prediction for a single feature vector.

        Args:
            features: Feature array (1D or 2D)
            feature_names: Optional list of feature names for SHAP explanations

        Returns:
            Dict with direction, confidence, probabilities, scale label,
            conformal prediction set, and optionally shap_drivers.
        """
        if self.model is None:
            logger.warning("No model loaded, returning neutral prediction")
            return self._neutral_prediction()

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Handle NaN features
        features = np.nan_to_num(features, nan=0.0)

        # --- P2: Use ensemble if available ---
        if self.ensemble is not None and self.use_ensemble:
            try:
                probs = self.ensemble.predict_proba(features)[0]
            except Exception:
                probs = self._get_base_probs(features)
        else:
            probs = self._get_base_probs(features)

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

        result = {
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

        # --- P2: Conformal prediction set ---
        if self.conformal is not None:
            try:
                conf_result = self.conformal.predict_single(probs)
                result["prediction_set"] = conf_result["prediction_set"]
                result["is_low_conviction"] = conf_result["is_low_conviction"]
                result["set_size"] = conf_result["set_size"]
            except Exception as e:
                logger.debug(f"Conformal prediction skipped: {e}")

        # --- P1: SHAP explanations (top 5 feature drivers) ---
        if feature_names is not None and self.model is not None:
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(features)
                if isinstance(shap_values, list):
                    sv = shap_values[pred_class][0]
                else:
                    sv = shap_values[0, :, pred_class] if shap_values.ndim == 3 else shap_values[0]
                top_idx = np.argsort(np.abs(sv))[-5:][::-1]
                drivers = []
                for idx in top_idx:
                    name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
                    drivers.append({
                        "feature": name,
                        "shap_value": round(float(sv[idx]), 4),
                        "feature_value": round(float(features[0, idx]), 4),
                    })
                result["shap_drivers"] = drivers
            except Exception as e:
                logger.debug(f"SHAP explanation skipped: {e}")

        return result

    def _get_base_probs(self, features: np.ndarray) -> np.ndarray:
        """Get probabilities from calibrated or raw XGBoost model."""
        if self.calibrator is not None:
            try:
                return self.calibrator.predict_proba(features)[0]
            except Exception:
                pass
        return self.model.predict_proba(features)[0]

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
            and not f.endswith("_meta.json")
        ])
        if not models:
            logger.info("No saved models found")
            return False

        model_path = os.path.join(self.model_dir, models[-1])
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        logger.info(f"Loaded model: {model_path}")

        # Load feature metadata if available
        self.trained_feature_names = None
        self.conformal = None
        meta_path = model_path.replace(".json", "_meta.json")
        if os.path.exists(meta_path):
            try:
                import json as _json
                with open(meta_path) as mf:
                    meta = _json.load(mf)
                self.trained_feature_names = meta.get("feature_names")
                logger.info(f"Loaded feature metadata: {len(self.trained_feature_names)} features")
                # Restore conformal calibrator if saved
                if meta.get("conformal_quantile") is not None:
                    from src.model.conformal import ConformalPredictor
                    self.conformal = ConformalPredictor(
                        significance=meta.get("conformal_significance", 0.10)
                    )
                    self.conformal.quantile = meta["conformal_quantile"]
                    logger.info(f"Conformal calibrator restored (quantile={self.conformal.quantile:.4f})")
            except Exception as e:
                logger.warning(f"Failed to load feature metadata: {e}")

        # Load conformal calibrator if available
        self.conformal = None
        conformal_path = model_path.replace(".json", "_conformal.pkl")
        if os.path.exists(conformal_path):
            try:
                import pickle
                with open(conformal_path, "rb") as cf:
                    self.conformal = pickle.load(cf)
                logger.info("Conformal calibrator loaded")
            except Exception as e:
                logger.warning(f"Failed to load conformal calibrator: {e}")

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

    def train_es_entry(self, df: pd.DataFrame, feature_cols: list[str],
                       credit_C: float = 10.0, use_gpu: bool = True) -> dict:
        """GAP 11: Train ES entry classifier using triple-barrier labels.

        Args:
            df: DataFrame with OHLCV + indicators + feature columns.
            feature_cols: List of feature column names.
            credit_C: Credit width for stop calculation.
            use_gpu: Use GPU if available.

        Returns:
            Training metrics dict.
        """
        entry_labels = label_entries_triple_barrier(df, credit_C=credit_C)
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        y = entry_labels.values

        logger.info(f"ES entry training: {y.sum()} positive / {len(y)} total "
                    f"({y.mean():.1%} hit rate)")
        return self.train(pd.DataFrame(X, columns=feature_cols),
                          pd.Series(y), use_gpu=use_gpu)

    def train_es_exit(self, df: pd.DataFrame, feature_cols: list[str],
                      use_gpu: bool = True) -> dict:
        """GAP 11: Train ES exit classifier using reversal labels.

        Args:
            df: DataFrame with OHLCV + indicators + feature columns.
            feature_cols: List of feature column names.
            use_gpu: Use GPU if available.

        Returns:
            Training metrics dict.
        """
        exit_labels = label_exits_reversal(df)
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        y = exit_labels.values

        logger.info(f"ES exit training: {y.sum()} reversals / {len(y)} total "
                    f"({y.mean():.1%} reversal rate)")
        return self.train(pd.DataFrame(X, columns=feature_cols),
                          pd.Series(y), use_gpu=use_gpu)


def evaluate_past_prediction(conn, date_str: str) -> Optional[dict]:
    """Compare yesterday's prediction to actual outcome.

    Returns dict with evaluation results or None if no prediction exists.
    """
    row = conn.execute(
        "SELECT direction, confidence FROM predictions WHERE date = ?", (date_str,)
    ).fetchone()
    if not row:
        return None

    predicted = row[0]
    pred_confidence = row[1] or 0

    # Get actual return
    prices = conn.execute(
        "SELECT close FROM prices WHERE date >= ? ORDER BY date LIMIT 2", (date_str,)
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

    # P1: Stratified accuracy dimensions
    # Confidence tier
    if pred_confidence >= 70:
        conf_tier = "high"
    elif pred_confidence >= 50:
        conf_tier = "medium"
    else:
        conf_tier = "low"

    # VIX regime
    macro_row = conn.execute(
        "SELECT vix FROM macro WHERE date = ?", (date_str,)
    ).fetchone()
    vix_val = macro_row[0] if macro_row and macro_row[0] else 18
    if vix_val < 15:
        vix_regime = "low"
    elif vix_val > 25:
        vix_regime = "high"
    else:
        vix_regime = "normal"

    # Day of week
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        dow = d.weekday()
    except Exception:
        dow = 0

    # Event proximity
    try:
        from src.data.calendar import has_nearby_event
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        event_prox = 1 if has_nearby_event(d) else 0
    except Exception:
        event_prox = 0

    # Update cumulative accuracy
    perf_rows = conn.execute("SELECT COUNT(*), SUM(correct) FROM performance").fetchone()
    total = (perf_rows[0] or 0) + 1
    correct_total = (perf_rows[1] or 0) + correct
    cum_accuracy = correct_total / total

    conn.execute(
        """INSERT OR REPLACE INTO performance
           (date, predicted, actual, correct, cumulative_accuracy,
            confidence_tier, vix_regime, day_of_week, event_proximity)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (date_str, predicted, actual, correct, cum_accuracy,
         conf_tier, vix_regime, dow, event_prox)
    )
    conn.commit()

    return {
        "date": date_str, "predicted": predicted, "actual": actual,
        "correct": bool(correct), "cumulative_accuracy": round(cum_accuracy, 3),
        "confidence_tier": conf_tier, "vix_regime": vix_regime,
    }
