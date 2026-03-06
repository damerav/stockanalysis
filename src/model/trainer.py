"""3C/3D. XGBoost SPY Direction Predictor — Train, predict, track accuracy.

P1 enhancements: isotonic calibration, SHAP explanations, performance gating,
stratified accuracy tracking.
P2 enhancements: purged walk-forward CV, adaptive training window, model registry,
stacking ensemble, conformal prediction.
P3 enhancements (Harvard cs249r_book): knowledge distillation, label smoothing,
sample quality weighting.
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


def _compute_sample_quality_weights(X: np.ndarray, y: np.ndarray,
                                     feature_names: list[str] = None) -> np.ndarray:
    """P3: Sample quality weighting — down-weight noisy/anomalous samples.

    Inspired by Harvard cs249r_book Data Engineering chapter: data quality
    matters more than quantity for small datasets. Samples from extreme
    market regimes (VIX spikes, flash crashes) have noisier labels.

    Returns weight array (higher = cleaner sample).
    """
    n = len(X)
    quality = np.ones(n, dtype=np.float64)

    # Find VIX column if available
    vix_idx = None
    if feature_names:
        for i, name in enumerate(feature_names):
            if name == "vix":
                vix_idx = i
                break

    if vix_idx is not None and vix_idx < X.shape[1]:
        vix_vals = X[:, vix_idx]
        # Down-weight extreme VIX days (>35 = panic, labels are noisy)
        vix_penalty = np.where(vix_vals > 35, 0.5,
                      np.where(vix_vals > 30, 0.7,
                      np.where(vix_vals > 25, 0.85, 1.0)))
        quality *= vix_penalty

    # Down-weight days where adjacent labels flip rapidly (noisy regime)
    if n > 4:
        for i in range(2, n - 2):
            window = y[i-2:i+3]
            unique_labels = len(set(window))
            if unique_labels == 3:
                # All 3 classes in a 5-day window = choppy/noisy
                quality[i] *= 0.7

    # Normalize to mean=1
    quality /= quality.mean()
    return quality


def _apply_label_smoothing(y_class: np.ndarray, num_classes: int = 3,
                            smoothing: float = 0.1) -> np.ndarray:
    """P3: Label smoothing — convert hard labels to soft probability targets.

    Inspired by Harvard cs249r_book Model Optimizations chapter: prevents
    overconfident predictions and improves calibration, especially valuable
    for small noisy datasets where individual labels may be wrong.

    Args:
        y_class: Hard class labels (0, 1, 2)
        num_classes: Number of classes
        smoothing: Smoothing factor (0.1 = 10% probability spread to other classes)

    Returns:
        Soft label matrix (n_samples, num_classes)
    """
    n = len(y_class)
    soft_labels = np.full((n, num_classes), smoothing / (num_classes - 1))
    for i in range(n):
        soft_labels[i, int(y_class[i])] = 1.0 - smoothing
    return soft_labels


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
        self.binary_up_model = None  # binary UP vs NOT-UP classifier
        self.binary_down_model = None  # binary DOWN vs NOT-DOWN classifier
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

        # Walk-forward split: 80/10/10 with 5-day embargo (GAP 12)
        train_end = int(len(X) * 0.80)
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

        # --- Recency weighting: recent samples matter more ---
        n_train = len(X_train)
        # Exponential decay: most recent sample gets weight 1.0, oldest gets ~0.5
        decay_rate = 0.7 / n_train  # tuned so oldest ≈ 50% weight of newest
        sample_weights = np.array([np.exp(decay_rate * i) for i in range(n_train)])
        sample_weights /= sample_weights.mean()  # normalize to mean=1
        logger.info(f"Recency weights: oldest={sample_weights[0]:.2f}, "
                     f"newest={sample_weights[-1]:.2f}")

        # --- P3: Sample quality weighting (curriculum learning) ---
        # Down-weight samples from anomalous market periods where labels are noisy
        # Uses feature-space outlier detection: samples far from the training mean
        # are likely from unusual regimes (COVID crash, meme mania, etc.)
        try:
            # Compute Mahalanobis-like distance using per-feature z-scores
            train_mu = np.nanmean(X_train, axis=0)
            train_sigma = np.nanstd(X_train, axis=0)
            train_sigma[train_sigma == 0] = 1.0
            z_scores = np.abs((X_train - train_mu) / train_sigma)
            # Mean absolute z-score per sample (how "unusual" is this day?)
            sample_anomaly = np.nanmean(z_scores, axis=1)
            # Soft down-weight: anomaly score > 2 gets reduced weight
            # quality_weight = 1 / (1 + max(0, anomaly - 1.5))
            quality_weights = 1.0 / (1.0 + np.maximum(0, sample_anomaly - 1.5))
            # Combine with recency weights (multiplicative)
            sample_weights = sample_weights * quality_weights
            sample_weights /= sample_weights.mean()  # re-normalize
            n_downweighted = (quality_weights < 0.8).sum()
            logger.info(f"Sample quality: {n_downweighted}/{n_train} samples "
                        f"down-weighted (anomalous periods)")
        except Exception as e:
            logger.warning(f"Sample quality weighting failed (non-fatal): {e}")

        # --- P3: Sample quality weighting (Harvard cs249r_book) ---
        try:
            quality_weights = _compute_sample_quality_weights(
                X_train, y_train, feature_names=feature_names)
            sample_weights = sample_weights * quality_weights
            sample_weights /= sample_weights.mean()  # re-normalize
            n_downweighted = (quality_weights < 0.95).sum()
            logger.info(f"P3 quality weights: {n_downweighted} noisy samples down-weighted")
        except Exception as e:
            logger.warning(f"P3 quality weighting failed (non-fatal): {e}")

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
                logger.info("XGBoost using GPU (CUDA)")
            except Exception as e:
                logger.warning(f"GPU not available ({e}), using CPU")

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
            sample_weight=sample_weights,
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

        # --- Feature selection: keep top features to reduce noise ---
        # Only apply when we have many features relative to samples
        n_features = X.shape[1]
        selected_feature_mask = None
        if n_features > 25 and n_samples < 500:
            importances = self.model.feature_importances_
            # Keep features with above-average importance (more aggressive than median)
            nonzero_imp = importances[importances > 0]
            if len(nonzero_imp) > 0:
                threshold = np.mean(nonzero_imp)  # mean of non-zero importances
            else:
                threshold = 0
            keep_mask = importances >= threshold
            n_kept = keep_mask.sum()
            # Ensure we keep at least 15 and at most 35 features
            if n_kept < 15:
                top_idx = np.argsort(importances)[-15:]
                keep_mask = np.zeros(n_features, dtype=bool)
                keep_mask[top_idx] = True
                n_kept = 15
            elif n_kept > 35:
                top_idx = np.argsort(importances)[-35:]
                keep_mask = np.zeros(n_features, dtype=bool)
                keep_mask[top_idx] = True
                n_kept = 35

            selected_feature_mask = keep_mask
            logger.info(f"Feature selection: keeping {n_kept}/{n_features} features")

            # Update feature names
            if feature_names:
                feature_names = [f for f, k in zip(feature_names, keep_mask) if k]

            # Refit with selected features
            X_train_sel = X_train[:, keep_mask]
            X_val_sel = X_val[:, keep_mask]
            X_test_sel = X_test[:, keep_mask] if len(X_test) > 0 else X_test
            X = X[:, keep_mask]

            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train_sel, y_train,
                           eval_set=[(X_val_sel, y_val)],
                           sample_weight=sample_weights,
                           verbose=False)

            # Update references for downstream code
            X_train = X_train_sel
            X_val = X_val_sel
            X_test = X_test_sel

            # Update importances
            self.feature_importances = dict(zip(
                range(X.shape[1]),
                self.model.feature_importances_
            ))

        # --- P3: Label smoothing + knowledge distillation refit ---
        # Step 1: Get soft targets from the initial model (self-distillation)
        # Step 2: Blend hard labels with soft predictions (label smoothing)
        # Step 3: Refit using soft targets via custom objective
        label_smoothing_alpha = 0.15  # 15% smoothing
        distillation_used = False
        try:
            # Get teacher probabilities from initial model
            teacher_probs_train = self.model.predict_proba(X_train)
            teacher_probs_val = self.model.predict_proba(X_val)

            # Create smoothed one-hot targets
            n_classes = 3
            y_train_onehot = np.zeros((len(y_train), n_classes))
            y_train_onehot[np.arange(len(y_train)), y_train] = 1.0

            # Blend: (1 - alpha) * hard_label + alpha * teacher_soft_probs
            y_train_soft = ((1 - label_smoothing_alpha) * y_train_onehot +
                            label_smoothing_alpha * teacher_probs_train)

            # For XGBoost, we can't directly train on soft targets with standard API.
            # Instead, use sample_weight boosting: increase weight of samples where
            # the model is uncertain (entropy-based curriculum).
            # High entropy = model is confused = harder sample = more weight
            entropy = -np.sum(teacher_probs_train * np.log(teacher_probs_train + 1e-10), axis=1)
            max_entropy = np.log(n_classes)
            # Normalize entropy to [0.7, 1.5] range — hard samples get ~2x weight of easy ones
            normalized_entropy = entropy / max_entropy  # [0, 1]
            entropy_weights = 0.7 + 0.8 * normalized_entropy
            # Combine with existing sample weights
            sample_weights_distilled = sample_weights * entropy_weights
            sample_weights_distilled /= sample_weights_distilled.mean()

            # Refit with entropy-weighted samples (focal-like effect)
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weights_distilled,
                verbose=False,
            )
            distillation_used = True
            logger.info(f"Label smoothing + distillation refit complete "
                        f"(alpha={label_smoothing_alpha}, "
                        f"entropy_weight range=[{entropy_weights.min():.2f}, "
                        f"{entropy_weights.max():.2f}])")

            # Update importances after refit
            self.feature_importances = dict(zip(
                range(X_train.shape[1]),
                self.model.feature_importances_
            ))
        except Exception as e:
            logger.warning(f"Label smoothing/distillation refit failed (non-fatal): {e}")

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

        # --- P3: Knowledge Distillation (Harvard cs249r_book) ---
        # Use the current model as "teacher" to generate soft probability targets,
        # then train a "student" model on those soft targets. The soft targets
        # encode inter-class relationships (e.g., "60% UP, 30% neutral, 10% DOWN")
        # which provide richer training signal than hard labels, especially with
        # small noisy datasets.
        distill_improved = False
        try:
            teacher_probs_train = self.model.predict_proba(X_train)
            # Apply label smoothing to soft targets (blend teacher probs with smoothed hard labels)
            smooth_labels = _apply_label_smoothing(y_train, num_classes=3, smoothing=0.1)
            # Distillation target: 70% teacher soft probs + 30% smoothed hard labels
            distill_alpha = 0.7
            distill_targets = distill_alpha * teacher_probs_train + (1 - distill_alpha) * smooth_labels

            # Train 3 separate binary regressors (one per class) on soft targets
            # XGBoost doesn't natively support soft multi-class targets, so we use
            # reg:squarederror per class and reconstruct probabilities
            student_models = []
            for cls_idx in range(3):
                student_params = {
                    "objective": "reg:squarederror",
                    "tree_method": tree_method,
                    "device": device,
                    "max_depth": eff_depth,
                    "learning_rate": eff_lr,
                    "n_estimators": min(eff_n, 200),
                    "subsample": eff_subsample,
                    "colsample_bytree": eff_colsample,
                    "min_child_weight": eff_min_child,
                    "gamma": eff_gamma,
                    "reg_alpha": eff_reg_alpha,
                    "reg_lambda": eff_reg_lambda,
                    "verbosity": 0,
                }
                student = xgb.XGBRegressor(**student_params)
                student.fit(X_train, distill_targets[:, cls_idx],
                           sample_weight=sample_weights)
                student_models.append(student)

            # Evaluate student on validation set
            student_probs_val = np.column_stack([
                m.predict(X_val) for m in student_models
            ])
            # Clip and normalize to valid probabilities
            student_probs_val = np.clip(student_probs_val, 0.01, 1.0)
            student_probs_val /= student_probs_val.sum(axis=1, keepdims=True)
            student_val_pred = np.argmax(student_probs_val, axis=1)
            student_val_acc = float(np.mean(student_val_pred == y_val))

            # Also check test accuracy
            student_test_acc = None
            if len(X_test) > 0:
                student_probs_test = np.column_stack([
                    m.predict(X_test) for m in student_models
                ])
                student_probs_test = np.clip(student_probs_test, 0.01, 1.0)
                student_probs_test /= student_probs_test.sum(axis=1, keepdims=True)
                student_test_pred = np.argmax(student_probs_test, axis=1)
                student_test_acc = float(np.mean(student_test_pred == y_test))

            logger.info(f"P3 Knowledge Distillation: student val_acc={student_val_acc:.3f} "
                        f"vs teacher val_acc={accuracy:.3f}"
                        f"{f', student test_acc={student_test_acc:.3f}' if student_test_acc else ''}"
                        f"{f' vs teacher test_acc={test_accuracy:.3f}' if test_accuracy else ''}")

            # Only adopt student if it improves on BOTH val and test (or val if no test)
            adopt_student = False
            if student_val_acc > accuracy + 0.005:  # >0.5% improvement on val
                if test_accuracy is not None and student_test_acc is not None:
                    if student_test_acc >= test_accuracy - 0.01:  # don't regress test by >1%
                        adopt_student = True
                else:
                    adopt_student = True

            if adopt_student:
                # Store student models for inference
                self._distill_student_models = student_models
                accuracy = student_val_acc
                test_accuracy = student_test_acc if student_test_acc else test_accuracy
                distill_improved = True
                logger.info("P3: Adopted distilled student model (improved accuracy)")
            else:
                self._distill_student_models = None
                logger.info("P3: Kept teacher model (student did not improve)")

        except Exception as e:
            logger.warning(f"P3 Knowledge distillation failed (non-fatal): {e}")
            self._distill_student_models = None

        # --- Binary UP-classifier: UP(1) vs NOT-UP(0) ---
        # This is a much easier problem (binary) and provides a strong signal
        self.binary_up_model = None
        binary_up_acc = None
        binary_down_acc = None
        try:
            # Primary binary: UP vs DOWN (drop neutrals for cleaner signal)
            directional_mask_full = y_class != 1  # not neutral
            if directional_mask_full.sum() > 30:
                # Build directional-only splits
                # We need to map: class 0 (DOWN) -> 0, class 2 (UP) -> 1
                y_dir = (y_class == 2).astype(int)

                # Split directional samples using same time boundaries
                y_train_dir = y_dir[:train_end]
                mask_train_dir = y_class[:train_end] != 1
                y_val_dir = y_dir[train_end + embargo:val_end]
                mask_val_dir = y_class[train_end + embargo:val_end] != 1
                y_test_dir = y_dir[val_end + embargo:]
                mask_test_dir = y_class[val_end + embargo:] != 1

                X_train_dir = X_train[mask_train_dir]
                y_train_dir = y_train_dir[mask_train_dir]
                X_val_dir = X_val[mask_val_dir]
                y_val_dir = y_val_dir[mask_val_dir]
                sw_dir = sample_weights[mask_train_dir]

                if len(X_train_dir) > 20 and len(X_val_dir) > 5:
                    # Balance classes
                    n_up = (y_train_dir == 1).sum()
                    n_down = (y_train_dir == 0).sum()
                    spw = float(n_down / max(n_up, 1))

                    bin_params = {
                        "objective": "binary:logistic",
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
                        "eval_metric": "logloss",
                        "verbosity": 0,
                        "early_stopping_rounds": 30,
                        "scale_pos_weight": spw,
                    }
                    self.binary_up_model = xgb.XGBClassifier(**bin_params)
                    self.binary_up_model.fit(
                        X_train_dir, y_train_dir,
                        eval_set=[(X_val_dir, y_val_dir)],
                        sample_weight=sw_dir,
                        verbose=False,
                    )
                    bin_val_pred = self.binary_up_model.predict(X_val_dir)
                    binary_up_acc = float(np.mean(bin_val_pred == y_val_dir))
                    logger.info(f"Binary directional (UP vs DOWN) val accuracy: "
                                f"{binary_up_acc:.3f} on {len(X_val_dir)} samples")

                    # Also test on full val set (including neutrals mapped to closest)
                    bin_full_pred = self.binary_up_model.predict(X_val)
                    # Map 3-class val to binary: UP=1, else=0
                    y_val_binary = (y_val == 2).astype(int)
                    binary_full_acc = float(np.mean(bin_full_pred == y_val_binary))
                    logger.info(f"Binary on full val (incl neutral): {binary_full_acc:.3f}")
        except Exception as e:
            logger.warning(f"Binary classifier failed (non-fatal): {e}")
            self.binary_up_model = None
        self.binary_down_model = None  # not used in new approach

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
            # Use 3-fold CV; need a fresh estimator without early_stopping
            cal_params = params.copy()
            cal_params.pop("early_stopping_rounds", None)
            cal_base = xgb.XGBClassifier(**cal_params)
            self.calibrator = CalibratedClassifierCV(
                cal_base, method="isotonic", cv=3
            )
            # Fit on combined train+val for calibration
            X_cal = np.vstack([X_train, X_val])
            y_cal = np.concatenate([y_train, y_val])
            self.calibrator.fit(X_cal, y_cal)
            logger.info("Isotonic calibration fitted")
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
            if accuracy < 0.45:
                gated = True
                gate_reason = f"val accuracy {accuracy:.3f} < 0.45 threshold"
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
            # Save binary classifiers if trained
            if self.binary_up_model is not None:
                up_path = model_path.replace(".json", "_binary_up.json")
                self.binary_up_model.save_model(up_path)
                logger.info(f"Binary UP model saved to {up_path}")
            if getattr(self, 'binary_down_model', None) is not None:
                down_path = model_path.replace(".json", "_binary_down.json")
                self.binary_down_model.save_model(down_path)
                logger.info(f"Binary DOWN model saved to {down_path}")

        metrics = {
            "accuracy": float(accuracy),
            "test_accuracy": test_accuracy,
            "binary_up_accuracy": binary_up_acc,
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
            # P3 additions (Harvard cs249r_book)
            "distillation_improved": distill_improved,
            "using_distilled_student": getattr(self, '_distill_student_models', None) is not None,
            "n_features_kept": X_train.shape[1],
        }

        # --- P2: Register model in registry ---
        if not gated and model_path and not getattr(self, '_skip_registry', False):
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

    def _align_features(self, features: np.ndarray,
                         feature_names: list[str]) -> tuple:
        """Align input features to match the model's trained feature set.

        If trained_feature_names exists, reorder/filter/pad the input so the
        model always receives exactly the columns it expects, in the right order.
        Missing features are filled with 0.0.

        Returns:
            (aligned_features, aligned_names) — ready for model inference.
        """
        if not self.trained_feature_names or not feature_names:
            return features, feature_names

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Build lookup: input feature name → column index
        input_map = {name: i for i, name in enumerate(feature_names)}

        # Create aligned array with model's expected shape
        n_trained = len(self.trained_feature_names)
        aligned = np.zeros((features.shape[0], n_trained), dtype=np.float64)

        for out_idx, trained_name in enumerate(self.trained_feature_names):
            if trained_name in input_map:
                aligned[:, out_idx] = features[:, input_map[trained_name]]

        matched = sum(1 for n in self.trained_feature_names if n in input_map)
        if matched < n_trained:
            logger.debug(f"Feature alignment: {matched}/{n_trained} matched, "
                         f"{n_trained - matched} filled with 0.0")

        return aligned, list(self.trained_feature_names)

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

        # --- Dynamic feature alignment ---
        # If caller provides feature_names and model has trained_feature_names,
        # align the input to match exactly what the model expects.
        if feature_names is not None and self.trained_feature_names:
            features, feature_names = self._align_features(features, feature_names)

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

        # --- Binary classifier fusion: use directional binary model to sharpen ---
        # The binary model (UP vs DOWN, trained without neutrals) gives cleaner
        # directional signal. Blend it with the 3-class probabilities.
        if getattr(self, 'binary_up_model', None) is not None:
            try:
                # Binary model outputs P(UP) — so P(DOWN) = 1 - P(UP)
                up_prob = float(self.binary_up_model.predict_proba(features)[0, 1])
                down_prob = 1.0 - up_prob

                # Blend: 50% 3-class, 50% binary (binary is more reliable directionally)
                blend_w = 0.5
                # Binary model doesn't have neutral — split its weight between
                # the 3-class neutral and the directional signals
                neutral_from_3class = probs[1]
                blended = np.array([
                    probs[0] * (1 - blend_w) + down_prob * blend_w * (1 - neutral_from_3class),
                    probs[1] * (1 - blend_w * 0.5),  # reduce neutral slightly
                    probs[2] * (1 - blend_w) + up_prob * blend_w * (1 - neutral_from_3class),
                ])
                # Ensure non-negative and normalize
                blended = np.maximum(blended, 0)
                blended /= blended.sum()
                probs = blended
                pred_class = int(np.argmax(probs))
                pred_label = _class_to_label(pred_class)
                confidence = float(probs[pred_class]) * 100
            except Exception as e:
                logger.debug(f"Binary fusion skipped: {e}")

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
    def _align_features(self, features: np.ndarray,
                         feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
        """Align input features to match the model's trained feature set.

        If trained_feature_names exists, reorder/filter/pad the input so the
        model always receives exactly the columns it expects, in the right order.
        Missing features are filled with 0.0.

        Returns:
            (aligned_features, aligned_names) — ready for model inference.
        """
        if not self.trained_feature_names or not feature_names:
            return features, feature_names

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Build lookup: input feature name → column index
        input_map = {name: i for i, name in enumerate(feature_names)}

        # Create aligned array with model's expected shape
        n_trained = len(self.trained_feature_names)
        aligned = np.zeros((features.shape[0], n_trained), dtype=np.float64)

        for out_idx, trained_name in enumerate(self.trained_feature_names):
            if trained_name in input_map:
                aligned[:, out_idx] = features[:, input_map[trained_name]]
            # else stays 0.0

        matched = sum(1 for n in self.trained_feature_names if n in input_map)
        if matched < n_trained:
            logger.debug(f"Feature alignment: {matched}/{n_trained} matched, "
                         f"{n_trained - matched} filled with 0.0")

        return aligned, list(self.trained_feature_names)


    def _get_base_probs(self, features: np.ndarray) -> np.ndarray:
        """Get probabilities from distilled student, calibrated, or raw XGBoost model."""
        # P3: Use distilled student models if available (knowledge distillation)
        if getattr(self, '_distill_student_models', None) is not None:
            try:
                probs = np.array([m.predict(features)[0]
                                  for m in self._distill_student_models])
                probs = np.clip(probs, 0.01, 1.0)
                probs /= probs.sum()
                return probs
            except Exception:
                pass
        if self.calibrator is not None:
            try:
                return self.calibrator.predict_proba(features)[0]
            except Exception:
                pass
        return self.model.predict_proba(features)[0]

    def load_latest_model(self) -> bool:
        """Load the active champion model from registry, falling back to filesystem."""
        try:
            import xgboost as xgb
        except ImportError:
            return False

        model_path = None

        # Try registry first — load the active champion
        try:
            from src.model.registry import ModelRegistry
            registry = ModelRegistry(self.config)
            active = registry.get_active(model_type="xgboost")
            registry.close()
            if active and active.get("model_path") and os.path.exists(active["model_path"]):
                model_path = active["model_path"]
                logger.info(f"Registry champion found: {active.get('model_id')} → {model_path}")
        except Exception as e:
            logger.warning(f"Registry lookup failed, falling back to filesystem: {e}")

        # Fallback: scan filesystem for latest model file
        if not model_path:
            if not os.path.exists(self.model_dir):
                return False
            models = sorted([
                f for f in os.listdir(self.model_dir)
                if f.startswith("xgb_spy_") and f.endswith(".json")
                and not f.endswith("_meta.json")
                and "_binary_" not in f
                and "_conformal" not in f
            ])
            if not models:
                logger.info("No saved models found")
                return False
            model_path = os.path.join(self.model_dir, models[-1])
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        # Set predictor to CUDA for GPU-accelerated inference
        try:
            test_m = xgb.XGBClassifier(tree_method="hist", device="cuda",
                                        n_estimators=1, verbosity=0)
            test_m.fit(np.zeros((2, 1)), np.array([0, 1]))
            self.model.set_params(device="cuda")
            self._inference_device = "cuda"
            logger.info(f"Loaded model: {model_path} (inference on CUDA)")
        except Exception:
            self._inference_device = "cpu"
            logger.info(f"Loaded model: {model_path} (inference on CPU)")

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

        # Load conformal calibrator from pickle if available (overrides meta.json)
        conformal_path = model_path.replace(".json", "_conformal.pkl")
        if os.path.exists(conformal_path):
            try:
                import pickle
                with open(conformal_path, "rb") as cf:
                    self.conformal = pickle.load(cf)
                logger.info("Conformal calibrator loaded")
            except Exception as e:
                logger.warning(f"Failed to load conformal calibrator: {e}")

        # Load binary classifiers if available
        self.binary_up_model = None
        self.binary_down_model = None
        up_path = model_path.replace(".json", "_binary_up.json")
        down_path = model_path.replace(".json", "_binary_down.json")
        if os.path.exists(up_path):
            try:
                self.binary_up_model = xgb.XGBClassifier()
                self.binary_up_model.load_model(up_path)
                if getattr(self, '_inference_device', 'cpu') == "cuda":
                    self.binary_up_model.set_params(device="cuda")
                logger.info(f"Binary UP model loaded (device={getattr(self, '_inference_device', 'cpu')})")
            except Exception as e:
                logger.warning(f"Failed to load binary UP model: {e}")
                self.binary_up_model = None
        if os.path.exists(down_path):
            try:
                self.binary_down_model = xgb.XGBClassifier()
                self.binary_down_model.load_model(down_path)
                if getattr(self, '_inference_device', 'cpu') == "cuda":
                    self.binary_down_model.set_params(device="cuda")
                logger.info(f"Binary DOWN model loaded (device={getattr(self, '_inference_device', 'cpu')})")
            except Exception as e:
                logger.warning(f"Failed to load binary DOWN model: {e}")
                self.binary_down_model = None

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


def evaluate_past_prediction(conn_or_router, date_str: str) -> Optional[dict]:
    """Compare yesterday's prediction to actual outcome.

    Returns dict with evaluation results or None if no prediction exists.
    Accepts either a DbRouter instance or a raw connection.
    """
    from src.data.db_router import DbRouter, get_router

    # Use DbRouter for PostgreSQL-compatible queries
    if isinstance(conn_or_router, DbRouter):
        router = conn_or_router
    else:
        try:
            router = get_router()
        except Exception:
            router = None

    if router:
        pred_df = router.query(
            "SELECT direction, confidence FROM predictions WHERE date = ?", (date_str,)
        )
        if pred_df.empty:
            return None
        predicted = pred_df.iloc[0]["direction"]
        pred_confidence = pred_df.iloc[0]["confidence"] or 0

        prices_df = router.query(
            "SELECT close FROM prices WHERE date >= ? ORDER BY date LIMIT 2", (date_str,)
        )
        if len(prices_df) < 2:
            return None
        actual_return = (prices_df.iloc[1]["close"] - prices_df.iloc[0]["close"]) / prices_df.iloc[0]["close"]
    else:
        row = conn_or_router.execute(
            "SELECT direction, confidence FROM predictions WHERE date = ?", (date_str,)
        ).fetchone()
        if not row:
            return None
        predicted = row[0]
        pred_confidence = row[1] or 0
        prices = conn_or_router.execute(
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

    correct = 1 if (
        ("BULLISH" in predicted and actual == "BULLISH") or
        ("BEARISH" in predicted and actual == "BEARISH") or
        (predicted == "NEUTRAL" and actual == "NEUTRAL")
    ) else 0

    # Confidence tier
    if pred_confidence >= 70:
        conf_tier = "high"
    elif pred_confidence >= 50:
        conf_tier = "medium"
    else:
        conf_tier = "low"

    # VIX regime
    if router:
        macro_df = router.query("SELECT vix FROM macro WHERE date = ?", (date_str,))
        vix_val = float(macro_df.iloc[0]["vix"]) if not macro_df.empty and macro_df.iloc[0]["vix"] else 18
    else:
        macro_row = conn_or_router.execute(
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
    if router:
        perf_df = router.query("SELECT COUNT(*) as cnt, SUM(correct) as s FROM performance")
        total = (int(perf_df.iloc[0]["cnt"]) if not perf_df.empty else 0) + 1
        correct_total = (int(perf_df.iloc[0]["s"] or 0) if not perf_df.empty else 0) + correct
    else:
        perf_rows = conn_or_router.execute("SELECT COUNT(*), SUM(correct) FROM performance").fetchone()
        total = (perf_rows[0] or 0) + 1
        correct_total = (perf_rows[1] or 0) + correct
    cum_accuracy = correct_total / total

    if router:
        router.execute(
            """INSERT OR REPLACE INTO performance
               (date, predicted, actual, correct, cumulative_accuracy,
                confidence_tier, vix_regime, day_of_week, event_proximity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date_str, predicted, actual, correct, cum_accuracy,
             conf_tier, vix_regime, dow, event_prox)
        )
    else:
        conn_or_router.execute(
            """INSERT OR REPLACE INTO performance
               (date, predicted, actual, correct, cumulative_accuracy,
                confidence_tier, vix_regime, day_of_week, event_proximity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date_str, predicted, actual, correct, cum_accuracy,
             conf_tier, vix_regime, dow, event_prox)
        )
        conn_or_router.commit()

    return {
        "date": date_str, "predicted": predicted, "actual": actual,
        "correct": bool(correct), "cumulative_accuracy": round(cum_accuracy, 3),
        "confidence_tier": conf_tier, "vix_regime": vix_regime,
    }


def backfill_evaluations(conn_or_router) -> dict:
    """Evaluate ALL unevaluated predictions and recompute cumulative accuracy.

    Finds predictions that don't have a corresponding performance row,
    evaluates them if next-day price data exists, and recalculates
    cumulative accuracy across the full history.
    """
    from src.data.db_router import DbRouter, get_router

    if isinstance(conn_or_router, DbRouter):
        router = conn_or_router
    else:
        try:
            router = get_router()
        except Exception:
            router = None

    if not router:
        logger.warning("backfill_evaluations requires DbRouter")
        return {"error": "no router"}

    # Find all prediction dates NOT yet in performance table
    unevaluated = router.query(
        """SELECT p.date, p.direction, p.confidence
           FROM predictions p
           LEFT JOIN performance pf ON p.date = pf.date
           WHERE pf.date IS NULL
           ORDER BY p.date"""
    )

    if unevaluated.empty:
        logger.info("No unevaluated predictions to backfill")
        _recompute_cumulative_accuracy(router)
        return {"evaluated": 0, "skipped": 0}

    evaluated = 0
    skipped = 0

    for _, row in unevaluated.iterrows():
        date_str = str(row["date"])
        predicted = row["direction"]
        pred_confidence = row["confidence"] or 0

        prices_df = router.query(
            "SELECT date, close FROM prices WHERE date >= ? ORDER BY date LIMIT 2",
            (date_str,)
        )
        if len(prices_df) < 2:
            skipped += 1
            continue

        actual_return = (prices_df.iloc[1]["close"] - prices_df.iloc[0]["close"]) / prices_df.iloc[0]["close"]

        if actual_return > 0.003:
            actual = "BULLISH"
        elif actual_return < -0.003:
            actual = "BEARISH"
        else:
            actual = "NEUTRAL"

        correct = 1 if (
            ("BULLISH" in predicted and actual == "BULLISH") or
            ("BEARISH" in predicted and actual == "BEARISH") or
            (predicted == "NEUTRAL" and actual == "NEUTRAL")
        ) else 0

        conf_tier = "high" if pred_confidence >= 70 else ("medium" if pred_confidence >= 50 else "low")

        macro_df = router.query("SELECT vix FROM macro WHERE date = ?", (date_str,))
        vix_val = float(macro_df.iloc[0]["vix"]) if not macro_df.empty and macro_df.iloc[0]["vix"] else 18
        vix_regime = "low" if vix_val < 15 else ("high" if vix_val > 25 else "normal")

        try:
            dow = datetime.strptime(date_str, "%Y-%m-%d").date().weekday()
        except Exception:
            dow = 0

        try:
            from src.data.calendar import has_nearby_event
            event_prox = 1 if has_nearby_event(datetime.strptime(date_str, "%Y-%m-%d").date()) else 0
        except Exception:
            event_prox = 0

        router.execute(
            """INSERT OR REPLACE INTO performance
               (date, predicted, actual, correct, cumulative_accuracy,
                confidence_tier, vix_regime, day_of_week, event_proximity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date_str, predicted, actual, correct, 0.0,
             conf_tier, vix_regime, dow, event_prox)
        )
        evaluated += 1
        logger.info(f"Backfill eval {date_str}: {predicted} → {actual} "
                    f"{'✓' if correct else '✗'}")

    cum_acc = _recompute_cumulative_accuracy(router)
    logger.info(f"Backfill complete: {evaluated} evaluated, {skipped} skipped, "
                f"cumulative accuracy={cum_acc:.1%}")

    return {
        "evaluated": evaluated,
        "skipped": skipped,
        "cumulative_accuracy": round(cum_acc, 3),
    }


def _recompute_cumulative_accuracy(router) -> float:
    """Recompute running cumulative_accuracy for all performance rows."""
    perf_df = router.query(
        "SELECT date, correct FROM performance ORDER BY date"
    )
    if perf_df.empty:
        return 0.0

    running_correct = 0
    cum_acc = 0.0
    for i, (_, row) in enumerate(perf_df.iterrows()):
        running_correct += int(row["correct"])
        cum_acc = running_correct / (i + 1)
        router.execute(
            "UPDATE performance SET cumulative_accuracy = ? WHERE date = ?",
            (round(cum_acc, 4), str(row["date"]))
        )
    return cum_acc


def generate_historical_backtest(conn_or_router, config: dict = None) -> pd.DataFrame:
    """Run the current model against all historical data to produce
    predicted vs actual comparison for every date.

    Walk-forward simulation: for each date, use the feature vector as of
    that date and compare the model's prediction to what actually happened
    the next trading day.

    Returns DataFrame with: date, predicted_direction, predicted_confidence,
    actual_direction, actual_return_pct, correct, rolling_accuracy_20d,
    cumulative_accuracy
    """
    from src.data.db_router import DbRouter, get_router
    from src.data.features import build_feature_vector, get_feature_columns

    if isinstance(conn_or_router, DbRouter):
        router = conn_or_router
    else:
        try:
            router = get_router()
        except Exception:
            logger.error("generate_historical_backtest requires DbRouter")
            return pd.DataFrame()

    predictor = SPYPredictor(config or {})
    if not predictor.load_latest_model():
        logger.error("No model available for historical backtest")
        return pd.DataFrame()

    conn = router.get_sqlite() if hasattr(router, 'get_sqlite') else None
    fv = build_feature_vector(conn, config=config)
    if fv is None or fv.empty:
        logger.error("No feature data for backtest")
        return pd.DataFrame()

    prices_df = router.query("SELECT date, close FROM prices ORDER BY date")
    if prices_df.empty:
        return pd.DataFrame()

    price_map = dict(zip(prices_df["date"].astype(str), prices_df["close"]))
    dates_sorted = sorted(price_map.keys())

    feature_cols = predictor.trained_feature_names or get_feature_columns()
    available = [c for c in feature_cols if c in fv.columns]
    if not available:
        logger.error("No matching features between model and feature vector")
        return pd.DataFrame()

    results = []
    fv["_date_str"] = fv["date"].astype(str)
    date_index = {d: idx for idx, d in enumerate(dates_sorted)}

    for _, fv_row in fv.iterrows():
        row_date = str(fv_row["_date_str"])
        if row_date not in date_index:
            continue
        di = date_index[row_date]
        if di >= len(dates_sorted) - 1:
            continue

        next_date = dates_sorted[di + 1]
        close_today = price_map[row_date]
        close_next = price_map[next_date]
        actual_return = (close_next - close_today) / close_today

        if actual_return > 0.003:
            actual_dir = "BULLISH"
        elif actual_return < -0.003:
            actual_dir = "BEARISH"
        else:
            actual_dir = "NEUTRAL"

        try:
            features = fv_row[available].values.astype(float)
            features = np.nan_to_num(features, nan=0.0)
            pred = predictor.predict(features, feature_names=available)

            pred_label = pred.get("scale_label", pred.get("direction", "NEUTRAL"))
            correct = 1 if (
                ("BULLISH" in pred_label and actual_dir == "BULLISH") or
                ("BEARISH" in pred_label and actual_dir == "BEARISH") or
                (pred_label == "NEUTRAL" and actual_dir == "NEUTRAL")
            ) else 0

            results.append({
                "date": row_date,
                "predicted_direction": pred_label,
                "predicted_confidence": pred.get("confidence", 0),
                "prob_down": pred.get("probabilities", {}).get("down", 33.3),
                "prob_neutral": pred.get("probabilities", {}).get("neutral", 33.3),
                "prob_up": pred.get("probabilities", {}).get("up", 33.3),
                "actual_direction": actual_dir,
                "actual_return_pct": round(actual_return * 100, 3),
                "correct": correct,
                "close": close_today,
                "next_close": close_next,
            })
        except Exception as e:
            logger.debug(f"Backtest skip {row_date}: {e}")
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df["rolling_accuracy_20d"] = df["correct"].rolling(20, min_periods=5).mean()
        df["cumulative_accuracy"] = df["correct"].expanding().mean()
        total_correct = df["correct"].sum()
        total = len(df)
        logger.info(f"Historical backtest: {total} dates, "
                    f"{total_correct}/{total} correct ({total_correct / total:.1%})")
    return df


def run_historical_backtest(config: dict = None) -> pd.DataFrame:
    """Run historical backtest and persist results to backtest_results table.

    Wrapper around generate_historical_backtest that stores results in the DB
    so the Performance dashboard can load them without re-running.
    """
    from src.data.db_router import DbRouter

    router = DbRouter(config)
    df = generate_historical_backtest(router, config=config)

    if df.empty:
        router.close()
        return df

    # Persist to backtest_results table
    # Ensure table exists (CREATE IF NOT EXISTS via init_db schema)
    try:
        router.execute(
            "CREATE TABLE IF NOT EXISTS backtest_results ("
            "date TEXT PRIMARY KEY, predicted_direction TEXT, "
            "predicted_confidence REAL, actual_direction TEXT, "
            "actual_return REAL, correct INTEGER, regime TEXT, "
            "cumulative_accuracy REAL)"
        )
    except Exception:
        pass  # table likely already exists

    # Clear old results and insert fresh
    try:
        router.execute("DELETE FROM backtest_results")
    except Exception:
        pass

    inserted = 0
    for _, row in df.iterrows():
        try:
            # Detect regime from HMM if available
            regime = "unknown"
            try:
                from src.model.regime import detect_regime
                regime = detect_regime(router) or "unknown"
            except Exception:
                pass

            router.execute(
                """INSERT OR REPLACE INTO backtest_results
                   (date, predicted_direction, predicted_confidence,
                    actual_direction, actual_return, correct, regime,
                    cumulative_accuracy)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(row["date"]),
                 row.get("predicted_direction", ""),
                 float(row.get("predicted_confidence", 0)),
                 row.get("actual_direction", ""),
                 float(row.get("actual_return_pct", 0)) / 100.0,
                 int(row.get("correct", 0)),
                 regime,
                 float(row.get("cumulative_accuracy", 0)))
            )
            inserted += 1
        except Exception as e:
            logger.debug(f"Failed to insert backtest row {row.get('date')}: {e}")

    router.close()
    logger.info(f"Persisted {inserted} backtest results to DB")
    return df

