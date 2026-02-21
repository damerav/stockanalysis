"""3E/3F. ES AI Models — XGBoost Entry Gate + CNN Exit Controller + Drift Monitor."""

import os
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


# --- 3E: ES Entry Gate (XGBoost-GPU) ---

ES_ENTRY_FEATURES = [
    "price_vs_kc_mid", "price_vs_vwap", "rsi", "roc_3", "atr_regime_pct",
    "volume_ratio", "kc_width", "ema9_slope", "macd_hist", "bb_width",
    "momentum_3bar", "momentum_5bar", "bars_since_trade", "daily_pnl",
    "time_sin", "time_cos", "spread_vs_atr",
]

# Regime-specific entry thresholds
REGIME_THRESHOLDS = {"Low": 0.58, "Med": 0.55, "High": 0.52}


class ESEntryGate:
    """XGBoost entry gate for ES futures strategy."""

    def __init__(self, config: dict = None):
        self.model = None
        self.model_dir = "./models"
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, X: np.ndarray, y: np.ndarray, use_gpu: bool = True) -> dict:
        """Train entry gate with triple-barrier meta-labels.

        y: 1 = TP1 hit before stop, 0 = stop hit or timeout
        """
        try:
            import xgboost as xgb
        except ImportError:
            return {"error": "xgboost not installed"}

        if len(X) < 30:
            return {"error": "insufficient data"}

        # Purged walk-forward split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Elastic sample weights: recent 6 months emphasized
        n = len(X_train)
        half = n // 2
        weights = np.ones(n)
        weights[half:] = 1.5  # recent half gets 50% more weight

        device = "cpu"
        if use_gpu:
            try:
                import xgboost as xgb
                test = xgb.XGBClassifier(device="cuda", n_estimators=1, verbosity=0)
                test.fit(X_train[:5], y_train[:5])
                device = "cuda"
            except Exception:
                pass

        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist", device=device,
            max_depth=5, learning_rate=0.03, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30, eval_metric="logloss",
        )
        self.model.fit(X_train, y_train, sample_weight=weights,
                       eval_set=[(X_val, y_val)], verbose=False)

        val_pred = self.model.predict(X_val)
        accuracy = np.mean(val_pred == y_val)

        model_path = os.path.join(self.model_dir, "es_entry_gate.json")
        self.model.save_model(model_path)
        logger.info(f"ES entry gate trained: accuracy={accuracy:.3f}")

        return {"accuracy": float(accuracy), "model_path": model_path}

    def predict(self, features: np.ndarray, regime: str = "Med") -> dict:
        """Predict entry probability and compute lot sizing.

        Returns:
            p_enter: probability of TP1 hit
            should_enter: bool based on regime threshold
            quantity: 1-3 lots based on probability
        """
        if self.model is None:
            return {"p_enter": 0.0, "should_enter": False, "quantity": 0}

        if features.ndim == 1:
            features = features.reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0)

        p_enter = float(self.model.predict_proba(features)[0, 1])
        threshold = REGIME_THRESHOLDS.get(regime, 0.55)
        should_enter = p_enter >= threshold

        # Sizing: qty = round(base × clip((p-p_min)/0.20, 0, 1))
        base = 3
        p_min = threshold
        sizing_factor = np.clip((p_enter - p_min) / 0.20, 0, 1)
        quantity = max(1, round(base * sizing_factor)) if should_enter else 0

        return {
            "p_enter": round(p_enter, 4),
            "should_enter": should_enter,
            "quantity": quantity,
            "regime": regime,
            "threshold": threshold,
        }

    def load(self) -> bool:
        try:
            import xgboost as xgb
            path = os.path.join(self.model_dir, "es_entry_gate.json")
            if os.path.exists(path):
                self.model = xgb.XGBClassifier()
                self.model.load_model(path)
                return True
        except Exception:
            pass
        return False


# --- 3F: ES Exit Controller (1D-CNN) ---

class ESExitController:
    """1D-CNN exit controller for ES futures strategy."""

    # Trail multiplier bounds by regime
    RUNNER_BOUNDS = {"Low": (1.2, 1.6), "Med": (1.3, 1.7), "High": (1.5, 2.0)}
    TP2_BOUNDS = {"Low": (0.9, 1.2), "Med": (1.0, 1.25), "High": (1.25, 1.5)}

    def __init__(self, n_features: int = 19, lookback: int = 20):
        self.n_features = n_features
        self.lookback = lookback
        self.model = None

    def build_model(self):
        """Build 1D-CNN: Conv1d(features→64→32→16) + Pool + FC."""
        try:
            import torch
            import torch.nn as nn

            class ExitCNN(nn.Module):
                def __init__(self, n_features, lookback):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv1d(64, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv1d(32, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool1d(1),
                    )
                    self.fc = nn.Sequential(
                        nn.Linear(16, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    # x: (batch, features, lookback)
                    x = self.conv(x)
                    x = x.squeeze(-1)
                    return self.fc(x)

            self.model = ExitCNN(self.n_features, self.lookback)
            logger.info("Exit CNN model built")
            return True
        except ImportError:
            logger.warning("PyTorch not installed, exit controller unavailable")
            return False

    def predict(self, bar_window: np.ndarray, regime: str = "Med") -> dict:
        """Predict continuation probability and compute trail multipliers.

        Args:
            bar_window: (lookback, n_features) array of recent bars
            regime: Current volatility regime

        Returns:
            p_cont_5: probability price continues 5 more bars
            runner_trail: trail multiplier for runner lot (× ATR)
            tp2_trail: trail multiplier for TP2 lot (× ATR)
        """
        if self.model is None:
            # Default to mid-range multipliers
            runner_lo, runner_hi = self.RUNNER_BOUNDS.get(regime, (1.3, 1.7))
            tp2_lo, tp2_hi = self.TP2_BOUNDS.get(regime, (1.0, 1.25))
            return {
                "p_cont_5": 0.5,
                "runner_trail": (runner_lo + runner_hi) / 2,
                "tp2_trail": (tp2_lo + tp2_hi) / 2,
            }

        try:
            import torch
            x = torch.FloatTensor(bar_window.T).unsqueeze(0)  # (1, features, lookback)
            with torch.no_grad():
                p_cont = float(self.model(x).item())
        except Exception as e:
            logger.warning(f"Exit CNN prediction failed: {e}")
            p_cont = 0.5

        # Map probability to trail multipliers within regime bounds
        runner_lo, runner_hi = self.RUNNER_BOUNDS.get(regime, (1.3, 1.7))
        tp2_lo, tp2_hi = self.TP2_BOUNDS.get(regime, (1.0, 1.25))

        runner_trail = runner_lo + p_cont * (runner_hi - runner_lo)
        tp2_trail = tp2_lo + p_cont * (tp2_hi - tp2_lo)

        return {
            "p_cont_5": round(p_cont, 4),
            "runner_trail": round(runner_trail, 3),
            "tp2_trail": round(tp2_trail, 3),
        }

    def load(self, path: str = "./models/es_exit_cnn.pt") -> bool:
        try:
            import torch
            if os.path.exists(path):
                self.build_model()
                self.model.load_state_dict(torch.load(path, weights_only=True))
                self.model.eval()
                return True
        except Exception:
            pass
        return False

    def save(self, path: str = "./models/es_exit_cnn.pt"):
        try:
            import torch
            if self.model:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(self.model.state_dict(), path)
        except Exception as e:
            logger.warning(f"Failed to save exit CNN: {e}")


# --- Drift Monitor ---

class DriftMonitor:
    """Monitors feature drift and AI performance degradation."""

    def __init__(self, psi_threshold: float = 0.2, ir_lookback: int = 100):
        self.psi_threshold = psi_threshold
        self.ir_lookback = ir_lookback
        self.reference_distributions: dict = {}
        self.ai_trades: list = []
        self.rules_trades: list = []

    def set_reference(self, feature_name: str, values: np.ndarray):
        """Set reference distribution for a feature."""
        hist, edges = np.histogram(values, bins=10, density=True)
        self.reference_distributions[feature_name] = (hist, edges)

    def compute_psi(self, feature_name: str, current_values: np.ndarray) -> float:
        """Compute Population Stability Index for a feature."""
        if feature_name not in self.reference_distributions:
            return 0.0

        ref_hist, edges = self.reference_distributions[feature_name]
        curr_hist, _ = np.histogram(current_values, bins=edges, density=True)

        # Avoid division by zero
        ref_hist = np.clip(ref_hist, 1e-6, None)
        curr_hist = np.clip(curr_hist, 1e-6, None)

        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()

        psi = np.sum((curr_hist - ref_hist) * np.log(curr_hist / ref_hist))
        return float(psi)

    def check_drift(self, features: dict[str, np.ndarray]) -> dict:
        """Check all features for drift. Returns action recommendation."""
        max_psi = 0.0
        drifted_features = []

        for name, values in features.items():
            psi = self.compute_psi(name, values)
            if psi > self.psi_threshold:
                drifted_features.append((name, psi))
            max_psi = max(max_psi, psi)

        action = "none"
        if max_psi > self.psi_threshold:
            action = "halve_size_and_refit"
            logger.warning(f"Drift detected: PSI={max_psi:.3f}, features={drifted_features}")

        return {
            "max_psi": round(max_psi, 4),
            "drifted_features": drifted_features,
            "action": action,
        }

    def record_trade(self, pnl: float, is_ai: bool):
        """Record a trade result for IR comparison."""
        if is_ai:
            self.ai_trades.append(pnl)
        else:
            self.rules_trades.append(pnl)

    def check_ai_performance(self) -> dict:
        """Compare AI vs rules performance. Disable AI if underperforming."""
        ai = self.ai_trades[-self.ir_lookback:]
        rules = self.rules_trades[-self.ir_lookback:]

        if len(ai) < 20 or len(rules) < 20:
            return {"action": "insufficient_data"}

        ai_mean = np.mean(ai)
        rules_mean = np.mean(rules)
        rules_std = np.std(rules) or 1.0

        # AI underperforms rules by 1σ → disable
        if ai_mean < rules_mean - rules_std:
            logger.warning(f"AI underperforming: AI IR={ai_mean:.2f} vs Rules={rules_mean:.2f}")
            return {"action": "disable_ai", "ai_mean": ai_mean, "rules_mean": rules_mean}

        return {"action": "none", "ai_mean": ai_mean, "rules_mean": rules_mean}
