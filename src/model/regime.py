"""HMM Regime Detector — 4-state market regime classification.

States: Bull Trending, Bear Trending, High-Vol Choppy, Low-Vol Ranging.
Trained on VIX, realised volatility, and rolling return features.
Used to select regime-specific model parameters and adjust thresholds.
"""

import logging
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

REGIME_NAMES = {
    0: "bull_trend",
    1: "bear_trend",
    2: "high_vol_choppy",
    3: "low_vol_range",
}

REGIME_LABELS = {
    "bull_trend": "🟢 Bull Trend",
    "bear_trend": "🔴 Bear Trend",
    "high_vol_choppy": "🟡 High-Vol Choppy",
    "low_vol_range": "🔵 Low-Vol Range",
}

# Neutral threshold per regime
REGIME_THRESHOLDS = {
    "bull_trend": 0.002,
    "bear_trend": 0.002,
    "high_vol_choppy": 0.005,
    "low_vol_range": 0.001,
}


class HMMRegimeDetector:
    """Gaussian HMM for market regime detection."""

    def __init__(self, n_states: int = 4, model_dir: str = "./models"):
        self.n_states = n_states
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.state_map = {}  # Maps HMM state indices to regime names
        self._model_path = os.path.join(model_dir, "hmm_regime.pkl")

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build HMM observation features from price/macro data.

        Features:
            - 5-day rolling return
            - 20-day realised volatility
            - VIX level (normalised)
            - VIX change (5-day)
            - Volume trend (vs 20-day avg)
        """
        close = df["close"].values.astype(float)
        vix = df["vix"].values.astype(float) if "vix" in df.columns else np.full(len(df), 18.0)
        volume = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(df))

        returns_5d = pd.Series(close).pct_change(5).values
        realised_vol = pd.Series(close).pct_change().rolling(20).std().values * np.sqrt(252)
        vix_norm = vix / 20.0  # Normalise around long-run mean
        vix_change = pd.Series(vix).diff(5).values
        vol_trend = volume / pd.Series(volume).rolling(20).mean().values

        features = np.column_stack([
            returns_5d, realised_vol, vix_norm, vix_change, vol_trend
        ])

        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0)
        return features

    def fit(self, df: pd.DataFrame) -> dict:
        """Train HMM on historical price/macro data.

        Args:
            df: DataFrame with close, vix, volume columns

        Returns:
            Training metrics dict
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn not installed. Run: pip install hmmlearn")
            return {"error": "hmmlearn not installed"}

        X = self._build_features(df)
        # Skip first 20 rows (NaN from rolling calcs)
        X = X[20:]

        # Standardize features for better HMM convergence
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        n_states = min(self.n_states, max(2, len(X) // 50))

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
            tol=0.01,
        )
        self.model.fit(X)

        # Decode states
        states = self.model.predict(X)

        # Map HMM states to regime names by characteristics
        self._map_states(X, states)

        os.makedirs(self.model_dir, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump({"model": self.model, "state_map": self.state_map,
                         "scaler": scaler}, f)

        # Compute regime distribution
        unique, counts = np.unique(states, return_counts=True)
        dist = {REGIME_NAMES.get(self.state_map.get(s, s), f"state_{s}"): int(c)
                for s, c in zip(unique, counts)}

        logger.info(f"HMM regime detector trained: {dist}")
        return {
            "n_states": self.n_states,
            "regime_distribution": dist,
            "log_likelihood": float(self.model.score(X)),
            "n_samples": len(X),
        }

    def _map_states(self, X: np.ndarray, states: np.ndarray):
        """Map HMM state indices to meaningful regime names.

        Uses mean return and volatility per state to classify:
        - Highest return + low vol → bull_trend
        - Lowest return + low vol → bear_trend
        - Highest vol → high_vol_choppy
        - Lowest vol + near-zero return → low_vol_range
        """
        state_stats = {}
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() == 0:
                continue
            state_stats[s] = {
                "mean_return": float(np.mean(X[mask, 0])),
                "mean_vol": float(np.mean(X[mask, 1])),
                "count": int(mask.sum()),
            }

        if not state_stats:
            self.state_map = {i: i for i in range(self.n_states)}
            return

        # Sort by volatility to find high/low vol states
        by_vol = sorted(state_stats.items(), key=lambda x: x[1]["mean_vol"])
        # Sort by return to find bull/bear
        by_ret = sorted(state_stats.items(), key=lambda x: x[1]["mean_return"])

        assigned = set()
        self.state_map = {}

        # Highest vol → high_vol_choppy (state 2)
        high_vol_state = by_vol[-1][0]
        self.state_map[high_vol_state] = 2
        assigned.add(high_vol_state)

        # Highest return (not already assigned) → bull_trend (state 0)
        for s, _ in reversed(by_ret):
            if s not in assigned:
                self.state_map[s] = 0
                assigned.add(s)
                break

        # Lowest return (not already assigned) → bear_trend (state 1)
        for s, _ in by_ret:
            if s not in assigned:
                self.state_map[s] = 1
                assigned.add(s)
                break

        # Remaining → low_vol_range (state 3)
        for s in range(self.n_states):
            if s not in assigned:
                self.state_map[s] = 3

    def predict(self, df: pd.DataFrame) -> str:
        """Predict current market regime.

        Args:
            df: Recent price/macro DataFrame (at least 25 rows)

        Returns:
            Regime name string (e.g., "bull_trend")
        """
        if self.model is None:
            if not self.load():
                return "low_vol_range"  # Safe default

        X = self._build_features(df)
        X = X[20:]  # Skip NaN rows
        if len(X) == 0:
            return "low_vol_range"

        if self.scaler is not None:
            X = self.scaler.transform(X)

        states = self.model.predict(X)
        current_state = int(states[-1])
        mapped = self.state_map.get(current_state, current_state)
        regime = REGIME_NAMES.get(mapped, "low_vol_range")
        return regime

    def predict_all(self, df: pd.DataFrame) -> list[str]:
        """Predict regime for every row in df. Returns list of regime name strings.

        Handles the 20-row NaN offset from _build_features by padding the
        first 20 entries with the earliest detected regime.
        Requires df to have 'close' column (and optionally 'vix', 'volume').
        """
        if self.model is None:
            if not self.load():
                return ["low_vol_range"] * len(df)

        X = self._build_features(df)
        if len(X) == 0:
            return ["low_vol_range"] * len(df)

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        # _build_features returns len(df) rows but first ~20 have NaN-derived
        # values (filled with 0). HMM predictions on those are unreliable.
        # We predict on all rows but mark the first 20 as matching row 20's regime.
        states = self.model.predict(X_scaled)
        regimes = []
        for s in states:
            mapped = self.state_map.get(int(s), int(s))
            regimes.append(REGIME_NAMES.get(mapped, "low_vol_range"))

        # Overwrite first 20 with the first reliable regime
        if len(regimes) > 20:
            for i in range(20):
                regimes[i] = regimes[20]

        return regimes


    def predict_with_probs(self, df: pd.DataFrame) -> dict:
        """Predict regime with state probabilities."""
        if self.model is None:
            if not self.load():
                return {"regime": "low_vol_range", "probabilities": {}}

        X = self._build_features(df)
        X = X[20:]
        if len(X) == 0:
            return {"regime": "low_vol_range", "probabilities": {}}

        if self.scaler is not None:
            X = self.scaler.transform(X)

        states = self.model.predict(X)
        posteriors = self.model.predict_proba(X)
        current_probs = posteriors[-1]

        regime = REGIME_NAMES.get(
            self.state_map.get(int(states[-1]), int(states[-1])),
            "low_vol_range"
        )
        probs = {}
        for s in range(self.n_states):
            mapped = self.state_map.get(s, s)
            name = REGIME_NAMES.get(mapped, f"state_{s}")
            probs[name] = round(float(current_probs[s]), 3)

        return {"regime": regime, "probabilities": probs}

    def load(self) -> bool:
        """Load saved HMM model."""
        if not os.path.exists(self._model_path):
            return False
        try:
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            self.model = data["model"]
            self.state_map = data.get("state_map", {})
            self.scaler = data.get("scaler")
            logger.info("HMM regime detector loaded")
            return True
        except Exception as e:
            logger.warning(f"Failed to load HMM: {e}")
            return False

    def get_threshold(self, regime: str) -> float:
        """Get neutral threshold for the given regime."""
        return REGIME_THRESHOLDS.get(regime, 0.003)
