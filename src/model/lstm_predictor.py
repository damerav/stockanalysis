"""LSTM Price Forecaster — Multi-day price regression using Keras.

Predicts next N days of closing prices using a sequence of technical features.
Inspired by SevilayMuni/stock-prediction-web-app.
"""

import os
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy imports for TensorFlow/Keras (heavy dependency)
_tf = None
_keras = None


def _ensure_keras():
    """Lazy-load TensorFlow/Keras."""
    global _tf, _keras
    if _tf is None:
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            _tf = tf
            _keras = tf.keras
        except ImportError:
            try:
                # Standalone Keras
                import keras
                _keras = keras
            except ImportError:
                raise ImportError(
                    "LSTM predictor requires tensorflow or keras. "
                    "Install with: pip install tensorflow"
                )
    return _keras


class LSTMPredictor:
    """Keras LSTM model for multi-day price forecasting.

    Scikit-learn compatible interface with fit/predict/save/load.
    """

    def __init__(self, n_past: int = 21, n_future: int = 5,
                 epochs: int = 50, batch_size: int = 32,
                 feature_list: list = None):
        self.n_past = n_past
        self.n_future = n_future
        self.epochs = epochs
        self.batch_size = batch_size
        self.feature_list = feature_list or ["close"]
        self.model = None
        self.scaler = None
        self.trained = False
        self.history = None

    def _build_model(self, n_features: int):
        """Build LSTM architecture."""
        keras = _ensure_keras()
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True,
                              input_shape=(self.n_past, n_features)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(self.n_future),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def _prepare_sequences(self, data: np.ndarray):
        """Create sliding window sequences for LSTM input."""
        X, y = [], []
        close_idx = 0  # close is always first feature
        for i in range(self.n_past, len(data) - self.n_future + 1):
            X.append(data[i - self.n_past:i])
            y.append(data[i:i + self.n_future, close_idx])
        return np.array(X), np.array(y)

    def _compute_extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived features needed by the LSTM."""
        df = df.copy()
        # Garman-Klass volatility
        if "garman_klass_vol" in self.feature_list and "garman_klass_vol" not in df.columns:
            log_hl = (np.log(df["high"] / df["low"])) ** 2
            log_co = (np.log(df["close"] / df["open"])) ** 2
            df["garman_klass_vol"] = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20).mean().apply(np.sqrt)
        # Dollar volume
        if "dollar_volume" in self.feature_list and "dollar_volume" not in df.columns:
            df["dollar_volume"] = df["close"] * df["volume"]
        # OBV
        if "obv" in self.feature_list and "obv" not in df.columns:
            direction = np.sign(df["close"].diff()).fillna(0)
            df["obv"] = (df["volume"].astype(float) * direction).cumsum()
        # 3-day MA
        if "ma_3_days" in self.feature_list and "ma_3_days" not in df.columns:
            df["ma_3_days"] = df["close"].rolling(3).mean()
        return df

    def fit(self, df: pd.DataFrame, verbose: int = 0) -> dict:
        """Train the LSTM on a price DataFrame.

        Args:
            df: DataFrame with at minimum: date, open, high, low, close, volume
            verbose: Keras verbosity level

        Returns:
            Training metrics dict
        """
        from sklearn.preprocessing import MinMaxScaler

        df = self._compute_extra_features(df)

        # Select features (ensure close is first)
        cols = ["close"] + [c for c in self.feature_list if c != "close" and c in df.columns]
        data = df[cols].dropna().values.astype(np.float64)

        if len(data) < self.n_past + self.n_future + 10:
            logger.warning(f"Insufficient data ({len(data)} rows) for LSTM training")
            return {"error": "insufficient_data", "rows": len(data)}

        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(data)

        X, y = self._prepare_sequences(scaled)
        if len(X) == 0:
            return {"error": "no_sequences"}

        # Scale targets back to close-only scaler range
        self.model = self._build_model(len(cols))
        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=verbose,
        )
        self.trained = True
        self.history = {
            "loss": history.history["loss"][-1],
            "val_loss": history.history.get("val_loss", [0])[-1],
            "epochs": self.epochs,
            "samples": len(X),
        }
        logger.info(f"LSTM trained: loss={self.history['loss']:.6f}, "
                     f"val_loss={self.history['val_loss']:.6f}")
        return self.history

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate n_future day forecast from latest data.

        Args:
            df: DataFrame with OHLCV data (uses last n_past rows)

        Returns:
            DataFrame with columns: day, predicted_close
        """
        if not self.trained or self.model is None:
            logger.warning("Model not trained — cannot predict")
            return pd.DataFrame()

        df = self._compute_extra_features(df)
        cols = ["close"] + [c for c in self.feature_list if c != "close" and c in df.columns]
        data = df[cols].dropna().values.astype(np.float64)

        if len(data) < self.n_past:
            return pd.DataFrame()

        recent = data[-self.n_past:]
        scaled = self.scaler.transform(recent)
        X = scaled.reshape(1, self.n_past, len(cols))

        pred_scaled = self.model.predict(X, verbose=0)[0]

        # Inverse transform: create dummy array with close in first column
        dummy = np.zeros((len(pred_scaled), len(cols)))
        dummy[:, 0] = pred_scaled
        pred_prices = self.scaler.inverse_transform(dummy)[:, 0]

        last_date = pd.to_datetime(df["date"].iloc[-1])
        dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                               periods=self.n_future)

        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "day": range(1, self.n_future + 1),
            "predicted_close": pred_prices,
        })

    def save(self, path: str = "./models/lstm_predictor"):
        """Save model weights and scaler."""
        os.makedirs(path, exist_ok=True)
        if self.model:
            self.model.save(os.path.join(path, "model.keras"))
        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "n_past": self.n_past,
                "n_future": self.n_future,
                "feature_list": self.feature_list,
                "trained": self.trained,
                "history": self.history,
                "saved_at": datetime.now().isoformat(),
            }, f)
        logger.info(f"LSTM model saved to {path}")

    def load(self, path: str = "./models/lstm_predictor"):
        """Load model weights and scaler."""
        keras = _ensure_keras()
        model_path = os.path.join(path, "model.keras")
        meta_path = os.path.join(path, "meta.pkl")
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.scaler = meta["scaler"]
        self.n_past = meta["n_past"]
        self.n_future = meta["n_future"]
        self.feature_list = meta["feature_list"]
        self.trained = meta["trained"]
        self.history = meta.get("history")
        logger.info(f"LSTM model loaded from {path}")
