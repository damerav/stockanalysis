"""BiLSTM Base Learner — Temporal sequence model for stacking ensemble.

Captures temporal dependencies that tree-based models miss.
Operates on the last `seq_len` days of features as a sequence.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class BiLSTMNet(nn.Module):
    """Bidirectional LSTM for 3-class direction prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]  # last timestep
        out = self.dropout(last)
        return self.fc(out)


class BiLSTMClassifier:
    """Sklearn-compatible wrapper around BiLSTM for use in stacking ensemble."""

    def __init__(self, input_dim: int = 50, seq_len: int = 20,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, lr: float = 1e-3,
                 epochs: int = 50, batch_size: int = 64,
                 device: str = "auto"):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = 3
        self.model: Optional[BiLSTMNet] = None

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"BiLSTM device: {self.device} "
                     f"(CUDA available: {torch.cuda.is_available()}"
                     f"{', ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''})")

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert flat feature matrix to overlapping sequences.

        Input:  (n_samples, n_features)
        Output: (n_samples - seq_len + 1, seq_len, n_features)
        """
        n = len(X)
        if n < self.seq_len:
            # Pad with zeros at the start
            pad = np.zeros((self.seq_len - n, X.shape[1]))
            X = np.vstack([pad, X])
            return X.reshape(1, self.seq_len, -1)

        seqs = []
        for i in range(self.seq_len - 1, n):
            seqs.append(X[i - self.seq_len + 1: i + 1])
        return np.array(seqs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train BiLSTM on feature sequences."""
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        self.input_dim = X.shape[1]

        seqs = self._build_sequences(X)
        # Align y to sequences (drop first seq_len-1 samples)
        y_seq = y[self.seq_len - 1:] if len(y) > self.seq_len else y[-len(seqs):]

        if len(seqs) != len(y_seq):
            min_len = min(len(seqs), len(y_seq))
            seqs = seqs[-min_len:]
            y_seq = y_seq[-min_len:]

        X_t = torch.FloatTensor(seqs).to(self.device)
        y_t = torch.LongTensor(y_seq.astype(int)).to(self.device)

        self.model = BiLSTMNet(
            self.input_dim, self.hidden_dim, self.num_layers,
            self.dropout, self.num_classes,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug(f"BiLSTM epoch {epoch+1}/{self.epochs}, loss={total_loss/len(loader):.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            n = max(1, len(X) - self.seq_len + 1) if len(X) >= self.seq_len else 1
            return np.full((n, self.num_classes), 1.0 / self.num_classes)

        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        seqs = self._build_sequences(X)
        X_t = torch.FloatTensor(seqs).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
