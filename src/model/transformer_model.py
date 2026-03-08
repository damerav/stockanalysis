"""Transformer Time-Series Model — Attention-based sequence model for stacking ensemble.

Captures long-range temporal dependencies and cross-feature interactions
that LSTM/BiLSTM miss. Uses multi-head self-attention over feature sequences.

Designed as a drop-in replacement/addition alongside BiLSTM in the ensemble.
"""

import logging
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """Transformer encoder for 3-class direction prediction."""

    def __init__(self, input_dim: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.2,
                 num_classes: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Use mean pooling over sequence (more stable than last-token)
        x = x.mean(dim=1)
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)


class TransformerClassifier:
    """Sklearn-compatible wrapper for use in stacking ensemble.

    Mirrors BiLSTMClassifier API: fit(), predict(), predict_proba().
    """

    def __init__(self, input_dim: int = 50, seq_len: int = 20,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.2, lr: float = 5e-4,
                 epochs: int = 50, batch_size: int = 64,
                 device: str = "auto"):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = 3
        self.model: Optional[TransformerPredictor] = None

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Transformer device: {self.device}")

    def _build_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert flat feature matrix to overlapping sequences.

        Input:  (n_samples, n_features)
        Output: (n_samples - seq_len + 1, seq_len, n_features)
        """
        n = len(X)
        if n < self.seq_len:
            pad = np.zeros((self.seq_len - n, X.shape[1]))
            X = np.vstack([pad, X])
            return X.reshape(1, self.seq_len, -1)

        seqs = []
        for i in range(self.seq_len - 1, n):
            seqs.append(X[i - self.seq_len + 1: i + 1])
        return np.array(seqs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train Transformer on feature sequences."""
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        self.input_dim = X.shape[1]

        seqs = self._build_sequences(X)
        y_seq = y[self.seq_len - 1:] if len(y) > self.seq_len else y[-len(seqs):]

        if len(seqs) != len(y_seq):
            min_len = min(len(seqs), len(y_seq))
            seqs = seqs[-min_len:]
            y_seq = y_seq[-min_len:]

        X_t = torch.FloatTensor(seqs).to(self.device)
        y_t = torch.LongTensor(y_seq.astype(int)).to(self.device)

        self.model = TransformerPredictor(
            self.input_dim, self.d_model, self.nhead,
            self.num_layers, self.dim_feedforward,
            self.dropout, self.num_classes,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
        )

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                # Gradient clipping for transformer stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            scheduler.step()

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.debug(f"Transformer early stop at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                logger.debug(f"Transformer epoch {epoch+1}/{self.epochs}, "
                             f"loss={avg_loss:.4f}")

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
