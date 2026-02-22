"""Model Registry — versioned model storage with metadata tracking.

Tracks training metrics, feature sets, deployment status, and enables
rollback to prior models. Stored as a SQLite table alongside spy.db.
"""

import os
import json
import uuid
import logging
import sqlite3
from datetime import datetime
from typing import Optional

from src.data.init_db import get_connection

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    training_date TEXT NOT NULL,
    model_path TEXT NOT NULL,
    model_type TEXT DEFAULT 'xgboost',
    training_window INTEGER,
    feature_count INTEGER,
    feature_set_hash TEXT,
    val_accuracy REAL,
    test_accuracy REAL,
    calibration_brier REAL,
    train_size INTEGER,
    val_size INTEGER,
    test_size INTEGER,
    embargo_days INTEGER,
    gated INTEGER DEFAULT 0,
    gate_reason TEXT,
    calibrated INTEGER DEFAULT 0,
    best_iteration INTEGER,
    top_features_json TEXT,
    deployment_status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    notes TEXT
);
"""


class ModelRegistry:
    """Lightweight model registry backed by SQLite."""

    def __init__(self, config: dict = None):
        self.conn = get_connection(config)
        self.conn.executescript(REGISTRY_SCHEMA)
        self.conn.commit()

    def register(self, metrics: dict, model_path: str,
                 feature_names: list[str] = None,
                 model_type: str = "xgboost") -> str:
        """Register a newly trained model with its metrics.

        Args:
            metrics: Training metrics dict from SPYPredictor.train()
            model_path: Path to saved model file
            feature_names: List of feature column names used
            model_type: Model type identifier

        Returns:
            model_id (UUID string)
        """
        model_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        training_date = datetime.now().strftime("%Y-%m-%d")

        # Feature set hash
        import hashlib
        feat_hash = ""
        if feature_names:
            feat_hash = hashlib.md5(",".join(sorted(feature_names)).encode()).hexdigest()[:12]

        # Top features JSON
        top_features = json.dumps(metrics.get("top_features", []))

        self.conn.execute(
            """INSERT INTO model_registry
               (model_id, training_date, model_path, model_type,
                training_window, feature_count, feature_set_hash,
                val_accuracy, test_accuracy, calibration_brier,
                train_size, val_size, test_size, embargo_days,
                gated, gate_reason, calibrated, best_iteration,
                top_features_json, deployment_status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (model_id, training_date, model_path, model_type,
             metrics.get("train_size", 0) + metrics.get("val_size", 0),
             len(feature_names) if feature_names else 0,
             feat_hash,
             metrics.get("accuracy"), metrics.get("test_accuracy"),
             metrics.get("brier_score"),
             metrics.get("train_size"), metrics.get("val_size"),
             metrics.get("test_size"), metrics.get("embargo_days", 5),
             int(metrics.get("gated", False)),
             metrics.get("gate_reason", ""),
             int(metrics.get("calibrated", False)),
             metrics.get("best_iteration"),
             top_features,
             "gated" if metrics.get("gated") else "active",
             now),
        )
        self.conn.commit()

        # Retire previous active models of same type
        if not metrics.get("gated"):
            self.conn.execute(
                """UPDATE model_registry SET deployment_status = 'retired'
                   WHERE model_type = ? AND model_id != ? AND deployment_status = 'active'""",
                (model_type, model_id),
            )
            self.conn.commit()

        logger.info(f"Model registered: {model_id} ({model_type}, "
                    f"acc={metrics.get('accuracy', 0):.3f})")
        return model_id

    def get_active(self, model_type: str = "xgboost") -> Optional[dict]:
        """Get the currently active model metadata."""
        row = self.conn.execute(
            """SELECT * FROM model_registry
               WHERE model_type = ? AND deployment_status = 'active'
               ORDER BY created_at DESC LIMIT 1""",
            (model_type,),
        ).fetchone()
        return dict(row) if row else None

    def get_history(self, model_type: str = "xgboost",
                    limit: int = 20) -> list[dict]:
        """Get recent model training history."""
        rows = self.conn.execute(
            """SELECT model_id, training_date, val_accuracy, test_accuracy,
                      feature_count, gated, deployment_status, created_at
               FROM model_registry
               WHERE model_type = ?
               ORDER BY created_at DESC LIMIT ?""",
            (model_type, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def rollback(self, model_id: str) -> bool:
        """Rollback to a specific model version."""
        row = self.conn.execute(
            "SELECT model_path FROM model_registry WHERE model_id = ?",
            (model_id,),
        ).fetchone()
        if not row:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        if not os.path.exists(row[0]):
            logger.warning(f"Model file {row[0]} not found on disk")
            return False

        # Retire current active, activate target
        self.conn.execute(
            "UPDATE model_registry SET deployment_status = 'retired' WHERE deployment_status = 'active'"
        )
        self.conn.execute(
            "UPDATE model_registry SET deployment_status = 'active' WHERE model_id = ?",
            (model_id,),
        )
        self.conn.commit()
        logger.info(f"Rolled back to model {model_id}")
        return True

    def get_accuracy_trend(self, model_type: str = "xgboost",
                           limit: int = 30) -> list[dict]:
        """Get accuracy trend over recent models."""
        rows = self.conn.execute(
            """SELECT training_date, val_accuracy, test_accuracy
               FROM model_registry
               WHERE model_type = ? AND gated = 0
               ORDER BY created_at DESC LIMIT ?""",
            (model_type, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if self.conn:
            self.conn.close()
