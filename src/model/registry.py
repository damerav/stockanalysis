"""Model Registry — versioned model storage with metadata tracking.

Tracks training metrics, feature sets, deployment status, and enables
rollback / promotion. Uses DbRouter (PostgreSQL primary, SQLite fallback).
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# PostgreSQL-compatible schema (no executescript, uses standard SQL)
REGISTRY_SCHEMA_PG = """
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
    """Model registry backed by PostgreSQL (primary) with SQLite fallback."""

    def __init__(self, config: dict = None):
        from src.data.db_router import DbRouter
        self._router = DbRouter(config)
        self._ensure_table()

    def _ensure_table(self):
        """Create model_registry table if it doesn't exist."""
        try:
            self._router.execute(REGISTRY_SCHEMA_PG)
        except Exception as e:
            logger.warning(f"Registry table creation: {e}")

    def register(self, metrics: dict, model_path: str,
                 feature_names: list[str] = None,
                 model_type: str = "xgboost",
                 status: str = "auto") -> str:
        """Register a newly trained model with its metrics.

        Args:
            metrics: Training metrics dict from SPYPredictor.train()
            model_path: Path to saved model file
            feature_names: List of feature column names used
            model_type: Model type identifier
            status: Deployment status — 'auto' (default: active unless gated),
                    'candidate' (backtest model awaiting promotion), or explicit status.

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

        top_features = json.dumps(metrics.get("top_features", []))

        # Determine deployment status
        if status == "auto":
            deploy_status = "gated" if metrics.get("gated") else "active"
        else:
            deploy_status = status

        self._router.execute(
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
             deploy_status,
             now),
        )

        # Retire previous active models of same type (only if new model is active)
        if deploy_status == "active":
            self._router.execute(
                """UPDATE model_registry SET deployment_status = 'retired'
                   WHERE model_type = ? AND model_id != ? AND deployment_status = 'active'""",
                (model_type, model_id),
            )

        logger.info(f"Model registered: {model_id} ({model_type}, "
                    f"status={deploy_status}, acc={metrics.get('accuracy', 0):.3f})")
        return model_id

    def get_active(self, model_type: str = "xgboost") -> Optional[dict]:
        """Get the currently active model metadata."""
        df = self._router.query(
            """SELECT * FROM model_registry
               WHERE model_type = ? AND deployment_status = 'active'
               ORDER BY created_at DESC LIMIT 1""",
            (model_type,),
        )
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    def get_history(self, model_type: str = "xgboost",
                    limit: int = 20) -> list[dict]:
        """Get recent model training history."""
        df = self._router.query(
            """SELECT model_id, training_date, val_accuracy, test_accuracy,
                      feature_count, gated, deployment_status, created_at
               FROM model_registry
               WHERE model_type = ?
               ORDER BY created_at DESC LIMIT ?""",
            (model_type, limit),
        )
        return df.to_dict("records") if not df.empty else []

    def promote_model(self, model_id: str) -> bool:
        """Promote a candidate model to active champion, retiring others."""
        df = self._router.query(
            "SELECT model_type, model_path FROM model_registry WHERE model_id = ?",
            (model_id,),
        )
        if df.empty:
            logger.error(f"Promotion failed: Model ID {model_id} not found.")
            return False

        model_type = df.iloc[0]["model_type"]
        model_path = df.iloc[0]["model_path"]

        if model_path and not os.path.exists(model_path):
            logger.error(f"Promotion failed: Model file {model_path} not found on disk.")
            return False

        # Retire all active models of this type
        self._router.execute(
            """UPDATE model_registry SET deployment_status = 'retired'
               WHERE model_type = ? AND deployment_status = 'active'""",
            (model_type,),
        )
        # Promote the candidate
        self._router.execute(
            "UPDATE model_registry SET deployment_status = 'active' WHERE model_id = ?",
            (model_id,),
        )
        logger.info(f"Model {model_id} promoted to active champion for '{model_type}'.")
        return True

    def rollback(self, model_id: str) -> bool:
        """Rollback to a specific model version."""
        df = self._router.query(
            "SELECT model_path FROM model_registry WHERE model_id = ?",
            (model_id,),
        )
        if df.empty:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        model_path = df.iloc[0]["model_path"]
        if model_path and not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found on disk")
            return False

        # Retire current active, activate target
        self._router.execute(
            "UPDATE model_registry SET deployment_status = 'retired' WHERE deployment_status = 'active'"
        )
        self._router.execute(
            "UPDATE model_registry SET deployment_status = 'active' WHERE model_id = ?",
            (model_id,),
        )
        logger.info(f"Rolled back to model {model_id}")
        return True

    def get_accuracy_trend(self, model_type: str = "xgboost",
                           limit: int = 30) -> list[dict]:
        """Get accuracy trend over recent models."""
        df = self._router.query(
            """SELECT training_date, val_accuracy, test_accuracy
               FROM model_registry
               WHERE model_type = ? AND gated = 0
               ORDER BY created_at DESC LIMIT ?""",
            (model_type, limit),
        )
        return df.to_dict("records") if not df.empty else []

    def close(self):
        """Close underlying connections."""
        try:
            if hasattr(self._router, '_pg_conn') and self._router._pg_conn:
                self._router._pg_conn.close()
            if hasattr(self._router, '_sqlite') and self._router._sqlite:
                self._router._sqlite.close()
        except Exception:
            pass
