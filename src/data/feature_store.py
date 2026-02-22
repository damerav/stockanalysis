"""Feature Store — SQLite-backed cache for computed feature vectors.

Eliminates redundant recomputation of features for historical dates.
Only computes features for new/missing dates, reducing pipeline time
from O(N) to O(1) for the common daily case.
"""

import logging
import sqlite3
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from src.data.init_db import get_connection, load_config

logger = logging.getLogger(__name__)

STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS feature_cache (
    date TEXT PRIMARY KEY,
    features_json TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    computed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feature_store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class FeatureStore:
    """SQLite-backed feature cache with version tracking."""

    def __init__(self, config: dict = None):
        self.conn = get_connection(config)
        self.conn.executescript(STORE_SCHEMA)
        self.conn.commit()

    def _feature_version(self, feature_cols: list[str]) -> str:
        """Hash of feature column list — invalidates cache on schema change."""
        h = hashlib.md5(",".join(sorted(feature_cols)).encode()).hexdigest()[:12]
        return h

    def get_cached_dates(self, feature_version: str) -> set[str]:
        """Return set of dates that have cached features for this version."""
        rows = self.conn.execute(
            "SELECT date FROM feature_cache WHERE feature_version = ?",
            (feature_version,),
        ).fetchall()
        return {r[0] for r in rows}

    def get_cached_features(self, feature_version: str,
                            feature_cols: list[str]) -> Optional[pd.DataFrame]:
        """Load all cached features for the given version."""
        rows = self.conn.execute(
            "SELECT date, features_json FROM feature_cache WHERE feature_version = ? ORDER BY date",
            (feature_version,),
        ).fetchall()
        if not rows:
            return None

        records = []
        for date_str, fj in rows:
            data = json.loads(fj)
            data["date"] = date_str
            records.append(data)

        df = pd.DataFrame(records)
        # Ensure all expected columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        return df

    def store_features(self, df: pd.DataFrame, feature_cols: list[str],
                       feature_version: str):
        """Cache computed features for given dates."""
        now = datetime.now().isoformat()
        stored = 0
        for _, row in df.iterrows():
            date_str = str(row.get("date", ""))
            if not date_str:
                continue
            # Serialize only the feature columns
            features = {}
            for col in feature_cols:
                val = row.get(col)
                if pd.notna(val):
                    features[col] = round(float(val), 8)
                else:
                    features[col] = None
            self.conn.execute(
                """INSERT OR REPLACE INTO feature_cache
                   (date, features_json, feature_version, computed_at)
                   VALUES (?, ?, ?, ?)""",
                (date_str, json.dumps(features), feature_version, now),
            )
            stored += 1
        self.conn.commit()
        logger.info(f"Feature store: cached {stored} dates (version {feature_version})")

    def get_features(self, feature_cols: list[str],
                     all_dates: list[str] = None,
                     build_fn=None) -> Optional[pd.DataFrame]:
        """Get features, using cache where possible, computing missing dates.

        Args:
            feature_cols: List of feature column names
            all_dates: All dates that should have features (optional)
            build_fn: Callable(missing_dates) -> DataFrame for computing missing features

        Returns:
            DataFrame with all features for all dates
        """
        version = self._feature_version(feature_cols)
        cached_dates = self.get_cached_dates(version)

        if all_dates is not None:
            missing = [d for d in all_dates if d not in cached_dates]
        else:
            missing = []

        # Load cached
        cached_df = self.get_cached_features(version, feature_cols)

        if not missing:
            if cached_df is not None and not cached_df.empty:
                logger.info(f"Feature store: {len(cached_df)} dates from cache, 0 to compute")
                return cached_df
            # No cache and no dates specified — compute everything
            missing = all_dates or []

        if missing and build_fn is not None:
            logger.info(f"Feature store: {len(cached_dates)} cached, {len(missing)} to compute")
            new_df = build_fn(missing)
            if new_df is not None and not new_df.empty:
                self.store_features(new_df, feature_cols, version)
                if cached_df is not None and not cached_df.empty:
                    combined = pd.concat([cached_df, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["date"], keep="last")
                    return combined.sort_values("date").reset_index(drop=True)
                return new_df

        return cached_df

    def invalidate(self, dates: list[str] = None):
        """Invalidate cache for specific dates or all."""
        if dates:
            placeholders = ",".join("?" * len(dates))
            self.conn.execute(
                f"DELETE FROM feature_cache WHERE date IN ({placeholders})", dates
            )
        else:
            self.conn.execute("DELETE FROM feature_cache")
        self.conn.commit()
        logger.info(f"Feature store: invalidated {'all' if not dates else len(dates)} entries")

    def stats(self) -> dict:
        """Return cache statistics."""
        row = self.conn.execute("SELECT COUNT(*), MAX(computed_at) FROM feature_cache").fetchone()
        versions = self.conn.execute(
            "SELECT feature_version, COUNT(*) FROM feature_cache GROUP BY feature_version"
        ).fetchall()
        return {
            "total_cached": row[0] or 0,
            "last_computed": row[1] or "never",
            "versions": {v: c for v, c in versions},
        }

    def close(self):
        if self.conn:
            self.conn.close()
