"""Feature Store — PostgreSQL-backed cache for computed feature vectors.

Eliminates redundant recomputation of features for historical dates.
Only computes features for new/missing dates, reducing pipeline time
from O(N) to O(1) for the common daily case.
"""

import logging
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from src.data.db_router import get_router

logger = logging.getLogger(__name__)

STORE_SCHEMA_PG = """
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
    """PostgreSQL-backed feature cache with version tracking."""

    def __init__(self, config: dict = None):
        self.router = get_router(config)
        # Ensure tables exist
        if self.router.using_postgres:
            pg = self.router.get_pg()
            cur = pg.cursor()
            for stmt in STORE_SCHEMA_PG.strip().split(';'):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)
            cur.close()
        else:
            logger.warning("FeatureStore: PostgreSQL unavailable")

    def _feature_version(self, feature_cols: list[str]) -> str:
        """Hash of feature column list — invalidates cache on schema change."""
        h = hashlib.md5(",".join(sorted(feature_cols)).encode()).hexdigest()[:12]
        return h

    def get_cached_dates(self, feature_version: str) -> set[str]:
        """Return set of dates that have cached features for this version."""
        df = self.router.query(
            "SELECT date FROM feature_cache WHERE feature_version = ?",
            (feature_version,),
        )
        return set(df["date"].tolist()) if not df.empty else set()

    def get_cached_features(self, feature_version: str,
                            feature_cols: list[str]) -> Optional[pd.DataFrame]:
        """Load all cached features for the given version."""
        df = self.router.query(
            "SELECT date, features_json FROM feature_cache WHERE feature_version = ? ORDER BY date",
            (feature_version,),
        )
        if df.empty:
            return None

        records = []
        for _, row in df.iterrows():
            data = json.loads(row["features_json"])
            data["date"] = row["date"]
            records.append(data)

        result = pd.DataFrame(records)
        for col in feature_cols:
            if col not in result.columns:
                result[col] = np.nan
        return result

    def store_features(self, df: pd.DataFrame, feature_cols: list[str],
                       feature_version: str):
        """Cache computed features for given dates."""
        now = datetime.now().isoformat()
        stored = 0
        for _, row in df.iterrows():
            date_str = str(row.get("date", ""))
            if not date_str:
                continue
            features = {}
            for col in feature_cols:
                val = row.get(col)
                if pd.notna(val):
                    features[col] = round(float(val), 8)
                else:
                    features[col] = None
            self.router.execute(
                """INSERT OR REPLACE INTO feature_cache
                   (date, features_json, feature_version, computed_at)
                   VALUES (?, ?, ?, ?)""",
                (date_str, json.dumps(features), feature_version, now),
            )
            stored += 1
        logger.info(f"Feature store: cached {stored} dates (version {feature_version})")

    def get_features(self, feature_cols: list[str],
                     all_dates: list[str] = None,
                     build_fn=None) -> Optional[pd.DataFrame]:
        """Get features, using cache where possible, computing missing dates."""
        version = self._feature_version(feature_cols)
        cached_dates = self.get_cached_dates(version)

        if all_dates is not None:
            missing = [d for d in all_dates if d not in cached_dates]
        else:
            missing = []

        cached_df = self.get_cached_features(version, feature_cols)

        if not missing:
            if cached_df is not None and not cached_df.empty:
                logger.info(f"Feature store: {len(cached_df)} dates from cache, 0 to compute")
                return cached_df
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
            for d in dates:
                self.router.execute("DELETE FROM feature_cache WHERE date = ?", (d,))
        else:
            self.router.execute("DELETE FROM feature_cache")
        logger.info(f"Feature store: invalidated {'all' if not dates else len(dates)} entries")

    def stats(self) -> dict:
        """Return cache statistics."""
        df = self.router.query("SELECT COUNT(*) as cnt, MAX(computed_at) as last_at FROM feature_cache")
        versions_df = self.router.query(
            "SELECT feature_version, COUNT(*) as cnt FROM feature_cache GROUP BY feature_version"
        )
        return {
            "total_cached": int(df.iloc[0]["cnt"]) if not df.empty else 0,
            "last_computed": df.iloc[0]["last_at"] if not df.empty else "never",
            "versions": {row["feature_version"]: int(row["cnt"]) for _, row in versions_df.iterrows()} if not versions_df.empty else {},
        }

    def close(self):
        pass  # Router manages connections
