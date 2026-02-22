"""Migrate analytics tables from SQLite to DuckDB.

Enhancement 26: One-time migration script that copies historical data from
the 5 analytics tables (prices, technicals, macro, intraday_bars, options_chain)
from SQLite into DuckDB. Validates row counts after migration.

Usage:
    python -m src.data.migrate_to_duckdb [--config config.yaml]
"""

import argparse
import logging
import sqlite3
import os

import duckdb
import pandas as pd

from src.data.init_db import load_config
from src.data.db_router import (
    _get_duckdb_path, _get_sqlite_path, ANALYTICS_TABLES, DUCKDB_SCHEMA,
)

logger = logging.getLogger(__name__)

TABLES_TO_MIGRATE = ["prices", "technicals", "macro", "intraday_bars", "options_chain"]


def migrate(config: dict = None):
    """Copy all rows from SQLite analytics tables into DuckDB."""
    if config is None:
        config = load_config()

    sqlite_path = _get_sqlite_path(config)
    duckdb_path = _get_duckdb_path(config)

    if not os.path.exists(sqlite_path):
        logger.error(f"SQLite database not found at {sqlite_path}")
        return False

    os.makedirs(os.path.dirname(duckdb_path) or ".", exist_ok=True)

    logger.info(f"Migrating from {sqlite_path} → {duckdb_path}")

    # Open connections
    sq = sqlite3.connect(sqlite_path)
    dk = duckdb.connect(duckdb_path)

    # Create DuckDB schema
    for stmt in DUCKDB_SCHEMA.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            dk.execute(stmt)

    results = {}
    for table in TABLES_TO_MIGRATE:
        try:
            # Read from SQLite
            df = pd.read_sql_query(f"SELECT * FROM {table}", sq)
            sqlite_count = len(df)

            if sqlite_count == 0:
                logger.info(f"  {table}: 0 rows (empty, skipping)")
                results[table] = {"sqlite": 0, "duckdb": 0, "ok": True}
                continue

            # Clear existing DuckDB data for this table (idempotent re-run)
            dk.execute(f"DELETE FROM {table}")

            # Insert into DuckDB using DuckDB's fast DataFrame ingestion
            dk.execute(f"INSERT INTO {table} SELECT * FROM df")

            # Validate
            duck_count = dk.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            ok = duck_count == sqlite_count
            status = "✓" if ok else "✗ MISMATCH"
            logger.info(f"  {table}: {sqlite_count} → {duck_count} {status}")
            results[table] = {"sqlite": sqlite_count, "duckdb": duck_count, "ok": ok}

        except Exception as e:
            logger.error(f"  {table}: FAILED — {e}")
            results[table] = {"sqlite": -1, "duckdb": -1, "ok": False, "error": str(e)}

    sq.close()
    dk.close()

    # Summary
    all_ok = all(r["ok"] for r in results.values())
    logger.info(f"\nMigration {'COMPLETE' if all_ok else 'FAILED'}")
    for t, r in results.items():
        logger.info(f"  {t}: SQLite={r['sqlite']} DuckDB={r['duckdb']} {'✓' if r['ok'] else '✗'}")

    return all_ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Migrate analytics tables to DuckDB")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()
    config = load_config(args.config)
    success = migrate(config)
    if not success:
        exit(1)
    print("Migration successful.")
