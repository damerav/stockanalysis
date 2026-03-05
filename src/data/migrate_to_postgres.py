"""Migrate all data from SQLite (spy.db + news.db) to PostgreSQL.

Usage:
    python -m src.data.migrate_to_postgres [--config config.yaml]
"""

import argparse
import logging
import os
import sqlite3

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from src.data.init_db import load_config

logger = logging.getLogger(__name__)


def get_pg_dsn(config: dict) -> str:
    """Build PostgreSQL DSN from config."""
    pg = config.get("database", {}).get("postgres", {})
    host = pg.get("host", "localhost")
    port = pg.get("port", 5432)
    dbname = pg.get("dbname", "stockanalysis")
    user = pg.get("user", "stockapp")
    password = os.environ.get("STOCKAPP_DB_PASSWORD", pg.get("password", ""))
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def _migrate_table(sq_conn, pg_conn, table: str, pk_cols: list = None,
                   date_cols: list = None, skip_cols: list = None):
    """Migrate a single table from SQLite to PostgreSQL."""
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", sq_conn)
    except Exception as e:
        logger.warning(f"  {table}: read failed — {e}")
        return 0

    if df.empty:
        logger.info(f"  {table}: 0 rows (empty)")
        return 0

    if skip_cols:
        df = df.drop(columns=[c for c in skip_cols if c in df.columns], errors="ignore")

    cols = list(df.columns)
    placeholders = ", ".join(["%s"] * len(cols))
    col_names = ", ".join(cols)

    # Build ON CONFLICT clause for upsert
    if pk_cols:
        pk_str = ", ".join(pk_cols)
        update_cols = [c for c in cols if c not in pk_cols]
        if update_cols:
            update_str = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
            upsert = f"ON CONFLICT ({pk_str}) DO UPDATE SET {update_str}"
        else:
            upsert = f"ON CONFLICT ({pk_str}) DO NOTHING"
    else:
        upsert = ""

    sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) {upsert}"

    cur = pg_conn.cursor()
    inserted = 0
    batch = []
    for _, row in df.iterrows():
        values = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                values.append(None)
            else:
                values.append(v)
        batch.append(tuple(values))

        if len(batch) >= 500:
            try:
                cur.executemany(sql, batch)
                inserted += len(batch)
            except Exception as e:
                pg_conn.rollback()
                # Try one by one
                for b in batch:
                    try:
                        cur.execute(sql, b)
                        inserted += 1
                    except Exception:
                        pg_conn.rollback()
            batch = []

    if batch:
        try:
            cur.executemany(sql, batch)
            inserted += len(batch)
        except Exception:
            pg_conn.rollback()
            for b in batch:
                try:
                    cur.execute(sql, b)
                    inserted += 1
                except Exception:
                    pg_conn.rollback()

    pg_conn.commit()
    cur.close()
    logger.info(f"  {table}: {inserted}/{len(df)} rows migrated")
    return inserted


def migrate(config: dict = None):
    """Migrate all data from SQLite to PostgreSQL."""
    if config is None:
        config = load_config()

    dsn = get_pg_dsn(config)
    spy_path = config.get("database", {}).get("path", "./data/spy.db")
    news_path = config.get("news_pipeline", {}).get("db_path", "./data/news.db")

    logger.info(f"Connecting to PostgreSQL: {dsn.split('password=')[0]}...")
    pg = psycopg2.connect(dsn)
    pg.autocommit = False

    results = {}

    # --- Migrate spy.db tables ---
    if os.path.exists(spy_path):
        logger.info(f"\n=== Migrating spy.db ({spy_path}) ===")
        sq = sqlite3.connect(spy_path)

        tables = {
            "prices": {"pk": ["date"]},
            "technicals": {"pk": ["date"]},
            "macro": {"pk": ["date"]},
            "intraday_bars": {"pk": ["timestamp", "ticker"]},
            "options_chain": {"pk": ["date", "contract_symbol"]},
            "daily_sentiment": {"pk": ["date"]},
            "predictions": {"pk": ["date"]},
            "options_analytics": {"pk": ["date"]},
            "intraday_features": {"pk": ["date"]},
            "performance": {"pk": ["date"]},
            "earnings_calendar": {"pk": ["date", "ticker"]},
            "fed_communications": {"pk": ["date"]},
            "users": {"pk": ["username"]},
            "news": {"pk": None, "skip_cols": ["id"]},
        }

        for table, opts in tables.items():
            try:
                count = _migrate_table(
                    sq, pg, table,
                    pk_cols=opts.get("pk"),
                    skip_cols=opts.get("skip_cols"),
                )
                results[f"spy.{table}"] = count
            except Exception as e:
                logger.error(f"  {table}: FAILED — {e}")
                results[f"spy.{table}"] = f"ERROR: {e}"

        sq.close()
    else:
        logger.warning(f"spy.db not found at {spy_path}")

    # --- Migrate news.db tables ---
    if os.path.exists(news_path):
        logger.info(f"\n=== Migrating news.db ({news_path}) ===")
        nq = sqlite3.connect(news_path)

        # raw_articles (without embedding column — that gets added later)
        try:
            df = pd.read_sql_query("SELECT * FROM raw_articles", nq)
            if not df.empty:
                # Drop SQLite auto-increment id, let PostgreSQL generate new ones
                if "id" in df.columns:
                    df = df.drop(columns=["id"])

                cols = list(df.columns)
                placeholders = ", ".join(["%s"] * len(cols))
                col_names = ", ".join(cols)
                sql = f"INSERT INTO raw_articles ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

                cur = pg.cursor()
                batch = []
                inserted = 0
                for _, row in df.iterrows():
                    values = tuple(None if pd.isna(row[c]) else row[c] for c in cols)
                    batch.append(values)
                    if len(batch) >= 500:
                        cur.executemany(sql, batch)
                        inserted += len(batch)
                        batch = []
                if batch:
                    cur.executemany(sql, batch)
                    inserted += len(batch)
                pg.commit()
                cur.close()
                logger.info(f"  raw_articles: {inserted}/{len(df)} rows migrated")
                results["news.raw_articles"] = inserted
            else:
                results["news.raw_articles"] = 0
        except Exception as e:
            logger.error(f"  raw_articles: FAILED — {e}")
            pg.rollback()
            results["news.raw_articles"] = f"ERROR: {e}"

        # finbert_cache — need to map old article_ids to new ones
        try:
            fb_df = pd.read_sql_query("SELECT * FROM finbert_cache", nq)
            if not fb_df.empty:
                # Get old article headlines for matching
                old_articles = pd.read_sql_query(
                    "SELECT id, headline FROM raw_articles", nq
                )
                old_map = dict(zip(old_articles["id"], old_articles["headline"]))

                # Get new article ids from PostgreSQL
                cur = pg.cursor()
                cur.execute("SELECT id, headline FROM raw_articles")
                new_rows = cur.fetchall()
                new_map = {}
                for nid, headline in new_rows:
                    new_map[headline] = nid

                mapped = 0
                for _, row in fb_df.iterrows():
                    old_id = int(row["article_id"])
                    headline = old_map.get(old_id)
                    if headline and headline in new_map:
                        new_id = new_map[headline]
                        try:
                            cur.execute(
                                "INSERT INTO finbert_cache (article_id, fb_positive, fb_negative, fb_neutral) "
                                "VALUES (%s, %s, %s, %s) ON CONFLICT (article_id) DO NOTHING",
                                (new_id,
                                 None if pd.isna(row["fb_positive"]) else float(row["fb_positive"]),
                                 None if pd.isna(row["fb_negative"]) else float(row["fb_negative"]),
                                 None if pd.isna(row["fb_neutral"]) else float(row["fb_neutral"]))
                            )
                            mapped += 1
                        except Exception:
                            pg.rollback()
                pg.commit()
                cur.close()
                logger.info(f"  finbert_cache: {mapped}/{len(fb_df)} rows migrated")
                results["news.finbert_cache"] = mapped
            else:
                results["news.finbert_cache"] = 0
        except Exception as e:
            logger.error(f"  finbert_cache: FAILED — {e}")
            pg.rollback()
            results["news.finbert_cache"] = f"ERROR: {e}"

        nq.close()
    else:
        logger.warning(f"news.db not found at {news_path}")

    # --- Validate ---
    logger.info("\n=== Validation ===")
    cur = pg.cursor()
    all_tables = [
        "prices", "technicals", "macro", "intraday_bars", "options_chain",
        "daily_sentiment", "predictions", "options_analytics", "intraday_features",
        "performance", "earnings_calendar", "fed_communications", "users",
        "news", "raw_articles", "finbert_cache",
    ]
    for t in all_tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            count = cur.fetchone()[0]
            logger.info(f"  {t}: {count} rows")
        except Exception:
            pg.rollback()
            logger.info(f"  {t}: table not found")
    cur.close()
    pg.close()

    logger.info("\n=== Migration Summary ===")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Migrate SQLite to PostgreSQL")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    migrate(config)
