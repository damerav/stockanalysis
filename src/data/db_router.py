"""Database Router - Routes queries to PostgreSQL (primary) with SQLite fallback.

PostgreSQL is the primary database with pgvector support for semantic search.
SQLite is kept as a fallback for environments without PostgreSQL.
"""

import os
import re
import sqlite3
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_TABLE_PKS = {
    "prices": "date", "technicals": "date", "macro": "date",
    "intraday_bars": "timestamp, ticker", "options_chain": "date, contract_symbol",
    "daily_sentiment": "date", "predictions": "date", "options_analytics": "date",
    "intraday_features": "date", "performance": "date", "feature_cache": "date",
    "earnings_calendar": "date, ticker", "fed_communications": "date",
    "users": "username", "news": "id", "raw_articles": "id",
    "finbert_cache": "article_id", "model_registry": "id",
}

ANALYTICS_TABLES = {"prices", "technicals", "macro", "intraday_bars", "options_chain"}


def _get_sqlite_path(config: dict = None) -> str:
    if config:
        return config.get("database", {}).get("path", "./data/spy.db")
    return "./data/spy.db"


def _get_pg_config(config: dict = None) -> Optional[dict]:
    if not config:
        return None
    pg = config.get("database", {}).get("postgres")
    if not pg:
        return None
    if pg.get("dbname") and pg.get("user"):
        return pg
    return None


def _sqlite_sql_to_pg(sql: str, params: tuple = None) -> tuple[str, tuple]:
    """Convert SQLite SQL to PostgreSQL (INSERT OR REPLACE -> ON CONFLICT, ? -> %s)."""
    converted = sql
    if re.search(r'INSERT\s+OR\s+IGNORE', converted, re.IGNORECASE):
        converted = re.sub(r'INSERT\s+OR\s+IGNORE', 'INSERT', converted, flags=re.IGNORECASE)
        converted = converted.rstrip().rstrip(';') + " ON CONFLICT DO NOTHING"
    elif re.search(r'INSERT\s+OR\s+REPLACE', converted, re.IGNORECASE):
        converted = re.sub(r'INSERT\s+OR\s+REPLACE', 'INSERT', converted, flags=re.IGNORECASE)
        m = re.search(r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)', converted, re.IGNORECASE)
        if m:
            table = m.group(1).lower()
            cols = [c.strip() for c in m.group(2).split(',')]
            pk_str = _TABLE_PKS.get(table, "")
            pk_cols = [c.strip() for c in pk_str.split(',')] if pk_str else []
            update_cols = [c for c in cols if c not in pk_cols]
            converted = converted.rstrip().rstrip(';')
            if pk_cols and update_cols:
                set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
                converted += f" ON CONFLICT ({pk_str}) DO UPDATE SET {set_clause}"
            elif pk_cols:
                converted += f" ON CONFLICT ({pk_str}) DO NOTHING"
    converted = converted.replace('?', '%s')
    return converted, params


class DbRouter:
    """Routes queries to PostgreSQL (primary) with SQLite fallback."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._sqlite_path = _get_sqlite_path(config)
        self._pg_config = _get_pg_config(config)
        self._pg_conn = None
        self._sqlite = None
        os.makedirs(os.path.dirname(self._sqlite_path) or ".", exist_ok=True)
        if self._pg_config:
            try:
                import psycopg2
                self._pg_conn = psycopg2.connect(
                    host=self._pg_config.get("host", "localhost"),
                    port=self._pg_config.get("port", 5432),
                    dbname=self._pg_config["dbname"],
                    user=self._pg_config["user"],
                    password=self._pg_config.get("password", ""),
                )
                self._pg_conn.autocommit = True
                logger.info(f"DbRouter: PostgreSQL connected ({self._pg_config['dbname']})")
            except Exception as e:
                logger.warning(f"PostgreSQL unavailable, falling back to SQLite: {e}")
                self._pg_conn = None
        self._sqlite = sqlite3.connect(self._sqlite_path, timeout=10)
        self._sqlite.row_factory = sqlite3.Row
        self._sqlite.execute("PRAGMA journal_mode=WAL")
        self._sqlite.execute("PRAGMA busy_timeout=5000")
        backend = "PostgreSQL" if self._pg_conn else "SQLite"
        logger.info(f"DbRouter ready: primary={backend}, SQLite={self._sqlite_path}")

    @property
    def using_postgres(self) -> bool:
        return self._pg_conn is not None

    def get_pg(self):
        return self._pg_conn

    def get_sqlite(self) -> sqlite3.Connection:
        return self._sqlite

    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        if self._pg_conn:
            try:
                pg_sql, pg_params = _sqlite_sql_to_pg(sql, params)
                return pd.read_sql_query(pg_sql, self._pg_conn,
                                         params=pg_params if pg_params else None)
            except Exception as e:
                logger.warning(f"PostgreSQL query failed, falling back: {e}")
        try:
            return pd.read_sql_query(sql, self._sqlite,
                                     params=params if params else None)
        except Exception as e:
            logger.error(f"SQLite query also failed: {e}")
            return pd.DataFrame()

    def execute(self, sql: str, params: tuple = None):
        if self._pg_conn:
            try:
                pg_sql, pg_params = _sqlite_sql_to_pg(sql, params)
                cur = self._pg_conn.cursor()
                cur.execute(pg_sql, pg_params)
                cur.close()
                return
            except Exception as e:
                logger.warning(f"PostgreSQL execute failed, falling back: {e}")
        if params:
            self._sqlite.execute(sql, params)
        else:
            self._sqlite.execute(sql)
        self._sqlite.commit()

    def read_analytics(self, sql: str, params: tuple = None) -> pd.DataFrame:
        return self.query(sql, params)

    def write_analytics(self, sql: str, params: tuple = None):
        self.execute(sql, params)

    def execute_analytics(self, sql: str, params: tuple = None):
        self.execute(sql, params)

    def get_analytics_conn(self):
        if self._pg_conn:
            return self._pg_conn
        return self._sqlite

    def read_sqlite(self, sql: str, params: tuple = None) -> pd.DataFrame:
        try:
            if params:
                return pd.read_sql_query(sql, self._sqlite, params=params)
            return pd.read_sql_query(sql, self._sqlite)
        except Exception as e:
            logger.error(f"SQLite read error: {e}")
            return pd.DataFrame()

    def read_feature_join(self, date: str = None) -> pd.DataFrame:
        date_filter = f" WHERE p.date = '{date}'" if date else ""
        sql = f"""
        SELECT p.date, p.open, p.high, p.low, p.close, p.volume,
            t.sma_20, t.sma_50, t.rsi_14, t.macd, t.macd_signal, t.macd_hist,
            t.bb_upper, t.bb_lower, t.atr_14,
            m.vix, m.vix_change, m.us10y_yield, m.dxy, m.fed_funds, m.gold, m.crude,
            m.vix9d, m.vix3m, m.vix6m, m.vvix, m.skew_index,
            m.hy_spread, m.tlt_spy_ratio, m.eem_spy_ratio,
            m.copper_gold_ratio, m.xlk_xlf_ratio, m.xlk_xle_ratio
        FROM prices p
        LEFT JOIN technicals t ON p.date = t.date
        LEFT JOIN macro m ON p.date = m.date
        {date_filter} ORDER BY p.date"""
        df = self.query(sql)
        if df.empty:
            return pd.DataFrame()
        dates_list = df["date"].tolist()
        if not dates_list:
            return df
        if self._pg_conn:
            try:
                df_sent = pd.read_sql_query(
                    "SELECT date, score as sentiment_score, confidence as sentiment_confidence, "
                    "article_count, positive_ratio, negative_ratio, macro_sentiment, "
                    "earnings_sentiment, geopolitical_sentiment, technical_sentiment, "
                    "sentiment_dispersion, sentiment_velocity "
                    "FROM daily_sentiment WHERE date = ANY(%s)", self._pg_conn, params=(dates_list,))
                df_intra = pd.read_sql_query(
                    "SELECT date, vwap_spread, intraday_momentum, intraday_range, volume_ratio "
                    "FROM intraday_features WHERE date = ANY(%s)", self._pg_conn, params=(dates_list,))
                df_opts = pd.read_sql_query(
                    "SELECT date, put_call_ratio, max_pain, iv_skew, gex, "
                    "vanna_exposure, charm_exposure, zero_dte_pcr "
                    "FROM options_analytics WHERE date = ANY(%s)", self._pg_conn, params=(dates_list,))
            except Exception:
                df_sent = df_intra = df_opts = pd.DataFrame()
        else:
            ph = ",".join(["?"] * len(dates_list))
            df_sent = pd.read_sql_query(
                f"SELECT date, score as sentiment_score, confidence as sentiment_confidence, "
                f"article_count, positive_ratio, negative_ratio, macro_sentiment, "
                f"earnings_sentiment, geopolitical_sentiment, technical_sentiment, "
                f"sentiment_dispersion, sentiment_velocity "
                f"FROM daily_sentiment WHERE date IN ({ph})", self._sqlite, params=dates_list)
            df_intra = pd.read_sql_query(
                f"SELECT date, vwap_spread, intraday_momentum, intraday_range, volume_ratio "
                f"FROM intraday_features WHERE date IN ({ph})", self._sqlite, params=dates_list)
            df_opts = pd.read_sql_query(
                f"SELECT date, put_call_ratio, max_pain, iv_skew, gex, "
                f"vanna_exposure, charm_exposure, zero_dte_pcr "
                f"FROM options_analytics WHERE date IN ({ph})", self._sqlite, params=dates_list)
        for src_df, cols in [
            (df_sent, ["sentiment_score", "sentiment_confidence", "article_count",
                       "positive_ratio", "negative_ratio", "macro_sentiment",
                       "earnings_sentiment", "geopolitical_sentiment",
                       "technical_sentiment", "sentiment_dispersion", "sentiment_velocity"]),
            (df_intra, ["vwap_spread", "intraday_momentum", "intraday_range", "volume_ratio"]),
            (df_opts, ["put_call_ratio", "max_pain", "iv_skew", "gex",
                       "vanna_exposure", "charm_exposure", "zero_dte_pcr"]),
        ]:
            if not src_df.empty:
                df = df.merge(src_df, on="date", how="left")
            else:
                for c in cols:
                    df[c] = None
        return df

    def store_embedding(self, article_id: int, embedding: list[float]):
        if not self._pg_conn:
            return
        try:
            cur = self._pg_conn.cursor()
            cur.execute("UPDATE raw_articles SET embedding = %s WHERE id = %s",
                        (str(embedding), article_id))
            cur.close()
        except Exception as e:
            logger.error(f"store_embedding failed: {e}")

    def vector_search(self, embedding: list[float], limit: int = 10,
                      category: str = None, days_back: int = 30) -> pd.DataFrame:
        if not self._pg_conn:
            return pd.DataFrame()
        try:
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            embed_str = str(embedding)
            where = "WHERE embedding IS NOT NULL AND published_at >= %s"
            params = [cutoff]
            if category:
                where += " AND category = %s"
                params.append(category)
            sql = f"""SELECT id, headline, source, category, published_at,
                   sentiment_compound, 1 - (embedding <=> %s::vector) as similarity
            FROM raw_articles {where}
            ORDER BY embedding <=> %s::vector LIMIT %s"""
            params_full = [embed_str] + params + [embed_str, limit]
            return pd.read_sql_query(sql, self._pg_conn, params=params_full)
        except Exception as e:
            logger.error(f"vector_search failed: {e}")
            return pd.DataFrame()

    def close(self):
        if self._pg_conn:
            try:
                self._pg_conn.close()
            except Exception:
                pass
        if self._sqlite:
            try:
                self._sqlite.close()
            except Exception:
                pass
        logger.info("DbRouter connections closed")


_router_instance: Optional[DbRouter] = None


def get_router(config: dict = None) -> DbRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = DbRouter(config)
    return _router_instance


def reset_router():
    global _router_instance
    if _router_instance:
        _router_instance.close()
    _router_instance = None
