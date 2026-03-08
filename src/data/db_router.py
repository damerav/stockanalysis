"""Database Router - Routes queries to PostgreSQL (primary) with SQLite fallback.

PostgreSQL is the primary database with pgvector support for semantic search.
TimescaleDB extension provides hypertables, continuous aggregates, and compression
for time-series tables when available. Falls back to regular PostgreSQL seamlessly.
SQLite is kept as a fallback for environments without PostgreSQL.
Uses SQLAlchemy engine for pandas read_sql_query to avoid psycopg2 warnings.
"""

import os
import re
import sqlite3
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_TABLE_PKS = {
    "prices": "date", "technicals": "date", "macro": "date",
    "intraday_bars": "timestamp, ticker", "options_chain": "date, contract_symbol",
    "daily_sentiment": "date", "predictions": "date", "options_analytics": "date",
    "intraday_features": "date", "performance": "date", "feature_cache": "date",
    "earnings_calendar": "date, ticker", "fed_communications": "date",
    "users": "username", "news": "id", "raw_articles": "id",
    "finbert_cache": "url_hash", "model_registry": "id",
    "news_features": "date", "feature_store_meta": "key",
    "strategy_rules": "rule_group, rule_key",
    "market_breadth": "date",
    "backtest_results": "date",
    "etf_flows": "date",
    "cot_data": "date",
    "deepseek_scores": "date",
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
        # Allow env var override for password
        if os.environ.get("STOCKAPP_DB_PASSWORD"):
            pg = dict(pg)  # don't mutate original
            pg["password"] = os.environ["STOCKAPP_DB_PASSWORD"]
        elif pg.get("password") in ("FROM_ENCRYPTED_DB", ""):
            # Try secrets manager
            try:
                from src.data.secrets_manager import get_secret
                db_pw = get_secret("db_password")
                if db_pw:
                    pg = dict(pg)
                    pg["password"] = db_pw
            except Exception:
                pass
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


def _pg_to_sqlalchemy(sql: str, params: tuple = None) -> tuple[str, dict]:
    """Convert %s positional params to :p0, :p1, ... named params for SQLAlchemy text()."""
    if not params:
        return sql, None
    sa_sql = sql
    sa_params = {}
    for i, val in enumerate(params):
        sa_sql = sa_sql.replace('%s', f':p{i}', 1)
        sa_params[f'p{i}'] = val
    return sa_sql, sa_params


class DbRouter:
    """Routes queries to PostgreSQL (primary) with SQLite fallback."""

    def __init__(self, config: dict = None):
        if config is None or not config:
            try:
                import yaml
                with open("config.yaml") as f:
                    config = yaml.safe_load(f) or {}
            except Exception:
                config = {}
        self.config = config
        self._sqlite_path = _get_sqlite_path(config)
        self._pg_config = _get_pg_config(config)
        self._pg_conn = None
        self._pg_engine = None
        self._sqlite = None
        if self._pg_config:
            try:
                import psycopg2
                host = self._pg_config.get("host", "localhost")
                port = self._pg_config.get("port", 5432)
                dbname = self._pg_config["dbname"]
                user = self._pg_config["user"]
                password = self._pg_config.get("password", "")
                self._pg_conn = psycopg2.connect(
                    host=host, port=port, dbname=dbname,
                    user=user, password=password,
                )
                self._pg_conn.autocommit = True
                # SQLAlchemy engine for pandas read_sql_query (suppresses psycopg2 warnings)
                self._pg_engine = create_engine(
                    f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}",
                    pool_pre_ping=True,
                )
                logger.info(f"DbRouter: PostgreSQL connected ({dbname})")
            except Exception as e:
                logger.warning(f"PostgreSQL unavailable: {e}")
                self._pg_conn = None
                self._pg_engine = None
        # Only create SQLite connection if PostgreSQL is unavailable
        if not self._pg_conn:
            os.makedirs(os.path.dirname(self._sqlite_path) or ".", exist_ok=True)
            try:
                self._sqlite = sqlite3.connect(self._sqlite_path, timeout=10)
                self._sqlite.row_factory = sqlite3.Row
                self._sqlite.execute("PRAGMA journal_mode=WAL")
                self._sqlite.execute("PRAGMA busy_timeout=5000")
            except sqlite3.DatabaseError:
                logger.warning(f"SQLite file corrupted ({self._sqlite_path}), recreating")
                try:
                    os.remove(self._sqlite_path)
                except OSError:
                    pass
                self._sqlite = sqlite3.connect(self._sqlite_path, timeout=10)
                self._sqlite.row_factory = sqlite3.Row
                self._sqlite.execute("PRAGMA journal_mode=WAL")
                self._sqlite.execute("PRAGMA busy_timeout=5000")
        backend = "PostgreSQL" if self._pg_conn else "SQLite"
        logger.info(f"DbRouter ready: primary={backend}")

        # Detect TimescaleDB
        self._has_timescaledb = False
        if self._pg_conn:
            try:
                cur = self._pg_conn.cursor()
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                row = cur.fetchone()
                if row:
                    self._has_timescaledb = True
                    logger.info(f"DbRouter: TimescaleDB {row[0]} detected")
                cur.close()
            except Exception:
                pass

    @property
    def using_postgres(self) -> bool:
        return self._pg_conn is not None

    @property
    def has_timescaledb(self) -> bool:
        """True if TimescaleDB extension is installed."""
        return self._has_timescaledb

    def get_pg(self):
        return self._pg_conn

    def get_sqlite(self):
        """Get SQLite connection (may be None if PostgreSQL is primary)."""
        return self._sqlite

    def query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        if self._pg_engine:
            try:
                pg_sql, pg_params = _sqlite_sql_to_pg(sql, params)
                sa_sql, sa_params = _pg_to_sqlalchemy(pg_sql, pg_params)
                return pd.read_sql_query(text(sa_sql), self._pg_engine,
                                         params=sa_params if sa_params else None)
            except Exception as e:
                logger.warning(f"PostgreSQL query failed: {e}")
                if not self._sqlite:
                    return pd.DataFrame()
        if self._sqlite:
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
                logger.warning(f"PostgreSQL execute failed: {e}")
                if not self._sqlite:
                    raise
        if self._sqlite:
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
        """Read from SQLite (fallback). Returns empty DataFrame if SQLite unavailable."""
        if not self._sqlite:
            return pd.DataFrame()
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
            t.adx_14, t.cci_20, t.aroon_up, t.aroon_down,
            t.psar_long, t.psar_short, t.dpo_20, t.trix_14,
            t.vortex_pos, t.vortex_neg, t.williams_r, t.mfi_14,
            t.rsi_2, t.rsi_9, t.rsi_21, t.cmo_14, t.ppo,
            t.roc_5, t.roc_21,
            t.kc_upper_20, t.kc_lower_20, t.atr_7, t.atr_21,
            t.donchian_high, t.donchian_low, t.ulcer_14,
            t.cmf_20, t.vwma_20, t.eom_14,
            t.ema_9, t.ema_21, t.ema_200,
            t.hma_20, t.wma_20, t.dema_20, t.tema_20, t.kama_10,
            t.ichi_tenkan, t.ichi_kijun, t.ichi_senkou_a, t.ichi_senkou_b,
            m.vix, m.vix_change, m.us10y_yield, m.dxy, m.fed_funds, m.gold, m.crude,
            m.vix9d, m.vix3m, m.vix6m, m.vvix, m.skew_index,
            m.hy_spread, m.tlt_spy_ratio, m.eem_spy_ratio,
            m.copper_gold_ratio, m.xlk_xlf_ratio, m.xlk_xle_ratio,
            m.us3m_yield, m.yield_curve_10y3m, m.sahm_rule, m.consumer_conf, m.ism_pmi,
            m.xlk, m.xlf, m.xle, m.xlv, m.xli, m.xlu, m.xlb, m.xlp, m.xly, m.xlre,
            m.qqq, m.iwm, m.dia,
            m.cpi, m.core_cpi, m.pce, m.core_pce, m.ppi,
            m.gdp, m.nfp, m.unemployment_rate, m.initial_claims, m.continuing_claims,
            m.retail_sales, m.industrial_production,
            m.housing_starts, m.building_permits, m.case_shiller_hpi
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
        if self._pg_engine:
            try:
                df_sent = pd.read_sql_query(
                    text("SELECT date, score as sentiment_score, confidence as sentiment_confidence, "
                    "article_count, positive_ratio, negative_ratio, macro_sentiment, "
                    "earnings_sentiment, geopolitical_sentiment, technical_sentiment, "
                    "sentiment_dispersion, sentiment_velocity "
                    "FROM daily_sentiment WHERE date = ANY(:p0)"), self._pg_engine, params={"p0": dates_list})
                df_intra = pd.read_sql_query(
                    text("SELECT date, vwap_spread, intraday_momentum, intraday_range, volume_ratio "
                    "FROM intraday_features WHERE date = ANY(:p0)"), self._pg_engine, params={"p0": dates_list})
                df_opts = pd.read_sql_query(
                    text("SELECT date, put_call_ratio, max_pain, iv_skew, gex, "
                    "vanna_exposure, charm_exposure, zero_dte_pcr "
                    "FROM options_analytics WHERE date = ANY(:p0)"), self._pg_engine, params={"p0": dates_list})
            except Exception:
                df_sent = df_intra = df_opts = pd.DataFrame()
        else:
            # SQLite fallback (if available)
            if not self._sqlite:
                return pd.DataFrame()
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
        if not self._pg_engine:
            return pd.DataFrame()
        try:
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            embed_str = str(embedding)
            where = "WHERE embedding IS NOT NULL AND published_at >= :cutoff"
            sa_params = {"embed1": embed_str, "cutoff": cutoff, "embed2": embed_str, "lim": limit}
            if category:
                where += " AND category = :cat"
                sa_params["cat"] = category
            sql = f"""SELECT id, headline, source, category, published_at,
                   sentiment_compound, 1 - (embedding <=> :embed1::vector) as similarity
            FROM raw_articles {where}
            ORDER BY embedding <=> :embed2::vector LIMIT :lim"""
            return pd.read_sql_query(text(sql), self._pg_engine, params=sa_params)
        except Exception as e:
            logger.error(f"vector_search failed: {e}")
            return pd.DataFrame()

    def vector_search_knowledge(self, embedding: list[float], limit: int = 10) -> pd.DataFrame:
        """Semantic search over the knowledge_vectors table (docs + code chunks).

        Returns a DataFrame with columns: source_path, chunk_text, similarity.
        Falls back to an empty DataFrame if pgvector is unavailable.
        """
        if not self._pg_engine:
            return pd.DataFrame()
        try:
            embed_str = str(embedding)
            # Use CAST() instead of :: to avoid SQLAlchemy parsing :emb::vector
            # as two named params (:emb and :vector)
            sql = text("""
                SELECT source_path,
                       chunk_text,
                       1 - (embedding <=> CAST(:emb AS vector)) AS similarity
                FROM   knowledge_vectors
                ORDER  BY embedding <=> CAST(:emb AS vector)
                LIMIT  :lim
            """)
            return pd.read_sql_query(
                sql, self._pg_engine, params={"emb": embed_str, "lim": limit}
            )
        except Exception as e:
            logger.error(f"vector_search_knowledge failed: {e}")
            return pd.DataFrame()


    def close(self):
        if self._pg_engine:
            try:
                self._pg_engine.dispose()
            except Exception:
                pass
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

    # ── TimescaleDB helpers ──────────────────────────────────────

    def query_time_bucket(self, table: str, time_col: str, bucket: str,
                          agg_cols: dict, where: str = "",
                          params: tuple = None) -> pd.DataFrame:
        """Query with time_bucket if TimescaleDB is available, else manual GROUP BY.

        Args:
            table: Table name
            time_col: Time column name (e.g. 'date', 'timestamp')
            bucket: Bucket size (e.g. '1 week', '1 month')
            agg_cols: Dict of {alias: sql_expr} e.g. {"avg_close": "AVG(close)"}
            where: Optional WHERE clause (without 'WHERE' keyword)
            params: Optional query params
        """
        agg_str = ", ".join(f"{expr} AS {alias}" for alias, expr in agg_cols.items())
        where_clause = f"WHERE {where}" if where else ""

        if self._has_timescaledb and self._pg_engine:
            sql = (f"SELECT time_bucket('{bucket}', {time_col}) AS bucket, "
                   f"{agg_str} FROM {table} {where_clause} "
                   f"GROUP BY bucket ORDER BY bucket")
        else:
            # Fallback: use date_trunc for PostgreSQL, strftime for SQLite
            if self._pg_engine:
                # Map bucket to date_trunc precision
                trunc_map = {"1 week": "week", "1 month": "month",
                             "1 day": "day", "1 hour": "hour"}
                precision = trunc_map.get(bucket, "day")
                sql = (f"SELECT date_trunc('{precision}', {time_col}::timestamp) AS bucket, "
                       f"{agg_str} FROM {table} {where_clause} "
                       f"GROUP BY bucket ORDER BY bucket")
            else:
                # SQLite fallback
                fmt_map = {"1 week": "%Y-%W", "1 month": "%Y-%m",
                           "1 day": "%Y-%m-%d", "1 hour": "%Y-%m-%d %H"}
                fmt = fmt_map.get(bucket, "%Y-%m-%d")
                sql = (f"SELECT strftime('{fmt}', {time_col}) AS bucket, "
                       f"{agg_str} FROM {table} {where_clause} "
                       f"GROUP BY bucket ORDER BY bucket")

        return self.query(sql, params)

    def query_continuous_aggregate(self, view_name: str, fallback_sql: str,
                                   params: tuple = None) -> pd.DataFrame:
        """Query a continuous aggregate view, falling back to raw SQL if unavailable.

        Args:
            view_name: TimescaleDB continuous aggregate name (e.g. 'cagg_prices_weekly')
            fallback_sql: SQL to run if the view doesn't exist
            params: Optional query params
        """
        if self._has_timescaledb and self._pg_engine:
            try:
                # Check if the view exists
                check_sql = ("SELECT COUNT(*) FROM information_schema.views "
                             "WHERE table_name = %s")
                cur = self._pg_conn.cursor()
                cur.execute(check_sql, (view_name,))
                exists = cur.fetchone()[0] > 0
                cur.close()

                if not exists:
                    # Also check materialized views
                    cur = self._pg_conn.cursor()
                    cur.execute(
                        "SELECT COUNT(*) FROM pg_matviews WHERE matviewname = %s",
                        (view_name,)
                    )
                    exists = cur.fetchone()[0] > 0
                    cur.close()

                if exists:
                    return self.query(f"SELECT * FROM {view_name} ORDER BY 1", params)
            except Exception as e:
                logger.debug(f"Continuous aggregate {view_name} query failed: {e}")

        # Fallback to raw SQL
        return self.query(fallback_sql, params)

    def get_weekly_prices(self) -> pd.DataFrame:
        """Get weekly OHLCV data — uses continuous aggregate if available."""
        return self.query_continuous_aggregate(
            "cagg_prices_weekly",
            "SELECT date_trunc('week', date::timestamp) AS week, "
            "MIN(open) AS open, MAX(high) AS high, MIN(low) AS low, "
            "(array_agg(close ORDER BY date DESC))[1] AS close, "
            "SUM(volume) AS volume, AVG(close) AS avg_close "
            "FROM prices GROUP BY week ORDER BY week"
            if self._pg_engine else
            "SELECT strftime('%Y-%W', date) AS week, "
            "MIN(open) AS open, MAX(high) AS high, MIN(low) AS low, "
            "close AS close, SUM(volume) AS volume, AVG(close) AS avg_close "
            "FROM prices GROUP BY week ORDER BY week"
        )

    def get_monthly_prices(self) -> pd.DataFrame:
        """Get monthly OHLCV data — uses continuous aggregate if available."""
        return self.query_continuous_aggregate(
            "cagg_prices_monthly",
            "SELECT date_trunc('month', date::timestamp) AS month, "
            "MIN(open) AS open, MAX(high) AS high, MIN(low) AS low, "
            "(array_agg(close ORDER BY date DESC))[1] AS close, "
            "SUM(volume) AS volume, AVG(close) AS avg_close "
            "FROM prices GROUP BY month ORDER BY month"
            if self._pg_engine else
            "SELECT strftime('%Y-%m', date) AS month, "
            "MIN(open) AS open, MAX(high) AS high, MIN(low) AS low, "
            "close AS close, SUM(volume) AS volume, AVG(close) AS avg_close "
            "FROM prices GROUP BY month ORDER BY month"
        )

    def get_intraday_daily_summary(self, ticker: str = "SPY") -> pd.DataFrame:
        """Get daily OHLCV from intraday bars — uses continuous aggregate if available."""
        if self._has_timescaledb:
            df = self.query_continuous_aggregate(
                "cagg_intraday_daily",
                ""  # fallback below
            )
            if not df.empty:
                if "ticker" in df.columns:
                    df = df[df["ticker"] == ticker]
                return df

        # Fallback: manual aggregation
        if self._pg_engine:
            sql = (
                "SELECT date_trunc('day', timestamp::timestamp) AS date, "
                "ticker, "
                "MIN(open) AS open, MAX(high) AS high, MIN(low) AS low, "
                "(array_agg(close ORDER BY timestamp DESC))[1] AS close, "
                "SUM(volume) AS volume "
                "FROM intraday_bars WHERE ticker = %s "
                "GROUP BY date_trunc('day', timestamp::timestamp), ticker "
                "ORDER BY date"
            )
        else:
            sql = (
                "SELECT substr(timestamp, 1, 10) AS date, ticker, "
                "MIN(open) AS open, MAX(high) AS high, MIN(low) AS low, "
                "close AS close, SUM(volume) AS volume "
                "FROM intraday_bars WHERE ticker = ? "
                "GROUP BY substr(timestamp, 1, 10), ticker ORDER BY date"
            )
        return self.query(sql, (ticker,))


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
