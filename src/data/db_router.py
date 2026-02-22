"""Database Router — Routes queries between DuckDB (analytics) and SQLite (operational).

Enhancement 26: Analytics tables (prices, technicals, macro, intraday_bars, options_chain)
are stored in DuckDB for fast columnar reads. Operational tables remain in SQLite.

Usage:
    router = get_router(config)
    df = router.read_analytics("SELECT * FROM prices WHERE date >= '2025-01-01'")
    router.write_analytics("INSERT INTO prices VALUES (?, ...)", params)
    conn = router.get_sqlite()  # for operational tables
    router.close()
"""

import os
import sqlite3
import logging
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# Tables stored in DuckDB (high-volume, read-heavy analytics)
ANALYTICS_TABLES = {"prices", "technicals", "macro", "intraday_bars", "options_chain"}

# Tables remaining in SQLite (write-heavy operational)
SQLITE_TABLES = {
    "daily_sentiment", "predictions", "performance", "news",
    "options_analytics", "intraday_features",
    "earnings_calendar", "fed_communications",
    "model_registry", "feature_cache",
}


def _get_duckdb_path(config: dict = None) -> str:
    if config:
        return config.get("database", {}).get("analytics_path", "./data/analytics.duckdb")
    return "./data/analytics.duckdb"


def _get_sqlite_path(config: dict = None) -> str:
    if config:
        return config.get("database", {}).get("path", "./data/spy.db")
    return "./data/spy.db"


# DuckDB schema for analytics tables
DUCKDB_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    date VARCHAR PRIMARY KEY,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume BIGINT, adjusted_close DOUBLE
);

CREATE TABLE IF NOT EXISTS technicals (
    date VARCHAR PRIMARY KEY,
    sma_20 DOUBLE, sma_50 DOUBLE, rsi_14 DOUBLE,
    macd DOUBLE, macd_signal DOUBLE, macd_hist DOUBLE,
    bb_upper DOUBLE, bb_lower DOUBLE, bb_mid DOUBLE,
    atr_14 DOUBLE
);

CREATE TABLE IF NOT EXISTS macro (
    date VARCHAR PRIMARY KEY,
    vix DOUBLE, vix_change DOUBLE,
    us10y_yield DOUBLE, dxy DOUBLE, fed_funds DOUBLE,
    gold DOUBLE, crude DOUBLE,
    vix9d DOUBLE, vix3m DOUBLE, vix6m DOUBLE, vvix DOUBLE, skew_index DOUBLE,
    hy_spread DOUBLE, tlt_spy_ratio DOUBLE, eem_spy_ratio DOUBLE,
    copper_gold_ratio DOUBLE, xlk_xlf_ratio DOUBLE, xlk_xle_ratio DOUBLE
);

CREATE TABLE IF NOT EXISTS intraday_bars (
    timestamp VARCHAR, ticker VARCHAR,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume BIGINT, vwap DOUBLE,
    PRIMARY KEY (timestamp, ticker)
);

CREATE TABLE IF NOT EXISTS options_chain (
    date VARCHAR, contract_symbol VARCHAR,
    strike DOUBLE, expiry VARCHAR, option_type VARCHAR,
    last_price DOUBLE, bid DOUBLE, ask DOUBLE,
    volume BIGINT, open_interest BIGINT,
    iv DOUBLE, delta DOUBLE, gamma DOUBLE, theta DOUBLE, vega DOUBLE,
    PRIMARY KEY (date, contract_symbol)
);
"""


class DbRouter:
    """Routes queries between DuckDB (analytics) and SQLite (operational)."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._duckdb_path = _get_duckdb_path(config)
        self._sqlite_path = _get_sqlite_path(config)

        # Ensure directories exist
        os.makedirs(os.path.dirname(self._duckdb_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self._sqlite_path) or ".", exist_ok=True)

        # Open connections
        self._duck = duckdb.connect(self._duckdb_path)
        self._sqlite = sqlite3.connect(self._sqlite_path)
        self._sqlite.row_factory = sqlite3.Row
        self._sqlite.execute("PRAGMA journal_mode=WAL")
        self._sqlite.execute("PRAGMA busy_timeout=5000")

        # Initialize DuckDB schema
        self._init_duckdb()
        logger.info(f"DbRouter ready: DuckDB={self._duckdb_path}, SQLite={self._sqlite_path}")

    def _init_duckdb(self):
        """Create analytics tables in DuckDB if they don't exist."""
        for stmt in DUCKDB_SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._duck.execute(stmt)

    # --- Analytics (DuckDB) operations ---

    def read_analytics(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """Execute a SELECT on DuckDB analytics tables, return DataFrame."""
        try:
            if params:
                return self._duck.execute(sql, params).fetchdf()
            return self._duck.execute(sql).fetchdf()
        except Exception as e:
            logger.error(f"DuckDB read error: {e}")
            return pd.DataFrame()

    def write_analytics(self, sql: str, params: tuple = None):
        """Execute an INSERT/UPDATE/DELETE on DuckDB analytics tables."""
        try:
            if params:
                self._duck.execute(sql, params)
            else:
                self._duck.execute(sql)
        except Exception as e:
            logger.error(f"DuckDB write error: {e}")
            raise

    def execute_analytics(self, sql: str, params: tuple = None):
        """Execute arbitrary SQL on DuckDB (for DDL, etc.)."""
        if params:
            self._duck.execute(sql, params)
        else:
            self._duck.execute(sql)

    def get_analytics_conn(self) -> duckdb.DuckDBPyConnection:
        """Return the raw DuckDB connection for advanced usage."""
        return self._duck

    # --- Operational (SQLite) operations ---

    def get_sqlite(self) -> sqlite3.Connection:
        """Return the SQLite connection for operational tables."""
        return self._sqlite

    def read_sqlite(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """Execute a SELECT on SQLite operational tables, return DataFrame."""
        try:
            if params:
                return pd.read_sql_query(sql, self._sqlite, params=params)
            return pd.read_sql_query(sql, self._sqlite)
        except Exception as e:
            logger.error(f"SQLite read error: {e}")
            return pd.DataFrame()

    # --- Cross-database join helper ---

    def read_feature_join(self, date: str = None) -> pd.DataFrame:
        """Read analytics tables from DuckDB and operational tables from SQLite,
        then merge in Python. This replaces the old single-JOIN query."""
        # Analytics from DuckDB
        date_filter = f" WHERE p.date = '{date}'" if date else ""
        analytics_sql = f"""
        SELECT
            p.date,
            p.open, p.high, p.low, p.close, p.volume,
            t.sma_20, t.sma_50, t.rsi_14,
            t.macd, t.macd_signal, t.macd_hist,
            t.bb_upper, t.bb_lower, t.atr_14,
            m.vix, m.vix_change, m.us10y_yield, m.dxy, m.fed_funds, m.gold, m.crude,
            m.vix9d, m.vix3m, m.vix6m, m.vvix, m.skew_index,
            m.hy_spread, m.tlt_spy_ratio, m.eem_spy_ratio,
            m.copper_gold_ratio, m.xlk_xlf_ratio, m.xlk_xle_ratio
        FROM prices p
        LEFT JOIN technicals t ON p.date = t.date
        LEFT JOIN macro m ON p.date = m.date
        {date_filter}
        ORDER BY p.date
        """
        df_analytics = self.read_analytics(analytics_sql)
        if df_analytics.empty:
            return pd.DataFrame()

        # Operational from SQLite
        dates_list = df_analytics["date"].tolist()
        if not dates_list:
            return df_analytics

        placeholders = ",".join(["?"] * len(dates_list))

        # Sentiment
        df_sent = pd.read_sql_query(
            f"""SELECT date, score as sentiment_score, confidence as sentiment_confidence,
                article_count, positive_ratio, negative_ratio,
                macro_sentiment, earnings_sentiment, geopolitical_sentiment,
                technical_sentiment, sentiment_dispersion, sentiment_velocity
            FROM daily_sentiment WHERE date IN ({placeholders})""",
            self._sqlite, params=dates_list,
        )

        # Intraday features
        df_intra = pd.read_sql_query(
            f"""SELECT date, vwap_spread, intraday_momentum, intraday_range, volume_ratio
            FROM intraday_features WHERE date IN ({placeholders})""",
            self._sqlite, params=dates_list,
        )

        # Options analytics
        df_opts = pd.read_sql_query(
            f"""SELECT date, put_call_ratio, max_pain, iv_skew, gex,
                vanna_exposure, charm_exposure, zero_dte_pcr
            FROM options_analytics WHERE date IN ({placeholders})""",
            self._sqlite, params=dates_list,
        )

        # Merge all
        df = df_analytics
        if not df_sent.empty:
            df = df.merge(df_sent, on="date", how="left")
        else:
            for c in ["sentiment_score", "sentiment_confidence", "article_count",
                       "positive_ratio", "negative_ratio", "macro_sentiment",
                       "earnings_sentiment", "geopolitical_sentiment",
                       "technical_sentiment", "sentiment_dispersion", "sentiment_velocity"]:
                df[c] = None

        if not df_intra.empty:
            df = df.merge(df_intra, on="date", how="left")
        else:
            for c in ["vwap_spread", "intraday_momentum", "intraday_range", "volume_ratio"]:
                df[c] = None

        if not df_opts.empty:
            df = df.merge(df_opts, on="date", how="left")
        else:
            for c in ["put_call_ratio", "max_pain", "iv_skew", "gex",
                       "vanna_exposure", "charm_exposure", "zero_dte_pcr"]:
                df[c] = None

        return df

    # --- Lifecycle ---

    def close(self):
        """Close both connections."""
        try:
            self._duck.close()
        except Exception:
            pass
        try:
            self._sqlite.close()
        except Exception:
            pass
        logger.info("DbRouter connections closed")


# Module-level singleton
_router_instance: Optional[DbRouter] = None


def get_router(config: dict = None) -> DbRouter:
    """Get or create the module-level DbRouter singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = DbRouter(config)
    return _router_instance


def reset_router():
    """Reset the singleton (for testing)."""
    global _router_instance
    if _router_instance:
        _router_instance.close()
    _router_instance = None
