"""1F. SQLite Database Schema — Creates 10 tables in spy.db."""

import sqlite3
import os
import yaml
import logging

logger = logging.getLogger(__name__)

SCHEMA = """
-- Daily OHLCV prices
CREATE TABLE IF NOT EXISTS prices (
    date TEXT PRIMARY KEY,
    open REAL, high REAL, low REAL, close REAL,
    volume INTEGER, adjusted_close REAL
);

-- Technical indicators
CREATE TABLE IF NOT EXISTS technicals (
    date TEXT PRIMARY KEY,
    sma_20 REAL, sma_50 REAL, sma_200 REAL,
    rsi_14 REAL,
    macd REAL, macd_signal REAL, macd_hist REAL,
    bb_upper REAL, bb_lower REAL, bb_mid REAL,
    atr_14 REAL,
    obv REAL, garman_klass_vol REAL,
    stoch_k REAL, stoch_d REAL
);

-- News headlines
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT, source TEXT, headline TEXT, summary TEXT,
    url TEXT, fetched_at TEXT
);

-- Daily aggregated sentiment
CREATE TABLE IF NOT EXISTS daily_sentiment (
    date TEXT PRIMARY KEY,
    score REAL, confidence REAL, article_count INTEGER,
    positive_ratio REAL, negative_ratio REAL, neutral_ratio REAL,
    -- P2: Structured sentiment decomposition
    macro_sentiment REAL,
    earnings_sentiment REAL,
    geopolitical_sentiment REAL,
    technical_sentiment REAL,
    sentiment_dispersion REAL,
    sentiment_velocity REAL
);

-- Macro indicators
CREATE TABLE IF NOT EXISTS macro (
    date TEXT PRIMARY KEY,
    vix REAL, vix_change REAL,
    us10y_yield REAL, dxy REAL, fed_funds REAL,
    gold REAL, crude REAL,
    -- VIX term structure (P1 enhancement)
    vix9d REAL, vix3m REAL, vix6m REAL, vvix REAL, skew_index REAL,
    -- Cross-asset signals (P1 enhancement)
    hy_spread REAL, tlt_spy_ratio REAL, eem_spy_ratio REAL,
    copper_gold_ratio REAL, xlk_xlf_ratio REAL, xlk_xle_ratio REAL
);

-- Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    date TEXT PRIMARY KEY,
    direction TEXT, confidence REAL,
    factors TEXT,  -- JSON blob of factor scores
    report_text TEXT,
    predicted_at TEXT
);

-- Intraday 5-second bars
CREATE TABLE IF NOT EXISTS intraday_bars (
    timestamp TEXT, ticker TEXT,
    open REAL, high REAL, low REAL, close REAL,
    volume INTEGER, vwap REAL,
    PRIMARY KEY (timestamp, ticker)
);

-- Options chain snapshots
CREATE TABLE IF NOT EXISTS options_chain (
    date TEXT, contract_symbol TEXT,
    strike REAL, expiry TEXT, option_type TEXT,
    last_price REAL, bid REAL, ask REAL,
    volume INTEGER, open_interest INTEGER,
    iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL,
    PRIMARY KEY (date, contract_symbol)
);

-- Computed options analytics
CREATE TABLE IF NOT EXISTS options_analytics (
    date TEXT PRIMARY KEY,
    put_call_ratio REAL, max_pain REAL,
    iv_skew REAL, gex REAL,
    -- P3: Extended dealer greek exposures
    vanna_exposure REAL, charm_exposure REAL, zero_dte_pcr REAL
);

-- Intraday derived features
CREATE TABLE IF NOT EXISTS intraday_features (
    date TEXT PRIMARY KEY,
    vwap_spread REAL, intraday_momentum REAL,
    intraday_range REAL, volume_ratio REAL
);

-- Prediction performance tracking (extended for stratified accuracy P1)
CREATE TABLE IF NOT EXISTS performance (
    date TEXT PRIMARY KEY,
    predicted TEXT, actual TEXT,
    correct INTEGER,
    cumulative_accuracy REAL,
    confidence_tier TEXT,       -- 'high' (>=70), 'medium' (50-70), 'low' (<50)
    vix_regime TEXT,            -- 'low' (<15), 'normal' (15-25), 'high' (>25)
    day_of_week INTEGER,       -- 0=Mon..4=Fri
    event_proximity INTEGER    -- 1 if within 2 days of FOMC/CPI/NFP
);
"""


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_db_path(config: dict = None) -> str:
    """Get database path from config, creating directories as needed."""
    if config is None:
        config = load_config()
    db_path = config.get("database", {}).get("path", "./data/spy.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return db_path


def init_db(config: dict = None) -> str:
    """Initialize the SQLite database with all tables. Returns db path.
    Also initializes DuckDB analytics database if duckdb is available."""
    db_path = get_db_path(config)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    # P1: Migrate existing tables to add new columns
    _migrate_schema(conn)
    conn.close()
    logger.info(f"Database initialized at {db_path}")

    # Enhancement 26: Initialize DuckDB analytics database
    try:
        from src.data.db_router import DbRouter, _get_duckdb_path
        duckdb_path = _get_duckdb_path(config)
        os.makedirs(os.path.dirname(duckdb_path) or ".", exist_ok=True)
        router = DbRouter(config)
        router.close()
        logger.info(f"DuckDB analytics initialized at {duckdb_path}")
    except ImportError:
        logger.debug("duckdb not installed — analytics layer skipped")
    except Exception as e:
        logger.warning(f"DuckDB init failed (non-fatal): {e}")

    return db_path


def _migrate_schema(conn: sqlite3.Connection):
    """Add new columns to existing tables if they don't exist yet."""
    # Macro table: VIX term structure + cross-asset columns
    macro_new_cols = [
        ("vix9d", "REAL"), ("vix3m", "REAL"), ("vix6m", "REAL"),
        ("vvix", "REAL"), ("skew_index", "REAL"),
        ("hy_spread", "REAL"), ("tlt_spy_ratio", "REAL"),
        ("eem_spy_ratio", "REAL"), ("copper_gold_ratio", "REAL"),
        ("xlk_xlf_ratio", "REAL"), ("xlk_xle_ratio", "REAL"),
    ]
    _add_columns_if_missing(conn, "macro", macro_new_cols)

    # Performance table: stratified accuracy columns
    perf_new_cols = [
        ("confidence_tier", "TEXT"), ("vix_regime", "TEXT"),
        ("day_of_week", "INTEGER"), ("event_proximity", "INTEGER"),
    ]
    _add_columns_if_missing(conn, "performance", perf_new_cols)

    # P3: Options analytics extended columns
    opts_new_cols = [
        ("vanna_exposure", "REAL"), ("charm_exposure", "REAL"),
        ("zero_dte_pcr", "REAL"),
    ]
    _add_columns_if_missing(conn, "options_analytics", opts_new_cols)

    # P2: Decomposed sentiment columns
    sentiment_new_cols = [
        ("macro_sentiment", "REAL"), ("earnings_sentiment", "REAL"),
        ("geopolitical_sentiment", "REAL"), ("technical_sentiment", "REAL"),
        ("sentiment_dispersion", "REAL"), ("sentiment_velocity", "REAL"),
    ]
    _add_columns_if_missing(conn, "daily_sentiment", sentiment_new_cols)

    # New technical indicator columns
    tech_new_cols = [
        ("sma_200", "REAL"), ("obv", "REAL"), ("garman_klass_vol", "REAL"),
        ("stoch_k", "REAL"), ("stoch_d", "REAL"),
    ]
    _add_columns_if_missing(conn, "technicals", tech_new_cols)

    # P3: Earnings calendar table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS earnings_calendar (
            date TEXT, ticker TEXT, eps_estimate REAL, eps_actual REAL,
            surprise_pct REAL, market_cap_pct REAL,
            PRIMARY KEY (date, ticker)
        )
    """)

    # P3: Fed communications table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fed_communications (
            date TEXT PRIMARY KEY,
            type TEXT,
            hawkish_score REAL,
            summary TEXT,
            scored_at TEXT
        )
    """)

    # Users table (bcrypt-hashed passwords)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            name TEXT,
            role TEXT DEFAULT 'viewer',
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()


def _add_columns_if_missing(conn: sqlite3.Connection, table: str,
                             columns: list[tuple[str, str]]):
    """Add columns to a table if they don't already exist."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    for col_name, col_type in columns:
        if col_name not in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                logger.info(f"Added column {col_name} to {table}")
            except Exception as e:
                logger.debug(f"Column {col_name} already exists or error: {e}")
    conn.commit()


def get_connection(config: dict = None) -> sqlite3.Connection:
    """Get a connection to the database, initializing if needed."""
    db_path = get_db_path(config)
    if not os.path.exists(db_path):
        init_db(config)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    # P1: Ensure schema is up to date
    _migrate_schema(conn)
    return conn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Database initialized successfully.")
