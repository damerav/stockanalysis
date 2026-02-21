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
    sma_20 REAL, sma_50 REAL, rsi_14 REAL,
    macd REAL, macd_signal REAL, macd_hist REAL,
    bb_upper REAL, bb_lower REAL, bb_mid REAL,
    atr_14 REAL
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
    positive_ratio REAL, negative_ratio REAL, neutral_ratio REAL
);

-- Macro indicators
CREATE TABLE IF NOT EXISTS macro (
    date TEXT PRIMARY KEY,
    vix REAL, vix_change REAL,
    us10y_yield REAL, dxy REAL, fed_funds REAL,
    gold REAL, crude REAL
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
    iv_skew REAL, gex REAL
);

-- Intraday derived features
CREATE TABLE IF NOT EXISTS intraday_features (
    date TEXT PRIMARY KEY,
    vwap_spread REAL, intraday_momentum REAL,
    intraday_range REAL, volume_ratio REAL
);

-- Prediction performance tracking
CREATE TABLE IF NOT EXISTS performance (
    date TEXT PRIMARY KEY,
    predicted TEXT, actual TEXT,
    correct INTEGER,
    cumulative_accuracy REAL
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
    """Initialize the SQLite database with all 10 tables. Returns db path."""
    db_path = get_db_path(config)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.close()
    logger.info(f"Database initialized at {db_path}")
    return db_path


def get_connection(config: dict = None) -> sqlite3.Connection:
    """Get a connection to the database, initializing if needed."""
    db_path = get_db_path(config)
    if not os.path.exists(db_path):
        init_db(config)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Database initialized successfully.")
