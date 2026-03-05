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

-- Market breadth & index fundamentals
CREATE TABLE IF NOT EXISTS market_breadth (
    date TEXT PRIMARY KEY,
    sp500_pe REAL,
    sp500_forward_pe REAL,
    sp500_earnings_yield REAL,
    sp500_dividend_yield REAL,
    pct_above_sma50 REAL,
    pct_above_sma200 REAL,
    advance_decline_ratio REAL,
    new_highs_52w INTEGER,
    new_lows_52w INTEGER,
    breadth_thrust REAL
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
    Also verifies PostgreSQL connection if configured."""
    db_path = get_db_path(config)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    # P1: Migrate existing tables to add new columns
    _migrate_schema(conn)
    conn.close()
    logger.info(f"Database initialized at {db_path}")

    # Initialize PostgreSQL connection (if configured)
    try:
        from src.data.db_router import DbRouter
        router = DbRouter(config)
        if router.using_postgres:
            logger.info("PostgreSQL connection verified")
            # Ensure strategy_rules table exists in PostgreSQL
            try:
                router.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_rules (
                        rule_group  TEXT NOT NULL, rule_key    TEXT NOT NULL,
                        rule_value  TEXT NOT NULL, value_type  TEXT NOT NULL DEFAULT 'float',
                        min_val     TEXT,          max_val     TEXT,
                        description TEXT,          updated_at  TEXT,
                        updated_by  TEXT,          PRIMARY KEY (rule_group, rule_key)
                    )
                """)
                router.execute("""
                    CREATE TABLE IF NOT EXISTS market_breadth (
                        date TEXT PRIMARY KEY,
                        sp500_pe REAL, sp500_forward_pe REAL,
                        sp500_earnings_yield REAL, sp500_dividend_yield REAL,
                        pct_above_sma50 REAL, pct_above_sma200 REAL,
                        advance_decline_ratio REAL,
                        new_highs_52w INTEGER, new_lows_52w INTEGER,
                        breadth_thrust REAL
                    )
                """)
                router.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_rules_history (
                        id SERIAL PRIMARY KEY,
                        rule_group  TEXT NOT NULL,
                        rule_key    TEXT NOT NULL,
                        old_value   TEXT,
                        new_value   TEXT NOT NULL,
                        changed_at  TEXT NOT NULL,
                        changed_by  TEXT NOT NULL DEFAULT 'system'
                    )
                """)
                from datetime import datetime
                _seed_strategy_rules_pg(router, datetime.now().isoformat())
                logger.info("strategy_rules table ready in PostgreSQL")
            except Exception as e:
                logger.warning(f"strategy_rules PostgreSQL setup failed: {e}")
        router.close()
    except Exception as e:
        logger.debug(f"PostgreSQL not available (non-fatal): {e}")

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

    # News table: add unique index on url for deduplication
    try:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_news_url ON news(url)")
    except Exception:
        pass  # may fail if duplicates already exist

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

    # Strategy rules table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_rules (
            rule_group  TEXT NOT NULL, rule_key    TEXT NOT NULL,
            rule_value  TEXT NOT NULL, value_type  TEXT NOT NULL DEFAULT 'float',
            min_val     TEXT,          max_val     TEXT,
            description TEXT,          updated_at  TEXT,
            updated_by  TEXT,          PRIMARY KEY (rule_group, rule_key)
        )
    """)

    # Strategy rules change history (for rollback)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_rules_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_group  TEXT NOT NULL,
            rule_key    TEXT NOT NULL,
            old_value   TEXT,
            new_value   TEXT NOT NULL,
            changed_at  TEXT NOT NULL,
            changed_by  TEXT NOT NULL DEFAULT 'system'
        )
    """)

    conn.commit()
    seed_strategy_rules(conn)


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


_STRATEGY_RULE_DEFAULTS = [
    ("spread", "strike_K", "6000.0", "float", "5000", "7000", "Sold strike price (K)"),
    ("spread", "credit_C", "10.0", "float", "1", "50", "Credit width (C) in points"),
    ("sizing", "max_lots", "3", "int", "1", "10", "Maximum lots per trade"),
    ("entry", "anti_chase_atr_pct", "0.5", "float", "0.1", "2.0", "Anti-chase gate fraction of ATR"),
    ("entry", "phase2_enabled", "true", "bool", None, None, "Enable Phase 2 entries"),
    ("entry", "phase2_min_filters", "2", "int", "1", "3", "Minimum Phase 2 confluence filters"),
    ("entry", "phase2_roc_threshold", "0.5", "float", "0.1", "2.0", "ROC threshold for Phase 2"),
    ("tp_low", "tp1_mult", "1.0", "float", "0.5", "5.0", "TP1 x ATR — Low"),
    ("tp_low", "tp2_mult", "1.5", "float", "0.5", "5.0", "TP2 x ATR — Low"),
    ("tp_low", "runner_trail_mult", "2.0", "float", "0.5", "8.0", "Runner trail x ATR — Low"),
    ("tp_med", "tp1_mult", "1.2", "float", "0.5", "5.0", "TP1 x ATR — Med"),
    ("tp_med", "tp2_mult", "1.8", "float", "0.5", "5.0", "TP2 x ATR — Med"),
    ("tp_med", "runner_trail_mult", "2.5", "float", "0.5", "8.0", "Runner trail x ATR — Med"),
    ("tp_high", "tp1_mult", "1.5", "float", "0.5", "5.0", "TP1 x ATR — High"),
    ("tp_high", "tp2_mult", "2.2", "float", "0.5", "5.0", "TP2 x ATR — High"),
    ("tp_high", "runner_trail_mult", "3.0", "float", "0.5", "8.0", "Runner trail x ATR — High"),
    ("risk", "emergency_stop_pct", "0.20", "float", "0.05", "0.50", "Emergency stop % of C"),
    ("risk", "jump_exit_points", "5.0", "float", "1.0", "20.0", "Jump exit threshold pts"),
    ("risk", "circuit_breaker_usd", "-2000.0", "float", "-10000", "-100", "Daily loss limit USD"),
    ("session", "session_close_ct", "15:55", "time", None, None, "Session close time CT"),
    ("session", "session_reset_ct", "17:00", "time", None, None, "Session reset time CT"),
    ("indicators", "kc_ema_period", "20", "int", "5", "100", "KC EMA period"),
    ("indicators", "kc_atr_period", "14", "int", "5", "50", "KC ATR period"),
    ("indicators", "kc_atr_multiplier", "2.0", "float", "0.5", "5.0", "KC ATR multiplier"),
    ("indicators", "rsi_period", "14", "int", "2", "50", "RSI period"),
    ("indicators", "roc_period", "3", "int", "1", "20", "ROC period"),
    ("indicators", "atr_period", "14", "int", "5", "50", "ATR period"),
    ("regime", "lookback_minutes", "10080", "int", "1440", "43200", "Regime lookback minutes"),
    ("regime", "pct_low", "33", "int", "10", "45", "Low regime percentile cutoff"),
    ("regime", "pct_high", "66", "int", "55", "90", "High regime percentile cutoff"),
    ("ai", "ai_enabled", "true", "bool", None, None, "Enable AI confidence layer"),
    ("ai", "ai_fail_closed", "true", "bool", None, None, "Fail-closed when AI unavailable"),
    ("ai", "entry_conf_threshold", "0.70", "float", "0.50", "0.99", "Entry confidence threshold"),
    ("ai", "exit_conf_threshold", "0.65", "float", "0.50", "0.99", "Exit hold confidence threshold"),
    ("ai", "trail_ai_enabled", "true", "bool", None, None, "Use CNN for dynamic trailing"),
    ("ai", "regime_thresholds_low", "0.58", "float", "0.50", "0.99", "Entry threshold Low regime"),
    ("ai", "regime_thresholds_med", "0.55", "float", "0.50", "0.99", "Entry threshold Med regime"),
    ("ai", "regime_thresholds_high", "0.52", "float", "0.50", "0.99", "Entry threshold High regime"),
    ("rl", "rl_alpha", "0.1", "float", "0.001", "0.5", "Q-learning rate"),
    ("rl", "rl_gamma", "0.95", "float", "0.5", "0.999", "Discount factor"),
    ("rl", "rl_epsilon", "0.1", "float", "0.0", "0.5", "Exploration rate"),
    ("rl", "rl_lambda_dd", "0.5", "float", "0.0", "2.0", "Drawdown penalty weight"),
]


def seed_strategy_rules(conn):
    """Seed strategy_rules table with defaults (SQLite — INSERT OR IGNORE)."""
    from datetime import datetime
    now = datetime.now().isoformat()
    for row in _STRATEGY_RULE_DEFAULTS:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO strategy_rules "
                "(rule_group, rule_key, rule_value, value_type, min_val, max_val, "
                "description, updated_at, updated_by) VALUES (?,?,?,?,?,?,?,?,?)",
                (*row, now, "system"),
            )
        except Exception:
            pass
    conn.commit()


def _seed_strategy_rules_pg(router, now: str):
    """Seed strategy_rules table with defaults (PostgreSQL — ON CONFLICT DO NOTHING)."""
    for row in _STRATEGY_RULE_DEFAULTS:
        try:
            router.execute(
                "INSERT INTO strategy_rules "
                "(rule_group,rule_key,rule_value,value_type,min_val,max_val,"
                "description,updated_at,updated_by) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT (rule_group,rule_key) DO NOTHING",
                (*row, now, "system"),
            )
        except Exception:
            pass


def get_connection(config: dict = None):
    """Get a database connection — PostgreSQL primary, SQLite fallback.

    Returns a psycopg2 connection if PostgreSQL is configured and reachable,
    otherwise falls back to SQLite. Callers should use %s placeholders for
    PostgreSQL or ? for SQLite — prefer using DbRouter instead for auto-conversion.
    """
    # Try PostgreSQL first
    try:
        from src.data.db_router import DbRouter
        router = DbRouter(config)
        if router.using_postgres:
            pg = router.get_pg()
            if pg:
                return pg
    except Exception:
        pass

    # Fallback to SQLite
    db_path = get_db_path(config)
    try:
        if not os.path.exists(db_path):
            init_db(config)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        _migrate_schema(conn)
        return conn
    except sqlite3.DatabaseError:
        # Corrupted SQLite file — remove and recreate
        logger.warning(f"SQLite file corrupted ({db_path}), recreating")
        try:
            os.remove(db_path)
        except OSError:
            pass
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
