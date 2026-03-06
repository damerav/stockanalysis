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
    predicted_at TEXT,
    regime TEXT,   -- HMM regime (bull_trend, bear_trend, high_vol_choppy, low_vol_range)
    -- Enhanced Prediction: model + institutional flow fusion
    enhanced_direction TEXT,
    enhanced_score REAL,
    flow_score REAL,
    flow_alert_count INTEGER
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
    event_proximity INTEGER,   -- 1 if within 2 days of FOMC/CPI/NFP
    -- Enhanced prediction tracking
    enhanced_predicted TEXT,
    enhanced_correct INTEGER,
    enhanced_cumulative_accuracy REAL
);

-- Historical backtest results (model predictions vs actuals)
CREATE TABLE IF NOT EXISTS backtest_results (
    date TEXT PRIMARY KEY,
    predicted_direction TEXT,
    predicted_confidence REAL,
    actual_direction TEXT,
    actual_return REAL,
    correct INTEGER,
    regime TEXT,
    cumulative_accuracy REAL
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
            # Enable TimescaleDB if available
            _init_timescaledb(router)
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
                # Inverted strangle tables (PostgreSQL)
                router.execute("""
                    CREATE TABLE IF NOT EXISTS inverted_strangle_positions (
                        id              SERIAL PRIMARY KEY,
                        trade_date      TEXT    NOT NULL,
                        underlying      TEXT    NOT NULL DEFAULT 'SPY',
                        spot_at_open    REAL    NOT NULL,
                        expiry_date     TEXT    NOT NULL,
                        dte_at_open     INTEGER NOT NULL,
                        status          TEXT    NOT NULL DEFAULT 'OPEN',
                        short_put       REAL    NOT NULL,
                        short_call      REAL    NOT NULL,
                        long_put        REAL    NOT NULL,
                        long_call       REAL    NOT NULL,
                        inversion_pts   REAL    NOT NULL DEFAULT 5.0,
                        wing_pts        REAL    NOT NULL DEFAULT 25.0,
                        initial_credit  REAL    NOT NULL,
                        credit_per_leg  TEXT,
                        profit_target   REAL    NOT NULL,
                        current_value   REAL,
                        current_pnl     REAL,
                        close_date      TEXT,
                        close_price     REAL,
                        close_reason    TEXT,
                        roll_count      INTEGER NOT NULL DEFAULT 0,
                        notes           TEXT,
                        vix_at_open     REAL,
                        position_delta  REAL,
                        entry_iv_rank   REAL,
                        entry_vix_term_structure REAL,
                        cost_to_close   REAL,
                        c2c_updated_at  TEXT,
                        c2c_extrinsic   REAL,
                        c2c_intrinsic   REAL,
                        credit_vs_width REAL,
                        loss_rule_2_1_breached INTEGER NOT NULL DEFAULT 0
                    )
                """)
                router.execute("""
                    CREATE TABLE IF NOT EXISTS inverted_strangle_adjustments (
                        id              SERIAL PRIMARY KEY,
                        position_id     INTEGER NOT NULL,
                        adj_date        TEXT    NOT NULL,
                        adj_type        TEXT    NOT NULL,
                        old_short_put   REAL,
                        old_short_call  REAL,
                        new_short_put   REAL,
                        new_short_call  REAL,
                        new_spot        REAL,
                        debit_paid      REAL,
                        notes           TEXT,
                        FOREIGN KEY (position_id) REFERENCES inverted_strangle_positions(id)
                    )
                """)
                logger.info("strategy_rules table ready in PostgreSQL")

                # v2.9.1: Enhanced prediction columns (PostgreSQL ALTER TABLE)
                _pg_conn = router._pg_conn
                for tbl, col, ctype in [
                    ("predictions", "regime", "TEXT"),
                    ("predictions", "enhanced_direction", "TEXT"),
                    ("predictions", "enhanced_score", "REAL"),
                    ("predictions", "flow_score", "REAL"),
                    ("predictions", "flow_alert_count", "INTEGER"),
                    ("performance", "enhanced_predicted", "TEXT"),
                    ("performance", "enhanced_correct", "INTEGER"),
                    ("performance", "enhanced_cumulative_accuracy", "REAL"),
                ]:
                    try:
                        _pg_conn.cursor().execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {ctype}")
                        _pg_conn.commit()
                    except Exception:
                        _pg_conn.rollback()

                # v2.9.2: finbert_cache table (PostgreSQL — url_hash keyed)
                try:
                    _pg_conn.cursor().execute("""
                        CREATE TABLE IF NOT EXISTS finbert_cache (
                            url_hash TEXT PRIMARY KEY,
                            article_id INTEGER,
                            fb_positive REAL NOT NULL,
                            fb_negative REAL NOT NULL,
                            fb_neutral REAL NOT NULL,
                            fb_label TEXT,
                            fb_score REAL,
                            scored_at TEXT NOT NULL
                        )
                    """)
                    _pg_conn.commit()
                    logger.info("finbert_cache table ready in PostgreSQL")
                except Exception:
                    _pg_conn.rollback()
                # Add new columns if old schema exists
                for col, ctype in [
                    ("url_hash", "TEXT"), ("fb_label", "TEXT"),
                    ("fb_score", "REAL"), ("scored_at", "TEXT"),
                ]:
                    try:
                        _pg_conn.cursor().execute(
                            f"ALTER TABLE finbert_cache ADD COLUMN {col} {ctype}")
                        _pg_conn.commit()
                    except Exception:
                        _pg_conn.rollback()

            except Exception as e:
                logger.warning(f"strategy_rules PostgreSQL setup failed: {e}")
        router.close()
    except Exception as e:
        logger.debug(f"PostgreSQL not available (non-fatal): {e}")

    return db_path


def _init_timescaledb(router):
    """Enable TimescaleDB extension and convert tables to hypertables if available.

    Safe to call multiple times — all operations use IF NOT EXISTS.
    Falls back silently if TimescaleDB is not installed.
    """
    if not router.using_postgres:
        return

    conn = router.get_pg()
    cur = conn.cursor()

    # Check if TimescaleDB is available
    try:
        cur.execute("SELECT default_version FROM pg_available_extensions WHERE name = 'timescaledb'")
        row = cur.fetchone()
        if not row:
            logger.debug("TimescaleDB not available, using regular PostgreSQL")
            cur.close()
            return
    except Exception:
        cur.close()
        return

    # Install extension
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        logger.info("TimescaleDB extension enabled")
    except Exception as e:
        logger.debug(f"TimescaleDB extension install skipped: {e}")
        cur.close()
        return

    # Convert time-series tables to hypertables
    # (table, time_col, chunk_interval, extra_unique_cols)
    hypertables = [
        ("prices",            "date", "1 month",  []),
        ("technicals",        "date", "1 month",  []),
        ("macro",             "date", "1 month",  []),
        ("daily_sentiment",   "date", "1 month",  []),
        ("options_analytics", "date", "1 month",  []),
        ("intraday_features", "date", "1 month",  []),
        ("market_breadth",    "date", "3 months", []),
        ("predictions",       "date", "3 months", []),
        ("performance",       "date", "3 months", []),
        ("backtest_results",  "date", "3 months", []),
        ("intraday_bars",     "timestamp", "1 week", ["ticker"]),
        ("options_chain",     "date", "1 month",  ["contract_symbol"]),
    ]

    for table, time_col, chunk_interval, extra_cols in hypertables:
        try:
            # Check if already a hypertable
            cur.execute(
                "SELECT COUNT(*) FROM timescaledb_information.hypertables "
                "WHERE hypertable_name = %s", (table,)
            )
            if cur.fetchone()[0] > 0:
                continue  # Already a hypertable

            # Check if table exists
            cur.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = %s", (table,)
            )
            if cur.fetchone()[0] == 0:
                continue  # Table doesn't exist yet

            # Convert TEXT date columns to DATE type (required by TimescaleDB)
            if time_col == "date":
                cur.execute("""
                    SELECT data_type FROM information_schema.columns
                    WHERE table_name = %s AND column_name = 'date'
                """, (table,))
                dtype_row = cur.fetchone()
                if dtype_row and dtype_row[0] in ('text', 'character varying'):
                    cur.execute(f"ALTER TABLE {table} "
                                f"ALTER COLUMN date TYPE DATE USING date::DATE")

            # Convert intraday_bars timestamp from TEXT to TIMESTAMPTZ if needed
            if table == "intraday_bars":
                cur.execute("""
                    SELECT data_type FROM information_schema.columns
                    WHERE table_name = %s AND column_name = 'timestamp'
                """, (table,))
                dtype_row = cur.fetchone()
                if dtype_row and dtype_row[0] in ('text', 'character varying'):
                    cur.execute(f"""
                        ALTER TABLE {table}
                        ALTER COLUMN timestamp TYPE TIMESTAMPTZ
                        USING timestamp::TIMESTAMPTZ
                    """)

            # Drop PK constraint (TimescaleDB needs time col in unique constraints)
            cur.execute("""
                SELECT constraint_name FROM information_schema.table_constraints
                WHERE table_name = %s AND constraint_type = 'PRIMARY KEY'
            """, (table,))
            pk_row = cur.fetchone()
            if pk_row:
                cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT {pk_row[0]}")

            # Add UNIQUE constraint
            unique_cols = [time_col] + extra_cols
            cols_str = ", ".join(unique_cols)
            uq_name = f"uq_{table}_{'_'.join(unique_cols)}"
            try:
                cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT {uq_name} "
                            f"UNIQUE ({cols_str})")
            except Exception:
                pass  # May already exist

            # Create hypertable
            cur.execute(
                f"SELECT create_hypertable('{table}', '{time_col}', "
                f"chunk_time_interval => INTERVAL '{chunk_interval}', "
                f"if_not_exists => TRUE, migrate_data => TRUE)"
            )
            logger.info(f"Converted {table} to hypertable (chunk={chunk_interval})")

        except Exception as e:
            logger.debug(f"Hypertable conversion skipped for {table}: {e}")
            try:
                conn.rollback()
                conn.autocommit = True
            except Exception:
                pass

    cur.close()


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

    # v2.8: pandas-ta comprehensive technical indicator columns
    pta_new_cols = [
        ("adx_14", "REAL"), ("cci_20", "REAL"), ("aroon_up", "REAL"), ("aroon_down", "REAL"),
        ("psar_long", "REAL"), ("psar_short", "REAL"), ("dpo_20", "REAL"), ("trix_14", "REAL"),
        ("vortex_pos", "REAL"), ("vortex_neg", "REAL"), ("williams_r", "REAL"), ("mfi_14", "REAL"),
        ("rsi_2", "REAL"), ("rsi_9", "REAL"), ("rsi_21", "REAL"), ("cmo_14", "REAL"), ("ppo", "REAL"),
        ("roc_5", "REAL"), ("roc_21", "REAL"),
        ("kc_upper_20", "REAL"), ("kc_lower_20", "REAL"), ("atr_7", "REAL"), ("atr_21", "REAL"),
        ("donchian_high", "REAL"), ("donchian_low", "REAL"), ("ulcer_14", "REAL"),
        ("cmf_20", "REAL"), ("vwma_20", "REAL"), ("eom_14", "REAL"),
        ("ema_9", "REAL"), ("ema_21", "REAL"), ("ema_200", "REAL"),
        ("hma_20", "REAL"), ("wma_20", "REAL"), ("dema_20", "REAL"), ("tema_20", "REAL"), ("kama_10", "REAL"),
        ("ichi_tenkan", "REAL"), ("ichi_kijun", "REAL"), ("ichi_senkou_a", "REAL"), ("ichi_senkou_b", "REAL"),
    ]
    _add_columns_if_missing(conn, "technicals", pta_new_cols)

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

    # Inverted strangle: guardrails columns
    strangle_new_cols = [
        ("cost_to_close", "REAL"),
        ("c2c_updated_at", "TEXT"),
        ("c2c_extrinsic", "REAL"),
        ("c2c_intrinsic", "REAL"),
        ("credit_vs_width", "REAL"),
        ("loss_rule_2_1_breached", "INTEGER DEFAULT 0"),
    ]
    _add_columns_if_missing(conn, "inverted_strangle_positions", strangle_new_cols)

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

    # Inverted strangle positions table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inverted_strangle_positions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date      TEXT    NOT NULL,
            underlying      TEXT    NOT NULL DEFAULT 'SPY',
            spot_at_open    REAL    NOT NULL,
            expiry_date     TEXT    NOT NULL,
            dte_at_open     INTEGER NOT NULL,
            status          TEXT    NOT NULL DEFAULT 'OPEN',
            short_put       REAL    NOT NULL,
            short_call      REAL    NOT NULL,
            long_put        REAL    NOT NULL,
            long_call       REAL    NOT NULL,
            inversion_pts   REAL    NOT NULL DEFAULT 5.0,
            wing_pts        REAL    NOT NULL DEFAULT 25.0,
            initial_credit  REAL    NOT NULL,
            credit_per_leg  TEXT,
            profit_target   REAL    NOT NULL,
            current_value   REAL,
            current_pnl     REAL,
            close_date      TEXT,
            close_price     REAL,
            close_reason    TEXT,
            roll_count      INTEGER NOT NULL DEFAULT 0,
            notes           TEXT,
            vix_at_open     REAL,
            position_delta  REAL,
            entry_iv_rank   REAL,
            entry_vix_term_structure REAL,
            cost_to_close   REAL,
            c2c_updated_at  TEXT,
            c2c_extrinsic   REAL,
            c2c_intrinsic   REAL,
            credit_vs_width REAL,
            loss_rule_2_1_breached INTEGER NOT NULL DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS inverted_strangle_adjustments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id     INTEGER NOT NULL,
            adj_date        TEXT    NOT NULL,
            adj_type        TEXT    NOT NULL,
            old_short_put   REAL,
            old_short_call  REAL,
            new_short_put   REAL,
            new_short_call  REAL,
            new_spot        REAL,
            debit_paid      REAL,
            notes           TEXT,
            FOREIGN KEY (position_id) REFERENCES inverted_strangle_positions(id)
        )
    """)

    # v2.8: New columns for market_breadth table
    breadth_new_cols = [
        ("sp500_cape", "REAL"),
        ("buffett_indicator", "REAL"),
        ("fear_greed_index", "INTEGER"),
        ("trin", "REAL"),
    ]
    _add_columns_if_missing(conn, "market_breadth", breadth_new_cols)

    # v2.8: New columns for macro table (extended FRED series)
    macro_ext_cols = [
        ("us3m_yield", "REAL"),
        ("yield_curve_10y3m", "REAL"),
        ("sahm_rule", "REAL"),
        ("consumer_conf", "REAL"),
        ("ism_pmi", "REAL"),
    ]
    _add_columns_if_missing(conn, "macro", macro_ext_cols)

    # v2.8: New columns for macro table (sector ETF prices)
    sector_etf_cols = [
        ("xlk", "REAL"), ("xlf", "REAL"), ("xle", "REAL"),
        ("xlv", "REAL"), ("xli", "REAL"), ("xlu", "REAL"), ("xlb", "REAL"),
        ("xlp", "REAL"), ("xly", "REAL"), ("xlre", "REAL"),
        ("qqq", "REAL"), ("iwm", "REAL"), ("dia", "REAL"),
    ]
    _add_columns_if_missing(conn, "macro", sector_etf_cols)

    # v2.9.2: HMM regime column on predictions table
    _add_columns_if_missing(conn, "predictions", [("regime", "TEXT")])

    # v2.9.1: Enhanced prediction columns on predictions table
    enhanced_pred_cols = [
        ("enhanced_direction", "TEXT"),
        ("enhanced_score", "REAL"),
        ("flow_score", "REAL"),
        ("flow_alert_count", "INTEGER"),
    ]
    _add_columns_if_missing(conn, "predictions", enhanced_pred_cols)

    # v2.9.1: Enhanced prediction tracking on performance table
    enhanced_perf_cols = [
        ("enhanced_predicted", "TEXT"),
        ("enhanced_correct", "INTEGER"),
        ("enhanced_cumulative_accuracy", "REAL"),
    ]
    _add_columns_if_missing(conn, "performance", enhanced_perf_cols)

    # v2.9.2: finbert_cache — migrate from integer-keyed to url_hash-keyed schema
    try:
        old_schema = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='finbert_cache'"
        ).fetchone()
        if old_schema and "url_hash" not in (old_schema[0] or ""):
            conn.execute("DROP TABLE IF EXISTS finbert_cache")
            logger.info("finbert_cache: dropped old integer-keyed table")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finbert_cache (
                url_hash TEXT PRIMARY KEY,
                article_id INTEGER,
                fb_positive REAL NOT NULL,
                fb_negative REAL NOT NULL,
                fb_neutral REAL NOT NULL,
                fb_label TEXT,
                fb_score REAL,
                scored_at TEXT NOT NULL
            )
        """)
    except Exception as e:
        logger.warning(f"finbert_cache migration: {e}")

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
    # SPY prediction model parameters
    ("prediction", "neutral_threshold", "0.004", "float", "0.001", "0.01", "Neutral band width for target labeling"),
    ("prediction", "lookback_days", "252", "int", "100", "1260", "Training window size in days"),
    ("prediction", "confidence_dampening_factor", "0.85", "float", "0.5", "1.0", "Confidence multiplier for choppy/bear regimes"),
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
