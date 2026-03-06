-- ============================================================
-- TimescaleDB + pgvector initialization
-- Creates extensions, tables, hypertables, continuous
-- aggregates, compression policies, and retention policies.
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Time-series tables (will become hypertables)
-- ============================================================

-- Daily OHLCV prices
CREATE TABLE IF NOT EXISTS prices (
    date DATE NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    adjusted_close DOUBLE PRECISION,
    UNIQUE (date)
);

-- Technical indicators (daily)
CREATE TABLE IF NOT EXISTS technicals (
    date DATE NOT NULL,
    sma_20 DOUBLE PRECISION, sma_50 DOUBLE PRECISION, sma_200 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION, macd_signal DOUBLE PRECISION, macd_hist DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION, bb_lower DOUBLE PRECISION, bb_mid DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION, obv DOUBLE PRECISION, garman_klass_vol DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION, stoch_d DOUBLE PRECISION,
    -- pandas-ta extended indicators
    adx_14 DOUBLE PRECISION, cci_20 DOUBLE PRECISION,
    aroon_up DOUBLE PRECISION, aroon_down DOUBLE PRECISION,
    psar_long DOUBLE PRECISION, psar_short DOUBLE PRECISION,
    dpo_20 DOUBLE PRECISION, trix_14 DOUBLE PRECISION,
    vortex_pos DOUBLE PRECISION, vortex_neg DOUBLE PRECISION,
    williams_r DOUBLE PRECISION, mfi_14 DOUBLE PRECISION,
    rsi_2 DOUBLE PRECISION, rsi_9 DOUBLE PRECISION, rsi_21 DOUBLE PRECISION,
    cmo_14 DOUBLE PRECISION, ppo DOUBLE PRECISION,
    roc_5 DOUBLE PRECISION, roc_21 DOUBLE PRECISION,
    kc_upper_20 DOUBLE PRECISION, kc_lower_20 DOUBLE PRECISION,
    atr_7 DOUBLE PRECISION, atr_21 DOUBLE PRECISION,
    donchian_high DOUBLE PRECISION, donchian_low DOUBLE PRECISION,
    ulcer_14 DOUBLE PRECISION,
    cmf_20 DOUBLE PRECISION, vwma_20 DOUBLE PRECISION, eom_14 DOUBLE PRECISION,
    ema_9 DOUBLE PRECISION, ema_21 DOUBLE PRECISION, ema_200 DOUBLE PRECISION,
    hma_20 DOUBLE PRECISION, wma_20 DOUBLE PRECISION,
    dema_20 DOUBLE PRECISION, tema_20 DOUBLE PRECISION, kama_10 DOUBLE PRECISION,
    ichi_tenkan DOUBLE PRECISION, ichi_kijun DOUBLE PRECISION,
    ichi_senkou_a DOUBLE PRECISION, ichi_senkou_b DOUBLE PRECISION,
    UNIQUE (date)
);

-- Macro indicators (daily)
CREATE TABLE IF NOT EXISTS macro (
    date DATE NOT NULL,
    vix DOUBLE PRECISION, vix_change DOUBLE PRECISION,
    us10y_yield DOUBLE PRECISION, dxy DOUBLE PRECISION,
    fed_funds DOUBLE PRECISION, gold DOUBLE PRECISION, crude DOUBLE PRECISION,
    vix9d DOUBLE PRECISION, vix3m DOUBLE PRECISION, vix6m DOUBLE PRECISION,
    vvix DOUBLE PRECISION, skew_index DOUBLE PRECISION,
    hy_spread DOUBLE PRECISION, tlt_spy_ratio DOUBLE PRECISION,
    eem_spy_ratio DOUBLE PRECISION, copper_gold_ratio DOUBLE PRECISION,
    xlk_xlf_ratio DOUBLE PRECISION, xlk_xle_ratio DOUBLE PRECISION,
    us3m_yield DOUBLE PRECISION, yield_curve_10y3m DOUBLE PRECISION,
    sahm_rule DOUBLE PRECISION, consumer_conf DOUBLE PRECISION, ism_pmi DOUBLE PRECISION,
    xlk DOUBLE PRECISION, xlf DOUBLE PRECISION, xle DOUBLE PRECISION,
    xlv DOUBLE PRECISION, xli DOUBLE PRECISION, xlu DOUBLE PRECISION,
    xlb DOUBLE PRECISION, xlp DOUBLE PRECISION, xly DOUBLE PRECISION,
    xlre DOUBLE PRECISION, qqq DOUBLE PRECISION, iwm DOUBLE PRECISION,
    dia DOUBLE PRECISION,
    UNIQUE (date)
);

-- Intraday bars (5-min or finer — highest volume table)
CREATE TABLE IF NOT EXISTS intraday_bars (
    timestamp TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    vwap DOUBLE PRECISION,
    UNIQUE (timestamp, ticker)
);

-- Options chain snapshots (daily, multi-contract)
CREATE TABLE IF NOT EXISTS options_chain (
    date DATE NOT NULL,
    contract_symbol VARCHAR(50) NOT NULL,
    strike DOUBLE PRECISION,
    expiry DATE,
    option_type VARCHAR(4),
    last_price DOUBLE PRECISION,
    bid DOUBLE PRECISION, ask DOUBLE PRECISION,
    volume BIGINT, open_interest BIGINT,
    iv DOUBLE PRECISION, delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION, theta DOUBLE PRECISION, vega DOUBLE PRECISION,
    UNIQUE (date, contract_symbol)
);

-- Daily sentiment
CREATE TABLE IF NOT EXISTS daily_sentiment (
    date DATE NOT NULL,
    score DOUBLE PRECISION, confidence DOUBLE PRECISION,
    article_count INTEGER,
    positive_ratio DOUBLE PRECISION, negative_ratio DOUBLE PRECISION,
    neutral_ratio DOUBLE PRECISION,
    macro_sentiment DOUBLE PRECISION, earnings_sentiment DOUBLE PRECISION,
    geopolitical_sentiment DOUBLE PRECISION, technical_sentiment DOUBLE PRECISION,
    sentiment_dispersion DOUBLE PRECISION, sentiment_velocity DOUBLE PRECISION,
    UNIQUE (date)
);

-- Options analytics (daily)
CREATE TABLE IF NOT EXISTS options_analytics (
    date DATE NOT NULL,
    put_call_ratio DOUBLE PRECISION, max_pain DOUBLE PRECISION,
    iv_skew DOUBLE PRECISION, gex DOUBLE PRECISION,
    vanna_exposure DOUBLE PRECISION, charm_exposure DOUBLE PRECISION,
    zero_dte_pcr DOUBLE PRECISION,
    UNIQUE (date)
);

-- Intraday features (daily aggregates)
CREATE TABLE IF NOT EXISTS intraday_features (
    date DATE NOT NULL,
    vwap_spread DOUBLE PRECISION, intraday_momentum DOUBLE PRECISION,
    intraday_range DOUBLE PRECISION, volume_ratio DOUBLE PRECISION,
    UNIQUE (date)
);

-- Market breadth & fundamentals
CREATE TABLE IF NOT EXISTS market_breadth (
    date DATE NOT NULL,
    sp500_pe DOUBLE PRECISION, sp500_forward_pe DOUBLE PRECISION,
    sp500_earnings_yield DOUBLE PRECISION, sp500_dividend_yield DOUBLE PRECISION,
    pct_above_sma50 DOUBLE PRECISION, pct_above_sma200 DOUBLE PRECISION,
    advance_decline_ratio DOUBLE PRECISION,
    new_highs_52w INTEGER, new_lows_52w INTEGER,
    breadth_thrust DOUBLE PRECISION,
    sp500_cape DOUBLE PRECISION, buffett_indicator DOUBLE PRECISION,
    fear_greed_index INTEGER, trin DOUBLE PRECISION,
    UNIQUE (date)
);

-- Predictions
CREATE TABLE IF NOT EXISTS predictions (
    date DATE NOT NULL,
    direction TEXT, confidence DOUBLE PRECISION,
    factors TEXT, report_text TEXT, predicted_at TEXT,
    UNIQUE (date)
);

-- Performance tracking
CREATE TABLE IF NOT EXISTS performance (
    date DATE NOT NULL,
    predicted TEXT, actual TEXT, correct INTEGER,
    cumulative_accuracy DOUBLE PRECISION,
    confidence_tier TEXT, vix_regime TEXT,
    day_of_week INTEGER, event_proximity INTEGER,
    UNIQUE (date)
);

-- Backtest results
CREATE TABLE IF NOT EXISTS backtest_results (
    date DATE NOT NULL,
    predicted_direction TEXT, predicted_confidence DOUBLE PRECISION,
    actual_direction TEXT, actual_return DOUBLE PRECISION,
    correct INTEGER, regime TEXT, cumulative_accuracy DOUBLE PRECISION,
    UNIQUE (date)
);

-- ============================================================
-- Non-time-series tables (regular PostgreSQL)
-- ============================================================

CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    date TEXT, source TEXT, headline TEXT, summary TEXT,
    url TEXT UNIQUE, fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS raw_articles (
    id SERIAL PRIMARY KEY,
    headline TEXT NOT NULL, source TEXT, url TEXT,
    published_at TEXT, category TEXT, ticker TEXT, summary TEXT,
    sentiment_compound DOUBLE PRECISION,
    sentiment_pos DOUBLE PRECISION, sentiment_neg DOUBLE PRECISION,
    sentiment_neu DOUBLE PRECISION,
    llm_sentiment DOUBLE PRECISION, blended_sentiment DOUBLE PRECISION,
    fetched_at TEXT, quality_score DOUBLE PRECISION,
    embedding vector(768)
);

CREATE TABLE IF NOT EXISTS finbert_cache (
    article_id INTEGER PRIMARY KEY REFERENCES raw_articles(id),
    fb_positive DOUBLE PRECISION, fb_negative DOUBLE PRECISION,
    fb_neutral DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS news_features (
    date DATE PRIMARY KEY,
    features_json TEXT, computed_at TEXT
);

CREATE TABLE IF NOT EXISTS feature_cache (
    date DATE PRIMARY KEY,
    features_json TEXT, computed_at TEXT,
    feature_version TEXT
);

CREATE TABLE IF NOT EXISTS feature_store_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name TEXT, model_path TEXT,
    accuracy DOUBLE PRECISION, test_accuracy DOUBLE PRECISION,
    feature_count INTEGER, training_date TEXT,
    metadata_json TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    date DATE NOT NULL, ticker VARCHAR(10) NOT NULL,
    eps_estimate DOUBLE PRECISION, eps_actual DOUBLE PRECISION,
    surprise_pct DOUBLE PRECISION, market_cap_pct DOUBLE PRECISION,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS fed_communications (
    date DATE PRIMARY KEY,
    type TEXT, hawkish_score DOUBLE PRECISION,
    summary TEXT, scored_at TEXT
);

CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL, name TEXT,
    role TEXT DEFAULT 'viewer',
    created_at TEXT, updated_at TEXT
);

CREATE TABLE IF NOT EXISTS app_secrets (
    key TEXT PRIMARY KEY,
    encrypted_value TEXT NOT NULL,
    created_at TEXT, updated_at TEXT
);

CREATE TABLE IF NOT EXISTS strategy_rules (
    rule_group TEXT NOT NULL, rule_key TEXT NOT NULL,
    rule_value TEXT NOT NULL, value_type TEXT NOT NULL DEFAULT 'float',
    min_val TEXT, max_val TEXT,
    description TEXT, updated_at TEXT, updated_by TEXT,
    PRIMARY KEY (rule_group, rule_key)
);

CREATE TABLE IF NOT EXISTS strategy_rules_history (
    id SERIAL PRIMARY KEY,
    rule_group TEXT NOT NULL, rule_key TEXT NOT NULL,
    old_value TEXT, new_value TEXT NOT NULL,
    changed_at TEXT NOT NULL, changed_by TEXT NOT NULL DEFAULT 'system'
);

CREATE TABLE IF NOT EXISTS inverted_strangle_positions (
    id SERIAL PRIMARY KEY,
    trade_date TEXT NOT NULL, underlying TEXT NOT NULL DEFAULT 'SPY',
    spot_at_open DOUBLE PRECISION NOT NULL, expiry_date TEXT NOT NULL,
    dte_at_open INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN',
    short_put DOUBLE PRECISION NOT NULL, short_call DOUBLE PRECISION NOT NULL,
    long_put DOUBLE PRECISION NOT NULL, long_call DOUBLE PRECISION NOT NULL,
    inversion_pts DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    wing_pts DOUBLE PRECISION NOT NULL DEFAULT 25.0,
    initial_credit DOUBLE PRECISION NOT NULL, credit_per_leg TEXT,
    profit_target DOUBLE PRECISION NOT NULL,
    current_value DOUBLE PRECISION, current_pnl DOUBLE PRECISION,
    close_date TEXT, close_price DOUBLE PRECISION, close_reason TEXT,
    roll_count INTEGER NOT NULL DEFAULT 0, notes TEXT,
    vix_at_open DOUBLE PRECISION, position_delta DOUBLE PRECISION,
    entry_iv_rank DOUBLE PRECISION, entry_vix_term_structure DOUBLE PRECISION,
    cost_to_close DOUBLE PRECISION, c2c_updated_at TEXT,
    c2c_extrinsic DOUBLE PRECISION, c2c_intrinsic DOUBLE PRECISION,
    credit_vs_width DOUBLE PRECISION,
    loss_rule_2_1_breached INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS inverted_strangle_adjustments (
    id SERIAL PRIMARY KEY,
    position_id INTEGER NOT NULL, adj_date TEXT NOT NULL,
    adj_type TEXT NOT NULL,
    old_short_put DOUBLE PRECISION, old_short_call DOUBLE PRECISION,
    new_short_put DOUBLE PRECISION, new_short_call DOUBLE PRECISION,
    new_spot DOUBLE PRECISION, debit_paid DOUBLE PRECISION, notes TEXT,
    FOREIGN KEY (position_id) REFERENCES inverted_strangle_positions(id)
);

-- ============================================================
-- Convert time-series tables to hypertables
-- ============================================================

-- Daily tables: chunk by 1 month (good for ~20 rows/day)
SELECT create_hypertable('prices', 'date', chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('technicals', 'date', chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('macro', 'date', chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('daily_sentiment', 'date', chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('options_analytics', 'date', chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('intraday_features', 'date', chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('market_breadth', 'date', chunk_time_interval => INTERVAL '3 months',
                         if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('predictions', 'date', chunk_time_interval => INTERVAL '3 months',
                         if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('performance', 'date', chunk_time_interval => INTERVAL '3 months',
                         if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('backtest_results', 'date', chunk_time_interval => INTERVAL '3 months',
                         if_not_exists => TRUE, migrate_data => TRUE);

-- Intraday: chunk by 1 week (high volume — ~100K+ rows)
SELECT create_hypertable('intraday_bars', 'timestamp', chunk_time_interval => INTERVAL '1 week',
                         if_not_exists => TRUE, migrate_data => TRUE);

-- Options chain: chunk by 1 month (multi-contract daily snapshots)
SELECT create_hypertable('options_chain', 'date', chunk_time_interval => INTERVAL '1 month',
                         if_not_exists => TRUE, migrate_data => TRUE);

-- ============================================================
-- Continuous Aggregates
-- ============================================================

-- 1) Daily OHLCV summary from intraday bars (auto-maintained)
CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_intraday_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS date,
    ticker,
    first(open, timestamp)  AS open,
    max(high)               AS high,
    min(low)                AS low,
    last(close, timestamp)  AS close,
    sum(volume)             AS volume,
    last(vwap, timestamp)   AS vwap,
    -- Microstructure features computed at DB level
    max(high) - min(low)    AS intraday_range,
    sum(volume)::DOUBLE PRECISION /
        NULLIF(avg(volume) OVER (), 0) AS volume_ratio_raw,
    count(*)                AS bar_count
FROM intraday_bars
GROUP BY time_bucket('1 day', timestamp), ticker
WITH NO DATA;

-- 2) Weekly price aggregates (for multi-timeframe features)
CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_prices_weekly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 week', date) AS week,
    first(open, date)   AS open,
    max(high)            AS high,
    min(low)             AS low,
    last(close, date)    AS close,
    sum(volume)          AS volume,
    avg(close)           AS avg_close
FROM prices
GROUP BY time_bucket('1 week', date)
WITH NO DATA;

-- 3) Monthly price aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_prices_monthly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', date) AS month,
    first(open, date)   AS open,
    max(high)            AS high,
    min(low)             AS low,
    last(close, date)    AS close,
    sum(volume)          AS volume,
    avg(close)           AS avg_close
FROM prices
GROUP BY time_bucket('1 month', date)
WITH NO DATA;

-- 4) Hourly intraday aggregates (for intraday pattern analysis)
CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_intraday_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS hour,
    ticker,
    first(open, timestamp)  AS open,
    max(high)               AS high,
    min(low)                AS low,
    last(close, timestamp)  AS close,
    sum(volume)             AS volume,
    last(vwap, timestamp)   AS vwap,
    count(*)                AS bar_count
FROM intraday_bars
GROUP BY time_bucket('1 hour', timestamp), ticker
WITH NO DATA;

-- 5) Daily options summary (aggregate across all contracts)
CREATE MATERIALIZED VIEW IF NOT EXISTS cagg_options_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', date) AS date,
    count(*)                                    AS contract_count,
    sum(volume)                                 AS total_volume,
    sum(open_interest)                          AS total_oi,
    avg(iv)                                     AS avg_iv,
    sum(CASE WHEN option_type = 'put' THEN volume ELSE 0 END)::DOUBLE PRECISION /
        NULLIF(sum(CASE WHEN option_type = 'call' THEN volume ELSE 0 END), 0)
                                                AS put_call_volume_ratio,
    sum(CASE WHEN option_type = 'put' THEN open_interest ELSE 0 END)::DOUBLE PRECISION /
        NULLIF(sum(CASE WHEN option_type = 'call' THEN open_interest ELSE 0 END), 0)
                                                AS put_call_oi_ratio
FROM options_chain
GROUP BY time_bucket('1 day', date)
WITH NO DATA;

-- ============================================================
-- Refresh policies for continuous aggregates
-- ============================================================

SELECT add_continuous_aggregate_policy('cagg_intraday_daily',
    start_offset => INTERVAL '3 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('cagg_prices_weekly',
    start_offset => INTERVAL '2 weeks',
    end_offset   => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('cagg_prices_monthly',
    start_offset => INTERVAL '2 months',
    end_offset   => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('cagg_intraday_hourly',
    start_offset => INTERVAL '3 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('cagg_options_daily',
    start_offset => INTERVAL '7 days',
    end_offset   => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ============================================================
-- Compression policies (older data gets compressed)
-- ============================================================

-- Intraday bars: compress after 7 days (highest volume)
ALTER TABLE intraday_bars SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'timestamp DESC'
);
SELECT add_compression_policy('intraday_bars', INTERVAL '7 days', if_not_exists => TRUE);

-- Options chain: compress after 30 days
ALTER TABLE options_chain SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'contract_symbol',
    timescaledb.compress_orderby = 'date DESC'
);
SELECT add_compression_policy('options_chain', INTERVAL '30 days', if_not_exists => TRUE);

-- Daily tables: compress after 6 months (small volume, keep recent fast)
ALTER TABLE prices SET (timescaledb.compress, timescaledb.compress_orderby = 'date DESC');
SELECT add_compression_policy('prices', INTERVAL '6 months', if_not_exists => TRUE);

ALTER TABLE technicals SET (timescaledb.compress, timescaledb.compress_orderby = 'date DESC');
SELECT add_compression_policy('technicals', INTERVAL '6 months', if_not_exists => TRUE);

ALTER TABLE macro SET (timescaledb.compress, timescaledb.compress_orderby = 'date DESC');
SELECT add_compression_policy('macro', INTERVAL '6 months', if_not_exists => TRUE);

ALTER TABLE daily_sentiment SET (timescaledb.compress, timescaledb.compress_orderby = 'date DESC');
SELECT add_compression_policy('daily_sentiment', INTERVAL '6 months', if_not_exists => TRUE);

-- ============================================================
-- Retention policy: drop raw intraday bars older than 1 year
-- (continuous aggregates preserve the daily/hourly summaries)
-- ============================================================

SELECT add_retention_policy('intraday_bars', INTERVAL '1 year', if_not_exists => TRUE);

-- ============================================================
-- Indexes (TimescaleDB auto-creates time index on hypertables)
-- ============================================================

-- Additional indexes for non-time lookups
CREATE INDEX IF NOT EXISTS idx_intraday_ticker ON intraday_bars(ticker, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_options_contract ON options_chain(contract_symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_articles_published ON raw_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_category ON raw_articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_source ON raw_articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_url ON raw_articles(url);
CREATE INDEX IF NOT EXISTS idx_news_url ON news(url);
CREATE INDEX IF NOT EXISTS idx_news_date ON news(date);
CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_calendar(date);
CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings_calendar(ticker);

-- ============================================================
-- Initial refresh of continuous aggregates
-- ============================================================

CALL refresh_continuous_aggregate('cagg_intraday_daily', NULL, NULL);
CALL refresh_continuous_aggregate('cagg_prices_weekly', NULL, NULL);
CALL refresh_continuous_aggregate('cagg_prices_monthly', NULL, NULL);
CALL refresh_continuous_aggregate('cagg_intraday_hourly', NULL, NULL);
CALL refresh_continuous_aggregate('cagg_options_daily', NULL, NULL);
