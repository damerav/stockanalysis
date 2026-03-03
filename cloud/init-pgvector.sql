-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- Analytics tables (migrated from DuckDB/SQLite)
-- ============================================================

CREATE TABLE IF NOT EXISTS prices (
    date DATE PRIMARY KEY,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    adjusted_close DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS technicals (
    date DATE PRIMARY KEY,
    sma_20 DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    bb_upper DOUBLE PRECISION,
    bb_lower DOUBLE PRECISION,
    bb_mid DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    obv DOUBLE PRECISION,
    garman_klass_vol DOUBLE PRECISION,
    stoch_k DOUBLE PRECISION,
    stoch_d DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS macro (
    date DATE PRIMARY KEY,
    vix DOUBLE PRECISION,
    vix_change DOUBLE PRECISION,
    us10y_yield DOUBLE PRECISION,
    dxy DOUBLE PRECISION,
    fed_funds DOUBLE PRECISION,
    gold DOUBLE PRECISION,
    crude DOUBLE PRECISION,
    vix9d DOUBLE PRECISION,
    vix3m DOUBLE PRECISION,
    vix6m DOUBLE PRECISION,
    vvix DOUBLE PRECISION,
    skew_index DOUBLE PRECISION,
    hy_spread DOUBLE PRECISION,
    tlt_spy_ratio DOUBLE PRECISION,
    eem_spy_ratio DOUBLE PRECISION,
    copper_gold_ratio DOUBLE PRECISION,
    xlk_xlf_ratio DOUBLE PRECISION,
    xlk_xle_ratio DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS intraday_bars (
    timestamp TEXT NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    vwap DOUBLE PRECISION,
    PRIMARY KEY (timestamp, ticker)
);

CREATE TABLE IF NOT EXISTS options_chain (
    date DATE NOT NULL,
    contract_symbol VARCHAR(50) NOT NULL,
    strike DOUBLE PRECISION,
    expiry DATE,
    option_type VARCHAR(4),
    last_price DOUBLE PRECISION,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION,
    volume BIGINT,
    open_interest BIGINT,
    iv DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION,
    theta DOUBLE PRECISION,
    vega DOUBLE PRECISION,
    PRIMARY KEY (date, contract_symbol)
);

-- ============================================================
-- Operational tables (migrated from SQLite)
-- ============================================================

CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    date TEXT,
    source TEXT,
    headline TEXT,
    summary TEXT,
    url TEXT UNIQUE,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS daily_sentiment (
    date DATE PRIMARY KEY,
    score DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    article_count INTEGER,
    positive_ratio DOUBLE PRECISION,
    negative_ratio DOUBLE PRECISION,
    neutral_ratio DOUBLE PRECISION,
    macro_sentiment DOUBLE PRECISION,
    earnings_sentiment DOUBLE PRECISION,
    geopolitical_sentiment DOUBLE PRECISION,
    technical_sentiment DOUBLE PRECISION,
    sentiment_dispersion DOUBLE PRECISION,
    sentiment_velocity DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS predictions (
    date DATE PRIMARY KEY,
    direction TEXT,
    confidence DOUBLE PRECISION,
    factors TEXT,
    report_text TEXT,
    predicted_at TEXT
);

CREATE TABLE IF NOT EXISTS options_analytics (
    date DATE PRIMARY KEY,
    put_call_ratio DOUBLE PRECISION,
    max_pain DOUBLE PRECISION,
    iv_skew DOUBLE PRECISION,
    gex DOUBLE PRECISION,
    vanna_exposure DOUBLE PRECISION,
    charm_exposure DOUBLE PRECISION,
    zero_dte_pcr DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS intraday_features (
    date DATE PRIMARY KEY,
    vwap_spread DOUBLE PRECISION,
    intraday_momentum DOUBLE PRECISION,
    intraday_range DOUBLE PRECISION,
    volume_ratio DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS performance (
    date DATE PRIMARY KEY,
    predicted TEXT,
    actual TEXT,
    correct INTEGER,
    cumulative_accuracy DOUBLE PRECISION,
    confidence_tier TEXT,
    vix_regime TEXT,
    day_of_week INTEGER,
    event_proximity INTEGER
);

CREATE TABLE IF NOT EXISTS feature_cache (
    date DATE PRIMARY KEY,
    features_json TEXT,
    computed_at TEXT
);

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name TEXT,
    model_path TEXT,
    accuracy DOUBLE PRECISION,
    test_accuracy DOUBLE PRECISION,
    feature_count INTEGER,
    training_date TEXT,
    metadata_json TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS earnings_calendar (
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    eps_estimate DOUBLE PRECISION,
    eps_actual DOUBLE PRECISION,
    surprise_pct DOUBLE PRECISION,
    market_cap_pct DOUBLE PRECISION,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS fed_communications (
    date DATE PRIMARY KEY,
    type TEXT,
    hawkish_score DOUBLE PRECISION,
    summary TEXT,
    scored_at TEXT
);

CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    name TEXT,
    role TEXT DEFAULT 'viewer',
    created_at TEXT,
    updated_at TEXT
);

-- ============================================================
-- News articles with vector embeddings (from news.db)
-- ============================================================

CREATE TABLE IF NOT EXISTS raw_articles (
    id SERIAL PRIMARY KEY,
    headline TEXT NOT NULL,
    source TEXT,
    url TEXT,
    published_at TEXT,
    category TEXT,
    ticker TEXT,
    summary TEXT,
    sentiment_compound DOUBLE PRECISION,
    sentiment_pos DOUBLE PRECISION,
    sentiment_neg DOUBLE PRECISION,
    sentiment_neu DOUBLE PRECISION,
    llm_sentiment DOUBLE PRECISION,
    blended_sentiment DOUBLE PRECISION,
    fetched_at TEXT,
    embedding vector(768)
);

CREATE TABLE IF NOT EXISTS finbert_cache (
    article_id INTEGER PRIMARY KEY REFERENCES raw_articles(id),
    fb_positive DOUBLE PRECISION,
    fb_negative DOUBLE PRECISION,
    fb_neutral DOUBLE PRECISION
);

-- ============================================================
-- Indexes for optimal query performance
-- ============================================================

-- Price lookups by date range (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

-- Technicals date lookup
CREATE INDEX IF NOT EXISTS idx_technicals_date ON technicals(date);

-- Macro date lookup
CREATE INDEX IF NOT EXISTS idx_macro_date ON macro(date);

-- Intraday bars: timestamp prefix search (WHERE timestamp LIKE '2025-01-%')
CREATE INDEX IF NOT EXISTS idx_intraday_timestamp ON intraday_bars(timestamp);
CREATE INDEX IF NOT EXISTS idx_intraday_ticker_ts ON intraday_bars(ticker, timestamp);

-- Options chain date lookup
CREATE INDEX IF NOT EXISTS idx_options_chain_date ON options_chain(date);

-- News: date + source for filtering
CREATE INDEX IF NOT EXISTS idx_news_date ON news(date);
CREATE INDEX IF NOT EXISTS idx_news_url ON news(url);

-- Sentiment date lookup
CREATE INDEX IF NOT EXISTS idx_sentiment_date ON daily_sentiment(date);

-- Predictions date lookup
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date);

-- Performance date + tiers for stratified queries
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date);
CREATE INDEX IF NOT EXISTS idx_performance_tier ON performance(confidence_tier);

-- Earnings calendar: date + ticker
CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_calendar(date);
CREATE INDEX IF NOT EXISTS idx_earnings_ticker ON earnings_calendar(ticker);

-- Fed communications date
CREATE INDEX IF NOT EXISTS idx_fed_date ON fed_communications(date);

-- Raw articles: published_at for time-range queries, category for filtering
CREATE INDEX IF NOT EXISTS idx_articles_published ON raw_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_category ON raw_articles(category);
CREATE INDEX IF NOT EXISTS idx_articles_source ON raw_articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_url ON raw_articles(url);

-- FinBERT cache article lookup
CREATE INDEX IF NOT EXISTS idx_finbert_article ON finbert_cache(article_id);

-- Vector similarity index (IVFFlat for fast approximate nearest neighbor)
-- Created after data is loaded for better index quality
-- CREATE INDEX IF NOT EXISTS idx_articles_embedding ON raw_articles
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
