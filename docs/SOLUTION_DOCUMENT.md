# Stock Analysis Platform — Solution Document

## 1. System Overview

The Stock Analysis Platform is a real-time market analysis and signal generation system built for an NVIDIA DGX Spark local GPU server with optional AWS cloud mirroring. It combines quantitative analysis, machine learning, and large language model inference to produce daily SPY/SPX direction predictions and intraday ES futures trading signals.

The system is signal-only — all trade execution is manual.

### Two Subsystems

1. **SPY/SPX Predictor** — Ingests multi-source market data, engineers 85 features across 8 categories, trains an XGBoost classifier daily on GPU (with optional BiLSTM+LightGBM stacking ensemble), and predicts next-day SPY direction on a 5-level scale (STRONG BULLISH to STRONG BEARISH). Uses a local 70B LLM for news sentiment scoring. Includes conformal prediction sets, HMM regime detection, earnings calendar integration, and Fed communication NLP.

2. **ES Futures Strategy** — Generates entry/exit signals for E-mini S&P 500 futures using Keltner Channel bands, a 3-lot tiered exit system, adaptive volatility regimes, and optional AI-enhanced entry/exit models.

Both subsystems share data infrastructure, dashboards, cloud sync, and the daily automated pipeline.

---

## 2. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Runtime | Python 3.12 | Application language |
| Database | SQLite 3.45 (WAL mode) | Local data store, 17 tables |
| ML Framework | XGBoost 2.0 (GPU) | SPY direction classifier, ES entry gate |
| Deep Learning | PyTorch 2.1 | CNN exit controller for ES strategy |
| LLM | Ollama + DeepSeek R1 70B | News sentiment analysis, report generation, what-if narratives |
| Dashboard | Streamlit 1.30 + Plotly 5.18 | Unified web UI on port 8501 |
| AI API | FastAPI + Uvicorn + Pydantic | Real-time confidence API on port 8100 |
| Data Sources | Polygon.io, yfinance, Finnhub, FRED, RSS | Market data, news, macro indicators |
| Cloud Relay | FastAPI + Uvicorn | Stateless relay on AWS EC2 |
| Containers | Docker + Docker Compose | Cloud deployment packaging |
| GPU | NVIDIA GB10 (DGX Spark) | XGBoost training, PyTorch inference, LLM hosting |

### Python Dependencies

```
websockets, aiohttp, yfinance, feedparser, requests, pandas, numpy,
pyyaml, xgboost, scikit-learn, torch, streamlit, plotly, fastapi, uvicorn,
pydantic, google-auth, google-auth-oauthlib, shap, transformers, scipy,
prometheus_client, hmmlearn, lightgbm
```

---

## 3. Component Inventory

### 3.1 Data Layer (`src/data/`)

#### `init_db.py` — Database Schema Manager
Creates and manages the SQLite database with 13 application tables (plus internal tables for feature store and model registry). Provides `get_connection()` for all database access with WAL journal mode and 5-second busy timeout. Includes automatic schema migration (`_migrate_schema`) that adds new columns and tables for P1/P2/P3 enhancements on every connection. Tables cover prices, technicals, news, sentiment, macro indicators, predictions, intraday bars, options chain, options analytics, intraday features, performance tracking, earnings calendar, and Fed communications.

#### `polygon_fetcher.py` — Polygon.io REST Client
Primary data source for market data. Provides:
- `get_daily_bars()` — daily OHLCV with adjusted prices
- `get_5s_bars()` — intraday 5-second granularity
- `get_options_chain()` — full chain with Greeks (delta, gamma, theta, vega, IV)
- `get_options_analytics()` — computed put/call ratio, max pain, IV skew, GEX, vanna exposure, charm exposure, zero-DTE put/call ratio (P3)

Includes retry logic (3 attempts, exponential backoff), pagination handling, and rate limit awareness (100 req/min).

#### `fetcher.py` — Fallback Data Sources
Provides alternative data when Polygon is unavailable:
- yfinance: price data fallback
- Finnhub: market news headlines
- RSS feeds: Yahoo Finance, CNBC, MarketWatch
- FRED: macro data (VIX, 10Y yield, DXY, fed funds, gold, crude oil)

#### `daily_pull.py` — Gap Detection and Backfill
Identifies missing dates in the database and backfills from Polygon (with yfinance fallback). Fetches macro data, updates options chain, computes technicals for all new dates, and validates data completeness. Called from the launcher scheduler, daily pipeline, and realtime pre-market.

#### `backfill.py` — Initial Bulk Load
First-time setup utility that loads 252 trading days (1 year) of historical data. Run once: `python -m src.data.backfill --days 252`.

#### `features.py` — Feature Engineering
Builds an 85-feature vector for each trading day from 8 categories:

| Category | Features | Source |
|----------|----------|--------|
| Technical | price_vs_sma20/50, rsi_14, macd, macd_signal, macd_hist, bb_upper/lower_dist, atr_14, sma slopes | technicals table |
| Macro | vix, vix_change, us10y_yield, dxy, fed_funds, gold, crude | macro table |
| VIX Term Structure (P1) | vix9d, vix3m, vix6m, vvix, skew_index, vix_term_slope, vix_term_curve, vix_realised_ratio | macro table |
| Cross-Asset (P1) | hy_spread, tlt_spy_ratio, eem_spy_ratio, copper_gold_ratio, xlk_xlf_ratio, xlk_xle_ratio | macro table |
| Sentiment | sentiment_score, article_count, positive/negative_ratio | daily_sentiment |
| Decomposed Sentiment (P2) | macro_sentiment, earnings_sentiment, geopolitical_sentiment, technical_sentiment, sentiment_dispersion, sentiment_velocity | daily_sentiment |
| Intraday | vwap_spread, intraday_momentum, intraday_range, volume_ratio | intraday_features |
| Options | put_call_ratio, max_pain_distance, iv_skew, gex_normalized | options_analytics |
| Extended Options (P3) | vanna_exposure, charm_exposure, zero_dte_pcr, gex_sign_change, max_pain_velocity, vanna_normalized, charm_normalized | options_analytics (derived) |
| Calendar/Events (P1) | days_to_fomc, is_fomc_week, is_fomc_day, days_to_cpi, days_to_nfp, days_to_opex, is_triple_witching, is_quarter_end, day_of_week, week_of_month | calendar module |
| Earnings (P3) | earnings_density, days_to_next_mega, earnings_week | earnings_calendar table |
| Fed Comms (P3) | fomc_hawkish_score, beige_book_score, fed_sentiment_avg | fed_communications table |
| Context (GAP 8) | vix_percentile, spy_es_zscore, rth_flag, minutes_to_close, event_proximity | derived |
| Derived | price_vs_sma_pct, rsi_divergence, volume_trend, atr_percentile, momentum_5d/10d | computed |

Also provides `get_target()` with adaptive VIX-scaled neutral threshold, and `get_feature_columns()` for listing all 85 features.

#### `earnings_calendar.py` — Earnings Calendar Integration (P3)
Fetches upcoming earnings dates for top 20 S&P 500 mega-caps from yfinance. Computes three features per trading day:
- `earnings_density`: number of mega-caps reporting within ±3 days
- `days_to_next_mega`: days until next mega-cap earnings (capped at 30)
- `earnings_week`: binary flag if any mega-cap reports this week

#### `fed_comms.py` — Federal Reserve Communication NLP (P3)
Fetches FOMC statements and Beige Book summaries from the Fed's RSS feeds. Scores each communication on a hawkish (+1) to dovish (-1) scale using the LLM (with keyword fallback). Computes three features:
- `fomc_hawkish_score`: most recent FOMC statement score
- `beige_book_score`: most recent Beige Book score
- `fed_sentiment_avg`: average of both scores

Only runs ~8 times per year (after each FOMC meeting). Cached in `fed_communications` table.

#### `calendar.py` — Economic Event Calendar (P1)
Provides `get_event_features()` returning 10 calendar-based features: days to FOMC/CPI/NFP/OpEx, triple witching flag, quarter-end flag, day of week, week of month.

#### `drift_monitor.py` — Feature Drift Detection (P1)
Computes Population Stability Index (PSI) and Kolmogorov-Smirnov tests between training and current feature distributions. PSI > 0.2 triggers alerts for model refit.

#### `feature_store.py` — Feature Cache with Version Tracking (P2)
SQLite-backed feature cache that stores computed feature vectors with version hashing. Avoids redundant recomputation during training and inference.

---

### 3.2 LLM Layer (`src/llm/`)

#### `analyzer.py` — LLM Sentiment Analyzer
Manages the Ollama/DeepSeek R1 70B integration:
- `check_health()` — verifies Ollama is running, model is available, inference works
- Auto-starts Ollama if not running
- Auto-downloads the model (~42GB) with progress logging
- `analyze_sentiment()` — batch processes up to 50 news articles, returns structured JSON per article with score (-1.0 to 1.0), confidence (0-100), and topics
- Aggregates daily sentiment as weighted average by recency and source credibility
- Graceful degradation: if LLM unavailable, returns neutral sentiment (score=0.0)

#### `reporter.py` — Daily Report Generator
Generates a ~400-word market brief using the LLM. Input: day's technicals, sentiment, model prediction, key signals. Output stored in `predictions.report_text`. Falls back to a template-based summary if LLM is unavailable.

---

### 3.3 Model Layer (`src/model/`)

#### `trainer.py` — XGBoost SPY Direction Predictor
- Target: next-day SPY direction — UP (+1) / DOWN (-1) / NEUTRAL (0)
- Neutral threshold: ±0.3% daily return (adaptive with VIX scaling)
- Objective: `multi:softprob` with 3 classes
- GPU acceleration: `tree_method='gpu_hist'`
- Training window: adaptive selection from [63, 126, 252, 504] days (P2)
- Validation: purged walk-forward time-series split (P2) — no data leakage
- Early stopping: 50 rounds on validation loss
- Output: 5-level prediction scale (STRONG BULLISH to STRONG BEARISH) with 0-100% confidence
- SHAP explanation: top prediction drivers with feature importance (P1)
- Model persistence: `./models/xgb_spy_{date}.json`
- Integrates ensemble, conformal prediction, regime detection, and model registry (P2)

#### `bilstm_model.py` — Bidirectional LSTM Classifier (P2)
PyTorch BiLSTM sequence model for 3-class direction prediction. Sklearn-compatible wrapper (`BiLSTMClassifier`) for use in the stacking ensemble. Configurable sequence length, hidden dimensions, dropout, and training epochs.

#### `ensemble.py` — Stacking Ensemble (P2)
XGBoost + BiLSTM + LightGBM stacking ensemble with LogisticRegression meta-learner. Each base model produces 3-class probabilities; the meta-learner combines them for final prediction. Optional — enabled via `config.yaml` `ensemble.enabled: true`.

#### `conformal.py` — Conformal Prediction Sets (P2)
Provides calibrated prediction sets with guaranteed coverage. At 90% significance, outputs a set of plausible classes (e.g., {UP, NEUTRAL}). Flags low-conviction predictions when the set contains multiple classes.

#### `regime.py` — HMM Regime Detection (P2)
4-state Hidden Markov Model that classifies market regimes: bull_trend, bear_trend, high_vol_choppy, low_vol_range. Uses StandardScaler + diagonal covariance for robustness. Regime is displayed on the dashboard and used to contextualize predictions.

#### `purged_cv.py` — Purged Walk-Forward Cross-Validation (P2)
Time-series cross-validation with purge gaps between train and test sets to prevent data leakage. Configurable number of splits and purge window.

#### `adaptive_window.py` — Adaptive Training Window Selection (P2)
Selects the optimal training window from candidates [63, 126, 252, 504] days by evaluating validation accuracy for each. Adapts to changing market conditions.

#### `registry.py` — SQLite-Backed Model Registry (P2)
Tracks all trained models with metadata: training date, validation/test accuracy, feature count, deployment status. Supports gating (blocking underperforming models) and active model selection.

---

### 3.4 Realtime Layer (`src/realtime/`)

#### `streamer.py` — Polygon WebSocket Streamer
Connects to Polygon Advanced WebSocket for real-time data:
- Stocks channel: subscribes to `T.SPY` for real-time trades
- Options channel: subscribes to `T.O:SPX*` for all SPX options trades
- Aggregates raw ticks into 5-second OHLCV bars
- Detects options sweeps (same option, multiple exchanges, <2 sec, >$50K) and block trades (>$100K)
- Computes live put/call ratio
- Auto-reconnect with exponential backoff (1s → 30s max)
- Heartbeat monitoring: reconnects if no data for 60 seconds during market hours

#### `dashboard_bridge.py` — State File Writer
Bridges backend computation to dashboard display:
- Writes `spy_state.json` and `es_state.json` to `./data/`
- Atomic writes (temp file → rename) to prevent partial reads
- Dashboards read these files on each refresh cycle

---

### 3.5 ES Strategy Layer (`src/es_strategy/`)

#### `indicators.py` — Technical Indicators
Computes: ATR(14), Keltner Channel (EMA20 ± 1.5×ATR), EMA(9), VWAP, RSI(14), ROC(3).

Includes `RegimeDetector`: classifies volatility as Low/Med/High based on ATR percentile over 10,080 bars with 3-bar hysteresis to prevent rapid regime switching.

Includes `CuMLRegimeClassifier`: cuML LogisticRegression-based regime classifier (falls back to sklearn if cuML unavailable). Trained on ATR percentile features, outputs Low/Mid/High classification.

#### `position.py` — 3-Lot Position Manager
Manages a tiered position with 3 lots:
- Lot 0: TP1 target, tightest trailing stop
- Lot 1: TP2 target, medium trail
- Lot 2: Runner, widest trail for extended moves

Tracks entry price, direction, per-lot status, stop levels, and P&L. No pyramiding — must be flat to enter.

#### `engine.py` — Strategy Engine
Core signal generation logic:
- Phase 1: pure-edge entry at K±C → TP1 → TP2 → Runner cascade
- Phase 2: confluence reload requiring K±C plus 2 of 3 filters (ROC, ATR expansion, VWAP alignment)
- Emergency stop: 20% × C, always active
- Jump exit: 5 points adverse during 1-minute hold period
- Session guards: flatten before 15:55 CT, flatten before FOMC/CPI/NFP events
- Circuit breaker: daily P&L ≤ -$2,000 → flatten all + disable until 17:00 CT reset

#### `ai_models.py` — AI Entry Gate and Exit Controller
Optional AI enhancement (enabled via `--ai` flag):

**XGBoost Entry Gate**: 17 normalized features, triple-barrier meta-labels, regime-adaptive thresholds (Low=0.58, Med=0.55, High=0.52). Outputs `p_enter` probability and position sizing (1-3 lots).

**CNN Exit Controller**: 1D-CNN (Conv1d → AdaptiveAvgPool → FC → sigmoid) over last 20 bars × 19 features. Outputs `P_cont_5` (probability price continues 5 more bars), maps to trailing stop multipliers within regime bounds.

**Drift Monitor**: tracks PSI (Population Stability Index). PSI > 0.2 triggers size reduction and refit. If AI underperforms rules by 1σ over 100 trades, AI is auto-disabled.

#### `runner.py` — Live/Backtest/Paper Runner
Execution modes:
- `--mode live` — Polygon real-time feed
- `--mode backtest --data file.csv` — historical CSV replay
- `--mode paper` — live data, signals logged only (no execution)
- `--ai` flag enables AI entry gate + CNN exit controller

#### `rl_trail.py` — Q-Learning Trailing Stop Agent
Tabular Q-learning agent that adaptively adjusts the runner lot's trailing stop:
- State: [regime, ATR percentile, unrealized P&L, bars held, RSI, ROC] — discretized into 9,375 states
- Actions: tighten (-0.1×ATR), hold, widen (+0.1×ATR)
- Reward: ΔEquity − λ×Drawdown (λ=0.5 default)
- Integrated into `engine.py` — adjusts runner trail each bar, updates Q-table, saves/loads from `models/rl_trail_qtable.json`

#### `labeling.py` — Triple-Barrier Entry and Exit Labels
Generates training labels per consolidated requirements:
- Entry: triple-barrier method — TP1 hit before emergency stop within 60 bars → 1, else → 0
- Exit: future adverse move ≥ 0.25×ATR within 5 bars → 1 (reversal), else → 0
- Integrated into `trainer.py` via `train_es_entry()` and `train_es_exit()` methods

### 3.5b AI Confidence API (`src/api/`)

#### `confidence_server.py` — FastAPI Real-Time Inference (port 8100)
Three endpoints for MT5/FxDreema integration:
- `POST /confidence` — entry gate inference returning `entry_conf`, `vol_regime`, `advice` (allow/block)
- `POST /exit` — exit controller returning `exit_conf_reversal`, `tp2_trail_atr`, `runner_trail_atr`
- `POST /spread` — dynamic spread update (strike_K, credit_C) from broker or manual entry
- `GET /health` — model load status and uptime

Features: fail-closed logic (blocks trades if AI unavailable), configurable entry threshold (default 0.70), JSONL audit logging to `./logs/trade_audit.jsonl`, latency measurement, feature vector hashing for auditability.

---

### 3.6 Dashboard Layer (`src/dashboard/`)

#### `app.py` — Unified Dashboard (port 8501)
Single Streamlit application with sidebar navigation across four pages:

1. **SPY Predictor**: prediction banner, P2 regime/conformal/ensemble info row, P3 earnings/Fed/extended-greeks row, SHAP prediction drivers (P1), history chart, stratified accuracy tracking (P1), indicators, options flow alerts. Auto-refreshes every 15 seconds.

2. **ES Strategy**: position banner, candlestick chart with Keltner Channel overlay, signal feed, status panel. Auto-refreshes every 5 seconds.

3. **What-If Analysis**: interactive scenario testing with ES parameter sweeps, SPY feature overrides, Monte Carlo simulations, stress tests, and feature ablation.

4. **Admin Console**: browser-based system management with 5 tabs:
   - System Status: health monitoring for database, LLM, and XGBoost model; data inventory with row counts and date ranges for all 13 application tables; latest prediction display; P2 model registry
   - Actions: ad-hoc execution of individual pipeline steps (data pull, news fetch, macro fetch, compute technicals, retrain XGBoost, generate prediction, LLM health check, generate report, full pipeline with skip-LLM option, send test alert)
   - Database: table browser with configurable limits, custom SQL queries (SELECT-only), vacuum, and integrity check
   - Configuration: key settings overview, full YAML editor with validation, save to disk
   - Logs: dashboard log viewer, pipeline history with report text, model file inventory

5. **Monitoring**: native Plotly monitoring dashboards replicating Grafana panels (SPY Predictor, ES Strategy, System Health, Confidence API, Pipeline Status) with dark theme.

6. **Grafana (compare)**: embedded Grafana iframes with Google OAuth proxy support for side-by-side comparison.

Auto-detects mode: if `RELAY_URL` environment variable is set, fetches data from cloud relay; otherwise reads local JSON state files.

#### `realtime_app.py` — Standalone SPY Dashboard (legacy, port 8501)
Original standalone SPY dashboard. Retained for backward compatibility.

#### `es_dashboard.py` — Standalone ES Dashboard (legacy, port 8502)
Original standalone ES dashboard. Retained for backward compatibility.

#### `whatif_app.py` — Standalone What-If Dashboard (legacy, port 8503)
Original standalone What-If dashboard. Retained for backward compatibility.

---

### 3.7 What-If Layer (`src/whatif/`)

#### `engine.py` — What-If Compute Engine
Core analysis module running on DGX (requires GPU + data + models):

| Method | Description |
|--------|-------------|
| `es_parameter_sweep(params_grid)` | Backtest ES strategy across a grid of parameter combinations |
| `es_compare_scenarios(scenario_list)` | Side-by-side backtest comparison of named scenarios |
| `spy_scenario_inject(overrides)` | Override specific features and re-run XGBoost inference |
| `spy_feature_ablation(drop_list)` | Zero out features and measure accuracy impact |
| `spy_monte_carlo(n_sims, noise_pct)` | Random perturbation of features across N simulations |
| `market_stress_test(scenario_name)` | Run pre-built stress scenarios |
| `llm_explain(result)` | Generate LLM narrative for what-if results |
| `list_stress_scenarios()` | List available pre-built stress scenarios |

#### `presets.py` — Stress Test Scenarios
Five pre-built market stress scenarios:

| Name | Description |
|------|-------------|
| `vix_spike_40` | VIX jumps to 40, sentiment drops to -0.8 |
| `gap_down_3pct` | 3% gap down with elevated VIX and negative sentiment |
| `march_2020_crash` | March 2020 crash conditions (VIX 65, extreme fear) |
| `fed_rate_cut` | Fed rate cut with positive sentiment and lower yields |
| `melt_up` | Melt-up rally with low VIX and extreme bullish sentiment |

#### `narrator.py` — LLM What-If Narrator
Sends what-if results to DeepSeek R1 70B for plain-English explanation. Supports both ES strategy and SPY predictor results. Temperature 0.5, max 800 tokens. Falls back to raw numbers if LLM is unavailable.

---

### 3.8 Pipeline Layer (`src/pipeline/`)

#### `daily_run.py` — 15-Step Pipeline Orchestrator
Sequential pipeline running at 4:30 PM ET Monday-Friday:

| Step | Name | Description |
|------|------|-------------|
| 0 | LLM Check | Verify Ollama + model availability |
| 0.5 | Data Pull | Gap detection + backfill missing dates |
| 1 | Evaluate | Compare yesterday's prediction to actual outcome |
| 2 | Prices | Fetch today's OHLCV (Polygon → yfinance fallback) |
| 3 | News | Fetch headlines (Finnhub + RSS) |
| 4 | Sentiment | LLM sentiment analysis with decomposed categories (P2) |
| 5 | Macro | Fetch VIX, yields, DXY, fed funds, gold, crude + VIX term structure + cross-asset signals (P1) |
| 6 | Options Chain | Fetch SPX options chain snapshot |
| 7 | Options Analytics | Compute P/C ratio, max pain, IV skew, GEX + vanna, charm, 0DTE PCR (P3) |
| 8 | Technicals | Compute SMA, RSI, MACD, BB, ATR |
| 9 | Intraday | Build VWAP spread, momentum, range features |
| 9.5 | Earnings Calendar | Fetch mega-cap earnings dates from yfinance (P3) |
| 9.6 | Fed Communications | Fetch + score FOMC statements and Beige Book (P3) |
| 10 | Retrain | Build 85-feature vector + retrain XGBoost on GPU (P2: with feature store, regime, adaptive window, ensemble, conformal, registry) |
| 11 | Predict | Generate next-day prediction with SHAP drivers (P1) + conformal set (P2) |
| 12 | Report | Generate LLM daily report |
| 13 | Alerts | Send Telegram + email notifications |

If any step fails, the pipeline logs the error and continues to the next step. LLM failure at Step 4 results in neutral sentiment (score=0.0) — the pipeline never aborts.

#### `alerts.py` — Notification System
Sends prediction alerts via two optional channels:
- Telegram: Bot API with formatted message (emoji, direction, confidence, report)
- Email: SMTP with HTML template

Both channels are independently configured and optional.

---

### 3.9 Cloud Sync Layer (`src/sync/`)

#### `publisher.py` — Cloud State Publisher
Runs on DGX alongside backends. Pushes state updates to AWS relay via HTTPS POST:
- `/push/prediction` — SPY prediction updates
- `/push/flow_alert` — options flow alerts
- `/push/es_state` — ES strategy state
- `/push/heartbeat` — 30-second keepalive

API key authentication via `X-API-Key` header. Retry on failure (3 attempts, 5s backoff). No-op if sync is disabled in config.

#### `relay_server.py` — FastAPI Cloud Relay
Runs on AWS EC2 t3.micro (~$8/mo). In-memory state only — no database. Receives POST pushes from DGX, serves state to cloud dashboards via GET endpoints and Server-Sent Events. Stale detection: marks source OFFLINE if no heartbeat for 90 seconds.

---

### 3.10 Launcher (`src/launcher.py`)

Single entry point for the entire system. Contains:
- `ProcessManager`: starts, stops, monitors, and restarts child processes
- `Scheduler`: background thread that triggers the daily pipeline at 4:30 PM ET
- `SystemLauncher`: orchestrates LLM check, ES strategy runner, dashboards, and scheduler
- CLI with mutually exclusive flags: `--all`, `--spy`, `--es`, `--dashboards-only`, `--check-llm`, `--pipeline`

---

### 3.11 Cloud Deployment (`cloud/`)

| File | Purpose |
|------|---------|
| `Dockerfile` | Combined relay + dashboard container |
| `Dockerfile.relay` | Relay-only container (FastAPI + Uvicorn) |
| `Dockerfile.dashboard` | Dashboard-only container (Streamlit) |
| `docker-compose.yml` | Two-service orchestration (relay:8000, dashboard:8501) |
| `deploy_aws.sh` | Automated ECR + EC2 deployment script |
| `start.sh` | Combined container entrypoint |
| `start_dashboards.sh` | Dashboard container entrypoint |
| `.env.example` | Environment variables template |

---

### 3.12 Operations Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `start.sh` | Start all components in order: Ollama, LLM check, dashboard (8501), ES runner (paper), scheduler (4:30 PM ET). Uses `.pids/` for PID tracking and `logs/` for log files. |
| `stop.sh` | Graceful shutdown in reverse order with 10-second timeout then force-kill. `--all` flag also stops Ollama. Cleans up orphaned processes on port 8501. |
| `status.sh` | Full health check: process status, port status, database health (size, row counts, latest dates), model info, disk usage, recent log activity. |
| `restart.sh` | Runs `stop.sh` then `start.sh` with a 2-second pause. Passes through arguments (e.g., `--all`). |

Scripts are self-contained — they auto-detect the project directory relative to their location and activate the Python virtual environment as needed.

---

## 4. Data Flow

### Daily Pipeline Flow

```
Polygon/yfinance → prices table
Finnhub/RSS      → news table
FRED             → macro table (+ VIX term structure, cross-asset signals)
Polygon          → options_chain, options_analytics tables (+ vanna, charm, 0DTE)
yfinance         → earnings_calendar table (P3)
Fed RSS          → fed_communications table (P3, LLM-scored)
                        │
                        ▼
              Feature Engineering (85 features, 8 categories)
                        │
                        ▼
              XGBoost Training (GPU, adaptive window)
              + Optional Ensemble (XGB + BiLSTM + LightGBM)
              + Conformal Prediction Sets
              + HMM Regime Detection
                        │
                        ▼
              Prediction → predictions table
                        │
                        ▼
              LLM Report → predictions.report_text
                        │
                        ▼
              Alerts → Telegram / Email
```

### Realtime Data Flow

```
Polygon WebSocket → 5-sec bars → intraday_bars table
                  → options flow → sweep/block detection
                        │
                        ▼
              Dashboard Bridge → spy_state.json / es_state.json
                        │
                        ▼
              Streamlit Dashboard (port 8501)
                        │
              Cloud Publisher (optional)
                        │
                        ▼
              AWS Relay → Cloud Dashboard
```

---

## 5. Deployment Topology

### Local (Primary)

All components run on the DGX Spark at `192.168.1.211`:
- Python processes managed by `src/launcher.py` or `scripts/start.sh`
- SQLite database at `./data/spy.db`
- Ollama serving DeepSeek R1 70B on localhost:11434
- Unified dashboard on port 8501 (SPY Predictor, ES Strategy, What-If Analysis, Admin Console)
- Operations scripts in `scripts/` for start/stop/status/restart
- Code synced from Windows dev machine via Mutagen

### Cloud (Optional Mirror)

AWS EC2 t3.micro running Docker containers:
- FastAPI relay on port 8000 (receives state pushes from DGX)
- Streamlit dashboard on port 8501 (reads from relay)
- No GPU, no database, no model inference — display only
- Estimated cost: ~$8-10/month

---

## 6. Failure Modes and Resilience

| Failure | System Behavior |
|---------|----------------|
| Polygon WebSocket disconnects | Auto-reconnect with exponential backoff (1s → 30s) |
| Polygon REST unavailable | Falls back to yfinance for price data |
| Ollama not running | Auto-start attempted; if fails, continues with neutral sentiment |
| DeepSeek model not downloaded | Auto-download with progress logging; if fails, neutral sentiment |
| LLM inference timeout | Skip sentiment step, use score=0.0 |
| SQLite locked | 5-second busy timeout with WAL mode for concurrent reads |
| Dashboard process crashes | Launcher detects and auto-restarts within 30 seconds |
| Cloud relay unreachable | Publisher retries 3 times then skips; local system unaffected |
| Circuit breaker triggered | ES strategy flattens all positions, pauses until 17:00 CT |
| Pipeline step fails | Logs error, continues to next step; pipeline never fully aborts |
