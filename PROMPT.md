# MASTER BUILD PROMPT — SPY/SPX Predictor + ES Futures Strategy

> **For Kiro IDE**: This is the master prompt for building the complete trading analysis system.
> Open this file, then tell Kiro: "Build this system following the 8 phases below."
> Each phase has a dedicated spec in `.kiro/specs/` — use Kiro's spec-driven development to execute them in order.

---

## WHAT WE ARE BUILDING

A **real-time market analysis and signal generation platform** that runs on an NVIDIA DGX Spark (local GPU server) with optional AWS cloud mirror. The system produces trading signals and predictions only — the user executes all trades manually.

**Two subsystems:**
1. **SPY/SPX Predictor** — Ingests Polygon.io real-time data, scores 9 market factors every 5 minutes, predicts next-day SPY direction with XGBoost, uses a local 70B LLM for news sentiment
2. **ES Futures Strategy** — Generates entry/exit signals for E-mini S&P 500 futures using Keltner Channel bands, 3-lot tiered exits, adaptive volatility regimes, and optional AI models

**Shared infrastructure:** Polygon data feed, SQLite database, Streamlit dashboards, cloud sync to AWS, daily automated pipeline, Telegram/email alerts.

---

## HARDWARE TARGET

| Component | Role | Details |
|-----------|------|---------|
| NVIDIA DGX Spark | Local always-on server | GPU for XGBoost, PyTorch CNN, Ollama (DeepSeek R1 70B = 42GB) |
| AWS EC2 t3.micro | Cloud relay (~$8/mo) | No GPU — proxies JSON state for remote dashboards |

---

## PHASE 1: DATA INGESTION FROM POLYGON.IO

> **Spec**: `.kiro/specs/01-polygon-ingestion/`
> **Skill**: `.kiro/skills/polygon-ingestion/`
> **Power**: `powers/power-polygon/`

### What To Build

**1A. Polygon WebSocket Streamer (`src/realtime/streamer.py`, ~494 lines)**
- Connect to Polygon Advanced WebSocket ($398/mo): stocks + options channels
- **Stocks channel** (`wss://socket.polygon.io/stocks`): Subscribe to `T.SPY` for real-time trades
- **Options channel** (`wss://socket.polygon.io/options`): Subscribe to `T.O:SPX*` for all SPX options trades
- Aggregate raw ticks into **5-second OHLCV bars** (open, high, low, close, volume, VWAP components)
- Track options flow: detect sweeps (same option, multiple exchanges, <2 sec, >$50K), block trades (>$100K notional), compute live put/call ratio
- Auto-reconnect with exponential backoff (1s → 2s → 4s → max 30s)
- Heartbeat monitoring: reconnect if no data for 60 seconds during market hours
- Two separate threads: one for stocks, one for options

**1B. Polygon REST Fetcher (`src/data/polygon_fetcher.py`, ~475 lines)**
- `get_daily_bars(ticker, from_date, to_date)` — daily OHLCV, adjusted
- `get_5s_bars(ticker, date)` — intraday 5-second granularity
- `get_options_chain(underlying, expiry)` — full chain with Greeks (delta, gamma, theta, vega, IV)
- `get_options_analytics(underlying)` — computed: P/C ratio, max pain, IV skew, GEX
- Retry logic: 3 attempts, exponential backoff
- Handle Polygon's `next_url` pagination for large result sets
- Rate limit awareness: 100 requests/minute

**1C. Fallback Data Sources (`src/data/fetcher.py`, ~223 lines)**
- **yfinance**: Price data fallback if Polygon REST fails
- **Finnhub** (free tier): Market news headlines via API
- **RSS feeds**: Yahoo Finance, CNBC, MarketWatch news scraping
- **FRED** (free): Macro data — VIX, 10Y yield, DXY, fed funds rate, gold, crude oil

**1D. Gap Detection & Backfill (`src/data/daily_pull.py`, ~262 lines)**
- Find last date in SQLite → identify missing weekdays → backfill from Polygon (fallback: yfinance)
- Fetch fresh macro data for gap dates
- Update options chain snapshot
- Compute technicals for all new dates
- Validate data completeness across all 10 tables
- Called from 3 places: launcher scheduler, daily pipeline Step 0.5, realtime pre-market

**1E. Initial Bulk Load (`src/data/backfill.py`, ~115 lines)**
- First-time setup: load 252 trading days (1 year) of historical data
- Command: `python -m src.data.backfill --days 252`

**1F. SQLite Database Schema (`src/data/init_db.py`, ~224 lines)**
Create 10 tables in `./data/spy.db`:
```sql
prices              — date PK, OHLCV
technicals          — date PK, SMA_20, SMA_50, RSI_14, MACD, BB, ATR
news                — auto-increment PK, date, source, headline, summary
daily_sentiment     — date PK, score, confidence, article_count, ratios
macro               — date PK, VIX, yields, DXY, fed_funds, gold, crude
predictions         — date PK, direction, confidence, factors, report_text
intraday_bars       — (timestamp, ticker) PK, OHLCV
options_chain       — (date, contract_symbol) PK, strike, expiry, Greeks
options_analytics   — date PK, P/C ratio, max_pain, IV_skew, GEX
intraday_features   — date PK, VWAP_spread, momentum, range, volume_ratio
performance         — date PK, predicted, actual, correct, cumulative_accuracy
```

### Acceptance Criteria
- [ ] WebSocket connects and receives SPY trades within 5 seconds
- [ ] 5-second bars are produced with correct OHLCV aggregation
- [ ] Options sweeps and block trades are detected and logged
- [ ] REST API fetches daily bars, options chain, and 5s bars successfully
- [ ] Gap detection finds and backfills missing dates automatically
- [ ] All 10 SQLite tables are created and populated with 1 year of history
- [ ] System continues operating if options WebSocket fails (stocks-only mode)
- [ ] yfinance fallback works when Polygon REST is unavailable

---

## PHASE 2: CHECK FOR 70B LLM ON DGX SPARK

> **Spec**: `.kiro/specs/02-llm-health-check/`
> **Skill**: `.kiro/skills/llm-health-check/`
> **Power**: `powers/power-dgx-spark/`

### What To Build

**2A. Ollama Health Check (`src/llm/analyzer.py` → `_check_ollama()` method)**
- Check if Ollama process is running: `GET http://localhost:11434/api/tags`
- If not running → attempt to start: `subprocess.Popen(["ollama", "serve"])`
- Wait up to 15 seconds for Ollama to become responsive
- If still not running → log warning, set `llm_available = False`, continue without LLM

**2B. Model Availability Check (`_model_available()` method)**
- Query Ollama's model list: `GET http://localhost:11434/api/tags`
- Look for `deepseek-r1:70b` (configurable via `config.yaml → llm.model`)
- If model not found → trigger auto-download

**2C. Auto-Download with Progress (`_pull_model()` method)**
- Call `POST http://localhost:11434/api/pull` with `{"name": "deepseek-r1:70b"}`
- Stream the response (model is ~42GB download)
- Parse streaming JSON: extract `total` and `completed` fields
- Log progress every 5%: `"Downloading deepseek-r1:70b: 45.2% (19.0GB / 42.1GB)"`
- On success → log "Model ready" and set `llm_available = True`
- On failure → log error, set `llm_available = False`, continue without LLM

**2D. Inference Validation**
- After model is confirmed available, run a quick test inference:
  ```python
  ollama.chat(model="deepseek-r1:70b", messages=[{"role": "user", "content": "Reply OK"}])
  ```
- Verify response is non-empty within 30-second timeout
- If inference fails → log warning, mark LLM as degraded

**2E. Graceful Degradation Pattern**
```
If Ollama not running    → try to start → if fails → continue without LLM
If model not downloaded  → auto-download → if fails → continue without LLM
If inference fails       → use neutral sentiment (score=0.0) + raw-data reports
Pipeline NEVER aborts due to LLM unavailability
```

**2F. Integration Points**
- Called from `launcher.py` on system start (Phase 0 of boot sequence)
- Called from `daily_run.py` Step 0 before each pipeline run
- Called from `realtime/main.py` during pre-market phase
- Config: `config.yaml → llm.model`, `llm.base_url`, `llm.temperature`

### Acceptance Criteria
- [ ] System detects whether Ollama is running or not
- [ ] System starts Ollama automatically if it's not running
- [ ] System detects whether `deepseek-r1:70b` is downloaded
- [ ] Auto-download streams with progress logging (no silent hang)
- [ ] Quick inference test validates the model actually works
- [ ] If LLM is completely unavailable, system continues with neutral sentiment
- [ ] LLM check completes in <30 seconds (excluding model download time)
- [ ] Works from all 3 call sites: launcher, daily pipeline, realtime pre-market

---

## PHASE 3: BUILD AND TRAIN THE FINANCIAL MODEL

> **Spec**: `.kiro/specs/03-financial-model/`
> **Skill**: `.kiro/skills/financial-model/`

### What To Build

**3A. Feature Engineering (`src/data/features.py`, ~400+ lines)**
Build a 125-feature vector for each trading day (32 kept after aggressive selection):
```
TECHNICAL (from technicals table):
  price_vs_sma20, price_vs_sma50, rsi_14, macd, macd_signal, macd_hist,
  bb_upper_dist, bb_lower_dist, atr_14, sma20_slope, sma50_slope

MACRO (from macro table):
  vix_level, vix_change, us10y_yield, dxy, fed_funds, gold, crude

SENTIMENT (from daily_sentiment table):
  llm_sentiment_score, news_count, positive_ratio, negative_ratio

NEWS (from expanded news pipeline — 17 sources, 1000+ articles/fetch):
  news_sentiment_mean, news_sentiment_std, news_volume_ratio, news_momentum

INTRADAY (from intraday_features table):
  vwap_spread, intraday_momentum, intraday_range, volume_ratio

OPTIONS (from options_analytics table):
  put_call_ratio, max_pain_distance, iv_skew, gex_normalized

GEOPOLITICAL (from src/data/geopolitical_features.py):
  geopolitical_risk_index, oil_shock_indicator, finbert_geopolitical_sentiment

DERIVED:
  price_vs_sma20_pct, price_vs_sma50_pct, rsi_divergence, volume_trend,
  atr_percentile, momentum_5d, momentum_10d

MICROSTRUCTURE, EARNINGS, FED NLP:
  (additional features from earnings calendar, Fed communications, market microstructure)
```

**3B. Technical Indicator Computation (`src/data/features.py`)**
- SMA(20), SMA(50) — Simple Moving Averages
- RSI(14) — Relative Strength Index
- MACD(12,26,9) — signal line + histogram
- Bollinger Bands — SMA(20) ± 2σ
- ATR(14) — Average True Range
- All configurable via `config.yaml → technicals` section

**3C. XGBoost SPY Direction Predictor (`src/model/trainer.py`, ~800+ lines)**
- **Target**: Next-day SPY direction — UP (+1) / DOWN (-1) / NEUTRAL (0)
- **Threshold**: ±0.3% daily return to classify (filters noise)
- **Objective**: `multi:softprob` with `num_class=3`
- **GPU**: `tree_method='gpu_hist'` for DGX Spark acceleration
- **Lookback**: Adaptive window from candidates [252, 504] days
- **Validation**: Walk-forward time-series split (80/20, no shuffle)
- **Early stopping**: 50 rounds on validation loss
- **Feature selection**: Aggressive — keeps ~32 of 125 available features
- **P3 Training Enhancements** (from Harvard cs249r_book research):
  - Label smoothing (α=0.15) — blends hard labels with teacher soft probabilities
  - Sample quality weighting — z-score anomaly detection + VIX-based penalty + label-flip detection
  - Entropy-weighted self-distillation refit — up-weights hard/uncertain samples
  - Knowledge distillation validation — trains student models, only adopts if accuracy improves
- **Hyperparameters** (from config):
  ```yaml
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 500
  subsample: 0.8
  colsample_bytree: 0.8
  ```
- **Output**: Prediction with class probabilities → map to STRONG_BULLISH through STRONG_BEARISH with confidence 0-100%
- **Persistence**: Save model to `./models/xgb_spy_{date}.json`, log feature importances
- **Current accuracy**: 3-class val=48.9%, test=47.8%, binary directional=54.2%

**3D. Daily Retraining Pipeline Integration**
- Retrained daily at 4:30 PM ET as part of the 13-step pipeline (Step 10)
- Uses latest data including today's close
- Generates prediction for NEXT trading day (Step 11)
- Stores prediction in `predictions` table with factors breakdown
- Tracks accuracy: compares yesterday's prediction to actual direction (Step 1)

**3E. ES Entry Gate — XGBoost-GPU (`src/es_strategy/ai_models.py`, ~485 lines)**
- **17 features** normalized by ATR/price: price_vs_kc_mid, price_vs_vwap, rsi, roc_3, atr_regime_pct, volume_ratio, kc_width, ema9_slope, macd_hist, bb_width, momentum_3bar, momentum_5bar, bars_since_trade, daily_pnl, time_sin, time_cos, spread_vs_atr
- **Triple-barrier meta-labels**: UP=TP1, DOWN=emergency_stop, TIME=60 bars
- **Output**: `p_enter` = Prob(TP1 hit before stop)
- **Regime thresholds**: Low=0.58, Med=0.55, High=0.52
- **Sizing**: `qty = round(base × clip((p-p_min)/0.20, 0, 1))` → 1-3 lots
- **Training**: Purged walk-forward CV, elastic sample weights (recent 6 months emphasized)

**3F. ES Exit Controller — 1D-CNN (`src/es_strategy/ai_models.py`)**
- **Architecture**: Conv1d(features→64→32→16) + AdaptiveAvgPool + FC(16→32→1→sigmoid)
- **Input**: Last 20 bars × 19 features
- **Output**: P_cont_5 (probability price continues 5 more bars)
- **Maps to trail multipliers** within regime bounds:
  - Runner: Low [1.2-1.6], Med [1.3-1.7], High [1.5-2.0] × ATR
  - TP2: Low [0.9-1.2], Med [1.0-1.25], High [1.25-1.5] × ATR
- **Drift monitor (DriftMonitor)**: PSI > 0.2 → halve size + refit; AI IR underperforms rules by 1σ over 100 trades → disable AI

**3G. LLM Sentiment Model (`src/llm/analyzer.py`, ~359 lines)**
- Batch process up to 50 news articles through DeepSeek R1 70B
- Structured JSON output per article: `{score: -1.0 to 1.0, confidence: 0-100, topics: []}`
- Aggregate daily sentiment: weighted average by recency + source credibility
- Runtime: ~60-90 minutes for 50 articles on DGX Spark
- Temperature: 0.3 for consistent structured output

**3H. Daily Report Generator (`src/llm/reporter.py`, ~192 lines)**
- Input: day's technicals, sentiment, model prediction, key signals
- Output: ~400-word market brief
- Stored in `predictions.report_text` column

### Acceptance Criteria
- [ ] Feature vector with 35+ features computed correctly for each date
- [ ] XGBoost trains on GPU (`gpu_hist`) without errors
- [ ] Walk-forward validation shows no data leakage (future data never in training set)
- [ ] Model saves to `./models/` with date-stamped filename
- [ ] Prediction output maps to 5-level scale (STRONG_BULLISH → STRONG_BEARISH)
- [ ] Accuracy tracking compares prior predictions to actual outcomes
- [ ] ES entry gate produces p_enter probabilities within expected ranges
- [ ] CNN exit controller adjusts trail multipliers within regime bounds
- [ ] Drift monitor detects PSI > 0.2 and triggers refit
- [ ] LLM sentiment analysis produces valid JSON output for each article
- [ ] System works with AI models disabled (pure rules fallback for ES)

---

## PHASE 4: BUILD STREAMLIT DASHBOARDS

> **Spec**: `.kiro/specs/04-streamlit-dashboards/`
> **Skill**: `.kiro/skills/streamlit-dashboards/`

### What To Build

**4A. Unified Dashboard (`src/dashboard/app.py`, port 8501)**

All dashboards are served from a single Streamlit app with sidebar navigation on one port (8501). The sidebar menu provides access to: SPY Predictor, ES Strategy, What-If Analysis, and Admin Console. Auto-detects local vs cloud mode via `RELAY_URL` env var.

**4A-1. SPY/SPX Predictor Page (formerly `src/dashboard/realtime_app.py`)**

Layout:
```
┌──────────────────────────────────────────────────┐
│  SPY/SPX PREDICTOR — Current Prediction Banner   │
│  ████ STRONG BULLISH  87% confidence ████         │
├──────────────────────┬───────────────────────────┤
│  Prediction History  │  Key Indicators           │
│  ┌────────────────┐  │  RSI(14):  62.3           │
│  │  Timeline chart │  │  MACD:     +0.42          │
│  │  (last 20 preds)│  │  VIX:      14.2 (↓0.8)   │
│  └────────────────┘  │  Vol Ratio: 1.24           │
├──────────────────────┴───────────────────────────┤
│  Options Flow Alerts (most recent)               │
│  🔴 14:23 PUT SWEEP SPX 5950 Jan17 $2.1M (12×)  │
│  🟢 14:21 CALL BLOCK SPX 6050 Jan17 $890K        │
│  🔴 14:18 PUT SWEEP SPX 5900 Jan17 $1.5M (8×)   │
└──────────────────────────────────────────────────┘
```

Features:
- Prediction banner: color-coded (green→red), shows direction + confidence %
- Prediction history: Plotly timeline of last 20 predictions with actual outcomes
- Key indicators: RSI, MACD, VIX, volume ratio — live during market hours
- Options flow feed: scrolling alerts for sweeps/blocks with size + direction
- **15-second auto-refresh** via `st.rerun()` with timer
- **Auto-detect mode**: if `RELAY_URL` env var set → cloud mode (fetch from relay), else → local mode (read JSON files from `./data/`)

**4A-2. ES Futures Strategy Page (formerly `src/dashboard/es_dashboard.py`)**

Layout:
```
┌──────────────────────────────────────────────────┐
│  ES STRATEGY — LONG 2 lots (TP1 filled)          │
│  Entry: 6010.25  |  P&L: +$375  |  Regime: Med   │
├──────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐ │
│  │  Plotly Candlestick Chart (1-min bars)      │ │
│  │  ● KC Upper band (shaded)                   │ │
│  │  ● KC Lower band (shaded)                   │ │
│  │  ● KC Mid (dashed line)                     │ │
│  │  ● VWAP overlay (dotted)                    │ │
│  │  ● Entry/exit markers (▲ long, ▼ short)     │ │
│  │  ● Stop levels (horizontal dashed red)      │ │
│  ├─────────────────────────────────────────────┤ │
│  │  RSI(14) subplot                            │ │
│  └─────────────────────────────────────────────┘ │
├──────────────────────┬───────────────────────────┤
│  Signal Feed         │  Status Panel             │
│  15:42 EXIT_TP1 Lot0 │  Circuit Breaker: OK      │
│  15:38 ENTRY_LONG 3  │  Daily P&L: +$375         │
│  15:35 AI_REJECT     │  Trades Today: 2          │
│  15:30 STOP_UPDATE   │  Session: Active          │
└──────────────────────┴───────────────────────────┘
```

Features:
- Position state banner: LONG/SHORT/FLAT with lot count, entry price, unrealized P&L
- Plotly candlestick with Keltner Channel as shaded bands + VWAP overlay
- Entry/exit markers on chart (triangle up=long, triangle down=short)
- Stop levels drawn as horizontal dashed red lines
- RSI subplot synced with main chart's x-axis
- Regime indicator: color-coded (green=Low, yellow=Med, red=High)
- Signal feed: scrolling log of all signals with timestamps
- Status panel: circuit breaker, daily P&L, trade count, session status
- **5-second auto-refresh**
- Same cloud/local mode detection as SPY dashboard

**4B. Dashboard Data Bridge (`src/realtime/dashboard_bridge.py`, ~68 lines)**
- Backends write state to JSON files in `./data/`:
  - `spy_state.json` — current prediction, indicators, flow alerts
  - `es_state.json` — position, signals, regime, P&L, chart data
- Atomic writes (write to temp → rename) to prevent partial reads
- Dashboards read these JSON files on each refresh cycle

### Acceptance Criteria
- [ ] SPY dashboard shows current prediction with confidence percentage
- [ ] SPY dashboard displays prediction history timeline chart
- [ ] Options flow alerts appear in real-time during market hours
- [ ] ES dashboard shows candlestick chart with Keltner Channel overlay
- [ ] ES dashboard displays per-lot position state and P&L
- [ ] Signal feed updates with all signal types (entries, exits, stops, AI rejects)
- [ ] 15-second auto-refresh works on SPY dashboard
- [ ] 5-second auto-refresh works on ES dashboard
- [ ] Local mode reads from `./data/*.json` files
- [ ] Cloud mode fetches from relay when `RELAY_URL` is set
- [ ] Dashboards handle missing/empty data gracefully (show "Waiting for data")

---

## PHASE 5: SYNC STREAMLIT DASHBOARDS TO THE CLOUD

> **Spec**: `.kiro/specs/05-cloud-sync/`
> **Skill**: `.kiro/skills/cloud-sync/`
> **Power**: `powers/power-aws-deploy/`

### What To Build

**5A. Cloud Publisher (`src/sync/publisher.py`, ~142 lines)**
- Runs on DGX Spark alongside backends
- HTTPS POST to AWS relay every time state changes:
  - `POST /push/prediction` — SPY prediction updates
  - `POST /push/flow_alert` — large SPX options trades
  - `POST /push/premarket` — pre-market summary
  - `POST /push/es_state` — ES strategy live state
  - `POST /push/heartbeat` — 30-second keepalive
- API key authentication in `X-API-Key` header
- Retry on network failure (3 attempts, 5s backoff)
- If sync disabled in config → publisher does nothing (no-op)

**5B. FastAPI Relay Server (`src/sync/relay_server.py`, ~344 lines)**
- Runs on AWS EC2 t3.micro (~$8/mo)
- **In-memory state only** — no database on cloud
- Receives POST pushes from DGX, stores latest state in memory
- Serves state to cloud Streamlit dashboards:
  - `GET /state` — full SPY state (prediction, indicators, alerts)
  - `GET /state/es` — ES strategy state (position, signals, regime)
  - `GET /stream` — Server-Sent Events for real-time push
  - `GET /health` — health check (shows last heartbeat time)
  - `POST /admin/reset` — clear all state (requires admin API key)
- Stale detection: if no heartbeat for 90 seconds → mark source as OFFLINE
- CORS enabled for dashboard access

**5C. Docker Containers (`cloud/`)**
```
cloud/
├── Dockerfile              # Combined relay + dashboard
├── Dockerfile.relay        # Relay only (FastAPI + uvicorn)
├── Dockerfile.dashboard    # Dashboard only (Streamlit × 2)
├── docker-compose.yml      # Two-container orchestration
├── deploy_aws.sh           # Automated EC2 deployment
├── start.sh                # Combined container entrypoint
└── .env.example            # Environment variables template
```

**5D. Automated AWS Deployment (`cloud/deploy_aws.sh`)**
Sequence:
1. Create ECR repository (if not exists)
2. Build Docker image(s) locally
3. Push to ECR
4. Create security group (ports 8000, 8501, 8502)
5. Launch EC2 t3.micro with ECR pull + docker run
6. Output: relay URL + dashboard URLs + API key
7. User copies relay URL to `config.yaml → sync.relay_url`

**5E. Cloud Dashboard Mode**
- Same `realtime_app.py` and `es_dashboard.py` code
- Detects `RELAY_URL` env var → switches data source from local JSON to relay GET
- No code duplication — single codebase serves both modes

### Acceptance Criteria
- [ ] Publisher sends state updates to relay via HTTPS POST
- [ ] Relay stores latest state in memory and serves via GET
- [ ] Cloud dashboards fetch and display same data as local dashboards
- [ ] SSE endpoint streams updates in real-time
- [ ] Heartbeat detects DGX going offline within 90 seconds
- [ ] Docker containers build and run correctly
- [ ] `deploy_aws.sh` creates ECR, launches EC2, configures security group
- [ ] Total cloud cost is ~$8-10/month on t3.micro
- [ ] System works 100% locally if cloud sync is disabled
- [ ] API key authentication prevents unauthorized state pushes

---

## PHASE 6: ES FUTURES STRATEGY ENGINE

> **Spec**: `.kiro/specs/06-es-strategy/`
> **Skill**: `.kiro/skills/es-strategy-engine/`

### What To Build

**6A. Indicators (`src/es_strategy/indicators.py`, ~235 lines)**
- ATR(14), Keltner Channel (EMA20 ± 1.5×ATR), EMA(9), VWAP, RSI(14), ROC(3)
- Regime detection: ATR percentile over 10,080 bars with 3-bar hysteresis
  - Low (<33rd pctile), Med (33-66th), High (>66th)

**6B. Position Manager (`src/es_strategy/position.py`, ~253 lines)**
- 3-lot tiered position: Lot 0 (TP1), Lot 1 (TP2), Lot 2 (Runner)
- Track: entry_price, direction, per-lot status, stops, P&L
- No pyramiding (must be flat to enter)

**6C. Strategy Engine (`src/es_strategy/engine.py`, ~705 lines)**
- Phase 1: Pure-edge entry at K±C → TP1 → TP2 → Runner cascade
- Phase 2: Confluence reload (K±C + 2/3 filters: ROC, ATR, VWAP)
- Session guards: flatten before 15:55 CT, flatten before FOMC/CPI/NFP
- Circuit breaker: daily P&L ≤ −$2,000 → flatten + disable until 17:00 CT

**6D. Live/Backtest/Paper Runner (`src/es_strategy/runner.py`, ~448 lines)**
- `--mode live` — Polygon feed
- `--mode backtest --data es_1min.csv` — historical CSV
- `--mode paper` — live data, signals logged only
- `--ai` flag enables XGBoost entry gate + CNN exit controller

### Acceptance Criteria
- [ ] Entry triggers at exactly K±C with anti-chase gate
- [ ] 3-lot exit cascade: TP1 → ratchet → TP2 → ratchet → Runner
- [ ] Emergency stop at 20% × C always active
- [ ] Jump exit at 5 pts adverse during 1-minute hold
- [ ] Phase 2 requires 2/3 confluence filters
- [ ] Circuit breaker flattens at −$2,000 daily P&L
- [ ] Backtest mode reproduces signals from CSV data

---

## PHASE 7: DAILY AUTOMATED PIPELINE

> **Spec**: `.kiro/specs/07-daily-pipeline/`
> **Skill**: `.kiro/skills/daily-pipeline/`

### What To Build

**7A. Pipeline Orchestrator (`src/pipeline/daily_run.py`, ~310 lines)**
13-step sequential pipeline running at 4:30 PM ET:
```
Step 0:   Check LLM model (auto-download if missing)
Step 0.5: Daily data pull (gap detection + backfill)
Step 1:   Evaluate past predictions (accuracy tracking)
Step 2:   Fetch daily prices (Polygon → yfinance fallback)
Step 3:   Fetch news (Finnhub + RSS)
Step 4:   LLM sentiment analysis (~60-90 min for 50 articles)
Step 5:   Fetch macro (VIX, yields, DXY, fed funds, gold, crude)
Step 6:   Fetch options chain snapshot
Step 7:   Compute options analytics (P/C ratio, max pain, IV skew, GEX)
Step 8:   Compute daily technicals (SMAs, RSI, MACD, BB, ATR)
Step 9:   Build intraday features (VWAP spread, momentum, range)
Step 10:  Build feature vector + retrain XGBoost
Step 11:  Generate prediction for next trading day
Step 12:  Generate LLM daily report
Step 13:  Send alerts (Telegram + email)
```

**7B. Alert System (`src/pipeline/alerts.py`, ~127 lines)**
- Telegram: Bot API with formatted prediction message
- Email: SMTP with HTML template
- Both optional, configured in `config.yaml → alerts`

**7C. Launcher & Scheduler (`src/launcher.py`, ~418 lines)**
- Single command starts everything: `python -m src.launcher --config config.yaml --all`
- Starts backends as subprocesses, dashboards as Streamlit processes
- Background thread checks time every 60 seconds, triggers pipeline at 4:30 PM ET (Mon-Fri)
- Health monitoring: watches subprocess health, logs crashes
- CLI flags: `--all`, `--spy`, `--es`, `--dashboards-only`, `--check-llm`

### Acceptance Criteria
- [ ] Pipeline runs all 13 steps in sequence without manual intervention
- [ ] LLM failure at Step 4 → skips to Step 5 with neutral sentiment
- [ ] XGBoost retrains with latest data and saves new model
- [ ] Prediction is generated and stored in SQLite
- [ ] Alerts sent via Telegram and/or email
- [ ] Scheduler triggers at 4:30 PM ET, skips weekends
- [ ] Launcher starts all components and monitors health

---

## PHASE 8: WHAT-IF ANALYSIS (DGX + CLOUD)

> **Spec**: `.kiro/specs/08-what-if-analysis/`
> **Skill**: `.kiro/skills/what-if-analysis/`

### Purpose
Let the user ask "what would happen if…?" across both subsystems — parameter sweeps on the ES strategy, scenario injection into the SPY predictor, and LLM-narrated explanations — all accessible from local DGX dashboards and cloud mirrors.

### What To Build

**8A. What-If Engine (`src/whatif/engine.py`, new)**
Core compute module that runs on DGX Spark (has GPU + data + models):

```
WhatIfEngine
├── es_parameter_sweep(params_grid)     # Backtest ES strategy with varied K/C/regime/risk
├── es_compare_scenarios(scenario_list)  # Side-by-side backtest comparison
├── spy_scenario_inject(overrides)       # Override features → re-run XGBoost inference
├── spy_feature_ablation(drop_list)      # Drop features → measure accuracy impact
├── spy_monte_carlo(n_sims, noise)       # Random perturbation → prediction distribution
├── market_stress_test(scenario_name)    # Pre-built: gap-down, VIX spike, flash crash
└── llm_explain(result)                  # Ask DeepSeek to narrate findings
```

**8B. ES Strategy What-If Scenarios**

| Scenario Type | Input | Output |
|---------------|-------|--------|
| **K/C Sweep** | K range (e.g. 5900-6100 step 25), C range (5-20 step 2.5) | Heatmap: total P&L per K/C combo over last N days |
| **Regime Tuning** | Change TP multipliers or percentile thresholds | Comparison table: old vs new regime P&L, win rate, max drawdown |
| **Lot Sizing** | Max lots = 1, 2, or 3 | Bar chart: P&L and risk per lot config |
| **Risk Limits** | Circuit breaker from −$1K to −$3K in $500 steps | Line chart: P&L vs protection trade-off |
| **AI On/Off** | Toggle AI entry gate and/or exit controller | Side-by-side: rules-only vs AI-enhanced metrics |
| **Historical Replay** | Pick a date range + params | Full signal log + equity curve for that period |

Implementation: reuses `src/es_strategy/runner.py --mode backtest` under the hood with injected config overrides. Each what-if scenario creates a temporary config dict, runs the backtest engine, and collects results.

**8C. SPY Predictor What-If Scenarios**

| Scenario Type | Input | Output |
|---------------|-------|--------|
| **Feature Override** | "What if VIX = 35?" or "sentiment = −0.8" | New prediction + confidence + which direction changed and why |
| **Feature Ablation** | Drop 1-5 features from the vector | Accuracy impact: which features matter most (bar chart) |
| **Threshold Sensitivity** | Change neutral threshold (±0.1% to ±0.5%) | Table: precision/recall/accuracy per threshold |
| **Monte Carlo** | Add Gaussian noise (1-5% σ) to all features, run N=500 sims | Distribution histogram: % bullish / bearish / neutral |
| **Market Stress** | Pre-built scenarios: "2020 March crash", "VIX spike to 40", "gap down 3%" | Prediction output + LLM narrative of implications |

Implementation: loads the current trained XGBoost model from `./models/`, constructs a modified feature vector, runs `model.predict()` — no retraining needed. GPU inference is sub-second per scenario.

**8D. LLM-Narrated Explanations (`src/whatif/narrator.py`, new)**
After each what-if run, optionally send results to DeepSeek R1 70B for a plain-English narrative:
```
Prompt: "Given these what-if results for the ES strategy with K=5950 and C=12:
  - Original P&L: +$2,400 over 20 days
  - Modified P&L: +$3,100 over 20 days
  - Win rate improved from 58% to 64%
  Explain why this parameter change improved results and what risks it introduces."
```
- Temperature: 0.5 (slightly more creative than sentiment analysis)
- Max tokens: 800
- Fallback: if LLM unavailable, show raw numbers without narrative

**8E. What-If Dashboard (`src/dashboard/whatif_app.py`, new, port 8503)**

Layout:
```
┌──────────────────────────────────────────────────────────┐
│  WHAT-IF ANALYSIS                                         │
│  ┌─────────────┐  ┌──────────────────────────────────┐   │
│  │ Scenario     │  │  Results Panel                    │   │
│  │ Selector     │  │                                   │   │
│  │              │  │  ┌─ ES Strategy Tab ────────────┐ │   │
│  │ ○ ES Params  │  │  │ Plotly heatmap: K/C sweep    │ │   │
│  │ ○ ES Compare │  │  │ Equity curves overlay        │ │   │
│  │ ○ SPY Inject │  │  │ Metrics comparison table     │ │   │
│  │ ○ SPY Monte  │  │  └──────────────────────────────┘ │   │
│  │ ○ Stress Test│  │  ┌─ SPY Predictor Tab ──────────┐ │   │
│  │              │  │  │ Feature importance bar chart  │ │   │
│  │ ┌──────────┐│  │  │ Monte Carlo histogram         │ │   │
│  │ │ Input     ││  │  │ Scenario comparison table     │ │   │
│  │ │ Controls  ││  │  └──────────────────────────────┘ │   │
│  │ │ (sliders, ││  │                                   │   │
│  │ │  dropdowns││  │  ┌─ LLM Narrative ──────────────┐ │   │
│  │ │  ranges)  ││  │  │ "Shifting K from 6000 to     │ │   │
│  │ └──────────┘│  │  │  5950 improved results by..." │ │   │
│  │              │  │  └──────────────────────────────┘ │   │
│  │ [▶ Run]     │  │                                   │   │
│  │ [📋 Compare]│  │  [💾 Export CSV] [📊 Save Report] │   │
│  └─────────────┘  └──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

Features:
- Left panel: scenario type selector + dynamic input controls (Streamlit sliders, number inputs, multiselects)
- Right panel: tabbed results (ES / SPY / LLM Narrative)
- Plotly heatmaps for K/C sweeps, overlaid equity curves for comparisons
- Monte Carlo histograms with confidence intervals
- Export results to CSV or generate a PDF report
- **Run button** triggers compute → shows spinner → displays results
- Saved scenarios: store what-if configs + results in SQLite for later comparison

**8F. What-If Cloud Sync**
Extends the existing cloud sync architecture:

```
DGX Spark (computes what-if)                    AWS Cloud
┌──────────────────────────┐    HTTPS POST    ┌─────────────────────┐
│ whatif/engine.py          │───────────────→  │ relay_server.py     │
│   runs GPU backtests      │  /push/whatif    │   stores results    │
│   runs XGBoost inference  │                  │                     │
│   runs LLM narrative      │    GET           │ whatif_app.py :8503 │
│                            │←───────────────  │   displays results  │
│ whatif_app.py :8503 (DGX) │                  └─────────────────────┘
└──────────────────────────┘
```

Two operating modes:
- **DGX mode** (local): User submits scenario → engine.py runs locally → results appear instantly
- **Cloud mode** (remote): User submits scenario via cloud dashboard → POST to DGX relay → DGX computes → POST results back to cloud relay → cloud dashboard displays

New relay endpoints:
| Method | Path | Purpose |
|--------|------|---------|
| POST | /push/whatif_request | Cloud → DGX: request a what-if scenario |
| POST | /push/whatif_result | DGX → Cloud: completed what-if results |
| GET | /state/whatif | Cloud dashboard fetches latest what-if results |
| GET | /whatif/presets | List available pre-built stress test scenarios |

**8G. Pre-Built Stress Test Scenarios (`src/whatif/presets.py`, new)**
```python
STRESS_PRESETS = {
    "march_2020_crash": {
        "description": "Simulate March 2020 sell-off conditions",
        "overrides": {"vix": 65, "vix_change": 20, "sentiment": -0.9, "volume_ratio": 3.5}
    },
    "vix_spike_40": {
        "description": "VIX jumps to 40 with negative sentiment",
        "overrides": {"vix": 40, "vix_change": 12, "sentiment": -0.6}
    },
    "gap_down_3pct": {
        "description": "Market gaps down 3% at open",
        "overrides": {"intraday_momentum": -3.0, "vwap_spread": -2.5, "rsi_14": 22}
    },
    "fomc_surprise_hawkish": {
        "description": "Unexpected rate hike, yields spike",
        "overrides": {"us10y": 5.5, "fed_funds": 6.0, "dxy": 108, "sentiment": -0.7}
    },
    "low_vol_melt_up": {
        "description": "Steady grind higher, very low volatility",
        "overrides": {"vix": 11, "vix_change": -2, "sentiment": 0.5, "rsi_14": 72}
    },
    "flash_crash_5min": {
        "description": "ES drops 50 points in 5 minutes then recovers half",
        "es_overrides": {"K_shift": -50, "atr_multiplier": 3.0, "regime_override": "High"}
    }
}
```

**8H. What-If SQLite Storage**
New tables in spy.db:
```sql
-- Saved what-if scenario configurations
CREATE TABLE whatif_scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    type TEXT,              -- 'es_sweep', 'spy_inject', 'monte_carlo', 'stress_test'
    params TEXT,            -- JSON blob of input parameters
    created_at TEXT
);

-- What-if results
CREATE TABLE whatif_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id INTEGER REFERENCES whatif_scenarios(id),
    result_data TEXT,       -- JSON blob of output (P&L, metrics, distributions)
    narrative TEXT,         -- LLM-generated explanation (nullable)
    computed_at TEXT,
    compute_time_sec REAL
);
```

### Acceptance Criteria
- [ ] ES K/C sweep produces a Plotly heatmap of P&L across parameter grid
- [ ] ES scenario comparison shows overlaid equity curves for 2+ configs
- [ ] SPY feature override changes prediction output in real-time
- [ ] SPY Monte Carlo generates histogram from 500+ simulations in <30 seconds on DGX
- [ ] Pre-built stress test scenarios produce valid predictions with overridden features
- [ ] LLM narrative explains what-if results in plain English (fallback: raw numbers)
- [ ] What-if dashboard (port 8503) works both locally and via cloud relay
- [ ] Cloud mode: user submits scenario from cloud → DGX computes → results appear on cloud
- [ ] Scenarios and results persist in SQLite for later comparison
- [ ] Export to CSV and PDF report works
- [ ] System handles what-if requests without disrupting live trading signals

---

## ADMIN CONSOLE (Integrated in Unified Dashboard)

The Admin Console is the 4th page in the unified Streamlit dashboard sidebar, providing full system management without SSH.

### Admin Console Tabs

**System Status Tab**
- Database health: file size, table count, integrity
- LLM status: Ollama running, model availability, inference readiness
- XGBoost model: latest model file, size, date
- Data inventory: row counts and date ranges for all 11 tables
- Latest prediction: date, direction, confidence

**Actions Tab — Ad-Hoc Operations**
| Action | Description |
|--------|-------------|
| 📥 Pull Latest Data | Gap detection + backfill prices and macro |
| 📰 Fetch News | Fetch from Finnhub + RSS, insert into news table |
| 📊 Fetch Macro Data | Fetch VIX, yields, DXY, gold, crude from FRED |
| 📈 Compute Technicals | Recompute SMA, RSI, MACD, BB, ATR |
| 🧠 Retrain XGBoost | Retrain SPY predictor on GPU with latest data |
| 🔮 Generate Prediction | Run inference for next trading day |
| 🩺 LLM Health Check | Check Ollama + model availability |
| 📝 Generate Report | Generate LLM daily report |
| 🚀 Run Full Pipeline | Execute all 13 pipeline steps (option to skip LLM) |
| 📨 Send Test Alert | Send test Telegram/email alert |

**Database Tab**
- Table browser with sort, limit, and order controls
- Custom SQL query editor (SELECT only, read-only)
- Vacuum database (reclaim space)
- Integrity check

**Configuration Tab**
- Quick view of key settings (LLM model, XGB lookback, ES max lots, cloud sync)
- Full YAML editor with syntax validation
- Save with validation — changes take effect on next restart

**Logs Tab**
- Dashboard log viewer (tail N lines)
- Pipeline run history with prediction reports
- Model file inventory with sizes and dates

---

## OPERATIONS SCRIPTS (`scripts/`)

Shell scripts for DGX Spark system lifecycle management. All scripts track PIDs in `.pids/` and write logs to `logs/`.

**`scripts/start.sh`** — Full system startup:
1. Checks for existing running instance (prevents double-start)
2. Activates Python virtual environment
3. Starts Ollama (if not already running)
4. Runs LLM health check
5. Starts unified Streamlit dashboard on port 8501
6. Starts ES strategy runner in paper mode
7. Starts pipeline scheduler (4:30 PM ET, Mon-Fri)
8. Prints summary with URLs and PIDs

**`scripts/stop.sh`** — Graceful shutdown:
- Stops scheduler, ES runner, dashboard in reverse order
- 10-second graceful timeout, then force-kill
- Kills orphaned processes on port 8501
- Use `--all` flag to also stop Ollama

**`scripts/status.sh`** — System health check:
- Process status (running/dead/not started) for all components
- Port checks (8501 dashboard, 11434 Ollama)
- Database size, row counts, latest dates for key tables
- Latest trained model file
- Disk usage and free space
- Recent log activity

**`scripts/restart.sh`** — Stop then start (pass `--all` to restart Ollama too)

---

## CONFIGURATION REFERENCE

All settings in `config.yaml`:
```yaml
polygon:
  api_key: "YOUR_POLYGON_KEY"
  ws_stocks_url: "wss://socket.polygon.io/stocks"
  ws_options_url: "wss://socket.polygon.io/options"

analysis:
  signal_interval: 300          # 5 minutes between scoring cycles
  flow_sweep_threshold: 50000   # $50K minimum for sweep detection
  flow_block_threshold: 100000  # $100K minimum for block detection

llm:
  model: "deepseek-r1:70b"
  base_url: "http://localhost:11434"
  temperature: 0.3

xgboost:
  lookback_days: 252
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 500
  neutral_threshold: 0.003      # ±0.3% for neutral classification

technicals:
  sma_periods: [20, 50]
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bb_period: 20
  bb_std: 2
  atr_period: 14

sync:
  enabled: false
  relay_url: "https://your-ec2-ip:8000"
  api_key: "YOUR_RELAY_API_KEY"

es_strategy:
  credit_C: 10.0
  strike_K: 6000.0
  max_lots: 3
  jump_exit_points: 5.0
  emergency_stop_pct: 0.20
  circuit_breaker_usd: -2000.0
  session_close_ct: "15:55"
  session_reset_ct: "17:00"
  ai_enabled: false
  regime_lookback: 10080
  regime_pct_low: 33
  regime_pct_high: 66

alerts:
  telegram_token: ""
  telegram_chat_id: ""
  email_smtp: ""
  email_from: ""
  email_to: ""

whatif:
  enabled: true
  max_sims: 1000              # Monte Carlo cap
  max_concurrent_cloud: 1     # Queue cloud requests
  narrator_temperature: 0.5
  narrator_max_tokens: 800
  es_default_lookback_days: 20
  spy_default_noise_pct: 2.0

database:
  path: "./data/spy.db"
```

---

## BUILD ORDER FOR KIRO

Tell Kiro to build in this order, using spec-driven development for each phase:

1. **Phase 1** → `01-polygon-ingestion` spec → Get data flowing into SQLite
2. **Phase 2** → `02-llm-health-check` spec → Verify DGX has the LLM ready
3. **Phase 3** → `03-financial-model` spec → Train XGBoost, build features
4. **Phase 4** → `04-streamlit-dashboards` spec → Build unified dashboard with sidebar nav
5. **Phase 5** → `05-cloud-sync` spec → Add relay + deploy to AWS
6. **Phase 6** → `06-es-strategy` spec → Build the ES futures strategy engine
7. **Phase 7** → `07-daily-pipeline` spec → Wire up the automated pipeline + launcher
8. **Phase 8** → `08-what-if-analysis` spec → Add what-if analysis on DGX + cloud
9. **Admin Console** → Integrated in unified dashboard → System management UI
10. **Operations Scripts** → `scripts/` → start.sh, stop.sh, status.sh, restart.sh

Each phase depends on the previous. After all phases, the complete system runs with:
```bash
# Using scripts (recommended)
./scripts/start.sh          # Start everything
./scripts/status.sh         # Check health
./scripts/stop.sh           # Graceful shutdown

# Or using the Python launcher directly
python -m src.launcher --config config.yaml --all
```

## PROJECT STRUCTURE

```
stockanalysis/
├── config.yaml                     # All configuration
├── requirements.txt                # Python dependencies
├── PROMPT.md                       # This file — master build prompt
├── src/
│   ├── data/                       # Data ingestion layer
│   │   ├── init_db.py              # SQLite schema (11 tables)
│   │   ├── polygon_fetcher.py      # Polygon.io REST client
│   │   ├── fetcher.py              # Fallback: yfinance, Finnhub, RSS, FRED
│   │   ├── daily_pull.py           # Gap detection + backfill
│   │   ├── backfill.py             # Initial bulk load (252 days)
│   │   └── features.py             # Feature engineering (37+ features)
│   ├── realtime/                   # Real-time data layer
│   │   ├── streamer.py             # Polygon WebSocket (stocks + options)
│   │   └── dashboard_bridge.py     # State file writer for dashboards
│   ├── llm/                        # LLM layer
│   │   ├── analyzer.py             # Ollama health check + sentiment analysis
│   │   └── reporter.py             # Daily report generator
│   ├── model/                      # ML model layer
│   │   └── trainer.py              # XGBoost SPY predictor (GPU)
│   ├── es_strategy/                # ES futures strategy layer
│   │   ├── indicators.py           # ATR, Keltner, EMA, VWAP, RSI, ROC
│   │   ├── position.py             # 3-lot position manager
│   │   ├── engine.py               # Strategy engine (Phase 1/2 entries)
│   │   ├── ai_models.py            # XGBoost entry gate + CNN exit controller
│   │   └── runner.py               # Live/backtest/paper runner
│   ├── dashboard/                  # Dashboard layer
│   │   ├── app.py                  # Unified dashboard (port 8501, 4 pages)
│   │   ├── realtime_app.py         # Standalone SPY dashboard (legacy)
│   │   ├── es_dashboard.py         # Standalone ES dashboard (legacy)
│   │   └── whatif_app.py           # Standalone What-If dashboard (legacy)
│   ├── whatif/                     # What-If analysis layer
│   │   ├── engine.py               # What-If compute engine
│   │   ├── presets.py              # 5 stress test scenarios
│   │   └── narrator.py             # LLM narrator for what-if results
│   ├── sync/                       # Cloud sync layer
│   │   ├── publisher.py            # DGX → AWS state publisher
│   │   └── relay_server.py         # FastAPI relay on EC2
│   ├── pipeline/                   # Pipeline layer
│   │   ├── daily_run.py            # 13-step pipeline orchestrator
│   │   └── alerts.py               # Telegram + email alerts
│   └── launcher.py                 # System launcher + scheduler
├── scripts/                        # Operations scripts
│   ├── start.sh                    # Full system startup
│   ├── stop.sh                     # Graceful shutdown
│   ├── status.sh                   # Health check
│   └── restart.sh                  # Stop + start
├── cloud/                          # Cloud deployment
│   ├── Dockerfile                  # Combined container
│   ├── Dockerfile.relay            # Relay-only container
│   ├── Dockerfile.dashboard        # Dashboard container
│   ├── docker-compose.yml          # Two-service orchestration
│   ├── deploy_aws.sh               # AWS deployment script
│   ├── start.sh                    # Container entrypoint
│   ├── start_dashboards.sh         # Dashboard entrypoint
│   └── .env.example                # Environment template
├── data/                           # Runtime data
│   ├── spy.db                      # SQLite database
│   ├── spy_state.json              # SPY dashboard state
│   └── es_state.json               # ES dashboard state
├── models/                         # Trained models
│   └── xgb_spy_YYYYMMDD.json      # Date-stamped XGBoost models
├── logs/                           # Runtime logs (created by scripts)
├── .pids/                          # PID files (created by scripts)
└── docs/                           # Documentation
    ├── USER_GUIDE.md               # End-user guide
    ├── ADMIN_GUIDE.md              # Administrator guide
    └── SOLUTION_DOCUMENT.md        # Technical solution document
```
