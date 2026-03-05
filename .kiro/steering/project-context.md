# Stock Analysis Platform — Project Context

This file provides permanent context for every chat session in this workspace.

## Project Overview

SPY/SPX Predictor + ES Futures Strategy system. ML-powered daily market predictions with a real-time ES futures trading engine, unified Streamlit dashboard, and full observability stack.

- **Current version**: v2.7.2 (Market Breadth + Index Fundamentals — S&P 500 PE/earnings yield/dividend yield, breadth indicators, encrypted secrets)
- **Git remote**: `https://github.com/damerav/stockanalysis.git`
- **Git user**: `damerav <damerav@gmail.com>`

## Architecture

- **125+ model features** available across price, technicals, macro, sentiment, options, microstructure, earnings, Fed NLP, geopolitical risk, oil shock, FinBERT NLP, and market breadth/fundamentals — **32 kept after aggressive feature selection** (10 new breadth features available)
- **17+ database tables** in PostgreSQL (primary) with SQLite fallback, via `src/data/db_router.py`. All reads use SQLAlchemy 2.0 engine (no psycopg2 DBAPI2 warnings). PostgreSQL runs in Docker container on DGX (`stockanalysis` database, user `stockapp`). Plus `news.db` (7000+ articles with FinBERT cache, category-tagged)
- **16-step daily pipeline** (`src/pipeline/daily_run.py`) with expanded news ingestion (44 categorized RSS feeds across 13 finance categories, 2800+ articles/fetch) and market breadth computation
- **Stacking ensemble**: XGBoost + BiLSTM + LightGBM with logistic meta-learner
- **HMM regime detection**: 4 states (bull_trend, bear_trend, high_vol_choppy, low_vol_range)
- **Conformal prediction**: 90% coverage prediction sets
- **Adaptive training window**: candidates [252, 504] days
- **P3 training enhancements** (from Harvard cs249r_book research):
  - Label smoothing (α=0.15) — blends hard labels with teacher soft probabilities
  - Sample quality weighting — z-score anomaly detection + VIX-based penalty + label-flip detection
  - Entropy-weighted self-distillation refit — up-weights hard/uncertain samples (focal-loss-like)
  - Knowledge distillation validation — trains student models on soft targets, only adopts if accuracy improves
- **Current model accuracy**: 3-class val=48.9%, test=47.8%, binary directional=54.2%

## Infrastructure

### Development Machine (Windows)
- **Path**: `F:\websites\stockanalysis`
- **Sync**: Mutagen syncs to DGX (~7-8 seconds delay)
- Code is written locally, tested/run on DGX

### DGX Spark (Ubuntu, Production)
- **Host**: `abidamera@192.168.1.211`
- **Path**: `~/stockanalysis`
- **Python**: 3.12.3 in venv at `~/stockanalysis/.venv`
- **Activate**: `source .venv/bin/activate`
- **Note**: `lsof` is NOT installed — use `fuser -k <port>/tcp` to kill processes
- **ML dependencies**: FinBERT (`ProsusAI/finbert`) + `transformers` + `torch` installed for NLP sentiment
- **PostgreSQL**: Docker container `postgres` on port 5432, database `stockanalysis`, user `stockapp` (password in encrypted `app_secrets` table — see `src/data/secrets_manager.py`)
- **Ollama**: Running with `deepseek-r1:70b` (42.5GB, deep analysis) and `deepseek-r1:14b` (9GB, fast routing/tool calls)
- **Mutagen**: Currently PAUSED — use SCP to transfer files directly

### Running Commands on DGX
- All commands via: `ssh abidamera@192.168.1.211 "command"`
- PowerShell quote escaping is problematic for complex commands
- **Pattern for complex commands**: Write a temp Python script locally, wait ~8s for Mutagen sync, run via SSH, then delete the temp script
- Always activate venv: `source .venv/bin/activate && ...`

### Services & Ports
| Service | Port | URL |
|---------|------|-----|
| Streamlit Dashboard | 8501 | `http://192.168.1.211:8501` |
| Confidence API | 8100 | `http://192.168.1.211:8100` |
| PostgreSQL | 5432 | `localhost:5432` (DGX only) |
| Ollama | 11434 | `http://localhost:11434` (DGX only) |
| Grafana | 3001 | `http://192.168.1.211:3001` (LAN only) |
| Prometheus | 9092 | `http://192.168.1.211:9092` |
| Metrics Exporter | 9190 | `http://192.168.1.211:9190` |
| External Access | 8501 | `trading.aiagenticinternational.org` (reverse-proxied) |

### Scheduler (src/launcher.py)
- Runs as background process on DGX via `python -m src.launcher --spy`
- Full pipeline at 4:30 PM ET (Mon-Fri)
- Intraday updates at 8:30 AM, 12:00 PM, 1:30 PM, 3:00 PM ET
- Also manages dashboard process (starts Streamlit on port 8501)
- Health monitoring loop restarts crashed processes automatically
- Dashboard Admin page detects scheduler via `pgrep -af src.launcher`

### Starting/Restarting System
```bash
# Start scheduler + dashboard (recommended — manages both)
ssh abidamera@192.168.1.211 "cd ~/stockanalysis && source .venv/bin/activate && nohup python -m src.launcher --spy --config config.yaml > logs/scheduler.log 2>&1 &"

# Dashboard only (standalone, no scheduler)
ssh abidamera@192.168.1.211 "fuser -k 8501/tcp 2>/dev/null; sleep 1; cd ~/stockanalysis && source .venv/bin/activate && nohup streamlit run src/dashboard/app.py --server.port 8501 --server.headless true > logs/dashboard.log 2>&1 &"
```

## Key Files

### State Files
- `data/spy_state.json` — Real-time SPY prediction state (direction, confidence, indicators, flow_alerts, updated_at)
- `data/es_state.json` — ES futures strategy state (P&L, positions, signals, regime)
- `data/spy.db` — SQLite fallback database (legacy, kept for environments without PostgreSQL)
- `data/analytics.duckdb` — DuckDB analytics layer (legacy, superseded by PostgreSQL)
- `data/news.db` — News article store (7000+ articles) with FinBERT sentiment cache

### Config
- `config.yaml` — Central config (API keys, model params, auth, grafana, ensemble, etc.)
- `.streamlit/config.toml` — Streamlit theme config (supports dark/light toggle)
- `grafana/grafana.ini` — Grafana config (anonymous auth enabled, must be chmod 644 after Mutagen sync)

### Dashboard Source
- `src/dashboard/app.py` — Main unified dashboard (~3200+ lines). Uses `st.navigation` with pages across Markets and Operations groups. Includes global live price ticker bar (`@st.fragment(run_every=15)`) with admin-configurable stock list. Contains Quant Agent chatbot page with 10 quick-action buttons in 3 rows (data-only, no LLM) + free-form LLM chat. Row 3 has Market Thesis and Vigilance Alerts (v2.6). ES Strategy page has AI Confidence overlay row (layer status, continuation probability, trail multipliers) + Reload Rules button (v2.7).
- `src/dashboard/performance_app.py` — Performance analytics dashboard: accuracy trends, confidence calibration, regime-stratified metrics, model comparison charts
- `src/dashboard/tuning_app.py` — Model tuning & backtest UI: hyperparameter grid, champion/challenger comparison, backtest runner
- `src/dashboard/rules_app.py` — Strategy Rules management page: 12 tabbed groups (spread, sizing, entry, TP low/med/high, risk, session, indicators, regime, AI, RL), live edit + save + reset, hot-reload flag for runner
- `src/dashboard/theme.py` — Theme system: CSS token-based design with DARK + LIGHT palettes (TradingView-inspired), dynamic CSS injection, `get_plotly_layout()`, `themed_metric_card()`, `get_theme()` helpers
- `src/dashboard/monitoring.py` — Native Plotly monitoring (6 tabs: SPY, ES, System Health, Confidence API, Pipeline, Data Sources). Uses thread-safe fresh DB connections per query (PostgreSQL primary, SQLite fallback) — no singleton router.
- `src/dashboard/single_stock_app.py` — Individual stock analysis with technical indicators
- `src/dashboard/realtime_app.py` — Real-time streaming dashboard
- `src/dashboard/es_dashboard.py` — ES strategy dashboard (standalone)
- `src/dashboard/whatif_app.py` — What-If analysis
- `src/dashboard/style.css` — CSS token-based design system (dark/light theme variables, pill tabs, compact headers, sidebar styling)
- `src/dashboard/template.py` — HTML template helpers for themed components
- `src/dashboard/forecast_app.py` — REMOVED from navigation (vanilla LSTM, slow, duplicated ensemble predictions). File kept on disk but disconnected.

### Core Modules
- `src/data/` — Data fetching, features (125 available), DB routing (PostgreSQL primary + SQLite fallback via `db_router.py`), backfill, calendar, drift monitoring, geopolitical risk features, news fetching (44 categorized RSS feeds across 13 finance categories), FinBERT sentiment caching, PostgreSQL migration tools
- `src/model/` — Trainer (with P3 label smoothing, sample quality weighting, entropy-weighted self-distillation, knowledge distillation), registry (champion/challenger framework with promote/rollback), ensemble, BiLSTM, conformal, regime, adaptive window, purged CV, LSTM predictor, news predictor
- `src/es_strategy/` — ES futures engine (loads all params from `strategy_rules` DB table via `rules_store`), indicators, position management, RL trailing, labeling, AI entry gate + CNN exit controller
- `src/strategy/` — `rules_store.py` DB-backed strategy parameter store (get/set/reset rules via DbRouter). 42 rules across 12 groups (spread, sizing, entry, tp_low/med/high, risk, session, indicators, regime, ai, rl)
- `src/llm/` — LLM analyzer and reporter (DeepSeek R1 via Ollama), Quant Agent with multi-model routing (14B fast + 70B deep) and tool-based architecture
- `src/pipeline/` — Daily pipeline orchestration (fully thread-safe — all DB ops via `_db_execute`/`_db_query`/`_db_fetchone` router helpers), alerts, news pipeline runner, and event-driven vigilance monitor
- `src/launcher.py` — System launcher with background scheduler (pipeline + intraday updates + vigilance monitoring every 5 min), process manager, health monitoring
- `src/api/` — Confidence API server, Prometheus metrics exporter
- `src/auth/` — Google OAuth + local auth with bcrypt, server-side session files
- `src/sync/` — Cloud relay publisher/server
- `src/whatif/` — What-If simulation engine, presets, narrator

## Authentication

- **Mode**: Local auth (bcrypt-hashed passwords in SQLite `users` table)
- **Seed users**: admin/admin123, user/user123 (auto-created on first run via `src/data/init_db.py`)
- **Session persistence**: Server-side session files in `data/.sessions/` (gitignored) with triple redundancy (query param + cookie + session_state)
- **User management**: Admin → Users tab in dashboard (add/edit/delete)
- **Grafana**: Anonymous auth enabled for embed; direct access credentials: admin/admin

## API Keys

- **FRED**: Stored in encrypted `app_secrets` table (key: `fred_api_key`)
- **Finnhub**: Stored in encrypted `app_secrets` table (key: `finnhub_api_key`)
- **Polygon**: Not yet configured
- All secrets managed via `src/data/secrets_manager.py` with Fernet encryption
- Keys are masked in the Admin config editor with a "Reveal sensitive values" toggle

## spy_state.json Fields

**Contains**: `updated_at`, `prediction` (direction, scale_label, confidence, probabilities), `indicators` (rsi_14, macd, atr_14, vix, vix_change, volume_ratio, sentiment_score), `flow_alerts`, `regime`, `prediction_set`, `vigilance_alerts` (array of event-driven alerts from vigilance monitor)

**Does NOT contain**: `last_close`, `ensemble_used`, `shap_drivers` — these come from the pipeline when it runs

## Data Consistency Rules

- SPY Predictor page and Monitoring SPY tab both read from `spy_state.json` for real-time metrics
- Historical charts read from PostgreSQL (primary) with SQLite fallback
- SPY close price comes from the `prices` table (not in state file)
- VIX in summary cards falls back to live FRED macro data
- Monitoring page uses thread-safe fresh connections per query (not the DbRouter singleton) to avoid SQLite cross-thread errors in Streamlit
- Pipeline (`daily_run.py`) uses thread-safe DB helpers (`_db_execute`, `_db_query`, `_db_fetchone`) routed through a fresh `DbRouter` instance per run. External functions that need a raw connection get `router.get_sqlite()` via `_get_conn()` (thread-safe, created in pipeline thread). No direct `self.conn` usage in any step method.
- Quick-action buttons in Quant Agent bypass LLM entirely — call tool methods directly for instant results

## Docker / Cloud

- `cloud/docker-compose.yml` — Grafana, Prometheus, relay, dashboard containers
- Grafana container: `cloud-grafana-1`, config mounted from `grafana/grafana.ini`
- After Mutagen sync, `grafana.ini` permissions may reset — fix with `chmod 644`

## Conventions

- Commit messages: conventional commits (`fix:`, `feat:`, `chore:`, etc.)
- Testing on DGX: write temp script locally → Mutagen sync → SSH run → delete script
- Dashboard uses TradingView-inspired dual theme (dark/light) managed by `src/dashboard/theme.py`
- Sidebar always stays dark navy in both themes (by design)
- All Plotly charts use `get_plotly_layout()` from `theme.py` (theme-aware)
- All metric cards use `themed_metric_card()` from `theme.py`
- `use_container_width=True` on all `st.plotly_chart()` calls
- Navigation: `st.navigation` with pages — Markets group (SPY Predictor, Performance, ES Strategy, Tune & Backtest, What-If, Single-Stock, Quant Agent) and Operations group (Monitoring, Grafana Dashboards, Strategy Rules, Admin). Forecast page removed.
- ES signal feed shows human-readable descriptions (e.g., "AI Rejected Signal" instead of raw `AI_REJECT`)
- ES regime badges use dark text on yellow/green for WCAG contrast compliance

## Theme Palettes (reference)

- **DARK**: bg `#131722`, surface `#1E222D`, accent `#2962FF`, bull `#26A69A`, bear `#EF5350`, text `#D1D4DC`/`#787B86`, border `#2A2E39`
- **LIGHT**: bg `#F0F2F5`, surface `#FFFFFF`, green `#0ECB81`, red `#F6465D`, yellow `#F0B90B`, text `#1E2329`/`#707A8A`, border `#E6E8EC`, card_border `#D1D4DC`

## Recent Changes (Post Phase 3)

- Expanded news pipeline: 44 categorized RSS feeds across 13 finance categories (markets, forex, bonds, commodities, crypto, centralbanks, economic, ipo, derivatives, fintech, regulation, institutional, analysis), 2800+ articles per fetch, VADER + FinBERT + LLM blended sentiment. Inspired by worldmonitor project's Google News RSS proxy technique — all feeds are free, no API keys needed.
- Geopolitical risk features: `src/data/geopolitical_features.py` (oil shock, geopolitical risk index, FinBERT-scored geopolitical headlines)
- P3 training enhancements from Harvard cs249r_book research (label smoothing, sample quality weighting, entropy-weighted self-distillation, knowledge distillation validation)
- Global live price ticker bar on all dashboard pages with admin-configurable stock list
- Forecast accuracy display fixed to use MAE instead of meaningless `(1 - MSE_loss) * 100`
- Single-stock news sorting fixed (RFC 2822 vs ISO date format mismatch)
- CSS token-based unified design system for dark/light theme
- Feature shape mismatch fix after auto-retrain (predictor.trained_feature_names now updates in memory)

### v2.5 Changes (PostgreSQL migration + Quant Agent + LLM routing)

- **PostgreSQL migration**: All 17+ tables migrated from SQLite/DuckDB to PostgreSQL (Docker container on DGX). `src/data/db_router.py` provides `DbRouter` class with PostgreSQL primary + SQLite fallback. `src/data/migrate_to_postgres.py` handles migration.
- **Quant Agent chatbot**: New page in dashboard with 10 quick-action buttons in 3 rows (Current Prediction, Feature Importance, News Sentiment, Regime History, Correlations, Risk Assessment, Alpha Ideas, Explain Regime, Market Thesis, Vigilance Alerts) — all bypass LLM for instant results. Free-form chat input routes through LLM for deep analysis.
- **Multi-model LLM routing**: `deepseek-r1:14b` (9GB, ~19s) for fast tool routing + `deepseek-r1:70b` (42.5GB) for deep interpretation. Automatic fallback from 70B to 14B on timeout. Sidebar shows both models.
- **Prediction formatter fix**: Emoji checked for BULLISH/BEARISH (not UP/DOWN), confidence no longer formatted as percentage of percentage, enriched with conformal set + regime + SHAP drivers.
- **Forecast page removed**: Vanilla LSTM on close-only prices was slow (TensorFlow import), simplistic, and duplicated the 32-feature stacking ensemble. Disconnected from navigation.
- **Monitoring page PostgreSQL migration**: All 6 monitoring tabs now query PostgreSQL via thread-safe fresh connections per query (not the DbRouter singleton). Fixes SQLite "objects created in a thread" errors in Streamlit's multi-threaded rendering. Direct `_pg_connect()` / `_sqlite_connect()` helpers create and close connections per query.
- **None sentiment fix**: `_fmt_risk` handles NULL sentiment values from PostgreSQL categories with no data.
- **Streamlit 1.54 compatibility**: Running on DGX with sklearn 1.8.0.
- **Pipeline thread-safety migration**: All 13 pipeline steps now use router-based DB helpers (`_db_execute`, `_db_query`, `_db_fetchone`) instead of direct `self.conn`. Fresh `DbRouter` instance created per pipeline run (not the singleton). External functions (`store_technicals`, `build_feature_vector`, `evaluate_past_prediction`, `store_earnings`, `update_fed_communications`) receive `router.get_sqlite()` via `_get_conn()`. Fixes "SQLite objects created in a thread" errors when pipeline is triggered from Streamlit Admin page.
- **Scheduler activated**: `src/launcher.py --spy` runs as background process on DGX. Manages dashboard process + scheduled pipeline runs (4:30 PM ET daily, intraday at 8:30/12:00/13:30/15:00). Admin page detects scheduler status via `pgrep`.

### v2.6 Changes (Agentic Intelligence — Vigilance + Quality Scoring + Thesis Tracker)

- **Event-driven vigilance monitor** (`src/pipeline/vigilance.py`): Lightweight polling loop (every 5 min during market hours) checks for VIX spikes (>20%), price gaps (>1.5%), HMM regime changes, and sentiment sign flips. Alerts written to `spy_state.json` → `vigilance_alerts` array and sent through existing Telegram/email channels. Integrated into `Scheduler._loop()` in `src/launcher.py`.
- **News quality scoring**: Each article now gets a `quality_score` (0.0-1.0) computed at fetch time from 4 factors: source credibility (40%, tiered by outlet), content depth (30%, text length proxy), sentiment confidence (15%, VADER signal strength), specificity (15%, numbers/$/%  presence). `get_category_sentiment_summary()` now returns `weighted_sentiment` alongside `avg_sentiment`. New `quality_score` column in `news.db` with auto-migration.
- **Source credibility tiers**: Bloomberg/Fed (0.95), CNBC (0.85), MarketWatch (0.80), Yahoo/SeekingAlpha (0.75), Investing.com (0.70), Google News aggregated (0.55), unknown (0.50).
- **Market thesis tracker**: New Quant Agent tool `get_market_thesis()` builds a structured thesis with 4 pillars (Trend, Macro/VIX, News Sentiment, Regime). Each pillar evaluated as supporting/weakening/conflicting with current prediction. Summary shows thesis strength (strong/moderate/weak) and conviction ratio.
- **Vigilance alerts tool**: New Quant Agent tool `get_vigilance_alerts()` surfaces recent event-driven alerts in the chatbot.
- **Dynamic feature alignment**: `SPYPredictor._align_features()` maps any input feature array to the model's trained feature set. `predict()` calls it automatically when `feature_names` and `trained_feature_names` are both available. Fixes What-If page `ValueError: feature shape mismatch`.
- **Model file sorting fix**: `load_latest_model()` now excludes `_binary_up/down` and `_conformal` files from the sort. Previously loaded the binary UP model instead of the main model, causing `trained_feature_names` to be None.
- **Streamlit duplicate element fix**: `_prediction_card()` now takes `key_suffix` param for unique `plotly_chart` keys on What-If comparison page.
- **Quant Agent UI expansion**: Added Row 3 quick-action buttons — "🎯 Market Thesis" and "🚨 Vigilance Alerts" — with dedicated formatters (`_fmt_thesis`, `_fmt_vigilance`). Total buttons now 10 across 3 rows.

### v2.7 Changes (Rules UI + AI Confidence Layer + Performance Analytics)

- **SQLAlchemy 2.0 migration**: All `pd.read_sql_query()` calls now use SQLAlchemy engine instead of raw psycopg2 connections. `db_router.py` creates `_pg_engine` alongside raw conn. `_pg_to_sqlalchemy()` helper converts `%s` positional params to `:p0, :p1` named params for `text()`. Zero DBAPI2 warnings. `monitoring.py` and `performance_app.py` use per-query SQLAlchemy engines. `sqlalchemy>=2.0` added to `requirements.txt`.
- **Full PostgreSQL migration**: All 20+ source files migrated from `sqlite3`/`get_connection` to `DbRouter`. No remaining direct SQLite imports outside of `init_db.py` fallback path and `db_router.py` itself.
- **Performance Dashboard** (`src/dashboard/performance_app.py`): Accuracy trends, confidence calibration curves, regime-stratified metrics, model comparison charts. Registered in Markets nav group.
- **Model Tuning & Backtest** (`src/dashboard/tuning_app.py`): Hyperparameter grid editor, champion/challenger comparison, backtest runner with walk-forward validation. Registered in Markets nav group.
- **Champion/Challenger Framework** (`src/model/registry.py`): `model_registry` PostgreSQL table (22 columns). `register()`, `get_active()`, `promote_model()`, `rollback()`, `get_accuracy_trend()` methods. Models tracked with status (active/candidate/retired), accuracy metrics, feature names, training metadata.
- **Strategy Rules DB table** (`strategy_rules`): 42 rules across 12 groups (spread, sizing, entry, tp_low/med/high, risk, session, indicators, regime, ai, rl). Seeded via `init_db.py` with `INSERT OR IGNORE` (SQLite) / `ON CONFLICT DO NOTHING` (PostgreSQL). Primary key: `(rule_group, rule_key)`.
- **Rules Store module** (`src/strategy/rules_store.py`): `get_rule()`, `get_group()`, `get_all_rules()`, `set_rule()`, `set_group()`, `reset_to_defaults()`. All SQL uses `?` placeholders — DbRouter converts automatically.
- **ES Engine rules migration**: `ESStrategyEngine.__init__()` loads all parameters from `rules_store` instead of hardcoded values. Anti-chase gate, Phase 2 thresholds, TP multipliers, RL params all DB-driven.
- **AI exit confidence wiring**: `runner.py` `_process_bar()` calls `self.ai.get_trail_multipliers()` and sets `self.engine._ai_trail_mults` before `process_bar()`. Engine `_check_exits()` uses CNN-provided TP2/runner multipliers when `trail_ai_enabled` is true.
- **Hot-reload**: Runner checks for `data/.reload_rules` flag file each loop iteration. When found, deletes flag and re-initializes `ESStrategyEngine` from DB. Rules UI and ES page both write this flag on save/reload.
- **Rules Management page** (`src/dashboard/rules_app.py`): 12 tabbed groups with editable fields (number_input for float/int, checkbox for bool, text_input for time/string). Min/max validation from DB. Save per-group + Reset per-group + Reset ALL. Registered in Operations nav group.
- **AI Confidence overlay**: ES Strategy page shows 4-metric row (AI Layer on/off, Continuation P, AI TP2 Trail, AI Runner Trail) from `get_state()`. Plus "Reload Rules" button.
- **Admin Config tab replaced**: Raw YAML editor removed. Now shows read-only key settings summary + info banner pointing to Strategy Rules page.
- **`get_state()` expanded**: Now includes `ai_enabled`, `trail_ai_enabled`, `ai_trail_mults` in the state dict for dashboard consumption.
- **Auth migrated to PostgreSQL**: `google_oauth.py` user management functions use DbRouter.

### v2.7.1 Changes (Schema Fixes + Pipeline Robustness + Dashboard Performance)

- **PostgreSQL schema fixes**: Added missing `quality_score` column to `raw_articles` table and `feature_version` column to `feature_cache` table via ALTER TABLE on DGX. News quality scoring and feature caching now work correctly.
- **DbRouter auto-config**: `DbRouter.__init__()` now auto-loads `config.yaml` when no config dict is passed. Previously, `get_router()` with no args would get an empty config, fail PostgreSQL, and crash on corrupted SQLite. Fixes all modules that call `get_router()` without explicit config.
- **DbRouter corrupted SQLite handling**: SQLite fallback path now catches `sqlite3.DatabaseError` and recreates the file instead of crashing.
- **`_TABLE_PKS` expanded**: Added `news_features`, `feature_store_meta`, and `strategy_rules` to the primary key mapping. Fixes `INSERT OR REPLACE` → PostgreSQL `ON CONFLICT DO UPDATE` conversion for these tables.
- **Pipeline Step 10 regime fix**: HMM regime detection now queries the `prices` + `macro` tables directly via SQL JOIN instead of relying on the feature store cache (which doesn't include `close`/`volume` columns). Eliminates `KeyError: 'close'`.
- **Dashboard load performance**: CSS file cached in `session_state` (no disk read per rerun). yfinance ticker price cached with `st.cache_data(ttl=15)` and lazy-imported. `spy_state.json`, `es_state.json`, prediction history, and performance queries all cached with appropriate TTLs (10-120s).
- **Single-stock lazy imports**: `yfinance` moved from module-level import to lazy import inside `_fetch_stock_data()` and `_fetch_company_info()`. Faster initial module load.
- **Predicted vs Actual chart**: New dual-axis line chart on Performance page — left axis shows predicted direction (blue) vs actual direction (orange dashed) with red X markers on misses; right axis overlays SPY OHLC candlesticks at 40% opacity.
- **news_features upsert**: `store_features()` now uses `INSERT OR REPLACE` with explicit column list for proper PostgreSQL upsert via DbRouter's `_TABLE_PKS` mapping. Fixes duplicate key errors on `news_features` table.

### v2.7.2 Changes (Market Breadth + Index Fundamentals + Encrypted Secrets)

- **Encrypted secrets manager** (`src/data/secrets_manager.py`): Fernet-encrypted `app_secrets` PostgreSQL table. 6 secrets migrated (API keys, DB password, session secret, seed passwords). All source files resolve secrets from encrypted DB with env var fallbacks. `config.yaml` API keys replaced with `FROM_ENCRYPTED_DB` placeholders. `cryptography>=42.0` added to requirements.
- **Market breadth & index fundamentals** (`src/data/market_breadth.py`): S&P 500 PE ratio, forward PE, earnings yield, dividend yield from yfinance SPY info. Market breadth: % stocks above 50/200-day SMA, advance/decline ratio, 52-week highs/lows, breadth thrust. Wikipedia S&P 500 constituent list with fallback to hardcoded top-100.
- **`market_breadth` PostgreSQL table**: Schema in `init_db.py`, PK mapping in `db_router.py`. Stores daily breadth + fundamental snapshots.
- **Pipeline step 9.7** (`_step97_market_breadth`): Fetches fundamentals + breadth, stores in DB. Runs as part of the 16-step daily pipeline.
- **10 new features in feature vector**: `sp500_pe`, `sp500_forward_pe`, `sp500_earnings_yield`, `sp500_dividend_yield`, `pct_above_sma50`, `pct_above_sma200`, `advance_decline_ratio`, `new_highs_52w`, `new_lows_52w`, `breadth_thrust`. Integrated into `build_feature_vector()` and `get_feature_columns()` in `features.py`.
- **np.float64 serialization fix**: `store_breadth_fundamentals()` casts numpy types to native Python before SQL to avoid PostgreSQL `schema "np" does not exist` errors.
- **pd.read_html FutureWarning fix**: Wikipedia scraper wraps `resp.text` in `StringIO`.
- **lxml>=5.0** added to requirements.txt for Wikipedia HTML parsing.

```
