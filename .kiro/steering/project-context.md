# Stock Analysis Platform — Project Context

This file provides permanent context for every chat session in this workspace.

## Project Overview

SPY/SPX Predictor + ES Futures Strategy system. ML-powered daily market predictions with a real-time ES futures trading engine, unified Streamlit dashboard, and full observability stack.

- **Current version**: v1.3+ (post Phase 3 enhancements)
- **Git remote**: `https://github.com/damerav/stockanalysis.git`
- **Git user**: `damerav <damerav@gmail.com>`

## Architecture

- **93 model features** across price, technicals, macro, sentiment, options, microstructure, earnings, and Fed NLP
- **17+ database tables** split across SQLite (operational) and DuckDB (analytics)
- **15-step daily pipeline** (`src/pipeline/daily_run.py`)
- **Stacking ensemble**: XGBoost + BiLSTM + LightGBM with logistic meta-learner
- **HMM regime detection**: 4 states (bull_trend, bear_trend, high_vol_choppy, low_vol_range)
- **Conformal prediction**: 90% coverage prediction sets
- **Adaptive training window**: candidates [63, 126, 252, 504] days

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
| Grafana | 3001 | `http://192.168.1.211:3001` (LAN only) |
| Prometheus | 9092 | `http://192.168.1.211:9092` |
| Metrics Exporter | 9190 | `http://192.168.1.211:9190` |
| External Access | 8501 | `trading.aiagenticinternational.org` (reverse-proxied) |

### Restarting Dashboard
```bash
ssh abidamera@192.168.1.211 "fuser -k 8501/tcp 2>/dev/null; sleep 1; cd ~/stockanalysis && source .venv/bin/activate && nohup streamlit run src/dashboard/app.py --server.port 8501 --server.headless true > logs/dashboard.log 2>&1 &"
```

## Key Files

### State Files
- `data/spy_state.json` — Real-time SPY prediction state (direction, confidence, indicators, flow_alerts, updated_at)
- `data/es_state.json` — ES futures strategy state (P&L, positions, signals, regime)
- `data/spy.db` — SQLite operational database
- `data/analytics.duckdb` — DuckDB analytics layer

### Config
- `config.yaml` — Central config (API keys, model params, auth, grafana, ensemble, etc.)
- `.streamlit/config.toml` — Streamlit dark theme config
- `grafana/grafana.ini` — Grafana config (anonymous auth enabled, must be chmod 644 after Mutagen sync)

### Dashboard Source
- `src/dashboard/app.py` — Main unified dashboard (SPY Predictor, ES Strategy, What-If, Monitoring, Grafana, Admin pages)
- `src/dashboard/monitoring.py` — Native Plotly monitoring (6 tabs: SPY, ES, System Health, Confidence API, Pipeline, Data Sources)
- `src/dashboard/realtime_app.py` — Real-time streaming dashboard
- `src/dashboard/es_dashboard.py` — ES strategy dashboard
- `src/dashboard/whatif_app.py` — What-If analysis

### Core Modules
- `src/data/` — Data fetching, features, DB routing, backfill, calendar, drift monitoring
- `src/model/` — Trainer, registry, ensemble, BiLSTM, conformal, regime, adaptive window, purged CV
- `src/es_strategy/` — ES futures engine, indicators, position management, RL trailing, labeling
- `src/llm/` — LLM analyzer and reporter (DeepSeek R1 70B via Ollama)
- `src/pipeline/` — Daily pipeline orchestration and alerts
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

- **FRED**: `dff4b18b046e602a474a8d1037619af1`
- **Finnhub**: `d6dn6u9r01qm89pk83m0d6dn6u9r01qm89pk83mg`
- **Polygon**: Not yet configured (placeholder in config.yaml)
- Keys are masked in the Admin config editor with a "Reveal sensitive values" toggle

## spy_state.json Fields

**Contains**: `updated_at`, `prediction` (direction, scale_label, confidence, probabilities), `indicators` (rsi_14, macd, atr_14, vix, vix_change, volume_ratio, sentiment_score), `flow_alerts`

**Does NOT contain**: `last_close`, `regime`, `ensemble_used`, `prediction_set`, `shap_drivers` — these come from the pipeline when it runs

## Data Consistency Rules

- SPY Predictor page and Monitoring SPY tab both read from `spy_state.json` for real-time metrics
- Historical charts read from SQLite/DuckDB
- SPY close price comes from the `prices` table (not in state file)
- VIX in summary cards falls back to live FRED macro data

## Docker / Cloud

- `cloud/docker-compose.yml` — Grafana, Prometheus, relay, dashboard containers
- Grafana container: `cloud-grafana-1`, config mounted from `grafana/grafana.ini`
- After Mutagen sync, `grafana.ini` permissions may reset — fix with `chmod 644`

## Conventions

- Commit messages: conventional commits (`fix:`, `feat:`, `chore:`, etc.)
- Testing on DGX: write temp script locally → Mutagen sync → SSH run → delete script
- Dashboard uses high-contrast dark theme throughout (CSS in app.py)
- All Plotly charts use `DARK_LAYOUT` template from monitoring.py
- `use_container_width=True` on all `st.plotly_chart()` calls
