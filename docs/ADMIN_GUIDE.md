# Stock Analysis Platform — Admin Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  NVIDIA DGX Spark (192.168.1.211)                           │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Polygon   │  │ yfinance │  │ Finnhub  │  │ FRED      │  │
│  │ WebSocket │  │ REST     │  │ RSS      │  │ Macro     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
│       └──────────────┴─────────────┴──────────────┘         │
│                          │                                  │
│                   ┌──────▼──────┐                           │
│                   │  SQLite DB  │  ./data/spy.db            │
│                   └──────┬──────┘                           │
│                          │                                  │
│  ┌───────────┐  ┌───────▼───────┐  ┌──────────────────┐   │
│  │ Ollama    │  │ Feature Eng.  │  │ ES Strategy      │   │
│  │ DeepSeek  │  │ + XGBoost     │  │ Engine           │   │
│  │ R1 70B    │  │ (GPU)         │  │ (Keltner/3-lot)  │   │
│  └─────┬─────┘  └───────┬───────┘  └────────┬─────────┘   │
│        │                │                    │              │
│        └────────────────┼────────────────────┘              │
│                         │                                   │
│              ┌──────────▼──────────┐                        │
│              │  Unified Dashboard  │  port 8501             │
│              │  (Streamlit)        │                        │
│              └──────────┬──────────┘                        │
│                         │                                   │
│              ┌──────────▼──────────┐                        │
│              │  Cloud Publisher    │  (optional)             │
│              └──────────┬──────────┘                        │
└─────────────────────────┼───────────────────────────────────┘
                          │ HTTPS
              ┌───────────▼───────────┐
              │  AWS EC2 t3.micro     │
              │  FastAPI Relay :8000  │
              │  Cloud Dashboard      │
              └───────────────────────┘
```

---

## Prerequisites

### Hardware
- NVIDIA DGX Spark with GB10 GPU
- 128GB+ RAM (DeepSeek R1 70B requires ~42GB VRAM)
- Network access to Polygon.io, Finnhub, Yahoo Finance, FRED

### Software
- Python 3.12+ with venv at `~/stockanalysis/.venv`
- Ollama 0.13+ (for LLM inference)
- SQLite 3.45+
- Git

### API Keys
- Polygon.io Advanced ($398/mo) — real-time stocks + options WebSocket
- Finnhub (free tier) — news headlines
- FRED (free) — macro data (no key needed for basic endpoints)
- Telegram Bot Token (optional) — for alerts
- SMTP credentials (optional) — for email alerts

---

## Installation

```bash
# Clone repository
git clone https://github.com/damerav/stockanalysis.git
cd stockanalysis

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -m src.data.init_db

# Backfill historical data (1 year)
python -m src.data.backfill --days 252

# Verify LLM
python -m src.launcher --check-llm

# Run first pipeline
python -m src.launcher --pipeline

# Start full system
python -m src.launcher --all
```

---

## Configuration Reference

All configuration is in `config.yaml` at the project root.

### polygon
| Key | Default | Description |
|-----|---------|-------------|
| `api_key` | `YOUR_POLYGON_KEY` | Polygon.io API key |
| `ws_stocks_url` | `wss://socket.polygon.io/stocks` | Stocks WebSocket endpoint |
| `ws_options_url` | `wss://socket.polygon.io/options` | Options WebSocket endpoint |

### analysis
| Key | Default | Description |
|-----|---------|-------------|
| `signal_interval` | `300` | Seconds between signal evaluations |
| `flow_sweep_threshold` | `50000` | Minimum notional ($) for sweep detection |
| `flow_block_threshold` | `100000` | Minimum notional ($) for block trade detection |

### llm
| Key | Default | Description |
|-----|---------|-------------|
| `model` | `deepseek-r1:70b` | Ollama model name |
| `base_url` | `http://localhost:11434` | Ollama API endpoint |
| `temperature` | `0.3` | LLM temperature for sentiment analysis |

### xgboost
| Key | Default | Description |
|-----|---------|-------------|
| `lookback_days` | `252` | Training window (trading days) |
| `max_depth` | `6` | Tree depth |
| `learning_rate` | `0.05` | Boosting learning rate |
| `n_estimators` | `500` | Max boosting rounds |
| `neutral_threshold` | `0.003` | ±0.3% daily return threshold for neutral classification |

### technicals
| Key | Default | Description |
|-----|---------|-------------|
| `sma_periods` | `[20, 50]` | Simple moving average periods |
| `rsi_period` | `14` | RSI lookback |
| `macd_fast` | `12` | MACD fast EMA |
| `macd_slow` | `26` | MACD slow EMA |
| `macd_signal` | `9` | MACD signal line |
| `bb_period` | `20` | Bollinger Band period |
| `bb_std` | `2` | Bollinger Band standard deviations |
| `atr_period` | `14` | ATR lookback |

### es_strategy
| Key | Default | Description |
|-----|---------|-------------|
| `credit_C` | `10.0` | Credit width for Keltner Channel entries |
| `strike_K` | `6000.0` | Strike reference price |
| `max_lots` | `3` | Maximum position size (1-3) |
| `jump_exit_points` | `5.0` | Points adverse for jump exit during 1-min hold |
| `emergency_stop_pct` | `0.20` | Emergency stop as % of C |
| `circuit_breaker_usd` | `-2000.0` | Daily loss limit to halt trading |
| `session_close_ct` | `15:55` | Flatten before this time (Central) |
| `session_reset_ct` | `17:00` | Reset circuit breaker at this time |
| `ai_enabled` | `false` | Enable XGBoost entry gate + CNN exit controller |
| `regime_lookback` | `10080` | Bars for ATR percentile regime detection |
| `regime_pct_low` | `33` | Percentile threshold for Low regime |
| `regime_pct_high` | `66` | Percentile threshold for High regime |

### sync
| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable cloud sync to AWS relay |
| `relay_url` | — | AWS relay server URL |
| `api_key` | — | API key for relay authentication |

### alerts
| Key | Default | Description |
|-----|---------|-------------|
| `telegram_token` | `""` | Telegram Bot API token |
| `telegram_chat_id` | `""` | Telegram chat ID for alerts |
| `email_smtp` | `""` | SMTP server address |
| `email_from` | `""` | Sender email address |
| `email_to` | `""` | Recipient email address |

### whatif
| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable What-If Analysis dashboard page |
| `max_sims` | `1000` | Maximum Monte Carlo simulations |
| `narrator_temperature` | `0.5` | LLM temperature for what-if narratives |
| `narrator_max_tokens` | `800` | Max tokens for LLM narrative |
| `es_default_lookback_days` | `20` | Default backtest window for ES what-if |
| `spy_default_noise_pct` | `2.0` | Default noise % for Monte Carlo |

### database
| Key | Default | Description |
|-----|---------|-------------|
| `path` | `./data/spy.db` | SQLite database file path |

---

## Database Schema

SQLite database at `./data/spy.db` with 11 tables:

| Table | Primary Key | Purpose |
|-------|-------------|---------|
| `prices` | date | Daily OHLCV for SPY |
| `technicals` | date | SMA, RSI, MACD, BB, ATR |
| `news` | id (auto) | News headlines from Finnhub + RSS |
| `daily_sentiment` | date | LLM-scored daily sentiment aggregate |
| `macro` | date | VIX, 10Y yield, DXY, fed funds, gold, crude |
| `predictions` | date | Model predictions with factors and report |
| `intraday_bars` | (timestamp, ticker) | 5-second OHLCV bars |
| `options_chain` | (date, contract_symbol) | Options chain with Greeks |
| `options_analytics` | date | Put/call ratio, max pain, IV skew, GEX |
| `intraday_features` | date | VWAP spread, momentum, range, volume ratio |
| `performance` | date | Prediction accuracy tracking |

Database uses WAL journal mode and 5-second busy timeout for concurrent access.

---

## Process Management

### Using Operations Scripts (Recommended)

Four shell scripts in `scripts/` provide the simplest way to manage the system:

```bash
./scripts/start.sh        # Start all components
./scripts/stop.sh         # Stop app components (keep Ollama)
./scripts/stop.sh --all   # Stop everything including Ollama
./scripts/status.sh       # Full health check
./scripts/restart.sh      # Stop + start
```

`start.sh` launches components in order:
1. Ollama (if not already running)
2. LLM health check
3. Unified dashboard on port 8501
4. ES strategy runner (paper mode)
5. Pipeline scheduler (daily at 4:30 PM ET)

`stop.sh` shuts down in reverse order with a 10-second graceful timeout before force-kill. Also cleans up orphaned processes on port 8501.

`status.sh` reports on:
- Process status (dashboard, ES strategy, scheduler, Ollama)
- Port status (8501, 11434)
- Database health (size, row counts, latest dates)
- Model info (latest file, total count)
- Disk usage and recent log activity

Scripts use `.pids/` for PID tracking and `logs/` for log files.

### Using the Python Launcher

Alternatively, use the Python launcher directly:

```bash
python -m src.launcher --all --config config.yaml
```

The Python launcher manages child processes and monitors their health:
- Restarts crashed processes automatically
- Logs process status every 30 seconds
- Graceful shutdown on Ctrl+C (SIGINT)

### Scheduler

The built-in scheduler triggers the daily pipeline at 4:30 PM ET, Monday through Friday. It checks the clock every 60 seconds and runs the pipeline once per day.

### Manual Pipeline Run

```bash
python -m src.launcher --pipeline
```

### ES Strategy Runner Modes

```bash
# Paper trading (signals logged, no execution)
python -m src.es_strategy.runner --mode paper --config config.yaml

# Backtesting from CSV
python -m src.es_strategy.runner --mode backtest --data es_1min.csv --config config.yaml

# With AI models enabled
python -m src.es_strategy.runner --mode paper --ai --config config.yaml
```

---

## Admin Console

The unified dashboard includes a built-in Admin Console accessible via the ⚙️ Admin sidebar option. It provides browser-based system management without SSH access.

### System Status Tab

Real-time health monitoring:
- Database: online/offline, size in MB, table count
- LLM (Ollama): connection status, target model availability, available models list
- XGBoost Model: latest model file name, size, total model count
- Data Inventory: row counts and date ranges for all 11 database tables
- Latest Prediction: date, direction, confidence, generation timestamp

### Actions Tab

Ad-hoc execution of pipeline steps and system operations:

| Action | Description |
|--------|-------------|
| 📥 Pull Latest Data | Gap detection + backfill prices and macro |
| 📰 Fetch News | Fetch latest headlines from Finnhub + RSS, insert into DB |
| 📊 Fetch Macro Data | Fetch VIX, yields, DXY, gold, crude from FRED |
| 📈 Compute Technicals | Recompute SMA, RSI, MACD, BB, ATR for all dates |
| 🧠 Retrain XGBoost | Retrain SPY predictor with latest data on GPU |
| 🔮 Generate Prediction | Run inference for next trading day, store in DB |
| 🩺 LLM Health Check | Verify Ollama + model availability |
| 📝 Generate Report | Generate LLM daily report for latest prediction |
| 🚀 Run Full Pipeline | Run all 13 pipeline steps (with "Skip LLM" checkbox for faster runs) |
| 📨 Send Test Alert | Send a test prediction alert via configured channels |

All actions show real-time progress spinners and display results (success/error with details) inline.

### Database Tab

- Table browser with configurable row limit (1-1000) and sort order (ASC/DESC)
- Custom SQL query editor (SELECT-only, write queries are blocked)
- Vacuum: reclaim unused space, shows bytes saved
- Integrity Check: runs `PRAGMA integrity_check`

### Configuration Tab

- Quick metrics: LLM model, XGB lookback days, ES max lots, cloud sync on/off
- Full YAML editor with syntax validation
- Save button writes directly to `config.yaml` — changes take effect on next component restart

### Logs Tab

- Dashboard Log: tail the most recent 20-200 lines from the Streamlit log
- Pipeline (last run): expandable view of the last 5 predictions with full report text
- Model Files: table of all trained models with file size and modification date

---

## Cloud Deployment (Optional)

### Architecture

The cloud deployment mirrors the local dashboards on AWS EC2 for remote access. The DGX pushes state updates to a FastAPI relay server; cloud dashboards read from the relay.

### Deployment Steps

```bash
cd cloud
chmod +x deploy_aws.sh
./deploy_aws.sh
```

This script:
1. Creates an ECR repository
2. Builds and pushes Docker images
3. Creates a security group (ports 8000, 8501)
4. Launches an EC2 t3.micro instance
5. Outputs the relay URL and dashboard URL

After deployment, update `config.yaml`:
```yaml
sync:
  enabled: true
  relay_url: "https://<ec2-ip>:8000"
  api_key: "<generated-key>"
```

### Docker Compose (Local Testing)

```bash
cd cloud
docker-compose up -d
```

Services:
- `relay` — FastAPI relay server on port 8000
- `dashboards` — Unified Streamlit dashboard on port 8501

### Relay Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/push/prediction` | POST | Push SPY prediction state |
| `/push/flow_alert` | POST | Push options flow alert |
| `/push/es_state` | POST | Push ES strategy state |
| `/push/heartbeat` | POST | 30-second keepalive |
| `/state` | GET | Get full SPY state |
| `/state/es` | GET | Get ES strategy state |
| `/stream` | GET | Server-Sent Events stream |
| `/health` | GET | Health check |
| `/admin/reset` | POST | Clear all state (admin key required) |

---

## Monitoring and Maintenance

### Log Monitoring

All components log to stdout. When run via the launcher, logs are captured from child processes.

### Database Maintenance

The SQLite database grows over time. To check size:
```bash
ls -lh data/spy.db
```

To vacuum (reclaim space after deletes):
```bash
sqlite3 data/spy.db "VACUUM;"
```

### Model Files

Trained XGBoost models are saved to `./models/` with date stamps:
```
models/xgb_spy_20260221.json
```

Old model files can be safely deleted. The system always uses the latest.

### Ollama Model Management

```bash
# Check model status
ollama list

# Pull/update model
ollama pull deepseek-r1:70b

# Check Ollama is running
curl http://localhost:11434/api/tags
```

### Health Checks

The quickest way to check system health is via the operations script:
```bash
./scripts/status.sh
```

Or use the Admin Console → System Status tab in the browser.

For individual checks via CLI:

```bash
# LLM health
python -m src.launcher --check-llm

# Database integrity
sqlite3 data/spy.db "PRAGMA integrity_check;"

# Check dashboard is responding
curl -s -o /dev/null -w "%{http_code}" http://localhost:8501
```

---

## Backup and Recovery

### What to Back Up

| Item | Path | Priority |
|------|------|----------|
| Database | `data/spy.db` | High — contains all historical data |
| Models | `models/*.json` | Medium — can be retrained |
| Config | `config.yaml` | High — contains all settings |
| State files | `data/*.json` | Low — regenerated on next run |

### Backup Command

```bash
cp data/spy.db data/spy.db.backup.$(date +%Y%m%d)
```

### Recovery

Replace the database file and restart the system. If the database is lost, re-run the backfill:
```bash
python -m src.data.backfill --days 252
python -m src.launcher --pipeline
```

---

## Security Notes

- The Polygon API key is stored in plain text in `config.yaml`. Restrict file permissions: `chmod 600 config.yaml`
- The relay server uses API key authentication via `X-API-Key` header
- Cloud relay stores state in memory only — no persistent data on AWS
- SMTP credentials for email alerts should use app-specific passwords
- The system does not execute trades — it generates signals only
