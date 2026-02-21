# Stock Analysis Platform — User Guide

## Overview

The Stock Analysis Platform is a real-time market analysis and signal generation system. It produces trading signals and predictions for SPY/SPX and E-mini S&P 500 (ES) futures. All trades are executed manually by the user — the system provides analysis only.

Access the unified dashboard at: `http://192.168.1.211:8501`

---

## Getting Started

### Launching the System

From the DGX Spark server:

```bash
cd ~/stockanalysis
./scripts/start.sh
```

Or using the Python launcher directly:

```bash
source .venv/bin/activate
python -m src.launcher --all
```

Both approaches start all components: LLM health check, ES strategy runner (paper mode), dashboards, and the daily scheduler.

### Quick-Start Options

| Command | What It Does |
|---------|-------------|
| `python -m src.launcher --all` | Start everything |
| `python -m src.launcher --spy` | SPY predictor + dashboard only |
| `python -m src.launcher --es` | ES strategy + dashboard only |
| `python -m src.launcher --dashboards-only` | Dashboards without backends |
| `python -m src.launcher --check-llm` | Verify LLM is available |
| `python -m src.launcher --pipeline` | Run the daily pipeline immediately |

### First-Time Setup

If this is a fresh install, backfill 1 year of historical data:

```bash
python -m src.data.backfill --days 252
```

This loads 252 trading days of SPY prices, technicals, macro data, and news into the SQLite database.

---

## Dashboard Navigation

The unified dashboard runs on port 8501 with a sidebar menu containing four pages.

### 1. SPY Predictor

Displays the current next-day SPY direction prediction.

- Prediction banner: color-coded from green (STRONG BULLISH) to red (STRONG BEARISH) with confidence percentage
- Probability breakdown: shows up/neutral/down percentages
- Prediction history: bar chart of the last 20 predictions with color-coded directions
- Accuracy tracking: cumulative accuracy of past predictions vs actual outcomes
- Key indicators: RSI(14), MACD, ATR(14), VIX, volume ratio, sentiment score
- Options flow alerts: real-time sweeps and block trades detected from SPX options

The prediction scale has 5 levels:

| Level | Meaning |
|-------|---------|
| STRONG BULLISH | High confidence SPY will rise significantly |
| BULLISH | SPY likely to rise |
| NEUTRAL | No clear directional signal |
| BEARISH | SPY likely to fall |
| STRONG BEARISH | High confidence SPY will fall significantly |

This page auto-refreshes every 15 seconds during market hours.

### 2. ES Strategy

Displays live ES futures strategy signals and position state.

- Position banner: shows LONG/SHORT/FLAT, lot count, entry price, unrealized P&L, and current volatility regime
- Price chart: Plotly candlestick with Keltner Channel bands (shaded), VWAP overlay (dotted), entry/exit markers, and stop levels
- RSI subplot: synced with the main chart
- Signal feed: scrolling log of entries, exits, stops, AI rejects, and circuit breaker events
- Status panel: circuit breaker status, daily P&L, trade count, session status, per-lot breakdown

Signal types you'll see in the feed:

| Signal | Meaning |
|--------|---------|
| 🟢 ENTRY_LONG | Long entry triggered |
| 🔴 ENTRY_SHORT | Short entry triggered |
| 💰 EXIT_TP1 / EXIT_TP2 | Take-profit hit on Lot 0 or Lot 1 |
| 🏃 EXIT_RUNNER | Runner lot exited |
| 🛑 STOP_HIT | Stop loss triggered |
| 📍 STOP_UPDATE | Trailing stop ratcheted |
| 🤖 AI_REJECT | AI model rejected the entry |
| ⚡ CIRCUIT_BREAKER | Daily loss limit hit, trading paused |
| 🕐 SESSION_FLATTEN | End-of-session auto-flatten |

This page auto-refreshes every 5 seconds.

### 3. What-If Analysis

Interactive scenario testing for both subsystems. No auto-refresh — results appear when you click "Run."

#### ES Strategy Tab

- K/C Sweep: test different Keltner Channel strike (K) and credit (C) combinations. Produces a heatmap of P&L across the parameter grid.
- Lot Sizing: compare 1-lot, 2-lot, and 3-lot configurations side by side.
- Risk Limits: sweep circuit breaker thresholds from -$1K to -$3K to see the P&L vs protection trade-off.
- Custom Compare: define two custom ES configurations and compare their backtest results.

#### SPY Predictor Tab

- Feature Override: change individual features (VIX, sentiment, RSI, put/call ratio) and see how the prediction changes.
- Feature Ablation: zero out selected features to measure which ones matter most for accuracy.
- Monte Carlo: add random noise to all features across 100-1000 simulations to see the prediction distribution.
- Stress Test: run pre-built scenarios (VIX spike to 40, 3% gap down, March 2020 crash, Fed rate cut, melt-up rally).
- Threshold Sensitivity: test how the neutral zone threshold affects prediction accuracy.

### 4. Admin Console

The Admin Console provides system management tools organized into 5 tabs.

#### System Status Tab

Displays real-time health of three core components:
- Database: online/offline status, size, table count
- LLM (Ollama): connection status, model availability
- XGBoost Model: latest model file, size, total model count

Also shows a data inventory table (row counts and date ranges for all 11 tables) and the latest prediction details.

#### Actions Tab

Run pipeline steps individually or trigger full operations on demand.

| Action | Description |
|--------|-------------|
| 📥 Pull Latest Data | Gap detection + backfill prices and macro |
| 📰 Fetch News | Fetch latest headlines from Finnhub + RSS |
| 📊 Fetch Macro Data | Fetch VIX, yields, DXY, gold, crude from FRED |
| 📈 Compute Technicals | Recompute SMA, RSI, MACD, BB, ATR |
| 🧠 Retrain XGBoost | Retrain SPY predictor with latest data (GPU) |
| 🔮 Generate Prediction | Run inference for next trading day |
| 🩺 LLM Health Check | Check Ollama + model availability |
| 📝 Generate Report | Generate LLM daily report for latest prediction |
| 🚀 Run Full Pipeline | Run all 13 pipeline steps (with optional "Skip LLM" checkbox) |
| 📨 Send Test Alert | Send a test prediction alert via Telegram/email |

#### Database Tab

- Table browser: select any table, set row limit and sort order, view data in a grid
- Custom SQL: run read-only SELECT queries against the database
- Maintenance: Vacuum (reclaim space) and Integrity Check buttons

#### Configuration Tab

- Quick view of key settings: LLM model, XGB lookback, ES max lots, cloud sync status
- Full YAML editor with validation — edit and save `config.yaml` directly from the browser
- Changes take effect on next component restart

#### Logs Tab

Three log sources:
- Dashboard Log: view the most recent 20-200 lines from `/tmp/dashboard.log`
- Pipeline (last run): expandable view of the last 5 predictions with report text
- Model Files: list of all trained model files with size and modification date

---

## Operations Scripts

Four shell scripts in `scripts/` manage the system on the DGX Spark. All scripts auto-detect the project directory and use `.pids/` for PID tracking and `logs/` for log files.

### Starting the System

```bash
./scripts/start.sh
```

Starts components in order:
1. Ollama (if not already running)
2. LLM health check
3. Unified dashboard on port 8501
4. ES strategy runner (paper mode)
5. Pipeline scheduler (daily at 4:30 PM ET)

Prints a summary with dashboard URL, PID locations, and log paths.

### Stopping the System

```bash
./scripts/stop.sh        # Stop app components (keep Ollama)
./scripts/stop.sh --all  # Stop everything including Ollama
```

Stops in reverse order with a 10-second graceful timeout before force-kill. Also cleans up orphaned processes on port 8501.

### Checking Status

```bash
./scripts/status.sh
```

Shows:
- Process status (dashboard, ES strategy, scheduler, Ollama)
- Port status (8501, 11434)
- Database health (size, row counts, latest dates for key tables)
- Model info (latest model file, total count)
- Disk usage (project size, free space)
- Recent log activity (last line from each log file)

### Restarting

```bash
./scripts/restart.sh
```

Runs `stop.sh` then `start.sh` with a 2-second pause between.

---

## Understanding the Predictions

### SPY Predictor

The XGBoost model uses 37+ features across 5 categories:

- Technical: RSI, MACD, Bollinger Bands, moving averages, ATR
- Macro: VIX, 10Y yield, DXY, fed funds rate, gold, crude oil
- Sentiment: LLM-scored news sentiment from 50 daily articles
- Intraday: VWAP spread, momentum, range, volume ratio
- Options: put/call ratio, max pain distance, IV skew, GEX

The model retrains daily at 4:30 PM ET with the latest data and generates a prediction for the next trading day.

### ES Strategy

The strategy uses Keltner Channel bands for entry signals with a 3-lot tiered exit system:

- Lot 0 (TP1): first take-profit target, tightest stop
- Lot 1 (TP2): second take-profit, medium trail
- Lot 2 (Runner): widest trail, captures extended moves

Volatility regime (Low/Med/High) adapts all parameters automatically based on recent ATR percentiles.

---

## Daily Pipeline

The automated pipeline runs at 4:30 PM ET Monday through Friday. It executes 13 steps:

1. Check LLM availability
2. Backfill any missing data
3. Evaluate yesterday's prediction accuracy
4. Fetch today's prices
5. Fetch news articles
6. Run LLM sentiment analysis (~60-90 min)
7. Fetch macro data (VIX, yields, etc.)
8. Fetch options chain snapshot
9. Compute options analytics
10. Compute technical indicators
11. Build features and retrain XGBoost
12. Generate next-day prediction
13. Generate LLM daily report
14. Send alerts (Telegram/email)

If the LLM is unavailable, the pipeline continues with neutral sentiment — it never aborts.

---

## Alerts

Predictions can be sent via Telegram and/or email after each daily pipeline run. Configure in `config.yaml` under the `alerts` section. Both channels are optional.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dashboard shows "Waiting for data" | Run the pipeline once: `python -m src.launcher --pipeline` or use Admin Console → Actions → Run Full Pipeline |
| LLM sentiment shows 0.0 | Check Ollama: `python -m src.launcher --check-llm` or Admin Console → Actions → LLM Health Check |
| No prediction history chart | Need at least 2 pipeline runs to build history |
| ES dashboard shows FLAT with no signals | Normal outside market hours or in paper mode |
| Dashboard not loading | Check status: `./scripts/status.sh` |
| Need to restart everything | Run `./scripts/restart.sh` |
| Check what's running | Run `./scripts/status.sh` for full health report |
