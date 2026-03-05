# Session Handoff — 2026-03-04

## What Was Done This Session

### 1. PostgreSQL Schema Fixes (deployed to DGX)
- Added `quality_score` column to `raw_articles` table
- Added `feature_version` column to `feature_cache` table
- Verified `news_features`, `feature_store_meta` tables exist

### 2. DbRouter Robustness
- Auto-loads `config.yaml` when no config passed (fixes `get_router()` without args)
- Handles corrupted SQLite gracefully (recreates instead of crashing)
- Added `news_features`, `strategy_rules`, `feature_store_meta` to `_TABLE_PKS`

### 3. Pipeline Fixes
- Step 10 regime detection: queries prices+macro tables directly instead of feature store cache (fixes `KeyError: 'close'`)
- news_features `store_features()`: proper upsert with column list
- Pipeline runs in ~37s total, all steps pass except Step 11 (no price data for today — data availability, not a bug)

### 4. Dashboard Performance Improvements
- CSS cached in `session_state` (no disk read per rerun)
- yfinance ticker cached with `st.cache_data(ttl=15)`, lazy-imported
- `spy_state.json`, `es_state.json`, predictions, performance all cached
- `single_stock_app.py`: yfinance lazy-imported (faster module load)

### 5. New Feature: Predicted vs Actual Chart
- Dual-axis line chart on Performance page
- Left axis: predicted direction (blue) vs actual (orange dashed), red X on misses
- Right axis: SPY OHLC candlesticks at 40% opacity

## Current System State

- **Version**: v2.7.1
- **Latest commit**: `1d76431` on `main`, pushed to `origin/main`
- **DGX Dashboard**: Running on port 8501 (PID on 8501 confirmed)
- **DGX Scheduler**: NOT running — needs manual restart (see below)
- **Mutagen**: PAUSED — all files SCP'd manually
- **PostgreSQL**: All schema fixes applied, all tables have correct columns
- **SQLite (`spy.db`)**: Corrupted on DGX — DbRouter handles this gracefully now, always uses PostgreSQL

## Commits This Session

| Hash | Message |
|------|---------|
| `64c70e6` | fix: PostgreSQL schema fixes, pipeline robustness, dashboard perf |
| `ff373ac` | feat: add predicted vs actual daily line chart to Performance page |
| `1d12867` | feat: add OHLC candlestick overlay to predicted vs actual chart |
| `1d76431` | fix: remove duplicate legend kwarg in pred vs actual chart |

## Known Issues / TODO

1. **Step 11 "no features for today"**: `build_feature_vector(date=today)` returns empty because no price row exists for today. This happens when yfinance doesn't have same-day data yet. Not a code bug — the pipeline generates predictions only when price data is available.

2. **Step 10 `'close'` KeyError**: FIXED in code but the fix was deployed mid-session. The latest `daily_run.py` on DGX has the fix (queries prices table directly for regime detection).

3. **Scheduler not running**: Was killed during debugging. Restart with:
   ```bash
   ssh abidamera@192.168.1.211 "cd ~/stockanalysis && source .venv/bin/activate && nohup python -m src.launcher --spy --config config.yaml > logs/scheduler.log 2>&1 &"
   ```

4. **Corrupted `spy.db` on DGX**: The SQLite file at `data/spy.db` is corrupted. Not critical since PostgreSQL is the primary database. DbRouter now handles this gracefully. To fix permanently: `ssh abidamera@192.168.1.211 "rm ~/stockanalysis/data/spy.db"` and let `init_db` recreate it.

5. **Temp files on DGX**: All cleaned up. No `_tmp_*.py` files remain.

6. **Untracked local files** (not committed, not needed):
   - `*.pdf`, `*.docx` — reference documents
   - `kiro_rules_and_ai_prompt*.md` — implementation prompts (historical)
   - `session-2026-03-03-handoff.md` — previous session handoff
   - `tmp_verify.py` — one-off validation script
   - `models/xgb_spy_20260302*` — model artifacts (gitignored on DGX)

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/data/db_router.py` | Auto-config loading, corrupted SQLite handling, expanded `_TABLE_PKS` |
| `src/data/init_db.py` | Corrupted SQLite handling in `get_connection()` |
| `src/data/news_features.py` | Proper upsert with column list in `store_features()` |
| `src/pipeline/daily_run.py` | Step 10 regime fix (prices table query), try/except around legacy conn |
| `src/dashboard/app.py` | CSS caching, yfinance ticker caching, state file caching, lazy imports |
| `src/dashboard/single_stock_app.py` | Lazy yfinance imports |
| `src/dashboard/performance_app.py` | New predicted vs actual + OHLC dual-axis chart |
| `.kiro/steering/project-context.md` | Updated with v2.7.1 changes |

## Quick Resume Commands

```bash
# Check dashboard status
ssh abidamera@192.168.1.211 "fuser 8501/tcp 2>/dev/null && echo UP || echo DOWN"

# Restart dashboard
ssh abidamera@192.168.1.211 "fuser -k 8501/tcp 2>/dev/null; sleep 2; cd ~/stockanalysis && source .venv/bin/activate && nohup streamlit run src/dashboard/app.py --server.port 8501 --server.headless true > logs/dashboard.log 2>&1 &"

# Start scheduler
ssh abidamera@192.168.1.211 "cd ~/stockanalysis && source .venv/bin/activate && nohup python -m src.launcher --spy --config config.yaml > logs/scheduler.log 2>&1 &"

# Run pipeline manually
ssh abidamera@192.168.1.211 "cd ~/stockanalysis && source .venv/bin/activate && python -m src.pipeline.daily_run --skip-llm"

# SCP a file to DGX
scp src/path/to/file.py abidamera@192.168.1.211:~/stockanalysis/src/path/to/
```
