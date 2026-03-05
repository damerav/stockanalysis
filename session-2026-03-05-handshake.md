# Session Handshake — March 4-5, 2026

## Version: v2.8.0 → v2.8.1

## Summary

This session delivered 6 features across 5 commits on `main`:

| Commit | Description |
|--------|-------------|
| `ed57576` | fix: Buffett Indicator — replaced retired FRED Wilshire 5000 with yfinance `^W5000` |
| `56a2333` | feat: shutdown/startup scripts (`scripts/shutdown.sh`, `scripts/startup.sh`) |
| `ad19a26` | feat: Inverted Strangle with Defined Risk dashboard (6 tabs, 2 new DB tables) |
| `7d9b9e2` | feat: Predictive & Risk Mitigation Engine (IV Rank, VIX term structure, SKEW, spike probability) |
| `601a65a` | feat: live spot price via yfinance in New Trade form |

## What Was Built

### 1. Buffett Indicator Fix
FRED retired all Wilshire 5000 series (WILL5000PR, WILL5000IND, WILL5000INDFC — all return 404). Replaced with yfinance `^W5000` where 1 index point ≈ $1B total US market cap. GDP still from FRED API. Validated: `(68,217 / 31,490) × 100 = 216.6%`. Data populated in `market_breadth` table.

**File**: `src/data/market_breadth.py`

### 2. Shutdown/Startup Scripts
- `scripts/shutdown.sh` — Kills scheduler + streamlit, frees ports 8501/8100, clears `__pycache__` and Streamlit cache
- `scripts/startup.sh` — Starts scheduler with `--spy --config config.yaml`, waits for HTTP 200 health check on port 8501

### 3. Inverted Strangle Dashboard
Full options strategy management page (`src/dashboard/strangle_app.py`, ~1060 lines) with 6 tabs:

| Tab | Purpose |
|-----|---------|
| Open Positions | Active trades with P&L curves, live Greeks (Polygon), adjustment rolls, close tracking |
| New Trade | Build new spread with live spot price, pre-trade checklist, auto-pricing via Polygon |
| Prediction & Risk | IV Rank gauge, VIX term structure, CBOE SKEW, heuristic spike probability |
| History & Performance | Cumulative P&L chart, win rate, avg P&L, close reason breakdown |
| Tracker | Rolling 10-trade P&L, rolling win rate, win/loss streaks, VIX regime breakdown |
| Strategy Guide | Quick reference for trade architecture, execution rules, Greek characteristics |

**Database**: 2 new PostgreSQL tables
- `inverted_strangle_positions` (27 columns) — trade records with entry conditions
- `inverted_strangle_adjustments` (11 columns) — roll/adjustment history

**Files**: `src/dashboard/strangle_app.py`, `src/data/init_db.py`, `src/dashboard/app.py`

### 4. Predictive & Risk Mitigation Engine
Four-layer framework integrated into the Prediction & Risk tab:

| Layer | Signal | Source |
|-------|--------|--------|
| 1 | IV Rank / IV Percentile | VIX 52-week history via yfinance |
| 2 | VIX Term Structure (contango/backwardation) | VIX vs VIX3M via yfinance |
| 3 | CBOE SKEW + Put/Call Ratio | yfinance SKEW + Polygon options analytics |
| 4 | VIX Spike Probability (heuristic) | Weighted score from all above factors |

Additional features:
- Pre-trade checklist (3 pass/fail checks) on New Trade form
- Adjustment alerts (21 DTE, 50% profit, Delta breach) on Open Positions tab
- Entry conditions stored per position (`entry_iv_rank`, `entry_vix_term_structure`)
- 3 new DB columns: `position_delta`, `entry_iv_rank`, `entry_vix_term_structure`

### 5. Live Spot Price
New Trade form now fetches live underlying price via yfinance (`_live_spot()` with 15-second cache) instead of hardcoded $540.00 default. Ticker-aware — changes when underlying input changes.

## Validation Results (DGX)
```
65/65 modules imported OK
20/20 database tables OK
3/3 new strangle columns OK (position_delta, entry_iv_rank, entry_vix_term_structure)
Market breadth: CAPE=30.81, Buffett=216.63
206/206 features (0 duplicates)
6/6 encrypted secrets OK
Dashboard HTTP 200
```

## System State
- **Scheduler**: PID 829513 on DGX
- **Streamlit**: PID 829998 on port 8501
- **Dashboard URL**: http://192.168.1.211:8501 / trading.aiagenticinternational.org
- **Git**: All commits pushed to `origin/main`
- **Documentation**: `.kiro/steering/project-context.md` updated with v2.8.1 changes
