# Session Handshake — 2026-03-11 — Regime Calibration + Bullish Bias Infrastructure

> **Resume instruction**: Open this file and tell Kiro "Resume from this handshake."

---

## Current System State

### Version & Git
- **Latest commit**: `2b1daf3` — regime-specific calibration (bullish_extra_margin + low_vol_range dampening)
- **Branch**: `main`, pushed to GitHub
- **Previous restore point**: `v2.9.2-prediction-calibration` tag (commit `36c72fd`)
- **Backup**: `backups/v2.9.2_20260308_130210/` on DGX (145.6 MB)

### Model
- **Active model**: `models/xgb_spy_20260307.json` (March 7 build)
- **Features**: 316 (no feature selection)
- **Classes**: 3-class (DOWN=0, NEUTRAL=1, UP=2)
- **Binary model**: DISABLED

### Backtest Accuracy (2514 days, adaptive threshold evaluation)
| Metric | Value |
|--------|-------|
| Overall | 70.2% |
| BULLISH | 83.6% (859 days) |
| BEARISH | 58.1% (614 days) |
| NEUTRAL | 66.4% (1041 days) |

> **Note on accuracy change**: Previous session reported 76.1% (2512 days). The drop to 70.2% is NOT caused by code changes — it's because the DGX previously ran a different version of `generate_historical_backtest` that evaluated actual labels differently. The current code uses `get_adaptive_neutral_threshold()` which widens the neutral band based on VIX, creating more NEUTRAL actual days (1041 vs 431). Prediction directions are unchanged.

### Strategy Rules DB (prediction group — 10 rules)
| Rule | Value | Note |
|------|-------|------|
| `use_binary_model` | `False` | Binary was hurting accuracy |
| `bearish_extra_margin` | `0.0` | Disabled — margin scaling sufficient |
| `bullish_extra_margin` | `0.0` | **NEW** — disabled, ready for tuning |
| `binary_confidence_gate` | `0.0` | N/A since binary disabled |
| `confidence_dampening_factor` | `0.85` | Applied in choppy/bear/low_vol regimes |
| `neutral_confidence_threshold` | `0.42` | 3-class neutral gate |
| `neutral_threshold` | `0.004` | Target labeling band (was 0.0015, fixed back) |
| `lookback_days` | `1260` | Training window (5 years) |

### Bullish Bias Data (validated)
- Model predicts BULLISH 57.6% vs actual 46.8% (over-predicts by ~11 points)
- Worst in `low_vol_range`: predicted BULL=340 vs actual=251
- Bear accuracy in low_vol_range: 63.4% (lowest among regimes)

### Services Running on DGX (192.168.1.211)
- Dashboard: port 8501 (restarted this session)
- PostgreSQL: Docker container `postgres`, port 5432
- Ollama: DeepSeek R1 70B + 14B
- Scheduler: may need restart (only dashboard was restarted)

---

## What Changed This Session

### Implemented (from "Regime-Specific Calibration" prompt)

1. **Validated bullish bias** — ran `tmp_bias_check.py` on DGX, confirmed model over-predicts BULLISH by ~11 points overall, worst in `low_vol_range` regime

2. **Added `bullish_extra_margin` rule** (set to 0.0, disabled)
   - Seeded in DB with full metadata (float, 0.0-0.10 range)
   - Added to `init_db.py` defaults
   - Loaded from rules_store in `SPYPredictor.__init__`
   - Bullish bias correction block added to `predict()` (after bearish correction, before margin scaling)
   - Disabled because: same approach with bearish_extra_margin hurt accuracy. Bull accuracy is 83.6% — too valuable to risk without probability-level simulation

3. **Added `low_vol_range` regime dampening** (factor * 0.95)
   - Added to regime dampening block in `predict()` alongside existing choppy (0.9) and bear (1.0) dampening
   - Only affects confidence, not direction — safe change
   - Addresses the regime where bullish bias is most pronounced

4. **Fixed `neutral_threshold`** — DB had 0.0015 (someone changed it), reset to 0.004 (training value)

5. **Added `bearish_extra_margin`** to `init_db.py` defaults (was only in DB, not in seed list)

6. **Rebuilt performance table** from backtest_results (2514 rows)

### Files Modified
- `src/model/trainer.py` — `__init__` (load bullish_extra_margin), `predict()` (bullish correction block + low_vol_range dampening)
- `src/data/init_db.py` — added bearish_extra_margin + bullish_extra_margin to `_STRATEGY_RULE_DEFAULTS`

### Previous Session Work (for context)
- v2.9.2 prediction calibration (margin scaling, binary model disabled, HC misses 14.1%)
- Renamed `backfill_2y.py` → `backfill_historical.py`
- Added rolling walk-forward accuracy (63-day, 126-day) to performance dashboard
- Full backup at `backups/v2.9.2_20260308_130210/`

---

## Known Issues / Future Work

- **Backtest accuracy discrepancy**: 70.2% (current adaptive threshold eval) vs 76.1% (previous eval). The adaptive threshold creates more NEUTRAL actual days. May want to investigate whether the adaptive threshold is too aggressive or if the previous eval was too lenient.
- **`bullish_extra_margin` ready but disabled**: Infrastructure is in place. To test, set it to 0.02 via Strategy Rules dashboard and run a backtest. Need probability columns in `backtest_results` table to do proper simulation first.
- **`backtest_results` table lacks probability columns**: `generate_historical_backtest` computes `prob_down/neutral/up` but `run_historical_backtest` doesn't persist them. Adding these would enable margin simulation without re-running the full backtest.
- **Scheduler may need restart**: Only dashboard was restarted this session. Check with `ssh abidamera@192.168.1.211 "pgrep -af src.launcher"`
- **Mutagen is PAUSED** — use SCP for file transfers

---

## Quick Reference Commands

```bash
# Start scheduler + dashboard
ssh abidamera@192.168.1.211 "cd ~/stockanalysis && source .venv/bin/activate && nohup python -m src.launcher --spy --config config.yaml > logs/scheduler.log 2>&1 &"

# Dashboard only
ssh abidamera@192.168.1.211 "fuser -k 8501/tcp 2>/dev/null; sleep 1; cd ~/stockanalysis && source .venv/bin/activate && nohup streamlit run src/dashboard/app.py --server.port 8501 --server.headless true > logs/dashboard.log 2>&1 &"

# SCP a file to DGX
scp <local_file> abidamera@192.168.1.211:~/stockanalysis/<remote_path>

# Run a script on DGX
ssh abidamera@192.168.1.211 "cd ~/stockanalysis && source .venv/bin/activate && python <script>"

# Restore from backup
ssh abidamera@192.168.1.211 "bash ~/stockanalysis/backups/v2.9.2_20260308_130210/restore.sh"

# Restore code only
git checkout v2.9.2-prediction-calibration
```
