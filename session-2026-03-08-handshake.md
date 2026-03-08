# Session Handshake — 2026-03-08 — v2.9.2 Prediction Calibration

> **Resume instruction**: Open this file and tell Kiro "Resume from this handshake."

---

## Current System State

### Version & Git
- **Version**: v2.9.2
- **Git tag**: `v2.9.2-prediction-calibration` (commit `36c72fd`)
- **Branch**: `main`, pushed to GitHub
- **Backup**: `backups/v2.9.2_20260308_130210/` on DGX (145.6 MB — PG dump, models, config, state)
- **Restore**: `bash backups/v2.9.2_20260308_130210/restore.sh` or `git checkout v2.9.2-prediction-calibration`

### Model
- **Active model**: `models/xgb_spy_20260307.json` (March 7 build)
- **Features**: 316 (no feature selection — full dataset performs best)
- **Classes**: 3-class (DOWN=0, NEUTRAL=1, UP=2)
- **Binary model**: DISABLED (`use_binary_model=False`)

### Backtest Accuracy (2512 days, all data)
| Metric | Value |
|--------|-------|
| Overall | 76.1% |
| BULLISH | 85.8% (1175 days) |
| BEARISH | 66.3% (906 days) |
| NEUTRAL | 70.1% (431 days) |
| High-confidence misses | 14.1% |

### Strategy Rules DB (prediction group)
| Rule | Value | Note |
|------|-------|------|
| `use_binary_model` | `False` | Binary was hurting accuracy |
| `bearish_extra_margin` | `0.0` | Disabled — margin scaling sufficient |
| `binary_confidence_gate` | `0.0` | N/A since binary disabled |
| `confidence_dampening_factor` | `0.85` | Applied in choppy/bear regimes |
| `neutral_confidence_threshold` | `0.42` | 3-class neutral gate |
| `neutral_threshold` | `0.004` | Target labeling band |
| `lookback_days` | `252` | Training window |

### Services Running on DGX (192.168.1.211)
- Scheduler: `python -m src.launcher --spy` (manages pipeline + dashboard)
- Dashboard: port 8501 (HTTP 200 confirmed)
- PostgreSQL: Docker container `postgres`, port 5432
- Ollama: DeepSeek R1 70B + 14B

### What Changed This Session

1. **Disabled binary model** — it was overriding correct 3-class predictions (bull accuracy dropped from 88.6% to 75.9% with binary on)
2. **Added margin-based confidence scaling** in `predict()` — when gap between top-2 probabilities is thin, confidence scales down. Reduced high-confidence misses from ~100% to 14.1%
3. **Added bearish bias correction** then disabled it (0.0) — hurt bear accuracy without enough benefit
4. **Enhanced regime dampening** — choppy regime gets `factor * 0.9`
5. **Reverted to March 7 model** — March 8 experimental models with aggressive feature selection performed worse (48% vs 76.1%)
6. **Removed duplicate `_align_features`** method in trainer.py
7. **Performance dashboard** defaults to "All Data" view, performance table rebuilt from backtest
8. **Full backup created** on DGX with restore script

### Files Modified
- `src/model/trainer.py` — predict() inference logic (margin scaling, bearish correction, regime dampening)
- `src/dashboard/performance_app.py` — default time range to "All Data"
- `src/data/init_db.py` — bearish_extra_margin rule seed
- `.kiro/steering/project-context.md` — version bump + v2.9.2 changelog

### Known Issues / Future Work
- **Recent 6-month accuracy is 58.1%** — the model struggles more on recent data (last 124 trading days). The 76.1% is all-time across 10 years. Recent market conditions are harder.
- **Bear day accuracy (66.3%)** is the weakest direction. The bearish_extra_margin was an attempt to fix this but it made things worse. Could explore regime-specific thresholds or separate bear-market models.
- **Confidence calibration** could be improved further — isotonic calibration is trained but not heavily tested in production.
- **Mutagen is PAUSED** — use SCP to transfer files: `scp <local> abidamera@192.168.1.211:~/stockanalysis/<remote>`

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
