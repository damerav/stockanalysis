# Session Handoff — 2026-03-08 — Prediction Calibration v2.9.2

## What Was Done

### Prediction Inference Fixes (trainer.py `predict()`)
1. **Binary model disabled** (`use_binary_model=False`): Was overriding correct 3-class predictions, dropping bull accuracy from 88.6% to 75.9%
2. **Margin-based confidence scaling** (NEW): Scales confidence down when top-2 probability gap is thin. Formula: `min(1.0, 0.7 + 2.0 * prob_margin)`. Reduces HC misses from ~100% to 14.1%
3. **Bearish bias correction** (added then disabled at 0.0): Hurt bear accuracy without enough benefit since margin scaling handles HC misses
4. **Enhanced regime dampening**: Choppy regime gets `factor * 0.9` (stronger than bear)
5. **Removed duplicate `_align_features`**: Dead code at line 1105 (Python used the one at 1332)

### Model State
- Using March 7 model: `models/xgb_spy_20260307.json` (316 features, no feature selection)
- March 8 experimental models deleted
- Backtest: 2512 days, 76.1% overall, bull=85.8%, bear=66.3%, neutral=70.1%, HC misses=14.1%

### DB Rules State
- `use_binary_model`: False
- `bearish_extra_margin`: 0.0 (disabled)
- `binary_confidence_gate`: 0.0
- `confidence_dampening_factor`: 0.85
- `neutral_confidence_threshold`: 0.42

### Dashboard Changes
- `performance_app.py`: Time range defaults to "All Data" instead of "Last 6 Months"
- `performance` table rebuilt from backtest_results (2512 rows, 76.1%)
- `backtest_results` table populated with fresh backtest

### Files Modified
- `src/model/trainer.py` — predict() inference logic
- `src/dashboard/performance_app.py` — default time range
- `src/data/init_db.py` — bearish_extra_margin rule seed
- `.kiro/steering/project-context.md` — version bump to v2.9.2

### System State
- Scheduler running on DGX (PID varies)
- Dashboard on port 8501 (HTTP 200)
- Git tag: `v2.9.2-prediction-calibration`
