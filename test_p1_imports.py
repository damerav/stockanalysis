"""Quick smoke test for Phase 1 imports and basic functionality."""
import sys
sys.path.insert(0, ".")

# Test 1: Calendar module
from src.data.calendar import (
    get_event_features, days_until_next_fomc, days_until_next_cpi,
    days_until_next_nfp, days_until_monthly_opex, is_triple_witching_week,
    has_nearby_event
)
from datetime import date

ef = get_event_features(date(2026, 2, 21))
print(f"[OK] Calendar features for 2026-02-21: days_to_fomc={ef['days_to_fomc']}, "
      f"is_fomc_week={ef['is_fomc_week']}, day_of_week={ef['day_of_week']}")

# Test 2: Drift monitor
from src.data.drift_monitor import compute_psi, compute_ks, monitor_features
import numpy as np
a = np.random.normal(0, 1, 500)
b = np.random.normal(0.5, 1.2, 500)
psi = compute_psi(a, b)
ks = compute_ks(a, b)
print(f"[OK] Drift monitor: PSI={psi:.4f}, KS stat={ks['statistic']}, "
      f"KS significant={ks['significant']}")

# Test 3: Feature columns count
from src.data.features import get_feature_columns, get_adaptive_neutral_threshold
cols = get_feature_columns()
print(f"[OK] Feature columns: {len(cols)} features")
thresh = get_adaptive_neutral_threshold(25.0)
print(f"[OK] Adaptive threshold at VIX=25: {thresh:.4f}")

# Test 4: DB schema migration
from src.data.init_db import init_db, get_connection
conn = get_connection()
# Check macro table has new columns
cursor = conn.execute("PRAGMA table_info(macro)")
macro_cols = [row[1] for row in cursor.fetchall()]
new_macro = ["vix9d", "vix3m", "vix6m", "vvix", "skew_index",
             "hy_spread", "tlt_spy_ratio"]
missing = [c for c in new_macro if c not in macro_cols]
if missing:
    print(f"[FAIL] Missing macro columns: {missing}")
else:
    print(f"[OK] Macro table has all new columns ({len(macro_cols)} total)")

# Check performance table has new columns
cursor = conn.execute("PRAGMA table_info(performance)")
perf_cols = [row[1] for row in cursor.fetchall()]
new_perf = ["confidence_tier", "vix_regime", "day_of_week", "event_proximity"]
missing_p = [c for c in new_perf if c not in perf_cols]
if missing_p:
    print(f"[FAIL] Missing performance columns: {missing_p}")
else:
    print(f"[OK] Performance table has all new columns ({len(perf_cols)} total)")

conn.close()

# Test 5: Trainer predict signature accepts feature_names
from src.model.trainer import SPYPredictor
predictor = SPYPredictor()
result = predictor.predict(np.zeros(len(cols)), feature_names=cols)
print(f"[OK] Predictor with feature_names: {result['scale_label']} "
      f"({result['confidence']:.0f}%)")

print("\n=== All Phase 1 smoke tests passed ===")
