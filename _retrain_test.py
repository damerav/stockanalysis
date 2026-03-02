#!/usr/bin/env python3
"""Test retrain with accuracy improvements."""
import sys
sys.path.insert(0, ".")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.data.init_db import get_connection, load_config
from src.data.features import build_feature_vector, get_feature_columns, get_target
from src.model.trainer import SPYPredictor
from collections import Counter
import numpy as np

config = load_config()
conn = get_connection(config)
fv = build_feature_vector(conn, config=config)
conn.close()

feature_cols = [c for c in get_feature_columns() if c in fv.columns]
target = get_target(fv)

# Show new target distribution
valid = target[target.notna()]
dist = Counter(valid)
total = len(valid)
print(f"\nTarget distribution (threshold=0.5%):")
for label, count in sorted(dist.items()):
    name = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}[label]
    print(f"  {name:8s}: {count:4d} ({count/total:.1%})")
print(f"  Baseline: {max(dist.values())/total:.1%}")

print(f"\nTotal features available: {len(feature_cols)}")
print(f"New features: return_1d, return_2d, return_3d, momentum_20d, overnight_gap, intraday_return, daily_range_pct, close_position, rsi_roc, volume_spike, vix_mean_reversion")

# Check new features exist
for f in ["return_1d", "return_2d", "return_3d", "overnight_gap", "close_position", "vix_mean_reversion"]:
    if f in fv.columns:
        nan_pct = fv[f].isna().mean()
        print(f"  {f}: {nan_pct:.1%} NaN")

predictor = SPYPredictor(config)
result = predictor.train(fv[feature_cols], target, feature_names=list(feature_cols), force_save=True)

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"  Accuracy: {result.get('accuracy', 0):.1%}")
print(f"  Test accuracy: {result.get('test_accuracy', 0):.1%}")
print(f"  Train size: {result.get('train_size')}")
print(f"  Val size: {result.get('val_size')}")
print(f"  Window: {result.get('adaptive_window')}")
print(f"  Gated: {result.get('gated')}")
print(f"  Features trained: {len(predictor.trained_feature_names) if predictor.trained_feature_names else 'N/A'}")
print(f"  Conformal: {predictor.conformal is not None}")
if predictor.conformal:
    print(f"  Conformal quantile: {predictor.conformal.quantile:.4f}")

# Quick prediction test
latest = fv[predictor.trained_feature_names].iloc[-1].values.astype(np.float64)
pred = predictor.predict(latest, feature_names=predictor.trained_feature_names)
print(f"\nPrediction: {pred.get('scale_label')} — {pred.get('confidence', 0):.0f}%")
print(f"Probabilities: {pred.get('probabilities')}")
if pred.get('prediction_set'):
    print(f"Conformal set: {pred.get('prediction_set')}")
