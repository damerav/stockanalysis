#!/usr/bin/env python3
"""Run retrain with accuracy improvements and report results."""
import sys
sys.path.insert(0, ".")

import sqlite3
import pandas as pd
import numpy as np
from src.data.features import build_feature_vector, get_feature_columns, get_target

conn = sqlite3.connect("data/spy.db")

print("Building feature vector...")
df = build_feature_vector(conn)
if df is None or df.empty:
    print("ERROR: No feature data")
    sys.exit(1)

print(f"Feature vector: {df.shape[0]} rows x {df.shape[1]} cols")

target = get_target(df, threshold=0.004, adaptive=True)
class_dist = target.value_counts().sort_index()
print(f"\nTarget distribution (±0.4% adaptive threshold):")
for cls, cnt in class_dist.items():
    label = {-1: "DOWN", 0: "NEUTRAL", 1: "UP"}[cls]
    print(f"  {label}: {cnt} ({cnt/len(target)*100:.1f}%)")

all_cols = get_feature_columns()
available = [c for c in all_cols if c in df.columns]
print(f"\nFeatures: {len(available)} available")

features_df = df[available].copy()

print("\n" + "="*60)
print("TRAINING WITH BINARY CLASSIFIERS + RECENCY WEIGHTING")
print("="*60)

from src.model.trainer import SPYPredictor
predictor = SPYPredictor()
metrics = predictor.train(features_df, target, use_gpu=True,
                          feature_names=available, force_save=True)

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"{'='*60}")
print(f"  3-class val accuracy:  {metrics.get('accuracy', 0):.1%}")
print(f"  3-class test accuracy: {metrics.get('test_accuracy', 0):.1%}")
print(f"  Binary UP accuracy:    {metrics.get('binary_up_accuracy', 'N/A')}")
print(f"  Train size:            {metrics.get('train_size', 0)}")
print(f"  Val size:              {metrics.get('val_size', 0)}")
print(f"  Test size:             {metrics.get('test_size', 0)}")
print(f"  Adaptive window:       {metrics.get('adaptive_window', 'N/A')}d")
print(f"  Window scores:         {metrics.get('window_scores', {})}")
print(f"  Gated:                 {metrics.get('gated', False)}")
print(f"  Model path:            {metrics.get('model_path', 'N/A')}")
n_feat = len(predictor.trained_feature_names) if predictor.trained_feature_names else 'N/A'
print(f"  Features after sel:    {n_feat}")

cv = metrics.get("purged_cv", {})
if cv:
    print(f"  Purged CV accuracy:    {cv.get('mean_accuracy', 0):.1%} ± {cv.get('std_accuracy', 0):.1%}")

if metrics.get("conformal_calibrated"):
    print(f"  Conformal:             calibrated")

# Test prediction with binary fusion
print(f"\n{'='*60}")
print("TESTING PREDICTION WITH BINARY FUSION")
print("="*60)
last_row = features_df.iloc[-1:].values
last_row = np.nan_to_num(last_row, nan=0.0)
# Align features if selection happened
if predictor.trained_feature_names:
    feat_idx = [available.index(f) for f in predictor.trained_feature_names if f in available]
    last_row = last_row[:, feat_idx]
pred = predictor.predict(last_row, feature_names=predictor.trained_feature_names)
print(f"  Direction:  {pred['direction']}")
print(f"  Confidence: {pred['confidence']:.1f}%")
print(f"  Probs:      {pred['probabilities']}")
if pred.get('prediction_set'):
    print(f"  Conf. set:  {pred['prediction_set']}")

conn.close()
print("\nDone.")
