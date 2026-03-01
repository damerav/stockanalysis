#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")
from src.data.init_db import get_connection
from src.data.features import build_feature_vector, get_feature_columns
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

conn = get_connection(config)
fv = build_feature_vector(conn, config=config)
conn.close()

feature_cols = [c for c in get_feature_columns() if c in fv.columns]

print(f"Total rows: {len(fv)}")
print(f"\n{'Feature':40s} {'NaN%':>6s}  {'Zero%':>6s}  {'Status'}")
print("-" * 70)
for c in feature_cols:
    nan_r = fv[c].isna().mean()
    zero_r = (fv[c] == 0).mean()
    if nan_r > 0.9:
        status = "EMPTY"
    elif nan_r > 0.5:
        status = "SPARSE"
    elif nan_r > 0.1:
        status = "PARTIAL"
    else:
        status = "OK"
    if nan_r > 0.05:
        print(f"  {c:40s} {nan_r:5.1%}  {zero_r:5.1%}  {status}")

# Summary
empty = sum(1 for c in feature_cols if fv[c].isna().mean() > 0.9)
sparse = sum(1 for c in feature_cols if 0.5 < fv[c].isna().mean() <= 0.9)
partial = sum(1 for c in feature_cols if 0.1 < fv[c].isna().mean() <= 0.5)
ok = sum(1 for c in feature_cols if fv[c].isna().mean() <= 0.1)
print(f"\nSummary: {ok} OK, {partial} partial, {sparse} sparse, {empty} empty (of {len(feature_cols)})")
