#!/usr/bin/env python3
"""Validation test for Enhancement 26 — DuckDB Migration."""
import sys, os, traceback
sys.path.insert(0, ".")

results = []
def check(name, fn):
    try:
        ok = fn()
        status = "PASS" if ok else "FAIL"
    except Exception as e:
        ok = False
        status = f"FAIL: {e}"
        traceback.print_exc()
    results.append((name, status))
    print(f"  {'✓' if ok else '✗'} {name}: {status}")
    return ok

# 1. duckdb import
def t_import():
    import duckdb
    return True
check("duckdb import", t_import)

# 2. db_router module loads
def t_router_module():
    from src.data.db_router import DbRouter, get_router, ANALYTICS_TABLES, DUCKDB_SCHEMA
    assert len(ANALYTICS_TABLES) == 5
    return True
check("db_router module loads", t_router_module)

# 3. config has analytics_path
def t_config():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "analytics_path" in cfg.get("database", {}), "analytics_path missing"
    return True
check("config.yaml analytics_path", t_config)

# 4. Create router, verify DuckDB file created
def t_router_create():
    from src.data.db_router import DbRouter, reset_router
    import yaml
    reset_router()
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    router = DbRouter(cfg)
    duck_path = cfg["database"]["analytics_path"]
    assert os.path.exists(duck_path), f"DuckDB file not created at {duck_path}"
    # Verify tables exist
    tables = router.read_analytics("SELECT table_name FROM information_schema.tables WHERE table_schema='main'")
    table_names = set(tables["table_name"].tolist()) if not tables.empty else set()
    for t in ["prices", "technicals", "macro", "intraday_bars", "options_chain"]:
        assert t in table_names, f"Table {t} missing in DuckDB"
    router.close()
    reset_router()
    return True
check("DbRouter creates DuckDB with schema", t_router_create)

# 5. Run migration
def t_migrate():
    from src.data.migrate_to_duckdb import migrate
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    ok = migrate(cfg)
    return ok
check("Migration script runs", t_migrate)

# 6. Verify row counts match
def t_row_counts():
    import sqlite3, yaml
    from src.data.db_router import DbRouter, reset_router
    reset_router()
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    sq = sqlite3.connect(cfg["database"]["path"])
    router = DbRouter(cfg)
    for table in ["prices", "technicals", "macro", "intraday_bars", "options_chain"]:
        sq_count = sq.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        dk_df = router.read_analytics(f"SELECT COUNT(*) as cnt FROM {table}")
        dk_count = int(dk_df.iloc[0]["cnt"]) if not dk_df.empty else 0
        if sq_count > 0:
            assert dk_count == sq_count, f"{table}: SQLite={sq_count} DuckDB={dk_count}"
            print(f"    {table}: {sq_count} rows ✓")
        else:
            print(f"    {table}: 0 rows (empty)")
    sq.close()
    router.close()
    reset_router()
    return True
check("Row counts match after migration", t_row_counts)

# 7. read_feature_join works
def t_feature_join():
    import yaml
    from src.data.db_router import get_router, reset_router
    reset_router()
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    router = get_router(cfg)
    df = router.read_feature_join()
    print(f"    Feature join returned {len(df)} rows, {df.shape[1]} columns")
    assert len(df) > 0, "Feature join returned empty"
    # Check key columns present
    for col in ["date", "open", "close", "vix", "sma_20"]:
        assert col in df.columns, f"Missing column: {col}"
    reset_router()
    return True
check("read_feature_join() works", t_feature_join)

# 8. build_feature_vector with DuckDB
def t_build_fv():
    import yaml, sqlite3
    from src.data.db_router import reset_router
    reset_router()
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    conn = sqlite3.connect(cfg["database"]["path"])
    from src.data.features import build_feature_vector
    fv = build_feature_vector(conn, config=cfg)
    conn.close()
    if fv is None or fv.empty:
        print("    WARNING: No feature data (expected if DB is fresh)")
        return True
    print(f"    Feature vector: {fv.shape[0]} rows, {fv.shape[1]} columns")
    assert fv.shape[1] > 80, f"Expected >80 columns, got {fv.shape[1]}"
    reset_router()
    return True
check("build_feature_vector with DuckDB", t_build_fv)

# 9. DuckDB analytics query speed
def t_speed():
    import time, yaml
    from src.data.db_router import get_router, reset_router
    reset_router()
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    router = get_router(cfg)
    start = time.time()
    df = router.read_analytics("""
        SELECT p.date, p.close, t.sma_20, t.rsi_14, m.vix
        FROM prices p
        LEFT JOIN technicals t ON p.date = t.date
        LEFT JOIN macro m ON p.date = m.date
        ORDER BY p.date
    """)
    elapsed = time.time() - start
    print(f"    Analytics JOIN query: {elapsed:.3f}s ({len(df)} rows)")
    assert elapsed < 5.0, f"Query took {elapsed:.1f}s (>5s limit)"
    reset_router()
    return True
check("Analytics query <5s", t_speed)

# 10. init_db creates DuckDB
def t_init_db():
    import yaml
    from src.data.init_db import init_db
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    init_db(cfg)
    duck_path = cfg["database"]["analytics_path"]
    assert os.path.exists(duck_path), "DuckDB not created by init_db"
    return True
check("init_db creates DuckDB", t_init_db)

# Summary
print(f"\n{'='*60}")
passed = sum(1 for _, s in results if s == "PASS")
total = len(results)
print(f"Results: {passed}/{total} passed")
for name, status in results:
    print(f"  {'✓' if status == 'PASS' else '✗'} {name}")
if passed < total:
    sys.exit(1)
