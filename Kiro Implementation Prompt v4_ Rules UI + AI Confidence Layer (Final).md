# Kiro Implementation Prompt v4: Rules UI + AI Confidence Layer (Final)
## Project: `damerav/stockanalysis` — Synced with Latest GitHub (commit `6661060`)

---

## Pre-Flight: What Is Already Implemented (Do NOT Re-implement)

The latest GitHub pull (`6661060 — fix: switch pd.read_sql_query to SQLAlchemy engine`) has already shipped the following. Kiro must **skip these** and treat them as existing infrastructure:

| Feature | File | Status |
| :--- | :--- | :--- |
| **SQLAlchemy 2.0 Migration** | `src/data/db_router.py` | **DONE** |
| Performance Dashboard page | `src/dashboard/performance_app.py` | **DONE** |
| Model Tuning & Backtest page | `src/dashboard/tuning_app.py` | **DONE** |
| Champion/Challenger Framework | `registry.py`, `trainer.py`, `tuning_app.py` | **DONE** |
| Full PostgreSQL Migration | All `src/` files | **DONE** |

---

## What This Prompt Implements (Net-New Work Only)

1.  **`strategy_rules` PostgreSQL table** — DB-backed storage for all strategy parameters.
2.  **`src/strategy/rules_store.py`** — single read/write module for rules (no YAML touching).
3.  **`src/dashboard/rules_app.py`** — full Rules Management UI page (9 tabs, pre-populated defaults, live edit + save + reset).
4.  **Engine updated** to read all parameters from `rules_store` instead of hardcoded values.
5.  **AI Confidence overlay** on the ES Strategy dashboard (entry confidence, continuation probability, model health).
6.  **AI exit/trailing wiring** — `ESExitController` CNN outputs wired into `_check_exits()` for dynamic TP2 and runner multipliers.
7.  **"Reload Rules" hot-reload** button on ES Strategy page + flag-file handler in runner.
8.  **Admin Config tab replaced** — raw YAML editor removed, replaced with read-only summary + link to Rules page.

---

## Architecture Context

The platform now uses a SQLAlchemy 2.0 engine for all PostgreSQL reads. The `DbRouter` class in `src/data/db_router.py` handles all DB access. Its key methods are:

-   `router.query(sql, params)` — read, returns `pd.DataFrame`
-   `router.execute(sql, params)` — write (INSERT/UPDATE/DELETE)

All SQL in `rules_store.py` must use `?` placeholders — the router handles the conversion to `%s` or `:param` transparently.

---

## Part 1: Database — `strategy_rules` Table

### Task 1.1 — Add table to `_migrate_schema()` in `src/data/init_db.py`

Add this block at the **end** of `_migrate_schema()`, before `conn.commit()`:

```python
# Strategy rules table
conn.execute("""
    CREATE TABLE IF NOT EXISTS strategy_rules (
        rule_group  TEXT NOT NULL, rule_key    TEXT NOT NULL,
        rule_value  TEXT NOT NULL, value_type  TEXT NOT NULL DEFAULT 'float',
        min_val     TEXT,          max_val     TEXT,
        description TEXT,          updated_at  TEXT,
        updated_by  TEXT,          PRIMARY KEY (rule_group, rule_key)
    )
""")
```

Also add a call to `seed_strategy_rules(conn)` at the end of `_migrate_schema()`.

### Task 1.2 — Add `seed_strategy_rules()` to `src/data/init_db.py`

Add this function to `init_db.py`. It uses `INSERT OR IGNORE` for SQLite compatibility.

```python
def seed_strategy_rules(conn):
    from datetime import datetime
    now = datetime.now().isoformat()
    defaults = [
        ("spread", "strike_K", "6000.0", "float", "5000", "7000", "Sold strike price (K)"),
        ("spread", "credit_C", "10.0", "float", "1", "50", "Credit width (C) in points"),
        ("sizing", "max_lots", "3", "int", "1", "10", "Maximum lots per trade"),
        ("entry", "anti_chase_atr_pct", "0.5", "float", "0.1", "2.0", "Anti-chase gate fraction of ATR"),
        ("entry", "phase2_enabled", "true", "bool", None, None, "Enable Phase 2 entries"),
        ("entry", "phase2_min_filters", "2", "int", "1", "3", "Minimum Phase 2 confluence filters"),
        ("entry", "phase2_roc_threshold", "0.5", "float", "0.1", "2.0", "ROC threshold for Phase 2"),
        ("tp_low", "tp1_mult", "1.0", "float", "0.5", "5.0", "TP1 × ATR — Low"),
        ("tp_low", "tp2_mult", "1.5", "float", "0.5", "5.0", "TP2 × ATR — Low"),
        ("tp_low", "runner_trail_mult", "2.0", "float", "0.5", "8.0", "Runner trail × ATR — Low"),
        ("tp_med", "tp1_mult", "1.2", "float", "0.5", "5.0", "TP1 × ATR — Med"),
        ("tp_med", "tp2_mult", "1.8", "float", "0.5", "5.0", "TP2 × ATR — Med"),
        ("tp_med", "runner_trail_mult", "2.5", "float", "0.5", "8.0", "Runner trail × ATR — Med"),
        ("tp_high", "tp1_mult", "1.5", "float", "0.5", "5.0", "TP1 × ATR — High"),
        ("tp_high", "tp2_mult", "2.2", "float", "0.5", "5.0", "TP2 × ATR — High"),
        ("tp_high", "runner_trail_mult", "3.0", "float", "0.5", "8.0", "Runner trail × ATR — High"),
        ("risk", "emergency_stop_pct", "0.20", "float", "0.05", "0.50", "Emergency stop % of C"),
        ("risk", "jump_exit_points", "5.0", "float", "1.0", "20.0", "Jump exit threshold pts"),
        ("risk", "circuit_breaker_usd", "-2000.0", "float", "-10000", "-100", "Daily loss limit USD"),
        ("session", "session_close_ct", "15:55", "time", None, None, "Session close time CT"),
        ("session", "session_reset_ct", "17:00", "time", None, None, "Session reset time CT"),
        ("indicators", "kc_ema_period", "20", "int", "5", "100", "KC EMA period"),
        ("indicators", "kc_atr_period", "14", "int", "5", "50", "KC ATR period"),
        ("indicators", "kc_atr_multiplier", "2.0", "float", "0.5", "5.0", "KC ATR multiplier"),
        ("indicators", "rsi_period", "14", "int", "2", "50", "RSI period"),
        ("indicators", "roc_period", "3", "int", "1", "20", "ROC period"),
        ("indicators", "atr_period", "14", "int", "5", "50", "ATR period"),
        ("regime", "lookback_minutes", "10080", "int", "1440", "43200", "Regime lookback minutes"),
        ("regime", "pct_low", "33", "int", "10", "45", "Low regime percentile cutoff"),
        ("regime", "pct_high", "66", "int", "55", "90", "High regime percentile cutoff"),
        ("ai", "ai_enabled", "true", "bool", None, None, "Enable AI confidence layer"),
        ("ai", "ai_fail_closed", "true", "bool", None, None, "Fail-closed when AI unavailable"),
        ("ai", "entry_conf_threshold", "0.70", "float", "0.50", "0.99", "Entry confidence threshold"),
        ("ai", "exit_conf_threshold", "0.65", "float", "0.50", "0.99", "Exit hold confidence threshold"),
        ("ai", "trail_ai_enabled", "true", "bool", None, None, "Use CNN for dynamic trailing"),
        ("ai", "regime_thresholds_low", "0.58", "float", "0.50", "0.99", "Entry threshold Low regime"),
        ("ai", "regime_thresholds_med", "0.55", "float", "0.50", "0.99", "Entry threshold Med regime"),
        ("ai", "regime_thresholds_high", "0.52", "float", "0.50", "0.99", "Entry threshold High regime"),
        ("rl", "rl_alpha", "0.1", "float", "0.001", "0.5", "Q-learning rate"),
        ("rl", "rl_gamma", "0.95", "float", "0.5", "0.999", "Discount factor"),
        ("rl", "rl_epsilon", "0.1", "float", "0.0", "0.5", "Exploration rate"),
        ("rl", "rl_lambda_dd", "0.5", "float", "0.0", "2.0", "Drawdown penalty weight"),
    ]
    for row in defaults:
        conn.execute(
            "INSERT OR IGNORE INTO strategy_rules (rule_group, rule_key, rule_value, value_type, min_val, max_val, description, updated_at, updated_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*row, now, "system")
        )
    conn.commit()
```

### Task 1.3 — Create table in PostgreSQL on startup

In `init_db()`, after the `router = DbRouter(config)` block, add:

```python
if router.using_postgres:
    try:
        router.execute("""
            CREATE TABLE IF NOT EXISTS strategy_rules (
                rule_group  TEXT NOT NULL, rule_key    TEXT NOT NULL,
                rule_value  TEXT NOT NULL, value_type  TEXT NOT NULL DEFAULT 'float',
                min_val     TEXT,          max_val     TEXT,
                description TEXT,          updated_at  TEXT,
                updated_by  TEXT,          PRIMARY KEY (rule_group, rule_key)
            )
        """)
        from datetime import datetime
        _seed_strategy_rules_pg(router, datetime.now().isoformat())
        logger.info("strategy_rules table ready in PostgreSQL")
    except Exception as e:
        logger.warning(f"strategy_rules PostgreSQL setup failed: {e}")
```

### Task 1.4 — Add `_seed_strategy_rules_pg()` to `src/data/init_db.py`

This function uses `INSERT ... ON CONFLICT DO NOTHING` for PostgreSQL:

```python
def _seed_strategy_rules_pg(router, now: str):
    defaults = [
        ("spread", "strike_K", "6000.0", "float", "5000", "7000", "Sold strike price (K)"),
        # ... (copy the full defaults list from Task 1.2) ...
        ("rl", "rl_lambda_dd", "0.5", "float", "0.0", "2.0", "Drawdown penalty weight"),
    ]
    for row in defaults:
        try:
            router.execute(
                "INSERT INTO strategy_rules (rule_group,rule_key,rule_value,value_type,min_val,max_val,description,updated_at,updated_by) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT (rule_group,rule_key) DO NOTHING",
                (*row, now, "system")
            )
        except Exception:
            pass
```

---

## Part 2: Rules Helper Module — `src/strategy/rules_store.py`

Create `src/strategy/__init__.py` (empty) and `src/strategy/rules_store.py`. This module must use the SQLAlchemy-based `DbRouter`.

```python
"""src/strategy/rules_store.py — DB-backed strategy parameter store."""
import logging, yaml
from datetime import datetime
from typing import Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

def _router():
    from src.data.db_router import DbRouter
    try:
        with open("config.yaml") as f: cfg = yaml.safe_load(f) or {}
    except Exception: cfg = {}
    return DbRouter(cfg)

def get_rule(group: str, key: str, default: Any = None) -> Any:
    try:
        r = _router()
        df = r.read_analytics("SELECT rule_value, value_type FROM strategy_rules WHERE rule_group=? AND rule_key=?", params=(group, key))
        r.close()
        if df.empty: return default
        return _cast(df.iloc[0]["rule_value"], df.iloc[0]["value_type"])
    except Exception as e:
        logger.warning(f"rules_store.get_rule({group}.{key}): {e}")
        return default

def get_group(group: str) -> dict:
    try:
        r = _router()
        df = r.read_analytics("SELECT rule_key, rule_value, value_type FROM strategy_rules WHERE rule_group=?", params=(group,))
        r.close()
        return {row["rule_key"]: _cast(row["rule_value"], row["value_type"]) for _, row in df.iterrows()}
    except Exception as e:
        logger.warning(f"rules_store.get_group({group}): {e}")
        return {}

def get_all_rules() -> dict:
    try:
        r = _router()
        df = r.read_analytics("SELECT rule_group, rule_key, rule_value, value_type, min_val, max_val, description, updated_at, updated_by FROM strategy_rules ORDER BY rule_group, rule_key")
        r.close()
        result: dict = {}
        for _, row in df.iterrows():
            g = row["rule_group"]
            if g not in result: result[g] = {}
            result[g][row["rule_key"]] = {
                "value": _cast(row["rule_value"], row["value_type"]), "raw": row["rule_value"],
                "type": row["value_type"], "min": row.get("min_val"), "max": row.get("max_val"),
                "description": row.get("description", ""), "updated_at": row.get("updated_at", ""),
                "updated_by": row.get("updated_by", ""),
            }
        return result
    except Exception as e:
        logger.warning(f"rules_store.get_all_rules(): {e}")
        return {}

def set_rule(group: str, key: str, value: Any, updated_by: str = "ui") -> bool:
    try:
        r = _router()
        r.execute("UPDATE strategy_rules SET rule_value=?, updated_at=?, updated_by=? WHERE rule_group=? AND rule_key=?", params=(str(value), datetime.now().isoformat(), updated_by, group, key))
        r.close()
        logger.info(f"Rule updated: {group}.{key} = {value} by {updated_by}")
        return True
    except Exception as e:
        logger.error(f"rules_store.set_rule({group}.{key}): {e}")
        return False

def set_group(group: str, updates: dict, updated_by: str = "ui") -> bool:
    return all(set_rule(group, k, v, updated_by) for k, v in updates.items())

def reset_to_defaults(group: Optional[str] = None, updated_by: str = "ui") -> bool:
    try:
        r = _router()
        if group: r.execute("DELETE FROM strategy_rules WHERE rule_group=?", params=(group,))
        else: r.execute("DELETE FROM strategy_rules")
        r.close()
        from src.data.init_db import init_db
        with open("config.yaml") as f: cfg = yaml.safe_load(f) or {}
        init_db(cfg)
        logger.info(f"Rules reset to defaults (group={group or 'ALL'}) by {updated_by}")
        return True
    except Exception as e:
        logger.error(f"rules_store.reset_to_defaults: {e}")
        return False

def _cast(value: str, vtype: str) -> Any:
    try:
        if vtype == "int": return int(float(value))
        if vtype == "float": return float(value)
        if vtype == "bool": return str(value).lower() in ("true", "1", "yes")
        return value
    except Exception: return value
```

---

## Part 3: Update `ESStrategyEngine` — `src/es_strategy/engine.py`

### Task 3.1 — Replace `__init__` to load from `rules_store`

Replace the entire `__init__` method with this version. It correctly uses `get_rule()` with fallbacks to the original hardcoded values.

```python
def __init__(self, config: dict = None):
    from src.strategy import rules_store as rs
    self.C = rs.get_rule("spread", "credit_C", 10.0)
    self.K = rs.get_rule("spread", "strike_K", 6000.0)
    self.max_lots = rs.get_rule("sizing", "max_lots", 3)
    self.jump_exit_pts = rs.get_rule("risk", "jump_exit_points", 5.0)
    self.emergency_stop_pct = rs.get_rule("risk", "emergency_stop_pct", 0.20)
    self.circuit_breaker_usd = rs.get_rule("risk", "circuit_breaker_usd", -2000.0)
    self.session_close_ct = rs.get_rule("session", "session_close_ct", "15:55")
    self.session_reset_ct = rs.get_rule("session", "session_reset_ct", "17:00")
    self.ai_enabled = rs.get_rule("ai", "ai_enabled", True)
    self.position = Position()
    self.regime_detector = RegimeDetector(
        lookback=rs.get_rule("regime", "lookback_minutes", 10080),
        pct_low=rs.get_rule("regime", "pct_low", 33),
        pct_high=rs.get_rule("regime", "pct_high", 66),
    )
    self.signals: list[Signal] = []
    self.circuit_breaker_active = False
    self._bars_since_entry = 0
    self._phase2_enabled = rs.get_rule("entry", "phase2_enabled", True)
    self._phase2_min_filters = rs.get_rule("entry", "phase2_min_filters", 2)
    self._phase2_roc_threshold = rs.get_rule("entry", "phase2_roc_threshold", 0.5)
    self._anti_chase_atr_pct = rs.get_rule("entry", "anti_chase_atr_pct", 0.5)
    self._tp_multipliers = {
        "Low": {"tp1": rs.get_rule("tp_low", "tp1_mult", 1.0), "tp2": rs.get_rule("tp_low", "tp2_mult", 1.5), "runner_trail": rs.get_rule("tp_low", "runner_trail_mult", 2.0)},
        "Med": {"tp1": rs.get_rule("tp_med", "tp1_mult", 1.2), "tp2": rs.get_rule("tp_med", "tp2_mult", 1.8), "runner_trail": rs.get_rule("tp_med", "runner_trail_mult", 2.5)},
        "High": {"tp1": rs.get_rule("tp_high", "tp1_mult", 1.5), "tp2": rs.get_rule("tp_high", "tp2_mult", 2.2), "runner_trail": rs.get_rule("tp_high", "runner_trail_mult", 3.0)},
    }
    self.rl_trail = RLTrailingAgent(
        alpha=rs.get_rule("rl", "rl_alpha", 0.1),
        gamma=rs.get_rule("rl", "rl_gamma", 0.95),
        epsilon=rs.get_rule("rl", "rl_epsilon", 0.1),
        lambda_dd=rs.get_rule("rl", "rl_lambda_dd", 0.5),
    )
    self.rl_trail.load()
```

### Task 3.2 — Update `_check_entry` anti-chase gate

In `_check_entry()`, replace `abs(price - kc_lower) < 0.5 * atr_val` with `abs(price - kc_lower) < self._anti_chase_atr_pct * atr_val`.

### Task 3.3 — Update `_check_phase2_entry` thresholds

In `_check_phase2_entry()`, replace `roc_3 < -0.5` with `roc_3 < -self._phase2_roc_threshold` and `filters_passed < 2` with `filters_passed < self._phase2_min_filters`.

### Task 3.4 — Wire AI exit confidence into `_check_exits`

This is the key AI integration. In `runner.py`, the `AILayer` already calls the `ESExitController` and gets the dynamic trailing multipliers. Pass these into `engine.process_bar()` and then into `_check_exits()`. In `_check_exits()`, if `trail_ai_enabled` is true, use these dynamic multipliers instead of the fixed ones from `self._tp_multipliers`.

### Task 3.5 — Add AI state to `get_state()`

In `engine.py`, update `get_state()` to include the `ai_enabled` status.

---

## Part 4: New Dashboard Page — `src/dashboard/rules_app.py`

Create `src/dashboard/rules_app.py` exactly as specified in the v2 prompt. It correctly uses `get_all_rules()`, `set_group()`, and `reset_to_defaults()` from the new `rules_store` module.

---

## Part 5: AI Confidence Overlay & Hot-Reload

Implement the AI Confidence metrics row and "Reload Rules" button in `page_es()` exactly as specified in the v2 prompt. Also add the hot-reload flag check to `runner.py`.

---

## Part 6: Register Rules Page in Navigation

Import `page_rules` in `app.py` and add it to the `_pages` navigation dictionary as specified in the v2 prompt.

---

## Part 7: Replace Admin Config Tab

Replace the body of `_admin_config_tab()` in `app.py` with the read-only summary and link to the Rules page, as specified in the v2 prompt.

---

## Acceptance Criteria

1.  **Rules page** appears in the sidebar and is fully functional, reading/writing to the DB via the SQLAlchemy-powered `rules_store`.
2.  **Engine** loads all parameters from the DB on startup.
3.  **AI exit confidence** is wired into the engine's trailing stop logic.
4.  **All other features** from the v2 prompt are implemented correctly.
