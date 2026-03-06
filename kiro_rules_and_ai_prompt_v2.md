# Kiro Implementation Prompt v2: Rules Management UI + AI Confidence Layer
## Project: `damerav/stockanalysis` — Updated for Latest GitHub State (commit `d60eab5`)

---

## Pre-Flight: What Is Already Implemented (Do NOT Re-implement)

The latest pull (`d60eab5 — feat: eliminate SQLite — full PostgreSQL migration`) has already shipped the following. Kiro must **skip these** and treat them as existing infrastructure:

| Feature | File | Status |
| :--- | :--- | :--- |
| Performance Dashboard page | `src/dashboard/performance_app.py` | **DONE** |
| Model Tuning & Backtest page | `src/dashboard/tuning_app.py` | **DONE** |
| `promote_model()` in registry | `src/model/registry.py` | **DONE** |
| Champion model loading in pipeline | `src/model/trainer.py` `load_latest_model()` | **DONE** |
| "Promote to Champion" button in tuning UI | `src/dashboard/tuning_app.py` | **DONE** |
| Both pages registered in `app.py` navigation | `src/dashboard/app.py` lines 38–39, 3173–3175 | **DONE** |
| Full PostgreSQL migration (all modules) | All `src/` files | **DONE** |

---

## What This Prompt Implements (Net-New Work Only)

1. **`strategy_rules` PostgreSQL table** — DB-backed storage for all strategy parameters
2. **`src/strategy/rules_store.py`** — single read/write module for rules (no YAML touching)
3. **`src/dashboard/rules_app.py`** — full Rules Management UI page (9 tabs, pre-populated defaults, live edit + save + reset)
4. **Engine updated** to read all parameters from `rules_store` instead of `config.yaml`
5. **AI Confidence overlay** on the ES Strategy dashboard (entry confidence, continuation probability, model health)
6. **AI exit/trailing wiring** — `ESExitController` CNN outputs wired into `_check_exits()` for dynamic TP2 and runner multipliers
7. **"Reload Rules" hot-reload** button on ES Strategy page + flag-file handler in runner
8. **Admin Config tab replaced** — raw YAML editor removed, replaced with read-only summary + link to Rules page

---

## Architecture Context

The platform runs on a DGX Spark server with **PostgreSQL as the primary database** (SQLite is a fallback only). The `DbRouter` class in `src/data/db_router.py` handles all DB access. Its key methods are:

- `router.query(sql, params)` — read, returns `pd.DataFrame`
- `router.read_analytics(sql, params)` — alias for `query()`
- `router.execute(sql, params)` — write (INSERT/UPDATE/DELETE)
- `router.write_analytics(sql, params)` — alias for `execute()`

The `_sqlite_sql_to_pg()` helper inside `db_router.py` auto-converts `?` placeholders to `%s` for PostgreSQL, so all SQL in `rules_store.py` must use `?` placeholders — the router handles the conversion transparently.

The existing `ai_enabled` toggle on the ES Strategy page currently writes directly to `config.yaml`. After this implementation, it must be updated to call `rules_store.set_rule("ai", "ai_enabled", ...)` instead.

---

## Part 1: Database — `strategy_rules` Table

### Task 1.1 — Add table to `_migrate_schema()` in `src/data/init_db.py`

The correct place to add new tables is inside `_migrate_schema(conn)`, not in `init_db()` directly. Add the following block at the **end** of `_migrate_schema()`, before `conn.commit()`:

```python
# Strategy rules table — DB-backed parameter store for the ES engine
conn.execute("""
    CREATE TABLE IF NOT EXISTS strategy_rules (
        rule_group  TEXT NOT NULL,
        rule_key    TEXT NOT NULL,
        rule_value  TEXT NOT NULL,
        value_type  TEXT NOT NULL DEFAULT 'float',
        min_val     TEXT,
        max_val     TEXT,
        description TEXT,
        updated_at  TEXT,
        updated_by  TEXT,
        PRIMARY KEY (rule_group, rule_key)
    )
""")
```

Also add a call to `seed_strategy_rules(conn)` at the end of `_migrate_schema()` (before `conn.commit()`), so defaults are populated on every `init_db()` call with `INSERT OR IGNORE` safety.

### Task 1.2 — Add `seed_strategy_rules()` to `src/data/init_db.py`

Add this function to `init_db.py`. It uses `INSERT OR IGNORE` so user-saved values are never overwritten on restart:

```python
def seed_strategy_rules(conn):
    from datetime import datetime
    now = datetime.now().isoformat()
    defaults = [
        # Spread
        ("spread",    "strike_K",              "6000.0",  "float",  "5000",   "7000",   "Sold strike price (K)"),
        ("spread",    "credit_C",              "10.0",    "float",  "1",      "50",     "Credit width (C) in points"),
        # Sizing
        ("sizing",    "max_lots",              "3",       "int",    "1",      "10",     "Maximum lots per trade"),
        # Entry
        ("entry",     "anti_chase_atr_pct",    "0.5",     "float",  "0.1",    "2.0",    "Anti-chase gate: max distance from KC band as fraction of ATR"),
        ("entry",     "phase2_enabled",        "true",    "bool",   None,     None,     "Enable Phase 2 confluence reload entries"),
        ("entry",     "phase2_min_filters",    "2",       "int",    "1",      "3",      "Minimum confluence filters for Phase 2 (out of 3)"),
        ("entry",     "phase2_roc_threshold",  "0.5",     "float",  "0.1",    "2.0",    "ROC momentum threshold for Phase 2 filter 1"),
        # TP multipliers by regime
        ("tp_low",    "tp1_mult",              "1.0",     "float",  "0.5",    "5.0",    "TP1 multiplier × ATR — Low regime"),
        ("tp_low",    "tp2_mult",              "1.5",     "float",  "0.5",    "5.0",    "TP2 multiplier × ATR — Low regime"),
        ("tp_low",    "runner_trail_mult",     "2.0",     "float",  "0.5",    "8.0",    "Runner trail multiplier × ATR — Low regime"),
        ("tp_med",    "tp1_mult",              "1.2",     "float",  "0.5",    "5.0",    "TP1 multiplier × ATR — Med regime"),
        ("tp_med",    "tp2_mult",              "1.8",     "float",  "0.5",    "5.0",    "TP2 multiplier × ATR — Med regime"),
        ("tp_med",    "runner_trail_mult",     "2.5",     "float",  "0.5",    "8.0",    "Runner trail multiplier × ATR — Med regime"),
        ("tp_high",   "tp1_mult",              "1.5",     "float",  "0.5",    "5.0",    "TP1 multiplier × ATR — High regime"),
        ("tp_high",   "tp2_mult",              "2.2",     "float",  "0.5",    "5.0",    "TP2 multiplier × ATR — High regime"),
        ("tp_high",   "runner_trail_mult",     "3.0",     "float",  "0.5",    "8.0",    "Runner trail multiplier × ATR — High regime"),
        # Risk
        ("risk",      "emergency_stop_pct",    "0.20",    "float",  "0.05",   "0.50",   "Emergency stop as fraction of credit C"),
        ("risk",      "jump_exit_points",      "5.0",     "float",  "1.0",    "20.0",   "Adverse move in points within first bar triggering immediate exit"),
        ("risk",      "circuit_breaker_usd",   "-2000.0", "float",  "-10000", "-100",   "Daily P&L loss limit in USD"),
        # Session
        ("session",   "session_close_ct",      "15:55",   "time",   None,     None,     "Flatten all positions before this time (CT)"),
        ("session",   "session_reset_ct",      "17:00",   "time",   None,     None,     "Reset daily counters at this time (CT)"),
        # Indicators
        ("indicators","kc_ema_period",         "20",      "int",    "5",      "100",    "Keltner Channel EMA period"),
        ("indicators","kc_atr_period",         "14",      "int",    "5",      "50",     "Keltner Channel ATR period"),
        ("indicators","kc_atr_multiplier",     "2.0",     "float",  "0.5",    "5.0",    "Keltner Channel ATR multiplier"),
        ("indicators","rsi_period",            "14",      "int",    "2",      "50",     "RSI period"),
        ("indicators","roc_period",            "3",       "int",    "1",      "20",     "Rate of Change period"),
        ("indicators","atr_period",            "14",      "int",    "5",      "50",     "ATR period"),
        # Regime
        ("regime",    "lookback_minutes",      "10080",   "int",    "1440",   "43200",  "ATR history lookback in minutes (10080 = 1 week)"),
        ("regime",    "pct_low",               "33",      "int",    "10",     "45",     "ATR percentile below which regime is Low"),
        ("regime",    "pct_high",              "66",      "int",    "55",     "90",     "ATR percentile above which regime is High"),
        # AI layer
        ("ai",        "ai_enabled",            "true",    "bool",   None,     None,     "Master switch: enable AI confidence layer"),
        ("ai",        "ai_fail_closed",        "true",    "bool",   None,     None,     "Fail-closed: block trades if AI model unavailable"),
        ("ai",        "entry_conf_threshold",  "0.70",    "float",  "0.50",   "0.99",   "Minimum AI entry confidence to allow a trade"),
        ("ai",        "exit_conf_threshold",   "0.65",    "float",  "0.50",   "0.99",   "AI continuation probability below which engine may exit"),
        ("ai",        "trail_ai_enabled",      "true",    "bool",   None,     None,     "Use CNN exit controller for dynamic trailing multipliers"),
        ("ai",        "regime_thresholds_low", "0.58",    "float",  "0.50",   "0.99",   "Entry gate threshold — Low regime"),
        ("ai",        "regime_thresholds_med", "0.55",    "float",  "0.50",   "0.99",   "Entry gate threshold — Med regime"),
        ("ai",        "regime_thresholds_high","0.52",    "float",  "0.50",   "0.99",   "Entry gate threshold — High regime"),
        # RL trailing agent
        ("rl",        "rl_alpha",              "0.1",     "float",  "0.001",  "0.5",    "Q-learning rate"),
        ("rl",        "rl_gamma",              "0.95",    "float",  "0.5",    "0.999",  "Discount factor"),
        ("rl",        "rl_epsilon",            "0.1",     "float",  "0.0",    "0.5",    "Exploration rate (epsilon-greedy)"),
        ("rl",        "rl_lambda_dd",          "0.5",     "float",  "0.0",    "2.0",    "Drawdown penalty weight"),
    ]
    for row in defaults:
        conn.execute(
            "INSERT OR IGNORE INTO strategy_rules "
            "(rule_group, rule_key, rule_value, value_type, min_val, max_val, description, updated_at, updated_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*row, now, "system")
        )
    conn.commit()
```

### Task 1.3 — Create the table in PostgreSQL on startup

In `init_db()`, after the existing `router = DbRouter(config)` block, add:

```python
if router.using_postgres:
    try:
        router.execute("""
            CREATE TABLE IF NOT EXISTS strategy_rules (
                rule_group  TEXT NOT NULL,
                rule_key    TEXT NOT NULL,
                rule_value  TEXT NOT NULL,
                value_type  TEXT NOT NULL DEFAULT 'float',
                min_val     TEXT,
                max_val     TEXT,
                description TEXT,
                updated_at  TEXT,
                updated_by  TEXT,
                PRIMARY KEY (rule_group, rule_key)
            )
        """)
        # Seed defaults into PostgreSQL using INSERT ... ON CONFLICT DO NOTHING
        from datetime import datetime
        now = datetime.now().isoformat()
        # Re-use the same defaults list from seed_strategy_rules()
        # Call a pg-compatible seed function (see Task 1.4)
        _seed_strategy_rules_pg(router, now)
        logger.info("strategy_rules table ready in PostgreSQL")
    except Exception as e:
        logger.warning(f"strategy_rules PostgreSQL setup failed: {e}")
```

### Task 1.4 — Add `_seed_strategy_rules_pg()` to `src/data/init_db.py`

This is the PostgreSQL-compatible version of the seed function. It uses `INSERT ... ON CONFLICT DO NOTHING` instead of `INSERT OR IGNORE`:

```python
def _seed_strategy_rules_pg(router, now: str):
    """Seed strategy_rules into PostgreSQL. Uses ON CONFLICT DO NOTHING."""
    from src.data.init_db import seed_strategy_rules as _get_defaults
    # We reuse the same defaults list by calling a helper
    # Since seed_strategy_rules() takes a sqlite conn, we replicate the defaults here
    # using the router's execute() method with %s placeholders
    defaults = [
        ("spread","strike_K","6000.0","float","5000","7000","Sold strike price (K)"),
        ("spread","credit_C","10.0","float","1","50","Credit width (C) in points"),
        ("sizing","max_lots","3","int","1","10","Maximum lots per trade"),
        ("entry","anti_chase_atr_pct","0.5","float","0.1","2.0","Anti-chase gate fraction of ATR"),
        ("entry","phase2_enabled","true","bool",None,None,"Enable Phase 2 entries"),
        ("entry","phase2_min_filters","2","int","1","3","Minimum Phase 2 confluence filters"),
        ("entry","phase2_roc_threshold","0.5","float","0.1","2.0","ROC threshold for Phase 2"),
        ("tp_low","tp1_mult","1.0","float","0.5","5.0","TP1 × ATR — Low"),
        ("tp_low","tp2_mult","1.5","float","0.5","5.0","TP2 × ATR — Low"),
        ("tp_low","runner_trail_mult","2.0","float","0.5","8.0","Runner trail × ATR — Low"),
        ("tp_med","tp1_mult","1.2","float","0.5","5.0","TP1 × ATR — Med"),
        ("tp_med","tp2_mult","1.8","float","0.5","5.0","TP2 × ATR — Med"),
        ("tp_med","runner_trail_mult","2.5","float","0.5","8.0","Runner trail × ATR — Med"),
        ("tp_high","tp1_mult","1.5","float","0.5","5.0","TP1 × ATR — High"),
        ("tp_high","tp2_mult","2.2","float","0.5","5.0","TP2 × ATR — High"),
        ("tp_high","runner_trail_mult","3.0","float","0.5","8.0","Runner trail × ATR — High"),
        ("risk","emergency_stop_pct","0.20","float","0.05","0.50","Emergency stop % of C"),
        ("risk","jump_exit_points","5.0","float","1.0","20.0","Jump exit threshold pts"),
        ("risk","circuit_breaker_usd","-2000.0","float","-10000","-100","Daily loss limit USD"),
        ("session","session_close_ct","15:55","time",None,None,"Session close time CT"),
        ("session","session_reset_ct","17:00","time",None,None,"Session reset time CT"),
        ("indicators","kc_ema_period","20","int","5","100","KC EMA period"),
        ("indicators","kc_atr_period","14","int","5","50","KC ATR period"),
        ("indicators","kc_atr_multiplier","2.0","float","0.5","5.0","KC ATR multiplier"),
        ("indicators","rsi_period","14","int","2","50","RSI period"),
        ("indicators","roc_period","3","int","1","20","ROC period"),
        ("indicators","atr_period","14","int","5","50","ATR period"),
        ("regime","lookback_minutes","10080","int","1440","43200","Regime lookback minutes"),
        ("regime","pct_low","33","int","10","45","Low regime percentile cutoff"),
        ("regime","pct_high","66","int","55","90","High regime percentile cutoff"),
        ("ai","ai_enabled","true","bool",None,None,"Enable AI confidence layer"),
        ("ai","ai_fail_closed","true","bool",None,None,"Fail-closed when AI unavailable"),
        ("ai","entry_conf_threshold","0.70","float","0.50","0.99","Entry confidence threshold"),
        ("ai","exit_conf_threshold","0.65","float","0.50","0.99","Exit hold confidence threshold"),
        ("ai","trail_ai_enabled","true","bool",None,None,"Use CNN for dynamic trailing"),
        ("ai","regime_thresholds_low","0.58","float","0.50","0.99","Entry threshold Low regime"),
        ("ai","regime_thresholds_med","0.55","float","0.50","0.99","Entry threshold Med regime"),
        ("ai","regime_thresholds_high","0.52","float","0.50","0.99","Entry threshold High regime"),
        ("rl","rl_alpha","0.1","float","0.001","0.5","Q-learning rate"),
        ("rl","rl_gamma","0.95","float","0.5","0.999","Discount factor"),
        ("rl","rl_epsilon","0.1","float","0.0","0.5","Exploration rate"),
        ("rl","rl_lambda_dd","0.5","float","0.0","2.0","Drawdown penalty weight"),
    ]
    for row in defaults:
        try:
            router.execute(
                "INSERT INTO strategy_rules "
                "(rule_group,rule_key,rule_value,value_type,min_val,max_val,description,updated_at,updated_by) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT (rule_group,rule_key) DO NOTHING",
                (*row, now, "system")
            )
        except Exception:
            pass
```

---

## Part 2: Rules Helper Module — `src/strategy/rules_store.py`

Create `src/strategy/__init__.py` (empty) and `src/strategy/rules_store.py`:

```python
"""src/strategy/rules_store.py — DB-backed strategy parameter store."""
import logging
import yaml
from datetime import datetime
from typing import Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

def _router():
    from src.data.db_router import DbRouter
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return DbRouter(cfg)

def get_rule(group: str, key: str, default: Any = None) -> Any:
    try:
        r = _router()
        df = r.read_analytics(
            "SELECT rule_value, value_type FROM strategy_rules WHERE rule_group=? AND rule_key=?",
            params=(group, key)
        )
        r.close()
        if df.empty:
            return default
        return _cast(df.iloc[0]["rule_value"], df.iloc[0]["value_type"])
    except Exception as e:
        logger.warning(f"rules_store.get_rule({group}.{key}): {e}")
        return default

def get_group(group: str) -> dict:
    try:
        r = _router()
        df = r.read_analytics(
            "SELECT rule_key, rule_value, value_type FROM strategy_rules WHERE rule_group=?",
            params=(group,)
        )
        r.close()
        return {row["rule_key"]: _cast(row["rule_value"], row["value_type"])
                for _, row in df.iterrows()}
    except Exception as e:
        logger.warning(f"rules_store.get_group({group}): {e}")
        return {}

def get_all_rules() -> dict:
    try:
        r = _router()
        df = r.read_analytics(
            "SELECT rule_group, rule_key, rule_value, value_type, "
            "min_val, max_val, description, updated_at, updated_by "
            "FROM strategy_rules ORDER BY rule_group, rule_key"
        )
        r.close()
        result: dict = {}
        for _, row in df.iterrows():
            g = row["rule_group"]
            if g not in result:
                result[g] = {}
            result[g][row["rule_key"]] = {
                "value":       _cast(row["rule_value"], row["value_type"]),
                "raw":         row["rule_value"],
                "type":        row["value_type"],
                "min":         row.get("min_val"),
                "max":         row.get("max_val"),
                "description": row.get("description", ""),
                "updated_at":  row.get("updated_at", ""),
                "updated_by":  row.get("updated_by", ""),
            }
        return result
    except Exception as e:
        logger.warning(f"rules_store.get_all_rules(): {e}")
        return {}

def set_rule(group: str, key: str, value: Any, updated_by: str = "ui") -> bool:
    try:
        r = _router()
        r.execute(
            "UPDATE strategy_rules SET rule_value=?, updated_at=?, updated_by=? "
            "WHERE rule_group=? AND rule_key=?",
            params=(str(value), datetime.now().isoformat(), updated_by, group, key)
        )
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
        if group:
            r.execute("DELETE FROM strategy_rules WHERE rule_group=?", params=(group,))
        else:
            r.execute("DELETE FROM strategy_rules")
        r.close()
        # Re-seed via init_db
        from src.data.init_db import init_db
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        init_db(cfg)
        logger.info(f"Rules reset to defaults (group={group or 'ALL'}) by {updated_by}")
        return True
    except Exception as e:
        logger.error(f"rules_store.reset_to_defaults: {e}")
        return False

def _cast(value: str, vtype: str) -> Any:
    try:
        if vtype == "int":    return int(float(value))
        if vtype == "float":  return float(value)
        if vtype == "bool":   return str(value).lower() in ("true", "1", "yes")
        return value
    except Exception:
        return value
```

---

## Part 3: Update `ESStrategyEngine` — `src/es_strategy/engine.py`

### Task 3.1 — Replace `__init__` to load from `rules_store`

Replace the entire `__init__` body with the following. The `cfg.get(...)` fallbacks ensure the engine still works if the DB is unavailable:

```python
def __init__(self, config: dict = None):
    from src.strategy.rules_store import get_rule
    cfg = (config or {}).get("es_strategy", {})

    self.C                   = get_rule("spread",     "credit_C",              cfg.get("credit_C", 10.0))
    self.K                   = get_rule("spread",     "strike_K",              cfg.get("strike_K", 6000.0))
    self.max_lots            = get_rule("sizing",     "max_lots",              cfg.get("max_lots", 3))
    self.jump_exit_pts       = get_rule("risk",       "jump_exit_points",      cfg.get("jump_exit_points", 5.0))
    self.emergency_stop_pct  = get_rule("risk",       "emergency_stop_pct",    cfg.get("emergency_stop_pct", 0.20))
    self.circuit_breaker_usd = get_rule("risk",       "circuit_breaker_usd",   cfg.get("circuit_breaker_usd", -2000.0))
    self.session_close_ct    = get_rule("session",    "session_close_ct",      cfg.get("session_close_ct", "15:55"))
    self.session_reset_ct    = get_rule("session",    "session_reset_ct",      cfg.get("session_reset_ct", "17:00"))
    self.ai_enabled          = get_rule("ai",         "ai_enabled",            cfg.get("ai_enabled", True))
    self._phase2_enabled     = get_rule("entry",      "phase2_enabled",        True)
    self._anti_chase_pct     = get_rule("entry",      "anti_chase_atr_pct",    0.5)
    self._phase2_min_filters = get_rule("entry",      "phase2_min_filters",    2)
    self._phase2_roc_thresh  = get_rule("entry",      "phase2_roc_threshold",  0.5)
    self._exit_conf_threshold= get_rule("ai",         "exit_conf_threshold",   0.65)
    self._trail_ai_enabled   = get_rule("ai",         "trail_ai_enabled",      True)

    self._tp_multipliers = {
        "Low":  {"tp1": get_rule("tp_low",  "tp1_mult", 1.0),
                 "tp2": get_rule("tp_low",  "tp2_mult", 1.5),
                 "runner_trail": get_rule("tp_low",  "runner_trail_mult", 2.0)},
        "Med":  {"tp1": get_rule("tp_med",  "tp1_mult", 1.2),
                 "tp2": get_rule("tp_med",  "tp2_mult", 1.8),
                 "runner_trail": get_rule("tp_med",  "runner_trail_mult", 2.5)},
        "High": {"tp1": get_rule("tp_high", "tp1_mult", 1.5),
                 "tp2": get_rule("tp_high", "tp2_mult", 2.2),
                 "runner_trail": get_rule("tp_high", "runner_trail_mult", 3.0)},
    }

    self.position = Position()
    self.regime_detector = RegimeDetector(
        lookback=get_rule("regime", "lookback_minutes", 10080),
        pct_low= get_rule("regime", "pct_low",          33),
        pct_high=get_rule("regime", "pct_high",         66),
    )
    self.signals: list[Signal] = []
    self.circuit_breaker_active = False
    self._bars_since_entry = 0
    self._phase2_enabled = True
    self._last_ai_entry_conf = None
    self._last_ai_exit_conf  = None

    self.rl_trail = RLTrailingAgent(
        alpha=    get_rule("rl", "rl_alpha",     0.1),
        gamma=    get_rule("rl", "rl_gamma",     0.95),
        epsilon=  get_rule("rl", "rl_epsilon",   0.1),
        lambda_dd=get_rule("rl", "rl_lambda_dd", 0.5),
    )
    self.rl_trail.load()

def reload_rules(self):
    """Hot-reload all parameters from DB without full restart."""
    self.__init__()
    logger.info("Strategy rules hot-reloaded")
```

### Task 3.2 — Update `_check_entry` anti-chase gate

In `_check_entry`, replace the hardcoded `0.5 * atr_val` distance check with `self._anti_chase_pct * atr_val` for both LONG and SHORT sides.

### Task 3.3 — Update `_check_phase2_entry` thresholds

Replace `roc_3 < -0.5` with `roc_3 < -self._phase2_roc_thresh`, `roc_3 > 0.5` with `roc_3 > self._phase2_roc_thresh`, and `filters_passed < 2` with `filters_passed < self._phase2_min_filters`.

### Task 3.4 — Wire AI exit confidence into `_check_exits`

At the top of `_check_exits`, after computing `mults`, add the AI trail multiplier override block:

```python
ai_trail_mults = mults.copy()
if self._trail_ai_enabled:
    try:
        from src.es_strategy.ai_models import ESExitController
        import numpy as np
        exit_ctrl = ESExitController()
        if exit_ctrl.load():
            bar_window = getattr(self, "_ai_bar_window", None)
            if bar_window is not None and len(bar_window) >= 20:
                result = exit_ctrl.predict(np.array(bar_window[-20:]), regime)
                p_cont = result.get("p_cont_5", 0.5)
                self._last_ai_exit_conf = round(p_cont, 4)
                if p_cont != 0.5:  # model actually ran
                    ai_trail_mults["tp2"]          = result.get("tp2_trail",    mults["tp2"])
                    ai_trail_mults["runner_trail"]  = result.get("runner_trail", mults["runner_trail"])
    except Exception as e:
        logger.warning(f"AI exit controller skipped: {e}")
```

Then replace `mults["tp2"]` with `ai_trail_mults["tp2"]` and `mults["runner_trail"]` with `ai_trail_mults["runner_trail"]` in the TP2 and runner trailing stop calculations below.

### Task 3.5 — Add AI state to `get_state()`

Add to the returned dict in `get_state()`:

```python
"ai_confidence": {
    "entry_conf":      self._last_ai_entry_conf,
    "exit_conf":       self._last_ai_exit_conf,
    "trail_ai_active": self._trail_ai_enabled,
    "ai_enabled":      self.ai_enabled,
},
```

### Task 3.6 — Update the `ai_enabled` toggle in `page_es()` to use `rules_store`

In `app.py`, find the existing `ai_enabled` toggle block (around line 2066) that writes directly to `config.yaml`. Replace the `config.yaml` write logic with:

```python
# BEFORE (remove this):
with open("config.yaml") as f:
    raw = f.read()
if ai_enabled:
    raw = raw.replace("ai_enabled: true", "ai_enabled: false")
else:
    raw = raw.replace("ai_enabled: false", "ai_enabled: true")
with open("config.yaml", "w") as f:
    f.write(raw)

# AFTER (replace with):
from src.strategy.rules_store import set_rule
set_rule("ai", "ai_enabled", str(new_ai).lower(),
         updated_by=st.session_state.get("username", "ui"))
```

---

## Part 4: New Dashboard Page — `src/dashboard/rules_app.py`

Create `src/dashboard/rules_app.py`. This page has 9 tabs, one per rule group. All values are loaded from `get_all_rules()` and saved via `set_group()`. No YAML is read or written.

```python
"""src/dashboard/rules_app.py — Strategy Rules Management UI."""
import streamlit as st
from src.strategy.rules_store import get_all_rules, set_group, reset_to_defaults
from src.dashboard.theme import apply_theme, page_header

def page_rules():
    apply_theme()
    st.markdown(page_header("⚙️ Strategy Rules"), unsafe_allow_html=True)
    st.caption(
        "All parameters are pre-populated with factory defaults from the engine. "
        "Changes are saved to the database and take effect on the next bar or after "
        "clicking **Reload Rules** on the ES Strategy page."
    )

    rules = get_all_rules()
    if not rules:
        st.error("Could not load rules. Ensure the database is initialized and reachable.")
        if st.button("Initialize Database"):
            from src.data.init_db import init_db
            import yaml
            with open("config.yaml") as f:
                cfg = yaml.safe_load(f) or {}
            init_db(cfg)
            st.rerun()
        return

    user = st.session_state.get("username", "ui")

    tabs = st.tabs([
        "📐 Spread & Sizing", "🟢 Entry", "🎯 Take-Profit",
        "🛡️ Risk & Stops", "🕐 Session", "📊 Indicators",
        "🌡️ Regime", "🤖 AI Layer", "🧠 RL Trailing",
    ])

    # ── Tab 1: Spread & Sizing ──────────────────────────────────────────
    with tabs[0]:
        st.subheader("📐 Spread Parameters")
        spread = rules.get("spread", {})
        sizing = rules.get("sizing", {})

        c1, c2 = st.columns(2)
        with c1:
            new_K = st.number_input("Strike K", min_value=5000.0, max_value=7000.0,
                value=float(spread.get("strike_K", {}).get("value", 6000.0)),
                step=50.0, key="r_strike_K",
                help=spread.get("strike_K", {}).get("description", ""))
            st.caption(f"Default: 6000.0  |  Last saved: {spread.get('strike_K', {}).get('updated_at', 'never')[:16]}")
        with c2:
            new_C = st.number_input("Credit C (pts)", min_value=1.0, max_value=50.0,
                value=float(spread.get("credit_C", {}).get("value", 10.0)),
                step=0.5, key="r_credit_C",
                help=spread.get("credit_C", {}).get("description", ""))
            st.caption(f"Default: 10.0  |  Last saved: {spread.get('credit_C', {}).get('updated_at', 'never')[:16]}")

        st.subheader("📦 Position Sizing")
        new_lots = st.slider("Max Lots Per Trade", 1, 10,
            value=int(sizing.get("max_lots", {}).get("value", 3)), key="r_max_lots",
            help=sizing.get("max_lots", {}).get("description", ""))
        st.caption("Default: 3. The AI entry gate may reduce this to 1–3 based on confidence.")

        col_save, col_reset = st.columns([1, 1])
        with col_save:
            if st.button("💾 Save", key="save_spread"):
                ok = set_group("spread", {"strike_K": new_K, "credit_C": new_C}, user)
                ok2 = set_group("sizing", {"max_lots": new_lots}, user)
                st.success("Saved.") if (ok and ok2) else st.error("Save failed.")
        with col_reset:
            if st.button("🔄 Reset Defaults", key="reset_spread"):
                reset_to_defaults("spread", user); reset_to_defaults("sizing", user)
                st.rerun()

    # ── Tab 2: Entry Rules ──────────────────────────────────────────────
    with tabs[1]:
        st.subheader("🟢 Entry Rules")
        entry = rules.get("entry", {})

        st.markdown("**Phase 1 — Anti-Chase Gate**")
        new_anti = st.slider("Anti-Chase Gate (× ATR)", 0.1, 2.0,
            value=float(entry.get("anti_chase_atr_pct", {}).get("value", 0.5)),
            step=0.05, key="r_anti_chase",
            help="Price must be within N × ATR of the KC band to qualify as an entry.")
        st.caption("Default: 0.5 × ATR. Tighter = fewer, higher-quality entries.")

        st.divider()
        st.markdown("**Phase 2 — Confluence Reload**")
        new_p2 = st.toggle("Enable Phase 2 Entries",
            value=bool(entry.get("phase2_enabled", {}).get("value", True)), key="r_p2_en")
        c1, c2 = st.columns(2)
        with c1:
            new_p2_f = st.selectbox("Min Confluence Filters (of 3)", [1, 2, 3],
                index=[1,2,3].index(int(entry.get("phase2_min_filters", {}).get("value", 2))),
                key="r_p2_filters")
        with c2:
            new_p2_roc = st.number_input("ROC Threshold", 0.1, 2.0,
                value=float(entry.get("phase2_roc_threshold", {}).get("value", 0.5)),
                step=0.05, key="r_p2_roc")

        st.markdown("""
| Filter | Condition |
|--------|-----------|
| 1 — Momentum | ROC confirms direction (above threshold) |
| 2 — Regime | Volatility is not High |
| 3 — VWAP | Price is on the correct side of VWAP |
        """)

        if st.button("💾 Save Entry Rules", key="save_entry"):
            ok = set_group("entry", {
                "anti_chase_atr_pct": new_anti,
                "phase2_enabled": str(new_p2).lower(),
                "phase2_min_filters": new_p2_f,
                "phase2_roc_threshold": new_p2_roc,
            }, user)
            st.success("Saved.") if ok else st.error("Save failed.")
        if st.button("🔄 Reset", key="reset_entry"):
            reset_to_defaults("entry", user); st.rerun()

    # ── Tab 3: Take-Profit ──────────────────────────────────────────────
    with tabs[2]:
        st.subheader("🎯 Take-Profit Multipliers (× ATR)")
        st.info(
            "Targets are calculated as **Multiplier × ATR** from entry price. "
            "When AI trailing is enabled, TP2 and Runner multipliers are dynamically "
            "adjusted by the CNN exit controller."
        )
        regime_tabs = st.tabs(["🟢 Low Volatility", "🟡 Medium Volatility", "🔴 High Volatility"])
        for rt, (grp, defs) in zip(regime_tabs, [
            ("tp_low",  {"tp1": 1.0, "tp2": 1.5, "runner": 2.0}),
            ("tp_med",  {"tp1": 1.2, "tp2": 1.8, "runner": 2.5}),
            ("tp_high", {"tp1": 1.5, "tp2": 2.2, "runner": 3.0}),
        ]):
            with rt:
                tp = rules.get(grp, {})
                c1, c2, c3 = st.columns(3)
                with c1:
                    v1 = st.number_input("TP1 × ATR", 0.5, 5.0,
                        value=float(tp.get("tp1_mult", {}).get("value", defs["tp1"])),
                        step=0.1, key=f"r_{grp}_tp1",
                        help=f"First lot exits at entry ± TP1 × ATR. Default: {defs['tp1']}")
                with c2:
                    v2 = st.number_input("TP2 × ATR", 0.5, 5.0,
                        value=float(tp.get("tp2_mult", {}).get("value", defs["tp2"])),
                        step=0.1, key=f"r_{grp}_tp2",
                        help=f"Second lot exits at entry ± TP2 × ATR. Default: {defs['tp2']}")
                with c3:
                    v3 = st.number_input("Runner Trail × ATR", 0.5, 8.0,
                        value=float(tp.get("runner_trail_mult", {}).get("value", defs["runner"])),
                        step=0.1, key=f"r_{grp}_runner",
                        help=f"Runner trailing stop distance × ATR. Default: {defs['runner']}")
                st.caption("TP1 → ratchets stop to breakeven. TP2 → ratchets to TP1. Runner → RL-adjusted trail.")
                if st.button(f"💾 Save", key=f"save_{grp}"):
                    ok = set_group(grp, {"tp1_mult": v1, "tp2_mult": v2, "runner_trail_mult": v3}, user)
                    st.success("Saved.") if ok else st.error("Save failed.")

    # ── Tab 4: Risk & Stops ─────────────────────────────────────────────
    with tabs[3]:
        st.subheader("🛡️ Stop & Risk Rules")
        risk = rules.get("risk", {})
        c1, c2 = st.columns(2)
        with c1:
            new_stop = st.slider("Emergency Stop (% of Credit C)", 5, 50,
                value=int(float(risk.get("emergency_stop_pct", {}).get("value", 0.20)) * 100),
                format="%d%%", key="r_stop_pct",
                help="Stop placed at Entry ± (stop_pct × C). Default: 20%")
            cur_C = float(rules.get("spread", {}).get("credit_C", {}).get("value", 10.0))
            st.caption(f"At C={cur_C:.1f}, stop = {cur_C * new_stop / 100:.1f} pts")
        with c2:
            new_jump = st.number_input("Jump Exit (pts)", 1.0, 20.0,
                value=float(risk.get("jump_exit_points", {}).get("value", 5.0)),
                step=0.5, key="r_jump",
                help="Exit immediately if price moves this many pts adversely within the first bar.")
            st.caption("Default: 5.0 pts")

        new_cb = st.number_input("Circuit Breaker — Daily Loss Limit (USD)",
            min_value=-10000.0, max_value=-100.0,
            value=float(risk.get("circuit_breaker_usd", {}).get("value", -2000.0)),
            step=100.0, key="r_circuit",
            help="All positions flattened and trading disabled when daily P&L hits this level.")
        st.warning(f"⚡ Circuit breaker triggers at **${new_cb:,.0f}** daily loss.")

        c_save, c_reset = st.columns(2)
        with c_save:
            if st.button("💾 Save Risk Rules", key="save_risk"):
                ok = set_group("risk", {
                    "emergency_stop_pct": new_stop / 100.0,
                    "jump_exit_points": new_jump,
                    "circuit_breaker_usd": new_cb,
                }, user)
                st.success("Saved.") if ok else st.error("Save failed.")
        with c_reset:
            if st.button("🔄 Reset", key="reset_risk"):
                reset_to_defaults("risk", user); st.rerun()

    # ── Tab 5: Session ──────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("🕐 Session Rules")
        st.info("All times are **Central Time (CT)**.")
        session = rules.get("session", {})
        c1, c2 = st.columns(2)
        with c1:
            new_close = st.text_input("Session Close (flatten all)",
                value=str(session.get("session_close_ct", {}).get("value", "15:55")),
                key="r_close", help="Format: HH:MM. Default: 15:55 CT")
            st.caption("5 minutes before regular session close")
        with c2:
            new_reset = st.text_input("Session Reset (daily counters)",
                value=str(session.get("session_reset_ct", {}).get("value", "17:00")),
                key="r_reset", help="Format: HH:MM. Default: 17:00 CT")
            st.caption("Start of overnight session")
        if st.button("💾 Save Session Rules", key="save_session"):
            ok = set_group("session", {"session_close_ct": new_close, "session_reset_ct": new_reset}, user)
            st.success("Saved.") if ok else st.error("Save failed.")

    # ── Tab 6: Indicators ───────────────────────────────────────────────
    with tabs[5]:
        st.subheader("📊 Indicator Parameters")
        st.info("Changes take effect on the next engine restart or hot-reload.")
        ind = rules.get("indicators", {})
        c1, c2, c3 = st.columns(3)
        with c1:
            v_kc_ema = st.number_input("KC EMA Period", 5, 100,
                value=int(ind.get("kc_ema_period", {}).get("value", 20)), key="r_kc_ema")
            st.caption("Default: 20")
            v_kc_atr = st.number_input("KC ATR Period", 5, 50,
                value=int(ind.get("kc_atr_period", {}).get("value", 14)), key="r_kc_atr")
            st.caption("Default: 14")
        with c2:
            v_kc_mult = st.number_input("KC ATR Multiplier", 0.5, 5.0,
                value=float(ind.get("kc_atr_multiplier", {}).get("value", 2.0)),
                step=0.1, key="r_kc_mult")
            st.caption("Default: 2.0")
            v_rsi = st.number_input("RSI Period", 2, 50,
                value=int(ind.get("rsi_period", {}).get("value", 14)), key="r_rsi")
            st.caption("Default: 14")
        with c3:
            v_roc = st.number_input("ROC Period", 1, 20,
                value=int(ind.get("roc_period", {}).get("value", 3)), key="r_roc")
            st.caption("Default: 3")
            v_atr = st.number_input("ATR Period", 5, 50,
                value=int(ind.get("atr_period", {}).get("value", 14)), key="r_atr")
            st.caption("Default: 14")
        if st.button("💾 Save Indicator Parameters", key="save_ind"):
            ok = set_group("indicators", {
                "kc_ema_period": v_kc_ema, "kc_atr_period": v_kc_atr,
                "kc_atr_multiplier": v_kc_mult, "rsi_period": v_rsi,
                "roc_period": v_roc, "atr_period": v_atr,
            }, user)
            st.success("Saved.") if ok else st.error("Save failed.")

    # ── Tab 7: Regime ───────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("🌡️ Regime Detection")
        st.info("Classifies volatility as Low / Medium / High based on ATR percentile over a rolling window.")
        regime = rules.get("regime", {})
        new_lb = st.number_input("ATR Lookback (minutes)", 1440, 43200,
            value=int(regime.get("lookback_minutes", {}).get("value", 10080)),
            step=1440, key="r_lookback",
            help="10080 = 1 week. 1440 = 1 day.")
        c1, c2 = st.columns(2)
        with c1:
            new_pl = st.slider("Low Regime Percentile Cutoff", 10, 45,
                value=int(regime.get("pct_low", {}).get("value", 33)), key="r_pct_low")
            st.caption(f"ATR below {new_pl}th percentile → Low regime")
        with c2:
            new_ph = st.slider("High Regime Percentile Cutoff", 55, 90,
                value=int(regime.get("pct_high", {}).get("value", 66)), key="r_pct_high")
            st.caption(f"ATR above {new_ph}th percentile → High regime")
        if st.button("💾 Save Regime Rules", key="save_regime"):
            ok = set_group("regime", {"lookback_minutes": new_lb, "pct_low": new_pl, "pct_high": new_ph}, user)
            st.success("Saved.") if ok else st.error("Save failed.")

    # ── Tab 8: AI Confidence Layer ──────────────────────────────────────
    with tabs[7]:
        st.subheader("🤖 AI Confidence Layer")
        ai = rules.get("ai", {})
        st.info(
            "The AI layer provides **confidence scores only** — it never executes trades. "
            "The rule-based engine remains the sole execution authority. "
            "The entry gate can block a trade; the exit CNN suggests dynamic trailing multipliers."
        )
        c1, c2 = st.columns(2)
        with c1:
            new_ai_en = st.toggle("Enable AI Layer",
                value=bool(ai.get("ai_enabled", {}).get("value", True)), key="r_ai_en")
            new_fc = st.toggle("Fail-Closed (block trades if AI unavailable)",
                value=bool(ai.get("ai_fail_closed", {}).get("value", True)), key="r_fc",
                help="Recommended for production. If AI model fails to load, all entries are blocked.")
            new_trail_ai = st.toggle("Use AI for Dynamic Trailing Multipliers",
                value=bool(ai.get("trail_ai_enabled", {}).get("value", True)), key="r_trail_ai",
                help="CNN exit controller dynamically adjusts TP2 and runner trailing stop multipliers.")
        with c2:
            new_entry_t = st.slider("Global Entry Confidence Threshold", 0.50, 0.99,
                value=float(ai.get("entry_conf_threshold", {}).get("value", 0.70)),
                step=0.01, format="%.2f", key="r_entry_t",
                help="Minimum AI confidence to allow a trade. Default: 0.70")
            st.caption("Higher = fewer but higher-confidence entries.")
            new_exit_t = st.slider("Exit Hold Confidence Threshold", 0.50, 0.99,
                value=float(ai.get("exit_conf_threshold", {}).get("value", 0.65)),
                step=0.01, format="%.2f", key="r_exit_t",
                help="If CNN continuation probability > threshold, engine holds rather than exiting.")
            st.caption("Default: 0.65")

        st.markdown("**Per-Regime Entry Thresholds** (override global threshold by regime)")
        c1, c2, c3 = st.columns(3)
        with c1:
            new_tl = st.number_input("Low Regime", 0.50, 0.99,
                value=float(ai.get("regime_thresholds_low", {}).get("value", 0.58)),
                step=0.01, format="%.2f", key="r_tl"); st.caption("Default: 0.58")
        with c2:
            new_tm = st.number_input("Med Regime", 0.50, 0.99,
                value=float(ai.get("regime_thresholds_med", {}).get("value", 0.55)),
                step=0.01, format="%.2f", key="r_tm"); st.caption("Default: 0.55")
        with c3:
            new_th = st.number_input("High Regime", 0.50, 0.99,
                value=float(ai.get("regime_thresholds_high", {}).get("value", 0.52)),
                step=0.01, format="%.2f", key="r_th"); st.caption("Default: 0.52")

        if st.button("💾 Save AI Layer Rules", key="save_ai"):
            ok = set_group("ai", {
                "ai_enabled": str(new_ai_en).lower(),
                "ai_fail_closed": str(new_fc).lower(),
                "trail_ai_enabled": str(new_trail_ai).lower(),
                "entry_conf_threshold": new_entry_t,
                "exit_conf_threshold": new_exit_t,
                "regime_thresholds_low": new_tl,
                "regime_thresholds_med": new_tm,
                "regime_thresholds_high": new_th,
            }, user)
            st.success("Saved.") if ok else st.error("Save failed.")

    # ── Tab 9: RL Trailing Agent ────────────────────────────────────────
    with tabs[8]:
        st.subheader("🧠 RL Trailing Stop Agent")
        rl = rules.get("rl", {})
        st.info(
            "The Q-Learning agent adjusts the runner lot's trailing stop each bar. "
            "It learns from trade outcomes to tighten or widen the trail based on "
            "regime, unrealized P&L, RSI, and ROC."
        )
        c1, c2 = st.columns(2)
        with c1:
            new_a = st.number_input("Learning Rate (α)", 0.001, 0.5,
                value=float(rl.get("rl_alpha", {}).get("value", 0.1)),
                step=0.005, format="%.3f", key="r_alpha"); st.caption("Default: 0.1")
            new_g = st.number_input("Discount Factor (γ)", 0.5, 0.999,
                value=float(rl.get("rl_gamma", {}).get("value", 0.95)),
                step=0.005, format="%.3f", key="r_gamma"); st.caption("Default: 0.95")
        with c2:
            new_e = st.number_input("Exploration Rate (ε)", 0.0, 0.5,
                value=float(rl.get("rl_epsilon", {}).get("value", 0.1)),
                step=0.01, format="%.2f", key="r_epsilon"); st.caption("Default: 0.1")
            new_l = st.number_input("Drawdown Penalty (λ)", 0.0, 2.0,
                value=float(rl.get("rl_lambda_dd", {}).get("value", 0.5)),
                step=0.05, format="%.2f", key="r_lambda"); st.caption("Default: 0.5")

        st.markdown("""
| Action | Effect |
|--------|--------|
| Tighten (−0.1 × ATR) | Move trailing stop closer to price |
| Hold (0.0) | Keep trailing stop unchanged |
| Widen (+0.1 × ATR) | Move trailing stop further from price |
        """)
        if st.button("💾 Save RL Rules", key="save_rl"):
            ok = set_group("rl", {
                "rl_alpha": new_a, "rl_gamma": new_g,
                "rl_epsilon": new_e, "rl_lambda_dd": new_l,
            }, user)
            st.success("Saved.") if ok else st.error("Save failed.")

    # ── Danger Zone ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚠️ Danger Zone — Reset All Rules"):
        st.warning("Resets **all** strategy rules to factory defaults. Cannot be undone.")
        if st.button("🔄 Reset ALL to Factory Defaults", key="reset_all", type="primary"):
            reset_to_defaults(updated_by=user)
            st.success("All rules reset.")
            st.rerun()
```

---

## Part 5: AI Confidence Overlay on ES Strategy Page

### Task 5.1 — Add AI Confidence metrics row to `page_es()` in `app.py`

In `page_es()`, after the position banner div and before the "Price Chart" subheader, add:

```python
# AI Confidence overlay — reads from engine state
ai_state = state.get("ai_confidence", {})
if ai_state.get("ai_enabled", False):
    st.subheader("🤖 AI Confidence")
    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        ec = ai_state.get("entry_conf")
        if ec is not None:
            st.metric("Entry Confidence", f"{ec:.1%}",
                      delta="Allow" if ec >= 0.70 else "Block",
                      delta_color="normal" if ec >= 0.70 else "inverse")
            st.progress(float(ec))
        else:
            st.metric("Entry Confidence", "—"); st.caption("No signal yet")
    with ac2:
        xc = ai_state.get("exit_conf")
        p_cont = 1.0 - xc if xc is not None else None
        if p_cont is not None:
            st.metric("Continuation Prob.", f"{p_cont:.1%}",
                      delta="Hold" if p_cont >= 0.65 else "Exit",
                      delta_color="normal" if p_cont >= 0.65 else "inverse")
            st.progress(float(p_cont))
        else:
            st.metric("Continuation Prob.", "—"); st.caption("No open position")
    with ac3:
        trail = ai_state.get("trail_ai_active", False)
        st.metric("AI Trailing", "Active ✅" if trail else "Disabled ⛔")
        st.caption("CNN adjusts TP2 & runner multipliers")
    with ac4:
        try:
            import requests as _rq
            h = _rq.get("http://localhost:8100/health", timeout=2).json()
            eg = h.get("entry_gate_loaded", False)
            ex = h.get("exit_ctrl_loaded", False)
            st.metric("AI Models", "Online ✅" if (eg and ex) else "Partial ⚠️")
            st.caption(f"Entry: {'✅' if eg else '❌'}  Exit CNN: {'✅' if ex else '❌'}")
        except Exception:
            st.metric("AI Models", "Offline ❌")
            st.caption("Confidence server not running on :8100")
```

### Task 5.2 — Add "Reload Rules" button to `page_es()`

At the very top of `page_es()`, after `st.markdown(page_header(...))`, add:

```python
_hdr, _btn = st.columns([5, 1])
with _btn:
    if st.button("🔄 Reload Rules", key="es_reload_rules",
                 help="Hot-reload strategy rules from DB without restarting"):
        import json, os
        os.makedirs("data", exist_ok=True)
        with open("data/reload_rules_flag.json", "w") as f:
            json.dump({"reload": True, "at": datetime.now().isoformat()}, f)
        st.toast("Rules reload requested — takes effect on next bar.", icon="🔄")
```

### Task 5.3 — Handle hot-reload flag in `src/es_strategy/runner.py`

In `runner.py`, in the main bar-processing method (whichever method calls `self.engine.process_bar()`), add at the top of that method:

```python
import json, os
_flag = "data/reload_rules_flag.json"
if os.path.exists(_flag):
    try:
        self.engine.reload_rules()
        os.remove(_flag)
        logger.info("Strategy rules hot-reloaded from DB")
    except Exception as e:
        logger.warning(f"Rules reload failed: {e}")
```

---

## Part 6: Register Rules Page in Navigation — `src/dashboard/app.py`

### Task 6.1 — Add import

At the top of `app.py`, alongside the other dashboard imports:

```python
from src.dashboard.rules_app import page_rules
```

### Task 6.2 — Add to `_pages` navigation dict

In the `_pages` dict (around line 3170), add `page_rules` between `page_es` and `page_tuning`:

```python
"Markets": [
    st.Page(page_spy,          title="SPY Predictor",    icon=":material/query_stats:"),
    st.Page(page_performance,  title="Performance",      icon=":material/verified:"),
    st.Page(page_es,           title="ES Strategy",      icon=":material/candlestick_chart:"),
    st.Page(page_rules,        title="Rules",            icon=":material/rule:"),        # NEW
    st.Page(page_tuning,       title="Tune & Backtest",  icon=":material/tune:"),
    st.Page(page_whatif,       title="What-If Analysis", icon=":material/science:"),
    st.Page(page_single_stock, title="Single-Stock",     icon=":material/search:"),
    st.Page(page_quant_agent,  title="Quant Agent",      icon=":material/smart_toy:"),
],
```

---

## Part 7: Replace Admin Config Tab with Read-Only Summary

### Task 7.1 — Replace `_admin_config_tab()` body in `app.py`

Find `def _admin_config_tab()` (around line 2358). Replace its entire body with:

```python
def _admin_config_tab():
    st.subheader("Configuration")
    st.info(
        "Strategy parameters are managed through the **⚙️ Rules** page in the sidebar. "
        "System-level settings (API keys, LLM model, database) are managed by the server administrator."
    )
    if st.button("→ Go to Rules Page", key="goto_rules_from_admin"):
        st.switch_page("rules")

    st.divider()
    st.markdown("**System Summary (read-only)**")
    try:
        config = _load_config()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LLM Model",    config.get("llm", {}).get("model", "—"))
        c2.metric("XGB Lookback", f"{config.get('xgboost', {}).get('lookback_days', '—')} days")
        c3.metric("Cloud Sync",   "On" if config.get("sync", {}).get("enabled") else "Off")
        c4.metric("DB Mode",      "PostgreSQL" if config.get("database", {}).get("postgres") else "SQLite")
    except Exception as e:
        st.warning(f"Could not load config summary: {e}")
```

---

## Summary of All Files to Create or Modify

| Action | File | Change |
| :--- | :--- | :--- |
| **Modify** | `src/data/init_db.py` | Add `strategy_rules` table to `_migrate_schema()`, add `seed_strategy_rules()`, add `_seed_strategy_rules_pg()`, call seed in `init_db()` PostgreSQL block |
| **Create** | `src/strategy/__init__.py` | Empty package init |
| **Create** | `src/strategy/rules_store.py` | Full DB-backed rules read/write module |
| **Modify** | `src/es_strategy/engine.py` | Replace `__init__` to use `rules_store`, add `reload_rules()`, update anti-chase gate, update Phase 2 thresholds, wire AI exit confidence into `_check_exits()`, add AI state to `get_state()` |
| **Modify** | `src/es_strategy/runner.py` | Add hot-reload flag check in bar-processing method |
| **Create** | `src/dashboard/rules_app.py` | Full Rules Management UI (9 tabs, pre-populated, save + reset) |
| **Modify** | `src/dashboard/app.py` | Import `page_rules`, add to navigation, add AI overlay to `page_es()`, add Reload Rules button, update `ai_enabled` toggle to use `rules_store`, replace `_admin_config_tab()` |

---

## Acceptance Criteria

1. **Rules page** appears in sidebar under "Markets" with icon `:material/rule:`.
2. **All 9 tabs** show pre-populated values matching the original `engine.py` hardcoded defaults.
3. **Saving any rule** persists to `strategy_rules` table in PostgreSQL (or SQLite fallback). No `config.yaml` is written.
4. **Reset to defaults** restores factory values without touching `config.yaml`.
5. **Engine reads all parameters** from `rules_store` at startup; `config.yaml` values are fallback only.
6. **AI Confidence overlay** is visible on the ES Strategy page with entry confidence, continuation probability, and model health metrics.
7. **AI trailing multipliers** from the CNN exit controller are applied to TP2 and runner trail when `trail_ai_enabled = true`.
8. **Reload Rules button** on ES Strategy page triggers hot-reload on next bar.
9. **Admin → Configuration tab** shows read-only summary and link to Rules page — no YAML text editor.
10. **`ai_enabled` toggle** on ES Strategy page writes to `strategy_rules` DB, not `config.yaml`.
11. **Performance and Tuning pages** are left untouched — they are already implemented.
