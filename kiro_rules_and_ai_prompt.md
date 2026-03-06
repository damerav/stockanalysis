# Kiro Implementation Prompt: Rules Management UI + AI Confidence Layer
## Project: `damerav/stockanalysis` — Full Feature Implementation

---

## Context & Architecture Philosophy

The `damerav/stockanalysis` platform is a Streamlit-based ES Futures trading dashboard running on a DGX Spark server. The core principle is a **strict separation of concerns**:

> The **rule-based execution engine** (`src/es_strategy/engine.py`) is the sole authority for all trade execution — entries, exits, take-profits, stops, and trailing logic. It executes deterministically based on parameters.
>
> The **AI confidence layer** sits entirely on top and acts as a **weighted suggestion system only**. It analyzes market data, volatility, and context metrics and outputs confidence scores for entry strength, hold/exit decisions, and trailing opportunities. The AI never executes trades — it only influences the parameters and gates that the rule-based engine uses.

All strategy parameters are currently hardcoded in `engine.py` or buried in `config.yaml`. **There is no UI to view or change rules.** This prompt implements:

1. A **Rules Management screen** — a full UI with pre-populated defaults and live editing capability, persisted to the database (not YAML files)
2. **AI Confidence Layer wiring** — connecting the already-built `ESEntryGate`, `ESExitController`, and `RLTrailingAgent` models into the engine's exit and trailing logic
3. An **AI Confidence Overlay** on the ES Strategy dashboard showing real-time confidence scores
4. A **`strategy_rules` database table** to persist all rule changes

---

## Part 1: Database Schema — `strategy_rules` Table

### Task 1.1 — Add `strategy_rules` table to `src/data/init_db.py`

In the `init_db()` function in `src/data/init_db.py`, add the following `CREATE TABLE` statement alongside the existing tables. This table stores all strategy rule parameters as key-value pairs so they can be read by the engine at runtime without touching any YAML file.

```python
conn.executescript("""
    CREATE TABLE IF NOT EXISTS strategy_rules (
        rule_group     TEXT NOT NULL,
        rule_key       TEXT NOT NULL,
        rule_value     TEXT NOT NULL,
        value_type     TEXT NOT NULL DEFAULT 'float',
        min_val        TEXT,
        max_val        TEXT,
        description    TEXT,
        updated_at     TEXT,
        updated_by     TEXT,
        PRIMARY KEY (rule_group, rule_key)
    );
""")
```

### Task 1.2 — Seed default values on first run

In `init_db.py`, after creating the table, add a `seed_strategy_rules()` function that inserts the following defaults using `INSERT OR IGNORE` so existing customizations are never overwritten. Call this function at the end of `init_db()`.

```python
def seed_strategy_rules(conn):
    """Seed default strategy rule values. INSERT OR IGNORE preserves user changes."""
    from datetime import datetime
    now = datetime.now().isoformat()
    defaults = [
        # --- Spread Parameters ---
        ("spread",    "strike_K",              "6000.0",  "float",  "5000", "7000",  "Sold strike price (K)"),
        ("spread",    "credit_C",              "10.0",    "float",  "1",    "50",    "Credit width (C) in points"),

        # --- Position Sizing ---
        ("sizing",    "max_lots",              "3",       "int",    "1",    "10",    "Maximum number of lots per trade"),

        # --- Entry Rules ---
        ("entry",     "anti_chase_atr_pct",    "0.5",     "float",  "0.1",  "2.0",  "Anti-chase gate: max distance from KC band as fraction of ATR"),
        ("entry",     "phase2_enabled",        "true",    "bool",   None,   None,   "Enable Phase 2 confluence reload entries"),
        ("entry",     "phase2_min_filters",    "2",       "int",    "1",    "3",    "Minimum confluence filters required for Phase 2 entry (out of 3)"),
        ("entry",     "phase2_roc_threshold",  "0.5",     "float",  "0.1",  "2.0",  "ROC momentum threshold for Phase 2 filter 1"),

        # --- Take-Profit Multipliers (× ATR) by Regime ---
        ("tp_low",    "tp1_mult",              "1.0",     "float",  "0.5",  "5.0",  "TP1 multiplier × ATR — Low volatility regime"),
        ("tp_low",    "tp2_mult",              "1.5",     "float",  "0.5",  "5.0",  "TP2 multiplier × ATR — Low volatility regime"),
        ("tp_low",    "runner_trail_mult",     "2.0",     "float",  "0.5",  "8.0",  "Runner trailing stop multiplier × ATR — Low volatility regime"),
        ("tp_med",    "tp1_mult",              "1.2",     "float",  "0.5",  "5.0",  "TP1 multiplier × ATR — Medium volatility regime"),
        ("tp_med",    "tp2_mult",              "1.8",     "float",  "0.5",  "5.0",  "TP2 multiplier × ATR — Medium volatility regime"),
        ("tp_med",    "runner_trail_mult",     "2.5",     "float",  "0.5",  "8.0",  "Runner trailing stop multiplier × ATR — Medium volatility regime"),
        ("tp_high",   "tp1_mult",              "1.5",     "float",  "0.5",  "5.0",  "TP1 multiplier × ATR — High volatility regime"),
        ("tp_high",   "tp2_mult",              "2.2",     "float",  "0.5",  "5.0",  "TP2 multiplier × ATR — High volatility regime"),
        ("tp_high",   "runner_trail_mult",     "3.0",     "float",  "0.5",  "8.0",  "Runner trailing stop multiplier × ATR — High volatility regime"),

        # --- Stop & Risk Rules ---
        ("risk",      "emergency_stop_pct",    "0.20",    "float",  "0.05", "0.50", "Emergency stop as fraction of credit C (e.g. 0.20 = 20% of C)"),
        ("risk",      "jump_exit_points",      "5.0",     "float",  "1.0",  "20.0", "Adverse move in points within first bar that triggers immediate exit"),
        ("risk",      "circuit_breaker_usd",   "-2000.0", "float",  "-10000", "-100", "Daily P&L loss limit in USD that triggers circuit breaker"),

        # --- Session Rules ---
        ("session",   "session_close_ct",      "15:55",   "time",   None,   None,   "Flatten all positions before this time (Central Time)"),
        ("session",   "session_reset_ct",      "17:00",   "time",   None,   None,   "Reset daily counters at this time (Central Time)"),

        # --- Keltner Channel Indicator ---
        ("indicators","kc_ema_period",         "20",      "int",    "5",    "100",  "Keltner Channel EMA period"),
        ("indicators","kc_atr_period",         "14",      "int",    "5",    "50",   "Keltner Channel ATR period"),
        ("indicators","kc_atr_multiplier",     "2.0",     "float",  "0.5",  "5.0",  "Keltner Channel ATR multiplier"),
        ("indicators","rsi_period",            "14",      "int",    "2",    "50",   "RSI period"),
        ("indicators","roc_period",            "3",       "int",    "1",    "20",   "Rate of Change period"),
        ("indicators","atr_period",            "14",      "int",    "5",    "50",   "ATR period for position sizing"),

        # --- Regime Detection ---
        ("regime",    "lookback_minutes",      "10080",   "int",    "1440", "43200","ATR history lookback in minutes for regime detection (10080 = 1 week)"),
        ("regime",    "pct_low",               "33",      "int",    "10",   "45",   "ATR percentile below which regime is 'Low'"),
        ("regime",    "pct_high",              "66",      "int",    "55",   "90",   "ATR percentile above which regime is 'High'"),

        # --- AI Confidence Layer ---
        ("ai",        "ai_enabled",            "true",    "bool",   None,   None,   "Master switch: enable AI confidence layer"),
        ("ai",        "ai_fail_closed",        "true",    "bool",   None,   None,   "Fail-closed: block trades if AI model is unavailable"),
        ("ai",        "entry_conf_threshold",  "0.70",    "float",  "0.50", "0.99", "Minimum AI entry confidence to allow a trade (0.50–0.99)"),
        ("ai",        "exit_conf_threshold",   "0.65",    "float",  "0.50", "0.99", "AI reversal confidence above which the engine holds (does not exit early)"),
        ("ai",        "trail_ai_enabled",      "true",    "bool",   None,   None,   "Use AI CNN exit controller to set dynamic trailing stop multipliers"),
        ("ai",        "regime_thresholds_low", "0.58",    "float",  "0.50", "0.99", "Entry gate threshold override for Low volatility regime"),
        ("ai",        "regime_thresholds_med", "0.55",    "float",  "0.50", "0.99", "Entry gate threshold override for Medium volatility regime"),
        ("ai",        "regime_thresholds_high","0.52",    "float",  "0.50", "0.99", "Entry gate threshold override for High volatility regime"),

        # --- RL Trailing Agent ---
        ("rl",        "rl_alpha",              "0.1",     "float",  "0.001","0.5",  "RL Q-learning rate"),
        ("rl",        "rl_gamma",              "0.95",    "float",  "0.5",  "0.999","RL discount factor"),
        ("rl",        "rl_epsilon",            "0.1",     "float",  "0.0",  "0.5",  "RL exploration rate (epsilon-greedy)"),
        ("rl",        "rl_lambda_dd",          "0.5",     "float",  "0.0",  "2.0",  "RL drawdown penalty weight"),
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

---

## Part 2: Rules Helper Module — `src/strategy/rules_store.py`

Create a new file `src/strategy/rules_store.py`. This module is the single source of truth for reading and writing strategy rules. The engine, runner, and dashboard all use this module — **no component reads config.yaml for strategy parameters**.

```python
"""
src/strategy/rules_store.py
Single source of truth for strategy rule parameters.
All reads/writes go through this module — no component reads config.yaml directly.
"""
import logging
from datetime import datetime
from typing import Any, Optional
logger = logging.getLogger(__name__)

def _get_conn():
    """Get a database connection using the existing db_router."""
    from src.data.db_router import get_router
    import yaml, os
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    router = get_router(cfg)
    return router

def get_rule(group: str, key: str, default: Any = None) -> Any:
    """Read a single rule value, cast to its stored type. Returns default on error."""
    try:
        router = _get_conn()
        df = router.read_analytics(
            "SELECT rule_value, value_type FROM strategy_rules WHERE rule_group=? AND rule_key=?",
            params=(group, key)
        )
        if df.empty:
            return default
        val = df.iloc[0]["rule_value"]
        vtype = df.iloc[0]["value_type"]
        return _cast(val, vtype)
    except Exception as e:
        logger.warning(f"rules_store.get_rule({group}.{key}) failed: {e}")
        return default

def get_group(group: str) -> dict:
    """Read all rules in a group as a dict of {key: cast_value}."""
    try:
        router = _get_conn()
        df = router.read_analytics(
            "SELECT rule_key, rule_value, value_type FROM strategy_rules WHERE rule_group=?",
            params=(group,)
        )
        return {row["rule_key"]: _cast(row["rule_value"], row["value_type"]) for _, row in df.iterrows()}
    except Exception as e:
        logger.warning(f"rules_store.get_group({group}) failed: {e}")
        return {}

def get_all_rules() -> dict:
    """Read all rules as nested dict: {group: {key: value}}."""
    try:
        router = _get_conn()
        df = router.read_analytics(
            "SELECT rule_group, rule_key, rule_value, value_type, min_val, max_val, description, updated_at, updated_by "
            "FROM strategy_rules ORDER BY rule_group, rule_key"
        )
        result = {}
        for _, row in df.iterrows():
            g = row["rule_group"]
            if g not in result:
                result[g] = {}
            result[g][row["rule_key"]] = {
                "value": _cast(row["rule_value"], row["value_type"]),
                "raw": row["rule_value"],
                "type": row["value_type"],
                "min": row.get("min_val"),
                "max": row.get("max_val"),
                "description": row.get("description", ""),
                "updated_at": row.get("updated_at", ""),
                "updated_by": row.get("updated_by", ""),
            }
        return result
    except Exception as e:
        logger.warning(f"rules_store.get_all_rules() failed: {e}")
        return {}

def set_rule(group: str, key: str, value: Any, updated_by: str = "ui") -> bool:
    """Update a single rule value. Returns True on success."""
    try:
        router = _get_conn()
        now = datetime.now().isoformat()
        router.execute_write(
            "UPDATE strategy_rules SET rule_value=?, updated_at=?, updated_by=? "
            "WHERE rule_group=? AND rule_key=?",
            params=(str(value), now, updated_by, group, key)
        )
        logger.info(f"Rule updated: {group}.{key} = {value} by {updated_by}")
        return True
    except Exception as e:
        logger.error(f"rules_store.set_rule({group}.{key}) failed: {e}")
        return False

def set_group(group: str, updates: dict, updated_by: str = "ui") -> bool:
    """Update multiple rules in a group atomically. Returns True if all succeed."""
    success = True
    for key, value in updates.items():
        if not set_rule(group, key, value, updated_by):
            success = False
    return success

def reset_to_defaults(group: Optional[str] = None, updated_by: str = "ui") -> bool:
    """Reset rules to factory defaults by re-running the seed with forced overwrite."""
    try:
        router = _get_conn()
        # Re-import and re-run seed with UPDATE instead of INSERT OR IGNORE
        from src.data.init_db import seed_strategy_rules
        # Get raw connection for seed function
        import sqlite3, os
        db_path = "./data/spy.db"
        conn = sqlite3.connect(db_path)
        if group:
            conn.execute("DELETE FROM strategy_rules WHERE rule_group=?", (group,))
        else:
            conn.execute("DELETE FROM strategy_rules")
        conn.commit()
        seed_strategy_rules(conn)
        conn.close()
        logger.info(f"Rules reset to defaults (group={group or 'ALL'}) by {updated_by}")
        return True
    except Exception as e:
        logger.error(f"rules_store.reset_to_defaults failed: {e}")
        return False

def _cast(value: str, vtype: str) -> Any:
    """Cast a string value to its proper Python type."""
    try:
        if vtype == "int":
            return int(float(value))
        elif vtype == "float":
            return float(value)
        elif vtype == "bool":
            return value.lower() in ("true", "1", "yes")
        else:
            return value  # str or time
    except Exception:
        return value
```

---

## Part 3: Update `ESStrategyEngine` to Read from `rules_store`

### Task 3.1 — Modify `src/es_strategy/engine.py`

Replace the `__init__` method's hardcoded `cfg.get(...)` calls with `rules_store` reads. The engine now loads its parameters from the database at startup. Add a `reload_rules()` method so the dashboard can hot-reload rules without restarting the engine.

Replace the `__init__` body with:

```python
def __init__(self, config: dict = None):
    # Load live rules from DB; fall back to config.yaml values if DB unavailable
    from src.strategy.rules_store import get_rule, get_group
    cfg = (config or {}).get("es_strategy", {})

    self.C                  = get_rule("spread",     "credit_C",             cfg.get("credit_C", 10.0))
    self.K                  = get_rule("spread",     "strike_K",             cfg.get("strike_K", 6000.0))
    self.max_lots           = get_rule("sizing",     "max_lots",             cfg.get("max_lots", 3))
    self.jump_exit_pts      = get_rule("risk",       "jump_exit_points",     cfg.get("jump_exit_points", 5.0))
    self.emergency_stop_pct = get_rule("risk",       "emergency_stop_pct",   cfg.get("emergency_stop_pct", 0.20))
    self.circuit_breaker_usd= get_rule("risk",       "circuit_breaker_usd",  cfg.get("circuit_breaker_usd", -2000.0))
    self.session_close_ct   = get_rule("session",    "session_close_ct",     cfg.get("session_close_ct", "15:55"))
    self.session_reset_ct   = get_rule("session",    "session_reset_ct",     cfg.get("session_reset_ct", "17:00"))
    self.ai_enabled         = get_rule("ai",         "ai_enabled",           cfg.get("ai_enabled", True))
    self._phase2_enabled    = get_rule("entry",      "phase2_enabled",       True)
    self._anti_chase_pct    = get_rule("entry",      "anti_chase_atr_pct",   0.5)
    self._phase2_min_filters= get_rule("entry",      "phase2_min_filters",   2)
    self._phase2_roc_thresh = get_rule("entry",      "phase2_roc_threshold", 0.5)

    # AI confidence thresholds
    self._exit_conf_threshold = get_rule("ai",       "exit_conf_threshold",  0.65)
    self._trail_ai_enabled    = get_rule("ai",       "trail_ai_enabled",     True)

    # TP multipliers loaded from DB by regime
    self._tp_multipliers = {
        "Low":  {
            "tp1":          get_rule("tp_low",  "tp1_mult",          1.0),
            "tp2":          get_rule("tp_low",  "tp2_mult",          1.5),
            "runner_trail": get_rule("tp_low",  "runner_trail_mult", 2.0),
        },
        "Med":  {
            "tp1":          get_rule("tp_med",  "tp1_mult",          1.2),
            "tp2":          get_rule("tp_med",  "tp2_mult",          1.8),
            "runner_trail": get_rule("tp_med",  "runner_trail_mult", 2.5),
        },
        "High": {
            "tp1":          get_rule("tp_high", "tp1_mult",          1.5),
            "tp2":          get_rule("tp_high", "tp2_mult",          2.2),
            "runner_trail": get_rule("tp_high", "runner_trail_mult", 3.0),
        },
    }

    self.position = Position()
    self.regime_detector = RegimeDetector(
        lookback=get_rule("regime", "lookback_minutes",  10080),
        pct_low= get_rule("regime", "pct_low",           33),
        pct_high=get_rule("regime", "pct_high",          66),
    )
    self.signals: list[Signal] = []
    self.circuit_breaker_active = False
    self._bars_since_entry = 0

    self.rl_trail = RLTrailingAgent(
        alpha=    get_rule("rl", "rl_alpha",     0.1),
        gamma=    get_rule("rl", "rl_gamma",     0.95),
        epsilon=  get_rule("rl", "rl_epsilon",   0.1),
        lambda_dd=get_rule("rl", "rl_lambda_dd", 0.5),
    )
    self.rl_trail.load()

def reload_rules(self):
    """Hot-reload all rule parameters from the database without restarting."""
    self.__init__()
    logger.info("Strategy rules reloaded from database")
```

### Task 3.2 — Update `_check_entry` to use DB-backed anti-chase gate

In `_check_entry`, replace the hardcoded `0.5 * atr_val` anti-chase gate with `self._anti_chase_pct`:

```python
# BEFORE:
if price <= kc_lower and abs(price - kc_lower) < 0.5 * atr_val:

# AFTER:
if price <= kc_lower and abs(price - kc_lower) < self._anti_chase_pct * atr_val:
```

Apply the same change for the SHORT side (`kc_upper`).

### Task 3.3 — Update `_check_phase2_entry` to use DB-backed thresholds

Replace the hardcoded `roc_3 < -0.5` and `roc_3 > 0.5` and `filters_passed < 2` with:

```python
if at_lower and roc_3 < -self._phase2_roc_thresh:
    filters_passed += 1
elif at_upper and roc_3 > self._phase2_roc_thresh:
    filters_passed += 1
# ...
if filters_passed < self._phase2_min_filters:
    return None
```

### Task 3.4 — Wire AI Exit Confidence into `_check_exits`

The `ESExitController` CNN already produces `exit_conf_reversal` (probability of reversal) and dynamic trail multipliers. These are currently unused in the engine. Wire them in as follows.

At the top of `_check_exits`, after computing `mults`, add:

```python
# AI Exit Confidence: ask CNN for reversal probability and dynamic trail multipliers
ai_trail_mults = mults  # default to rule-based multipliers
if self._trail_ai_enabled and hasattr(self, '_exit_ctrl') and self._exit_ctrl is not None:
    try:
        bar_window = getattr(self, '_ai_bar_window', None)
        if bar_window is not None and len(bar_window) >= 20:
            ai_result = self._exit_ctrl.predict(
                np.array(bar_window[-20:]), regime
            )
            # Only use AI trail multipliers if CNN is loaded
            if ai_result.get("p_cont_5", 0.5) != 0.5:
                ai_trail_mults = {
                    "tp1":          mults["tp1"],   # TP1 stays rule-based
                    "tp2":          ai_result.get("tp2_trail", mults["tp2"]),
                    "runner_trail": ai_result.get("runner_trail", mults["runner_trail"]),
                }
                # Store AI confidence on engine state for dashboard display
                self._last_ai_exit_conf = round(ai_result.get("p_cont_5", 0.5), 4)
    except Exception as e:
        logger.warning(f"AI exit controller failed, using rule-based: {e}")
```

Then replace `mults["runner_trail"]` with `ai_trail_mults["runner_trail"]` and `mults["tp2"]` with `ai_trail_mults["tp2"]` in the TP2 and runner trailing stop calculations below.

### Task 3.5 — Expose AI state in `get_state()`

Add AI confidence scores to the engine's state dict so the dashboard can display them:

```python
def get_state(self) -> dict:
    return {
        "position": self.position.to_dict(),
        "regime": self.regime,
        "spread": {"strike_K": self.K, "credit_C": self.C},
        "pnl": {
            "daily": round(self.position.daily_pnl, 2),
            "unrealized": round(self.position.unrealized_pnl, 2),
        },
        "circuit_breaker": "ACTIVE" if self.circuit_breaker_active else "OK",
        "trade_count": self.position.trade_count,
        "signals": [s.to_dict() for s in self.signals[-50:]],
        # NEW: AI confidence scores for dashboard overlay
        "ai_confidence": {
            "entry_conf":    getattr(self, "_last_ai_entry_conf", None),
            "exit_conf":     getattr(self, "_last_ai_exit_conf", None),
            "trail_ai_active": self._trail_ai_enabled,
        },
    }
```

---

## Part 4: New Dashboard Page — Rules Management (`src/dashboard/rules_app.py`)

Create a new file `src/dashboard/rules_app.py`. This is a full Streamlit page with tabbed sections for each rule group. All values are pre-populated from the database (which was seeded with the defaults from Part 1). Changes are saved back to the database immediately on button click.

```python
"""
src/dashboard/rules_app.py
Strategy Rules Management page.
All parameters are loaded from the strategy_rules DB table and saved back on change.
No YAML files are read or written from this page.
"""
import streamlit as st
from src.strategy.rules_store import get_all_rules, set_group, reset_to_defaults
from src.dashboard.theme import apply_theme

GROUP_LABELS = {
    "spread":     "📐 Spread Parameters",
    "sizing":     "📦 Position Sizing",
    "entry":      "🟢 Entry Rules",
    "tp_low":     "🎯 Take-Profit — Low Volatility",
    "tp_med":     "🎯 Take-Profit — Medium Volatility",
    "tp_high":    "🎯 Take-Profit — High Volatility",
    "risk":       "🛡️ Stop & Risk Rules",
    "session":    "🕐 Session Rules",
    "indicators": "📊 Indicator Parameters",
    "regime":     "🌡️ Regime Detection",
    "ai":         "🤖 AI Confidence Layer",
    "rl":         "🧠 RL Trailing Agent",
}

def page_rules():
    apply_theme()
    st.title("⚙️ Strategy Rules")
    st.caption(
        "All parameters are pre-populated with factory defaults. "
        "Changes are saved to the database immediately and take effect on the next bar processed by the engine. "
        "Use **Reload Rules** on the ES Strategy page to apply changes to a running engine without restarting."
    )

    rules = get_all_rules()
    if not rules:
        st.error("Could not load rules from database. Ensure the database has been initialized.")
        return

    # --- Tab layout ---
    tab_spread, tab_entry, tab_tp, tab_risk, tab_session, tab_indicators, tab_regime, tab_ai, tab_rl = st.tabs([
        "📐 Spread & Sizing",
        "🟢 Entry",
        "🎯 Take-Profit",
        "🛡️ Risk & Stops",
        "🕐 Session",
        "📊 Indicators",
        "🌡️ Regime",
        "🤖 AI Layer",
        "🧠 RL Trailing",
    ])

    # ------------------------------------------------------------------ #
    # TAB 1: Spread & Sizing
    # ------------------------------------------------------------------ #
    with tab_spread:
        st.subheader("📐 Spread Parameters")
        st.info("These define the Keltner Channel spread used for entry zone calculation.")
        spread = rules.get("spread", {})
        sizing = rules.get("sizing", {})

        col1, col2 = st.columns(2)
        with col1:
            new_K = st.number_input(
                "Strike K (Sold Strike Price)",
                min_value=float(spread.get("strike_K", {}).get("min", 5000)),
                max_value=float(spread.get("strike_K", {}).get("max", 7000)),
                value=float(spread.get("strike_K", {}).get("value", 6000.0)),
                step=50.0,
                help=spread.get("strike_K", {}).get("description", ""),
                key="rule_strike_K",
            )
            st.caption(f"Default: 6000.0 | Last saved: {spread.get('strike_K', {}).get('updated_at', 'never')[:16]}")
        with col2:
            new_C = st.number_input(
                "Credit C (Width in Points)",
                min_value=float(spread.get("credit_C", {}).get("min", 1)),
                max_value=float(spread.get("credit_C", {}).get("max", 50)),
                value=float(spread.get("credit_C", {}).get("value", 10.0)),
                step=0.5,
                help=spread.get("credit_C", {}).get("description", ""),
                key="rule_credit_C",
            )
            st.caption(f"Default: 10.0 | Last saved: {spread.get('credit_C', {}).get('updated_at', 'never')[:16]}")

        st.subheader("📦 Position Sizing")
        new_lots = st.slider(
            "Maximum Lots Per Trade",
            min_value=int(sizing.get("max_lots", {}).get("min", 1)),
            max_value=int(sizing.get("max_lots", {}).get("max", 10)),
            value=int(sizing.get("max_lots", {}).get("value", 3)),
            help=sizing.get("max_lots", {}).get("description", ""),
            key="rule_max_lots",
        )
        st.caption("Default: 3. The AI entry gate may reduce this based on confidence (1–3 lots).")

        if st.button("💾 Save Spread & Sizing", key="save_spread"):
            ok1 = set_group("spread", {"strike_K": new_K, "credit_C": new_C},
                            updated_by=st.session_state.get("username", "ui"))
            ok2 = set_group("sizing", {"max_lots": new_lots},
                            updated_by=st.session_state.get("username", "ui"))
            if ok1 and ok2:
                st.success("✅ Spread & Sizing rules saved.")
            else:
                st.error("❌ Failed to save. Check database connection.")

        if st.button("🔄 Reset to Defaults", key="reset_spread"):
            reset_to_defaults("spread")
            reset_to_defaults("sizing")
            st.success("Reset to factory defaults.")
            st.rerun()

    # ------------------------------------------------------------------ #
    # TAB 2: Entry Rules
    # ------------------------------------------------------------------ #
    with tab_entry:
        st.subheader("🟢 Entry Rules")
        entry = rules.get("entry", {})

        st.markdown("**Phase 1 Entry — Anti-Chase Gate**")
        st.caption("Price must be within N × ATR of the Keltner Channel band to be considered a valid entry.")
        new_anti_chase = st.slider(
            "Anti-Chase Gate (fraction of ATR)",
            min_value=0.1, max_value=2.0,
            value=float(entry.get("anti_chase_atr_pct", {}).get("value", 0.5)),
            step=0.05,
            help=entry.get("anti_chase_atr_pct", {}).get("description", ""),
            key="rule_anti_chase",
        )
        st.caption(f"Default: 0.5 × ATR. Tighter = fewer but higher-quality entries.")

        st.divider()
        st.markdown("**Phase 2 Entry — Confluence Reload**")
        st.caption("Phase 2 allows re-entry after TP1 is hit, requiring multiple confluence filters.")

        new_phase2_enabled = st.toggle(
            "Enable Phase 2 Confluence Entries",
            value=bool(entry.get("phase2_enabled", {}).get("value", True)),
            key="rule_phase2_enabled",
        )
        col1, col2 = st.columns(2)
        with col1:
            new_phase2_filters = st.selectbox(
                "Minimum Confluence Filters Required (out of 3)",
                options=[1, 2, 3],
                index=[1, 2, 3].index(int(entry.get("phase2_min_filters", {}).get("value", 2))),
                help=entry.get("phase2_min_filters", {}).get("description", ""),
                key="rule_phase2_filters",
            )
        with col2:
            new_phase2_roc = st.number_input(
                "ROC Momentum Threshold",
                min_value=0.1, max_value=2.0,
                value=float(entry.get("phase2_roc_threshold", {}).get("value", 0.5)),
                step=0.05,
                help=entry.get("phase2_roc_threshold", {}).get("description", ""),
                key="rule_phase2_roc",
            )

        st.markdown("""
        **Confluence Filter Logic (Phase 2):**
        | Filter | Condition |
        |--------|-----------|
        | 1 — Momentum | ROC confirms direction (above threshold) |
        | 2 — Regime   | Volatility regime is not 'High' |
        | 3 — VWAP     | Price is on the correct side of VWAP |
        """)

        if st.button("💾 Save Entry Rules", key="save_entry"):
            ok = set_group("entry", {
                "anti_chase_atr_pct":   new_anti_chase,
                "phase2_enabled":       str(new_phase2_enabled).lower(),
                "phase2_min_filters":   new_phase2_filters,
                "phase2_roc_threshold": new_phase2_roc,
            }, updated_by=st.session_state.get("username", "ui"))
            st.success("✅ Entry rules saved.") if ok else st.error("❌ Save failed.")

        if st.button("🔄 Reset Entry to Defaults", key="reset_entry"):
            reset_to_defaults("entry")
            st.rerun()

    # ------------------------------------------------------------------ #
    # TAB 3: Take-Profit Multipliers
    # ------------------------------------------------------------------ #
    with tab_tp:
        st.subheader("🎯 Take-Profit Multipliers (× ATR)")
        st.info(
            "Take-profit targets are calculated as **Multiplier × ATR** from the entry price. "
            "The regime (Low / Medium / High volatility) determines which set of multipliers is active. "
            "When the AI trailing layer is enabled, TP2 and Runner multipliers are dynamically adjusted by the CNN model."
        )

        regime_tabs = st.tabs(["🟢 Low Volatility", "🟡 Medium Volatility", "🔴 High Volatility"])
        regime_map = [("tp_low", "Low"), ("tp_med", "Medium"), ("tp_high", "High")]

        for tab, (group, label) in zip(regime_tabs, regime_map):
            with tab:
                tp = rules.get(group, {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_tp1 = st.number_input(
                        "TP1 Multiplier × ATR",
                        min_value=0.5, max_value=5.0,
                        value=float(tp.get("tp1_mult", {}).get("value", 1.0 if group == "tp_low" else 1.2 if group == "tp_med" else 1.5)),
                        step=0.1,
                        help=f"First lot exits at entry ± TP1 × ATR. Default ({label}): {1.0 if group=='tp_low' else 1.2 if group=='tp_med' else 1.5}",
                        key=f"rule_{group}_tp1",
                    )
                with col2:
                    new_tp2 = st.number_input(
                        "TP2 Multiplier × ATR",
                        min_value=0.5, max_value=5.0,
                        value=float(tp.get("tp2_mult", {}).get("value", 1.5 if group == "tp_low" else 1.8 if group == "tp_med" else 2.2)),
                        step=0.1,
                        help=f"Second lot exits at entry ± TP2 × ATR. Default ({label}): {1.5 if group=='tp_low' else 1.8 if group=='tp_med' else 2.2}",
                        key=f"rule_{group}_tp2",
                    )
                with col3:
                    new_runner = st.number_input(
                        "Runner Trail Multiplier × ATR",
                        min_value=0.5, max_value=8.0,
                        value=float(tp.get("runner_trail_mult", {}).get("value", 2.0 if group == "tp_low" else 2.5 if group == "tp_med" else 3.0)),
                        step=0.1,
                        help=f"Runner lot trailing stop distance × ATR. Default ({label}): {2.0 if group=='tp_low' else 2.5 if group=='tp_med' else 3.0}",
                        key=f"rule_{group}_runner",
                    )

                st.caption("TP1 → ratchets stop to breakeven. TP2 → ratchets stop to TP1. Runner → trails with RL agent adjustment.")

                if st.button(f"💾 Save {label} TP Rules", key=f"save_{group}"):
                    ok = set_group(group, {
                        "tp1_mult": new_tp1,
                        "tp2_mult": new_tp2,
                        "runner_trail_mult": new_runner,
                    }, updated_by=st.session_state.get("username", "ui"))
                    st.success(f"✅ {label} TP rules saved.") if ok else st.error("❌ Save failed.")

    # ------------------------------------------------------------------ #
    # TAB 4: Risk & Stops
    # ------------------------------------------------------------------ #
    with tab_risk:
        st.subheader("🛡️ Stop & Risk Rules")
        risk = rules.get("risk", {})

        col1, col2 = st.columns(2)
        with col1:
            new_stop_pct = st.slider(
                "Emergency Stop (% of Credit C)",
                min_value=5, max_value=50,
                value=int(float(risk.get("emergency_stop_pct", {}).get("value", 0.20)) * 100),
                step=1,
                format="%d%%",
                help="Stop is placed at Entry ± (emergency_stop_pct × C). Default: 20% of C.",
                key="rule_stop_pct",
            )
            st.caption(f"At current C={rules.get('spread', {}).get('credit_C', {}).get('value', 10.0):.1f}, stop = {float(rules.get('spread', {}).get('credit_C', {}).get('value', 10.0)) * new_stop_pct / 100:.1f} pts")
        with col2:
            new_jump_exit = st.number_input(
                "Jump Exit Threshold (points)",
                min_value=1.0, max_value=20.0,
                value=float(risk.get("jump_exit_points", {}).get("value", 5.0)),
                step=0.5,
                help="If price moves adversely by this many points within the first bar, exit immediately.",
                key="rule_jump_exit",
            )
            st.caption("Default: 5.0 pts. Protects against gap moves immediately after entry.")

        new_circuit = st.number_input(
            "Circuit Breaker — Daily Loss Limit (USD)",
            min_value=-10000.0, max_value=-100.0,
            value=float(risk.get("circuit_breaker_usd", {}).get("value", -2000.0)),
            step=100.0,
            help="When daily P&L hits this level, all positions are flattened and trading is disabled for the session.",
            key="rule_circuit",
        )
        st.caption("Default: -$2,000. Set to a more negative value to allow larger drawdowns.")
        st.warning(f"⚡ Circuit breaker will trigger at **${new_circuit:,.0f}** daily loss.")

        if st.button("💾 Save Risk Rules", key="save_risk"):
            ok = set_group("risk", {
                "emergency_stop_pct":  new_stop_pct / 100.0,
                "jump_exit_points":    new_jump_exit,
                "circuit_breaker_usd": new_circuit,
            }, updated_by=st.session_state.get("username", "ui"))
            st.success("✅ Risk rules saved.") if ok else st.error("❌ Save failed.")

        if st.button("🔄 Reset Risk to Defaults", key="reset_risk"):
            reset_to_defaults("risk")
            st.rerun()

    # ------------------------------------------------------------------ #
    # TAB 5: Session Rules
    # ------------------------------------------------------------------ #
    with tab_session:
        st.subheader("🕐 Session Rules")
        session = rules.get("session", {})
        st.info("All times are in **Central Time (CT)**.")

        col1, col2 = st.columns(2)
        with col1:
            new_close = st.text_input(
                "Session Close Time (flatten all positions)",
                value=str(session.get("session_close_ct", {}).get("value", "15:55")),
                help="All open positions are flattened at this time. Default: 15:55 CT",
                key="rule_session_close",
            )
            st.caption("Default: 15:55 CT (5 minutes before regular session close)")
        with col2:
            new_reset = st.text_input(
                "Session Reset Time (reset daily counters)",
                value=str(session.get("session_reset_ct", {}).get("value", "17:00")),
                help="Daily P&L counters and circuit breaker reset at this time. Default: 17:00 CT",
                key="rule_session_reset",
            )
            st.caption("Default: 17:00 CT (start of overnight session)")

        if st.button("💾 Save Session Rules", key="save_session"):
            ok = set_group("session", {
                "session_close_ct": new_close,
                "session_reset_ct": new_reset,
            }, updated_by=st.session_state.get("username", "ui"))
            st.success("✅ Session rules saved.") if ok else st.error("❌ Save failed.")

    # ------------------------------------------------------------------ #
    # TAB 6: Indicator Parameters
    # ------------------------------------------------------------------ #
    with tab_indicators:
        st.subheader("📊 Indicator Parameters")
        ind = rules.get("indicators", {})
        st.info("These parameters control the technical indicators used by the entry/exit engine. Changes take effect on the next engine restart.")

        col1, col2, col3 = st.columns(3)
        with col1:
            new_kc_ema = st.number_input("KC EMA Period", min_value=5, max_value=100,
                value=int(ind.get("kc_ema_period", {}).get("value", 20)), step=1, key="rule_kc_ema")
            st.caption("Default: 20")
            new_kc_atr = st.number_input("KC ATR Period", min_value=5, max_value=50,
                value=int(ind.get("kc_atr_period", {}).get("value", 14)), step=1, key="rule_kc_atr")
            st.caption("Default: 14")
        with col2:
            new_kc_mult = st.number_input("KC ATR Multiplier", min_value=0.5, max_value=5.0,
                value=float(ind.get("kc_atr_multiplier", {}).get("value", 2.0)), step=0.1, key="rule_kc_mult")
            st.caption("Default: 2.0")
            new_rsi = st.number_input("RSI Period", min_value=2, max_value=50,
                value=int(ind.get("rsi_period", {}).get("value", 14)), step=1, key="rule_rsi")
            st.caption("Default: 14")
        with col3:
            new_roc = st.number_input("ROC Period", min_value=1, max_value=20,
                value=int(ind.get("roc_period", {}).get("value", 3)), step=1, key="rule_roc")
            st.caption("Default: 3")
            new_atr = st.number_input("ATR Period", min_value=5, max_value=50,
                value=int(ind.get("atr_period", {}).get("value", 14)), step=1, key="rule_atr")
            st.caption("Default: 14")

        if st.button("💾 Save Indicator Parameters", key="save_indicators"):
            ok = set_group("indicators", {
                "kc_ema_period": new_kc_ema, "kc_atr_period": new_kc_atr,
                "kc_atr_multiplier": new_kc_mult, "rsi_period": new_rsi,
                "roc_period": new_roc, "atr_period": new_atr,
            }, updated_by=st.session_state.get("username", "ui"))
            st.success("✅ Indicator parameters saved.") if ok else st.error("❌ Save failed.")

    # ------------------------------------------------------------------ #
    # TAB 7: Regime Detection
    # ------------------------------------------------------------------ #
    with tab_regime:
        st.subheader("🌡️ Regime Detection")
        regime = rules.get("regime", {})
        st.info("The regime detector classifies current volatility as Low / Medium / High based on the ATR percentile over a rolling lookback window.")

        new_lookback = st.number_input(
            "ATR Lookback Window (minutes)",
            min_value=1440, max_value=43200,
            value=int(regime.get("lookback_minutes", {}).get("value", 10080)),
            step=1440,
            help="10080 = 1 week of 1-minute bars. 1440 = 1 day.",
            key="rule_regime_lookback",
        )
        col1, col2 = st.columns(2)
        with col1:
            new_pct_low = st.slider("Low Regime Percentile Cutoff", 10, 45,
                value=int(regime.get("pct_low", {}).get("value", 33)), key="rule_pct_low")
            st.caption(f"ATR below the {new_pct_low}th percentile → Low regime")
        with col2:
            new_pct_high = st.slider("High Regime Percentile Cutoff", 55, 90,
                value=int(regime.get("pct_high", {}).get("value", 66)), key="rule_pct_high")
            st.caption(f"ATR above the {new_pct_high}th percentile → High regime")

        if st.button("💾 Save Regime Rules", key="save_regime"):
            ok = set_group("regime", {
                "lookback_minutes": new_lookback,
                "pct_low": new_pct_low,
                "pct_high": new_pct_high,
            }, updated_by=st.session_state.get("username", "ui"))
            st.success("✅ Regime rules saved.") if ok else st.error("❌ Save failed.")

    # ------------------------------------------------------------------ #
    # TAB 8: AI Confidence Layer
    # ------------------------------------------------------------------ #
    with tab_ai:
        st.subheader("🤖 AI Confidence Layer")
        ai = rules.get("ai", {})
        st.info(
            "The AI layer provides **confidence scores only** — it never executes trades directly. "
            "The rule-based engine remains the sole execution authority. "
            "The AI gate can block a trade (if confidence is below threshold) or suggest dynamic trailing multipliers."
        )

        col1, col2 = st.columns(2)
        with col1:
            new_ai_enabled = st.toggle(
                "🤖 Enable AI Confidence Layer",
                value=bool(ai.get("ai_enabled", {}).get("value", True)),
                key="rule_ai_enabled",
            )
            new_fail_closed = st.toggle(
                "🔒 Fail-Closed (block trades if AI unavailable)",
                value=bool(ai.get("ai_fail_closed", {}).get("value", True)),
                help="If enabled and the AI model fails to load, all entries are blocked. Recommended for production.",
                key="rule_fail_closed",
            )
            new_trail_ai = st.toggle(
                "📈 Use AI for Dynamic Trailing Multipliers",
                value=bool(ai.get("trail_ai_enabled", {}).get("value", True)),
                help="The CNN exit controller dynamically adjusts TP2 and runner trailing stop multipliers based on continuation probability.",
                key="rule_trail_ai",
            )
        with col2:
            st.markdown("**Entry Gate Thresholds by Regime**")
            new_entry_thresh = st.slider(
                "Global Entry Confidence Threshold",
                min_value=0.50, max_value=0.99,
                value=float(ai.get("entry_conf_threshold", {}).get("value", 0.70)),
                step=0.01,
                format="%.2f",
                help="Minimum AI confidence required to allow a trade. Default: 0.70",
                key="rule_entry_thresh",
            )
            st.caption(f"Default: 0.70. Higher = fewer but higher-confidence entries.")

        st.markdown("**Per-Regime Entry Thresholds** (override global threshold)")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_thresh_low = st.number_input("Low Regime Threshold", min_value=0.50, max_value=0.99,
                value=float(ai.get("regime_thresholds_low", {}).get("value", 0.58)),
                step=0.01, format="%.2f", key="rule_thresh_low")
            st.caption("Default: 0.58")
        with col2:
            new_thresh_med = st.number_input("Med Regime Threshold", min_value=0.50, max_value=0.99,
                value=float(ai.get("regime_thresholds_med", {}).get("value", 0.55)),
                step=0.01, format="%.2f", key="rule_thresh_med")
            st.caption("Default: 0.55")
        with col3:
            new_thresh_high = st.number_input("High Regime Threshold", min_value=0.50, max_value=0.99,
                value=float(ai.get("regime_thresholds_high", {}).get("value", 0.52)),
                step=0.01, format="%.2f", key="rule_thresh_high")
            st.caption("Default: 0.52")

        new_exit_thresh = st.slider(
            "Exit Hold Confidence Threshold",
            min_value=0.50, max_value=0.99,
            value=float(ai.get("exit_conf_threshold", {}).get("value", 0.65)),
            step=0.01, format="%.2f",
            help="If the CNN's continuation probability exceeds this, the engine holds the position rather than exiting at TP2.",
            key="rule_exit_thresh",
        )
        st.caption("Default: 0.65. When p_cont_5 > threshold, the engine extends the runner.")

        if st.button("💾 Save AI Layer Rules", key="save_ai"):
            ok = set_group("ai", {
                "ai_enabled":              str(new_ai_enabled).lower(),
                "ai_fail_closed":          str(new_fail_closed).lower(),
                "trail_ai_enabled":        str(new_trail_ai).lower(),
                "entry_conf_threshold":    new_entry_thresh,
                "exit_conf_threshold":     new_exit_thresh,
                "regime_thresholds_low":   new_thresh_low,
                "regime_thresholds_med":   new_thresh_med,
                "regime_thresholds_high":  new_thresh_high,
            }, updated_by=st.session_state.get("username", "ui"))
            st.success("✅ AI layer rules saved.") if ok else st.error("❌ Save failed.")

    # ------------------------------------------------------------------ #
    # TAB 9: RL Trailing Agent
    # ------------------------------------------------------------------ #
    with tab_rl:
        st.subheader("🧠 RL Trailing Stop Agent")
        rl = rules.get("rl", {})
        st.info(
            "The Q-Learning agent adjusts the runner lot's trailing stop in real time. "
            "It learns from trade outcomes to tighten or widen the trail based on regime, unrealized P&L, RSI, and ROC. "
            "These hyperparameters control the learning behavior."
        )

        col1, col2 = st.columns(2)
        with col1:
            new_alpha = st.number_input("Learning Rate (α)", min_value=0.001, max_value=0.5,
                value=float(rl.get("rl_alpha", {}).get("value", 0.1)),
                step=0.005, format="%.3f", key="rule_rl_alpha",
                help="How quickly the agent updates its Q-table. Default: 0.1")
            new_gamma = st.number_input("Discount Factor (γ)", min_value=0.5, max_value=0.999,
                value=float(rl.get("rl_gamma", {}).get("value", 0.95)),
                step=0.005, format="%.3f", key="rule_rl_gamma",
                help="How much the agent values future rewards. Default: 0.95")
        with col2:
            new_epsilon = st.number_input("Exploration Rate (ε)", min_value=0.0, max_value=0.5,
                value=float(rl.get("rl_epsilon", {}).get("value", 0.1)),
                step=0.01, format="%.2f", key="rule_rl_epsilon",
                help="Probability of taking a random action (exploration vs exploitation). Default: 0.1")
            new_lambda = st.number_input("Drawdown Penalty (λ)", min_value=0.0, max_value=2.0,
                value=float(rl.get("rl_lambda_dd", {}).get("value", 0.5)),
                step=0.05, format="%.2f", key="rule_rl_lambda",
                help="Weight of drawdown penalty in the reward function. Higher = more conservative trailing. Default: 0.5")

        st.markdown("""
        **RL Agent Actions:** The agent chooses from three actions each bar:
        | Action | Effect |
        |--------|--------|
        | Tighten (−0.1 × ATR) | Move trailing stop closer to price |
        | Hold (0.0) | Keep trailing stop unchanged |
        | Widen (+0.1 × ATR) | Move trailing stop further from price |
        """)

        if st.button("💾 Save RL Agent Rules", key="save_rl"):
            ok = set_group("rl", {
                "rl_alpha": new_alpha, "rl_gamma": new_gamma,
                "rl_epsilon": new_epsilon, "rl_lambda_dd": new_lambda,
            }, updated_by=st.session_state.get("username", "ui"))
            st.success("✅ RL agent rules saved.") if ok else st.error("❌ Save failed.")

    # ------------------------------------------------------------------ #
    # Footer: Full Reset
    # ------------------------------------------------------------------ #
    st.divider()
    with st.expander("⚠️ Danger Zone — Reset All Rules"):
        st.warning("This will reset **all** strategy rules to factory defaults. This cannot be undone.")
        if st.button("🔄 Reset ALL Rules to Factory Defaults", key="reset_all", type="primary"):
            reset_to_defaults(updated_by=st.session_state.get("username", "ui"))
            st.success("All rules reset to factory defaults.")
            st.rerun()
```

---

## Part 5: AI Confidence Overlay on ES Strategy Dashboard

### Task 5.1 — Add AI Confidence Panel to `page_es()` in `src/dashboard/app.py`

In the `page_es()` function in `app.py`, after the existing regime/position metrics row, add a new AI Confidence row. This reads from the engine's `get_state()["ai_confidence"]` dict (added in Task 3.5) and from the confidence server's `/health` endpoint.

Add this block immediately after the regime/spread metrics section in `page_es()`:

```python
# --- AI Confidence Overlay ---
st.subheader("🤖 AI Confidence Layer")
ai_state = engine_state.get("ai_confidence", {})
conf_col1, conf_col2, conf_col3, conf_col4 = st.columns(4)

with conf_col1:
    entry_conf = ai_state.get("entry_conf")
    if entry_conf is not None:
        color = "green" if entry_conf >= 0.70 else "orange" if entry_conf >= 0.55 else "red"
        st.metric("Entry Confidence", f"{entry_conf:.1%}",
                  delta="Allow" if entry_conf >= 0.70 else "Block",
                  delta_color="normal" if entry_conf >= 0.70 else "inverse")
        st.progress(entry_conf, text=f"Gate: {entry_conf:.1%}")
    else:
        st.metric("Entry Confidence", "—")
        st.caption("No signal yet")

with conf_col2:
    exit_conf = ai_state.get("exit_conf")
    if exit_conf is not None:
        st.metric("Continuation Probability", f"{1.0 - exit_conf:.1%}",
                  delta="Hold" if (1.0 - exit_conf) >= 0.65 else "Exit",
                  delta_color="normal" if (1.0 - exit_conf) >= 0.65 else "inverse")
        st.progress(1.0 - exit_conf, text=f"p_cont: {1.0 - exit_conf:.1%}")
    else:
        st.metric("Continuation Probability", "—")
        st.caption("No open position")

with conf_col3:
    trail_active = ai_state.get("trail_ai_active", False)
    st.metric("AI Trailing", "Active ✅" if trail_active else "Disabled ⛔")
    st.caption("CNN dynamically adjusts TP2 & runner multipliers")

with conf_col4:
    # Check confidence server health
    try:
        import requests as _req
        health = _req.get("http://localhost:8100/health", timeout=2).json()
        entry_loaded = health.get("entry_gate_loaded", False)
        exit_loaded = health.get("exit_ctrl_loaded", False)
        st.metric("AI Models", "Online ✅" if (entry_loaded and exit_loaded) else "Partial ⚠️")
        st.caption(f"Entry gate: {'✅' if entry_loaded else '❌'} | Exit CNN: {'✅' if exit_loaded else '❌'}")
    except Exception:
        st.metric("AI Models", "Offline ❌")
        st.caption("Confidence server not running on port 8100")
```

### Task 5.2 — Add "Reload Rules" button to ES Strategy page

At the top of `page_es()`, after the title, add:

```python
col_title, col_reload = st.columns([4, 1])
with col_reload:
    if st.button("🔄 Reload Rules", help="Hot-reload strategy rules from database without restarting"):
        # Signal the runner to reload rules on next bar
        import json, os
        reload_flag = {"reload_rules": True, "requested_at": datetime.now().isoformat()}
        os.makedirs("data", exist_ok=True)
        with open("data/reload_rules_flag.json", "w") as f:
            json.dump(reload_flag, f)
        st.success("Rules reload requested. Takes effect on next bar.")
```

In `src/es_strategy/runner.py`, in the `_process_bar` method, add a check for this flag:

```python
def _process_bar(self, bar: dict, indicators: dict) -> list:
    # Check for hot-reload request from dashboard
    reload_flag_path = "data/reload_rules_flag.json"
    if os.path.exists(reload_flag_path):
        try:
            self.engine.reload_rules()
            os.remove(reload_flag_path)
            logger.info("Strategy rules hot-reloaded from database")
        except Exception as e:
            logger.warning(f"Rules reload failed: {e}")
    # ... rest of existing _process_bar logic
```

---

## Part 6: Register the Rules Page in Navigation

### Task 6.1 — Import and register `page_rules` in `src/dashboard/app.py`

At the top of `app.py`, add the import:

```python
from src.dashboard.rules_app import page_rules
```

In the `_pages` dictionary inside the `st.navigation` block (around line 3207), add the Rules page under the "Markets" group, between "ES Strategy" and "Tune & Backtest":

```python
_pages = {
    "Markets": [
        st.Page(page_spy,          title="SPY Predictor",    icon=":material/query_stats:"),
        st.Page(page_performance,  title="Performance",      icon=":material/verified:"),
        st.Page(page_es,           title="ES Strategy",      icon=":material/candlestick_chart:"),
        st.Page(page_rules,        title="Rules",            icon=":material/rule:"),          # NEW
        st.Page(page_tuning,       title="Tune & Backtest",  icon=":material/tune:"),
        st.Page(page_whatif,       title="What-If Analysis", icon=":material/science:"),
        st.Page(page_single_stock, title="Single-Stock",     icon=":material/search:"),
        st.Page(page_quant_agent,  title="Quant Agent",      icon=":material/smart_toy:"),
    ],
    "Operations": [
        st.Page(page_monitoring,   title="Monitoring",       icon=":material/monitor_heart:"),
        st.Page(page_grafana,      title="Grafana Dashboards",icon=":material/dashboard:"),
        st.Page(page_admin,        title="Admin",            icon=":material/settings:"),
    ],
}
```

---

## Part 7: Remove YAML Config Tab from Admin Page

### Task 7.1 — Replace `_admin_config_tab()` with a redirect

The existing Admin → Configuration tab exposes raw YAML editing. Replace its content with a clean redirect to the new Rules page and a read-only system summary. This ensures no user ever edits YAML directly.

Replace the body of `_admin_config_tab()` with:

```python
def _admin_config_tab():
    st.subheader("Configuration")
    st.info(
        "Strategy parameters are now managed through the dedicated **Rules** page. "
        "Use the sidebar to navigate to ⚙️ Rules to view and edit all strategy parameters with a full UI."
    )
    if st.button("Go to Rules Page →", key="goto_rules"):
        st.switch_page("rules")

    st.divider()
    st.markdown("**System Configuration Summary (read-only)**")
    config = _load_config()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM Model",        config.get("llm", {}).get("model", "—"))
    col2.metric("XGB Lookback",     f"{config.get('xgboost', {}).get('lookback_days', '—')} days")
    col3.metric("Cloud Sync",       "On" if config.get("sync", {}).get("enabled") else "Off")
    col4.metric("DB Mode",          "PostgreSQL" if config.get("database", {}).get("postgres") else "SQLite")
    st.caption("To change system-level settings (LLM model, API keys, database), contact the system administrator.")
```

---

## Part 8: `src/strategy/__init__.py`

Create the `src/strategy/` package by adding an empty `__init__.py`:

```python
# src/strategy/__init__.py
```

---

## Summary of All Files to Create or Modify

| Action | File | Description |
| :--- | :--- | :--- |
| **Modify** | `src/data/init_db.py` | Add `strategy_rules` table and `seed_strategy_rules()` function |
| **Create** | `src/strategy/__init__.py` | New package init |
| **Create** | `src/strategy/rules_store.py` | DB-backed rules read/write module |
| **Modify** | `src/es_strategy/engine.py` | Load all params from `rules_store`, wire AI exit/trail confidence, add `reload_rules()` |
| **Create** | `src/dashboard/rules_app.py` | Full Rules Management UI page (9 tabs, all pre-populated) |
| **Modify** | `src/dashboard/app.py` | Import `page_rules`, add to navigation, add AI overlay to `page_es()`, add Reload Rules button, replace `_admin_config_tab()` |
| **Modify** | `src/es_strategy/runner.py` | Check for hot-reload flag in `_process_bar()` |

---

## Acceptance Criteria

After implementation, the following must be true:

1. **Rules page exists** in the sidebar under "Markets" with icon `:material/rule:`.
2. **All 9 rule groups** are visible as tabs with pre-populated default values matching the hardcoded values in the original `engine.py`.
3. **Saving any rule** persists it to the `strategy_rules` database table and does not touch `config.yaml`.
4. **Resetting to defaults** restores the exact values listed in `seed_strategy_rules()`.
5. **The engine reads rules from the DB** at startup and on hot-reload, not from `config.yaml`.
6. **AI Confidence overlay** is visible on the ES Strategy page showing entry confidence, continuation probability, and model health.
7. **AI exit/trailing confidence** from the CNN exit controller is used to dynamically set TP2 and runner trail multipliers when `trail_ai_enabled = true`.
8. **Admin → Configuration tab** shows a read-only summary and a link to the Rules page — no raw YAML editing.
9. **"Reload Rules" button** on the ES Strategy page triggers a hot-reload without restarting the engine.
