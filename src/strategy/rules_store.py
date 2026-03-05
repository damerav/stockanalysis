"""src/strategy/rules_store.py — DB-backed strategy parameter store.

All reads/writes go through DbRouter (PostgreSQL primary, SQLite fallback).
SQL uses ? placeholders — the router converts automatically.
"""

import logging
import yaml
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _router():
    from src.data.db_router import DbRouter
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return DbRouter(cfg)


def _cast(value: str, vtype: str) -> Any:
    """Cast a string value to its declared type."""
    try:
        if vtype == "int":
            return int(float(value))
        if vtype == "float":
            return float(value)
        if vtype == "bool":
            return str(value).lower() in ("true", "1", "yes")
        return value
    except Exception:
        return value


def get_rule(group: str, key: str, default: Any = None) -> Any:
    """Get a single rule value, cast to its declared type."""
    try:
        r = _router()
        df = r.query(
            "SELECT rule_value, value_type FROM strategy_rules "
            "WHERE rule_group=? AND rule_key=?",
            (group, key),
        )
        r.close()
        if df.empty:
            return default
        return _cast(df.iloc[0]["rule_value"], df.iloc[0]["value_type"])
    except Exception as e:
        logger.warning(f"rules_store.get_rule({group}.{key}): {e}")
        return default


def get_group(group: str) -> dict:
    """Get all rules in a group as {key: typed_value}."""
    try:
        r = _router()
        df = r.query(
            "SELECT rule_key, rule_value, value_type FROM strategy_rules "
            "WHERE rule_group=?",
            (group,),
        )
        r.close()
        return {
            row["rule_key"]: _cast(row["rule_value"], row["value_type"])
            for _, row in df.iterrows()
        }
    except Exception as e:
        logger.warning(f"rules_store.get_group({group}): {e}")
        return {}


def get_all_rules() -> dict:
    """Get all rules grouped by rule_group with full metadata."""
    try:
        r = _router()
        df = r.query(
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
        logger.warning(f"rules_store.get_all_rules(): {e}")
        return {}


def set_rule(group: str, key: str, value: Any, updated_by: str = "ui") -> bool:
    """Update a single rule value. Logs the old value to history for rollback."""
    try:
        r = _router()
        # Read old value for history
        df = r.query(
            "SELECT rule_value FROM strategy_rules WHERE rule_group=? AND rule_key=?",
            (group, key),
        )
        old_value = df.iloc[0]["rule_value"] if not df.empty else None

        now = datetime.now().isoformat()
        r.execute(
            "UPDATE strategy_rules SET rule_value=?, updated_at=?, updated_by=? "
            "WHERE rule_group=? AND rule_key=?",
            (str(value), now, updated_by, group, key),
        )
        # Log to history
        if old_value is not None and str(old_value) != str(value):
            r.execute(
                "INSERT INTO strategy_rules_history "
                "(rule_group, rule_key, old_value, new_value, changed_at, changed_by) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (group, key, str(old_value), str(value), now, updated_by),
            )
        r.close()
        logger.info(f"Rule updated: {group}.{key} = {value} (was {old_value}) by {updated_by}")
        return True
    except Exception as e:
        logger.error(f"rules_store.set_rule({group}.{key}): {e}")
        return False


def set_group(group: str, updates: dict, updated_by: str = "ui") -> bool:
    """Update multiple rules in a group."""
    return all(set_rule(group, k, v, updated_by) for k, v in updates.items())


def reset_to_defaults(group: Optional[str] = None, updated_by: str = "ui") -> bool:
    """Reset rules to factory defaults by deleting and re-seeding.
    Logs all current values to history before resetting."""
    try:
        r = _router()
        # Log current values to history before reset
        if group:
            df = r.query(
                "SELECT rule_group, rule_key, rule_value FROM strategy_rules "
                "WHERE rule_group=?", (group,),
            )
        else:
            df = r.query("SELECT rule_group, rule_key, rule_value FROM strategy_rules")
        now = datetime.now().isoformat()
        for _, row in df.iterrows():
            r.execute(
                "INSERT INTO strategy_rules_history "
                "(rule_group, rule_key, old_value, new_value, changed_at, changed_by) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (row["rule_group"], row["rule_key"], row["rule_value"],
                 "RESET_TO_DEFAULT", now, updated_by),
            )
        # Delete and re-seed
        if group:
            r.execute("DELETE FROM strategy_rules WHERE rule_group=?", (group,))
        else:
            r.execute("DELETE FROM strategy_rules")
        r.close()
        # Re-seed
        from src.data.init_db import init_db
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        init_db(cfg)
        logger.info(f"Rules reset to defaults (group={group or 'ALL'}) by {updated_by}")
        return True
    except Exception as e:
        logger.error(f"rules_store.reset_to_defaults: {e}")
        return False


def get_history(group: str = None, key: str = None, limit: int = 50) -> list[dict]:
    """Get change history for rules. Newest first."""
    try:
        r = _router()
        if group and key:
            df = r.query(
                "SELECT * FROM strategy_rules_history "
                "WHERE rule_group=? AND rule_key=? ORDER BY changed_at DESC",
                (group, key),
            )
        elif group:
            df = r.query(
                "SELECT * FROM strategy_rules_history "
                "WHERE rule_group=? ORDER BY changed_at DESC",
                (group,),
            )
        else:
            df = r.query(
                "SELECT * FROM strategy_rules_history ORDER BY changed_at DESC"
            )
        r.close()
        if df.empty:
            return []
        return df.head(limit).to_dict("records")
    except Exception as e:
        logger.warning(f"rules_store.get_history: {e}")
        return []


def revert_rule(group: str, key: str, updated_by: str = "ui") -> bool:
    """Revert a single rule to its previous value from history."""
    try:
        history = get_history(group, key, limit=1)
        if not history:
            logger.warning(f"No history for {group}.{key} — nothing to revert")
            return False
        entry = history[0]
        old_val = entry.get("old_value")
        if old_val is None or old_val == "RESET_TO_DEFAULT":
            logger.warning(f"Cannot revert {group}.{key} — previous was a reset")
            return False
        return set_rule(group, key, old_val, updated_by=f"revert:{updated_by}")
    except Exception as e:
        logger.error(f"rules_store.revert_rule({group}.{key}): {e}")
        return False


def revert_group(group: str, updated_by: str = "ui") -> int:
    """Revert all rules in a group to their most recent previous values.
    Returns count of rules reverted."""
    try:
        r = _router()
        # Get the latest history entry per key in this group
        df = r.query(
            "SELECT DISTINCT ON (rule_key) rule_key, old_value, changed_at "
            "FROM strategy_rules_history WHERE rule_group=? "
            "ORDER BY rule_key, changed_at DESC",
            (group,),
        )
        r.close()
        if df.empty:
            # Fallback for SQLite (no DISTINCT ON)
            r2 = _router()
            df = r2.query(
                "SELECT rule_key, old_value, changed_at FROM strategy_rules_history "
                "WHERE rule_group=? ORDER BY changed_at DESC",
                (group,),
            )
            r2.close()
            if df.empty:
                return 0
            # Deduplicate: keep first (most recent) per key
            df = df.drop_duplicates(subset=["rule_key"], keep="first")

        count = 0
        for _, row in df.iterrows():
            old_val = row["old_value"]
            if old_val and old_val != "RESET_TO_DEFAULT":
                if set_rule(group, row["rule_key"], old_val,
                            updated_by=f"revert:{updated_by}"):
                    count += 1
        return count
    except Exception as e:
        logger.error(f"rules_store.revert_group({group}): {e}")
        return 0
