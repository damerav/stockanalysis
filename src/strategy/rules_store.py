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
    """Update a single rule value."""
    try:
        r = _router()
        r.execute(
            "UPDATE strategy_rules SET rule_value=?, updated_at=?, updated_by=? "
            "WHERE rule_group=? AND rule_key=?",
            (str(value), datetime.now().isoformat(), updated_by, group, key),
        )
        r.close()
        logger.info(f"Rule updated: {group}.{key} = {value} by {updated_by}")
        return True
    except Exception as e:
        logger.error(f"rules_store.set_rule({group}.{key}): {e}")
        return False


def set_group(group: str, updates: dict, updated_by: str = "ui") -> bool:
    """Update multiple rules in a group."""
    return all(set_rule(group, k, v, updated_by) for k, v in updates.items())


def reset_to_defaults(group: Optional[str] = None, updated_by: str = "ui") -> bool:
    """Reset rules to factory defaults by deleting and re-seeding."""
    try:
        r = _router()
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
