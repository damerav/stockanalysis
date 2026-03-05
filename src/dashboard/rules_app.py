"""Rules Management Dashboard — Edit all ES strategy parameters from the DB."""

import streamlit as st
from src.dashboard.theme import get_colors, page_header
from src.strategy import rules_store as rs

# Display labels for each rule group tab
_GROUP_LABELS = {
    "spread": "📐 Spread",
    "sizing": "📦 Sizing",
    "entry": "🚪 Entry",
    "tp_low": "🎯 TP Low",
    "tp_med": "🎯 TP Med",
    "tp_high": "🎯 TP High",
    "risk": "🛡️ Risk",
    "session": "🕐 Session",
    "indicators": "📊 Indicators",
    "regime": "🌡️ Regime",
    "ai": "🤖 AI",
    "rl": "🧠 RL",
}


def page_rules():
    c = get_colors()
    st.markdown(page_header("⚙️ Strategy Rules"), unsafe_allow_html=True)

    all_rules = rs.get_all_rules()
    if not all_rules:
        st.warning("No rules found in database. Run init_db to seed defaults.")
        if st.button("Seed Defaults Now"):
            rs.reset_to_defaults(updated_by="admin_ui")
            st.rerun()
        return

    # Group tabs
    groups = list(all_rules.keys())
    tab_labels = [_GROUP_LABELS.get(g, g) for g in groups]
    tabs = st.tabs(tab_labels)

    for tab, group in zip(tabs, groups):
        with tab:
            rules = all_rules[group]
            _render_group(group, rules, c)

    # Global reset
    st.divider()
    col_reset, col_spacer = st.columns([1, 4])
    with col_reset:
        if st.button("🔄 Reset ALL to Defaults", type="secondary", key="reset_all"):
            rs.reset_to_defaults(updated_by="admin_ui")
            st.success("All rules reset to factory defaults.")
            st.rerun()

    # ── Backtest pending changes ──
    st.divider()
    st.subheader("📊 Backtest Rule Changes")
    st.caption("Compare your pending edits against current live rules before saving.")

    # Collect all pending changes across all groups
    pending = _collect_pending_changes(all_rules)

    if not pending:
        st.info("No pending changes to backtest. Edit rules above, then come here.")
    else:
        st.markdown(f"**{len(pending)} pending change(s):**")
        for label in pending.values():
            st.markdown(f"- `{label['display']}`")

        if st.button("🧪 Run Backtest: Current vs Proposed", key="run_rules_bt",
                      type="primary"):
            _run_rules_backtest(pending)


def _collect_pending_changes(all_rules: dict) -> dict:
    """Scan session_state for widget values that differ from DB values."""
    changes = {}
    for group, rules in all_rules.items():
        for key, meta in rules.items():
            widget_key = f"rule_{group}_{key}"
            if widget_key not in st.session_state:
                continue
            widget_val = st.session_state[widget_key]
            db_val = meta["value"]
            vtype = meta["type"]

            changed = False
            if vtype == "float":
                changed = abs(float(widget_val) - float(db_val)) > 1e-8
            elif vtype == "int":
                changed = int(widget_val) != int(db_val)
            elif vtype == "bool":
                changed = bool(widget_val) != bool(db_val)
            else:
                changed = str(widget_val) != str(db_val)

            if changed:
                flat_key = f"{group}.{key}"
                changes[flat_key] = {
                    "value": widget_val,
                    "old": db_val,
                    "display": f"{group}.{key}: {db_val} → {widget_val}",
                }
    return changes


def _run_rules_backtest(pending: dict):
    """Execute the What-If rules backtest and render results."""
    from src.whatif.engine import WhatIfEngine

    overrides = {k: v["value"] for k, v in pending.items()}

    with st.spinner("Running backtests (current rules vs proposed)..."):
        engine = WhatIfEngine()
        result = engine.es_rules_backtest(overrides)

    if "error" in result:
        st.error(f"Backtest failed: {result['error']}")
        return

    baseline = result["baseline"]
    proposed = result["proposed"]
    diff = result["diff"]

    # Verdict banner
    verdict = diff["verdict"]
    if verdict == "IMPROVED":
        st.success(f"✅ Proposed rules IMPROVED P&L by ${diff['pnl_delta']:+,.0f} "
                    f"({diff['pnl_pct_change']:+.1f}%)")
    elif verdict == "DEGRADED":
        st.error(f"⚠️ Proposed rules DEGRADED P&L by ${diff['pnl_delta']:+,.0f} "
                  f"({diff['pnl_pct_change']:+.1f}%)")
    else:
        st.info("➖ No P&L difference between current and proposed rules.")

    # Side-by-side metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current P&L", f"${baseline['total_pnl']:+,.0f}",
                   help=f"{baseline['trades']} trades")
    with col2:
        st.metric("Proposed P&L", f"${proposed['total_pnl']:+,.0f}",
                   delta=f"${diff['pnl_delta']:+,.0f}",
                   help=f"{proposed['trades']} trades")
    with col3:
        st.metric("Trade Count Δ", f"{diff['trade_delta']:+d}",
                   help=f"Current: {baseline['trades']}, Proposed: {proposed['trades']}")


def _render_group(group: str, rules: dict, c: dict):
    """Render editable fields for one rule group."""
    changes = {}
    for key, meta in rules.items():
        val = meta["value"]
        vtype = meta["type"]
        desc = meta.get("description", "")
        min_v = meta.get("min")
        max_v = meta.get("max")
        updated = meta.get("updated_at", "")
        updated_by = meta.get("updated_by", "")

        label = f"{key}"
        help_text = desc
        if updated:
            help_text += f" (last: {updated[:16]} by {updated_by})"

        widget_key = f"rule_{group}_{key}"

        if vtype == "float":
            min_f = float(min_v) if min_v else None
            max_f = float(max_v) if max_v else None
            new_val = st.number_input(
                label, value=float(val), min_value=min_f, max_value=max_f,
                step=0.01, format="%.4f", key=widget_key, help=help_text,
            )
            if abs(new_val - float(val)) > 1e-8:
                changes[key] = new_val

        elif vtype == "int":
            min_i = int(float(min_v)) if min_v else None
            max_i = int(float(max_v)) if max_v else None
            new_val = st.number_input(
                label, value=int(val), min_value=min_i, max_value=max_i,
                step=1, key=widget_key, help=help_text,
            )
            if new_val != int(val):
                changes[key] = new_val

        elif vtype == "bool":
            new_val = st.checkbox(label, value=bool(val), key=widget_key, help=help_text)
            if new_val != bool(val):
                changes[key] = str(new_val).lower()

        elif vtype == "time":
            new_val = st.text_input(label, value=str(val), key=widget_key, help=help_text)
            if new_val != str(val):
                changes[key] = new_val

        else:
            new_val = st.text_input(label, value=str(val), key=widget_key, help=help_text)
            if new_val != str(val):
                changes[key] = new_val

    # Save + Reset + Revert buttons for this group
    col_save, col_reset, col_revert, _ = st.columns([1, 1, 1, 2])
    with col_save:
        if st.button(f"💾 Save {group}", key=f"save_{group}"):
            if changes:
                ok = rs.set_group(group, changes, updated_by="admin_ui")
                if ok:
                    st.success(f"Saved {len(changes)} change(s) to {group}.")
                    # Write hot-reload flag so runner picks up changes
                    import os
                    try:
                        with open(os.path.join("data", ".reload_rules"), "w") as f:
                            f.write("1")
                    except Exception:
                        pass
                    st.rerun()
                else:
                    st.error("Save failed — check logs.")
            else:
                st.info("No changes to save.")
    with col_reset:
        if st.button(f"🔄 Reset {group}", key=f"reset_{group}"):
            rs.reset_to_defaults(group=group, updated_by="admin_ui")
            st.success(f"{group} reset to defaults.")
            st.rerun()
    with col_revert:
        if st.button(f"⏪ Revert {group}", key=f"revert_{group}"):
            count = rs.revert_group(group, updated_by="admin_ui")
            if count > 0:
                st.success(f"Reverted {count} rule(s) in {group} to previous values.")
                import os
                try:
                    with open(os.path.join("data", ".reload_rules"), "w") as f:
                        f.write("1")
                except Exception:
                    pass
                st.rerun()
            else:
                st.warning(f"No history found for {group} — nothing to revert.")

    # Show recent history for this group
    with st.expander(f"📜 Change History ({group})", expanded=False):
        history = rs.get_history(group=group, limit=20)
        if history:
            for entry in history:
                old_v = entry.get("old_value", "?")
                new_v = entry.get("new_value", "?")
                at = (entry.get("changed_at") or "")[:16]
                by = entry.get("changed_by", "?")
                key_name = entry.get("rule_key", "?")
                if new_v == "RESET_TO_DEFAULT":
                    st.caption(f"🔄 `{key_name}` reset to default (was {old_v}) — {at} by {by}")
                else:
                    st.caption(f"`{key_name}`: {old_v} → {new_v} — {at} by {by}")
        else:
            st.caption("No changes recorded yet.")
