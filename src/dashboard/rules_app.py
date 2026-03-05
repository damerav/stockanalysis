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

    # Save + Reset buttons for this group
    col_save, col_reset, _ = st.columns([1, 1, 3])
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
