"""System Management — users, logs, configuration, and system health.

Extracted from the Admin console to provide a focused, single-purpose
page for all system-related administrative tasks.
"""

import streamlit as st
from src.dashboard.theme import page_header


def page_system_management():
    """Renders the System Management page."""
    st.markdown(page_header("⚙️ System Management"), unsafe_allow_html=True)
    st.caption(
        "Manage users, view system logs, review configuration, and check system health. "
        "For pipeline actions and database operations, see the Data Management page."
    )

    tab_status, tab_users, tab_config, tab_logs = st.tabs([
        "ℹ️ System Status", "👤 Users", "📝 Configuration", "📜 Logs"
    ])

    with tab_status:
        from src.dashboard.app import _admin_status_tab
        _admin_status_tab()

    with tab_users:
        from src.dashboard.app import _admin_users_tab
        _admin_users_tab()

    with tab_config:
        from src.dashboard.app import _admin_config_tab
        _admin_config_tab()

    with tab_logs:
        from src.dashboard.app import _admin_logs_tab
        _admin_logs_tab()
