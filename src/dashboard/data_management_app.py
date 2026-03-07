"""Data Management — database explorer and ad-hoc pipeline actions.

Extracted from the Admin console to provide a focused, single-purpose
page for all data-related administrative tasks.
"""

import streamlit as st
from src.dashboard.theme import page_header


def page_data_management():
    """Renders the Data Management page."""
    from src.dashboard.chatbot_widget import render_chatbot_widget
    render_chatbot_widget(page_key="data_mgmt", page_title="Data Management")
    st.markdown(page_header("🗃️ Data Management"), unsafe_allow_html=True)
    st.caption(
        "Explore the database, run ad-hoc pipeline steps, and manage data operations. "
        "For system health and user management, see the System Management page."
    )

    tab_actions, tab_db = st.tabs(["▶️ Pipeline Actions", "🗃️ Database Explorer"])

    with tab_actions:
        from src.dashboard.app import _admin_actions_tab
        _admin_actions_tab()

    with tab_db:
        from src.dashboard.app import _admin_db_tab
        _admin_db_tab()
