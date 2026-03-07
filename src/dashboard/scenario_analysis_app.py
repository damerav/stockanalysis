"""Scenario Analysis — consolidated What-If simulation engine.

Wraps the existing What-If engine for ES Strategy and SPY Predictor
under a cleaner, more descriptive page name.
"""

import streamlit as st
from src.dashboard.theme import page_header


def page_scenario_analysis():
    """Renders the Scenario Analysis page."""
    from src.dashboard.chatbot_widget import render_chatbot_widget
    render_chatbot_widget(page_key="scenario", page_title="Scenario Analysis")
    st.markdown(page_header("🔬 Scenario Analysis"), unsafe_allow_html=True)
    st.caption(
        "Simulate how changes to market conditions, model features, or strategy rules "
        "would affect outcomes. Use the ES Strategy tab for futures simulations and "
        "the SPY Predictor tab for prediction sensitivity analysis."
    )

    # Reuse the existing What-If tab functions from app.py
    from src.dashboard.app import get_whatif_engine, _whatif_es_tab, _whatif_spy_tab

    engine = get_whatif_engine()
    tab_es, tab_spy = st.tabs(["📈 ES Strategy", "🔮 SPY Predictor"])

    with tab_es:
        _whatif_es_tab(engine)
    with tab_spy:
        _whatif_spy_tab(engine)
