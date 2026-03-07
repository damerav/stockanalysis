"""Reports — Download Center for PDF and CSV exports.

Provides 9 report types with flexible date range selection:
  7 days / 30 days / 90 days / Year-to-Date / Custom
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import streamlit as st

from src.dashboard.theme import page_header

logger = logging.getLogger(__name__)

REPORT_CATALOGUE = [
    ("Prediction History",       "prediction_history",       True, True),
    ("Model Performance",        "model_performance",        True, True),
    ("Feature Importance",       "feature_importance",       True, True),
    ("ES Strategy P&L",          "es_strategy_pnl",          True, True),
    ("Market Data Export",       "market_data_export",       True, True),
    ("Options Analytics Export", "options_analytics_export", True, True),
    ("News & Sentiment Export",  "news_sentiment_export",    True, True),
    ("Macro Indicators Export",  "macro_indicators_export",  True, True),
    ("Platform Health Report",   "platform_health",          True, True),
]

DATE_PRESETS = {
    "Last 7 Days":  timedelta(days=7),
    "Last 30 Days": timedelta(days=30),
    "Last 90 Days": timedelta(days=90),
    "Year to Date": "ytd",
    "Custom Range": "custom",
}


def page_reports():
    """Renders the Reports download page."""
    from src.dashboard.chatbot_widget import render_chatbot_widget
    render_chatbot_widget(page_key="reports", page_title="Reports")

    st.markdown(page_header("📥 Reports & Downloads"), unsafe_allow_html=True)
    st.caption(
        "Generate and download platform reports as PDF (light-themed, with charts) "
        "or CSV. All data is sourced directly from your local database."
    )

    # Lazy-init report generator
    if "report_generator" not in st.session_state:
        from src.llm.report_generator import ReportGenerator
        st.session_state.report_generator = ReportGenerator()
    gen = st.session_state.report_generator

    st.divider()

    report_names = [r[0] for r in REPORT_CATALOGUE]
    selected_name = st.selectbox("Report Type", report_names)
    selected = next(r for r in REPORT_CATALOGUE if r[0] == selected_name)
    _, method_name, supports_pdf, supports_csv = selected

    st.markdown("**Date Range**")
    preset = st.radio(
        "Preset", list(DATE_PRESETS.keys()), horizontal=True, label_visibility="collapsed"
    )

    today = date.today()
    if DATE_PRESETS[preset] == "ytd":
        start_date = today.replace(month=1, day=1)
        end_date = today
    elif DATE_PRESETS[preset] == "custom":
        col_s, col_e = st.columns(2)
        start_date = col_s.date_input("Start date", today - timedelta(days=30))
        end_date = col_e.date_input("End date", today)
    else:
        start_date = today - DATE_PRESETS[preset]
        end_date = today

    st.caption(f"Selected range: **{start_date}** to **{end_date}**")
    st.divider()

    col_pdf, col_csv = st.columns(2)

    if supports_pdf:
        with col_pdf:
            if st.button("Generate PDF", use_container_width=True, key="btn_pdf"):
                with st.spinner(f"Generating {selected_name} PDF..."):
                    try:
                        data = getattr(gen, method_name)(start_date, end_date, fmt="pdf")
                        st.download_button(
                            label="Download PDF",
                            data=data,
                            file_name=f"{method_name}_{start_date}_{end_date}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")

    if supports_csv:
        with col_csv:
            if st.button("Generate CSV", use_container_width=True, key="btn_csv"):
                with st.spinner(f"Generating {selected_name} CSV..."):
                    try:
                        data = getattr(gen, method_name)(start_date, end_date, fmt="csv")
                        st.download_button(
                            label="Download CSV",
                            data=data,
                            file_name=f"{method_name}_{start_date}_{end_date}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.error(f"CSV generation failed: {e}")
