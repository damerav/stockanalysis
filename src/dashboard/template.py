"""Shared UI components for the Stock Analysis Platform design system.

All components use CSS classes defined in style.css which reference
CSS custom properties (tokens) for automatic dark/light theming.

Usage:
    from src.dashboard.template import page_header, kpi_card, signal_banner, stale_banner
"""

import streamlit as st


def page_header(title: str):
    """Render a compact page header."""
    st.markdown(
        f'<p style="margin:0;padding:4px 0;color:var(--color-text-primary);'
        f'font-size:1.1rem;font-weight:600;">{title}</p>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, color: str = "", unit: str = "", sub: str = "") -> str:
    """Return HTML for a themed KPI card.

    Args:
        label: Card label (uppercase)
        value: Main value to display
        color: Optional CSS color override for the value (e.g. 'var(--color-positive-500)')
        unit: Optional unit suffix
        sub: Optional subtitle text
    """
    val_style = f"color:{color};" if color else "color:var(--color-text-primary);"
    unit_html = f' <span style="font-size:0.9rem;">{unit}</span>' if unit else ""
    sub_html = (
        f'<div class="kpi-sub">{sub}</div>' if sub else ""
    )
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="{val_style}">{value}{unit_html}</div>'
        f'{sub_html}</div>'
    )


def signal_banner(label: str, sentiment: str):
    """Render a signal banner (positive/negative/neutral)."""
    sentiment_class = sentiment.lower()
    st.markdown(
        f'<div class="signal-banner {sentiment_class}">{label}</div>',
        unsafe_allow_html=True,
    )


def stale_banner(message: str):
    """Render a stale-data warning banner."""
    st.markdown(
        f'<div class="stale-banner">⚠️ {message}</div>',
        unsafe_allow_html=True,
    )


def badge_html(label: str, online: bool) -> str:
    """Return HTML for a status badge (ONLINE/OFFLINE)."""
    color = "var(--color-positive-500)" if online else "var(--color-negative-500)"
    text = "ONLINE" if online else "OFFLINE"
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color};">{text}</div>'
        f'</div>'
    )
