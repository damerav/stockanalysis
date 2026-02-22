"""Unified Dashboard — All dashboards on a single port.

Usage:
    streamlit run src/dashboard/app.py --server.port 8501 --server.headless true
"""

import os
import sys
import json
import time
import logging
import urllib.parse
import requests
import numpy as np
import yaml
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.auth.google_oauth import (
    is_authenticated,
    get_user,
    get_session_token,
    handle_oauth_callback,
    render_login_page,
    logout,
)
from src.dashboard.monitoring import page_monitoring

logger = logging.getLogger(__name__)

# --- Mode Detection ---
RELAY_URL = os.environ.get("RELAY_URL", "")
IS_CLOUD = bool(RELAY_URL)
DATA_DIR = "./data"

st.set_page_config(page_title="Stock Analysis", layout="wide", page_icon="📊")

# --- Global dark mode CSS (supplements .streamlit/config.toml dark theme) ---
st.markdown(
    """<style>
    /* Sidebar accent */
    .stSidebar { background-color: #111317 !important; }
    .stDivider { border-color: #2c3035 !important; }

    /* Dropdown menus */
    [data-baseweb="popover"] { background-color: #1f2329 !important; }
    [data-baseweb="popover"] li:hover { background-color: #2c3035 !important; }
    [role="listbox"] { background-color: #1f2329 !important; }
    [role="option"]:hover { background-color: #2c3035 !important; }

    /* Form containers */
    [data-testid="stForm"] {
        background-color: #1f2329 !important;
        border: 1px solid #3a3f47 !important;
        border-radius: 10px;
        padding: 24px !important;
    }

    /* Text inputs — visible borders */
    [data-testid="stForm"] input {
        background-color: #272b33 !important;
        border: 1px solid #3a3f47 !important;
        border-radius: 6px !important;
        color: #e8e9ea !important;
        padding: 10px 12px !important;
    }
    [data-testid="stForm"] input:focus {
        border-color: #5794F2 !important;
        box-shadow: 0 0 0 1px #5794F2 !important;
    }

    /* Sign In button */
    [data-testid="stForm"] button[kind="secondaryFormSubmit"],
    [data-testid="stForm"] button {
        background-color: #5794F2 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 0 !important;
        font-weight: 600 !important;
    }
    [data-testid="stForm"] button:hover {
        background-color: #4080e0 !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

# --- OAuth Callback Handling ---
if "code" in st.query_params and not is_authenticated():
    handle_oauth_callback()
    st.rerun()

# --- Auth Gate ---
if not is_authenticated():
    if not render_login_page():
        st.stop()

# --- User Info in Sidebar ---
user = get_user()

# --- Sidebar Navigation ---
st.sidebar.title("📊 Stock Analysis")
if user:
    st.sidebar.caption(f"👤 {user.get('name', user.get('email', ''))}")
page = st.sidebar.radio(
    "Navigate",
    ["📈 SPY Predictor", "📊 ES Strategy", "🔬 What-If Analysis",
     "📉 Monitoring", "📉 Grafana (compare)", "⚙️ Admin"],
    label_visibility="collapsed",
)
st.sidebar.divider()
mode_label = "☁️ Cloud" if IS_CLOUD else "🖥️ Local"
st.sidebar.caption(f"{mode_label} mode")
if user and user.get("email") != "anonymous":
    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        logout()
        st.rerun()


# ======================================================================
# SPY PREDICTOR PAGE
# ======================================================================

def load_spy_state() -> dict:
    if IS_CLOUD:
        try:
            resp = requests.get(f"{RELAY_URL}/state", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}
    else:
        try:
            with open(os.path.join(DATA_DIR, "spy_state.json"), "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}


def load_prediction_history(n: int = 20) -> pd.DataFrame:
    if IS_CLOUD:
        return pd.DataFrame()
    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
        df = pd.read_sql_query(
            f"SELECT date, direction, confidence FROM predictions ORDER BY date DESC LIMIT {n}",
            conn,
        )
        conn.close()
        return df.iloc[::-1]
    except Exception:
        return pd.DataFrame()


def load_performance() -> pd.DataFrame:
    if IS_CLOUD:
        return pd.DataFrame()
    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
        df = pd.read_sql_query(
            "SELECT date, predicted, actual, correct, cumulative_accuracy, "
            "confidence_tier, vix_regime, day_of_week, event_proximity "
            "FROM performance ORDER BY date DESC LIMIT 30",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def page_spy():
    state = load_spy_state()
    prediction = state.get("prediction", {})
    indicators = state.get("indicators", {})
    flow_alerts = state.get("flow_alerts", [])
    updated_at = state.get("updated_at", "")

    st.title("📈 SPY/SPX Predictor")
    if st.button("📉 View in Grafana", key="spy_to_grafana"):
        st.session_state["_nav_target"] = "📉 Grafana Monitoring"
        st.rerun()

    direction = prediction.get("direction", "NEUTRAL")
    scale_label = prediction.get("scale_label", "NEUTRAL")
    confidence = prediction.get("confidence", 0)
    probs = prediction.get("probabilities", {})

    color_map = {
        "STRONG_BULLISH": "#00C853", "BULLISH": "#4CAF50",
        "NEUTRAL": "#FFC107",
        "BEARISH": "#FF5722", "STRONG_BEARISH": "#D50000",
    }
    banner_color = color_map.get(scale_label, "#FFC107")

    if prediction:
        st.markdown(
            f"""<div style="background-color:{banner_color}; padding:20px; border-radius:10px;
            text-align:center; margin-bottom:20px;">
            <h1 style="color:white; margin:0;">{scale_label.replace('_', ' ')}</h1>
            <h2 style="color:white; margin:5px 0;">{confidence:.0f}% confidence</h2>
            <p style="color:rgba(255,255,255,0.8); margin:0;">
            ↑ {probs.get('up', 0):.0f}% | — {probs.get('neutral', 0):.0f}% | ↓ {probs.get('down', 0):.0f}%
            </p></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for prediction data...")

    # P2: Regime + Conformal prediction info row
    p2_col1, p2_col2, p2_col3 = st.columns(3)
    with p2_col1:
        # Regime display
        regime = prediction.get("regime", "")
        if regime:
            regime_labels = {
                "bull_trend": "🟢 Bull Trend",
                "bear_trend": "🔴 Bear Trend",
                "high_vol_choppy": "🟡 High-Vol Choppy",
                "low_vol_range": "🔵 Low-Vol Range",
            }
            st.metric("Market Regime", regime_labels.get(regime, regime))
    with p2_col2:
        # Conformal prediction set
        pred_set = prediction.get("prediction_set", [])
        is_low_conv = prediction.get("is_low_conviction", False)
        if pred_set:
            set_str = " / ".join(pred_set)
            if is_low_conv:
                st.metric("Prediction Set", f"⚠️ {set_str}")
                st.caption("LOW CONVICTION — multiple classes in set")
            else:
                st.metric("Prediction Set", f"✅ {set_str}")
    with p2_col3:
        # Ensemble info
        if prediction.get("ensemble_used"):
            st.metric("Model", "🔗 Ensemble")
            st.caption("XGB + BiLSTM + LightGBM")
        else:
            st.metric("Model", "🌲 XGBoost")

    # P3: Earnings + Fed + Extended Options row
    try:
        import sqlite3 as _sql
        _conn = _sql.connect(os.path.join(DATA_DIR, "spy.db"))
        _today = datetime.now().strftime("%Y-%m-%d")

        p3_col1, p3_col2, p3_col3 = st.columns(3)
        with p3_col1:
            # Earnings calendar
            from src.data.earnings_calendar import get_earnings_features as _get_earn
            earn = _get_earn(_conn, _today)
            density = earn.get("earnings_density", 0)
            days_next = earn.get("days_to_next_mega", 30)
            earn_week = earn.get("earnings_week", 0)
            st.metric("📅 Earnings Density", f"{density} mega-caps",
                      delta="Earnings Week" if earn_week else None,
                      delta_color="normal" if earn_week else "off")
            st.caption(f"Next mega-cap in {days_next}d")

        with p3_col2:
            # Fed sentiment
            from src.data.fed_comms import get_fed_features as _get_fed
            fed = _get_fed(_conn, _today)
            fomc = fed.get("fomc_hawkish_score", 0)
            bb = fed.get("beige_book_score", 0)
            avg = fed.get("fed_sentiment_avg", 0)
            label = "🦅 Hawkish" if avg > 0.2 else "🕊️ Dovish" if avg < -0.2 else "⚖️ Neutral"
            st.metric("Fed Sentiment", label, delta=f"{avg:+.2f}")
            st.caption(f"FOMC: {fomc:+.2f} | Beige Book: {bb:+.2f}")

        with p3_col3:
            # Extended options greeks
            opt_row = _conn.execute(
                "SELECT vanna_exposure, charm_exposure, zero_dte_pcr "
                "FROM options_analytics WHERE date = ? ORDER BY date DESC LIMIT 1",
                (_today,),
            ).fetchone()
            if opt_row and opt_row[0] is not None:
                st.metric("Vanna", f"{opt_row[0]:,.0f}")
                st.caption(f"Charm: {opt_row[1]:,.0f} | 0DTE P/C: {opt_row[2]:.2f}" if opt_row[1] else "")
            else:
                st.metric("Extended Greeks", "—")
                st.caption("Waiting for options data")

        _conn.close()
    except Exception:
        pass  # P3 display is non-critical

    # P1: SHAP prediction drivers
    shap_drivers = prediction.get("shap_drivers", [])
    if shap_drivers:
        st.subheader("🔍 Prediction Drivers (SHAP)")
        driver_df = pd.DataFrame(shap_drivers)
        fig_shap = go.Figure()
        colors = ["#00C853" if v > 0 else "#FF5722" for v in driver_df["shap_value"]]
        fig_shap.add_trace(go.Bar(
            y=driver_df["feature"], x=driver_df["shap_value"],
            orientation="h", marker_color=colors,
            text=[f"{v:+.3f}" for v in driver_df["shap_value"]],
            textposition="outside",
            hovertemplate="Feature: %{y}<br>SHAP: %{x:.4f}<br>Value: %{customdata:.4f}",
            customdata=driver_df["feature_value"],
        ))
        fig_shap.update_layout(
            height=200, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="SHAP contribution", yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Prediction History")
        hist_df = load_prediction_history(20)
        if not hist_df.empty:
            colors = hist_df["direction"].map({
                "BULLISH": "green", "STRONG_BULLISH": "darkgreen",
                "BEARISH": "red", "STRONG_BEARISH": "darkred",
                "NEUTRAL": "gray",
            }).fillna("gray")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hist_df["date"], y=hist_df["confidence"],
                marker_color=colors.tolist(),
                text=hist_df["direction"], textposition="outside",
                hovertemplate="Date: %{x}<br>Direction: %{text}<br>Confidence: %{y:.0f}%",
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20),
                              yaxis_title="Confidence %", yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No prediction history yet")

        perf_df = load_performance()
        if not perf_df.empty:
            st.subheader("Accuracy Tracking")
            latest_acc = perf_df.iloc[0]["cumulative_accuracy"] if len(perf_df) > 0 else 0
            st.metric("Cumulative Accuracy", f"{latest_acc:.1%}")

            # P1: Stratified accuracy breakdown
            if "confidence_tier" in perf_df.columns:
                st.caption("Accuracy by Confidence Tier")
                for tier in ["high", "medium", "low"]:
                    tier_df = perf_df[perf_df["confidence_tier"] == tier]
                    if not tier_df.empty:
                        tier_acc = tier_df["correct"].mean()
                        st.markdown(
                            f'<span style="color:#d8d9da;">'
                            f'{"🟢" if tier_acc >= 0.55 else "🟡" if tier_acc >= 0.50 else "🔴"} '
                            f'{tier.title()}: {tier_acc:.1%} ({len(tier_df)} predictions)</span>',
                            unsafe_allow_html=True,
                        )
            if "vix_regime" in perf_df.columns:
                st.caption("Accuracy by VIX Regime")
                for regime in ["low", "normal", "high"]:
                    reg_df = perf_df[perf_df["vix_regime"] == regime]
                    if not reg_df.empty:
                        reg_acc = reg_df["correct"].mean()
                        st.markdown(
                            f'<span style="color:#d8d9da;">'
                            f'VIX {regime}: {reg_acc:.1%} ({len(reg_df)})</span>',
                            unsafe_allow_html=True,
                        )

            st.dataframe(perf_df.head(10), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Key Indicators")
        if indicators:
            ic1, ic2 = st.columns(2)
            with ic1:
                st.metric("RSI(14)", f"{indicators.get('rsi_14', 'N/A')}")
                st.metric("MACD", f"{indicators.get('macd', 'N/A')}")
                st.metric("ATR(14)", f"{indicators.get('atr_14', 'N/A')}")
            with ic2:
                st.metric("VIX", f"{indicators.get('vix', 'N/A')}",
                          delta=f"{indicators.get('vix_change', 0):+.1f}" if indicators.get('vix_change') else None)
                st.metric("Vol Ratio", f"{indicators.get('volume_ratio', 'N/A')}")
                st.metric("Sentiment", f"{indicators.get('sentiment_score', 'N/A')}")
        else:
            st.caption("Waiting for indicator data...")

    st.subheader("Options Flow Alerts")
    if flow_alerts:
        for alert in flow_alerts[-15:]:
            direction_emoji = "🔴" if alert.get("direction") == "PUT" else "🟢"
            notional = alert.get("notional", 0)
            legs = alert.get("legs", "")
            legs_str = f" ({legs}×)" if legs else ""
            st.markdown(
                f'<span style="color:#d8d9da;">{direction_emoji} '
                f'<b>{alert.get("timestamp", "")[:19]}</b> '
                f'{alert.get("direction", "")} {alert.get("type", "")} '
                f'{alert.get("symbol", "")} ${notional:,.0f}{legs_str}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No options flow alerts yet")

    st.divider()
    st.markdown(f'<span style="color:#888;">Last updated: {updated_at or "N/A"}</span>',
                unsafe_allow_html=True)


# ======================================================================
# ES STRATEGY PAGE
# ======================================================================

def load_es_state() -> dict:
    if IS_CLOUD:
        try:
            resp = requests.get(f"{RELAY_URL}/state/es", timeout=5)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {}
    else:
        try:
            with open(os.path.join(DATA_DIR, "es_state.json"), "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}


def page_es():
    state = load_es_state()
    position = state.get("position", {"status": "FLAT", "lots": 0})
    signals = state.get("signals", [])
    regime = state.get("regime", "Med")
    pnl = state.get("pnl", {"daily": 0.0, "unrealized": 0.0})
    chart_data = state.get("chart_data", {})
    updated_at = state.get("updated_at", "")

    st.title("📊 ES Futures Strategy")
    if st.button("📉 View in Grafana", key="es_to_grafana"):
        st.session_state["_nav_target"] = "📉 Grafana Monitoring"
        st.rerun()

    pos_status = position.get("status", "FLAT")
    pos_lots = position.get("lots", 0)
    entry_price = position.get("entry_price", 0)
    unrealized = pnl.get("unrealized", 0)
    daily_pnl = pnl.get("daily", 0)

    regime_colors = {"Low": "#4CAF50", "Med": "#FFC107", "High": "#FF5722"}
    pos_colors = {"LONG": "#00C853", "SHORT": "#D50000", "FLAT": "#9E9E9E"}
    banner_color = pos_colors.get(pos_status, "#9E9E9E")
    regime_color = regime_colors.get(regime, "#FFC107")

    st.markdown(
        f"""<div style="background-color:{banner_color}; padding:15px; border-radius:10px;
        text-align:center; margin-bottom:15px; display:flex; justify-content:space-around; align-items:center;">
        <div><h2 style="color:white; margin:0;">{pos_status} {pos_lots} lots</h2></div>
        <div><p style="color:white; margin:0;">Entry: {entry_price}</p></div>
        <div><p style="color:white; margin:0;">P&L: ${unrealized:+,.0f}</p></div>
        <div style="background-color:{regime_color}; padding:5px 15px; border-radius:5px;">
        <p style="color:white; margin:0;">Regime: {regime}</p></div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Chart
    st.subheader("Price Chart")
    bars = chart_data.get("bars", [])
    if bars:
        df = pd.DataFrame(bars)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.75, 0.25], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(
                x=df["timestamp"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name="ES",
            ), row=1, col=1)

            if "kc_upper" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["kc_upper"], mode="lines",
                    line=dict(color="rgba(100,149,237,0.3)"), name="KC Upper",
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["kc_lower"], mode="lines",
                    fill="tonexty", fillcolor="rgba(100,149,237,0.1)",
                    line=dict(color="rgba(100,149,237,0.3)"), name="KC Lower",
                ), row=1, col=1)
            if "kc_mid" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["kc_mid"], mode="lines",
                    line=dict(color="cornflowerblue", dash="dash", width=1), name="KC Mid",
                ), row=1, col=1)
            if "vwap" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["vwap"], mode="lines",
                    line=dict(color="orange", dash="dot", width=1), name="VWAP",
                ), row=1, col=1)

            for e in chart_data.get("entries", []):
                marker = "triangle-up" if e.get("direction") == "LONG" else "triangle-down"
                color = "green" if e.get("direction") == "LONG" else "red"
                fig.add_trace(go.Scatter(
                    x=[e["timestamp"]], y=[e["price"]], mode="markers",
                    marker=dict(symbol=marker, size=14, color=color),
                    name=e.get("label", "Entry"), showlegend=False,
                ), row=1, col=1)
            for ex in chart_data.get("exits", []):
                fig.add_trace(go.Scatter(
                    x=[ex["timestamp"]], y=[ex["price"]], mode="markers",
                    marker=dict(symbol="x", size=12, color="blue"),
                    name=ex.get("label", "Exit"), showlegend=False,
                ), row=1, col=1)
            for s in chart_data.get("stop_levels", []):
                fig.add_hline(y=s, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

            if "rsi" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["rsi"], mode="lines",
                    line=dict(color="purple", width=1), name="RSI(14)",
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.3, row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.3, row=2, col=1)

            fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20),
                              xaxis_rangeslider_visible=False, showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Waiting for chart data...")

    # Signal Feed + Status
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Signal Feed")
        if signals:
            type_emojis = {
                "ENTRY_LONG": "🟢", "ENTRY_SHORT": "🔴",
                "EXIT_TP1": "💰", "EXIT_TP2": "💰", "EXIT_RUNNER": "🏃",
                "STOP_HIT": "🛑", "STOP_UPDATE": "📍",
                "AI_REJECT": "🤖", "CIRCUIT_BREAKER": "⚡", "SESSION_FLATTEN": "🕐",
            }
            for sig in reversed(signals[-20:]):
                emoji = type_emojis.get(sig.get("type", ""), "📌")
                st.markdown(
                    f'<span style="color:#d8d9da; font-family:monospace; font-size:0.85em;">'
                    f'{emoji} {sig.get("timestamp", "")[:8]} {sig.get("type", "")} '
                    f'{sig.get("detail", "")}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No signals yet")

    with col2:
        st.subheader("Status Panel")
        cb_status = state.get("circuit_breaker", "OK")
        st.markdown(f"{'🟢' if cb_status == 'OK' else '🔴'} Circuit Breaker: **{cb_status}**")
        st.metric("Daily P&L", f"${daily_pnl:+,.0f}")
        st.metric("Trades Today", state.get("trade_count", 0))
        st.markdown(f"Session: **{state.get('session_status', 'Inactive')}**")

        lots_detail = position.get("lots_detail", [])
        if lots_detail:
            st.subheader("Lot Status")
            for lot in lots_detail:
                st.markdown(
                    f'<span style="color:#d8d9da; font-family:monospace;">'
                    f'  Lot {lot.get("id", "?")}: {lot.get("status", "?")} '
                    f'(${lot.get("pnl", 0):+,.0f})</span>',
                    unsafe_allow_html=True,
                )

    st.divider()
    st.caption(f"Last updated: {updated_at or 'N/A'}")


# ======================================================================
# WHAT-IF ANALYSIS PAGE
# ======================================================================

from src.whatif.engine import WhatIfEngine
from src.whatif.presets import STRESS_SCENARIOS
from src.data.features import get_feature_columns


@st.cache_resource
def get_whatif_engine():
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        config = {}
    return WhatIfEngine(config)


def page_whatif():
    st.title("🔬 What-If Analysis")
    engine = get_whatif_engine()

    tab_es, tab_spy = st.tabs(["ES Strategy", "SPY Predictor"])

    with tab_es:
        _whatif_es_tab(engine)
    with tab_spy:
        _whatif_spy_tab(engine)


def _whatif_es_tab(engine: WhatIfEngine):
    scenario = st.selectbox(
        "Scenario Type",
        ["K/C Sweep", "Lot Sizing", "Risk Limits", "Custom Compare"],
        key="es_scenario",
    )

    if scenario == "K/C Sweep":
        st.subheader("Keltner Channel Parameter Sweep")
        col1, col2 = st.columns(2)
        with col1:
            c_min = st.number_input("C min", value=6.0, step=1.0, key="c_min")
            c_max = st.number_input("C max", value=16.0, step=1.0, key="c_max")
            c_step = st.number_input("C step", value=2.0, step=0.5, key="c_step")
        with col2:
            k_min = st.number_input("K min", value=5900.0, step=25.0, key="k_min")
            k_max = st.number_input("K max", value=6100.0, step=25.0, key="k_max")
            k_step = st.number_input("K step", value=50.0, step=25.0, key="k_step")
        if st.button("Run K/C Sweep", key="run_kc"):
            c_vals = list(np.arange(c_min, c_max + 0.01, c_step))
            k_vals = list(np.arange(k_min, k_max + 0.01, k_step))
            with st.spinner(f"Running {len(c_vals) * len(k_vals)} backtests..."):
                result = engine.es_parameter_sweep({"credit_C": c_vals, "strike_K": k_vals})
            _render_sweep_heatmap(result, c_vals, k_vals)

    elif scenario == "Lot Sizing":
        st.subheader("Lot Sizing Comparison")
        if st.button("Compare 1 / 2 / 3 lots", key="run_lots"):
            scenarios = [
                {"label": "1 lot", "overrides": {"max_lots": 1}},
                {"label": "2 lots", "overrides": {"max_lots": 2}},
                {"label": "3 lots", "overrides": {"max_lots": 3}},
            ]
            with st.spinner("Running 3 backtests..."):
                result = engine.es_compare_scenarios(scenarios)
            _render_comparison_bar(result)

    elif scenario == "Risk Limits":
        st.subheader("Circuit Breaker Sensitivity")
        if st.button("Sweep -$1K to -$3K", key="run_risk"):
            vals = list(np.arange(-3000, -500, 500))
            with st.spinner(f"Running {len(vals)} backtests..."):
                result = engine.es_parameter_sweep({"circuit_breaker_usd": vals})
            _render_risk_chart(result, vals)

    elif scenario == "Custom Compare":
        st.subheader("Custom Scenario Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Scenario A")
            a_c = st.number_input("Credit C (A)", value=10.0, key="a_c")
            a_lots = st.number_input("Max lots (A)", value=3, min_value=1, max_value=3, key="a_lots")
        with col2:
            st.markdown("Scenario B")
            b_c = st.number_input("Credit C (B)", value=12.0, key="b_c")
            b_lots = st.number_input("Max lots (B)", value=2, min_value=1, max_value=3, key="b_lots")
        if st.button("Compare", key="run_custom"):
            scenarios = [
                {"label": f"A: C={a_c}, lots={a_lots}", "overrides": {"credit_C": a_c, "max_lots": int(a_lots)}},
                {"label": f"B: C={b_c}, lots={b_lots}", "overrides": {"credit_C": b_c, "max_lots": int(b_lots)}},
            ]
            with st.spinner("Running backtests..."):
                result = engine.es_compare_scenarios(scenarios)
            _render_comparison_bar(result)


def _whatif_spy_tab(engine: WhatIfEngine):
    scenario = st.selectbox(
        "Scenario Type",
        ["Feature Override", "Feature Ablation", "Monte Carlo",
         "Stress Test", "Threshold Sensitivity"],
        key="spy_scenario",
    )

    if scenario == "Feature Override":
        st.subheader("What if a feature changes?")
        col1, col2 = st.columns(2)
        overrides = {}
        with col1:
            vix = st.number_input("VIX", value=0.0, step=1.0, key="ov_vix", help="Set to 0 to skip")
            if vix > 0:
                overrides["vix"] = vix
            sent = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.05, key="ov_sent")
            if sent != 0:
                overrides["sentiment_score"] = sent
        with col2:
            rsi = st.number_input("RSI(14)", value=0.0, step=1.0, key="ov_rsi", help="Set to 0 to skip")
            if rsi > 0:
                overrides["rsi_14"] = rsi
            pc = st.number_input("Put/Call Ratio", value=0.0, step=0.1, key="ov_pc", help="Set to 0 to skip")
            if pc > 0:
                overrides["put_call_ratio"] = pc
        if st.button("Run Override", key="run_override") and overrides:
            with st.spinner("Running inference..."):
                result = engine.spy_scenario_inject(overrides)
            _render_spy_comparison(result)

    elif scenario == "Feature Ablation":
        st.subheader("Which features matter most?")
        all_features = get_feature_columns()
        drop = st.multiselect("Features to drop (zero out)", all_features,
                              default=["sentiment_score", "put_call_ratio"], key="ablation_drop")
        if st.button("Run Ablation", key="run_ablation") and drop:
            with st.spinner("Computing accuracy impact..."):
                result = engine.spy_feature_ablation(drop)
            _render_ablation(result)

    elif scenario == "Monte Carlo":
        st.subheader("Prediction Distribution Under Noise")
        col1, col2 = st.columns(2)
        with col1:
            n_sims = st.slider("Simulations", 100, 1000, 500, 50, key="mc_sims")
        with col2:
            noise = st.slider("Noise %", 0.5, 10.0, 2.0, 0.5, key="mc_noise")
        if st.button("Run Monte Carlo", key="run_mc"):
            with st.spinner(f"Running {n_sims} simulations..."):
                result = engine.spy_monte_carlo(n_sims, noise)
            _render_monte_carlo(result)

    elif scenario == "Stress Test":
        st.subheader("Pre-built Market Stress Scenarios")
        scenarios = WhatIfEngine.list_stress_scenarios()
        names = [s["name"] for s in scenarios]
        labels = [f"{s['label']} — {s['description']}" for s in scenarios]
        choice = st.selectbox("Scenario", labels, key="stress_choice")
        idx = labels.index(choice)
        if st.button("Run Stress Test", key="run_stress"):
            with st.spinner(f"Running {scenarios[idx]['label']}..."):
                result = engine.market_stress_test(names[idx])
            _render_spy_comparison(result)

    elif scenario == "Threshold Sensitivity":
        st.subheader("Neutral Threshold Sensitivity")
        if st.button("Run Threshold Sweep", key="run_thresh"):
            with st.spinner("Sweeping thresholds..."):
                results = _run_threshold_sweep(engine)
            _render_threshold_sweep(results)


# --- What-If chart helpers ---

def _render_sweep_heatmap(result, c_vals, k_vals):
    data = result.get("results", {})
    if not data:
        st.warning("No results"); return
    z = []
    for c in c_vals:
        row = []
        for k in k_vals:
            entry = data.get(f"credit_C={c}, strike_K={k}", {})
            row.append(entry.get("total_pnl", 0))
        z.append(row)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"K={k:.0f}" for k in k_vals], y=[f"C={c:.1f}" for c in c_vals],
        colorscale="RdYlGn", colorbar_title="P&L ($)",
    ))
    fig.update_layout(title="K/C Sweep — Total P&L", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe([
        {"params": k, "P&L": f"${v.get('total_pnl', 0):+,.0f}", "trades": v.get("trades", 0)}
        for k, v in data.items() if "error" not in v
    ])


def _render_comparison_bar(result):
    items = result.get("results", [])
    if not items:
        st.warning("No results"); return
    labels = [r.get("label", "?") for r in items]
    pnls = [r.get("total_pnl", 0) for r in items]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="P&L ($)", x=labels, y=pnls,
                         marker_color=["green" if p > 0 else "red" for p in pnls]))
    fig.update_layout(title="Scenario Comparison — P&L", height=400)
    st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    for i, item in enumerate(items):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.metric(item.get("label", "?"), f"${item.get('total_pnl', 0):+,.0f}",
                      f"{item.get('trades', 0)} trades")


def _render_risk_chart(result, vals):
    data = result.get("results", {})
    pnls = [data.get(f"circuit_breaker_usd={v}", {}).get("total_pnl", 0) for v in vals]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[f"${v:,.0f}" for v in vals], y=pnls, mode="lines+markers"))
    fig.update_layout(title="Circuit Breaker vs P&L", xaxis_title="Breaker Limit",
                      yaxis_title="Total P&L ($)", height=400)
    st.plotly_chart(fig, use_container_width=True)


def _render_spy_comparison(result):
    if "error" in result:
        st.error(result["error"]); return
    orig = result.get("original", {})
    mod = result.get("modified", {})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Prediction**")
        _prediction_card(orig)
    with col2:
        st.markdown("**Modified Prediction**")
        _prediction_card(mod)
    overrides = result.get("overrides", {})
    if overrides:
        st.markdown("**Features Changed:**")
        st.json(overrides)
    desc = result.get("description", "")
    if desc:
        st.info(desc)


def _prediction_card(pred):
    direction = pred.get("direction", "N/A")
    confidence = pred.get("confidence", 0)
    probs = pred.get("probabilities", {})
    color = "🟢" if "BULLISH" in direction else "🔴" if "BEARISH" in direction else "⚪"
    st.markdown(f"### {color} {pred.get('scale_label', direction)}")
    st.markdown(f"Confidence: **{confidence:.0f}%**")
    if probs:
        fig = go.Figure(go.Bar(
            x=list(probs.values()), y=list(probs.keys()),
            orientation="h", marker_color=["red", "gray", "green"],
        ))
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), xaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)


def _render_ablation(result):
    if "error" in result:
        st.error(result["error"]); return
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Accuracy", f"{result['baseline_accuracy']:.1%}")
    c2.metric("Ablated Accuracy", f"{result['ablated_accuracy']:.1%}")
    c3.metric("Impact", f"{result['accuracy_impact']:+.1%}", delta_color="inverse")
    st.markdown(f"Dropped: `{', '.join(result['dropped'])}` | {result['samples']} samples")


def _render_monte_carlo(result):
    if "error" in result:
        st.error(result["error"]); return
    dist = result.get("distribution", {})
    fig = go.Figure(go.Bar(x=list(dist.keys()), y=list(dist.values()),
                           marker_color=["green", "red", "gray"]))
    fig.update_layout(title=f"Monte Carlo ({result['n_sims']} sims, {result['noise_pct']}% noise)",
                      yaxis_title="% of simulations", height=400)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Avg Confidence", f"{result['avg_confidence']:.1f}%")
    c2.metric("Std Confidence", f"{result['std_confidence']:.1f}%")


def _run_threshold_sweep(engine):
    from src.data.features import get_target
    fv = engine._get_features()
    if fv is None:
        return []
    predictor = engine._get_predictor()
    available = [c for c in engine._feature_cols if c in fv.columns]
    X = fv[available]
    results = []
    for thresh in [0.001, 0.002, 0.003, 0.004, 0.005]:
        y = get_target(fv, threshold=thresh)
        tail = min(50, len(X) - 1)
        correct = total = 0
        for i in range(len(X) - tail, len(X)):
            pred = predictor.predict(X.iloc[i].values.astype(np.float64))
            actual = int(y.iloc[i])
            if not np.isnan(actual):
                total += 1
                if pred["predicted_class"] == actual:
                    correct += 1
        acc = correct / total if total > 0 else 0
        results.append({"threshold": f"±{thresh*100:.1f}%", "accuracy": round(acc, 4), "samples": total})
    return results


def _render_threshold_sweep(results):
    if not results:
        st.warning("No results"); return
    fig = go.Figure(go.Bar(
        x=[r["threshold"] for r in results],
        y=[r["accuracy"] * 100 for r in results],
        marker_color="steelblue",
    ))
    fig.update_layout(title="Accuracy by Neutral Threshold", yaxis_title="Accuracy %", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(results)


# ======================================================================
# ADMIN PAGE
# ======================================================================

import subprocess
import sqlite3
import threading


def _load_config() -> dict:
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _run_in_thread(target, status_key: str, args=()):
    """Run a function in a background thread, tracking status in session_state."""
    def wrapper():
        st.session_state[status_key] = "running"
        try:
            result = target(*args)
            st.session_state[f"{status_key}_result"] = result
            st.session_state[status_key] = "done"
        except Exception as e:
            st.session_state[f"{status_key}_result"] = str(e)
            st.session_state[status_key] = "error"
    t = threading.Thread(target=wrapper, daemon=True)
    t.start()


def page_admin():
    st.title("⚙️ Admin Console")

    tab_status, tab_actions, tab_db, tab_config, tab_logs = st.tabs([
        "System Status", "Actions", "Database", "Configuration", "Logs",
    ])

    with tab_status:
        _admin_status_tab()
    with tab_actions:
        _admin_actions_tab()
    with tab_db:
        _admin_db_tab()
    with tab_config:
        _admin_config_tab()
    with tab_logs:
        _admin_logs_tab()


# --- System Status Tab ---

def _admin_status_tab():
    st.subheader("System Health")

    if st.button("🔄 Refresh Status", key="refresh_status"):
        st.rerun()

    col1, col2, col3 = st.columns(3)

    # Database status
    with col1:
        st.markdown("**Database**")
        db_path = os.path.join(DATA_DIR, "spy.db")
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            st.success(f"Online — {size_mb:.1f} MB")
            try:
                conn = sqlite3.connect(db_path)
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                conn.close()
                st.caption(f"{len(tables)} tables")
            except Exception as e:
                st.warning(f"Read error: {e}")
        else:
            st.error("Database not found")

    # LLM status
    with col2:
        st.markdown("**LLM (Ollama)**")
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                config = _load_config()
                target_model = config.get("llm", {}).get("model", "deepseek-r1:70b")
                if any(target_model in n for n in model_names):
                    st.success(f"Online — {target_model}")
                else:
                    st.warning(f"Ollama running but {target_model} not found")
                    st.caption(f"Available: {', '.join(model_names) or 'none'}")
            else:
                st.error("Ollama not responding")
        except Exception:
            st.error("Ollama offline")

    # Model status
    with col3:
        st.markdown("**XGBoost Model**")
        model_dir = "./models"
        if os.path.exists(model_dir):
            models = sorted([f for f in os.listdir(model_dir) if f.endswith(".json")], reverse=True)
            if models:
                latest = models[0]
                size_kb = os.path.getsize(os.path.join(model_dir, latest)) / 1024
                st.success(f"{latest} ({size_kb:.0f} KB)")
                if len(models) > 1:
                    st.caption(f"{len(models)} models total")
            else:
                st.warning("No trained models found")
        else:
            st.error("Models directory missing")

    st.divider()

    # Table row counts
    st.subheader("Data Inventory")
    db_path = os.path.join(DATA_DIR, "spy.db")
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            tables = ["prices", "technicals", "news", "daily_sentiment", "macro",
                       "predictions", "intraday_bars", "options_chain",
                       "options_analytics", "intraday_features", "performance",
                       "earnings_calendar", "fed_communications"]
            rows = []
            for t in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    min_date = conn.execute(f"SELECT MIN(date) FROM {t}").fetchone()[0] if t != "intraday_bars" else "—"
                    max_date = conn.execute(f"SELECT MAX(date) FROM {t}").fetchone()[0] if t != "intraday_bars" else "—"
                except Exception:
                    count, min_date, max_date = 0, "—", "—"
                rows.append({"Table": t, "Rows": count, "From": min_date or "—", "To": max_date or "—"})
            conn.close()
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error reading database: {e}")

    # Latest prediction
    st.subheader("Latest Prediction")
    try:
        conn = sqlite3.connect(db_path)
        pred = conn.execute(
            "SELECT date, direction, confidence, predicted_at FROM predictions ORDER BY date DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if pred:
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Date", pred[0])
            pc2.metric("Direction", pred[1])
            pc3.metric("Confidence", f"{pred[2]:.0f}%")
            pc4.metric("Generated", pred[3] or "—")
        else:
            st.info("No predictions yet")
    except Exception:
        st.info("No predictions yet")

    # P2: Model Registry
    st.subheader("Model Registry")
    try:
        conn = sqlite3.connect(db_path)
        reg_rows = conn.execute(
            """SELECT model_id, training_date, val_accuracy, test_accuracy,
                      feature_count, gated, deployment_status, created_at
               FROM model_registry ORDER BY created_at DESC LIMIT 10"""
        ).fetchall()
        conn.close()
        if reg_rows:
            reg_df = pd.DataFrame(reg_rows, columns=[
                "ID", "Date", "Val Acc", "Test Acc", "Features", "Gated", "Status", "Created"
            ])
            reg_df["Val Acc"] = reg_df["Val Acc"].apply(lambda x: f"{x:.3f}" if x else "—")
            reg_df["Test Acc"] = reg_df["Test Acc"].apply(lambda x: f"{x:.3f}" if x else "—")
            reg_df["Gated"] = reg_df["Gated"].map({0: "✅", 1: "🚫"})
            status_map = {"active": "🟢 Active", "retired": "⚪ Retired", "gated": "🚫 Gated"}
            reg_df["Status"] = reg_df["Status"].map(status_map).fillna(reg_df["Status"])
            st.dataframe(reg_df, use_container_width=True, hide_index=True)
        else:
            st.info("No models registered yet — will populate after next training run")
    except Exception:
        st.info("Model registry not initialized yet")


# --- Actions Tab ---

def _admin_actions_tab():
    st.subheader("Ad-Hoc Actions")
    st.caption("Run pipeline steps individually or trigger full operations on demand.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Data Operations**")

        if st.button("📥 Pull Latest Data", key="act_pull", help="Gap detection + backfill prices and macro"):
            with st.spinner("Running daily data pull..."):
                try:
                    from src.data.daily_pull import run_daily_pull
                    config = _load_config()
                    counts = run_daily_pull(config)
                    st.success("Data pull complete")
                    st.json(counts)
                except Exception as e:
                    st.error(f"Data pull failed: {e}")

        if st.button("📰 Fetch News", key="act_news", help="Fetch latest news from Finnhub + RSS"):
            with st.spinner("Fetching news..."):
                try:
                    from src.data.fetcher import FallbackFetcher
                    fetcher = FallbackFetcher()
                    articles = fetcher.get_news()
                    config = _load_config()
                    conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
                    inserted = 0
                    today = datetime.now().strftime("%Y-%m-%d")
                    for a in articles:
                        try:
                            conn.execute(
                                "INSERT INTO news (date, source, headline, summary, url, fetched_at) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (today, a.get("source", ""), a.get("headline", ""),
                                 a.get("summary", ""), a.get("url", ""),
                                 datetime.now().isoformat())
                            )
                            inserted += 1
                        except Exception:
                            pass
                    conn.commit()
                    conn.close()
                    st.success(f"Fetched {len(articles)} articles, inserted {inserted} new")
                except Exception as e:
                    st.error(f"News fetch failed: {e}")

        if st.button("📊 Fetch Macro Data", key="act_macro", help="Fetch VIX, yields, DXY, gold, crude from FRED"):
            with st.spinner("Fetching macro data..."):
                try:
                    from src.data.fetcher import FallbackFetcher
                    fetcher = FallbackFetcher()
                    macro = fetcher.get_macro_fred()
                    today = datetime.now().strftime("%Y-%m-%d")
                    conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
                    conn.execute(
                        "INSERT OR REPLACE INTO macro (date, vix, vix_change, us10y_yield, dxy, fed_funds, gold, crude) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (today, macro.get("vix"), macro.get("vix_change"),
                         macro.get("us10y_yield"), macro.get("dxy"),
                         macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
                    )
                    conn.commit()
                    conn.close()
                    st.success("Macro data updated")
                    st.json(macro)
                except Exception as e:
                    st.error(f"Macro fetch failed: {e}")

        if st.button("📈 Compute Technicals", key="act_tech", help="Recompute SMA, RSI, MACD, BB, ATR"):
            with st.spinner("Computing technicals..."):
                try:
                    from src.data.features import compute_technicals
                    config = _load_config()
                    from src.data.init_db import get_connection
                    conn = get_connection(config)
                    compute_technicals(conn, config)
                    conn.close()
                    st.success("Technicals computed")
                except Exception as e:
                    st.error(f"Technicals failed: {e}")

    with col2:
        st.markdown("**Model Operations**")

        if st.button("🧠 Retrain XGBoost", key="act_train", help="Retrain SPY predictor with latest data (GPU)"):
            with st.spinner("Training XGBoost on GPU... this may take a minute."):
                try:
                    from src.data.init_db import get_connection
                    from src.data.features import build_feature_vector, get_target
                    from src.model.trainer import SPYPredictor
                    config = _load_config()
                    conn = get_connection(config)
                    fv = build_feature_vector(conn)
                    conn.close()
                    target = get_target(fv)
                    predictor = SPYPredictor(config)
                    feature_cols = [c for c in fv.columns if c not in ["date", "close", "target", "next_return"]]
                    X = fv[feature_cols]
                    result = predictor.train(X, target)
                    st.success(f"Training complete — accuracy: {result.get('val_accuracy', 0):.1%}")
                    st.json(result)
                except Exception as e:
                    st.error(f"Training failed: {e}")

        if st.button("🔮 Generate Prediction", key="act_predict", help="Run inference for next trading day"):
            with st.spinner("Generating prediction..."):
                try:
                    from src.data.init_db import get_connection
                    from src.data.features import build_feature_vector, get_feature_columns
                    from src.model.trainer import SPYPredictor
                    config = _load_config()
                    conn = get_connection(config)
                    fv = build_feature_vector(conn)
                    predictor = SPYPredictor(config)
                    if not predictor.load_latest_model():
                        st.error("No trained model found — train first")
                    else:
                        feature_cols = [c for c in get_feature_columns() if c in fv.columns]
                        latest = fv[feature_cols].iloc[-1].values.astype(np.float64)
                        pred = predictor.predict(latest)
                        # Store in DB
                        today = datetime.now().strftime("%Y-%m-%d")
                        conn.execute(
                            "INSERT OR REPLACE INTO predictions (date, direction, confidence, predicted_at) "
                            "VALUES (?, ?, ?, ?)",
                            (today, pred.get("direction", ""), pred.get("confidence", 0),
                             datetime.now().isoformat())
                        )
                        conn.commit()
                        st.success(f"{pred.get('scale_label', pred.get('direction'))} — {pred.get('confidence', 0):.0f}% confidence")
                        st.json(pred)
                    conn.close()
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        if st.button("🩺 LLM Health Check", key="act_llm", help="Check Ollama + model availability"):
            with st.spinner("Checking LLM health..."):
                try:
                    from src.llm.analyzer import LLMAnalyzer
                    config = _load_config()
                    llm = LLMAnalyzer(config)
                    ok = llm.check_health()
                    if ok:
                        st.success("LLM is healthy and ready")
                    else:
                        st.warning("LLM unavailable — system will use neutral sentiment")
                except Exception as e:
                    st.error(f"LLM check failed: {e}")

        if st.button("📝 Generate Report", key="act_report", help="Generate LLM daily report for latest prediction"):
            with st.spinner("Generating LLM report (this may take a few minutes)..."):
                try:
                    from src.llm.reporter import ReportGenerator
                    config = _load_config()
                    from src.data.init_db import get_connection
                    conn = get_connection(config)
                    pred = conn.execute(
                        "SELECT date, direction, confidence FROM predictions ORDER BY date DESC LIMIT 1"
                    ).fetchone()
                    if not pred:
                        st.error("No prediction found — generate one first")
                    else:
                        reporter = ReportGenerator(config)
                        report = reporter.generate(conn, pred[0])
                        conn.execute(
                            "UPDATE predictions SET report_text = ? WHERE date = ?",
                            (report, pred[0])
                        )
                        conn.commit()
                        st.success("Report generated")
                        st.markdown(report)
                    conn.close()
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

    st.divider()

    # Full pipeline
    st.markdown("**Full Pipeline**")
    pc1, pc2 = st.columns(2)
    with pc1:
        skip_llm = st.checkbox("Skip LLM steps (faster)", value=True, key="skip_llm_check")
    with pc2:
        if st.button("🚀 Run Full Pipeline Now", key="act_pipeline", type="primary"):
            with st.spinner("Running full 13-step pipeline... this may take several minutes."):
                try:
                    from src.pipeline.daily_run import DailyPipeline
                    config = _load_config()
                    pipeline = DailyPipeline(config)
                    results = pipeline.run(skip_llm=skip_llm)
                    elapsed = results.get("total_elapsed", 0)
                    errors = sum(1 for k, v in results.items() if isinstance(v, dict) and v.get("status") == "error")
                    if errors:
                        st.warning(f"Pipeline complete in {elapsed:.0f}s with {errors} error(s)")
                    else:
                        st.success(f"Pipeline complete in {elapsed:.0f}s — all steps OK")
                    # Show step results
                    step_rows = []
                    for k, v in sorted(results.items()):
                        if k.startswith("step_") and isinstance(v, dict):
                            step_rows.append({
                                "Step": k.replace("step_", ""),
                                "Status": "✅" if v.get("status") == "ok" else "❌",
                                "Time": f"{v.get('elapsed', 0):.1f}s",
                                "Detail": str(v.get("error", ""))[:80] if v.get("status") == "error" else "",
                            })
                    if step_rows:
                        st.dataframe(pd.DataFrame(step_rows), use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")

    st.divider()

    # Send test alert
    st.markdown("**Alerts**")
    if st.button("📨 Send Test Alert", key="act_alert", help="Send a test prediction alert"):
        try:
            from src.pipeline.alerts import send_alerts
            config = _load_config()
            test_data = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "direction": "TEST",
                "confidence": 99.9,
                "report": "This is a test alert from the admin console.",
            }
            result = send_alerts(config, test_data)
            if result.get("sent"):
                st.success(f"Alert sent — Telegram: {result.get('telegram')}, Email: {result.get('email')}")
            else:
                st.warning("No alert channels configured in config.yaml")
        except Exception as e:
            st.error(f"Alert failed: {e}")


# --- Database Tab ---

def _admin_db_tab():
    st.subheader("Database Explorer")

    db_path = os.path.join(DATA_DIR, "spy.db")
    if not os.path.exists(db_path):
        st.error("Database not found")
        return

    conn = sqlite3.connect(db_path)

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]

    selected = st.selectbox("Table", tables, key="db_table")

    col1, col2 = st.columns([1, 1])
    with col1:
        limit = st.number_input("Row limit", value=50, min_value=1, max_value=1000, key="db_limit")
    with col2:
        order = st.selectbox("Order", ["DESC", "ASC"], key="db_order")

    # Determine date column
    date_col = "date"
    if selected == "intraday_bars":
        date_col = "timestamp"
    elif selected == "news":
        date_col = "id"

    try:
        df = pd.read_sql_query(
            f"SELECT * FROM {selected} ORDER BY {date_col} {order} LIMIT {limit}",
            conn,
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df)} of {conn.execute(f'SELECT COUNT(*) FROM {selected}').fetchone()[0]} rows")
    except Exception as e:
        st.error(f"Query error: {e}")

    st.divider()

    # Custom SQL query
    st.subheader("Custom Query")
    query = st.text_area("SQL (read-only, SELECT only)", value=f"SELECT * FROM {selected} LIMIT 10", key="db_query", height=80)
    if st.button("Run Query", key="run_query"):
        if not query.strip().upper().startswith("SELECT"):
            st.error("Only SELECT queries are allowed")
        else:
            try:
                df = pd.read_sql_query(query, conn)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"{len(df)} rows returned")
            except Exception as e:
                st.error(f"Query error: {e}")

    conn.close()

    st.divider()

    # DB maintenance
    st.subheader("Maintenance")
    mc1, mc2 = st.columns(2)
    with mc1:
        if st.button("🗜️ Vacuum Database", key="db_vacuum", help="Reclaim unused space"):
            try:
                conn = sqlite3.connect(db_path)
                size_before = os.path.getsize(db_path)
                conn.execute("VACUUM")
                conn.close()
                size_after = os.path.getsize(db_path)
                saved = (size_before - size_after) / 1024
                st.success(f"Vacuum complete — saved {saved:.0f} KB")
            except Exception as e:
                st.error(f"Vacuum failed: {e}")
    with mc2:
        if st.button("🔍 Integrity Check", key="db_integrity"):
            try:
                conn = sqlite3.connect(db_path)
                result = conn.execute("PRAGMA integrity_check").fetchone()[0]
                conn.close()
                if result == "ok":
                    st.success("Database integrity: OK")
                else:
                    st.error(f"Integrity issue: {result}")
            except Exception as e:
                st.error(f"Check failed: {e}")


# --- Configuration Tab ---

def _admin_config_tab():
    st.subheader("Configuration")

    config_path = "config.yaml"
    try:
        with open(config_path) as f:
            raw = f.read()
    except Exception:
        st.error("config.yaml not found")
        return

    config = yaml.safe_load(raw) or {}

    # Quick view of key settings
    st.markdown("**Key Settings**")
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("LLM Model", config.get("llm", {}).get("model", "—"))
    kc2.metric("XGB Lookback", f"{config.get('xgboost', {}).get('lookback_days', '—')} days")
    kc3.metric("ES Max Lots", config.get("es_strategy", {}).get("max_lots", "—"))
    kc4.metric("Cloud Sync", "On" if config.get("sync", {}).get("enabled") else "Off")

    st.divider()

    # Editable config
    st.markdown("**Edit Configuration**")
    st.caption("Edit the YAML below and click Save. Changes take effect on next component restart.")
    edited = st.text_area("config.yaml", value=raw, height=400, key="config_editor")

    if st.button("💾 Save Configuration", key="save_config"):
        try:
            # Validate YAML
            parsed = yaml.safe_load(edited)
            if not isinstance(parsed, dict):
                st.error("Invalid YAML — must be a mapping")
            else:
                with open(config_path, "w") as f:
                    f.write(edited)
                st.success("Configuration saved. Restart components to apply changes.")
        except yaml.YAMLError as e:
            st.error(f"YAML syntax error: {e}")


# --- Logs Tab ---

def _admin_logs_tab():
    st.subheader("System Logs")

    log_source = st.selectbox("Log Source", [
        "Dashboard Log (/tmp/dashboard.log)",
        "Pipeline (last run)",
        "Model Files",
    ], key="log_source")

    if log_source.startswith("Dashboard"):
        log_path = "/tmp/dashboard.log"
        try:
            with open(log_path) as f:
                lines = f.readlines()
            n = st.slider("Lines (most recent)", 20, 200, 50, key="log_lines")
            st.code("".join(lines[-n:]), language="log")
        except FileNotFoundError:
            st.info("No dashboard log found at /tmp/dashboard.log")
        except Exception as e:
            st.error(f"Error reading log: {e}")

    elif log_source.startswith("Pipeline"):
        st.markdown("**Last Pipeline Results**")
        db_path = os.path.join(DATA_DIR, "spy.db")
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                pred = conn.execute(
                    "SELECT date, direction, confidence, report_text, predicted_at "
                    "FROM predictions ORDER BY date DESC LIMIT 5"
                ).fetchall()
                conn.close()
                if pred:
                    for p in pred:
                        with st.expander(f"{p[0]} — {p[1]} ({p[2]:.0f}%) at {p[4] or '—'}"):
                            if p[3]:
                                st.markdown(p[3])
                            else:
                                st.caption("No report text")
                else:
                    st.info("No pipeline results yet")
            except Exception as e:
                st.error(f"Error: {e}")

    elif log_source.startswith("Model"):
        st.markdown("**Trained Models**")
        model_dir = "./models"
        if os.path.exists(model_dir):
            models = sorted(os.listdir(model_dir), reverse=True)
            if models:
                rows = []
                for m in models:
                    path = os.path.join(model_dir, m)
                    size_kb = os.path.getsize(path) / 1024
                    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
                    rows.append({"File": m, "Size": f"{size_kb:.0f} KB", "Modified": mtime})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No model files found")
        else:
            st.info("Models directory not found")


# ======================================================================
# GRAFANA MONITORING PAGE
# ======================================================================


def _grafana_summary_cards():
    """Show quick-glance summary cards above Grafana for seamless context."""
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # SPY data
    spy_state = load_spy_state()
    pred = spy_state.get("prediction", {})
    indicators = spy_state.get("indicators", {})

    with col1:
        close = indicators.get("last_close", 0)
        st.metric("SPY", f"${close:.2f}" if close else "—")
    with col2:
        direction = pred.get("direction", "—")
        conf = pred.get("confidence", 0)
        st.metric("Signal", direction, f"{conf:.0f}%")
    with col3:
        vix = indicators.get("vix", 0)
        st.metric("VIX", f"{vix:.1f}" if vix else "—")

    # ES data
    es_state = load_es_state()
    with col4:
        pnl = es_state.get("daily_pnl", 0)
        st.metric("ES Daily P&L", f"${pnl:+,.0f}" if pnl else "$0")
    with col5:
        total = es_state.get("total_pnl", 0)
        st.metric("ES Total P&L", f"${total:+,.0f}" if total else "$0")
    with col6:
        wr = es_state.get("win_rate", 0)
        st.metric("Win Rate", f"{wr*100:.0f}%" if wr else "—")


def page_grafana():
    """Embed Grafana dashboards with seamless Streamlit integration."""

    # --- Quick-glance summary cards ---
    _grafana_summary_cards()
    st.divider()

    # --- Cross-navigation buttons ---
    nav_cols = st.columns([1, 1, 1, 1, 3])
    with nav_cols[0]:
        if st.button("📈 SPY Details", use_container_width=True):
            st.session_state["_nav_target"] = "📈 SPY Predictor"
            st.rerun()
    with nav_cols[1]:
        if st.button("📊 ES Details", use_container_width=True):
            st.session_state["_nav_target"] = "📊 ES Strategy"
            st.rerun()
    with nav_cols[2]:
        if st.button("🔬 What-If", use_container_width=True):
            st.session_state["_nav_target"] = "🔬 What-If Analysis"
            st.rerun()
    with nav_cols[3]:
        if st.button("⚙️ Admin", use_container_width=True):
            st.session_state["_nav_target"] = "⚙️ Admin"
            st.rerun()

    # --- Grafana dashboard selector ---
    # Use the DGX LAN IP so the browser (on the user's machine) can reach Grafana
    _dgx_ip = os.environ.get("DGX_IP", "192.168.1.211")
    proxy_host = os.environ.get("GRAFANA_PROXY_HOST", f"http://{_dgx_ip}:9190")
    grafana_host = os.environ.get("GRAFANA_HOST", f"http://{_dgx_ip}:3001")

    grafana_tab = st.radio(
        "Grafana Dashboard",
        ["SPY Predictor", "ES Strategy", "System Health", "Confidence API", "Pipeline Status"],
        horizontal=True,
        label_visibility="collapsed",
    )

    dashboard_map = {
        "SPY Predictor": "spy-predictor",
        "ES Strategy": "es-strategy",
        "System Health": "system-health",
        "Confidence API": "confidence-api",
        "Pipeline Status": "pipeline-status",
    }

    uid = dashboard_map.get(grafana_tab, "spy-predictor")

    # Build embed URL — go directly to Grafana (anonymous access is enabled).
    # Only route through the auth proxy when Google OAuth is active.
    user_info = get_user()
    use_proxy = (
        user_info
        and user_info.get("email", "").endswith("@local") is False
        and "@local" not in user_info.get("email", "")
    )
    token = get_session_token()
    if token and use_proxy:
        embed_url = (
            f"{proxy_host}/grafana-proxy/d/{uid}"
            f"?orgId=1&kiosk&auth_token={urllib.parse.quote(token)}"
        )
    else:
        embed_url = f"{grafana_host}/d/{uid}?orgId=1&kiosk"

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Grafana Settings**")
    height = st.sidebar.slider("Panel Height", 400, 1200, 800, 50)
    st.sidebar.caption(f"[Open full Grafana ↗]({grafana_host}/d/{uid})")

    user_info = get_user()
    if user_info and user_info.get("email") != "anonymous":
        st.sidebar.caption(f"Grafana user: {user_info['email']}")

    # Embed Grafana with JS bridge for intercepting navigation
    st.components.v1.html(
        f"""
        <style>
            iframe#grafana-embed {{
                width: 100%; height: {height}px; border: none;
                border-radius: 8px; background: #181b1f;
            }}
        </style>
        <iframe id="grafana-embed" src="{embed_url}"></iframe>
        <script>
            // Intercept clicks inside Grafana that try to navigate away
            window.addEventListener('message', function(e) {{
                if (e.data && e.data.type === 'navigate') {{
                    // Could be used to trigger Streamlit navigation in the future
                    console.log('Grafana navigation:', e.data.url);
                }}
            }});
        </script>
        """,
        height=height + 10,
    )


# ======================================================================
# ROUTER
# ======================================================================

# Handle cross-page navigation from Grafana page buttons
if "_nav_target" in st.session_state:
    page = st.session_state.pop("_nav_target")

if page == "📈 SPY Predictor":
    page_spy()
    time.sleep(15)
    st.rerun()
elif page == "📊 ES Strategy":
    page_es()
    time.sleep(5)
    st.rerun()
elif page == "🔬 What-If Analysis":
    page_whatif()
elif page == "📉 Monitoring":
    page_monitoring()
elif page == "📉 Grafana (compare)":
    page_grafana()
elif page == "⚙️ Admin":
    page_admin()
