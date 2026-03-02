"""4B. ES Futures Strategy Dashboard — Streamlit app on port 8502."""

import os
import json
import time
import requests
import streamlit as st
import pandas as pd
from datetime import datetime

# --- Mode Detection ---
RELAY_URL = os.environ.get("RELAY_URL", "")
IS_CLOUD = bool(RELAY_URL)
DATA_DIR = "./data"

st.set_page_config(page_title="ES Strategy", layout="wide", page_icon="📊")


def load_state() -> dict:
    """Load ES state from local JSON or cloud relay."""
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


# --- Load State ---
state = load_state()
position = state.get("position", {"status": "FLAT", "lots": 0})
signals = state.get("signals", [])
regime = state.get("regime", "Med")
pnl = state.get("pnl", {"daily": 0.0, "unrealized": 0.0})
chart_data = state.get("chart_data", {})
updated_at = state.get("updated_at", "")

st.title("📊 ES Futures Strategy")

# --- Position Banner ---
pos_status = position.get("status", "FLAT")
pos_lots = position.get("lots", 0)
entry_price = position.get("entry_price", 0)
unrealized = pnl.get("unrealized", 0)
daily_pnl = pnl.get("daily", 0)

regime_colors = {"Low": "var(--color-positive-500)", "Med": "var(--color-warning-500)", "High": "var(--color-negative-500)"}
pos_colors = {"LONG": "var(--color-positive-500)", "SHORT": "var(--color-negative-500)", "FLAT": "var(--color-text-tertiary)"}

banner_color = pos_colors.get(pos_status, "var(--color-text-tertiary)")
regime_color = regime_colors.get(regime, "var(--color-warning-500)")

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

# --- Chart ---
st.subheader("Price Chart")
bars = chart_data.get("bars", [])
if bars:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        df = pd.DataFrame(bars)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.75, 0.25], vertical_spacing=0.05)

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df["timestamp"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"], name="ES",
            ), row=1, col=1)

            # Keltner Channel bands (shaded)
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

            # KC Mid (dashed)
            if "kc_mid" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["kc_mid"], mode="lines",
                    line=dict(color="cornflowerblue", dash="dash", width=1),
                    name="KC Mid",
                ), row=1, col=1)

            # VWAP overlay (dotted)
            if "vwap" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["vwap"], mode="lines",
                    line=dict(color="orange", dash="dot", width=1), name="VWAP",
                ), row=1, col=1)

            # Entry/exit markers
            entries = chart_data.get("entries", [])
            for e in entries:
                marker = "triangle-up" if e.get("direction") == "LONG" else "triangle-down"
                color = "green" if e.get("direction") == "LONG" else "red"
                fig.add_trace(go.Scatter(
                    x=[e["timestamp"]], y=[e["price"]], mode="markers",
                    marker=dict(symbol=marker, size=14, color=color),
                    name=e.get("label", "Entry"), showlegend=False,
                ), row=1, col=1)

            exits = chart_data.get("exits", [])
            for ex in exits:
                fig.add_trace(go.Scatter(
                    x=[ex["timestamp"]], y=[ex["price"]], mode="markers",
                    marker=dict(symbol="x", size=12, color="blue"),
                    name=ex.get("label", "Exit"), showlegend=False,
                ), row=1, col=1)

            # Stop levels (horizontal dashed red)
            stops = chart_data.get("stop_levels", [])
            for s in stops:
                fig.add_hline(y=s, line_dash="dash", line_color="red",
                              opacity=0.5, row=1, col=1)

            # RSI subplot
            if "rsi" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=df["rsi"], mode="lines",
                    line=dict(color="purple", width=1), name="RSI(14)",
                ), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red",
                              opacity=0.3, row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green",
                              opacity=0.3, row=2, col=1)

            fig.update_layout(
                height=500, margin=dict(l=20, r=20, t=30, b=20),
                xaxis_rangeslider_visible=False, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

            st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.warning("Install plotly: pip install plotly")
else:
    st.caption("Waiting for chart data...")

# --- Signal Feed + Status Panel ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Signal Feed")
    if signals:
        for sig in reversed(signals[-20:]):
            ts = sig.get("timestamp", "")[:8]
            sig_type = sig.get("type", "")
            detail = sig.get("detail", "")

            type_colors = {
                "ENTRY_LONG": "🟢", "ENTRY_SHORT": "🔴",
                "EXIT_TP1": "💰", "EXIT_TP2": "💰", "EXIT_RUNNER": "🏃",
                "STOP_HIT": "🛑", "STOP_UPDATE": "📍",
                "AI_REJECT": "🤖", "CIRCUIT_BREAKER": "⚡",
                "SESSION_FLATTEN": "🕐",
            }
            emoji = type_colors.get(sig_type, "📌")
            st.text(f"{emoji} {ts} {sig_type} {detail}")
    else:
        st.caption("No signals yet")

with col2:
    st.subheader("Status Panel")

    # Circuit breaker
    cb_status = state.get("circuit_breaker", "OK")
    cb_color = "🟢" if cb_status == "OK" else "🔴"
    st.markdown(f"{cb_color} Circuit Breaker: **{cb_status}**")

    # Daily P&L
    pnl_color = "green" if daily_pnl >= 0 else "red"
    st.metric("Daily P&L", f"${daily_pnl:+,.0f}")

    # Trade count
    trade_count = state.get("trade_count", 0)
    st.metric("Trades Today", trade_count)

    # Session status
    session = state.get("session_status", "Inactive")
    st.markdown(f"Session: **{session}**")

    # Per-lot status
    lots_detail = position.get("lots_detail", [])
    if lots_detail:
        st.subheader("Lot Status")
        for lot in lots_detail:
            lot_id = lot.get("id", "?")
            lot_status = lot.get("status", "?")
            lot_pnl = lot.get("pnl", 0)
            st.text(f"  Lot {lot_id}: {lot_status} (${lot_pnl:+,.0f})")

# --- Footer ---
st.divider()
mode_label = "☁️ Cloud" if IS_CLOUD else "🖥️ Local"
st.caption(f"{mode_label} mode | Last updated: {updated_at or 'N/A'}")

# --- Auto-refresh (5 seconds) ---
time.sleep(5)
st.rerun()
