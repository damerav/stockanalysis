"""4A. SPY/SPX Predictor Dashboard — Streamlit app on port 8501."""

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

st.set_page_config(page_title="SPY/SPX Predictor", layout="wide", page_icon="📈")


def load_state() -> dict:
    """Load state from local JSON or cloud relay."""
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
    """Load recent predictions from SQLite (local mode only)."""
    if IS_CLOUD:
        return pd.DataFrame()
    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
        df = pd.read_sql_query(
            f"SELECT date, direction, confidence FROM predictions ORDER BY date DESC LIMIT {n}",
            conn
        )
        conn.close()
        return df.iloc[::-1]  # reverse to chronological
    except Exception:
        return pd.DataFrame()


def load_performance() -> pd.DataFrame:
    """Load performance tracking from SQLite."""
    if IS_CLOUD:
        return pd.DataFrame()
    try:
        import sqlite3
        conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
        df = pd.read_sql_query(
            "SELECT date, predicted, actual, correct, cumulative_accuracy "
            "FROM performance ORDER BY date DESC LIMIT 30", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# --- Main Layout ---
state = load_state()
prediction = state.get("prediction", {})
indicators = state.get("indicators", {})
flow_alerts = state.get("flow_alerts", [])
updated_at = state.get("updated_at", "")

# Title
st.title("📈 SPY/SPX Predictor")

# --- Prediction Banner ---
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
        </p>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.info("Waiting for prediction data...")

# --- Two-column layout ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Prediction History")
    hist_df = load_prediction_history(20)
    if not hist_df.empty:
        import plotly.graph_objects as go

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
        fig.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Confidence %", xaxis_title="",
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No prediction history yet")

    # Performance tracking
    perf_df = load_performance()
    if not perf_df.empty:
        st.subheader("Accuracy Tracking")
        latest_acc = perf_df.iloc[0]["cumulative_accuracy"] if len(perf_df) > 0 else 0
        st.metric("Cumulative Accuracy", f"{latest_acc:.1%}")
        st.dataframe(perf_df.head(10), use_container_width=True, hide_index=True)

with col2:
    st.subheader("Key Indicators")
    if indicators:
        ind_col1, ind_col2 = st.columns(2)
        with ind_col1:
            st.metric("RSI(14)", f"{indicators.get('rsi_14', 'N/A')}")
            st.metric("MACD", f"{indicators.get('macd', 'N/A')}")
            st.metric("ATR(14)", f"{indicators.get('atr_14', 'N/A')}")
        with ind_col2:
            st.metric("VIX", f"{indicators.get('vix', 'N/A')}",
                      delta=f"{indicators.get('vix_change', 0):+.1f}" if indicators.get('vix_change') else None)
            st.metric("Vol Ratio", f"{indicators.get('volume_ratio', 'N/A')}")
            st.metric("Sentiment", f"{indicators.get('sentiment_score', 'N/A')}")
    else:
        st.caption("Waiting for indicator data...")

# --- Options Flow Alerts ---
st.subheader("Options Flow Alerts")
if flow_alerts:
    for alert in flow_alerts[-15:]:  # show last 15
        direction_emoji = "🔴" if alert.get("direction") == "PUT" else "🟢"
        alert_type = alert.get("type", "")
        symbol = alert.get("symbol", "")
        notional = alert.get("notional", 0)
        ts = alert.get("timestamp", "")[:19]
        legs = alert.get("legs", "")
        legs_str = f" ({legs}×)" if legs else ""

        st.markdown(
            f"{direction_emoji} **{ts}** {alert.get('direction', '')} "
            f"{alert_type} {symbol} ${notional:,.0f}{legs_str}"
        )
else:
    st.caption("No options flow alerts yet")

# --- Footer ---
st.divider()
mode_label = "☁️ Cloud" if IS_CLOUD else "🖥️ Local"
st.caption(f"{mode_label} mode | Last updated: {updated_at or 'N/A'}")

# --- Auto-refresh (15 seconds) ---
time.sleep(15)
st.rerun()
