"""🔮 Forecast Page — LSTM multi-day price prediction UI.

Displays 5-day price forecast with chart and insight box.
Includes a live price ticker that auto-refreshes via yfinance.
"""

import os
import sqlite3
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = "./data"
MODELS_DIR = "./models"

# Theme-aware colors and layout
from src.dashboard.theme import (
    get_colors as _get_theme_colors,
    get_plotly_layout as _get_theme_layout,
    page_header, is_dark,
)


def _refresh_theme():
    global DARK_LAYOUT, COLORS
    COLORS = _get_theme_colors()
    DARK_LAYOUT = _get_theme_layout()

COLORS = _get_theme_colors()
DARK_LAYOUT = _get_theme_layout()


@st.cache_data(ttl=600)
def _load_prices(ticker: str = "SPY", period: str = "1y") -> pd.DataFrame:
    """Load price data — from DB for SPY, yfinance for others."""
    if ticker == "SPY":
        try:
            conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM prices ORDER BY date",
                conn,
            )
            conn.close()
            if not df.empty:
                return df
        except Exception:
            pass
    # Fallback to yfinance
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index()
        data.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in data.columns]
        data["date"] = data["date"].dt.strftime("%Y-%m-%d")
        return data[["date", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def page_forecast():
    """Render the 🔮 Forecast page."""
    _refresh_theme()
    # ── Compact toolbar: title + controls on one line ──
    _t, _c1, _c2 = st.columns([4, 1, 1])
    with _t:
        st.markdown(page_header('🔮 Price Forecast'), unsafe_allow_html=True)
    with _c1:
        ticker = st.selectbox("Ticker", ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
                              key="forecast_ticker", label_visibility="collapsed")
    with _c2:
        forecast_days = st.select_slider("Days", options=list(range(3, 11)), value=5,
                                         key="forecast_days", label_visibility="collapsed")

    prices = _load_prices(ticker)
    if prices.empty:
        st.warning(f"No price data available for {ticker}")
        return

    last_close = float(prices["close"].iloc[-1])
    last_date = prices["date"].iloc[-1]

    # Try to load trained LSTM model
    model_path = os.path.join(MODELS_DIR, "lstm_predictor")
    forecast_df = pd.DataFrame()
    model_info = ""

    try:
        from src.model.lstm_predictor import LSTMPredictor
        predictor = LSTMPredictor(n_future=forecast_days)

        if os.path.exists(os.path.join(model_path, "meta.pkl")):
            predictor.load(model_path)
            # Override n_future to match user's slider selection
            predictor.n_future = forecast_days
            forecast_df = predictor.predict(prices)
            if predictor.history:
                # Show MAE in scaled units (0-1 range from MinMaxScaler)
                mae = predictor.history.get("val_mae",
                       predictor.history.get("mae",
                       predictor.history.get("val_loss",
                       predictor.history.get("loss", 0))))
                model_info = f"LSTM model · MAE: {mae:.4f}"
            else:
                model_info = "Pre-trained LSTM"
        else:
            # Train on the fly with available data
            st.info("No pre-trained model found. Training on available data...")
            predictor = LSTMPredictor(n_future=forecast_days, epochs=25)
            with st.spinner("Training LSTM..."):
                metrics = predictor.fit(prices, verbose=0)
            if "error" not in metrics:
                predictor.save(model_path)
                forecast_df = predictor.predict(prices)
                val_loss = metrics.get("val_mae", metrics.get("mae",
                           metrics.get("val_loss", metrics.get("loss", 0))))
                model_info = f"Just trained · MAE: {val_loss:.4f}"
            else:
                st.error(f"Training failed: {metrics.get('error')}")
    except ImportError:
        st.warning("TensorFlow not installed. Install with: `pip install tensorflow`")
    except Exception as e:
        st.warning(f"LSTM forecast unavailable: {e}")
        logger.exception("LSTM forecast error")

    if forecast_df.empty:
        # Show just the historical chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices["date"].tail(90), y=prices["close"].tail(90),
            name=f"{ticker} Close", line=dict(color="#2962FF", width=2),
        ))
        fig.update_layout(**DARK_LAYOUT, title=dict(text=f"{ticker} Price History", font=dict(color=COLORS["text"], size=14)),
                          height=400, yaxis_title="USD")
        st.plotly_chart(fig, use_container_width=True)
        return

    # ── Insight box + Forecast table ──
    pred_last = float(forecast_df["predicted_close"].iloc[-1])
    pct_change = (pred_last - last_close) / last_close * 100
    direction = "📈 UP" if pct_change > 0.3 else "📉 DOWN" if pct_change < -0.3 else "➡️ FLAT"
    dir_color = COLORS["green"] if pct_change > 0 else COLORS["red"] if pct_change < 0 else COLORS["yellow"]

    col_table, col_insight = st.columns([2, 1])

    with col_table:
        st.markdown("**Forecast Table**")
        display_df = forecast_df[["date", "day", "predicted_close"]].copy()
        display_df["predicted_close"] = display_df["predicted_close"].map(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with col_insight:
        st.markdown(
            f'<div style="background:{COLORS["card"]}; {"backdrop-filter:blur(12px);" if is_dark() else "box-shadow:0 1px 3px rgba(0,0,0,0.08);"} border:1px solid {COLORS["card_border"]}; border-radius:10px; padding:20px;">'
            f'<div style="color:{COLORS["text_secondary"]}; font-size:0.8em;">FORECAST INSIGHT</div>'
            f'<div style="color:{dir_color}; font-size:1.8em; font-weight:bold; margin:8px 0;">{direction}</div>'
            f'<div style="color:{COLORS["text"]};">Current: <b>${last_close:,.2f}</b></div>'
            f'<div style="color:{COLORS["text"]};">Day {len(forecast_df)}: <b>${pred_last:,.2f}</b></div>'
            f'<div style="color:{dir_color}; font-size:1.2em; margin-top:8px;">{pct_change:+.2f}%</div>'
            f'<div style="color:{COLORS["text_secondary"]}; font-size:0.75em; margin-top:12px;">{model_info}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Chart: Historical + Forecast ──
    hist_tail = prices.tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_tail["date"], y=hist_tail["close"],
        name=f"{ticker} Historical", line=dict(color="#2962FF", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[hist_tail["date"].iloc[-1]] + forecast_df["date"].tolist(),
        y=[last_close] + forecast_df["predicted_close"].tolist(),
        name="Forecast", line=dict(color=COLORS["yellow"], width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=6, color=COLORS["yellow"]),
    ))
    # "Today" divider — use annotation instead of add_vline to avoid type mismatch
    fig.add_annotation(
        x=str(last_date), y=1, yref="paper",
        text="Today", showarrow=False,
        font=dict(color=COLORS["text_secondary"], size=10),
        yanchor="bottom",
    )
    fig.add_shape(
        type="line", x0=str(last_date), x1=str(last_date),
        y0=0, y1=1, yref="paper",
        line=dict(color=COLORS["border"], width=1, dash="dot"),
    )
    actual_days = len(forecast_df)
    fig.update_layout(**DARK_LAYOUT,
                      title=dict(text=f"{ticker} — {actual_days}-Day Forecast", font=dict(color=COLORS["text"], size=14)),
                      height=350, yaxis_title="USD")
    st.plotly_chart(fig, use_container_width=True)
