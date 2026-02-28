"""Unified Dashboard — All dashboards on a single port.

Usage:
    streamlit run src/dashboard/app.py --server.port 8501 --server.headless true
"""

import os
import sys
import json
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
    db_list_users,
    db_create_user,
    db_update_user,
    db_delete_user,
    db_user_count,
)
from src.dashboard.monitoring import page_monitoring
from src.dashboard.forecast_app import page_forecast
from src.dashboard.single_stock_app import page_single_stock
from src.data.db_router import get_router, ANALYTICS_TABLES
from src.data.fetcher import FallbackFetcher
from src.dashboard.theme import (
    get_theme, get_colors, get_plotly_layout, get_title_font,
    metric_card as _theme_metric_card, badge_html as _theme_badge,
    page_header, render_theme_toggle, theme_css, is_dark,
)

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)
def _fetch_live_macro() -> dict:
    """Cached live FRED macro values (5-min TTL)."""
    try:
        config = _load_config_cached()
        fetcher = FallbackFetcher(config=config)
        return fetcher.get_macro_fred()
    except Exception:
        return {}


@st.cache_data(ttl=300)
def _fetch_live_news_counts() -> dict:
    """Cached live news source counts (5-min TTL)."""
    try:
        config = _load_config_cached()
        fetcher = FallbackFetcher(config=config)
        finnhub = fetcher.get_news_finnhub()
        rss = fetcher.get_news_rss()
        rss_by_source = {}
        for a in rss:
            src = a.get("source", "unknown")
            rss_by_source[src] = rss_by_source.get(src, 0) + 1
        return {
            "finnhub_count": len(finnhub),
            "finnhub_headline": finnhub[0].get("headline", "") if finnhub else "",
            "rss_by_source": rss_by_source,
            "rss_total": len(rss),
        }
    except Exception:
        return {}


@st.cache_data(ttl=600)
def _load_config_cached() -> dict:
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

# --- Mode Detection ---
RELAY_URL = os.environ.get("RELAY_URL", "")
IS_CLOUD = bool(RELAY_URL)
DATA_DIR = "./data"

st.set_page_config(page_title="Stock Analysis", layout="wide", page_icon="📊")

# --- Dynamic theme CSS ---
st.markdown(f"<style>{theme_css()}</style>", unsafe_allow_html=True)

# --- Load external CSS for pill tabs + sidebar styling ---
_css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(_css_path):
    with open(_css_path) as _css_f:
        st.markdown(f"<style>{_css_f.read()}</style>", unsafe_allow_html=True)

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

# --- Sidebar header ---
st.sidebar.title("📊 Stock Analysis")
if user:
    st.sidebar.caption(f"👤 {user.get('name', user.get('email', ''))}")
render_theme_toggle()

# NOTE: st.navigation is called after all page functions are defined (see bottom of file)


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


def load_prediction_history(n: int = 30) -> pd.DataFrame:
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
    c = get_colors()
    state = load_spy_state()
    prediction = state.get("prediction", {})
    indicators = state.get("indicators", {})
    flow_alerts = state.get("flow_alerts", [])
    updated_at = state.get("updated_at", "")

    # --- Compact header ---
    st.markdown(page_header('📈 SPY/SPX Predictor'), unsafe_allow_html=True)

    direction = prediction.get("direction", "NEUTRAL")
    scale_label = prediction.get("scale_label", "NEUTRAL")
    confidence = prediction.get("confidence", 0)
    probs = prediction.get("probabilities", {})

    color_map = {
        "STRONG_BULLISH": c["green"], "BULLISH": c["green"],
        "NEUTRAL": c["yellow"],
        "BEARISH": c["red"], "STRONG_BEARISH": c["red"],
    }
    banner_color = color_map.get(scale_label, c["yellow"])

    # --- Compact prediction banner ---
    if prediction:
        # Confidence interpretation
        conf_interp = "weak" if confidence < 55 else "moderate" if confidence < 70 else "strong" if confidence < 85 else "very strong"
        st.markdown(
            f"""<div style="background:{banner_color}; padding:8px 16px; border-radius:8px;
            text-align:center; margin-bottom:6px; display:flex; align-items:center; justify-content:center; gap:20px;">
            <span style="color:#fff; font-size:1.2rem; font-weight:700;">{scale_label.replace('_', ' ')}</span>
            <span style="color:#fff; font-size:1.05rem; font-weight:600;">{confidence:.0f}%</span>
            <span style="color:rgba(255,255,255,0.85); font-size:0.85rem;">
            ↑{probs.get('up', 0):.0f}% · —{probs.get('neutral', 0):.0f}% · ↓{probs.get('down', 0):.0f}%
            </span>
            <span style="color:rgba(255,255,255,0.7); font-size:0.75rem; font-style:italic;">
            ({conf_interp} signal)
            </span></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for prediction data...")

    # --- Combined info row: Regime + Model + Conf Set + Macro (2 rows of 4) ---
    macro = _fetch_live_macro()
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        regime = prediction.get("regime", "")
        regime_labels = {
            "bull_trend": "🟢 Bull", "bear_trend": "🔴 Bear",
            "high_vol_choppy": "🟡 Choppy", "low_vol_range": "🔵 Range",
        }
        st.metric("Regime", regime_labels.get(regime, regime or "—"),
                  help="HMM-detected market regime: Bull Trend, Bear Trend, High-Vol Choppy, or Low-Vol Range. Affects neutral threshold and model weighting.")
    with r2:
        if prediction.get("ensemble_used"):
            st.metric("Model", "🔗 Ensemble",
                      help="Stacking ensemble (XGBoost + BiLSTM + LightGBM) with logistic meta-learner.")
        else:
            st.metric("Model", "🌲 XGB",
                      help="XGBoost gradient-boosted tree model with isotonic calibration.")
    with r3:
        pred_set = prediction.get("prediction_set", [])
        is_low_conv = prediction.get("is_low_conviction", False)
        if pred_set:
            set_str = "/".join(pred_set)
            st.metric("Conf. Set", f"{'⚠️' if is_low_conv else '✅'} {set_str}",
                      help="Conformal prediction set at 90% coverage. Multiple directions = low conviction. Single direction = high conviction.")
        else:
            st.metric("Conf. Set", "—",
                      help="Conformal prediction set — shows which directions are statistically plausible.")
    with r4:
        v = macro.get("vix") if macro else None
        vc = macro.get("vix_change") if macro else None
        st.metric("VIX", f"{v:.1f}" if v else "—",
                  delta=f"{vc:+.1f}" if vc else None, delta_color="inverse",
                  help="CBOE Volatility Index. <15 = low vol (range-bound), 15-25 = normal, >25 = high vol (trending). Inversely correlated with SPY.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        v = macro.get("us10y_yield") if macro else None
        st.metric("10Y Yield", f"{v:.2f}%" if v else "—",
                  help="US 10-Year Treasury yield. Rising yields = tighter financial conditions, typically bearish for equities.")
    with m2:
        v = macro.get("dxy") if macro else None
        st.metric("DXY", f"{v:.1f}" if v else "—",
                  help="US Dollar Index. Strong dollar = headwind for multinational earnings and risk assets.")
    with m3:
        v = macro.get("gold") if macro else None
        st.metric("Gold", f"${v:,.0f}" if v else "—",
                  help="Gold spot price. Safe-haven asset — rising gold often signals risk-off sentiment.")
    with m4:
        v = macro.get("crude") if macro else None
        st.metric("Crude", f"${v:.1f}" if v else "—",
                  help="WTI Crude Oil. Impacts energy sector and inflation expectations.")

    # --- P3: Earnings + Fed + Options (compact row) ---
    try:
        import sqlite3 as _sql
        _conn = _sql.connect(os.path.join(DATA_DIR, "spy.db"))
        _today = datetime.now().strftime("%Y-%m-%d")

        p3_col1, p3_col2, p3_col3 = st.columns(3)
        with p3_col1:
            from src.data.earnings_calendar import get_earnings_features as _get_earn
            earn = _get_earn(_conn, _today)
            density = earn.get("earnings_density", 0)
            days_next = earn.get("days_to_next_mega", 30)
            earn_week = earn.get("earnings_week", 0)
            st.metric("📅 Earnings", f"{density} mega-caps",
                      delta="Earnings Week" if earn_week else None,
                      delta_color="normal" if earn_week else "off")
            st.caption(f"Next in {days_next}d")

        with p3_col2:
            from src.data.fed_comms import get_fed_features as _get_fed
            fed = _get_fed(_conn, _today)
            avg = fed.get("fed_sentiment_avg", 0)
            label = "🦅 Hawkish" if avg > 0.2 else "🕊️ Dovish" if avg < -0.2 else "⚖️ Neutral"
            st.metric("Fed", label, delta=f"{avg:+.2f}")

        with p3_col3:
            opt_row = _conn.execute(
                "SELECT vanna_exposure, charm_exposure, zero_dte_pcr "
                "FROM options_analytics WHERE date = ? ORDER BY date DESC LIMIT 1",
                (_today,),
            ).fetchone()
            if opt_row and opt_row[0] is not None:
                st.metric("Vanna", f"{opt_row[0]:,.0f}")
                st.caption(f"Charm: {opt_row[1]:,.0f} | 0DTE P/C: {opt_row[2]:.2f}" if opt_row[1] else "")
            else:
                st.metric("Greeks", "—")

        _conn.close()
    except Exception:
        pass

    # --- Microstructure (collapsible) ---
    try:
        import sqlite3 as _sql2
        _conn2 = _sql2.connect(os.path.join(DATA_DIR, "spy.db"))
        _today2 = datetime.now().strftime("%Y-%m-%d")
        from src.data.features import compute_intraday_microstructure as _get_micro
        micro = _get_micro(_conn2, _today2)
        _conn2.close()

        import math
        has_data = any(not (isinstance(v, float) and math.isnan(v)) for v in micro.values())
        if has_data:
            with st.expander("🔬 Intraday Microstructure", expanded=False):
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    gap = micro.get("opening_gap_pct", 0) or 0
                    gap_emoji = "🟢" if gap > 0 else "🔴" if gap < 0 else "⚪"
                    st.metric("Gap", f"{gap_emoji} {gap:+.2%}")
                    orb = micro.get("opening_range_breakout", 0) or 0
                    orb_label = "▲ Break" if orb > 0 else "▼ Down" if orb < 0 else "— In"
                    st.caption(f"30m: {orb_label}")
                with mc2:
                    cvh = micro.get("close_vs_high_pct", 0) or 0
                    cvl = micro.get("close_vs_low_pct", 0) or 0
                    st.metric("Cls/High", f"{cvh:+.2%}")
                    st.caption(f"Cls/Low: {cvl:+.2%}")
                with mc3:
                    rev = micro.get("afternoon_reversal", 0) or 0
                    st.metric("PM", "⚡ Rev" if rev else "→ Cont")
                    ihv = micro.get("institutional_hour_vol", 0) or 0
                    st.caption(f"AM/PM: {ihv:.2f}")
                with mc4:
                    vrc = micro.get("vwap_reclaim_count", 0) or 0
                    st.metric("VWAP×", f"{int(vrc)}")
                    td = micro.get("tick_divergence", 0) or 0
                    st.caption(f"Tick div: {td:.3f}")
    except Exception:
        pass

    # --- SHAP drivers (compact) ---
    shap_drivers = prediction.get("shap_drivers", [])
    if shap_drivers:
        driver_df = pd.DataFrame(shap_drivers)
        fig_shap = go.Figure()
        colors = [c["green"] if v > 0 else c["red"] for v in driver_df["shap_value"]]
        fig_shap.add_trace(go.Bar(
            y=driver_df["feature"], x=driver_df["shap_value"],
            orientation="h", marker_color=colors,
            text=[f"{v:+.3f}" for v in driver_df["shap_value"]],
            textposition="outside",
            textfont=dict(color=c["text"], size=11),
            hovertemplate="Feature: %{y}<br>SHAP: %{x:.4f}<br>Value: %{customdata:.4f}",
            customdata=driver_df["feature_value"],
        ))
        _shap_bg = "rgba(0,0,0,0)" if is_dark() else c["surface"]
        fig_shap.update_layout(
            height=160, margin=dict(l=10, r=10, t=5, b=5),
            xaxis_title="SHAP", yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=_shap_bg,
            font=dict(color=c["text"], size=11),
            xaxis=dict(gridcolor=c["grid"], zerolinecolor=c["zeroline"]),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # --- Main content: History + Indicators side by side ---
    col1, col2 = st.columns([5, 3])

    with col1:
        st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.9rem;margin-bottom:4px;">PREDICTION HISTORY</p>',
                    unsafe_allow_html=True)
        hist_df = load_prediction_history(30)
        if not hist_df.empty:
            colors = hist_df["direction"].map({
                "BULLISH": c["green"], "STRONG_BULLISH": c["green"],
                "BEARISH": c["red"], "STRONG_BEARISH": c["red"],
                "NEUTRAL": c["text_secondary"],
            }).fillna(c["text_muted"])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hist_df["date"], y=hist_df["confidence"],
                marker_color=colors.tolist(),
                text=hist_df["direction"], textposition="outside",
                textfont=dict(color=c["text"], size=9),
                hovertemplate="Date: %{x}<br>Direction: %{text}<br>Confidence: %{y:.0f}%",
            ))
            _hist_bg = "rgba(0,0,0,0)" if is_dark() else c["surface"]
            fig.update_layout(
                height=220, margin=dict(l=10, r=10, t=5, b=25),
                yaxis_title="Conf %", yaxis_range=[0, 100],
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=_hist_bg,
                font=dict(color=c["text_secondary"], size=10),
                xaxis=dict(gridcolor=c["grid"], tickfont=dict(size=9),
                           tickformat="%b %d", dtick="D1"),
                yaxis=dict(gridcolor=c["grid"]),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No prediction history yet")

        # Accuracy tracking (collapsible)
        perf_df = load_performance()
        if not perf_df.empty:
            with st.expander("📊 Accuracy Tracking", expanded=False):
                latest_acc = perf_df.iloc[0]["cumulative_accuracy"] if len(perf_df) > 0 else 0
                st.metric("Cumulative Accuracy", f"{latest_acc:.1%}")

                if "confidence_tier" in perf_df.columns:
                    st.caption("By Confidence Tier")
                    for tier in ["high", "medium", "low"]:
                        tier_df = perf_df[perf_df["confidence_tier"] == tier]
                        if not tier_df.empty:
                            tier_acc = tier_df["correct"].mean()
                            st.markdown(
                                f'<span style="color:{c["text"]};">'
                                f'{"🟢" if tier_acc >= 0.55 else "🟡" if tier_acc >= 0.50 else "🔴"} '
                                f'{tier.title()}: {tier_acc:.1%} ({len(tier_df)})</span>',
                                unsafe_allow_html=True,
                            )
                if "vix_regime" in perf_df.columns:
                    st.caption("By VIX Regime")
                    for regime in ["low", "normal", "high"]:
                        reg_df = perf_df[perf_df["vix_regime"] == regime]
                        if not reg_df.empty:
                            reg_acc = reg_df["correct"].mean()
                            st.markdown(
                                f'<span style="color:{c["text"]};">'
                                f'VIX {regime}: {reg_acc:.1%} ({len(reg_df)})</span>',
                                unsafe_allow_html=True,
                            )

                st.dataframe(perf_df.head(10), use_container_width=True, hide_index=True)

    with col2:
        st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.9rem;margin-bottom:4px;">KEY INDICATORS</p>',
                    unsafe_allow_html=True)
        if indicators:
            # Use macro data for VIX to stay consistent with the top row
            _vix_val = macro.get("vix") if macro else None
            _vix_chg = macro.get("vix_change") if macro else None
            if _vix_val is None:
                _vix_val = indicators.get("vix")
                _vix_chg = indicators.get("vix_change")
            ic1, ic2 = st.columns(2)
            with ic1:
                st.metric("RSI(14)", f"{indicators.get('rsi_14', 'N/A')}",
                          help="RSI 14-day. >70 overbought, <30 oversold.")
                st.metric("MACD", f"{indicators.get('macd', 'N/A')}",
                          help="MACD. Positive = bullish, negative = bearish.")
                st.metric("ATR(14)", f"{indicators.get('atr_14', 'N/A')}",
                          help="Average True Range 14-day volatility.")
            with ic2:
                st.metric("VIX", f"{_vix_val:.1f}" if _vix_val else "N/A",
                          delta=f"{_vix_chg:+.1f}" if _vix_chg else None, delta_color="inverse",
                          help="CBOE Volatility Index.")
                st.metric("Vol Ratio", f"{indicators.get('volume_ratio', 'N/A')}",
                          help="Volume vs 20-day avg. >1.5 = high.")
                st.metric("Sentiment", f"{indicators.get('sentiment_score', 'N/A')}",
                          help="News sentiment -1 to +1.")
        else:
            st.caption("Waiting for indicator data...")

        # Options flow (compact, in expander)
        with st.expander(f"⚡ Options Flow ({len(flow_alerts)} alerts)", expanded=bool(flow_alerts)):
            if flow_alerts:
                for alert in flow_alerts[-8:]:
                    direction_emoji = "🔴" if alert.get("direction") == "PUT" else "🟢"
                    notional = alert.get("notional", 0)
                    legs = alert.get("legs", "")
                    legs_str = f" ({legs}×)" if legs else ""
                    st.markdown(
                        f'<span style="color:{c["text"]}; font-family:monospace; font-size:0.8rem;">'
                        f'{direction_emoji} {alert.get("timestamp", "")[:16]} '
                        f'{alert.get("direction", "")} {alert.get("type", "")} '
                        f'${notional:,.0f}{legs_str}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No alerts yet")

    # Staleness indicator
    _stale_color = c["text_secondary"]
    _stale_label = ""
    if updated_at:
        try:
            _upd_dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00")) if "T" in updated_at else datetime.strptime(updated_at[:19], "%Y-%m-%d %H:%M:%S")
            _age_min = (datetime.now() - _upd_dt).total_seconds() / 60
            if _age_min > 60:
                _stale_color = "#EF4444"
                _stale_label = " ⚠️ STALE"
            elif _age_min > 30:
                _stale_color = "#F59E0B"
                _stale_label = " ⏳"
        except Exception:
            pass
    st.markdown(f'<p style="color:{_stale_color}; font-size:0.75rem; text-align:right; margin-top:4px;">'
                f'Updated: {updated_at or "N/A"}{_stale_label}</p>', unsafe_allow_html=True)


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
    c = get_colors()
    state = load_es_state()
    position = state.get("position", {"status": "FLAT", "lots": 0})
    signals = state.get("signals", [])
    regime = state.get("regime", "Med")
    pnl = state.get("pnl", {"daily": 0.0, "unrealized": 0.0})
    chart_data = state.get("chart_data", {})
    updated_at = state.get("updated_at", "")

    st.markdown(page_header('📊 ES Futures Strategy'), unsafe_allow_html=True)

    pos_status = position.get("status", "FLAT")
    pos_lots = position.get("lots", 0)
    entry_price = position.get("entry_price", 0)
    unrealized = pnl.get("unrealized", 0)
    daily_pnl = pnl.get("daily", 0)

    regime_colors = {"Low": c["green"], "Med": c["yellow"], "High": c["red"]}
    pos_colors = {"LONG": c["green"], "SHORT": c["red"], "FLAT": c["text_secondary"]}
    banner_color = pos_colors.get(pos_status, c["text_secondary"])
    regime_color = regime_colors.get(regime, c["yellow"])
    # Use dark text on yellow/green badges for WCAG contrast
    regime_text = "#1E2329" if regime in ("Low", "Med") else "#FFFFFF"

    st.markdown(
        f"""<div style="background-color:{banner_color}; padding:10px; border-radius:8px;
        text-align:center; margin-bottom:10px; display:flex; justify-content:space-around; align-items:center;">
        <div><h2 style="color:white; margin:0; font-size:1.1rem;">{pos_status} {pos_lots} lots</h2></div>
        <div><p style="color:white; margin:0; font-size:0.9rem;">Entry: {entry_price}</p></div>
        <div><p style="color:white; margin:0; font-size:0.9rem;">P&L: ${unrealized:+,.0f}</p></div>
        <div style="background-color:{regime_color}; padding:4px 12px; border-radius:5px;">
        <p style="color:{regime_text}; margin:0; font-size:0.9rem;">Regime: {regime}</p></div>
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

            _bg = "rgba(0,0,0,0)" if is_dark() else c["surface"]
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20),
                              xaxis_rangeslider_visible=False, showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=_bg,
                              font=dict(color=c["text"], size=11))
            fig.update_xaxes(gridcolor=c["grid"])
            fig.update_yaxes(gridcolor=c["grid"])
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No chart data available. This typically means the intraday data pipeline "
                   "(Polygon.io) is not running or the API key is not configured. "
                   "Check Admin → System Status → Data Sources for connection status.")

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
            type_labels = {
                "ENTRY_LONG": "Entered Long",
                "ENTRY_SHORT": "Entered Short",
                "EXIT_TP1": "Take-Profit 1 Hit",
                "EXIT_TP2": "Take-Profit 2 Hit",
                "EXIT_RUNNER": "Runner Exited",
                "STOP_HIT": "Stop-Loss Hit",
                "STOP_UPDATE": "Stop Updated",
                "AI_REJECT": "AI Rejected Signal",
                "CIRCUIT_BREAKER": "Circuit Breaker",
                "SESSION_FLATTEN": "Session Flattened",
            }
            for sig in reversed(signals[-20:]):
                sig_type = sig.get("type", "")
                emoji = type_emojis.get(sig_type, "📌")
                label = type_labels.get(sig_type, sig_type)
                detail = sig.get("detail", "")
                # Make detail human-readable
                if sig_type == "AI_REJECT" and "p_enter=" in detail:
                    import re as _re
                    m = _re.search(r"p_enter=([\d.]+)\s*<\s*([\d.]+)", detail)
                    if m:
                        detail = f"confidence {float(m.group(1)):.0%} below {float(m.group(2)):.0%} threshold"
                elif sig_type in ("ENTRY_LONG", "ENTRY_SHORT") and "@" in detail:
                    detail = detail.replace("@", "at")
                st.markdown(
                    f'<span style="color:{c["text"]}; font-family:monospace; font-size:0.85em;">'
                    f'{emoji} {sig.get("timestamp", "")[:8]} {label} '
                    f'{detail}</span>',
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
                    f'<span style="color:{c["text"]}; font-family:monospace;">'
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
    st.markdown(page_header('🔬 What-If Analysis'), unsafe_allow_html=True)
    engine = get_whatif_engine()

    tab_es, tab_spy = st.tabs(["📈 ES Strategy", "🔮 SPY Predictor"])

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
    st.markdown(page_header('⚙️ Admin Console'), unsafe_allow_html=True)

    tab_status, tab_actions, tab_users, tab_db, tab_config, tab_logs = st.tabs([
        "ℹ️ System Status", "▶️ Actions", "👤 Users", "🗃️ Database", "📝 Configuration", "📜 Logs",
    ])

    with tab_status:
        _admin_status_tab()
    with tab_actions:
        _admin_actions_tab()
    with tab_users:
        _admin_users_tab()
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
        duck_path = os.path.join(DATA_DIR, "analytics.duckdb")
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)
            duck_size = ""
            if os.path.exists(duck_path):
                duck_mb = os.path.getsize(duck_path) / (1024 * 1024)
                duck_size = f" + 🦆 {duck_mb:.1f} MB"
            st.success(f"Online — SQLite {size_mb:.1f} MB{duck_size}")
            try:
                conn = sqlite3.connect(db_path)
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
                conn.close()
                st.caption(f"{len(tables)} SQLite tables" + (" + 5 DuckDB" if os.path.exists(duck_path) else ""))
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
            # Enhancement 26: Check DuckDB for analytics tables
            duck_router = None
            try:
                config = _load_config()
                duck_router = get_router(config)
            except Exception:
                pass

            tables = ["prices", "technicals", "news", "daily_sentiment", "macro",
                       "predictions", "intraday_bars", "options_chain",
                       "options_analytics", "intraday_features", "performance",
                       "earnings_calendar", "fed_communications"]
            rows = []
            for t in tables:
                try:
                    if duck_router and t in ANALYTICS_TABLES:
                        count_df = duck_router.read_analytics(f"SELECT COUNT(*) as cnt FROM {t}")
                        count = int(count_df.iloc[0]["cnt"]) if not count_df.empty else 0
                        if t != "intraday_bars":
                            min_df = duck_router.read_analytics(f"SELECT MIN(date) as d FROM {t}")
                            max_df = duck_router.read_analytics(f"SELECT MAX(date) as d FROM {t}")
                            min_date = min_df.iloc[0]["d"] if not min_df.empty else "—"
                            max_date = max_df.iloc[0]["d"] if not max_df.empty else "—"
                        else:
                            min_date, max_date = "—", "—"
                        source = "🦆"
                    else:
                        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                        min_date = conn.execute(f"SELECT MIN(date) FROM {t}").fetchone()[0] if t != "intraday_bars" else "—"
                        max_date = conn.execute(f"SELECT MAX(date) FROM {t}").fetchone()[0] if t != "intraday_bars" else "—"
                        source = "📦"
                except Exception:
                    count, min_date, max_date, source = 0, "—", "—", "?"
                rows.append({"Table": t, "Rows": count, "From": min_date or "—",
                             "To": max_date or "—", "DB": source})
            conn.close()
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("🦆 = DuckDB analytics, 📦 = SQLite operational")
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

    # Live Data Sources Status
    st.subheader("📡 Data Sources")
    ds1, ds2 = st.columns(2)

    with ds1:
        # FRED API
        st.markdown("**FRED API**")
        macro = _fetch_live_macro()
        if macro and any(v is not None for k, v in macro.items() if k != "vix_change"):
            items = []
            for k in ["vix", "us10y_yield", "dxy", "fed_funds", "gold", "crude"]:
                v = macro.get(k)
                items.append(f"{k}: {v:.2f}" if v is not None else f"{k}: —")
            st.success("Online — " + " | ".join(items))
        else:
            st.error("FRED unavailable or no data")

        # yfinance
        st.markdown("**yfinance**")
        try:
            import yfinance as yf
            spy_data = yf.download("SPY", period="2d", progress=False)
            if not spy_data.empty:
                if isinstance(spy_data.columns, pd.MultiIndex):
                    spy_data.columns = spy_data.columns.get_level_values(0)
                last_close = float(spy_data["Close"].iloc[-1])
                last_date = str(spy_data.index[-1].date())
                st.success(f"Online — SPY ${last_close:.2f} ({last_date})")
            else:
                st.warning("yfinance returned no data")
        except Exception as e:
            st.error(f"yfinance offline: {e}")

    with ds2:
        # Finnhub + RSS
        st.markdown("**Finnhub + RSS Feeds**")
        news = _fetch_live_news_counts()
        if news:
            fh_count = news.get("finnhub_count", 0)
            fh_headline = news.get("finnhub_headline", "")
            rss_total = news.get("rss_total", 0)
            rss_src = news.get("rss_by_source", {})
            if fh_count > 0:
                st.success(f"Finnhub: {fh_count} articles")
                if fh_headline:
                    st.caption(f"Latest: {fh_headline[:80]}...")
            else:
                st.warning("Finnhub: 0 articles (check API key)")
            if rss_total > 0:
                parts = [f"{src}: {cnt}" for src, cnt in rss_src.items()]
                st.success(f"RSS: {rss_total} articles — " + ", ".join(parts))
            else:
                st.warning("RSS: 0 articles")
        else:
            st.error("News fetch failed")

        # Ollama
        st.markdown("**Ollama / DeepSeek**")
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                config = _load_config()
                target = config.get("llm", {}).get("model", "deepseek-r1:70b")
                if any(target in n for n in models):
                    st.success(f"Online — {target}")
                else:
                    st.warning(f"Running but {target} not loaded")
            else:
                st.error("Ollama not responding")
        except Exception:
            st.error("Ollama offline")


# --- Users Tab ---

def _admin_users_tab():
    c = get_colors()
    st.subheader("User Management")

    config = _load_config()
    auth = config.get("auth", {})
    mode = auth.get("mode", "local")

    # Current user info
    current_user = get_user()
    current_role = current_user.get("role", "viewer") if current_user else "viewer"
    is_admin = current_role == "admin"

    st.markdown(f'<p style="color:{c["text_secondary"]};font-size:0.85rem;">Auth mode: <b>{mode}</b> · '
                f'Logged in as: <b>{current_user.get("name", "—")}</b> ({current_role}) · '
                f'Storage: <b>Database (bcrypt)</b></p>',
                unsafe_allow_html=True)

    # --- User list ---
    users = db_list_users()

    if not is_admin:
        st.warning("Only admin users can manage users.")
        if users:
            rows = [{"Username": u["username"], "Name": u["name"], "Role": u["role"]} for u in users]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        return

    st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.85rem;">CURRENT USERS</p>',
                unsafe_allow_html=True)

    if users:
        rows = [{
            "Username": u["username"],
            "Name": u["name"],
            "Role": u["role"],
            "Password": "🔒 bcrypt",
            "Created": (u.get("created_at") or "—")[:19],
            "Updated": (u.get("updated_at") or "—")[:19],
        } for u in users]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No users configured")

    st.divider()

    # --- Add new user ---
    st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.85rem;">ADD USER</p>',
                unsafe_allow_html=True)

    with st.form("add_user_form", clear_on_submit=True):
        ac1, ac2 = st.columns(2)
        with ac1:
            new_username = st.text_input("Username", key="new_username")
            new_password = st.text_input("Password", type="password", key="new_password")
        with ac2:
            new_name = st.text_input("Display Name", key="new_name")
            new_role = st.selectbox("Role", ["viewer", "admin"], key="new_role")

        if st.form_submit_button("➕ Add User"):
            if not new_username or not new_password:
                st.error("Username and password are required")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                ok = db_create_user(new_username, new_password, new_name or new_username, new_role)
                if ok:
                    st.success(f"User '{new_username}' created (password bcrypt-hashed)")
                    st.rerun()
                else:
                    st.error(f"User '{new_username}' already exists")

    st.divider()

    # --- Edit / Delete existing users ---
    st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.85rem;">EDIT / DELETE USER</p>',
                unsafe_allow_html=True)

    usernames = [u["username"] for u in users]
    if usernames:
        sel_user = st.selectbox("Select user", usernames, key="edit_user_select")
        sel_data = next((u for u in users if u["username"] == sel_user), {})

        with st.form("edit_user_form"):
            ec1, ec2 = st.columns(2)
            with ec1:
                edit_name = st.text_input("Display Name", value=sel_data.get("name", ""), key="edit_name")
                edit_role = st.selectbox("Role", ["viewer", "admin"],
                                         index=0 if sel_data.get("role", "viewer") == "viewer" else 1,
                                         key="edit_role")
            with ec2:
                edit_password = st.text_input("New Password (leave blank to keep current)",
                                              type="password", key="edit_password")
                if edit_password:
                    st.caption("Password will be bcrypt-hashed before storage")

            fc1, fc2 = st.columns(2)
            with fc1:
                save_clicked = st.form_submit_button("💾 Save Changes")
            with fc2:
                delete_clicked = st.form_submit_button("🗑️ Delete User")

            if save_clicked:
                if edit_password and len(edit_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    db_update_user(sel_user, name=edit_name, role=edit_role,
                                   password=edit_password if edit_password else None)
                    st.success(f"User '{sel_user}' updated")
                    st.rerun()

            if delete_clicked:
                if sel_user == current_user.get("username"):
                    st.error("Cannot delete your own account")
                elif db_user_count() <= 1:
                    st.error("Cannot delete the last user")
                else:
                    db_delete_user(sel_user)
                    st.success(f"User '{sel_user}' deleted")
                    st.rerun()

    # --- Google OAuth settings (if mode is google) ---
    if mode == "google":
        st.divider()
        st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.85rem;">GOOGLE OAUTH SETTINGS</p>',
                    unsafe_allow_html=True)
        st.caption("These are read from config.yaml or environment variables.")
        gc1, gc2 = st.columns(2)
        with gc1:
            client_id = auth.get("google_client_id", "") or os.environ.get("GOOGLE_CLIENT_ID", "")
            st.text_input("Client ID", value=client_id[:20] + "..." if len(client_id) > 20 else client_id,
                          disabled=True)
            domains = auth.get("allowed_domains", [])
            st.text_input("Allowed Domains", value=", ".join(domains) if domains else "Any",
                          disabled=True)
        with gc2:
            emails = auth.get("allowed_emails", [])
            st.text_input("Allowed Emails", value=", ".join(emails) if emails else "Any",
                          disabled=True)


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
                    config = _load_config()
                    fetcher = FallbackFetcher(config=config)
                    articles = fetcher.get_news()
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
                    config = _load_config()
                    fetcher = FallbackFetcher(config=config)
                    macro = fetcher.get_macro_fred()
                    today = datetime.now().strftime("%Y-%m-%d")
                    # Enhancement 26: Write to DuckDB
                    try:
                        config = _load_config()
                        router = get_router(config)
                        router.write_analytics(
                            "INSERT OR REPLACE INTO macro (date, vix, vix_change, us10y_yield, dxy, fed_funds, gold, crude) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            (today, macro.get("vix"), macro.get("vix_change"),
                             macro.get("us10y_yield"), macro.get("dxy"),
                             macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
                        )
                    except Exception:
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
                    fv = build_feature_vector(conn, config=config)
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
                    fv = build_feature_vector(conn, config=config)
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

    st.divider()

    # ES Strategy AI Training Controls
    st.markdown("**ES Strategy — AI Models**")
    st.caption("Train the entry gate and exit controller for AI-assisted ES futures trading.")

    # Model status indicators
    entry_path = os.path.join("./models", "es_entry_gate.json")
    exit_path = os.path.join("./models", "es_exit_cnn.pt")
    entry_exists = os.path.exists(entry_path)
    exit_exists = os.path.exists(exit_path)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        if entry_exists:
            size_kb = os.path.getsize(entry_path) / 1024
            mod_time = datetime.fromtimestamp(os.path.getmtime(entry_path)).strftime("%Y-%m-%d %H:%M")
            st.success(f"✅ Entry Gate — {size_kb:.0f} KB")
            st.caption(f"Last trained: {mod_time}")
        else:
            st.error("❌ Entry Gate — not trained")
    with mc2:
        if exit_exists:
            size_kb = os.path.getsize(exit_path) / 1024
            mod_time = datetime.fromtimestamp(os.path.getmtime(exit_path)).strftime("%Y-%m-%d %H:%M")
            st.success(f"✅ Exit Controller — {size_kb:.0f} KB")
            st.caption(f"Last trained: {mod_time}")
        else:
            st.error("❌ Exit Controller — not trained")
    with mc3:
        config = _load_config()
        ai_enabled = config.get("es_strategy", {}).get("ai_enabled", False)
        new_ai = st.toggle("AI Enabled", value=ai_enabled, key="es_ai_toggle",
                            help="Toggle es_strategy.ai_enabled in config.yaml")
        if new_ai != ai_enabled:
            try:
                with open("config.yaml") as f:
                    raw = f.read()
                if ai_enabled:
                    raw = raw.replace("ai_enabled: true", "ai_enabled: false")
                else:
                    raw = raw.replace("ai_enabled: false", "ai_enabled: true")
                with open("config.yaml", "w") as f:
                    f.write(raw)
                st.success(f"AI {'enabled' if new_ai else 'disabled'} — restart ES strategy to apply")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Config update failed: {e}")

    tc1, tc2 = st.columns(2)
    with tc1:
        if st.button("🎯 Train Entry Gate", key="act_train_entry",
                      help="Train XGBoost entry gate using triple-barrier labels on intraday data"):
            with st.spinner("Training ES Entry Gate..."):
                try:
                    from src.es_strategy.ai_models import ESEntryGate
                    from src.es_strategy.labeling import generate_training_dataset
                    config = _load_config()

                    # Load intraday bars for training
                    conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
                    bars_df = pd.read_sql_query(
                        "SELECT timestamp, open, high, low, close, volume, vwap "
                        "FROM intraday_bars WHERE ticker='SPY' ORDER BY timestamp",
                        conn,
                    )
                    conn.close()

                    if bars_df.empty or len(bars_df) < 100:
                        st.error(f"Insufficient intraday data ({len(bars_df)} bars). Need 100+.")
                    else:
                        # Compute indicators on bars
                        from src.data.features import compute_atr
                        bars_df["atr_14"] = compute_atr(bars_df["high"], bars_df["low"],
                                                         bars_df["close"], 14)
                        bars_df = bars_df.dropna(subset=["atr_14"])

                        credit_C = config.get("es_strategy", {}).get("credit_C", 10.0)
                        labels = generate_training_dataset(bars_df, credit_C=credit_C)

                        # Build feature matrix from OHLCV + indicators
                        feat_cols = ["open", "high", "low", "close", "volume"]
                        if "vwap" in bars_df.columns:
                            feat_cols.append("vwap")
                        feat_cols.append("atr_14")
                        X = bars_df[feat_cols].values.astype(np.float64)
                        X = np.nan_to_num(X, nan=0.0)
                        y = labels["entry_labels"].values

                        gate = ESEntryGate(config)
                        result = gate.train(X, y)
                        if "error" in result:
                            st.error(f"Training failed: {result['error']}")
                        else:
                            st.success(f"Entry Gate trained — accuracy: {result['accuracy']:.1%}")
                            st.json(result)
                except Exception as e:
                    st.error(f"Entry Gate training failed: {e}")

    with tc2:
        if st.button("🛡️ Train Exit Controller", key="act_train_exit",
                      help="Train CNN exit controller using reversal labels on intraday data"):
            with st.spinner("Training ES Exit Controller..."):
                try:
                    from src.es_strategy.ai_models import ESExitController
                    from src.es_strategy.labeling import generate_training_dataset
                    config = _load_config()

                    conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
                    bars_df = pd.read_sql_query(
                        "SELECT timestamp, open, high, low, close, volume, vwap "
                        "FROM intraday_bars WHERE ticker='SPY' ORDER BY timestamp",
                        conn,
                    )
                    conn.close()

                    if bars_df.empty or len(bars_df) < 100:
                        st.error(f"Insufficient intraday data ({len(bars_df)} bars). Need 100+.")
                    else:
                        from src.data.features import compute_atr
                        bars_df["atr_14"] = compute_atr(bars_df["high"], bars_df["low"],
                                                         bars_df["close"], 14)
                        bars_df = bars_df.dropna(subset=["atr_14"])

                        credit_C = config.get("es_strategy", {}).get("credit_C", 10.0)
                        labels = generate_training_dataset(bars_df, credit_C=credit_C)

                        feat_cols = ["open", "high", "low", "close", "volume"]
                        if "vwap" in bars_df.columns:
                            feat_cols.append("vwap")
                        feat_cols.append("atr_14")

                        # Build windowed sequences for CNN
                        lookback = 20
                        n_features = len(feat_cols)
                        data = bars_df[feat_cols].values.astype(np.float64)
                        data = np.nan_to_num(data, nan=0.0)
                        y_exit = labels["exit_labels"].values

                        X_windows = []
                        y_windows = []
                        for i in range(lookback, len(data)):
                            X_windows.append(data[i - lookback:i])
                            y_windows.append(y_exit[i])
                        X_windows = np.array(X_windows)
                        y_windows = np.array(y_windows)

                        if len(X_windows) < 50:
                            st.error("Not enough windowed samples for training")
                        else:
                            import torch
                            import torch.nn as nn

                            controller = ESExitController(n_features=n_features, lookback=lookback)
                            controller.build_model()

                            # Simple training loop
                            X_t = torch.FloatTensor(X_windows).unsqueeze(1)
                            y_t = torch.FloatTensor(y_windows)
                            split = int(len(X_t) * 0.8)

                            optimizer = torch.optim.Adam(controller.model.parameters(), lr=0.001)
                            loss_fn = nn.BCEWithLogitsLoss()

                            controller.model.train()
                            for epoch in range(30):
                                optimizer.zero_grad()
                                out = controller.model(X_t[:split]).squeeze()
                                loss = loss_fn(out, y_t[:split])
                                loss.backward()
                                optimizer.step()

                            # Validation
                            controller.model.eval()
                            with torch.no_grad():
                                val_out = torch.sigmoid(controller.model(X_t[split:]).squeeze())
                                val_pred = (val_out > 0.5).float()
                                val_acc = (val_pred == y_t[split:]).float().mean().item()

                            controller.save()
                            st.success(f"Exit Controller trained — val accuracy: {val_acc:.1%}")
                            st.json({"val_accuracy": round(val_acc, 3), "samples": len(X_windows),
                                     "model_path": "./models/es_exit_cnn.pt"})
                except Exception as e:
                    st.error(f"Exit Controller training failed: {e}")


# --- Database Tab ---

def _admin_db_tab():
    st.subheader("Database Explorer")

    db_path = os.path.join(DATA_DIR, "spy.db")
    if not os.path.exists(db_path):
        st.error("Database not found")
        return

    conn = sqlite3.connect(db_path)

    # Enhancement 26: Get tables from both DBs
    duck_router = None
    try:
        config = _load_config()
        duck_router = get_router(config)
    except Exception:
        pass

    sqlite_tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]

    # Combine: analytics tables from DuckDB, rest from SQLite
    all_tables = sorted(set(sqlite_tables) | ANALYTICS_TABLES)

    selected = st.selectbox("Table", all_tables, key="db_table")
    is_duck_table = duck_router and selected in ANALYTICS_TABLES

    if is_duck_table:
        st.caption(f"🦆 Reading from DuckDB analytics")
    else:
        st.caption(f"📦 Reading from SQLite")

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
        if is_duck_table:
            df = duck_router.read_analytics(
                f"SELECT * FROM {selected} ORDER BY {date_col} {order} LIMIT {limit}"
            )
            count_df = duck_router.read_analytics(f"SELECT COUNT(*) as cnt FROM {selected}")
            total = int(count_df.iloc[0]["cnt"]) if not count_df.empty else 0
        else:
            df = pd.read_sql_query(
                f"SELECT * FROM {selected} ORDER BY {date_col} {order} LIMIT {limit}",
                conn,
            )
            total = conn.execute(f"SELECT COUNT(*) FROM {selected}").fetchone()[0]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df)} of {total} rows")
    except Exception as e:
        st.error(f"Query error: {e}")

    st.divider()

    # Custom SQL query
    st.subheader("Custom Query")
    db_target = st.radio("Target", ["Auto-detect", "DuckDB", "SQLite"], horizontal=True, key="db_target")
    query = st.text_area("SQL (read-only, SELECT only)", value=f"SELECT * FROM {selected} LIMIT 10", key="db_query", height=80)
    if st.button("Run Query", key="run_query"):
        if not query.strip().upper().startswith("SELECT"):
            st.error("Only SELECT queries are allowed")
        else:
            try:
                use_duck = False
                if db_target == "DuckDB" and duck_router:
                    use_duck = True
                elif db_target == "Auto-detect" and duck_router:
                    # Check if query references analytics tables
                    q_upper = query.upper()
                    use_duck = any(t.upper() in q_upper for t in ANALYTICS_TABLES)

                if use_duck:
                    df = duck_router.read_analytics(query)
                    st.caption("🦆 Executed on DuckDB")
                else:
                    df = pd.read_sql_query(query, conn)
                    st.caption("📦 Executed on SQLite")
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

def _mask_sensitive_yaml(raw: str) -> str:
    """Mask API keys and secrets in YAML text for display."""
    import re
    # Match lines like:  api_key: "value"  or  session_secret: "value"
    sensitive_keys = r'(api_key|api_secret|client_id|client_secret|session_secret|token|password)'
    def _mask_line(m):
        prefix = m.group(1)
        value = m.group(2).strip().strip('"').strip("'")
        if not value or value.startswith("YOUR_") or value == "":
            return m.group(0)  # don't mask placeholders
        masked = value[:4] + "****" + value[-2:] if len(value) > 8 else "********"
        return f'{prefix}"{masked}"'
    return re.sub(
        rf'({sensitive_keys}\s*:\s*)"?([^"\n]+)"?',
        _mask_line, raw, flags=re.IGNORECASE
    )


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

    # Editable config — mask sensitive values in display
    st.markdown("**Edit Configuration**")
    st.caption("API keys and secrets are masked. Edit the YAML below and click Save. "
               "Changes take effect on next component restart.")

    # Toggle to reveal/mask sensitive values
    show_secrets = st.checkbox("🔓 Reveal sensitive values", value=False, key="reveal_secrets")
    display_raw = raw if show_secrets else _mask_sensitive_yaml(raw)

    edited = st.text_area("config.yaml", value=display_raw, height=400, key="config_editor")

    if st.button("💾 Save Configuration", key="save_config"):
        # Prevent saving masked values
        if not show_secrets and "****" in edited:
            st.error("Cannot save masked values. Enable 'Reveal sensitive values' first, then edit and save.")
        else:
            try:
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

    log_files = {
        "Dashboard": "./logs/dashboard.log",
        "Scheduler": "./logs/scheduler.log",
        "ES Strategy": "./logs/es_strategy.log",
        "Confidence API": "./logs/confidence_api.log",
        "Metrics Exporter": "./logs/metrics_exporter.log",
    }

    log_source = st.selectbox("Log Source", [
        *[f"{name} ({path})" for name, path in log_files.items()],
        "Pipeline (last run)",
        "Model Files",
    ], key="log_source")

    # Check if it's a log file selection
    log_match = next((path for name, path in log_files.items()
                      if log_source.startswith(name)), None)

    if log_match:
        try:
            with open(log_match) as f:
                lines = f.readlines()
            n = st.slider("Lines (most recent)", 20, 200, 50, key="log_lines")
            # Filter option to hide deprecation warnings
            hide_deprecation = st.checkbox("Hide deprecation warnings", value=True, key="hide_depr")
            filtered = lines
            if hide_deprecation:
                filtered = [l for l in lines if "DeprecationWarning" not in l and "FutureWarning" not in l]
            st.code("".join(filtered[-n:]), language="log")
            if hide_deprecation and len(filtered) < len(lines):
                st.caption(f"Filtered out {len(lines) - len(filtered)} deprecation warnings")
        except FileNotFoundError:
            st.info(f"No log found at {log_match}")
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

    # SPY data
    spy_state = load_spy_state()
    pred = spy_state.get("prediction", {})
    indicators = spy_state.get("indicators", {})

    # Get SPY last close from DB (spy_state.json doesn't carry it)
    spy_close = None
    try:
        import sqlite3 as _sql3
        _c = _sql3.connect(os.path.join(DATA_DIR, "spy.db"))
        row = _c.execute(
            "SELECT close FROM prices ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row:
            spy_close = row[0]
        _c.close()
    except Exception:
        pass

    # ES data
    es_state = load_es_state()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SPY", f"${spy_close:.2f}" if spy_close else "—")
    with col2:
        direction = pred.get("direction", "—")
        conf = pred.get("confidence", 0)
        st.metric("Signal", direction, f"{conf:.0f}%")
    with col3:
        vix = indicators.get("vix")
        if vix is None:
            macro = _fetch_live_macro()
            vix = macro.get("vix")
        st.metric("VIX", f"{vix:.1f}" if vix else "—")

    col4, col5, col6 = st.columns(3)
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
    """Embed Grafana dashboards or fall back to native Plotly monitoring."""

    # --- Check if Grafana is reachable from the server ---
    config = _load_config_cached()
    grafana_cfg = config.get("grafana", {})
    _dgx_ip = os.environ.get("DGX_IP", "192.168.1.211")
    grafana_lan = grafana_cfg.get("lan_url", "") or os.environ.get("GRAFANA_HOST", f"http://{_dgx_ip}:3001")
    proxy_host = os.environ.get("GRAFANA_PROXY_HOST", f"http://{_dgx_ip}:9190")

    # Server-side check: can we reach Grafana from the DGX itself?
    grafana_ok = False
    try:
        _r = requests.get(f"{grafana_lan}/api/health", timeout=3)
        grafana_ok = _r.status_code == 200
    except Exception:
        grafana_ok = False

    # --- Quick-glance summary cards ---
    _grafana_summary_cards()
    st.divider()

    # --- Mode toggle: Grafana iframe vs native Plotly ---
    if grafana_ok:
        view_mode = st.radio(
            "View Mode",
            ["📊 Native Charts", "🔗 Grafana Embed"],
            horizontal=True,
            label_visibility="collapsed",
            help="Native Charts work everywhere. Grafana Embed requires LAN access to port 3001.",
        )
    else:
        view_mode = "📊 Native Charts"
        st.info("ℹ️ Grafana is not reachable — showing native Plotly dashboards instead.")

    if view_mode == "📊 Native Charts":
        # Render the same monitoring dashboards (all 5 tabs) using Plotly
        page_monitoring()
        return

    # --- Grafana iframe mode (LAN only) ---
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

    # Build embed URL — anonymous access is enabled in Grafana for kiosk mode.
    user_info = get_user()
    use_proxy = (
        user_info
        and "@local" not in user_info.get("email", "")
    )
    token = get_session_token()
    if token and use_proxy:
        embed_url = (
            f"{proxy_host}/grafana-proxy/d/{uid}"
            f"?orgId=1&kiosk&auth_token={urllib.parse.quote(token)}"
        )
    else:
        embed_url = f"{grafana_lan}/d/{uid}?orgId=1&kiosk"

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Grafana Settings**")
    height = st.sidebar.slider("Panel Height", 400, 1200, 800, 50)
    st.sidebar.caption(f"[Open full Grafana ↗]({grafana_lan}/d/{uid})")

    user_info = get_user()
    if user_info and user_info.get("email") != "anonymous":
        st.sidebar.caption(f"Grafana user: {user_info['email']}")

    # Embed Grafana iframe
    st.components.v1.html(
        f"""
        <style>
            iframe#grafana-embed {{
                width: 100%; height: {height}px; border: none;
                border-radius: 8px; background: #181b1f;
            }}
        </style>
        <iframe id="grafana-embed" src="{embed_url}"></iframe>
        """,
        height=height + 10,
    )


# ======================================================================
# ROUTER — st.navigation handles page dispatch
# ======================================================================

_pages = {
    "Markets": [
        st.Page(page_spy, title="SPY Predictor", icon=":material/query_stats:", default=True),
        st.Page(page_es, title="ES Strategy", icon=":material/candlestick_chart:"),
        st.Page(page_whatif, title="What-If Analysis", icon=":material/science:"),
        st.Page(page_forecast, title="Forecast", icon=":material/trending_up:"),
        st.Page(page_single_stock, title="Single-Stock", icon=":material/search:"),
    ],
    "Operations": [
        st.Page(page_monitoring, title="Monitoring", icon=":material/monitor_heart:"),
        st.Page(page_grafana, title="Grafana Dashboards", icon=":material/dashboard:"),
        st.Page(page_admin, title="Admin", icon=":material/settings:"),
    ],
}

_pg = st.navigation(_pages)

st.sidebar.divider()
mode_label = "☁️ Cloud" if IS_CLOUD else "🖥️ Local"
st.sidebar.caption(f"{mode_label} mode")
if user and user.get("email") != "anonymous":
    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        logout()
        st.rerun()

_pg.run()
