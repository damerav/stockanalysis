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
from src.dashboard.single_stock_app import page_single_stock
from src.dashboard.performance_app import page_performance
from src.dashboard.tuning_app import page_tuning
from src.dashboard.rules_app import page_rules
from src.dashboard.strangle_app import page_strangle
from src.dashboard.market_overview_app import page_market_overview
from src.dashboard.scenario_analysis_app import page_scenario_analysis
from src.dashboard.data_management_app import page_data_management
from src.dashboard.system_management_app import page_system_management
from src.data.db_router import get_router, ANALYTICS_TABLES
from src.data.fetcher import FallbackFetcher
from src.dashboard.theme import (
    get_theme, get_colors, get_plotly_layout, get_title_font,
    metric_card as _theme_metric_card, badge_html as _theme_badge,
    page_header, render_theme_toggle, is_dark,
    _sync_config_toml,
)
from src.dashboard.template import (
    kpi_card as _tpl_kpi_card,
    signal_banner as _tpl_signal_banner,
    stale_banner as _tpl_stale_banner,
    badge_html as _tpl_badge,
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

# --- Guard: Only run UI setup when executed as main Streamlit script ---
# When imported as a module (e.g., from scenario_analysis_app), skip all UI setup.
# Streamlit sets __name__ to "__main__" for the entry script.
_IS_MAIN_SCRIPT = (__name__ == "__main__")

if _IS_MAIN_SCRIPT:
    st.set_page_config(page_title="Stock Analysis", layout="wide", page_icon="📊")

    # --- Sync config.toml to match session theme ---
    _sync_config_toml(get_theme())

    # --- Load unified design system CSS (cached in session to avoid disk reads) ---
    if "_css_cache" not in st.session_state:
        _css_path = os.path.join(os.path.dirname(__file__), "style.css")
        if os.path.exists(_css_path):
            with open(_css_path) as _css_f:
                st.session_state["_css_cache"] = _css_f.read()
        else:
            st.session_state["_css_cache"] = ""
    if st.session_state["_css_cache"]:
        st.markdown(f"<style>{st.session_state['_css_cache']}</style>", unsafe_allow_html=True)

    # --- Inject light theme overrides if in light mode ---
    if get_theme() == "light":
        st.markdown("""<style>
        :root {
            --color-bg-primary: #FFFFFF;
            --color-bg-secondary: #F8F9FA;
            --color-bg-tertiary: #E9ECEF;
            --color-border-primary: #DEE2E6;
            --color-border-secondary: #D1D4DC;
            --color-text-primary: #212529;
            --color-text-secondary: #6C757D;
            --color-text-tertiary: #ADB5BD;
            --color-grid: #E6E8EC;
            --color-zeroline: #B7BDC6;
            --color-card-bg: #FFFFFF;
            --color-card-hover: rgba(0,123,255,0.15);
            --color-tab-bg: #E6E8EC;
            --color-input-bg: #FFFFFF;
            --color-input-border: #DEE2E6;
            --color-form-bg: #FFFFFF;
            --color-btn-bg: #FFFFFF;
            --color-btn-border: #B7BDC6;
            --color-btn-text: #212529;
            --color-btn-hover-bg: #F0F2F5;
            --color-btn-hover-border: #6C757D;
            --color-expander-bg: #FFFFFF;
            --color-scrollbar: #D1D4DC;
            --color-scrollbar-hover: #B7BDC6;
            --color-popover-bg: #FFFFFF;
            --color-popover-hover: #F0F2F5;
            --sidebar-bg: linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%);
            --sidebar-border: #DEE2E6;
            --sidebar-text: #212529;
            --sidebar-text-muted: #6C757D;
            --sidebar-btn-bg: #F0F2F5;
            --sidebar-btn-border: #D1D4DC;
            --sidebar-btn-hover-bg: #E6E8EC;
            --sidebar-btn-hover-border: #B7BDC6;
            --sidebar-nav-hover: rgba(0, 0, 0, 0.04);
            --sidebar-nav-active-bg: rgba(0, 123, 255, 0.08);
            --sidebar-divider: #DEE2E6;
            --backdrop-filter: none;
            --card-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }
        </style>""", unsafe_allow_html=True)

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

    # --- Sidebar live ticker symbol selector ---
    _TICKER_PRESETS = ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
                       "QQQ", "IWM", "DIA", "VIX"]
    if "live_ticker_symbol" not in st.session_state:
        st.session_state["live_ticker_symbol"] = "SPY"
    if "_ticker_pending" in st.session_state:
        st.session_state["live_ticker_symbol"] = st.session_state.pop("_ticker_pending")
    _current_sym = st.session_state["live_ticker_symbol"]
    _sidebar_options = _TICKER_PRESETS if _current_sym in _TICKER_PRESETS else [_current_sym] + _TICKER_PRESETS
    st.sidebar.selectbox(
        "📈 Live Ticker",
        _sidebar_options,
        key="live_ticker_symbol",
    )

# Module-level user fallback for import path
if not _IS_MAIN_SCRIPT:
    user = None

# NOTE: st.navigation is called after all page functions are defined (see bottom of file)


# ======================================================================
# SPY PREDICTOR PAGE
# ======================================================================

@st.cache_data(ttl=30, show_spinner=False)
def _load_spy_state_cached() -> dict:
    """Cached spy_state.json read (30s TTL)."""
    try:
        with open(os.path.join(DATA_DIR, "spy_state.json"), "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


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
        return _load_spy_state_cached()


@st.cache_data(ttl=120, show_spinner=False)
def load_prediction_history(n: int = 30) -> pd.DataFrame:
    if IS_CLOUD:
        return pd.DataFrame()
    try:
        config = _load_config_cached()
        router = get_router(config)
        df = router.read_analytics(
            f"SELECT date, direction, confidence FROM predictions ORDER BY date DESC LIMIT {n}"
        )
        return df.iloc[::-1] if not df.empty else df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_performance() -> pd.DataFrame:
    if IS_CLOUD:
        return pd.DataFrame()
    try:
        config = _load_config_cached()
        router = get_router(config)
        df = router.read_analytics(
            "SELECT date, predicted, actual, correct, cumulative_accuracy, "
            "confidence_tier, vix_regime, day_of_week, event_proximity "
            "FROM performance ORDER BY date DESC LIMIT 30"
        )
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
        "STRONG_BULLISH": c["green"], "BULLISH": c["green"], "WEAK_BULLISH": c["green"],
        "NEUTRAL": c["yellow"],
        "WEAK_BEARISH": c["red"], "BEARISH": c["red"], "STRONG_BEARISH": c["red"],
    }
    banner_color = color_map.get(scale_label, c["yellow"])

    # --- Hero prediction card ---
    if prediction:
        conf_interp = "Weak" if confidence < 55 else "Moderate" if confidence < 70 else "Strong" if confidence < 85 else "Very Strong"
        arrow = "▲" if "BULLISH" in scale_label else "▼" if "BEARISH" in scale_label else "◆"
        up_pct = probs.get('up', 0)
        neutral_pct = probs.get('neutral', 0)
        down_pct = probs.get('down', 0)
        total = max(up_pct + neutral_pct + down_pct, 1)
        up_w = up_pct / total * 100
        neut_w = neutral_pct / total * 100
        down_w = down_pct / total * 100
        updated_str = updated_at[:16].replace('T', ' ') if updated_at else "—"
        # Solid banner with glow
        glow = f"box-shadow: 0 6px 32px {banner_color}55, 0 2px 8px rgba(0,0,0,0.3);"
        st.markdown(
            f"""<div style="background: linear-gradient(135deg, {banner_color}ee 0%, {banner_color} 100%);
            border-radius:16px; padding:28px 32px; margin-bottom:16px; {glow}">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;">
                <div style="display:flex; align-items:center; gap:18px;">
                    <span style="font-size:3.2rem; color:#fff; line-height:1; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.3));">{arrow}</span>
                    <div>
                        <div style="font-size:2.2rem; font-weight:800; color:#fff; letter-spacing:1px; text-shadow:0 2px 4px rgba(0,0,0,0.2);">
                            {scale_label.replace('_', ' ')}
                        </div>
                        <div style="font-size:0.9rem; color:rgba(255,255,255,0.8); margin-top:4px;">
                            Next-day SPY direction · {conf_interp} signal
                        </div>
                    </div>
                </div>
                <div style="text-align:center; background:rgba(0,0,0,0.15); border-radius:12px; padding:12px 24px; min-width:120px;">
                    <div style="font-size:3rem; font-weight:900; color:#fff; line-height:1; text-shadow:0 2px 4px rgba(0,0,0,0.2);">{confidence:.0f}%</div>
                    <div style="font-size:0.75rem; color:rgba(255,255,255,0.7); text-transform:uppercase; letter-spacing:0.1em; margin-top:2px;">Confidence</div>
                </div>
            </div>
            <div style="margin-top:18px;">
                <div style="display:flex; border-radius:8px; overflow:hidden; height:14px; background:rgba(0,0,0,0.2);">
                    <div style="width:{up_w:.1f}%; background:{c['green']}; transition:width 0.5s ease;" title="Up {up_pct:.0f}%"></div>
                    <div style="width:{neut_w:.1f}%; background:{c['yellow']}; transition:width 0.5s ease;" title="Neutral {neutral_pct:.0f}%"></div>
                    <div style="width:{down_w:.1f}%; background:{c['red']}; transition:width 0.5s ease;" title="Down {down_pct:.0f}%"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:6px;">
                    <span style="color:rgba(255,255,255,0.9); font-size:0.82rem; font-weight:600;">↑ Up {up_pct:.1f}%</span>
                    <span style="color:rgba(255,255,255,0.9); font-size:0.82rem; font-weight:600;">— Neutral {neutral_pct:.1f}%</span>
                    <span style="color:rgba(255,255,255,0.9); font-size:0.82rem; font-weight:600;">↓ Down {down_pct:.1f}%</span>
                </div>
            </div>
            <div style="text-align:right; margin-top:8px; font-size:0.72rem; color:rgba(255,255,255,0.5);">
                Updated {updated_str}
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("Waiting for prediction data...")

    # --- P3: Earnings + Fed + Options (compact row) ---
    try:
        _p3_router = get_router(_load_config())
        _today = datetime.now().strftime("%Y-%m-%d")

        p3_col1, p3_col2, p3_col3 = st.columns(3)
        with p3_col1:
            from src.data.earnings_calendar import get_earnings_features as _get_earn
            earn = _get_earn(_p3_router, _today)
            density = earn.get("earnings_density", 0)
            days_next = earn.get("days_to_next_mega", 30)
            earn_week = earn.get("earnings_week", 0)
            st.metric("📅 Earnings", f"{density} mega-caps",
                      delta="Earnings Week" if earn_week else None,
                      delta_color="normal" if earn_week else "off")
            st.caption("Earnings today" if days_next == 0 else f"Next in {days_next}d")

        with p3_col2:
            from src.data.fed_comms import get_fed_features as _get_fed
            fed = _get_fed(_p3_router, _today)
            avg = fed.get("fed_sentiment_avg", 0)
            label = "🦅 Hawkish" if avg > 0.2 else "🕊️ Dovish" if avg < -0.2 else "⚖️ Neutral"
            st.metric("Fed", label, delta=f"{avg:+.2f}")

        with p3_col3:
            opt_df = _p3_router.query(
                "SELECT vanna_exposure, charm_exposure, zero_dte_pcr "
                "FROM options_analytics WHERE date = ? ORDER BY date DESC LIMIT 1",
                (_today,),
            )
            if not opt_df.empty and opt_df.iloc[0]["vanna_exposure"] is not None:
                st.metric("Vanna", f"{opt_df.iloc[0]['vanna_exposure']:,.0f}")
                charm = opt_df.iloc[0]["charm_exposure"]
                dte = opt_df.iloc[0]["zero_dte_pcr"]
                st.caption(f"Charm: {charm:,.0f} | 0DTE P/C: {dte:.2f}" if charm else "")
            else:
                st.metric("Greeks", "—")
    except Exception:
        pass

    # --- Microstructure (collapsible) ---
    try:
        _micro_router = get_router(_load_config())
        _today2 = datetime.now().strftime("%Y-%m-%d")
        from src.data.features import compute_intraday_microstructure as _get_micro
        micro = _get_micro(_micro_router, _today2)

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
            # Use macro data for VIX to stay consistent with the overview page
            _live_macro = _fetch_live_macro()
            _vix_val = _live_macro.get("vix") if _live_macro else None
            _vix_chg = _live_macro.get("vix_change") if _live_macro else None
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
                _stale_color = c["red"]
                _stale_label = " ⚠️ STALE"
            elif _age_min > 30:
                _stale_color = c["yellow"]
                _stale_label = " ⏳"
        except Exception:
            pass
    st.markdown(f'<p style="color:{_stale_color}; font-size:0.75rem; text-align:right; margin-top:4px;">'
                f'Updated: {updated_at or "N/A"}{_stale_label}</p>', unsafe_allow_html=True)


# ======================================================================
# ES STRATEGY PAGE
# ======================================================================

@st.cache_data(ttl=10, show_spinner=False)
def _load_es_state_cached() -> dict:
    """Cached es_state.json read (10s TTL)."""
    try:
        with open(os.path.join(DATA_DIR, "es_state.json"), "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


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
        return _load_es_state_cached()


@st.cache_data(ttl=60)
def _es_chart_from_db():
    """Load latest intraday bars from DB, resample to 1-min candles with RSI."""
    try:
        from src.data.db_router import get_router
        router = get_router()
        df = router.query(
            "SELECT timestamp, open, high, low, close, volume, vwap "
            "FROM intraday_bars WHERE ticker='SPY' "
            "ORDER BY timestamp DESC LIMIT 5000"
        )
        router.close()
        if df is None or df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        # Resample 5s bars to 1-min candles
        df = df.set_index("timestamp")
        ohlcv = df.resample("1min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum", "vwap": "mean"
        }).dropna(subset=["close"])
        ohlcv = ohlcv.reset_index()
        # Compute RSI(14)
        if len(ohlcv) >= 15:
            delta = ohlcv["close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, 1e-10)
            ohlcv["rsi"] = 100 - (100 / (1 + rs))
        return ohlcv
    except Exception:
        return None


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
    regime_text = "#212529" if regime in ("Low", "Med") else "#FFFFFF"

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

    # --- AI Confidence Overlay ---
    ai_enabled = state.get("ai_enabled", False)
    trail_ai = state.get("trail_ai_enabled", False)
    ai_mults = state.get("ai_trail_mults") or {}
    ai_c1, ai_c2, ai_c3, ai_c4 = st.columns(4)
    with ai_c1:
        _ai_icon = "🟢" if ai_enabled else "🔴"
        st.metric("AI Layer", f"{_ai_icon} {'On' if ai_enabled else 'Off'}")
    with ai_c2:
        _p_cont = ai_mults.get("p_cont_5", 0) if ai_mults else 0
        st.metric("Continuation P", f"{_p_cont:.0%}" if _p_cont else "—",
                  help="CNN-predicted probability the trend continues 5 bars")
    with ai_c3:
        _tp2_m = ai_mults.get("tp2_trail") if ai_mults else None
        _run_m = ai_mults.get("runner_trail") if ai_mults else None
        st.metric("AI TP2 Trail", f"{_tp2_m:.2f}×" if _tp2_m else "—",
                  help="CNN-adjusted TP2 trailing multiplier")
    with ai_c4:
        st.metric("AI Runner Trail", f"{_run_m:.2f}×" if _run_m else "—",
                  help="CNN-adjusted runner trailing multiplier")

    # Reload Rules button
    if st.button("🔄 Reload Rules", key="es_reload_rules",
                 help="Write hot-reload flag so the live runner re-reads strategy_rules from DB"):
        import os as _os
        try:
            with open(_os.path.join("data", ".reload_rules"), "w") as _rf:
                _rf.write("1")
            st.success("Reload flag written — runner will pick up new rules on next bar.")
        except Exception as _e:
            st.error(f"Failed to write reload flag: {_e}")

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
        # Fallback: load intraday bars from DB and resample to 1-min candles
        _db_bars = _es_chart_from_db()
        if _db_bars is not None and not _db_bars.empty:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.75, 0.25], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(
                x=_db_bars["timestamp"], open=_db_bars["open"], high=_db_bars["high"],
                low=_db_bars["low"], close=_db_bars["close"], name="SPY",
            ), row=1, col=1)
            if "vwap" in _db_bars.columns:
                fig.add_trace(go.Scatter(
                    x=_db_bars["timestamp"], y=_db_bars["vwap"], mode="lines",
                    line=dict(color="orange", dash="dot", width=1), name="VWAP",
                ), row=1, col=1)
            if "rsi" in _db_bars.columns:
                fig.add_trace(go.Scatter(
                    x=_db_bars["timestamp"], y=_db_bars["rsi"], mode="lines",
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
            st.plotly_chart(fig, use_container_width=True, key="es_db_chart")
            st.caption("📡 Showing latest intraday data from database (1-min candles)")
        else:
            st.warning("⚠️ No chart data available. The ES runner is not active and no "
                       "intraday bars found in the database. Run the daily pipeline or "
                       "check Polygon.io connectivity in Admin → Data Sources.")

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
        ["Rules What-If", "K/C Sweep", "Lot Sizing", "Risk Limits", "Custom Compare"],
        key="es_scenario",
    )

    if scenario == "Rules What-If":
        _whatif_rules(engine)

    elif scenario == "K/C Sweep":
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


def _whatif_rules(engine: WhatIfEngine):
    """Rules-aware What-If: load rules, let user tweak, backtest."""
    from src.strategy import rules_store as rs

    st.subheader("Strategy Rules What-If")
    st.caption("Tweak any rule below and backtest against current live settings. "
               "Nothing is saved — this is simulation only.")

    all_rules = rs.get_all_rules()
    if not all_rules:
        st.warning("No rules in database.")
        return

    groups = sorted(all_rules.keys())
    selected_groups = st.multiselect(
        "Rule groups to edit", groups,
        default=[g for g in ["spread", "tp_high", "risk"] if g in groups],
        key="wi_rule_groups",
    )
    if not selected_groups:
        st.info("Select at least one rule group above.")
        return

    proposed = {}
    for group in selected_groups:
        rules = all_rules[group]
        st.markdown(f"**{group}**")
        cols = st.columns(min(len(rules), 4))
        for i, (key, meta) in enumerate(rules.items()):
            col = cols[i % len(cols)]
            val, vtype = meta["value"], meta["type"]
            wk = f"wi_{group}_{key}"
            desc = meta.get("description", "")
            with col:
                if vtype == "float":
                    nv = st.number_input(key, value=float(val), step=0.01,
                                         format="%.4f", key=wk, help=desc)
                    if abs(nv - float(val)) > 1e-8:
                        proposed[f"{group}.{key}"] = nv
                elif vtype == "int":
                    nv = st.number_input(key, value=int(val), step=1, key=wk, help=desc)
                    if nv != int(val):
                        proposed[f"{group}.{key}"] = nv
                elif vtype == "bool":
                    nv = st.checkbox(key, value=bool(val), key=wk, help=desc)
                    if nv != bool(val):
                        proposed[f"{group}.{key}"] = nv
                else:
                    nv = st.text_input(key, value=str(val), key=wk, help=desc)
                    if nv != str(val):
                        proposed[f"{group}.{key}"] = nv

    st.divider()
    if not proposed:
        st.info("Change any rule value above to enable backtesting.")
        return

    st.markdown(f"**{len(proposed)} proposed change(s):**")
    for k, v in proposed.items():
        g, ky = k.split(".", 1)
        old = all_rules[g][ky]["value"]
        st.markdown(f"- `{k}`: {old} → **{v}**")

    if st.button("🧪 Backtest: Current vs Proposed", key="wi_run_bt", type="primary"):
        with st.spinner("Running two backtests (current rules vs proposed)..."):
            result = engine.es_rules_backtest(proposed)
        _whatif_rules_result(result)


def _whatif_rules_result(result: dict):
    """Render side-by-side backtest comparison for rules what-if."""
    if "error" in result:
        st.error(f"Backtest failed: {result['error']}")
        return

    baseline = result["baseline"]
    proposed = result["proposed"]
    diff = result["diff"]

    verdict = diff["verdict"]
    if verdict == "IMPROVED":
        st.success(f"✅ Proposed rules IMPROVED P&L by ${diff['pnl_delta']:+,.0f} "
                    f"({diff['pnl_pct_change']:+.1f}%)")
    elif verdict == "DEGRADED":
        st.error(f"⚠️ Proposed rules DEGRADED P&L by ${diff['pnl_delta']:+,.0f} "
                  f"({diff['pnl_pct_change']:+.1f}%)")
    else:
        st.info("➖ No P&L difference between current and proposed rules.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current P&L", f"${baseline['total_pnl']:+,.0f}",
                   help=f"{baseline['trades']} trades")
    with col2:
        st.metric("Proposed P&L", f"${proposed['total_pnl']:+,.0f}",
                   delta=f"${diff['pnl_delta']:+,.0f}",
                   help=f"{proposed['trades']} trades")
    with col3:
        st.metric("Trade Count Δ", f"{diff['trade_delta']:+d}",
                   help=f"Current: {baseline['trades']}, Proposed: {proposed['trades']}")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=["P&L", "Trades"],
                         y=[baseline["total_pnl"], baseline["trades"]],
                         marker_color="#2962FF"))
    fig.add_trace(go.Bar(name="Proposed", x=["P&L", "Trades"],
                         y=[proposed["total_pnl"], proposed["trades"]],
                         marker_color="#26A69A"))
    fig.update_layout(barmode="group", title="Current vs Proposed Rules", height=350)
    st.plotly_chart(fig, use_container_width=True)


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
        _prediction_card(orig, key_suffix="orig")
    with col2:
        st.markdown("**Modified Prediction**")
        _prediction_card(mod, key_suffix="mod")
    overrides = result.get("overrides", {})
    if overrides:
        st.markdown("**Features Changed:**")
        st.json(overrides)
    desc = result.get("description", "")
    if desc:
        st.info(desc)


def _prediction_card(pred, key_suffix=""):
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
        st.plotly_chart(fig, use_container_width=True, key=f"pred_card_{key_suffix}")


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


# --- System Status Tab ---

def _admin_status_tab():
    # ── Live Ticker Settings (moved from old page_admin) ──
    _colors = get_colors()
    with st.expander("📈 Live Ticker Settings", expanded=False):
        st.caption("Pick a preset or type any valid ticker symbol.")
        _col_preset, _col_custom = st.columns([1, 1])
        with _col_preset:
            _preset = st.selectbox(
                "Preset symbols",
                ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
                 "QQQ", "IWM", "DIA", "NFLX", "AMD", "COIN", "SOFI", "PLTR"],
                key="admin_ticker_preset",
            )
            if st.button("Apply preset", key="apply_preset_ticker"):
                st.session_state["_ticker_pending"] = _preset
                st.rerun()
        with _col_custom:
            _custom = st.text_input(
                "Custom symbol",
                placeholder="e.g. SOFI, BTC-USD, ^GSPC",
                key="admin_ticker_custom",
            )
            if st.button("Apply custom", key="apply_custom_ticker") and _custom.strip():
                st.session_state["_ticker_pending"] = _custom.strip().upper()
                st.rerun()
        st.caption(f"Currently tracking: **{st.session_state.get('live_ticker_symbol', 'SPY')}**")

    st.subheader("System Health")

    if st.button("\u21BB Refresh Status", key="refresh_status"):
        st.rerun()

    col1, col2, col3 = st.columns(3)

    # Database status
    with col1:
        st.markdown("**Database**")
        try:
            config = _load_config()
            router = get_router(config)
            if router.using_postgres:
                # Get PostgreSQL database size
                try:
                    size_df = router.read_analytics(
                        "SELECT pg_size_pretty(pg_database_size(current_database())) as size"
                    )
                    pg_size = size_df.iloc[0]["size"] if not size_df.empty else "?"
                except Exception:
                    pg_size = "connected"
                st.success(f"🐘 PostgreSQL ({pg_size})")
                try:
                    tbl_df = router.read_analytics(
                        "SELECT COUNT(*) as cnt FROM information_schema.tables "
                        "WHERE table_schema = 'public'"
                    )
                    tbl_count = int(tbl_df.iloc[0]["cnt"]) if not tbl_df.empty else 0
                    st.caption(f"{tbl_count} PostgreSQL tables")
                except Exception:
                    pass
            else:
                if os.path.exists(db_path):
                    size_mb = os.path.getsize(db_path) / (1024 * 1024)
                    st.success(f"Online — SQLite {size_mb:.1f} MB")
                else:
                    st.error("Database not found")
        except Exception:
            if os.path.exists(db_path):
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
                st.warning(f"SQLite only — {size_mb:.1f} MB (PostgreSQL unavailable)")
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
    try:
        config = _load_config()
        router = get_router(config)

        tables = ["prices", "technicals", "news", "daily_sentiment", "macro",
                   "predictions", "intraday_bars", "options_chain",
                   "options_analytics", "intraday_features", "performance",
                   "earnings_calendar", "fed_communications"]
        rows = []
        for t in tables:
            try:
                count_df = router.read_analytics(f"SELECT COUNT(*) as cnt FROM {t}")
                count = int(count_df.iloc[0]["cnt"]) if not count_df.empty else 0
                if t != "intraday_bars":
                    min_df = router.read_analytics(f"SELECT MIN(date) as d FROM {t}")
                    max_df = router.read_analytics(f"SELECT MAX(date) as d FROM {t}")
                    min_date = min_df.iloc[0]["d"] if not min_df.empty else "—"
                    max_date = max_df.iloc[0]["d"] if not max_df.empty else "—"
                    # Convert date objects to strings
                    if hasattr(min_date, "strftime"):
                        min_date = min_date.strftime("%Y-%m-%d")
                    if hasattr(max_date, "strftime"):
                        max_date = max_date.strftime("%Y-%m-%d")
                else:
                    min_date, max_date = "—", "—"
                source = "🐘" if router.using_postgres else "📦"
            except Exception:
                count, min_date, max_date, source = 0, "—", "—", "?"
            rows.append({"Table": t, "Rows": count, "From": min_date or "—",
                         "To": max_date or "—", "DB": source})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("🐘 = PostgreSQL" + (" (primary)" if router.using_postgres else "") + ", 📦 = SQLite")
    except Exception as e:
        st.error(f"Error reading database: {e}")

    # Latest prediction
    st.subheader("Latest Prediction")
    try:
        config = _load_config()
        router = get_router(config)
        pred_df = router.read_analytics(
            "SELECT date, direction, confidence, predicted_at FROM predictions ORDER BY date DESC LIMIT 1"
        )
        if not pred_df.empty:
            pred = pred_df.iloc[0]
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Date", pred["date"])
            pc2.metric("Direction", pred["direction"])
            pc3.metric("Confidence", f"{pred['confidence']:.0f}%")
            pc4.metric("Generated", pred["predicted_at"] or "—")
        else:
            st.info("No predictions yet")
    except Exception:
        st.info("No predictions yet")

    # P2: Model Registry
    st.subheader("Model Registry")
    try:
        config = _load_config()
        router = get_router(config)
        reg_df = router.read_analytics(
            """SELECT model_id, training_date, val_accuracy, test_accuracy,
                      feature_count, gated, deployment_status, created_at
               FROM model_registry ORDER BY created_at DESC LIMIT 10"""
        )
        if not reg_df.empty:
            reg_df.columns = ["ID", "Date", "Val Acc", "Test Acc", "Features", "Gated", "Status", "Created"]
            reg_df["Val Acc"] = reg_df["Val Acc"].apply(lambda x: f"{x:.3f}" if x else "—")
            reg_df["Test Acc"] = reg_df["Test Acc"].apply(lambda x: f"{x:.3f}" if x else "—")
            reg_df["Gated"] = reg_df["Gated"].map({0: "✅", 1: "🚫", False: "✅", True: "🚫"})
            status_map = {"active": "🟢 Active", "retired": "⚪ Retired", "gated": "🚫 Gated",
                          "candidate": "🔵 Candidate"}
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

        if st.form_submit_button("+ Add User"):
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
                save_clicked = st.form_submit_button("Save Changes")
            with fc2:
                delete_clicked = st.form_submit_button("Delete User")

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

        if st.button("Pull Latest Data", key="act_pull", help="Gap detection + backfill prices and macro"):
            with st.spinner("Running daily data pull..."):
                try:
                    from src.data.daily_pull import run_daily_pull
                    config = _load_config()
                    counts = run_daily_pull(config)
                    st.success("Data pull complete")
                    st.json(counts)
                except Exception as e:
                    st.error(f"Data pull failed: {e}")

        if st.button("Fetch News", key="act_news", help="Fetch latest news from Finnhub + RSS"):
            with st.spinner("Fetching news..."):
                try:
                    from src.data.fetcher import FallbackFetcher
                    config = _load_config()
                    fetcher = FallbackFetcher(config=config)
                    articles = fetcher.get_news()
                    router = get_router(config)
                    inserted = 0
                    today = datetime.now().strftime("%Y-%m-%d")
                    for a in articles:
                        try:
                            router.execute(
                                "INSERT INTO news (date, source, headline, summary, url, fetched_at) "
                                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING"
                                if router.using_postgres else
                                "INSERT OR IGNORE INTO news (date, source, headline, summary, url, fetched_at) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (today, a.get("source", ""), a.get("headline", ""),
                                 a.get("summary", ""), a.get("url", ""),
                                 datetime.now().isoformat())
                            )
                            inserted += 1
                        except Exception:
                            pass
                    st.success(f"Fetched {len(articles)} articles, inserted {inserted} new")
                except Exception as e:
                    st.error(f"News fetch failed: {e}")

        if st.button("Fetch Macro Data", key="act_macro", help="Fetch VIX, yields, DXY, gold, crude from FRED"):
            with st.spinner("Fetching macro data..."):
                try:
                    from src.data.fetcher import FallbackFetcher
                    config = _load_config()
                    fetcher = FallbackFetcher(config=config)
                    macro = fetcher.get_macro_fred()
                    today = datetime.now().strftime("%Y-%m-%d")
                    # Write to database (PostgreSQL via router)
                    try:
                        config = _load_config()
                        router = get_router(config)
                        if router.using_postgres:
                            router.execute(
                                "INSERT INTO macro (date, vix, vix_change, us10y_yield, dxy, fed_funds, gold, crude) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                                "ON CONFLICT (date) DO UPDATE SET vix=EXCLUDED.vix, vix_change=EXCLUDED.vix_change, "
                                "us10y_yield=EXCLUDED.us10y_yield, dxy=EXCLUDED.dxy, fed_funds=EXCLUDED.fed_funds, "
                                "gold=EXCLUDED.gold, crude=EXCLUDED.crude",
                                (today, macro.get("vix"), macro.get("vix_change"),
                                 macro.get("us10y_yield"), macro.get("dxy"),
                                 macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
                            )
                        else:
                            router.write_analytics(
                                "INSERT OR REPLACE INTO macro (date, vix, vix_change, us10y_yield, dxy, fed_funds, gold, crude) "
                                "VALUES (?,?,?,?,?,?,?,?)",
                                (today, macro.get("vix"), macro.get("vix_change"),
                                 macro.get("us10y_yield"), macro.get("dxy"),
                                 macro.get("fed_funds"), macro.get("gold"), macro.get("crude"))
                            )
                    except Exception as db_err:
                        st.warning(f"DB write failed: {db_err}")
                    st.success("Macro data updated")
                    st.json(macro)
                except Exception as e:
                    st.error(f"Macro fetch failed: {e}")

        if st.button("Compute Technicals", key="act_tech", help="Recompute SMA, RSI, MACD, BB, ATR"):
            with st.spinner("Computing technicals..."):
                try:
                    from src.data.features import compute_all_technicals, store_technicals
                    config = _load_config()
                    router = get_router(config)
                    df = router.query("SELECT date, open, high, low, close, volume FROM prices ORDER BY date")
                    tech_df = compute_all_technicals(df, config)
                    store_technicals(None, tech_df, config)
                    st.success(f"Technicals computed — {len(tech_df)} rows")
                except Exception as e:
                    st.error(f"Technicals failed: {e}")

    with col2:
        st.markdown("**Model Operations**")

        if st.button("Retrain XGBoost", key="act_train", help="Retrain SPY predictor with latest data (GPU)"):
            with st.spinner("Training XGBoost on GPU... this may take a minute."):
                try:
                    from src.data.features import build_feature_vector, get_feature_columns, get_target
                    from src.model.trainer import SPYPredictor
                    config = _load_config()
                    router = get_router(config)
                    fv = build_feature_vector(router, config=config)
                    target = get_target(fv)
                    predictor = SPYPredictor(config)
                    feature_cols = [c for c in get_feature_columns() if c in fv.columns]
                    X = fv[feature_cols]
                    result = predictor.train(X, target, feature_names=list(feature_cols))
                    st.success(f"Training complete — accuracy: {result.get('accuracy', 0):.1%}")
                    st.json(result)
                except Exception as e:
                    st.error(f"Training failed: {e}")

        if st.button("Run Backtest", key="act_backtest", help="Walk-forward backtest on recent N days"):
            bt_days = st.session_state.get("_bt_days", 60)
            with st.spinner(f"Running {bt_days}-day backtest..."):
                try:
                    from src.data.features import build_feature_vector, get_feature_columns, get_target
                    from src.model.trainer import SPYPredictor
                    config = _load_config()
                    router = get_router(config)
                    fv = build_feature_vector(router, config=config)
                    if fv is None or fv.empty:
                        st.error("No feature data available")
                    else:
                        feature_cols = [c for c in get_feature_columns() if c in fv.columns]
                        target = get_target(fv)
                        bt_days = min(bt_days, len(fv) - 100)
                        test_start = len(fv) - bt_days
                        predictor = SPYPredictor(config)
                        train_fv = fv.iloc[:test_start]
                        train_target = target.iloc[:test_start]
                        result = predictor.train(train_fv[feature_cols], train_target,
                                                 feature_names=list(feature_cols), force_save=False)
                        if result.get("error"):
                            st.error(f"Training failed: {result['error']}")
                        else:
                            test_fv = fv.iloc[test_start:]
                            test_target = target.iloc[test_start:]
                            correct, total = 0, 0
                            for i in range(len(test_fv)):
                                if pd.isna(test_target.iloc[i]):
                                    continue
                                features = test_fv[feature_cols].iloc[i].values
                                pred = predictor.predict(features, feature_names=feature_cols)
                                pred_dir = 1 if "BULLISH" in pred.get("direction", "") else (-1 if "BEARISH" in pred.get("direction", "") else 0)
                                actual = int(test_target.iloc[i])
                                correct += int(pred_dir == actual)
                                total += 1
                            accuracy = correct / max(total, 1)
                            st.success(f"Backtest: {accuracy:.1%} accuracy ({correct}/{total} correct over {bt_days} days)")
                            st.metric("Train Accuracy", f"{result.get('accuracy', 0):.1%}")
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
        bt_days_val = st.number_input("Backtest days", min_value=20, max_value=252, value=60, key="_bt_days")

        if st.button("Generate Prediction", key="act_predict", help="Run inference for next trading day"):
            with st.spinner("Fetching latest news + generating prediction..."):
                try:
                    # Fetch fresh news first so sentiment features are current
                    try:
                        from src.data.news_fetcher import NewsFetcher
                        nf = NewsFetcher(_load_config())
                        news_count = nf.fetch_all()

                        # Quick sentiment update from expanded news corpus
                        if news_count > 0:
                            st.info(f"Fetched {news_count} news articles")
                            try:
                                from src.data.news_features import NewsFeatureProcessor
                                processor = NewsFeatureProcessor(_load_config())
                                article_df = processor.process_articles()
                                today = datetime.now().strftime("%Y-%m-%d")
                                today_arts = article_df[article_df["date"] == today] if not article_df.empty else article_df
                                if not today_arts.empty:
                                    avg_sent = float(today_arts["sentiment_compound"].mean())
                                    art_count = len(today_arts)
                                    pos_ratio = float((today_arts["sentiment_compound"] > 0.05).mean())
                                    neg_ratio = float((today_arts["sentiment_compound"] < -0.05).mean())
                                    _sent_router = get_router(_load_config())
                                    _sent_router.execute(
                                        """INSERT OR REPLACE INTO daily_sentiment
                                           (date, score, confidence, article_count,
                                            positive_ratio, negative_ratio, neutral_ratio)
                                           VALUES (?,?,?,?,?,?,?)""",
                                        (today, avg_sent, min(art_count / 50, 1.0),
                                         art_count, pos_ratio, neg_ratio,
                                         1 - pos_ratio - neg_ratio),
                                    )
                                    st.info(f"Sentiment updated: {art_count} articles, score={avg_sent:.3f}")
                                processor.close()
                            except Exception as se:
                                logger.warning(f"Quick sentiment update failed: {se}")
                        nf.close()
                    except Exception as ne:
                        logger.warning(f"News fetch failed (non-fatal): {ne}")

                    from src.data.features import build_feature_vector, get_feature_columns, get_target
                    from src.model.trainer import SPYPredictor
                    config = _load_config()
                    _pred_router = get_router(config)
                    fv = build_feature_vector(_pred_router, config=config)
                    predictor = SPYPredictor(config)
                    all_feature_cols = [c for c in get_feature_columns() if c in fv.columns]

                    needs_retrain = False
                    if not predictor.load_latest_model():
                        needs_retrain = True
                    else:
                        expected_n = getattr(predictor.model, "n_features_in_", None)
                        if predictor.trained_feature_names:
                            feature_cols = [c for c in predictor.trained_feature_names if c in fv.columns]
                        elif expected_n and expected_n != len(all_feature_cols):
                            needs_retrain = True
                        else:
                            feature_cols = all_feature_cols

                    if needs_retrain:
                        st.info("Model outdated or missing — auto-retraining with current features...")
                        target = get_target(fv)
                        result = predictor.train(fv[all_feature_cols], target,
                                                 feature_names=list(all_feature_cols), force_save=True)
                        if result.get("error"):
                            st.error(f"Auto-retrain failed: {result['error']}")
                        else:
                            st.success(f"Auto-retrained — accuracy: {result.get('accuracy', 0):.1%}")
                            feature_cols = predictor.trained_feature_names or all_feature_cols

                    can_predict = not needs_retrain or not (needs_retrain and (result or {}).get("error"))
                    if can_predict:
                        latest = fv[feature_cols].iloc[-1].values.astype(np.float64)
                        pred = predictor.predict(latest, feature_names=feature_cols)

                        # Regime detection
                        try:
                            from src.model.regime import HMMRegimeDetector
                            regime_det = HMMRegimeDetector()
                            price_df = _pred_router.query(
                                "SELECT close, volume FROM prices ORDER BY date DESC LIMIT 60")
                            macro_df = _pred_router.query(
                                "SELECT vix FROM macro ORDER BY date DESC LIMIT 60")
                            price_df["vix"] = macro_df["vix"].values if not macro_df.empty else 18.0
                            regime = regime_det.predict(price_df)
                            pred["regime"] = regime
                        except Exception:
                            pred["regime"] = ""

                        pred["ensemble_used"] = predictor.ensemble is not None and predictor.use_ensemble

                        today = datetime.now().strftime("%Y-%m-%d")
                        _pred_router.execute(
                            "INSERT OR REPLACE INTO predictions (date, direction, confidence, predicted_at) "
                            "VALUES (?, ?, ?, ?)",
                            (today, pred.get("direction", ""), pred.get("confidence", 0),
                             datetime.now().isoformat())
                        )

                        # Update spy_state.json so SPY Predictor page shows fresh data
                        from src.realtime.dashboard_bridge import write_spy_state
                        ind = {}
                        tech_df = _pred_router.query(
                            "SELECT rsi_14, macd, atr_14 FROM technicals ORDER BY date DESC LIMIT 1"
                        )
                        if not tech_df.empty:
                            ind = {"rsi_14": tech_df.iloc[0]["rsi_14"], "macd": tech_df.iloc[0]["macd"], "atr_14": tech_df.iloc[0]["atr_14"]}
                        macro_df = _pred_router.query(
                            "SELECT vix, vix_change FROM macro ORDER BY date DESC LIMIT 1"
                        )
                        if not macro_df.empty:
                            ind["vix"] = macro_df.iloc[0]["vix"]
                            ind["vix_change"] = macro_df.iloc[0]["vix_change"]
                        if "volume_ratio" in fv.columns:
                            ind["volume_ratio"] = round(float(fv["volume_ratio"].iloc[-1]), 2) if pd.notna(fv["volume_ratio"].iloc[-1]) else None
                        if "sentiment_score" in fv.columns:
                            ind["sentiment_score"] = round(float(fv["sentiment_score"].iloc[-1]), 2) if pd.notna(fv["sentiment_score"].iloc[-1]) else None
                        write_spy_state(prediction=pred, indicators=ind)

                        st.success(f"{pred.get('scale_label', pred.get('direction'))} — {pred.get('confidence', 0):.0f}% confidence")
                        st.json(pred)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        if st.button("LLM Health Check", key="act_llm", help="Check Ollama + model availability"):
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

        if st.button("Generate Report", key="act_report", help="Generate LLM daily report for latest prediction"):
            with st.spinner("Generating LLM report (this may take a few minutes)..."):
                try:
                    from src.llm.reporter import DailyReporter
                    from src.llm.analyzer import LLMAnalyzer
                    config = _load_config()
                    _rpt_router = get_router(config)
                    pred_df = _rpt_router.query(
                        "SELECT date, direction, confidence FROM predictions ORDER BY date DESC LIMIT 1"
                    )
                    if pred_df.empty:
                        st.error("No prediction found — generate one first")
                    else:
                        pred_date = pred_df.iloc[0]["date"]
                        tech_df = _rpt_router.query("SELECT * FROM technicals WHERE date = ?", (pred_date,))
                        sent_df = _rpt_router.query("SELECT * FROM daily_sentiment WHERE date = ?", (pred_date,))
                        macro_df = _rpt_router.query("SELECT * FROM macro WHERE date = ?", (pred_date,))
                        context = {
                            "prediction": {"direction": pred_df.iloc[0]["direction"], "confidence": pred_df.iloc[0]["confidence"]},
                            "technicals": tech_df.iloc[0].to_dict() if not tech_df.empty else {},
                            "sentiment": sent_df.iloc[0].to_dict() if not sent_df.empty else {},
                            "macro": macro_df.iloc[0].to_dict() if not macro_df.empty else {},
                        }
                        llm = LLMAnalyzer(config)
                        reporter = DailyReporter(config)
                        report = reporter.generate_report(context, llm_available=llm.llm_available)
                        _rpt_router.execute("INSERT OR REPLACE INTO predictions (date, report_text) VALUES (?, ?)", (pred_date, report))
                        st.success(f"Report generated ({len(report)} chars)")
                        st.markdown(report)
                except Exception as e:
                    st.error(f"Report generation failed: {e}")

    st.divider()

    # Full pipeline
    st.markdown("**Full Pipeline**")
    pc1, pc2 = st.columns(2)
    with pc1:
        skip_llm = st.checkbox("Skip LLM steps (faster)", value=True, key="skip_llm_check")
    with pc2:
        if st.button("Run Full Pipeline", key="act_pipeline", type="primary"):
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

    # Pipeline Schedule Info
    st.markdown("**Pipeline Schedule**")
    st.caption("The daily pipeline runs automatically via the scheduler (src/launcher.py).")
    sched_c1, sched_c2 = st.columns(2)
    with sched_c1:
        st.markdown(f"""
- Full pipeline: **4:30 PM ET** (Mon–Fri)
- Steps: data pull → news → sentiment → macro → options → technicals → intraday → earnings → fed → **retrain** → **predict** → report → alerts
""")
    with sched_c2:
        st.markdown(f"""
- Intraday updates: **8:30 AM, 12:00 PM, 1:30 PM, 3:00 PM** ET
- Intraday steps: news → sentiment → macro → technicals → intraday → predict
""")
    # Check if scheduler is running
    try:
        import subprocess as _sp
        sched_check = _sp.run(
            ["pgrep", "-af", "src.launcher"],
            capture_output=True, text=True, timeout=5)
        if sched_check.stdout.strip():
            pid = sched_check.stdout.strip().split("\n")[0].split()[0]
            st.success(f"Scheduler is running (PID: {pid})")
        else:
            st.warning("Scheduler not detected — pipeline won't run automatically")
    except Exception:
        st.info("Could not check scheduler status")

    st.divider()

    # Send test alert
    st.markdown("**Alerts**")
    if st.button("Send Test Alert", key="act_alert", help="Send a test prediction alert"):
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
        if st.button("Train Entry Gate", key="act_train_entry",
                      help="Train XGBoost entry gate using triple-barrier labels on intraday data"):
            with st.spinner("Training ES Entry Gate..."):
                try:
                    from src.es_strategy.ai_models import ESEntryGate
                    from src.es_strategy.labeling import generate_training_dataset
                    config = _load_config()

                    # Load intraday bars for training
                    _es_router = get_router(config)
                    bars_df = _es_router.query(
                        "SELECT timestamp, open, high, low, close, volume, vwap "
                        "FROM intraday_bars WHERE ticker='SPY' ORDER BY timestamp"
                    )

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
        if st.button("Train Exit Controller", key="act_train_exit",
                      help="Train CNN exit controller using reversal labels on intraday data"):
            with st.spinner("Training ES Exit Controller..."):
                try:
                    from src.es_strategy.ai_models import ESExitController
                    from src.es_strategy.labeling import generate_training_dataset
                    config = _load_config()

                    from src.data.db_router import get_router as _get_exit_router
                    _exit_router = _get_exit_router(config)
                    bars_df = _exit_router.query(
                        "SELECT timestamp, open, high, low, close, volume, vwap "
                        "FROM intraday_bars WHERE ticker='SPY' ORDER BY timestamp"
                    )

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

    try:
        config = _load_config()
        router = get_router(config)
    except Exception:
        router = None

    # Get table list from the active database
    all_tables = []
    if router and router.using_postgres:
        try:
            tbl_df = router.read_analytics(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
            if not tbl_df.empty:
                all_tables = tbl_df["table_name"].tolist()
        except Exception:
            pass

    if not all_tables:
        # No PostgreSQL tables found
        st.warning("No tables found in PostgreSQL")

    if not all_tables:
        st.error("No database tables found")
        return

    selected = st.selectbox("Table", all_tables, key="db_table")
    db_label = "🐘 PostgreSQL" if (router and router.using_postgres) else "📦 SQLite"
    st.caption(f"Reading from {db_label}")

    col1, col2 = st.columns([1, 1])
    with col1:
        limit = st.number_input("Row limit", value=50, min_value=1, max_value=1000, key="db_limit")
    with col2:
        order = st.selectbox("Order", ["DESC", "ASC"], key="db_order")

    # Determine date column
    date_col = "date"
    if selected == "intraday_bars":
        date_col = "timestamp"
    elif selected in ("news", "raw_articles", "model_registry"):
        date_col = "id"

    try:
        if router:
            df = router.read_analytics(
                f"SELECT * FROM {selected} ORDER BY {date_col} {order} LIMIT {limit}"
            )
            count_df = router.read_analytics(f"SELECT COUNT(*) as cnt FROM {selected}")
            total = int(count_df.iloc[0]["cnt"]) if not count_df.empty else 0
        else:
            st.error("Database connection unavailable")
            return
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df)} of {total} rows")
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
                if router:
                    df = router.read_analytics(query)
                    st.caption(f"{db_label} query executed")
                else:
                    st.error("Database connection unavailable")
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"{len(df)} rows returned")
            except Exception as e:
                st.error(f"Query error: {e}")

    st.divider()

    # DB maintenance
    st.subheader("Maintenance")
    mc1, mc2 = st.columns(2)
    with mc1:
        if st.button("Vacuum Database", key="db_vacuum", help="Reclaim unused space"):
            try:
                if router and router.using_postgres:
                    pg = router.get_pg()
                    old_isolation = pg.isolation_level
                    pg.set_isolation_level(0)  # AUTOCOMMIT required for VACUUM
                    pg.cursor().execute("VACUUM ANALYZE")
                    pg.set_isolation_level(old_isolation)
                    pg.close()
                    st.success("PostgreSQL VACUUM ANALYZE complete")
                else:
                    st.warning("No database connection available")
            except Exception as e:
                st.error(f"Vacuum failed: {e}")
    with mc2:
        if st.button("Integrity Check", key="db_integrity"):
            try:
                if router and router.using_postgres:
                    check_df = router.read_analytics(
                        "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public'"
                    )
                    tbl_count = len(check_df) if not check_df.empty else 0
                    st.success(f"PostgreSQL: {tbl_count} tables in public schema — OK")
                else:
                    st.warning("No database connection available")
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

    # Read-only summary of key settings
    st.markdown("**Key Settings** (read-only)")
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("LLM Model", config.get("llm", {}).get("model", "—"))
    kc2.metric("XGB Lookback", f"{config.get('xgboost', {}).get('lookback_days', '—')} days")
    kc3.metric("ES Max Lots", config.get("es_strategy", {}).get("max_lots", "—"))
    kc4.metric("Cloud Sync", "On" if config.get("sync", {}).get("enabled") else "Off")

    st.divider()
    st.info("📋 Strategy parameters have moved to the **Strategy Rules** page in the Operations sidebar. "
            "All ES engine settings (spread, sizing, entry, TP, risk, AI, RL) are now managed from the database "
            "and can be edited live without restarting.")


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
        try:
            config = _load_config()
            router = get_router(config)
            pred_df = router.read_analytics(
                "SELECT date, direction, confidence, report_text, predicted_at "
                "FROM predictions ORDER BY date DESC LIMIT 5"
            )
            if not pred_df.empty:
                for _, p in pred_df.iterrows():
                    with st.expander(f"{p['date']} — {p['direction']} ({p['confidence']:.0f}%) at {p['predicted_at'] or '—'}"):
                        if p['report_text']:
                            st.markdown(p['report_text'])
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
        config = _load_config_cached()
        router = get_router(config)
        close_df = router.read_analytics(
            "SELECT close FROM prices ORDER BY date DESC LIMIT 1"
        )
        if not close_df.empty:
            spy_close = close_df.iloc[0]["close"]
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
    st.markdown(page_header("📡 System Monitoring"), unsafe_allow_html=True)
    st.caption(
        "Live system monitoring powered by Grafana. Dashboards cover SPY Predictor performance, "
        "ES Strategy P&L, system health, and the data pipeline status."
    )

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
# QUANT AGENT PAGE (Admin-only)
# ======================================================================


def page_quant_agent():
    c = get_colors()
    st.markdown(page_header('🤖 Quant Agent'), unsafe_allow_html=True)

    # Admin-only gate
    current_user = get_user()
    current_role = current_user.get("role", "viewer") if current_user else "viewer"
    if current_role != "admin":
        st.warning("Quant Agent is available to admin users only.")
        return

    st.caption("Conversational AI assistant powered by DeepSeek R1 70B. "
               "Ask about predictions, news sentiment, query databases, or analyze features.")

    # Initialize agent in session state
    if "quant_agent" not in st.session_state:
        from src.llm.quant_agent import QuantAgent
        st.session_state.quant_agent = QuantAgent(_load_config())
    if "quant_messages" not in st.session_state:
        st.session_state.quant_messages = []

    agent = st.session_state.quant_agent

    # ── Helper: format tool results as markdown (no LLM needed) ──
    def _fmt_prediction(data):
        if "error" in data:
            return f"⚠️ Error: {data['error']}"
        pred = data.get("prediction", {})
        ind = data.get("indicators", {})
        direction = pred.get("direction", "N/A")
        conf = pred.get("confidence", 0)
        scale = pred.get("scale_label", "")
        probs = pred.get("probabilities", {})
        emoji = "🟢" if "BULLISH" in direction else "🔴" if "BEARISH" in direction else "⚪"
        lines = [
            f"## {emoji} SPY Prediction: **{scale or direction}**",
            f"**Confidence**: {conf:.1f}%",
            f"**Updated**: {data.get('updated_at', 'N/A')}",
            "", "**Probabilities**:",
        ]
        for k, v in probs.items():
            lines.append(f"- {k}: {v:.1f}%" if isinstance(v, (int, float)) else f"- {k}: {v}")
        if pred.get("prediction_set"):
            lines += ["", f"**Conformal Set**: {', '.join(pred['prediction_set'])} (size {pred.get('set_size', '?')})"]
        if pred.get("regime"):
            regime_emoji = {"bull_trend": "🟢", "bear_trend": "🔴", "high_vol_choppy": "🟡", "low_vol_range": "🔵"}
            r = pred["regime"]
            lines.append(f"**Regime**: {regime_emoji.get(r, '⚪')} {r}")
        if pred.get("shap_drivers"):
            lines += ["", "**Top SHAP Drivers**:"]
            for d in pred["shap_drivers"][:5]:
                sign = "+" if d["shap_value"] > 0 else ""
                lines.append(f"- `{d['feature']}`: {sign}{d['shap_value']:.4f} (val={d['feature_value']:.4f})")
        if ind:
            lines += ["", "**Key Indicators**:"]
            for k, v in ind.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"- {k}: {val}")
        return "\n".join(lines)

    def _fmt_features(data):
        if "error" in data:
            return f"⚠️ Error: {data['error']}"
        lines = [
            f"## 🔬 Feature Importance",
            f"**Total features**: {data['total_features']} | **Top {len(data['top_features'])} account for {data['top_feature_pct']}%**",
            "",
            "| Rank | Feature | Importance |",
            "|------|---------|------------|",
        ]
        for i, f in enumerate(data["top_features"], 1):
            bar = "█" * int(f["importance"] * 100)
            lines.append(f"| {i} | `{f['name']}` | {f['importance']:.4f} {bar} |")
        return "\n".join(lines)

    def _fmt_news(data):
        if "error" in data:
            return f"⚠️ Error: {data['error']}"
        avg = data["avg_sentiment"]
        emoji = "🟢" if avg > 0.05 else "🔴" if avg < -0.05 else "⚪"
        lines = [
            f"## 📰 News Sentiment Summary",
            f"**Articles**: {data['total_articles']} | **Sources**: {data['unique_sources']} | **Avg Sentiment**: {emoji} {avg:.4f}",
        ]
        if data.get("category_breakdown"):
            lines += ["", "**By Category**:", "| Category | Count | Avg Sentiment |", "|----------|-------|---------------|"]
            for cat in data["category_breakdown"]:
                s = cat.get("avg_sent", 0) or 0
                ce = "🟢" if s > 0.05 else "🔴" if s < -0.05 else "⚪"
                lines.append(f"| {cat.get('category', 'N/A')} | {cat.get('count', 0)} | {ce} {s:.4f} |")
        if data.get("top_headlines"):
            lines += ["", "**Top Headlines** (by sentiment strength):"]
            for h in data["top_headlines"][:7]:
                s = h.get("sentiment_compound", 0) or 0
                he = "🟢" if s > 0 else "🔴" if s < 0 else "⚪"
                lines.append(f"- {he} **{h.get('headline', 'N/A')}** ({h.get('source', '')}, {s:+.3f})")
        return "\n".join(lines)

    def _fmt_regime(data):
        if "error" in data:
            return f"⚠️ Error: {data['error']}"
        regime_emoji = {"bull_trend": "🟢", "bear_trend": "🔴", "high_vol_choppy": "🟡", "low_vol_range": "🔵"}
        current = data.get("current_regime", "unknown")
        lines = [
            f"## {regime_emoji.get(current, '⚪')} Regime History ({data['days']}d)",
            f"**Current Regime**: {current}",
            "",
            "**Distribution**:",
        ]
        for regime, count in data.get("regime_distribution", {}).items():
            lines.append(f"- {regime_emoji.get(regime, '⚪')} {regime}: {count} days")
        if data.get("regime_history"):
            lines += ["", "**Recent History**:", "| Date | Regime |", "|------|--------|"]
            for entry in data["regime_history"]:
                r = entry["regime"]
                lines.append(f"| {entry['date']} | {regime_emoji.get(r, '⚪')} {r} |")
        return "\n".join(lines)

    def _fmt_correlations(data):
        if "error" in data:
            return f"⚠️ Error: {data['error']}"
        lines = [
            f"## 🔗 Feature Correlations",
            f"**Features analyzed**: {data['total_features']} | **Threshold**: {data['threshold']} | **High-corr pairs**: {data['high_corr_count']}",
        ]
        if data.get("high_corr_pairs"):
            lines += ["", "**Highly Correlated Pairs**:", "| Feature 1 | Feature 2 | Correlation |", "|-----------|-----------|-------------|"]
            for p in data["high_corr_pairs"][:10]:
                lines.append(f"| `{p['feature_1']}` | `{p['feature_2']}` | {p['correlation']:+.3f} |")
        if data.get("vif_top10"):
            lines += ["", "**Top VIF Scores** (>10 = severe multicollinearity):"]
            for v in data["vif_top10"][:7]:
                if "note" in v:
                    lines.append(f"- ⚠️ {v['note']}")
                else:
                    flag = "🔴" if v["vif"] > 10 else "🟡" if v["vif"] > 5 else "🟢"
                    lines.append(f"- {flag} `{v['feature']}`: VIF={v['vif']}")
        if data.get("drop_suggestions"):
            lines += ["", f"**Suggested drops**: {', '.join(f'`{d}`' for d in data['drop_suggestions'])}"]
        return "\n".join(lines)

    def _fmt_risk(data):
        if "error" in data:
            return f"⚠️ Error: {data['error']}"
        level = data.get("risk_level", "UNKNOWN")
        le = "🔴" if level == "HIGH" else "🟡" if level == "MODERATE" else "🟢"
        avg_s = data.get('avg_sentiment') or 0
        neg_r = data.get('negative_ratio') or 0
        lines = [
            f"## {le} News Risk Assessment: **{level}**",
            f"**Articles scanned**: {data.get('total_articles', 0)} | "
            f"**Avg sentiment**: {avg_s:.4f} | "
            f"**Negative ratio**: {neg_r:.0%}",
        ]
        if data.get("high_impact"):
            lines += ["", "**High-Impact Headlines** (strongest sentiment):"]
            for a in data["high_impact"][:10]:
                s = a.get("sentiment_compound", 0) or 0
                se = "🔴" if s < -0.3 else "🟡" if s < 0 else "🟢"
                hl = a["headline"][:65] + ("..." if len(a["headline"]) > 65 else "")
                lines.append(f"- {se} **{hl}** ({a.get('source', '')}, {s:+.3f})")
        if data.get("category_risk"):
            lines += ["", "**Risk by Category**:",
                       "| Category | Articles | Avg Sentiment | Risk |",
                       "|----------|----------|---------------|------|"]
            for cr in data["category_risk"]:
                s = cr.get("avg_sent") or 0
                rl = "🔴 HIGH" if s < -0.15 else "🟡 MED" if s < 0 else "🟢 LOW"
                lines.append(f"| {cr.get('category', 'N/A')} | {cr.get('count', 0)} | {s:+.4f} | {rl} |")
        return "\n".join(lines)

    def _fmt_alpha(data):
        if "error" in data:
            return f"⚠️ Error: {data['error']}"
        lines = [
            f"## 💡 Alpha Factor Analysis",
            f"**Current regime**: {data.get('current_regime', 'N/A')} | "
            f"**Active features**: {data.get('current_features', '?')} | "
            f"**Model confidence**: {data.get('confidence', 'N/A')}",
        ]
        if data.get("weak_features"):
            lines += ["", "**Weakest Features** (lowest importance — candidates for replacement):"]
            for wf in data["weak_features"]:
                lines.append(f"- `{wf['name']}`: importance={wf['importance']:.4f}")
        if data.get("missing_categories"):
            lines += ["", "**Unexplored Feature Categories**:"]
            for cat in data["missing_categories"]:
                lines.append(f"- 💡 {cat}")
        if data.get("regime_suggestions"):
            lines += ["", f"**Regime-Specific Ideas** ({data.get('current_regime', '')}):"]
            for s in data["regime_suggestions"]:
                lines.append(f"- {s}")
        lines += ["", "*For AI-generated hypotheses, ask in chat: \"Generate alpha factor ideas\"*"]
        return "\n".join(lines)

    def _fmt_explain_regime(data):
        if "error" in data:
            return f"⚠️ Error: {data['error']}"
        remoji = {"bull_trend": "🟢", "bear_trend": "🔴", "high_vol_choppy": "🟡", "low_vol_range": "🔵"}
        regime_desc = {
            "bull_trend": "Sustained upward momentum with low volatility. Trend-following strategies tend to outperform.",
            "bear_trend": "Persistent selling pressure. Defensive positioning and hedging recommended.",
            "high_vol_choppy": "Elevated volatility with no clear direction. Mean-reversion strategies may work. Reduce position sizes.",
            "low_vol_range": "Low volatility, range-bound market. Breakout signals are unreliable. Patience required.",
        }
        current = data.get("current_regime", "unknown")
        ki = data.get("key_indicators", {})
        lines = [
            f"## {remoji.get(current, '⚪')} Market Regime: **{current}**",
            f"*{regime_desc.get(current, 'Unknown regime state.')}*",
            "",
        ]
        if ki:
            vix = ki.get("vix")
            rsi = ki.get("rsi_14")
            macd = ki.get("macd")
            vol_ratio = ki.get("volume_ratio")
            sent = ki.get("news_sentiment")
            pct1 = ki.get("spy_1d_pct", 0)
            pct5 = ki.get("spy_5d_pct", 0)
            lines.append("**Key Indicators**:")
            if vix is not None:
                vix_label = "🔴 Elevated" if vix > 25 else "🟡 Normal" if vix > 15 else "🟢 Low"
                lines.append(f"- VIX: {vix} ({vix_label})")
            if rsi is not None:
                rsi_label = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "⚪ Neutral"
                lines.append(f"- RSI(14): {rsi:.1f} ({rsi_label})")
            if macd is not None:
                macd_label = "🟢 Bullish" if macd > 0 else "🔴 Bearish"
                lines.append(f"- MACD: {macd:.4f} ({macd_label})")
            if vol_ratio is not None:
                vol_label = "🔴 Heavy" if vol_ratio > 1.3 else "🟢 Light" if vol_ratio < 0.7 else "⚪ Normal"
                lines.append(f"- Volume Ratio: {vol_ratio:.2f} ({vol_label})")
            if sent is not None:
                sent_label = "🟢 Positive" if sent > 0.05 else "🔴 Negative" if sent < -0.05 else "⚪ Neutral"
                lines.append(f"- News Sentiment: {sent:.4f} ({sent_label})")
            lines.append(f"- SPY 1d: {pct1:+.2f}% | 5d: {pct5:+.2f}%")
        if data.get("regime_distribution"):
            lines += ["", "**Regime Distribution (14d)**:"]
            for regime, count in data["regime_distribution"].items():
                lines.append(f"- {remoji.get(regime, '⚪')} {regime}: {count} days")
        if data.get("watch_for"):
            lines += ["", "**Watch For** (regime change signals):"]
            for w in data["watch_for"]:
                lines.append(f"- ⚡ {w}")
        lines += ["", "*For deeper AI analysis, ask in chat: \"Explain the current regime in detail\"*"]
        return "\n".join(lines)

    def _data_risk_assessment():
        """Data-only risk assessment using VADER sentiment scores — no LLM."""
        try:
            from src.data.db_router import get_router
            router = get_router(_load_config())
            from datetime import datetime, timedelta
            cutoff = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            stats = router.query(
                f"SELECT COUNT(*) as count, AVG(sentiment_compound) as avg_sent, "
                f"SUM(CASE WHEN sentiment_compound < -0.15 THEN 1 ELSE 0 END) as neg_count "
                f"FROM raw_articles WHERE published_at >= '{cutoff}'"
            )
            top = router.query(
                f"SELECT headline, source, category, sentiment_compound "
                f"FROM raw_articles WHERE published_at >= '{cutoff}' "
                f"ORDER BY ABS(sentiment_compound) DESC LIMIT 10"
            )
            cats = router.query(
                f"SELECT category, COUNT(*) as count, AVG(sentiment_compound) as avg_sent "
                f"FROM raw_articles WHERE published_at >= '{cutoff}' AND category IS NOT NULL "
                f"GROUP BY category ORDER BY AVG(sentiment_compound) ASC"
            )
            total = int(stats.iloc[0]["count"]) if not stats.empty else 0
            avg_s = float(stats.iloc[0]["avg_sent"] or 0) if not stats.empty else 0
            neg_c = int(stats.iloc[0]["neg_count"] or 0) if not stats.empty else 0
            neg_ratio = neg_c / max(total, 1)
            level = "HIGH" if avg_s < -0.1 or neg_ratio > 0.4 else ("MODERATE" if avg_s < 0 or neg_ratio > 0.25 else "LOW")
            return {
                "total_articles": total, "avg_sentiment": avg_s,
                "negative_ratio": neg_ratio, "risk_level": level,
                "high_impact": top.to_dict(orient="records") if not top.empty else [],
                "category_risk": cats.to_dict(orient="records") if not cats.empty else [],
            }
        except Exception as e:
            return {"error": str(e)}

    def _data_alpha_analysis():
        """Data-only alpha analysis — shows model gaps and suggestions, no LLM."""
        try:
            import glob, json as _json
            import xgboost as xgb
            state = agent._tool_get_prediction_state()
            pred = state.get("prediction", {})
            regime = pred.get("regime", "unknown")
            conf = pred.get("confidence", "N/A")
            # Load feature importances
            model_files = sorted(glob.glob("./models/xgb_spy_*.json"))
            model_files = [f for f in model_files if "_meta" not in f and "_binary" not in f and "_conformal" not in f]
            weak_features = []
            n_features = 0
            if model_files:
                model = xgb.XGBClassifier()
                model.load_model(model_files[-1])
                meta_file = model_files[-1].replace(".json", "_meta.json")
                feature_names = [f"f{i}" for i in range(model.n_features_in_)]
                if os.path.exists(meta_file):
                    with open(meta_file) as f:
                        meta = _json.load(f)
                        feature_names = meta.get("feature_names", feature_names)
                importances = model.feature_importances_
                pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])
                weak_features = [{"name": n, "importance": round(float(v), 4)} for n, v in pairs[:5]]
                n_features = len(feature_names)
            existing_cats = {"price", "momentum", "volatility", "volume", "macro", "sentiment", "options", "microstructure", "earnings", "fed", "geopolitical"}
            missing = []
            for idea in ["Volatility surface (skew, term structure)", "Cross-asset momentum divergence (bonds vs equities)",
                         "Options flow imbalance (put/call volume ratio changes)", "Credit spreads (HY-IG spread dynamics)",
                         "Institutional positioning (COT report features)", "Intraday momentum patterns (first/last hour returns)"]:
                missing.append(idea)
            regime_suggestions = {
                "bull_trend": ["Momentum acceleration features", "Breadth thrust indicators", "Sector rotation signals"],
                "bear_trend": ["Credit stress indicators", "Safe-haven flow ratios (gold/bonds)", "Volatility term structure inversion"],
                "high_vol_choppy": ["Mean-reversion speed features", "Realized vs implied vol spread", "Gamma exposure estimates"],
                "low_vol_range": ["Breakout probability features", "Volume compression indicators", "Bollinger Band squeeze duration"],
            }
            return {
                "current_regime": regime, "current_features": n_features,
                "confidence": f"{conf:.1f}%" if isinstance(conf, (int, float)) else conf,
                "weak_features": weak_features,
                "missing_categories": missing[:4],
                "regime_suggestions": regime_suggestions.get(regime, ["No specific suggestions for this regime"]),
            }
        except Exception as e:
            return {"error": str(e)}

    def _data_explain_regime():
        """Data-only regime explanation — indicators + rules, no LLM."""
        try:
            state = agent._tool_get_prediction_state()
            regime_info = agent._tool_get_regime_history(days=14)
            news = agent._tool_get_news_summary(days=2)
            indicators = state.get("indicators", {})
            prediction = state.get("prediction", {})
            current = regime_info.get("current_regime", "unknown")
            regime_dist = regime_info.get("regime_distribution", {})
            # Price action
            from src.data.db_router import get_router
            router = get_router(_load_config())
            prices = router.query("SELECT date, close FROM prices ORDER BY date DESC LIMIT 10")
            pct1, pct5 = 0, 0
            if not prices.empty and len(prices) >= 2:
                pct1 = round((prices.iloc[0]["close"] / prices.iloc[1]["close"] - 1) * 100, 2)
            if not prices.empty and len(prices) >= 5:
                pct5 = round((prices.iloc[0]["close"] / prices.iloc[4]["close"] - 1) * 100, 2)
            # Watch-for signals
            vix = indicators.get("vix", 20)
            rsi = indicators.get("rsi_14", 50)
            watch = []
            if current == "low_vol_range":
                if vix < 14: watch.append("VIX extremely low — complacency risk, potential vol spike")
                if abs(pct5) < 0.5: watch.append("5-day range very tight — breakout imminent")
                watch.append("Watch for volume surge as breakout catalyst")
            elif current == "bull_trend":
                if rsi > 65: watch.append(f"RSI at {rsi:.0f} — approaching overbought territory")
                if vix < 13: watch.append("VIX very low — potential mean reversion in vol")
                watch.append("Watch for breadth divergence (fewer stocks making new highs)")
            elif current == "bear_trend":
                if rsi < 35: watch.append(f"RSI at {rsi:.0f} — approaching oversold, bounce possible")
                if vix > 30: watch.append(f"VIX at {vix:.0f} — fear elevated, capitulation watch")
                watch.append("Watch for credit spread widening as contagion signal")
            elif current == "high_vol_choppy":
                watch.append("Watch for VIX term structure normalization")
                watch.append("Consecutive closes in same direction = potential regime shift")
            return {
                "current_regime": current,
                "key_indicators": {
                    "vix": indicators.get("vix"), "rsi_14": indicators.get("rsi_14"),
                    "macd": indicators.get("macd"), "volume_ratio": indicators.get("volume_ratio"),
                    "news_sentiment": news.get("avg_sentiment", 0),
                    "spy_1d_pct": pct1, "spy_5d_pct": pct5,
                },
                "regime_distribution": regime_dist,
                "watch_for": watch,
            }
        except Exception as e:
            return {"error": str(e)}

    def _run_direct_tool(tool_fn, formatter, label, **kwargs):
        """Call a tool directly, format result, append to chat. No LLM."""
        st.session_state.quant_messages.append({"role": "user", "content": label})
        with st.spinner("Fetching data..."):
            result = tool_fn(**kwargs)
        md = formatter(result)
        st.session_state.quant_messages.append({"role": "assistant", "content": md})
        st.rerun()

    # Quick action buttons — Row 1 (instant — no LLM)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("📊 Current Prediction", key="qa_pred", use_container_width=True):
            _run_direct_tool(agent._tool_get_prediction_state, _fmt_prediction,
                             "📊 Current Prediction")
    with q2:
        if st.button("🔬 Feature Importance", key="qa_feat", use_container_width=True):
            _run_direct_tool(agent._tool_get_feature_importance, _fmt_features,
                             "🔬 Feature Importance")
    with q3:
        if st.button("📰 News Sentiment", key="qa_news", use_container_width=True):
            _run_direct_tool(agent._tool_get_news_summary, _fmt_news,
                             "📰 News Sentiment")
    with q4:
        if st.button("📊 Regime History", key="qa_regime_hist", use_container_width=True):
            _run_direct_tool(agent._tool_get_regime_history, _fmt_regime,
                             "📊 Regime History")

    # ── Quick action buttons — Row 2 (all instant, data-only) ──
    q5, q6, q7, q8 = st.columns(4)
    with q5:
        if st.button("⚠️ Risk Assessment", key="qa_risk", use_container_width=True):
            _run_direct_tool(_data_risk_assessment, _fmt_risk,
                             "⚠️ Risk Assessment")
    with q6:
        if st.button("💡 Alpha Ideas", key="qa_alpha", use_container_width=True):
            _run_direct_tool(_data_alpha_analysis, _fmt_alpha,
                             "💡 Alpha Analysis")
    with q7:
        if st.button("🔗 Correlations", key="qa_corr", use_container_width=True):
            _run_direct_tool(agent._tool_analyze_feature_correlations, _fmt_correlations,
                             "🔗 Feature Correlations")
    with q8:
        if st.button("🌊 Explain Regime", key="qa_regime", use_container_width=True):
            _run_direct_tool(_data_explain_regime, _fmt_explain_regime,
                             "🌊 Explain Regime")

    # ── Quick action buttons — Row 3 (agentic intelligence) ──
    def _fmt_thesis(data: dict) -> str:
        if "error" in data:
            return f"❌ Error: {data['error']}"
        lines = [f"**🎯 Market Thesis** — {data.get('direction', '?')} (conf {data.get('confidence', 0):.1%})"]
        lines.append(f"Regime: `{data.get('regime', 'unknown')}`\n")
        for p in data.get("pillars", []):
            icon = "✅" if p["status"] == "supporting" else ("⚠️" if p["status"] == "weakening" else "❌")
            lines.append(f"{icon} **{p['name']}** [{p['strength']}] — {p['detail']}")
        s = data.get("summary", {})
        if s:
            lines.append(f"\n**Thesis: {s.get('thesis_strength', '?').upper()}** — {s.get('conviction', '')}")
        return "\n".join(lines)

    def _fmt_vigilance(data: dict) -> str:
        if "error" in data:
            return f"❌ Error: {data['error']}"
        msg = data.get("message", "")
        alerts = data.get("alerts", [])
        if not alerts:
            return f"✅ {msg}"
        lines = [f"🚨 **Vigilance Alerts** — {msg}\n"]
        for a in alerts:
            ts = a.get("time", "")[:19] if a.get("time") else ""
            lines.append(f"- **{a.get('type', '?')}** ({ts}): {a.get('message', '')}")
        return "\n".join(lines)

    q9, q10, _q11, _q12 = st.columns(4)
    with q9:
        if st.button("🎯 Market Thesis", key="qa_thesis", use_container_width=True):
            _run_direct_tool(agent._tool_get_market_thesis, _fmt_thesis,
                             "🎯 Market Thesis")
    with q10:
        if st.button("🚨 Vigilance Alerts", key="qa_vigil", use_container_width=True):
            _run_direct_tool(agent._tool_get_vigilance_alerts, _fmt_vigilance,
                             "🚨 Vigilance Alerts")

    st.divider()

    # Chat history display
    for msg in st.session_state.quant_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart"):
                import plotly.graph_objects as go
                fig = go.Figure(msg["chart"])
                fig.update_layout(**get_plotly_layout())
                st.plotly_chart(fig, use_container_width=True)

    # Process pending typed chat message
    if (st.session_state.quant_messages
            and st.session_state.quant_messages[-1]["role"] == "user"):
        pending = st.session_state.quant_messages[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, chart_data = agent.chat(pending)
            st.markdown(response)
            msg_data = {"role": "assistant", "content": response}
            if chart_data:
                msg_data["chart"] = chart_data
                import plotly.graph_objects as go
                fig = go.Figure(chart_data)
                fig.update_layout(**get_plotly_layout())
                st.plotly_chart(fig, use_container_width=True)
            st.session_state.quant_messages.append(msg_data)

    # Chat input
    if prompt := st.chat_input("Ask the quant agent anything..."):
        st.session_state.quant_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        st.rerun()

    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Agent Controls**")
    if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.quant_messages = []
        st.session_state.quant_agent = None
        st.rerun()
    st.sidebar.caption(f"Models: {agent.model_fast} (fast) / {agent.model} (deep)")
    st.sidebar.caption(f"History: {len(st.session_state.quant_messages)} messages")




# ======================================================================
# ROUTER — st.navigation handles page dispatch
# ======================================================================

# ── Global live price ticker helper (defined at module level for reuse) ──
@st.cache_data(ttl=15, show_spinner=False)
def _fetch_ticker_price(symbol: str) -> tuple:
    """Cached live price fetch (15s TTL). Returns (price, prev_close) or (0, 0)."""
    try:
        import yfinance as _yf
        _fi = _yf.Ticker(symbol).fast_info
        _price = float(getattr(_fi, "last_price", 0) or 0)
        _prev = float(getattr(_fi, "previous_close", 0) or 0)
        return (_price, _prev)
    except Exception:
        return (0.0, 0.0)


def _global_live_ticker():
    _colors = get_colors()
    _sym = st.session_state.get("live_ticker_symbol", "SPY")
    try:
        _price, _prev = _fetch_ticker_price(_sym)
        if _price <= 0 or _prev <= 0:
            return
        _chg = _price - _prev
        _pct = (_chg / _prev) * 100
        _arrow = "▲" if _chg >= 0 else "▼"
        _c = _colors["green"] if _chg >= 0 else _colors["red"]
        _now = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:14px; '
            f'padding:6px 14px; background:{_colors["card"]}; '
            f'border:1px solid {_colors["card_border"]}; border-radius:8px; '
            f'margin-bottom:8px;">'
            f'<span style="color:{_colors["text"]}; font-weight:600;">{_sym}</span>'
            f'<span style="color:{_colors["text"]}; font-size:1.2em; font-weight:700;">'
            f'${_price:,.2f}</span>'
            f'<span style="color:{_c}; font-weight:600;">'
            f'{_arrow} {_chg:+.2f} ({_pct:+.2f}%)</span>'
            f'<span style="color:{_colors["text_secondary"]}; font-size:0.75em; '
            f'margin-left:auto;">Live · {_now}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass


# ── Only run navigation + page dispatch when this is the main Streamlit script ──
if _IS_MAIN_SCRIPT:
    _pages = {
        "Dashboards": [
            st.Page(page_market_overview, title="Market Overview", icon=":material/space_dashboard:", default=True),
            st.Page(page_spy, title="SPY Predictor", icon=":material/query_stats:"),
            st.Page(page_es, title="ES Strategy", icon=":material/candlestick_chart:"),
            st.Page(page_strangle, title="Inverted Strangle", icon=":material/mediation:"),
        ],
        "Analysis": [
            st.Page(page_single_stock, title="Single-Stock Analysis", icon=":material/search:"),
            st.Page(page_performance, title="Performance Tracking", icon=":material/verified:"),
            st.Page(page_scenario_analysis, title="Scenario Analysis", icon=":material/science:"),
            st.Page(page_tuning, title="Model Tuning", icon=":material/tune:"),
        ],
        "Administration": [
            st.Page(page_rules, title="Strategy Rules", icon=":material/rule:"),
            st.Page(page_grafana, title="System Monitoring", icon=":material/dashboard:"),
            st.Page(page_data_management, title="Data Management", icon=":material/database:"),
            st.Page(page_system_management, title="System Management", icon=":material/settings:"),
            st.Page(page_quant_agent, title="Quant Agent", icon=":material/smart_toy:"),
        ],
    }

    _pg = st.navigation(_pages)

    @st.fragment(run_every=15)
    def _live_ticker_fragment():
        _global_live_ticker()

    _live_ticker_fragment()

    st.sidebar.divider()
    mode_label = "☁️ Cloud" if IS_CLOUD else "🖥️ Local"
    st.sidebar.caption(f"{mode_label} mode")
    if user and user.get("email") != "anonymous":
        if st.sidebar.button("🚪 Sign Out", use_container_width=True):
            logout()
            st.rerun()

    _pg.run()
