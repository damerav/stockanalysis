"""Market Overview — High-level at-a-glance summary dashboard.

This is the default landing page. It provides a concise summary of:
  - The current SPY prediction signal and confidence
  - System health status (DB, LLM, Scheduler, Data Sources)
  - Key market indicator gauges (VIX, Fear & Greed, TRIN, Buffett Indicator)
  - Top SHAP drivers (compact, no interaction required)
"""

import os
import logging
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def page_market_overview():
    """Renders the Market Overview dashboard."""
    # ── Lazy imports to avoid circular dependency ──────────────────────
    import yaml
    from src.dashboard.theme import get_colors, page_header, metric_card, is_dark
    from src.data.db_router import get_router

    def _load_cfg():
        try:
            with open("config.yaml") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def _spy_state():
        relay = os.environ.get("RELAY_URL", "")
        if relay:
            try:
                r = requests.get(f"{relay}/state", timeout=5)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                pass
            return {}
        try:
            import json
            with open("./data/spy_state.json") as f:
                return json.load(f)
        except Exception:
            return {}

    c = get_colors()
    state = _spy_state()
    prediction = state.get("prediction", {})

    # ── Page header ────────────────────────────────────────────────────
    st.markdown(page_header("📊 Market Overview"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: HERO PREDICTION SIGNAL
    # ══════════════════════════════════════════════════════════════════
    if prediction:
        scale_label = prediction.get("scale_label", prediction.get("direction", "NEUTRAL"))
        confidence = prediction.get("confidence", 0)
        conf_set = prediction.get("prediction_set", [])
        regime = prediction.get("regime", "")
        is_low_conv = prediction.get("is_low_conviction", False)
        updated_at = state.get("updated_at", "")

        color_map = {
            "STRONG_BULLISH": c["green"], "BULLISH": c["green"], "WEAK_BULLISH": c["green"],
            "NEUTRAL": c["yellow"],
            "WEAK_BEARISH": c["red"], "BEARISH": c["red"], "STRONG_BEARISH": c["red"],
        }
        banner_color = color_map.get(scale_label, c["yellow"])
        arrow = "▲" if "BULLISH" in scale_label else "▼" if "BEARISH" in scale_label else "◆"
        conf_interp = "Weak" if confidence < 55 else "Moderate" if confidence < 70 else "Strong" if confidence < 85 else "Very Strong"
        regime_labels = {
            "bull_trend": "🟢 Bull Trend", "bear_trend": "🔴 Bear Trend",
            "high_vol_choppy": "🟡 Choppy", "low_vol_range": "🔵 Range-Bound",
        }
        updated_str = updated_at[:16].replace("T", " ") if updated_at else "—"

        glow = f"box-shadow: 0 6px 32px {banner_color}55, 0 2px 8px rgba(0,0,0,0.3);"
        st.markdown(
            f"""<div style="background: linear-gradient(135deg, {banner_color}ee 0%, {banner_color} 100%);
            border-radius:16px; padding:24px 28px; margin-bottom:16px; {glow}">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
                <div style="display:flex; align-items:center; gap:14px;">
                    <span style="font-size:2.8rem; color:#fff; line-height:1;">{arrow}</span>
                    <div>
                        <div style="font-size:1.9rem; font-weight:800; color:#fff; letter-spacing:1px;">
                            {scale_label.replace('_', ' ')}
                        </div>
                        <div style="font-size:0.85rem; color:rgba(255,255,255,0.8); margin-top:2px;">
                            Next-day SPY direction · {conf_interp} signal
                            {' · ⚠️ Low Conviction' if is_low_conv else ''}
                        </div>
                    </div>
                </div>
                <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
                    <div style="text-align:center; background:rgba(0,0,0,0.15); border-radius:10px; padding:10px 20px;">
                        <div style="font-size:2.5rem; font-weight:900; color:#fff; line-height:1;">{confidence:.0f}%</div>
                        <div style="font-size:0.7rem; color:rgba(255,255,255,0.7); text-transform:uppercase; letter-spacing:0.1em;">Confidence</div>
                    </div>
                    <div style="text-align:center; background:rgba(0,0,0,0.15); border-radius:10px; padding:10px 20px;">
                        <div style="font-size:0.95rem; font-weight:700; color:#fff;">{regime_labels.get(regime, regime or '—')}</div>
                        <div style="font-size:0.7rem; color:rgba(255,255,255,0.7); text-transform:uppercase; letter-spacing:0.1em;">Regime</div>
                    </div>
                    <div style="text-align:center; background:rgba(0,0,0,0.15); border-radius:10px; padding:10px 20px;">
                        <div style="font-size:0.95rem; font-weight:700; color:#fff;">{', '.join(conf_set) if conf_set else '—'}</div>
                        <div style="font-size:0.7rem; color:rgba(255,255,255,0.7); text-transform:uppercase; letter-spacing:0.1em;">Conf. Set</div>
                    </div>
                </div>
            </div>
            <div style="text-align:right; margin-top:8px; font-size:0.7rem; color:rgba(255,255,255,0.5);">
                Updated {updated_str}
            </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("⏳ Waiting for prediction data. Run the daily pipeline from Data Management to generate a prediction.")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: KEY MARKET INDICATORS (2 rows of 4)
    # ══════════════════════════════════════════════════════════════════
    st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.85rem;margin-top:8px;margin-bottom:4px;">KEY MARKET INDICATORS</p>', unsafe_allow_html=True)

    # Row 1: Macro
    try:
        from src.data.fetcher import FallbackFetcher
        cfg = _load_cfg()
        fetcher = FallbackFetcher(config=cfg)
        live_macro = fetcher.get_macro_fred()
    except Exception:
        live_macro = {}

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        v = live_macro.get("vix")
        vc = live_macro.get("vix_change")
        st.metric("VIX", f"{v:.1f}" if v else "—",
                  delta=f"{vc:+.1f}" if vc else None, delta_color="inverse",
                  help="CBOE Volatility Index. <15 = low vol, 15-25 = normal, >25 = high vol.")
    with m2:
        v = live_macro.get("us10y_yield")
        st.metric("10Y Yield", f"{v:.2f}%" if v else "—",
                  help="US 10-Year Treasury yield. Rising = tighter conditions, bearish for equities.")
    with m3:
        v = live_macro.get("dxy")
        st.metric("DXY", f"{v:.1f}" if v else "—",
                  help="US Dollar Index. Strong dollar = headwind for risk assets.")
    with m4:
        v = live_macro.get("gold")
        st.metric("Gold", f"${v:,.0f}" if v else "—",
                  help="Gold spot price. Rising gold = risk-off sentiment.")

    # Row 2: Breadth & Valuation
    try:
        cfg = _load_cfg()
        router = get_router(cfg)
        breadth_df = router.read_analytics("SELECT * FROM market_breadth ORDER BY date DESC LIMIT 1")
        row = breadth_df.iloc[0] if not breadth_df.empty else None
    except Exception:
        row = None

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        fg = row.get("fear_greed_index") if row is not None else None
        if fg is not None and pd.notna(fg):
            fg = int(fg)
            fg_label = "🔥 Extreme Greed" if fg > 75 else ("😀 Greed" if fg > 55 else ("😐 Neutral" if fg > 45 else ("😨 Fear" if fg > 25 else "🥶 Extreme Fear")))
            st.metric("Fear & Greed", f"{fg}", help="CNN Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed).")
            st.caption(fg_label)
        else:
            st.metric("Fear & Greed", "N/A", help="CNN Fear & Greed Index. Run pipeline to populate.")
    with b2:
        trin = row.get("trin") if row is not None else None
        if trin is not None and pd.notna(trin):
            trin = float(trin)
            trin_label = "🟢 Buying Pressure" if trin < 0.8 else ("🔴 Selling Pressure" if trin > 1.2 else "😐 Neutral")
            st.metric("TRIN", f"{trin:.2f}", help="Arms Index. <1.0 = buying pressure, >1.0 = selling pressure.")
            st.caption(trin_label)
        else:
            st.metric("TRIN", "N/A")
    with b3:
        buffett = row.get("buffett_indicator") if row is not None else None
        if buffett is not None and pd.notna(buffett):
            buffett = float(buffett)
            buffett_label = "🔴 Strongly OV" if buffett > 150 else ("🟡 Overvalued" if buffett > 100 else "🟢 Fair Value")
            st.metric("Buffett Indicator", f"{buffett:.0f}%", help="Market Cap / GDP. >100% = overvalued.")
            st.caption(buffett_label)
        else:
            st.metric("Buffett Indicator", "N/A")
    with b4:
        cape = row.get("sp500_cape") if row is not None else None
        if cape is not None and pd.notna(cape):
            cape = float(cape)
            cape_label = "🔴 Overvalued" if cape > 30 else ("🟡 Elevated" if cape > 20 else "🟢 Fair Value")
            st.metric("Shiller CAPE", f"{cape:.1f}", help="Cyclically Adjusted P/E. Historical avg ~17.")
            st.caption(cape_label)
        else:
            st.metric("Shiller CAPE", "N/A")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: SYSTEM HEALTH (compact, non-collapsible)
    # ══════════════════════════════════════════════════════════════════
    st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.85rem;margin-top:16px;margin-bottom:4px;">SYSTEM HEALTH</p>', unsafe_allow_html=True)
    h1, h2, h3, h4, h5 = st.columns(5)

    with h1:
        try:
            cfg = _load_cfg()
            router = get_router(cfg)
            if router.using_postgres:
                st.markdown(metric_card("Database", "🟢 PostgreSQL", "green"), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Database", "🟡 SQLite", "yellow"), unsafe_allow_html=True)
        except Exception:
            st.markdown(metric_card("Database", "🔴 Offline", "red"), unsafe_allow_html=True)

    with h2:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                st.markdown(metric_card("LLM (Ollama)", "🟢 Online", "green"), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("LLM (Ollama)", "🔴 Offline", "red"), unsafe_allow_html=True)
        except Exception:
            st.markdown(metric_card("LLM (Ollama)", "🔴 Offline", "red"), unsafe_allow_html=True)

    with h3:
        try:
            import subprocess
            result = subprocess.run(["pgrep", "-af", "src.launcher"], capture_output=True, text=True, timeout=3)
            if result.stdout.strip():
                st.markdown(metric_card("Scheduler", "🟢 Running", "green"), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Scheduler", "🔴 Stopped", "red"), unsafe_allow_html=True)
        except Exception:
            st.markdown(metric_card("Scheduler", "⚪ Unknown", "white"), unsafe_allow_html=True)

    with h4:
        try:
            import json as _json
            with open("./data/streamer_state.json") as _sf:
                _ss = _json.load(_sf)
            stocks_ok = _ss.get("is_stocks_alive", False)
            opts_ok = _ss.get("is_options_alive", False)
            if stocks_ok and opts_ok:
                st.markdown(metric_card("Streamer", "🟢 Both OK", "green"), unsafe_allow_html=True)
            elif stocks_ok:
                st.markdown(metric_card("Streamer", "🟡 Stocks Only", "yellow"), unsafe_allow_html=True)
            elif opts_ok:
                st.markdown(metric_card("Streamer", "🟡 Options Only", "yellow"), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Streamer", "🔴 Disconnected", "red"), unsafe_allow_html=True)
        except Exception:
            st.markdown(metric_card("Streamer", "⚪ Inactive", "white"), unsafe_allow_html=True)

    with h5:
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY").fast_info
            price = float(getattr(spy, "last_price", 0) or 0)
            if price > 0:
                st.markdown(metric_card("Data Feed", f"🟢 ${price:,.2f}", "green"), unsafe_allow_html=True)
            else:
                st.markdown(metric_card("Data Feed", "🔴 No Data", "red"), unsafe_allow_html=True)
        except Exception:
            st.markdown(metric_card("Data Feed", "🔴 Offline", "red"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4: TOP SHAP DRIVERS (compact, always visible)
    # ══════════════════════════════════════════════════════════════════
    shap_drivers = prediction.get("shap_drivers", []) if prediction else []
    if shap_drivers:
        st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.85rem;margin-top:16px;margin-bottom:4px;">TOP PREDICTION DRIVERS</p>', unsafe_allow_html=True)
        driver_df = pd.DataFrame(shap_drivers[:8])
        fig_shap = go.Figure()
        bar_colors = [c["green"] if v > 0 else c["red"] for v in driver_df["shap_value"]]
        fig_shap.add_trace(go.Bar(
            y=driver_df["feature"], x=driver_df["shap_value"],
            orientation="h", marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in driver_df["shap_value"]],
            textposition="outside",
            textfont=dict(color=c["text"], size=10),
        ))
        _bg = "rgba(0,0,0,0)" if is_dark() else c["surface"]
        fig_shap.update_layout(
            height=180, margin=dict(l=10, r=10, t=5, b=5),
            xaxis_title="SHAP Value", yaxis=dict(autorange="reversed"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=_bg,
            font=dict(color=c["text"], size=10),
            xaxis=dict(gridcolor=c.get("grid", "#2A2E39"), zerolinecolor=c.get("zeroline", "#363A45")),
        )
        st.plotly_chart(fig_shap, use_container_width=True, key="overview_shap")
        st.caption("For full prediction details, SHAP analysis, and historical charts, visit the **SPY Predictor** page.")
