# Kiro Prompt: Comprehensive UI/UX Restructuring

**Repository:** `damerav/stockanalysis` (latest commit: 6661060)
**Target File:** `src/dashboard/app.py` (3,439 lines) plus new files

---

## Background & Objective

The platform currently has two navigation groups (`Markets`, `Operations`) containing 12 pages. Through organic growth, several problems have emerged:

1. The `SPY Predictor` page is the default landing page but contains too many unrelated sections (macro metrics, valuation context, SHAP drivers, prediction history, options flow, intraday microstructure) all stacked vertically with critical data hidden in collapsible expanders.
2. System monitoring is duplicated across three places: `monitoring.py`, `page_grafana`, and the `Admin` console's System Status tab.
3. The `Admin` page is a catch-all with 6 tabs mixing unrelated concerns (user management, database explorer, ad-hoc pipeline actions, logs, configuration).
4. The `Quant Agent` page (admin-only) is grouped with general market pages.
5. The `What-If Analysis` page duplicates the `Tune & Backtest` page's conceptual space.

**Goal:** Restructure into three logical navigation groups (`Dashboards`, `Analysis`, `Administration`), create a new `Market Overview` default page, and split the `Admin` page into two focused pages.

---

## Constraints & Patterns

- All new files must follow the existing pattern: import from `src.dashboard.theme`, use `get_colors()`, `page_header()`, `metric_card()`, `get_plotly_layout()`.
- All database access must use the `DbRouter` via `get_router(config)` from `src.data.db_router`.
- All new pages must be importable as `from src.dashboard.<module> import page_<name>`.
- Do NOT use `st.set_page_config()` in any sub-page file — it is only called once in `app.py`.
- The `_load_config()` helper in `app.py` (line 1370) must be used for all config loading.
- The `load_spy_state()` function (line 224) must be used for reading the prediction state.
- The `_fetch_live_macro()` cached function (line 61) must be used for macro data.

---

## Part 1: Create `src/dashboard/market_overview_app.py` (New Default Page)

Create this file from scratch. It is the new default landing page and must provide a complete, at-a-glance market summary without requiring any scrolling or clicking to expand sections.

```python
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
    macro = state.get("macro", {})

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
        if fg is not None:
            fg_label = "🔥 Extreme Greed" if fg > 75 else ("😀 Greed" if fg > 55 else ("😐 Neutral" if fg > 45 else ("😨 Fear" if fg > 25 else "🥶 Extreme Fear")))
            st.metric("Fear & Greed", f"{fg:.0f}", help="CNN Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed).")
            st.caption(fg_label)
        else:
            st.metric("Fear & Greed", "N/A", help="CNN Fear & Greed Index. Run pipeline to populate.")
    with b2:
        trin = row.get("trin") if row is not None else None
        if trin is not None:
            trin_label = "🟢 Buying Pressure" if trin < 0.8 else ("🔴 Selling Pressure" if trin > 1.2 else "😐 Neutral")
            st.metric("TRIN", f"{trin:.2f}", help="Arms Index. <1.0 = buying pressure, >1.0 = selling pressure.")
            st.caption(trin_label)
        else:
            st.metric("TRIN", "N/A")
    with b3:
        buffett = row.get("buffett_indicator") if row is not None else None
        if buffett is not None:
            buffett_label = "🔴 Strongly OV" if buffett > 150 else ("🟡 Overvalued" if buffett > 100 else "🟢 Fair Value")
            st.metric("Buffett Indicator", f"{buffett:.0f}%", help="Market Cap / GDP. >100% = overvalued.")
            st.caption(buffett_label)
        else:
            st.metric("Buffett Indicator", "N/A")
    with b4:
        cape = row.get("sp500_cape") if row is not None else None
        if cape is not None:
            cape_label = "🔴 Overvalued" if cape > 30 else ("🟡 Elevated" if cape > 20 else "🟢 Fair Value")
            st.metric("Shiller CAPE", f"{cape:.1f}", help="Cyclically Adjusted P/E. Historical avg ~17.")
            st.caption(cape_label)
        else:
            st.metric("Shiller CAPE", "N/A")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: SYSTEM HEALTH (compact, non-collapsible)
    # ══════════════════════════════════════════════════════════════════
    st.markdown(f'<p style="color:{c["text_secondary"]};font-weight:600;font-size:0.85rem;margin-top:16px;margin-bottom:4px;">SYSTEM HEALTH</p>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)

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
            xaxis=dict(gridcolor=c["grid"], zerolinecolor=c["zeroline"]),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("For full prediction details, SHAP analysis, and historical charts, visit the **SPY Predictor** page.")
```

---

## Part 2: Create `src/dashboard/scenario_analysis_app.py`

This is a thin wrapper that renames the existing `What-If Analysis` page and gives it a proper header.

```python
"""Scenario Analysis — consolidated What-If simulation engine.

Wraps the existing What-If engine for ES Strategy and SPY Predictor
under a cleaner, more descriptive page name.
"""

import streamlit as st
from src.dashboard.theme import page_header


def page_scenario_analysis():
    """Renders the Scenario Analysis page."""
    st.markdown(page_header("🔬 Scenario Analysis"), unsafe_allow_html=True)
    st.caption(
        "Simulate how changes to market conditions, model features, or strategy rules "
        "would affect outcomes. Use the ES Strategy tab for futures simulations and "
        "the SPY Predictor tab for prediction sensitivity analysis."
    )

    # Import and call the existing What-If tab functions
    import numpy as np
    import yaml
    import plotly.graph_objects as go
    from src.whatif.engine import WhatIfEngine
    from src.whatif.presets import STRESS_SCENARIOS
    from src.data.features import get_feature_columns

    @st.cache_resource
    def _get_engine():
        try:
            with open("config.yaml") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            config = {}
        return WhatIfEngine(config)

    engine = _get_engine()
    tab_es, tab_spy = st.tabs(["📈 ES Strategy", "🔮 SPY Predictor"])

    # Import the tab renderers from app.py
    from src.dashboard.app import _whatif_es_tab, _whatif_spy_tab
    with tab_es:
        _whatif_es_tab(engine)
    with tab_spy:
        _whatif_spy_tab(engine)
```

---

## Part 3: Create `src/dashboard/data_management_app.py`

This page consolidates all data-related admin tasks from the old `Admin` page.

```python
"""Data Management — database explorer and ad-hoc pipeline actions.

Extracted from the Admin console to provide a focused, single-purpose
page for all data-related administrative tasks.
"""

import streamlit as st
from src.dashboard.theme import page_header


def page_data_management():
    """Renders the Data Management page."""
    st.markdown(page_header("🗃️ Data Management"), unsafe_allow_html=True)
    st.caption(
        "Explore the database, run ad-hoc pipeline steps, and manage data operations. "
        "For system health and user management, see the System Management page."
    )

    tab_actions, tab_db = st.tabs(["▶️ Pipeline Actions", "🗃️ Database Explorer"])

    with tab_actions:
        from src.dashboard.app import _admin_actions_tab
        _admin_actions_tab()

    with tab_db:
        from src.dashboard.app import _admin_db_tab
        _admin_db_tab()
```

---

## Part 4: Create `src/dashboard/system_management_app.py`

This page consolidates all system-related admin tasks from the old `Admin` page.

```python
"""System Management — users, logs, configuration, and system health.

Extracted from the Admin console to provide a focused, single-purpose
page for all system-related administrative tasks.
"""

import streamlit as st
from src.dashboard.theme import page_header


def page_system_management():
    """Renders the System Management page."""
    st.markdown(page_header("⚙️ System Management"), unsafe_allow_html=True)
    st.caption(
        "Manage users, view system logs, review configuration, and check system health. "
        "For pipeline actions and database operations, see the Data Management page."
    )

    tab_status, tab_users, tab_config, tab_logs = st.tabs([
        "ℹ️ System Status", "👤 Users", "📝 Configuration", "📜 Logs"
    ])

    with tab_status:
        from src.dashboard.app import _admin_status_tab
        _admin_status_tab()

    with tab_users:
        from src.dashboard.app import _admin_users_tab
        _admin_users_tab()

    with tab_config:
        from src.dashboard.app import _admin_config_tab
        _admin_config_tab()

    with tab_logs:
        from src.dashboard.app import _admin_logs_tab
        _admin_logs_tab()
```

---

## Part 5: Update `src/dashboard/app.py`

Make the following targeted changes to `app.py`:

**5.1. Add new imports at the top (after existing imports, around line 55):**

```python
from src.dashboard.market_overview_app import page_market_overview
from src.dashboard.scenario_analysis_app import page_scenario_analysis
from src.dashboard.data_management_app import page_data_management
from src.dashboard.system_management_app import page_system_management
```

**5.2. Replace the `_pages` dictionary (around line 3362) with the new structure:**

```python
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
```

**5.3. Remove the old `page_admin` function and its helper tabs:**

Delete the following functions from `app.py` as they are now in the new dedicated files:
- `page_admin()` (around line 1393)
- `_admin_status_tab()` (around line 1442)
- `_admin_actions_tab()` (around line 1827)
- `_admin_users_tab()` (around line 1684)
- `_admin_db_tab()` (around line 2437)
- `_admin_config_tab()` (around line 2574)
- `_admin_logs_tab()` (around line 2601)

> **Important:** Before deleting these functions, confirm that the new `data_management_app.py` and `system_management_app.py` files have been created and are importing them correctly. If the import approach causes circular dependency issues, copy the function bodies directly into the new files instead of importing them.

**5.4. Remove the old `page_monitoring` import and function:**

- Delete the import: `from src.dashboard.monitoring import page_monitoring` (around line 36)
- The `monitoring.py` file itself can remain as-is for now but is no longer used.

**5.5. Slim down the `page_spy` function:**

In the `page_spy()` function, the following sections are now redundant because they appear on the new `Market Overview` page. Remove them to reduce clutter:

- The 4-column macro row (`r1, r2, r3, r4`) showing Regime, Model, Conf. Set, VIX (lines ~349–381).
- The 4-column macro row (`m1, m2, m3, m4`) showing 10Y Yield, DXY, Gold, Crude (lines ~382–398).
- The `Market Valuation Context` expander (lines ~477–523).

The `page_spy` page should now start directly with the hero prediction banner, then go straight to the SHAP drivers chart, followed by the Prediction History chart and Key Indicators panel. The Earnings/Fed/Options row and the Intraday Microstructure expander can remain as they provide deep-dive detail not shown on the overview page.

**5.6. Update the `page_grafana` function title:**

Find the `page_grafana` function and add a page header and caption at the very top:

```python
def page_grafana():
    st.markdown(page_header("📡 System Monitoring"), unsafe_allow_html=True)
    st.caption(
        "Live system monitoring powered by Grafana. Dashboards cover SPY Predictor performance, "
        "ES Strategy P&L, system health, and the data pipeline status."
    )
    # ... rest of the existing function body unchanged ...
```

---

## Part 6: Validation Checklist

After implementation, verify the following:

| Check | Expected Result |
| :--- | :--- |
| Navigate to `/` | `Market Overview` page loads as default with prediction signal, indicator gauges, and system health |
| Navigate to `SPY Predictor` | Page shows hero banner, SHAP chart, prediction history, and key indicators — without the macro rows or valuation context |
| Navigate to `Scenario Analysis` | Page shows the same ES and SPY What-If tabs as before |
| Navigate to `Data Management` | Page shows `Pipeline Actions` and `Database Explorer` tabs |
| Navigate to `System Management` | Page shows `System Status`, `Users`, `Configuration`, and `Logs` tabs |
| Navigate to `System Monitoring` | Grafana iframe loads with new title and caption |
| Navigate to `Quant Agent` | Admin-only gate still works correctly |
| No broken imports | `streamlit run src/dashboard/app.py` starts without `ImportError` |
| Theme toggle works | Dark/light mode applies correctly to the new `Market Overview` page |
