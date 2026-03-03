"""Native Plotly monitoring dashboards — Grafana-quality, zero dependencies.

Replicates all 5 Grafana dashboards:
  1. SPY Predictor  — price, MAs, Bollinger, MACD, RSI, VIX
  2. ES Strategy    — P&L, positions, Keltner, RSI
  3. System Health  — service status, DB size, model info
  4. Confidence API — latency, allow/block, audit log
  5. Pipeline       — process status, data inventory
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

logger = logging.getLogger(__name__)

# Database router support (PostgreSQL primary, SQLite fallback)
try:
    from src.data.db_router import get_router, ANALYTICS_TABLES
    _HAS_ROUTER = True
except ImportError:
    _HAS_ROUTER = False
    ANALYTICS_TABLES = set()

DATA_DIR = "./data"
LOGS_DIR = "./logs"
MODELS_DIR = "./models"
PIDS_DIR = "./.pids"

# ── TradingView-inspired dark theme for Plotly ────────────────────────
# Now theme-aware — imports from shared theme module
from src.dashboard.theme import (
    get_colors as _get_theme_colors,
    get_plotly_layout as _get_theme_layout,
    get_title_font as _get_theme_title_font,
    metric_card as _theme_metric_card,
    badge_html as _theme_badge,
    is_dark,
)

def _refresh_theme():
    """Refresh theme-dependent module globals."""
    global DARK_LAYOUT, TITLE_FONT, COLORS
    COLORS = _get_theme_colors()
    DARK_LAYOUT = _get_theme_layout()
    TITLE_FONT = _get_theme_title_font()

# Initialize with defaults
COLORS = _get_theme_colors()
DARK_LAYOUT = _get_theme_layout()
TITLE_FONT = _get_theme_title_font()


# ── Helper: status badge HTML ─────────────────────────────────────────
def _badge(label: str, online: bool) -> str:
    _refresh_theme()
    return _theme_badge(label, online)


def _metric_card(label: str, value: str, color: str = "white", sub: str = "") -> str:
    _refresh_theme()
    return _theme_metric_card(label, value, color, sub)


def _gauge_chart(value: float, title: str, min_val=0, max_val=100,
                 thresholds=None, suffix="%", height=140):
    """Create a Plotly gauge chart mimicking Grafana gauges."""
    _refresh_theme()
    if thresholds is None:
        thresholds = [(0, "red"), (40, "yellow"), (60, "green")]

    steps = []
    for i, (thresh, color) in enumerate(thresholds):
        upper = thresholds[i + 1][0] if i + 1 < len(thresholds) else max_val
        steps.append(dict(range=[thresh, upper], color=COLORS.get(color, color)))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=14, color=COLORS["text"])),
        number=dict(suffix=suffix, font=dict(size=24)),
        gauge=dict(
            axis=dict(range=[min_val, max_val], tickcolor="#666"),
            bar=dict(color=COLORS["blue"]),
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["border"],
            steps=steps,
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"]),
        height=height, margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# ── DB query helpers ──────────────────────────────────────────────────
def _get_db():
    db_path = os.path.join(DATA_DIR, "spy.db")
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _query_df(sql: str, params=()) -> pd.DataFrame:
    """Query helper — routes through DbRouter (PostgreSQL → SQLite fallback)."""
    if _HAS_ROUTER:
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(DATA_DIR), "config.yaml")
            if not os.path.exists(config_path):
                config_path = "config.yaml"
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            except Exception:
                config = None
            router = get_router(config)
            return router.read_analytics(sql, params if params else None)
        except Exception:
            pass  # Fall through to SQLite

    conn = _get_db()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _load_es_state() -> dict:
    try:
        with open(os.path.join(DATA_DIR, "es_state.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def _check_service(url: str, timeout: int = 3) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _check_pid(name: str) -> bool:
    pid_file = os.path.join(PIDS_DIR, f"{name}.pid")
    if not os.path.exists(pid_file):
        return False
    try:
        pid = int(open(pid_file).read().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, PermissionError, OSError):
        return False


# ══════════════════════════════════════════════════════════════════════
# TAB 1: SPY PREDICTOR
# ══════════════════════════════════════════════════════════════════════

def tab_spy_predictor(days: int = 90):
    """SPY Predictor monitoring — mirrors Grafana spy-predictor dashboard."""

    # ── Row 1: Stat cards ──
    # Use spy_state.json for real-time metrics (same source as SPY Predictor page)
    # to avoid inconsistencies between pages. DB is used for historical charts below.
    conn = _get_db()
    close_val = rsi_val = atr_val = vix_val = 0.0
    direction_str = "—"
    confidence_val = 0.0

    # Load real-time state (same source as SPY Predictor page)
    spy_state = {}
    try:
        with open(os.path.join(DATA_DIR, "spy_state.json")) as f:
            spy_state = json.load(f)
    except Exception:
        pass

    pred = spy_state.get("prediction", {})
    indicators = spy_state.get("indicators", {})

    # Real-time values from state file
    direction_str = pred.get("direction", "—")
    confidence_val = pred.get("confidence", 0)
    rsi_val = indicators.get("rsi_14", 0) or 0
    vix_val = indicators.get("vix", 0) or 0

    # SPY close from DB (state file doesn't carry it)
    if conn:
        row = conn.execute("SELECT close FROM prices ORDER BY date DESC LIMIT 1").fetchone()
        if row:
            close_val = float(row[0] or 0)

        # ATR from DB (state file has it but let's be consistent)
        atr_val = indicators.get("atr_14", 0) or 0

        # Fall back to DB if state file has no data
        if not direction_str or direction_str == "—":
            row = conn.execute("SELECT direction, confidence FROM predictions ORDER BY date DESC LIMIT 1").fetchone()
            if row:
                direction_str = str(row[0] or "—")
                confidence_val = float(row[1] or 0)

        if not rsi_val:
            row = conn.execute("SELECT rsi_14, atr_14 FROM technicals ORDER BY date DESC LIMIT 1").fetchone()
            if row:
                rsi_val = float(row[0] or 0)
                atr_val = float(row[1] or 0)

        if not vix_val:
            row = conn.execute("SELECT vix FROM macro ORDER BY date DESC LIMIT 1").fetchone()
            if row:
                vix_val = float(row[0] or 0)

        conn.close()

    dir_color = "green" if "BULL" in direction_str.upper() else "red" if "BEAR" in direction_str.upper() else "yellow"
    vix_color = "green" if vix_val < 20 else "yellow" if vix_val < 30 else "red"

    cards = [
        ("SPY Last Close", f"${close_val:,.2f}", "white"),
        ("Prediction", direction_str, dir_color),
        ("Confidence", f"{confidence_val:.1f}%", dir_color),
        ("VIX", f"{vix_val:.1f}", vix_color),
        ("RSI (14)", f"{rsi_val:.1f}", "blue"),
        ("ATR", f"{atr_val:.2f}", "cyan"),
    ]
    for row_start in range(0, len(cards), 3):
        row_cards = cards[row_start:row_start + 3]
        cols = st.columns(len(row_cards))
        for col, (label, val, color) in zip(cols, row_cards):
            col.markdown(_metric_card(label, val, color), unsafe_allow_html=True)

    # Staleness indicator (same logic as SPY Predictor page)
    updated_at = spy_state.get("updated_at", "")
    _stale_color = "#64748B"
    _stale_label = ""
    if updated_at:
        try:
            _upd_dt = (datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                       if "T" in updated_at
                       else datetime.strptime(updated_at[:19], "%Y-%m-%d %H:%M:%S"))
            _age_min = (datetime.now() - _upd_dt).total_seconds() / 60
            if _age_min > 60:
                _stale_color = "#DC3545"
                _stale_label = " ⚠️ STALE"
            elif _age_min > 30:
                _stale_color = "#FFC107"
                _stale_label = " ⏳"
        except Exception:
            pass
    st.markdown(
        f'<p style="color:{_stale_color}; font-size:0.75rem; text-align:right; margin-top:2px; margin-bottom:0;">'
        f'Updated: {updated_at or "N/A"}{_stale_label}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Price vs Moving Averages | Bollinger Bands ──
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    prices = _query_df("SELECT date, close FROM prices WHERE date >= ? ORDER BY date", (cutoff,))
    techs = _query_df(
        "SELECT date, sma_20, sma_50, bb_upper, bb_lower, macd, macd_signal, rsi_14 "
        "FROM technicals WHERE date >= ? ORDER BY date", (cutoff,)
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        if not prices.empty:
            fig.add_trace(go.Scatter(x=prices["date"], y=prices["close"],
                                     name="SPY Close", line=dict(color=COLORS["white"], width=2)))
        if not techs.empty:
            fig.add_trace(go.Scatter(x=techs["date"], y=techs["sma_20"],
                                     name="SMA 20", line=dict(color=COLORS["yellow"], width=1.5)))
            fig.add_trace(go.Scatter(x=techs["date"], y=techs["sma_50"],
                                     name="SMA 50", line=dict(color=COLORS["orange"], width=1.5)))
        fig.update_layout(**DARK_LAYOUT, title=dict(text="SPY Price vs Moving Averages", font=TITLE_FONT),
                          yaxis_title="USD", height=250)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        if not prices.empty:
            fig.add_trace(go.Scatter(x=prices["date"], y=prices["close"],
                                     name="SPY Close", line=dict(color=COLORS["white"], width=2)))
        if not techs.empty:
            fig.add_trace(go.Scatter(x=techs["date"], y=techs["bb_upper"],
                                     name="BB Upper", line=dict(color=COLORS["red"], width=1, dash="dash")))
            fig.add_trace(go.Scatter(x=techs["date"], y=techs["bb_lower"],
                                     name="BB Lower", line=dict(color=COLORS["blue"], width=1, dash="dash"),
                                     fill="tonexty", fillcolor="rgba(87,148,242,0.08)"))
        fig.update_layout(**DARK_LAYOUT, title=dict(text="Bollinger Bands", font=TITLE_FONT),
                          yaxis_title="USD", height=250)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: MACD | RSI History ──
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure()
        if not techs.empty:
            fig.add_trace(go.Scatter(x=techs["date"], y=techs["macd"],
                                     name="MACD", line=dict(color=COLORS["cyan"], width=2)))
            fig.add_trace(go.Scatter(x=techs["date"], y=techs["macd_signal"],
                                     name="Signal", line=dict(color=COLORS["orange"], width=1.5)))
            hist = techs["macd"].astype(float) - techs["macd_signal"].astype(float)
            colors_hist = [COLORS["green"] if v >= 0 else COLORS["red"] for v in hist]
            fig.add_trace(go.Bar(x=techs["date"], y=hist, name="Histogram",
                                 marker_color=colors_hist, opacity=0.5))
        fig.update_layout(**DARK_LAYOUT, title=dict(text="MACD", font=TITLE_FONT), height=220)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        if not techs.empty:
            fig.add_trace(go.Scatter(x=techs["date"], y=techs["rsi_14"],
                                     name="RSI", line=dict(color=COLORS["purple"], width=2)))
            fig.add_hrect(y0=70, y1=100, fillcolor="rgba(242,73,92,0.1)", line_width=0)
            fig.add_hrect(y0=0, y1=30, fillcolor="rgba(115,191,105,0.1)", line_width=0)
            fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], opacity=0.5)
        rsi_layout = {**DARK_LAYOUT, "yaxis": dict(range=[0, 100], gridcolor=COLORS["border"], showgrid=True)}
        fig.update_layout(**rsi_layout, title=dict(text="RSI (14)", font=TITLE_FONT), height=220)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: VIX History | Data Inventory ──
    c1, c2 = st.columns(2)

    with c1:
        vix_df = _query_df("SELECT date, vix FROM macro WHERE date >= ? ORDER BY date", (cutoff,))
        fig = go.Figure()
        if not vix_df.empty:
            fig.add_trace(go.Scatter(x=vix_df["date"], y=vix_df["vix"],
                                     name="VIX", line=dict(color=COLORS["yellow"], width=2),
                                     fill="tozeroy", fillcolor="rgba(255,152,48,0.1)"))
            fig.add_hline(y=20, line_dash="dash", line_color=COLORS["yellow"], opacity=0.4)
            fig.add_hline(y=30, line_dash="dash", line_color=COLORS["red"], opacity=0.4)
        fig.update_layout(**DARK_LAYOUT, title=dict(text="VIX History", font=TITLE_FONT), height=220)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        conn = _get_db()
        if conn:
            tables = ["prices", "technicals", "news", "daily_sentiment", "macro",
                       "predictions", "intraday_bars", "options_chain",
                       "options_analytics", "intraday_features", "performance",
                       "earnings_calendar", "fed_communications"]
            rows = []
            # Route through DbRouter for all tables
            db_router = None
            if _HAS_ROUTER:
                try:
                    import yaml
                    with open("config.yaml") as f:
                        config = yaml.safe_load(f)
                    db_router = get_router(config)
                except Exception:
                    pass

            for t in tables:
                try:
                    if db_router:
                        cnt_df = db_router.read_analytics(f"SELECT COUNT(*) as cnt FROM {t}")
                        count = int(cnt_df.iloc[0]["cnt"]) if not cnt_df.empty else 0
                    else:
                        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    rows.append({"Table": t, "Rows": count})
                except Exception:
                    rows.append({"Table": t, "Rows": 0})
            conn.close()
            df = pd.DataFrame(rows)
            fig = go.Figure(go.Bar(
                x=df["Rows"], y=df["Table"], orientation="h",
                marker=dict(color=COLORS["blue"], line=dict(width=0)),
            ))
            inv_layout = {**DARK_LAYOUT, "yaxis": dict(autorange="reversed", gridcolor=COLORS["border"], showgrid=True)}
            fig.update_layout(**inv_layout, title=dict(text="Data Inventory", font=TITLE_FONT), height=220,
                              xaxis_title="Row Count")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2: ES STRATEGY
# ══════════════════════════════════════════════════════════════════════

def tab_es_strategy():
    """ES Futures Strategy monitoring — mirrors Grafana es-strategy dashboard."""
    state = _load_es_state()

    daily_pnl = float(state.get("daily_pnl", 0))
    total_pnl = float(state.get("total_pnl", 0))
    open_lots = int(state.get("open_lots", 0))
    trades_today = int(state.get("trades_today", 0))
    win_rate = float(state.get("win_rate", 0))
    sharpe = float(state.get("sharpe_ratio", 0))
    max_dd = float(state.get("max_drawdown", 0))
    regime_str = state.get("vol_regime", "Med")

    # ── Row 1: Stat cards ──
    pnl_color = "green" if daily_pnl >= 0 else "red"
    total_color = "green" if total_pnl >= 0 else "red"
    regime_color = {"Low": "green", "Med": "yellow", "High": "red"}.get(regime_str, "yellow")
    sharpe_color = "green" if sharpe >= 1.0 else "yellow" if sharpe >= 0.5 else "red"

    card_data = [
        ("Daily P&L", f"${daily_pnl:+,.0f}", pnl_color),
        ("Total P&L", f"${total_pnl:+,.0f}", total_color),
        ("Open Lots", str(open_lots), "blue"),
        ("Trades Today", str(trades_today), "cyan"),
        ("Win Rate", f"{win_rate*100:.0f}%", "green" if win_rate > 0.55 else "yellow"),
        ("Vol Regime", regime_str.upper(), regime_color),
        ("Sharpe", f"{sharpe:.2f}", sharpe_color),
    ]
    for row_start in range(0, len(card_data), 4):
        row_cards = card_data[row_start:row_start + 4]
        cols = st.columns(len(row_cards))
        for col, (label, val, color) in zip(cols, row_cards):
            col.markdown(_metric_card(label, val, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Gauges ──
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = _gauge_chart(win_rate * 100, "Win Rate", 0, 100,
                           [(0, "red"), (40, "yellow"), (55, "green")])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = _gauge_chart(sharpe, "Sharpe Ratio", -1, 3,
                           [(-1, "red"), (0.5, "yellow"), (1.0, "green")], suffix="")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = _gauge_chart(abs(max_dd), "Max Drawdown", 0, 3000,
                           [(0, "green"), (500, "yellow"), (1000, "red")], suffix=" USD")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Keltner Channel | P&L placeholder ──
    c1, c2 = st.columns(2)

    with c1:
        price = float(state.get("current_price", 0))
        kc_upper = float(state.get("kc_upper", 0))
        kc_mid = float(state.get("kc_mid", 0))
        kc_lower = float(state.get("kc_lower", 0))
        vwap = float(state.get("vwap", 0))

        fig = go.Figure()
        # Show current values as horizontal reference lines
        if kc_upper > 0:
            fig.add_hline(y=kc_upper, line_dash="dash", line_color=COLORS["red"],
                          annotation_text=f"KC Upper: {kc_upper:.0f}")
            fig.add_hline(y=kc_mid, line_dash="dot", line_color=COLORS["yellow"],
                          annotation_text=f"KC Mid: {kc_mid:.0f}")
            fig.add_hline(y=kc_lower, line_dash="dash", line_color=COLORS["green"],
                          annotation_text=f"KC Lower: {kc_lower:.0f}")
        if vwap > 0:
            fig.add_hline(y=vwap, line_dash="dashdot", line_color=COLORS["purple"],
                          annotation_text=f"VWAP: {vwap:.0f}")
        if price > 0:
            fig.add_hline(y=price, line_color=COLORS["white"], line_width=2,
                          annotation_text=f"ES: {price:.0f}")
            y_min = min(kc_lower, price) - 10 if kc_lower > 0 else price - 30
            y_max = max(kc_upper, price) + 10 if kc_upper > 0 else price + 30
            fig.update_yaxes(range=[y_min, y_max])

        fig.update_layout(**DARK_LAYOUT, title=dict(text="ES Price vs Keltner Channel", font=TITLE_FONT),
                          yaxis_title="USD", height=260)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Position details table
        pos = state.get("position", {})
        if isinstance(pos, dict) and pos:
            pos_df = pd.DataFrame([{
                "Lots": pos.get("lots", 0),
                "Entry Price": f"${float(pos.get('entry_price', 0)):,.2f}",
                "Unrealized P&L": f"${float(pos.get('unrealized_pnl', 0)):+,.2f}",
                "Status": pos.get("status", "FLAT"),
            }])
            st.markdown("**Position Details**")
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open position")

        # Signal history
        signals = state.get("signals", [])
        if signals and isinstance(signals, list):
            st.markdown(f"**Recent Signals** ({len(signals)} total)")
            recent = signals[-10:] if len(signals) > 10 else signals
            for sig in reversed(recent):
                if isinstance(sig, dict):
                    st.caption(f"{sig.get('time', '—')} | {sig.get('action', '—')} | {sig.get('reason', '—')}")


# ══════════════════════════════════════════════════════════════════════
# TAB 3: SYSTEM HEALTH
# ══════════════════════════════════════════════════════════════════════

def tab_system_health():
    """System Health monitoring — mirrors Grafana system-health dashboard."""

    # ── Row 1: Service status badges ──
    db_online = os.path.exists(os.path.join(DATA_DIR, "spy.db"))
    ollama_online = _check_service("http://localhost:11434/api/tags")
    dashboard_online = _check_service("http://localhost:8501")
    api_online = _check_service("http://localhost:8100/health")

    # Model check
    model_loaded = False
    model_count = 0
    model_size_kb = 0
    if os.path.exists(MODELS_DIR):
        model_files = sorted([f for f in os.listdir(MODELS_DIR) if f.endswith(".json")], reverse=True)
        model_count = len(model_files)
        model_loaded = model_count > 0
        if model_files:
            model_size_kb = os.path.getsize(os.path.join(MODELS_DIR, model_files[0])) / 1024

    # Target LLM
    target_loaded = False
    ollama_model_count = 0
    if ollama_online:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            models = resp.json().get("models", [])
            ollama_model_count = len(models)
            import yaml
            with open("config.yaml") as f:
                cfg = yaml.safe_load(f) or {}
            target = cfg.get("llm", {}).get("model", "deepseek-r1:70b")
            target_loaded = any(target in m.get("name", "") for m in models)
        except Exception:
            pass

    badges = [
        ("Database", db_online),
        ("Ollama LLM", ollama_online),
        ("XGBoost Model", model_loaded),
        ("Confidence API", api_online),
        ("Dashboard", dashboard_online),
        ("Target LLM", target_loaded),
    ]
    for row_start in range(0, len(badges), 3):
        row_badges = badges[row_start:row_start + 3]
        cols = st.columns(len(row_badges))
        for col, (label, status) in zip(cols, row_badges):
            col.markdown(_badge(label, status), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: DB size, API uptime, Model info ──
    c1, c2, c3 = st.columns(3)

    with c1:
        db_path = os.path.join(DATA_DIR, "spy.db")
        db_size = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0
        st.markdown(_metric_card("Database Size", f"{db_size:.1f} MB", "blue"), unsafe_allow_html=True)

    with c2:
        uptime_str = "—"
        if api_online:
            try:
                resp = requests.get("http://localhost:8100/health", timeout=3)
                data = resp.json()
                uptime_s = float(data.get("uptime_seconds", 0))
                hours = int(uptime_s // 3600)
                mins = int((uptime_s % 3600) // 60)
                uptime_str = f"{hours}h {mins}m"
            except Exception:
                pass
        st.markdown(_metric_card("API Uptime", uptime_str, "green"), unsafe_allow_html=True)

    with c3:
        st.markdown(
            _metric_card("Models", f"{model_count} files", "cyan",
                         sub=f"Latest: {model_size_kb:.0f} KB" if model_size_kb else ""),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 3: Confidence API details ──
    c1, c2 = st.columns(2)

    with c1:
        entry_loaded = False
        exit_loaded = False
        if api_online:
            try:
                resp = requests.get("http://localhost:8100/health", timeout=3)
                data = resp.json()
                entry_loaded = bool(data.get("entry_gate_loaded"))
                exit_loaded = bool(data.get("exit_ctrl_loaded"))
            except Exception:
                pass
        cols_inner = st.columns(2)
        cols_inner[0].markdown(_badge("Entry Gate", entry_loaded), unsafe_allow_html=True)
        cols_inner[1].markdown(_badge("Exit Controller", exit_loaded), unsafe_allow_html=True)

    with c2:
        st.markdown(
            _metric_card("Ollama Models", str(ollama_model_count), "purple",
                         sub=f"Target: {'✓' if target_loaded else '✗'}"),
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
# TAB 4: CONFIDENCE API
# ══════════════════════════════════════════════════════════════════════

def tab_confidence_api():
    """Confidence API monitoring — mirrors Grafana confidence-api dashboard."""

    api_online = _check_service("http://localhost:8100/health")

    # Fetch detailed health from API
    api_health = {}
    if api_online:
        try:
            r = requests.get("http://localhost:8100/health", timeout=3)
            if r.status_code == 200:
                api_health = r.json()
        except Exception:
            pass

    entry_loaded = api_health.get("entry_gate_loaded", False)
    exit_loaded = api_health.get("exit_ctrl_loaded", False)
    uptime = api_health.get("uptime_seconds", 0)

    # Parse audit log
    audit_path = os.path.join(LOGS_DIR, "trade_audit.jsonl")
    entries = []
    if os.path.exists(audit_path):
        try:
            with open(audit_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass

    recent = entries[-100:] if len(entries) > 100 else entries

    latencies = [e.get("latency_ms", 0) for e in recent if e.get("latency_ms")]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    max_lat = max(latencies) if latencies else 0
    p99_lat = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
    allows = sum(1 for e in recent if e.get("advice") == "allow")
    blocks = sum(1 for e in recent if e.get("advice") == "block")

    # Format uptime
    if uptime > 3600:
        uptime_str = f"{uptime / 3600:.1f} hrs"
    elif uptime > 60:
        uptime_str = f"{uptime / 60:.0f} min"
    else:
        uptime_str = f"{uptime:.0f} sec"

    # ── Row 1: Stat cards ──
    card_data = [
        ("API Status", "ONLINE" if api_online else "OFFLINE", "green" if api_online else "red"),
        ("Uptime", uptime_str if api_online else "—", "blue" if api_online else "red"),
        ("Entry Gate", "LOADED" if entry_loaded else "NOT LOADED", "green" if entry_loaded else "yellow"),
        ("Exit Controller", "LOADED" if exit_loaded else "NOT LOADED", "green" if exit_loaded else "yellow"),
        ("Recent Allows", str(allows), "green"),
        ("Recent Blocks", str(blocks), "orange"),
    ]
    for row_start in range(0, len(card_data), 3):
        row_cards = card_data[row_start:row_start + 3]
        cols = st.columns(len(row_cards))
        for col, (label, val, color) in zip(cols, row_cards):
            col.markdown(_metric_card(label, val, color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Model status detail + Latency ──
    c1, c2 = st.columns(2)

    with c1:
        if not entry_loaded or not exit_loaded:
            st.markdown(
                f'<div style="background:{COLORS["card_bg"]}; border:1px solid {COLORS["border"]}; '
                f'border-radius:8px; padding:10px 12px;">'
                f'<div style="color:{COLORS["yellow"]}; font-size:0.95em; font-weight:bold; margin-bottom:6px;">'
                f'⚠️ Models Not Loaded</div>'
                f'<div style="color:{COLORS["text"]}; font-size:0.9em; line-height:1.6;">'
                f'{"❌ Entry Gate (es_entry_gate.json)" if not entry_loaded else "✅ Entry Gate"}<br>'
                f'{"❌ Exit Controller (es_exit_cnn.pt)" if not exit_loaded else "✅ Exit Controller"}<br><br>'
                f'The ES strategy models need to be trained first.<br>'
                f'Run the daily pipeline or train manually:<br>'
                f'<code style="color:{COLORS["cyan"]};">python -m src.pipeline.daily_run</code>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        elif latencies:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=latencies, mode="lines",
                                     name="Latency", line=dict(color=COLORS["cyan"], width=1.5)))
            fig.add_hline(y=avg_lat, line_dash="dash", line_color=COLORS["yellow"],
                          annotation_text=f"Avg: {avg_lat:.1f}ms")
            fig.update_layout(**DARK_LAYOUT, title=dict(text="Latency (recent 100 requests)", font=TITLE_FONT),
                              yaxis_title="ms", height=260)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Models loaded — waiting for trade requests from ES strategy runner")

    with c2:
        if allows or blocks:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Allow", "Block"], y=[allows, blocks],
                                 marker_color=[COLORS["green"], COLORS["orange"]]))
            fig.update_layout(**DARK_LAYOUT, title=dict(text="Allow vs Block (recent 100)", font=TITLE_FONT),
                              height=260)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show latency stats if we have them, otherwise endpoint info
            st.markdown(
                f'<div style="background:{COLORS["card_bg"]}; border:1px solid {COLORS["border"]}; '
                f'border-radius:8px; padding:10px 12px;">'
                f'<div style="color:{COLORS["blue"]}; font-size:0.95em; font-weight:bold; margin-bottom:6px;">'
                f'📡 API Endpoints</div>'
                f'<div style="color:{COLORS["text"]}; font-size:0.9em; line-height:1.8; font-family:monospace;">'
                f'GET  /health &nbsp;&nbsp;&nbsp;→ Service health<br>'
                f'POST /confidence → Entry gate score<br>'
                f'POST /exit &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Exit signal<br>'
                f'POST /spread &nbsp;&nbsp;&nbsp;→ Update spread<br>'
                f'</div>'
                f'<div style="color:{COLORS["text_secondary"]}; font-size:0.8em; margin-top:10px;">'
                f'Base URL: http://localhost:8100</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Row 3: Audit log stats ──
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    if os.path.exists(audit_path):
        size_kb = os.path.getsize(audit_path) / 1024
        c1.markdown(_metric_card("Audit Log Size", f"{size_kb:.1f} KB", "blue"), unsafe_allow_html=True)
        c2.markdown(_metric_card("Total Entries", str(len(entries)), "cyan"), unsafe_allow_html=True)
    else:
        c1.markdown(_metric_card("Audit Log", "No log yet", "yellow"), unsafe_allow_html=True)
        c2.markdown(_metric_card("Total Entries", "0", "yellow"), unsafe_allow_html=True)
    if latencies:
        c3.markdown(_metric_card("Avg Latency", f"{avg_lat:.1f} ms",
                                 "green" if avg_lat < 20 else "yellow"), unsafe_allow_html=True)
    else:
        c3.markdown(_metric_card("Avg Latency", "—", "yellow"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5: PIPELINE STATUS
# ══════════════════════════════════════════════════════════════════════

def tab_pipeline_status():
    """Pipeline Status monitoring — mirrors Grafana pipeline-status dashboard."""

    # ── Row 1: Process status badges ──
    scheduler_running = _check_pid("scheduler")
    es_running = _check_pid("es_strategy")
    api_online = _check_service("http://localhost:8100/health")
    dashboard_online = _check_service("http://localhost:8501")

    cols = st.columns(4)
    badges = [
        ("Scheduler", scheduler_running),
        ("ES Strategy Runner", es_running),
        ("Confidence API", api_online),
        ("Dashboard", dashboard_online),
    ]
    for col, (label, status) in zip(cols, badges):
        text = "RUNNING" if status else "STOPPED"
        color = COLORS["green"] if status else COLORS["red"]
        col.markdown(
            f'<div style="background:{COLORS["card_bg"]}; border:1px solid {COLORS["border"]}; '
            f'border-radius:6px; padding:10px 12px; text-align:center;">'
            f'<div style="color:{COLORS["text_secondary"]}; font-size:0.75em; margin-bottom:3px;">{label}</div>'
            f'<div style="color:{color}; font-size:1.1em; font-weight:bold;">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Data inventory bar chart ──
    conn = _get_db()
    if conn:
        tables = ["prices", "technicals", "news", "daily_sentiment", "macro",
                   "predictions", "intraday_bars", "options_chain",
                   "options_analytics", "intraday_features", "performance",
                   "earnings_calendar", "fed_communications"]
        rows = []
        # Route through DbRouter for all tables
        db_router = None
        if _HAS_ROUTER:
            try:
                import yaml
                with open("config.yaml") as f:
                    config = yaml.safe_load(f)
                db_router = get_router(config)
            except Exception:
                pass

        for t in tables:
            try:
                if db_router:
                    cnt_df = db_router.read_analytics(f"SELECT COUNT(*) as cnt FROM {t}")
                    count = int(cnt_df.iloc[0]["cnt"]) if not cnt_df.empty else 0
                else:
                    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                rows.append({"Table": t, "Rows": count})
            except Exception:
                rows.append({"Table": t, "Rows": 0})
        conn.close()

        df = pd.DataFrame(rows).sort_values("Rows", ascending=True)
        colors_bar = [COLORS["green"] if r > 100 else COLORS["blue"] for r in df["Rows"]]
        fig = go.Figure(go.Bar(
            x=df["Rows"], y=df["Table"], orientation="h",
            marker=dict(color=colors_bar),
            text=df["Rows"].apply(lambda x: f"{x:,}"),
            textposition="auto",
        ))
        fig.update_layout(**DARK_LAYOUT, title=dict(text="Data Table Row Counts", font=TITLE_FONT),
                          height=300, xaxis_title="Rows")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Audit trail info ──
    audit_path = os.path.join(LOGS_DIR, "trade_audit.jsonl")
    c1, c2 = st.columns(2)
    if os.path.exists(audit_path):
        size_kb = os.path.getsize(audit_path) / 1024
        with open(audit_path) as f:
            line_count = sum(1 for _ in f)
        c1.markdown(_metric_card("Audit Entries", f"{line_count:,}", "blue"), unsafe_allow_html=True)
        c2.markdown(_metric_card("Audit Size", f"{size_kb:.1f} KB", "cyan"), unsafe_allow_html=True)
    else:
        c1.markdown(_metric_card("Audit Entries", "0", "yellow"), unsafe_allow_html=True)
        c2.markdown(_metric_card("Audit Size", "—", "yellow"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 6: DATA SOURCES
# ══════════════════════════════════════════════════════════════════════

def tab_data_sources():
    """Data Sources monitoring — live status + historical data for every feed."""

    import yaml
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        config = {}

    from src.data.fetcher import FallbackFetcher
    fetcher = FallbackFetcher(config=config)

    # Sub-tabs for each data source category
    src_tabs = st.tabs([
        "📰 Finnhub",
        "📡 Yahoo",
        "📡 CNBC",
        "📡 MarketWatch",
        "📊 FRED",
        "💹 yfinance",
        "📅 Earnings",
        "🏛️ Fed Comms",
        "🗄️ All News",
    ])

    # ── Finnhub News ──
    with src_tabs[0]:
        _ds_finnhub(fetcher)

    # ── Yahoo RSS ──
    with src_tabs[1]:
        _ds_rss_source(fetcher, "yahoo",
                       "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US")

    # ── CNBC RSS ──
    with src_tabs[2]:
        _ds_rss_source(fetcher, "cnbc",
                       "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114")

    # ── MarketWatch RSS ──
    with src_tabs[3]:
        _ds_rss_source(fetcher, "marketwatch",
                       "https://feeds.marketwatch.com/marketwatch/topstories/")

    # ── FRED Macro ──
    with src_tabs[4]:
        _ds_fred(fetcher)

    # ── yfinance ──
    with src_tabs[5]:
        _ds_yfinance()

    # ── Earnings Calendar ──
    with src_tabs[6]:
        _ds_earnings()

    # ── Fed Communications ──
    with src_tabs[7]:
        _ds_fed_comms()

    # ── All News from DB ──
    with src_tabs[8]:
        _ds_all_news_db()


def _ds_finnhub(fetcher):
    """Finnhub news sub-tab."""
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">FINNHUB NEWS API</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        api_key = fetcher.finnhub_key
        if api_key:
            st.success(f"API Key: ...{api_key[-6:]}")
        else:
            st.error("No API key configured")

    articles = fetcher.get_news_finnhub(days=3)
    with c2:
        if articles:
            st.success(f"{len(articles)} articles fetched (last 3 days)")
        else:
            st.warning("0 articles — check API key or rate limits")

    if articles:
        rows = []
        for a in articles:
            rows.append({
                "Date": a.get("date", ""),
                "Source": a.get("source", ""),
                "Headline": a.get("headline", "")[:120],
                "URL": a.get("url", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                      column_config={"URL": st.column_config.LinkColumn("Link", display_text="🔗")})

    # Historical from DB
    st.markdown("---")
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">HISTORICAL (from DB)</p>',
                unsafe_allow_html=True)
    hist = _query_df(
        "SELECT date, source, headline, url FROM news WHERE source = 'finnhub' "
        "ORDER BY id DESC LIMIT 50"
    )
    if not hist.empty:
        st.caption(f"{len(hist)} recent Finnhub articles in database")
        st.dataframe(hist, use_container_width=True, hide_index=True)
    else:
        st.caption("No Finnhub articles stored yet")


def _ds_rss_source(fetcher, source_name: str, feed_url: str):
    """RSS feed sub-tab for a specific source."""
    import feedparser

    label = source_name.upper()
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">{label} RSS FEED</p>',
                unsafe_allow_html=True)

    # Live fetch
    articles = []
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:30]:
            pub_date = ""
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                from datetime import datetime as _dt
                pub_date = _dt(*entry.published_parsed[:6]).strftime("%Y-%m-%d %H:%M")
            articles.append({
                "Published": pub_date,
                "Headline": entry.get("title", "")[:140],
                "URL": entry.get("link", ""),
            })
        st.success(f"Live: {len(articles)} articles from {label}")
    except Exception as e:
        st.error(f"Feed error: {e}")

    if articles:
        df = pd.DataFrame(articles)
        st.dataframe(df, use_container_width=True, hide_index=True,
                      column_config={"URL": st.column_config.LinkColumn("Link", display_text="🔗")})

    # Historical from DB
    st.markdown("---")
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">HISTORICAL (from DB)</p>',
                unsafe_allow_html=True)
    hist = _query_df(
        "SELECT date, headline, url FROM news WHERE source = ? ORDER BY id DESC LIMIT 50",
        (source_name,)
    )
    if not hist.empty:
        st.caption(f"{len(hist)} stored {label} articles")
        st.dataframe(hist, use_container_width=True, hide_index=True)
    else:
        st.caption(f"No {label} articles stored yet")


def _ds_fred(fetcher):
    """FRED macro data sub-tab."""
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">FRED MACRO DATA</p>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1, 3])
    with c1:
        api_key = fetcher.fred_key
        if api_key:
            st.success(f"API Key: ...{api_key[-6:]}")
        else:
            st.warning("No key — using CSV fallback")

    # Live fetch
    macro = fetcher.get_macro_fred()
    with c2:
        ok = sum(1 for k, v in macro.items() if v is not None and k != "vix_change")
        st.success(f"Live: {ok}/6 indicators") if ok >= 4 else st.warning(f"Live: {ok}/6 indicators")

    # Show current values
    if macro:
        m1, m2, m3 = st.columns(3)
        with m1:
            v = macro.get("vix")
            st.metric("VIX", f"{v:.2f}" if v else "—")
        with m2:
            v = macro.get("us10y_yield")
            st.metric("10Y Yield", f"{v:.2f}%" if v else "—")
        with m3:
            v = macro.get("dxy")
            st.metric("DXY", f"{v:.2f}" if v else "—")
        m4, m5, m6 = st.columns(3)
        with m4:
            v = macro.get("fed_funds")
            st.metric("Fed Funds", f"{v:.2f}%" if v else "—")
        with m5:
            v = macro.get("gold")
            st.metric("Gold", f"${v:,.0f}" if v else "—")
        with m6:
            v = macro.get("crude")
            st.metric("Crude", f"${v:.2f}" if v else "—")

    # Historical chart from DB
    st.markdown("---")
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">HISTORICAL (from DB)</p>',
                unsafe_allow_html=True)
    hist = _query_df("SELECT date, vix, us10y_yield, dxy, gold, crude FROM macro ORDER BY date DESC LIMIT 90")
    if not hist.empty:
        hist = hist.iloc[::-1]  # chronological
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=["VIX & 10Y Yield", "Gold & Crude"])
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["vix"], name="VIX",
                                 line=dict(color=COLORS["red"], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["us10y_yield"], name="10Y",
                                 line=dict(color=COLORS["blue"], width=2), yaxis="y2"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["gold"], name="Gold",
                                 line=dict(color=COLORS["yellow"], width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["crude"], name="Crude",
                                 line=dict(color=COLORS["green"], width=2)), row=2, col=1)
        fig.update_layout(**DARK_LAYOUT, height=300,
                          title=dict(text="Macro History", font=TITLE_FONT))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(hist.tail(20).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("No macro history in database yet")


def _ds_yfinance():
    """yfinance data sub-tab."""
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">YFINANCE (SPY PRICES)</p>',
                unsafe_allow_html=True)

    # Live check
    try:
        import yfinance as yf
        spy = yf.download("SPY", period="5d", progress=False)
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            last_close = float(spy["Close"].iloc[-1])
            last_date = str(spy.index[-1].date())
            st.success(f"Live: SPY ${last_close:.2f} ({last_date})")
        else:
            st.warning("yfinance returned no data")
    except Exception as e:
        st.error(f"yfinance error: {e}")

    # Historical from DB
    st.markdown("---")
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">PRICE HISTORY (from DB)</p>',
                unsafe_allow_html=True)
    hist = _query_df("SELECT date, open, high, low, close, volume FROM prices ORDER BY date DESC LIMIT 60")
    if not hist.empty:
        hist_c = hist.iloc[::-1]
        fig = go.Figure(go.Candlestick(
            x=hist_c["date"], open=hist_c["open"], high=hist_c["high"],
            low=hist_c["low"], close=hist_c["close"], name="SPY",
        ))
        fig.update_layout(**DARK_LAYOUT, height=280,
                          title=dict(text="SPY Price (last 60 days)", font=TITLE_FONT),
                          xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(hist.head(20), use_container_width=True, hide_index=True)
    else:
        st.caption("No price data in database yet")


def _ds_earnings():
    """Earnings calendar sub-tab."""
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">EARNINGS CALENDAR</p>',
                unsafe_allow_html=True)

    conn = _get_db()
    if not conn:
        st.warning("Database not available")
        return

    try:
        count = conn.execute("SELECT COUNT(*) FROM earnings_calendar").fetchone()[0]
        st.metric("Total Entries", f"{count:,}")

        upcoming = pd.read_sql_query(
            "SELECT date, ticker, market_cap_pct FROM earnings_calendar "
            "WHERE date >= date('now') ORDER BY date ASC LIMIT 30", conn
        )
        if not upcoming.empty:
            st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">UPCOMING</p>',
                        unsafe_allow_html=True)
            st.dataframe(upcoming, use_container_width=True, hide_index=True)

        recent = pd.read_sql_query(
            "SELECT date, ticker, market_cap_pct FROM earnings_calendar "
            "WHERE date < date('now') ORDER BY date DESC LIMIT 30", conn
        )
        if not recent.empty:
            st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">RECENT</p>',
                        unsafe_allow_html=True)
            st.dataframe(recent, use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Earnings table not available: {e}")
    finally:
        conn.close()


def _ds_fed_comms():
    """Fed communications sub-tab."""
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">FED COMMUNICATIONS</p>',
                unsafe_allow_html=True)

    conn = _get_db()
    if not conn:
        st.warning("Database not available")
        return

    try:
        count = conn.execute("SELECT COUNT(*) FROM fed_communications").fetchone()[0]
        st.metric("Total Entries", f"{count:,}")

        recent = pd.read_sql_query(
            "SELECT date, type, hawkish_score, summary "
            "FROM fed_communications ORDER BY date DESC LIMIT 30", conn
        )
        if not recent.empty:
            # Sentiment chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=recent["date"], y=recent["hawkish_score"],
                marker_color=[COLORS["red"] if v > 0 else COLORS["green"] for v in recent["hawkish_score"]],
                name="Hawkish Score (+ = hawkish)",
            ))
            fig.update_layout(**DARK_LAYOUT, height=220,
                              title=dict(text="Fed Sentiment (hawkish score)", font=TITLE_FONT))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.caption("No fed communications stored yet")
    except Exception as e:
        st.warning(f"Fed comms table not available: {e}")
    finally:
        conn.close()


def _ds_all_news_db():
    """All news from database — filterable by source."""
    st.markdown(f'<p style="color:{COLORS["text_secondary"]};font-weight:600;font-size:0.85rem;">ALL NEWS (DATABASE)</p>',
                unsafe_allow_html=True)

    conn = _get_db()
    if not conn:
        st.warning("Database not available")
        return

    try:
        # Source breakdown
        sources = pd.read_sql_query(
            "SELECT source, COUNT(*) as count FROM news GROUP BY source ORDER BY count DESC", conn
        )
        if not sources.empty:
            fig = go.Figure(go.Bar(
                x=sources["count"], y=sources["source"], orientation="h",
                marker_color=COLORS["blue"],
                text=sources["count"].apply(lambda x: f"{x:,}"),
                textposition="auto",
                textfont=dict(color=COLORS["text"]),
            ))
            fig.update_layout(**DARK_LAYOUT, height=180,
                              title=dict(text="Articles by Source", font=TITLE_FONT),
                              xaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        # Filter
        all_sources = ["All"] + sources["source"].tolist() if not sources.empty else ["All"]
        f1, f2 = st.columns([1, 2])
        with f1:
            sel_source = st.selectbox("Source", all_sources, key="ds_news_source")
        with f2:
            limit = st.slider("Rows", 10, 200, 50, key="ds_news_limit")

        if sel_source == "All":
            df = pd.read_sql_query(
                f"SELECT date, source, headline, url FROM news ORDER BY id DESC LIMIT {limit}", conn
            )
        else:
            df = pd.read_sql_query(
                f"SELECT date, source, headline, url FROM news WHERE source = ? ORDER BY id DESC LIMIT {limit}",
                conn, params=(sel_source,)
            )

        if not df.empty:
            total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
            st.caption(f"Showing {len(df)} of {total:,} total articles")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No articles in database")
    except Exception as e:
        st.warning(f"News query error: {e}")
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════
# MAIN PAGE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def page_monitoring():
    """Main monitoring page with tabbed sub-dashboards."""
    _refresh_theme()

    # Theme-aware tab styling (inherits from app.py global CSS, just override tab styling)
    st.markdown(
        f"""
        <style>
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; background-color: {COLORS["tab_bg"]}; border-radius: 8px; padding: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border-radius: 6px;
            padding: 8px 16px;
            color: {COLORS["tab_text"]};
            font-weight: 500;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLORS["tab_active"]} !important;
            color: #FFFFFF !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Top control bar (Grafana-style) ──────────────────────────────────
    ctrl_cols = st.columns([1, 2, 1, 1, 2, 1])

    with ctrl_cols[0]:
        if st.button("\u25C0", key="mon_back", help="Shift time range back"):
            if "mon_offset" not in st.session_state:
                st.session_state["mon_offset"] = 0
            st.session_state["mon_offset"] += 1
            st.rerun()

    with ctrl_cols[1]:
        time_options = {
            "Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90,
            "Last 180 days": 180, "Last 1 year": 365,
        }
        time_label = st.selectbox(
            "Time Range", list(time_options.keys()), index=2,
            label_visibility="collapsed", key="mon_time_range",
        )
        days = time_options[time_label]

    with ctrl_cols[2]:
        if st.button("\u25B6", key="mon_fwd", help="Shift time range forward"):
            if st.session_state.get("mon_offset", 0) > 0:
                st.session_state["mon_offset"] -= 1
                st.rerun()

    with ctrl_cols[3]:
        if st.button("\u2212", key="mon_zoom_out", help="Zoom out (double range)"):
            current_idx = list(time_options.values()).index(days)
            if current_idx < len(time_options) - 1:
                st.session_state["mon_time_range"] = list(time_options.keys())[current_idx + 1]
                st.rerun()

    with ctrl_cols[4]:
        refresh_options = {"Off": 0, "10s": 10, "30s": 30, "1m": 60, "5m": 300}
        refresh_label = st.selectbox(
            "Refresh", list(refresh_options.keys()), index=2,
            label_visibility="collapsed", key="mon_refresh_interval",
        )
        refresh_secs = refresh_options[refresh_label]

    with ctrl_cols[5]:
        if st.button("\u21BB Refresh", key="mon_manual_refresh"):
            st.rerun()

    # Apply time offset for back/forward navigation
    offset = st.session_state.get("mon_offset", 0)
    if offset > 0:
        st.caption(f"⏪ Shifted back {offset} × {days} days")

    st.markdown("---")

    # Tabs
    tabs = st.tabs([
        "📈 SPY",
        "📊 ES",
        "🖥️ Health",
        "🤖 API",
        "⚙️ Pipeline",
        "📡 Sources",
    ])

    with tabs[0]:
        tab_spy_predictor(days)
    with tabs[1]:
        tab_es_strategy()
    with tabs[2]:
        tab_system_health()
    with tabs[3]:
        tab_confidence_api()
    with tabs[4]:
        tab_pipeline_status()
    with tabs[5]:
        tab_data_sources()

    if refresh_secs > 0:
        import time as _time
        _time.sleep(refresh_secs)
        st.rerun()
