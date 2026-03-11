"""LLM Comparison Panel — QwQ:32b vs Claude Opus 4.6 side-by-side predictions.

Shows both models' market analysis on the same prompt, tracks historical
accuracy, and lets users run on-demand comparisons.
"""

import json
import logging
import os
import threading
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.theme import (
    get_colors, get_plotly_layout, page_header, metric_card, is_dark,
)

logger = logging.getLogger(__name__)


def _load_spy_state() -> dict:
    """Load spy_state.json."""
    try:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "spy_state.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_macro() -> dict:
    """Load live macro data for prompt building."""
    try:
        from src.dashboard.app import _fetch_live_macro
        return _fetch_live_macro()
    except Exception:
        return {}


def _get_router():
    try:
        from src.data.db_router import get_router
        return get_router()
    except Exception:
        return None


def _store_comparison(router, date: str, qwq_result: dict, claude_result: dict,
                      prompt: str):
    """Store comparison results in DB."""
    if not router:
        return
    try:
        router.execute(
            """INSERT INTO llm_comparison_results
               (date, prompt_hash, qwq_direction, qwq_confidence, qwq_reasoning,
                qwq_latency, claude_direction, claude_confidence, claude_reasoning,
                claude_latency, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                date,
                str(hash(prompt))[-8:],
                qwq_result.get("direction", ""),
                qwq_result.get("confidence", 0),
                qwq_result.get("reasoning", "")[:500],
                qwq_result.get("latency_s", 0),
                claude_result.get("direction", ""),
                claude_result.get("confidence", 0),
                claude_result.get("reasoning", "")[:500],
                claude_result.get("latency_s", 0),
                datetime.now().isoformat(),
            ),
        )
    except Exception as e:
        logger.warning("Failed to store comparison: %s", e)


def _load_history(router, limit: int = 30) -> pd.DataFrame:
    """Load comparison history from DB."""
    if not router:
        return pd.DataFrame()
    try:
        from sqlalchemy import text as sa_text
        engine = router._pg_engine
        if engine:
            df = pd.read_sql_query(
                sa_text("SELECT * FROM llm_comparison_results ORDER BY created_at DESC LIMIT :lim"),
                engine, params={"lim": limit},
            )
            return df
    except Exception:
        pass
    try:
        df = router.query(
            "SELECT * FROM llm_comparison_results ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _run_comparison(state: dict, macro: dict) -> tuple:
    """Run both models in parallel and return (qwq_result, claude_result, prompt)."""
    from src.llm.claude_client import (
        analyze_market, analyze_market_qwq, build_market_prompt,
    )
    prompt = build_market_prompt(state, macro)
    qwq_result = {}
    claude_result = {}

    def _run_qwq():
        nonlocal qwq_result
        qwq_result = analyze_market_qwq(prompt)

    def _run_claude():
        nonlocal claude_result
        claude_result = analyze_market(prompt)

    t1 = threading.Thread(target=_run_qwq)
    t2 = threading.Thread(target=_run_claude)
    t1.start()
    t2.start()
    t1.join(timeout=180)
    t2.join(timeout=180)

    return qwq_result, claude_result, prompt


def _direction_color(direction: str, colors: dict) -> str:
    d = direction.upper()
    if d == "BULLISH":
        return colors["green"]
    elif d == "BEARISH":
        return colors["red"]
    return colors["yellow"]


def _direction_emoji(direction: str) -> str:
    d = direction.upper()
    if d == "BULLISH":
        return "🟢"
    elif d == "BEARISH":
        return "🔴"
    return "🟡"


def _render_model_card(title: str, result: dict, colors: dict):
    """Render a single model's prediction card."""
    error = result.get("error")
    if error:
        st.error(f"{title}: {error}")
        return

    direction = result.get("direction", "N/A")
    confidence = result.get("confidence", 0)
    reasoning = result.get("reasoning", "")
    factors = result.get("key_factors", [])
    risk = result.get("risk_level", "N/A")
    latency = result.get("latency_s", 0)
    model = result.get("model", "")

    dir_color = _direction_color(direction, colors)
    emoji = _direction_emoji(direction)

    shadow = "box-shadow:0 1px 3px rgba(0,0,0,0.08);" if not is_dark() else ""
    blur = "backdrop-filter:blur(12px);" if is_dark() else ""

    st.markdown(
        f'<div style="background:{colors["card"]}; {blur} '
        f'border:1px solid {colors["card_border"]}; border-radius:10px; '
        f'padding:16px; {shadow}">'
        f'<div style="color:{colors["text_secondary"]}; font-size:0.7em; '
        f'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;">'
        f'{title}</div>'
        f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">'
        f'<span style="font-size:2em;">{emoji}</span>'
        f'<span style="color:{dir_color}; font-size:1.6em; font-weight:700;">'
        f'{direction}</span>'
        f'</div>'
        f'<div style="display:flex; gap:16px; margin-bottom:10px;">'
        f'<div><span style="color:{colors["text_secondary"]}; font-size:0.7em;">'
        f'Confidence</span><br/>'
        f'<span style="color:{colors["text"]}; font-size:1.2em; font-weight:600;">'
        f'{confidence}%</span></div>'
        f'<div><span style="color:{colors["text_secondary"]}; font-size:0.7em;">'
        f'Risk</span><br/>'
        f'<span style="color:{colors["text"]}; font-size:1.2em; font-weight:600;">'
        f'{risk}</span></div>'
        f'<div><span style="color:{colors["text_secondary"]}; font-size:0.7em;">'
        f'Latency</span><br/>'
        f'<span style="color:{colors["text"]}; font-size:1.2em; font-weight:600;">'
        f'{latency}s</span></div>'
        f'</div>'
        f'<div style="color:{colors["text"]}; font-size:0.85em; line-height:1.5; '
        f'margin-bottom:8px;">{reasoning}</div>'
        f'<div style="color:{colors["text_secondary"]}; font-size:0.7em;">'
        f'Model: {model}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if factors:
        factor_html = " ".join(
            f'<span style="background:{colors["surface"]}; border:1px solid '
            f'{colors["border"]}; border-radius:4px; padding:2px 8px; '
            f'font-size:0.75em; color:{colors["text"]};">{f}</span>'
            for f in factors[:5]
        )
        st.markdown(
            f'<div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:4px;">'
            f'{factor_html}</div>',
            unsafe_allow_html=True,
        )


def _render_history_chart(df: pd.DataFrame, colors: dict):
    """Render agreement/accuracy chart from historical comparisons."""
    if df.empty:
        st.info("No comparison history yet. Run a comparison to start tracking.")
        return

    layout = get_plotly_layout()

    # Agreement over time
    df = df.sort_values("date")
    df["agree"] = df["qwq_direction"] == df["claude_direction"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["qwq_confidence"],
        name="QwQ:32b", mode="lines+markers",
        line=dict(color=colors["cyan"], width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["claude_confidence"],
        name="Claude Opus 4.6", mode="lines+markers",
        line=dict(color=colors["purple"], width=2),
        marker=dict(size=6),
    ))

    # Mark disagreements
    disagree = df[~df["agree"]]
    if not disagree.empty:
        fig.add_trace(go.Scatter(
            x=disagree["date"],
            y=[105] * len(disagree),
            name="Disagree",
            mode="markers",
            marker=dict(symbol="x", size=10, color=colors["red"]),
        ))

    fig.update_layout(**layout, title="Confidence Over Time", yaxis_title="Confidence %")
    st.plotly_chart(fig, use_container_width=True)


def _render_agreement_stats(df: pd.DataFrame, colors: dict):
    """Render agreement statistics."""
    if df.empty:
        return

    total = len(df)
    agree = (df["qwq_direction"] == df["claude_direction"]).sum()
    agree_pct = agree / total * 100 if total > 0 else 0

    avg_qwq_lat = df["qwq_latency"].mean() if "qwq_latency" in df.columns else 0
    avg_claude_lat = df["claude_latency"].mean() if "claude_latency" in df.columns else 0

    cols = st.columns(4)
    cols[0].markdown(metric_card("Total Comparisons", str(total)), unsafe_allow_html=True)
    cols[1].markdown(metric_card("Agreement Rate", f"{agree_pct:.1f}%",
                                 "green" if agree_pct > 70 else "yellow"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("Avg QwQ Latency", f"{avg_qwq_lat:.1f}s", "cyan"),
                     unsafe_allow_html=True)
    cols[3].markdown(metric_card("Avg Claude Latency", f"{avg_claude_lat:.1f}s", "purple"),
                     unsafe_allow_html=True)


def page_llm_comparison():
    """Main LLM Comparison page."""
    c = get_colors()
    st.markdown(page_header("LLM Comparison — QwQ:32b vs Claude Opus 4.6"),
                unsafe_allow_html=True)

    router = _get_router()

    # ── Run Comparison Section ──
    st.markdown(f'<p style="color:{c["text_secondary"]}; font-size:0.85em;">'
                f'Send the same market analysis prompt to both models and compare '
                f'their predictions side-by-side.</p>', unsafe_allow_html=True)

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("Run Comparison", type="primary", use_container_width=True)

    if run_btn:
        state = _load_spy_state()
        if not state:
            st.warning("No spy_state.json found. Run the pipeline first.")
            return

        macro = _load_macro()

        with st.spinner("Running both models in parallel..."):
            qwq_result, claude_result, prompt = _run_comparison(state, macro)

        # Store results
        today = datetime.now().strftime("%Y-%m-%d")
        _store_comparison(router, today, qwq_result, claude_result, prompt)

        # Cache in session for display
        st.session_state["llm_cmp_qwq"] = qwq_result
        st.session_state["llm_cmp_claude"] = claude_result
        st.session_state["llm_cmp_prompt"] = prompt

    # ── Display Latest Results ──
    qwq = st.session_state.get("llm_cmp_qwq")
    claude = st.session_state.get("llm_cmp_claude")

    if qwq and claude:
        col1, col2 = st.columns(2)
        with col1:
            _render_model_card("QwQ:32b (Local)", qwq, c)
        with col2:
            _render_model_card("Claude Opus 4.6 (API)", claude, c)

        # Agreement indicator
        qwq_dir = qwq.get("direction", "")
        claude_dir = claude.get("direction", "")
        if qwq_dir and claude_dir:
            if qwq_dir == claude_dir:
                st.success(f"Both models agree: {qwq_dir}")
            else:
                st.warning(f"Models disagree — QwQ: {qwq_dir} vs Claude: {claude_dir}")

        # Show prompt in expander
        prompt_text = st.session_state.get("llm_cmp_prompt", "")
        if prompt_text:
            with st.expander("View Prompt Sent to Both Models"):
                st.code(prompt_text, language="text")

    # ── Historical Comparison ──
    st.divider()
    st.markdown(f'<p style="color:{c["text_heading"]}; font-size:0.95em; font-weight:600;">'
                f'Comparison History</p>', unsafe_allow_html=True)

    history = _load_history(router)
    _render_agreement_stats(history, c)
    _render_history_chart(history, c)

    if not history.empty:
        with st.expander("Raw Comparison Data"):
            display_cols = [col for col in [
                "date", "qwq_direction", "qwq_confidence", "qwq_latency",
                "claude_direction", "claude_confidence", "claude_latency",
            ] if col in history.columns]
            st.dataframe(history[display_cols], use_container_width=True)
