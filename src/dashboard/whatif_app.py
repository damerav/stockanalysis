"""8E. What-If Analysis Dashboard — Streamlit app on port 8503.

Usage:
    streamlit run src/dashboard/whatif_app.py --server.port 8503
"""

import sys
import os
import yaml
import logging
import streamlit as st
import plotly.graph_objects as go
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.whatif.engine import WhatIfEngine
from src.whatif.presets import STRESS_SCENARIOS

logger = logging.getLogger(__name__)


@st.cache_resource
def get_engine():
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        config = {}
    return WhatIfEngine(config)


def main():
    st.set_page_config(page_title="What-If Analysis", layout="wide")
    st.title("What-If Analysis")

    engine = get_engine()

    tab_es, tab_spy = st.tabs(["📈 ES Strategy", "🔮 SPY Predictor"])

    with tab_es:
        render_es_tab(engine)

    with tab_spy:
        render_spy_tab(engine)


# ------------------------------------------------------------------
# ES Strategy Tab
# ------------------------------------------------------------------

def render_es_tab(engine: WhatIfEngine):
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
                result = engine.es_parameter_sweep({
                    "credit_C": c_vals, "strike_K": k_vals,
                })
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
        st.info("Define two ES config variants to compare side-by-side.")
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


def _render_sweep_heatmap(result: dict, c_vals: list, k_vals: list):
    data = result.get("results", {})
    if not data:
        st.warning("No results")
        return

    z = []
    for c in c_vals:
        row = []
        for k in k_vals:
            label = f"credit_C={c}, strike_K={k}"
            entry = data.get(label, {})
            row.append(entry.get("total_pnl", 0))
        z.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"K={k:.0f}" for k in k_vals],
        y=[f"C={c:.1f}" for c in c_vals],
        colorscale="RdYlGn", colorbar_title="P&L ($)",
    ))
    fig.update_layout(title="K/C Sweep — Total P&L", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.dataframe([
        {"params": k, "P&L": f"${v.get('total_pnl', 0):+,.0f}", "trades": v.get("trades", 0)}
        for k, v in data.items() if "error" not in v
    ])


def _render_comparison_bar(result: dict):
    items = result.get("results", [])
    if not items:
        st.warning("No results")
        return

    labels = [r.get("label", "?") for r in items]
    pnls = [r.get("total_pnl", 0) for r in items]
    trades = [r.get("trades", 0) for r in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="P&L ($)", x=labels, y=pnls,
                         marker_color=["green" if p > 0 else "red" for p in pnls]))
    fig.update_layout(title="Scenario Comparison — P&L", height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    for i, item in enumerate(items):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.metric(item.get("label", "?"),
                      f"${item.get('total_pnl', 0):+,.0f}",
                      f"{item.get('trades', 0)} trades")


def _render_risk_chart(result: dict, vals: list):
    data = result.get("results", {})
    pnls = []
    for v in vals:
        label = f"circuit_breaker_usd={v}"
        entry = data.get(label, {})
        pnls.append(entry.get("total_pnl", 0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f"${v:,.0f}" for v in vals], y=pnls,
        mode="lines+markers", name="P&L",
    ))
    fig.update_layout(title="Circuit Breaker vs P&L", xaxis_title="Breaker Limit",
                      yaxis_title="Total P&L ($)", height=400)
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# SPY Predictor Tab
# ------------------------------------------------------------------

def render_spy_tab(engine: WhatIfEngine):
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
            vix = st.number_input("VIX", value=0.0, step=1.0, key="ov_vix",
                                  help="Set to 0 to skip")
            if vix > 0:
                overrides["vix"] = vix
            sent = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.05, key="ov_sent")
            if sent != 0:
                overrides["sentiment_score"] = sent
        with col2:
            rsi = st.number_input("RSI(14)", value=0.0, step=1.0, key="ov_rsi",
                                  help="Set to 0 to skip")
            if rsi > 0:
                overrides["rsi_14"] = rsi
            pc = st.number_input("Put/Call Ratio", value=0.0, step=0.1, key="ov_pc",
                                 help="Set to 0 to skip")
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
                              default=["sentiment_score", "put_call_ratio"],
                              key="ablation_drop")
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
        st.info("Tests how the neutral zone threshold affects prediction distribution.")
        if st.button("Run Threshold Sweep", key="run_thresh"):
            with st.spinner("Sweeping thresholds..."):
                results = _run_threshold_sweep(engine)
            _render_threshold_sweep(results)


def _render_spy_comparison(result: dict):
    if "error" in result:
        st.error(result["error"])
        return

    orig = result.get("original", {})
    mod = result.get("modified", {})

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Prediction**")
        _prediction_card(orig)
    with col2:
        st.markdown("**Modified Prediction**")
        _prediction_card(mod)

    # Show overrides
    overrides = result.get("overrides", {})
    if overrides:
        st.markdown("**Features Changed:**")
        st.json(overrides)

    # Scenario description
    desc = result.get("description", "")
    if desc:
        st.info(desc)

    # Narrative placeholder
    st.markdown("---")
    st.markdown("**LLM Narrative** *(run with LLM enabled for AI explanation)*")


def _prediction_card(pred: dict):
    direction = pred.get("direction", "N/A")
    confidence = pred.get("confidence", 0)
    probs = pred.get("probabilities", {})

    color = "🟢" if "BULLISH" in direction else "🔴" if "BEARISH" in direction else "⚪"
    st.markdown(f"### {color} {pred.get('scale_label', direction)}")
    st.markdown(f"Confidence: **{confidence:.0f}%**")

    if probs:
        fig = go.Figure(go.Bar(
            x=list(probs.values()), y=list(probs.keys()),
            orientation="h",
            marker_color=["red", "gray", "green"],
        ))
        fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)


def _render_ablation(result: dict):
    if "error" in result:
        st.error(result["error"])
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Accuracy", f"{result['baseline_accuracy']:.1%}")
    col2.metric("Ablated Accuracy", f"{result['ablated_accuracy']:.1%}")
    col3.metric("Impact", f"{result['accuracy_impact']:+.1%}",
                delta_color="inverse")

    st.markdown(f"Dropped features: `{', '.join(result['dropped'])}`")
    st.markdown(f"Tested on {result['samples']} samples")


def _render_monte_carlo(result: dict):
    if "error" in result:
        st.error(result["error"])
        return

    dist = result.get("distribution", {})

    fig = go.Figure(go.Bar(
        x=list(dist.keys()), y=list(dist.values()),
        marker_color=["green", "red", "gray"],
    ))
    fig.update_layout(title=f"Monte Carlo ({result['n_sims']} sims, {result['noise_pct']}% noise)",
                      yaxis_title="% of simulations", height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Avg Confidence", f"{result['avg_confidence']:.1f}%")
    col2.metric("Std Confidence", f"{result['std_confidence']:.1f}%")


def _run_threshold_sweep(engine: WhatIfEngine) -> list[dict]:
    """Sweep neutral threshold from 0.1% to 0.5%."""
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
        correct = 0
        total = 0
        for i in range(len(X) - tail, len(X)):
            pred = predictor.predict(X.iloc[i].values)
            actual = int(y.iloc[i])
            if not np.isnan(actual):
                total += 1
                if pred["predicted_class"] == actual:
                    correct += 1
        acc = correct / total if total > 0 else 0
        results.append({"threshold": f"±{thresh*100:.1f}%", "accuracy": round(acc, 4),
                         "samples": total})
    return results


def _render_threshold_sweep(results: list[dict]):
    if not results:
        st.warning("No results")
        return

    fig = go.Figure(go.Bar(
        x=[r["threshold"] for r in results],
        y=[r["accuracy"] * 100 for r in results],
        marker_color="steelblue",
    ))
    fig.update_layout(title="Accuracy by Neutral Threshold",
                      yaxis_title="Accuracy %", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(results)


# Need this import for ablation feature list
from src.data.features import get_feature_columns

if __name__ == "__main__":
    main()
