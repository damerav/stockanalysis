"""Model Tuning & Backtesting — interactive hyperparameter tuning and
feature selection with historical backtest execution.

Runs a full SPYPredictor training cycle with user-chosen config and
registers the result as a 'candidate' model for optional promotion.
"""

import logging
import streamlit as st
import yaml

from src.dashboard.theme import (
    get_colors,
    get_plotly_layout,
    page_header,
    metric_card,
)
from src.data.features import get_feature_columns

logger = logging.getLogger(__name__)

# Feature categories mapped to keyword prefixes in get_feature_columns()
FEATURE_CATEGORIES = {
    "Technical": ["price_vs_sma", "rsi", "macd", "bb_", "atr", "sma20_slope", "sma50_slope"],
    "Macro": ["vix", "us10y_yield", "dxy", "fed_funds", "gold", "crude"],
    "VIX Term Structure": ["vix9d", "vix3m", "vix6m", "vvix", "skew_index",
                           "vix_term", "vix_realised"],
    "Cross-Asset": ["hy_spread", "tlt_spy", "eem_spy", "copper_gold", "xlk_xl"],
    "Sentiment": ["sentiment", "article_count", "positive_ratio", "negative_ratio",
                  "macro_sentiment", "earnings_sentiment", "geopolitical_sentiment",
                  "technical_sentiment", "sentiment_dispersion", "sentiment_velocity"],
    "Options": ["put_call", "max_pain", "iv_skew", "gex", "vanna", "charm"],
    "Intraday": ["vwap", "intraday", "opening_gap", "reversal"],
    "Calendar": ["fomc", "cpi", "nfp", "opex", "witching", "quarter_end", "day_of", "week_of"],
}


def _categorize_features() -> dict[str, list[str]]:
    """Map each feature column to its category."""
    all_cols = get_feature_columns()
    result = {}
    for cat, keywords in FEATURE_CATEGORIES.items():
        matched = [col for col in all_cols if any(kw in col for kw in keywords)]
        if matched:
            result[cat] = matched
    # Catch any uncategorized features
    categorized = set()
    for feats in result.values():
        categorized.update(feats)
    uncategorized = [c for c in all_cols if c not in categorized]
    if uncategorized:
        result["Other"] = uncategorized
    return result


def page_tuning():
    """Renders the model tuning & backtesting page."""
    from src.dashboard.chatbot_widget import render_chatbot_widget
    render_chatbot_widget(page_key="tuning", page_title="Model Tuning")
    st.markdown(page_header("🛠️ Model Tuning & Backtesting"), unsafe_allow_html=True)

    colors = get_colors()
    categories = _categorize_features()

    with st.form(key="tuning_form"):
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                        f'font-size:0.95rem;">XGBoost Hyperparameters</p>',
                        unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                max_depth = st.slider("Max Depth", 2, 10, 6)
                learning_rate = st.slider("Learning Rate", 0.01, 0.30, 0.05, 0.01)
            with c2:
                n_estimators = st.slider("Num. Estimators", 100, 1500, 500, 50)
                subsample = st.slider("Subsample", 0.5, 1.0, 0.8, 0.05)

        with col_right:
            st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                        f'font-size:0.95rem;">Feature Selection</p>',
                        unsafe_allow_html=True)
            enabled_cats = {}
            for cat, feats in categories.items():
                enabled_cats[cat] = st.checkbox(f"{cat} ({len(feats)})", value=True)

        # Training enhancements row
        st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                    f'font-size:0.95rem;">Training Enhancements</p>',
                    unsafe_allow_html=True)
        enh1, enh2, enh3 = st.columns(3)
        with enh1:
            use_focal_loss = st.checkbox(
                "Focal Loss (hard-example mining)",
                value=True,
                help="Replaces softmax CE with focal loss (γ=1.5). "
                     "Down-weights easy examples, boosts BEARISH class (α=1.3)."
            )
        with enh2:
            regime_boost = st.slider(
                "Regime Sample Boost", 1.0, 3.0, 1.5, 0.1,
                help="Weight multiplier for training samples matching the current "
                     "market regime. Higher = more regime-adaptive."
            )
        with enh3:
            register_candidate = st.checkbox(
                "Register as candidate model", value=True,
                help="Save and register as 'candidate' in the model registry."
            )

        submit = st.form_submit_button("🚀 Run Backtest", type="primary")

    if submit:
        # Build feature list from selected categories
        selected_features = []
        for cat, feats in categories.items():
            if enabled_cats.get(cat, False):
                selected_features.extend(feats)

        if not selected_features:
            st.error("Select at least one feature category.")
            return

        model_config = {
            "xgboost": {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "subsample": subsample,
            }
        }

        with st.spinner("Running historical backtest... This may take several minutes on DGX."):
            try:
                import yaml as _yaml
                try:
                    with open("config.yaml") as f:
                        _cfg = _yaml.safe_load(f) or {}
                except Exception:
                    _cfg = {}
                from src.whatif.engine import WhatIfEngine
                engine = WhatIfEngine(config=_cfg)
                results = engine.spy_backtest(
                    model_config=model_config,
                    feature_list=selected_features,
                    register_as_candidate=register_candidate,
                )
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                logger.exception("Backtest error")
                return

        if "error" in results:
            st.error(f"Backtest error: {results['error']}")
            return

        # ── Results Display ──────────────────────────────────────────
        st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                    f'font-size:0.95rem;">Backtest Results</p>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        test_acc = results.get("test_accuracy") or 0
        val_acc = results.get("val_accuracy") or 0
        brier = results.get("brier_score") or 0
        feat_count = results.get("feature_count", 0)

        c1.markdown(metric_card("Test Accuracy", f"{test_acc*100:.1f}%",
                    color="green" if test_acc > 0.5 else "red"), unsafe_allow_html=True)
        c2.markdown(metric_card("Val Accuracy", f"{val_acc*100:.1f}%",
                    color="green" if val_acc > 0.5 else "red"), unsafe_allow_html=True)
        c3.markdown(metric_card("Brier Score", f"{brier:.4f}",
                    color="green" if brier < 0.25 else "yellow"), unsafe_allow_html=True)
        c4.markdown(metric_card("Features Used", str(feat_count)), unsafe_allow_html=True)

        if results.get("gated"):
            st.warning(f"⚠️ Model was gated: {results.get('gate_reason', 'unknown')}")

        # Top features
        top_feats = results.get("top_features", [])
        if top_feats:
            with st.expander("🏆 Top Features", expanded=True):
                import pandas as pd
                if isinstance(top_feats[0], (list, tuple)):
                    feat_df = pd.DataFrame(top_feats, columns=["Feature", "Importance"])
                else:
                    feat_df = pd.DataFrame({"Feature": top_feats})
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # ── Champion Promotion ───────────────────────────────────────
        model_id = results.get("model_id")
        if model_id:
            st.success(f"Candidate model registered: `{model_id}`")

            if st.button("👑 Promote to Champion", key=f"promote_{model_id}",
                         type="primary"):
                try:
                    import yaml as _yaml
                    try:
                        with open("config.yaml") as f:
                            _cfg = _yaml.safe_load(f) or {}
                    except Exception:
                        _cfg = {}
                    from src.model.registry import ModelRegistry
                    registry = ModelRegistry(_cfg)
                    success = registry.promote_model(model_id)
                    registry.close()
                    if success:
                        st.toast(f"Model {model_id} promoted to champion!", icon="🎉")
                        st.balloons()
                    else:
                        st.error("Promotion failed — check logs.")
                except Exception as e:
                    st.error(f"Promotion error: {e}")
        elif results.get("gated"):
            st.info("Gated models cannot be promoted.")
        elif not register_candidate:
            st.info("Model was not registered (checkbox was unchecked).")
