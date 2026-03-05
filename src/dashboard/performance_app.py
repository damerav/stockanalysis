"""Performance Dashboard — historical model accuracy tracking.

Visualizes the performance table with rolling accuracy, stratified
breakdowns by confidence tier / VIX regime, and confusion matrix.
Uses thread-safe fresh PostgreSQL connections per query.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yaml

from src.dashboard.theme import (
    get_colors,
    get_plotly_layout,
    get_title_font,
    page_header,
    metric_card,
    is_dark,
)

logger = logging.getLogger(__name__)

# ── PostgreSQL connection helper ─────────────────────────────────────

_pg_cfg = None
_pg_cfg_loaded = False


def _load_pg_config() -> dict | None:
    global _pg_cfg, _pg_cfg_loaded
    if _pg_cfg_loaded:
        return _pg_cfg
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        pg = cfg.get("database", {}).get("postgres")
        if pg and pg.get("dbname") and pg.get("user"):
            _pg_cfg = pg
    except Exception:
        pass
    _pg_cfg_loaded = True
    return _pg_cfg


def _pg_connect():
    """Create a fresh SQLAlchemy engine (thread-safe, suppresses psycopg2 warnings)."""
    pg = _load_pg_config()
    if not pg:
        return None
    try:
        from sqlalchemy import create_engine
        host = pg.get("host", "localhost")
        port = pg.get("port", 5432)
        dbname = pg["dbname"]
        user = pg["user"]
        password = pg.get("password", "")
        engine = create_engine(
            f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}",
            pool_pre_ping=True,
        )
        return engine
    except Exception:
        return None


def _query_df(sql: str, params=()) -> pd.DataFrame:
    """Thread-safe PostgreSQL query via SQLAlchemy engine — fresh per call."""
    from sqlalchemy import text as sa_text
    engine = _pg_connect()
    if not engine:
        st.error("PostgreSQL connection unavailable. Check config.yaml database settings.")
        return pd.DataFrame()
    try:
        # Convert %s positional params to :p0, :p1, ... named params
        sa_sql = sql
        sa_params = None
        if params:
            sa_dict = {}
            for i, val in enumerate(params):
                sa_sql = sa_sql.replace('%s', f':p{i}', 1)
                sa_dict[f'p{i}'] = val
            sa_params = sa_dict
        df = pd.read_sql_query(sa_text(sa_sql), engine,
                               params=sa_params if sa_params else None)
        return df
    except Exception as e:
        logger.error(f"PostgreSQL query failed: {e}")
        return pd.DataFrame()
    finally:
        try:
            engine.dispose()
        except Exception:
            pass


def _load_performance() -> pd.DataFrame:
    """Load full performance history from PostgreSQL."""
    df = _query_df(
        "SELECT date, predicted, actual, correct, cumulative_accuracy, "
        "confidence_tier, vix_regime, day_of_week, event_proximity "
        "FROM performance ORDER BY date ASC"
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["correct"] = df["correct"].astype(int)
    return df


def _load_registry_history() -> pd.DataFrame:
    """Load model registry history via DbRouter (PostgreSQL primary)."""
    try:
        import yaml as _yaml
        try:
            with open("config.yaml") as f:
                cfg = _yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
        from src.model.registry import ModelRegistry
        registry = ModelRegistry(cfg)
        rows = registry.get_history(limit=50)
        registry.close()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["training_date"] = pd.to_datetime(df["training_date"])
        return df
    except Exception as e:
        logger.error(f"Failed to load registry history: {e}")
        return pd.DataFrame()


def page_performance():
    """Renders the model performance analysis page."""
    st.markdown(page_header("🎯 Model Performance"), unsafe_allow_html=True)

    df = _load_performance()
    if df.empty:
        st.warning("No performance data found. Run the daily pipeline to start tracking.")
        return

    colors = get_colors()
    layout = get_plotly_layout()

    # ── Key Metrics Row ──────────────────────────────────────────────
    total = len(df)
    overall_acc = df["correct"].mean()
    recent_30 = df.tail(30)
    recent_acc = recent_30["correct"].mean() if len(recent_30) > 0 else 0
    streak = 0
    for v in df["correct"].iloc[::-1]:
        if v == 1:
            streak += 1
        else:
            break

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Total Predictions", str(total)), unsafe_allow_html=True)
    c2.markdown(metric_card("Overall Accuracy", f"{overall_acc:.1%}",
                            color="green" if overall_acc > 0.5 else "red"),
                unsafe_allow_html=True)
    c3.markdown(metric_card("Last 30 Days", f"{recent_acc:.1%}",
                            color="green" if recent_acc > 0.5 else "red"),
                unsafe_allow_html=True)
    c4.markdown(metric_card("Current Streak", f"{streak}✓" if streak > 0 else "0"),
                unsafe_allow_html=True)

    st.markdown("")

    # ── Accuracy Over Time ───────────────────────────────────────────
    st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                f'font-size:0.95rem;">Accuracy Trend</p>', unsafe_allow_html=True)

    df["rolling_20d"] = df["correct"].rolling(window=20, min_periods=5).mean()
    df["rolling_50d"] = df["correct"].rolling(window=50, min_periods=10).mean()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["cumulative_accuracy"],
        name="Cumulative", line=dict(color=colors["blue"], width=2),
    ))
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["rolling_20d"],
        name="20-Day Rolling", line=dict(color=colors["orange"], width=2, dash="dash"),
    ))
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["rolling_50d"],
        name="50-Day Rolling", line=dict(color=colors["cyan"], width=1.5, dash="dot"),
    ))
    fig1.add_hline(y=0.5, line_dash="dot", line_color=colors["text_muted"],
                   annotation_text="50%", annotation_position="bottom right")
    fig1.update_layout(**layout, yaxis_title="Accuracy", yaxis_tickformat=".0%",
                       height=340)
    st.plotly_chart(fig1, use_container_width=True, key="perf_accuracy_trend")

    # ── Predicted vs Actual (daily line chart) ───────────────────────
    st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                f'font-size:0.95rem;">Predicted vs Actual (Daily)</p>',
                unsafe_allow_html=True)

    _dir_map = {
        "STRONG_BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
        "BEARISH": -1, "STRONG_BEARISH": -2,
        # lowercase fallbacks
        "strong_bullish": 2, "bullish": 1, "neutral": 0,
        "bearish": -1, "strong_bearish": -2,
        "UP": 1, "DOWN": -1, "FLAT": 0,
        "up": 1, "down": -1, "flat": 0,
    }
    pv = df[["date", "predicted", "actual", "correct"]].copy()
    pv["pred_val"] = pv["predicted"].map(_dir_map).fillna(0)
    pv["actual_val"] = pv["actual"].map(_dir_map).fillna(0)

    fig_pva = go.Figure()
    fig_pva.add_trace(go.Scatter(
        x=pv["date"], y=pv["pred_val"], name="Predicted",
        mode="lines+markers",
        line=dict(color=colors["blue"], width=2),
        marker=dict(size=6, symbol="circle"),
    ))
    fig_pva.add_trace(go.Scatter(
        x=pv["date"], y=pv["actual_val"], name="Actual",
        mode="lines+markers",
        line=dict(color=colors["orange"], width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ))
    # Highlight misses with red markers
    misses = pv[pv["correct"] == 0]
    if not misses.empty:
        fig_pva.add_trace(go.Scatter(
            x=misses["date"], y=misses["actual_val"], name="Miss",
            mode="markers",
            marker=dict(size=10, color=colors["red"], symbol="x", line=dict(width=2)),
        ))
    fig_pva.update_layout(
        **layout, height=320,
        yaxis=dict(
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["Strong Bear", "Bearish", "Neutral", "Bullish", "Strong Bull"],
            gridcolor=colors["grid"], zeroline=True,
            zerolinecolor=colors["text_muted"], zerolinewidth=1,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_pva, use_container_width=True, key="perf_pred_vs_actual")

    # ── Stratified Performance ───────────────────────────────────────
    st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                f'font-size:0.95rem;">Stratified Performance</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # By Confidence Tier
    with col1:
        conf_data = df.groupby("confidence_tier").agg(
            accuracy=("correct", "mean"),
            count=("correct", "count"),
        ).reindex(["high", "medium", "low"])
        conf_data = conf_data.dropna()
        if not conf_data.empty:
            bar_colors = [colors["green"], colors["yellow"], colors["red"]][:len(conf_data)]
            fig2 = go.Figure(go.Bar(
                x=conf_data.index, y=conf_data["accuracy"],
                marker_color=bar_colors,
                text=[f"{v:.0%}<br>n={c}" for v, c in
                      zip(conf_data["accuracy"], conf_data["count"])],
                textposition="outside",
            ))
            fig2.update_layout(**layout, title=dict(text="By Confidence Tier",
                               font=get_title_font()),
                               yaxis_tickformat=".0%", height=300,
                               yaxis_range=[0, min(1.0, conf_data["accuracy"].max() + 0.15)])
            st.plotly_chart(fig2, use_container_width=True, key="perf_conf_tier")
        else:
            st.caption("No confidence tier data")

    # By VIX Regime
    with col2:
        vix_data = df.groupby("vix_regime").agg(
            accuracy=("correct", "mean"),
            count=("correct", "count"),
        ).reindex(["low", "normal", "high"])
        vix_data = vix_data.dropna()
        if not vix_data.empty:
            bar_colors = [colors["green"], colors["blue"], colors["red"]][:len(vix_data)]
            fig3 = go.Figure(go.Bar(
                x=vix_data.index, y=vix_data["accuracy"],
                marker_color=bar_colors,
                text=[f"{v:.0%}<br>n={c}" for v, c in
                      zip(vix_data["accuracy"], vix_data["count"])],
                textposition="outside",
            ))
            fig3.update_layout(**layout, title=dict(text="By VIX Regime",
                               font=get_title_font()),
                               yaxis_tickformat=".0%", height=300,
                               yaxis_range=[0, min(1.0, vix_data["accuracy"].max() + 0.15)])
            st.plotly_chart(fig3, use_container_width=True, key="perf_vix_regime")
        else:
            st.caption("No VIX regime data")

    # By Day of Week
    with col3:
        if "day_of_week" in df.columns:
            dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
            dow_data = df.groupby("day_of_week").agg(
                accuracy=("correct", "mean"),
                count=("correct", "count"),
            )
            dow_data.index = dow_data.index.map(lambda x: dow_map.get(int(x), str(x)))
            if not dow_data.empty:
                fig_dow = go.Figure(go.Bar(
                    x=dow_data.index, y=dow_data["accuracy"],
                    marker_color=colors["blue"],
                    text=[f"{v:.0%}<br>n={c}" for v, c in
                          zip(dow_data["accuracy"], dow_data["count"])],
                    textposition="outside",
                ))
                fig_dow.update_layout(**layout, title=dict(text="By Day of Week",
                                     font=get_title_font()),
                                     yaxis_tickformat=".0%", height=300,
                                     yaxis_range=[0, min(1.0, dow_data["accuracy"].max() + 0.15)])
                st.plotly_chart(fig_dow, use_container_width=True, key="perf_dow")
            else:
                st.caption("No day-of-week data")

    # ── Confusion Matrix ─────────────────────────────────────────────
    st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                f'font-size:0.95rem;">Confusion Matrix</p>', unsafe_allow_html=True)

    cm = pd.crosstab(df["actual"], df["predicted"],
                     rownames=["Actual"], colnames=["Predicted"])
    if not cm.empty:
        fig4 = go.Figure(data=go.Heatmap(
            z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
            colorscale="Blues" if is_dark() else "YlGnBu",
            text=cm.values, texttemplate="%{text}",
            textfont=dict(size=14),
            showscale=False,
        ))
        fig4.update_layout(**layout, height=320,
                           xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(fig4, use_container_width=True, key="perf_confusion")

    # ── Model Registry Timeline ──────────────────────────────────────
    reg_df = _load_registry_history()
    if not reg_df.empty:
        with st.expander("📋 Model Registry History", expanded=False):
            display_cols = [c for c in ["model_id", "training_date", "val_accuracy",
                            "test_accuracy", "feature_count", "deployment_status"]
                           if c in reg_df.columns]
            st.dataframe(reg_df[display_cols], use_container_width=True, hide_index=True)

    # ── Recent Predictions Table ─────────────────────────────────────
    with st.expander("📝 Recent Predictions", expanded=False):
        recent = df.tail(20).iloc[::-1]
        display_cols = [c for c in ["date", "predicted", "actual", "correct",
                        "confidence_tier", "vix_regime", "cumulative_accuracy"]
                       if c in recent.columns]
        st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)
