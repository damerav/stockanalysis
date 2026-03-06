"""Performance Dashboard — historical model accuracy tracking.

Visualizes the performance table with rolling accuracy, stratified
breakdowns by confidence tier / VIX regime, and confusion matrix.
Uses thread-safe fresh PostgreSQL connections per query.
"""

import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def _load_backtest_results() -> pd.DataFrame:
    """Load historical backtest results from PostgreSQL."""
    df = _query_df(
        "SELECT date, predicted_direction, predicted_confidence, "
        "actual_direction, actual_return, correct, regime, cumulative_accuracy "
        "FROM backtest_results ORDER BY date ASC"
    )
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["correct"] = df["correct"].astype(int)
    return df


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

    # ── Predicted vs Actual (daily line chart) with OHLC ────────────
    st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                f'font-size:0.95rem;">Predicted vs Actual (Daily) + Price</p>',
                unsafe_allow_html=True)

    _dir_map = {
        "STRONG_BULLISH": 2, "BULLISH": 1, "NEUTRAL": 0,
        "BEARISH": -1, "STRONG_BEARISH": -2,
        "strong_bullish": 2, "bullish": 1, "neutral": 0,
        "bearish": -1, "strong_bearish": -2,
        "UP": 1, "DOWN": -1, "FLAT": 0,
        "up": 1, "down": -1, "flat": 0,
    }
    pv = df[["date", "predicted", "actual", "correct"]].copy()
    pv["pred_val"] = pv["predicted"].map(_dir_map).fillna(0)
    pv["actual_val"] = pv["actual"].map(_dir_map).fillna(0)

    # Fetch OHLC prices for the same date range
    _ohlc = pd.DataFrame()
    try:
        _min_date = pv["date"].min().strftime("%Y-%m-%d")
        _max_date = pv["date"].max().strftime("%Y-%m-%d")
        _ohlc = _query_df(
            "SELECT date, open, high, low, close FROM prices "
            "WHERE date >= %s AND date <= %s ORDER BY date",
            (_min_date, _max_date),
        )
        if not _ohlc.empty:
            _ohlc["date"] = pd.to_datetime(_ohlc["date"])
    except Exception:
        pass

    fig_pva = make_subplots(
        specs=[[{"secondary_y": True}]],
    )

    # OHLC candlestick on secondary y-axis (behind prediction lines)
    if not _ohlc.empty:
        fig_pva.add_trace(go.Candlestick(
            x=_ohlc["date"], open=_ohlc["open"], high=_ohlc["high"],
            low=_ohlc["low"], close=_ohlc["close"],
            name="OHLC", opacity=0.4,
            increasing_line_color=colors["green"],
            decreasing_line_color=colors["red"],
        ), secondary_y=True)

    # Predicted line
    fig_pva.add_trace(go.Scatter(
        x=pv["date"], y=pv["pred_val"], name="Predicted",
        mode="lines+markers",
        line=dict(color=colors["blue"], width=2),
        marker=dict(size=6, symbol="circle"),
    ), secondary_y=False)

    # Actual line
    fig_pva.add_trace(go.Scatter(
        x=pv["date"], y=pv["actual_val"], name="Actual",
        mode="lines+markers",
        line=dict(color=colors["orange"], width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ), secondary_y=False)

    # Highlight misses
    misses = pv[pv["correct"] == 0]
    if not misses.empty:
        fig_pva.add_trace(go.Scatter(
            x=misses["date"], y=misses["actual_val"], name="Miss",
            mode="markers",
            marker=dict(size=10, color=colors["red"], symbol="x",
                        line=dict(width=2)),
        ), secondary_y=False)

    _pva_layout = {k: v for k, v in layout.items() if k != "legend"}
    fig_pva.update_layout(
        **_pva_layout, height=420, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig_pva.update_yaxes(
        tickvals=[-2, -1, 0, 1, 2],
        ticktext=["Strong Bear", "Bearish", "Neutral", "Bullish", "Strong Bull"],
        gridcolor=colors["grid"], zeroline=True,
        zerolinecolor=colors["text_muted"], zerolinewidth=1,
        title_text="Direction", secondary_y=False,
    )
    fig_pva.update_yaxes(
        title_text="SPY Price ($)", gridcolor=colors["grid"],
        secondary_y=True,
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

    # ── Historical Backtest: Predicted vs Actual ─────────────────────
    st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                f'font-size:0.95rem;">Historical Backtest — Predicted vs Actual</p>',
                unsafe_allow_html=True)

    bt_df = _load_backtest_results()
    if bt_df.empty:
        st.info("No backtest data yet. Click below to run a historical backtest "
                "using the current model against all available price history.")
        if st.button("🔬 Run Historical Backtest", key="run_backtest_btn"):
            with st.spinner("Running backtest across all historical data..."):
                try:
                    import yaml as _yaml
                    try:
                        with open("config.yaml") as f:
                            cfg = _yaml.safe_load(f) or {}
                    except Exception:
                        cfg = {}
                    from src.model.trainer import run_historical_backtest
                    bt_df = run_historical_backtest(cfg)
                    if not bt_df.empty:
                        st.success(f"Backtest complete: {len(bt_df)} days analyzed")
                        st.rerun()
                    else:
                        st.error("Backtest returned no results — check model and data availability")
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
    else:
        # Summary metrics
        bt_total = len(bt_df)
        bt_acc = bt_df["correct"].mean()
        dir_mask = bt_df["actual_direction"] != "NEUTRAL"
        bt_dir_acc = bt_df.loc[dir_mask, "correct"].mean() if dir_mask.any() else 0

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.markdown(metric_card("Backtest Days", str(bt_total)), unsafe_allow_html=True)
        bc2.markdown(metric_card("3-Class Accuracy", f"{bt_acc:.1%}",
                                 color="green" if bt_acc > 0.4 else "red"),
                     unsafe_allow_html=True)
        bc3.markdown(metric_card("Directional Accuracy", f"{bt_dir_acc:.1%}",
                                 color="green" if bt_dir_acc > 0.5 else "red"),
                     unsafe_allow_html=True)
        # Regime breakdown
        regime_counts = bt_df["regime"].value_counts()
        top_regime = regime_counts.index[0] if not regime_counts.empty else "N/A"
        bc4.markdown(metric_card("Dominant Regime", top_regime), unsafe_allow_html=True)

        st.markdown("")

        # Cumulative accuracy trend chart
        fig_bt = make_subplots(specs=[[{"secondary_y": True}]])

        fig_bt.add_trace(go.Scatter(
            x=bt_df["date"], y=bt_df["cumulative_accuracy"],
            name="Cumulative 3-Class", line=dict(color=colors["blue"], width=2),
        ), secondary_y=False)

        # Rolling 20-day accuracy
        bt_df["rolling_20d"] = bt_df["correct"].rolling(window=20, min_periods=5).mean()
        fig_bt.add_trace(go.Scatter(
            x=bt_df["date"], y=bt_df["rolling_20d"],
            name="20-Day Rolling", line=dict(color=colors["orange"], width=2, dash="dash"),
        ), secondary_y=False)

        fig_bt.add_hline(y=0.5, line_dash="dot", line_color=colors["text_muted"],
                         annotation_text="50%", annotation_position="bottom right")
        fig_bt.add_hline(y=0.333, line_dash="dot", line_color=colors["red"],
                         annotation_text="Random (33%)", annotation_position="bottom left")

        # Overlay actual returns on secondary axis
        fig_bt.add_trace(go.Bar(
            x=bt_df["date"], y=bt_df["actual_return"],
            name="Actual Return",
            marker_color=[colors["green"] if r > 0 else colors["red"]
                          for r in bt_df["actual_return"]],
            opacity=0.3,
        ), secondary_y=True)

        fig_bt.update_layout(**layout, height=420, xaxis_rangeslider_visible=False,
                             legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                         xanchor="right", x=1))
        fig_bt.update_yaxes(title_text="Accuracy", tickformat=".0%",
                            secondary_y=False)
        fig_bt.update_yaxes(title_text="Daily Return", tickformat=".1%",
                            secondary_y=True)
        st.plotly_chart(fig_bt, use_container_width=True, key="perf_backtest_trend")

        # Predicted vs Actual direction scatter
        st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                    f'font-size:0.95rem;">Direction Prediction Accuracy Over Time</p>',
                    unsafe_allow_html=True)

        _bt_dir_map = {
            "STRONG_BULLISH": 2, "WEAK_BULLISH": 1, "BULLISH": 1,
            "NEUTRAL": 0,
            "WEAK_BEARISH": -1, "BEARISH": -1, "STRONG_BEARISH": -2,
        }
        bt_pv = bt_df[["date", "predicted_direction", "actual_direction", "correct"]].copy()
        bt_pv["pred_val"] = bt_pv["predicted_direction"].map(_bt_dir_map).fillna(0)
        bt_pv["actual_val"] = bt_pv["actual_direction"].map(_bt_dir_map).fillna(0)

        fig_bt_pv = go.Figure()
        fig_bt_pv.add_trace(go.Scatter(
            x=bt_pv["date"], y=bt_pv["pred_val"], name="Predicted",
            mode="lines", line=dict(color=colors["blue"], width=1.5),
        ))
        fig_bt_pv.add_trace(go.Scatter(
            x=bt_pv["date"], y=bt_pv["actual_val"], name="Actual",
            mode="lines", line=dict(color=colors["orange"], width=1.5, dash="dash"),
        ))
        # Highlight misses
        bt_misses = bt_pv[bt_pv["correct"] == 0]
        if not bt_misses.empty and len(bt_misses) < 500:
            fig_bt_pv.add_trace(go.Scatter(
                x=bt_misses["date"], y=bt_misses["actual_val"], name="Miss",
                mode="markers",
                marker=dict(size=5, color=colors["red"], symbol="x", opacity=0.5),
            ))

        fig_bt_pv.update_layout(**layout, height=340,
                                yaxis=dict(tickvals=[-2, -1, 0, 1, 2],
                                           ticktext=["Strong Bear", "Bearish", "Neutral",
                                                     "Bullish", "Strong Bull"]),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                            xanchor="right", x=1))
        st.plotly_chart(fig_bt_pv, use_container_width=True, key="perf_backtest_pv")

        # Regime-stratified backtest accuracy
        st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                    f'font-size:0.95rem;">Backtest Accuracy by Regime</p>',
                    unsafe_allow_html=True)

        regime_acc = bt_df.groupby("regime").agg(
            accuracy=("correct", "mean"),
            count=("correct", "count"),
        ).sort_values("count", ascending=False)

        if not regime_acc.empty:
            regime_colors = {
                "bull_trend": colors["green"], "bear_trend": colors["red"],
                "high_vol_choppy": colors["orange"], "low_vol_range": colors["blue"],
            }
            bar_c = [regime_colors.get(r, colors["text_muted"]) for r in regime_acc.index]
            fig_regime = go.Figure(go.Bar(
                x=regime_acc.index, y=regime_acc["accuracy"],
                marker_color=bar_c,
                text=[f"{v:.0%}<br>n={c}" for v, c in
                      zip(regime_acc["accuracy"], regime_acc["count"])],
                textposition="outside",
            ))
            fig_regime.update_layout(**layout, height=300,
                                     yaxis_tickformat=".0%",
                                     yaxis_range=[0, min(1.0, regime_acc["accuracy"].max() + 0.15)])
            st.plotly_chart(fig_regime, use_container_width=True, key="perf_backtest_regime")

        # Confidence calibration: predicted confidence vs actual hit rate
        st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                    f'font-size:0.95rem;">Confidence Calibration</p>',
                    unsafe_allow_html=True)

        bt_df["conf_bucket"] = pd.cut(bt_df["predicted_confidence"],
                                       bins=[0, 35, 40, 45, 50, 55, 60, 100],
                                       labels=["<35%", "35-40%", "40-45%",
                                               "45-50%", "50-55%", "55-60%", ">60%"])
        cal_data = bt_df.groupby("conf_bucket", observed=True).agg(
            hit_rate=("correct", "mean"),
            count=("correct", "count"),
        ).dropna()

        if not cal_data.empty and len(cal_data) > 1:
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Bar(
                x=cal_data.index.astype(str), y=cal_data["hit_rate"],
                name="Actual Hit Rate",
                marker_color=colors["blue"],
                text=[f"{v:.0%}<br>n={c}" for v, c in
                      zip(cal_data["hit_rate"], cal_data["count"])],
                textposition="outside",
            ))
            # Perfect calibration line
            fig_cal.add_trace(go.Scatter(
                x=cal_data.index.astype(str),
                y=[0.35, 0.375, 0.425, 0.475, 0.525, 0.575, 0.65][:len(cal_data)],
                name="Perfect Calibration",
                mode="lines+markers",
                line=dict(color=colors["text_muted"], dash="dot"),
            ))
            fig_cal.update_layout(**layout, height=300,
                                  yaxis_tickformat=".0%",
                                  yaxis_range=[0, min(1.0, cal_data["hit_rate"].max() + 0.15)],
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                              xanchor="right", x=1))
            st.plotly_chart(fig_cal, use_container_width=True, key="perf_backtest_cal")

        # Re-run backtest button
        with st.expander("🔄 Re-run Backtest", expanded=False):
            st.caption("Re-run the historical backtest with the current model. "
                       "This will overwrite existing backtest results.")
            if st.button("Re-run Backtest", key="rerun_backtest_btn"):
                with st.spinner("Running backtest..."):
                    try:
                        import yaml as _yaml
                        try:
                            with open("config.yaml") as f:
                                cfg = _yaml.safe_load(f) or {}
                        except Exception:
                            cfg = {}
                        from src.model.trainer import run_historical_backtest
                        bt_df = run_historical_backtest(cfg)
                        if not bt_df.empty:
                            st.success(f"Backtest complete: {len(bt_df)} days")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")

    # ── Recent Predictions Table ─────────────────────────────────────
    with st.expander("📝 Recent Predictions", expanded=False):
        recent = df.tail(20).iloc[::-1]
        display_cols = [c for c in ["date", "predicted", "actual", "correct",
                        "confidence_tier", "vix_regime", "cumulative_accuracy"]
                       if c in recent.columns]
        st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)

    # ── Historical Backtest ──────────────────────────────────────────
    st.markdown(f'<p style="color:{colors["text_heading"]};font-weight:600;'
                f'font-size:0.95rem;">Historical Backtest (Model vs Market)</p>',
                unsafe_allow_html=True)
    st.caption("Runs the current model against all historical data to show "
               "predicted vs actual direction for every trading day.")

    if st.button("🔬 Run Historical Backtest", key="run_backtest"):
        with st.spinner("Running backtest across all historical dates..."):
            try:
                import yaml as _yaml
                try:
                    with open("config.yaml") as f:
                        cfg = _yaml.safe_load(f) or {}
                except Exception:
                    cfg = {}
                from src.model.trainer import generate_historical_backtest
                from src.data.db_router import DbRouter
                router = DbRouter(cfg)
                bt = generate_historical_backtest(router, config=cfg)
                router.close()
                if bt.empty:
                    st.warning("No backtest data generated. Ensure model and price data exist.")
                else:
                    st.session_state["_backtest_df"] = bt
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                logger.error(f"Historical backtest error: {e}", exc_info=True)

    bt = st.session_state.get("_backtest_df")
    if bt is not None and not bt.empty:
        # Summary metrics
        bt_total = len(bt)
        bt_correct = bt["correct"].sum()
        bt_acc = bt_correct / bt_total
        bt_bull_mask = bt["actual_direction"] == "BULLISH"
        bt_bear_mask = bt["actual_direction"] == "BEARISH"
        bt_bull_acc = bt.loc[bt_bull_mask, "correct"].mean() if bt_bull_mask.any() else 0
        bt_bear_acc = bt.loc[bt_bear_mask, "correct"].mean() if bt_bear_mask.any() else 0

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.markdown(metric_card("Backtest Days", str(bt_total)), unsafe_allow_html=True)
        bc2.markdown(metric_card("Backtest Accuracy", f"{bt_acc:.1%}",
                                 color="green" if bt_acc > 0.45 else "red"),
                     unsafe_allow_html=True)
        bc3.markdown(metric_card("Bull Day Accuracy", f"{bt_bull_acc:.1%}",
                                 color="green" if bt_bull_acc > 0.5 else "red"),
                     unsafe_allow_html=True)
        bc4.markdown(metric_card("Bear Day Accuracy", f"{bt_bear_acc:.1%}",
                                 color="green" if bt_bear_acc > 0.5 else "red"),
                     unsafe_allow_html=True)

        st.markdown("")

        # Rolling accuracy chart
        fig_bt = make_subplots(specs=[[{"secondary_y": True}]])

        fig_bt.add_trace(go.Scatter(
            x=bt["date"], y=bt["cumulative_accuracy"],
            name="Cumulative Accuracy", line=dict(color=colors["blue"], width=2),
        ), secondary_y=False)
        fig_bt.add_trace(go.Scatter(
            x=bt["date"], y=bt["rolling_accuracy_20d"],
            name="20-Day Rolling", line=dict(color=colors["orange"], width=2, dash="dash"),
        ), secondary_y=False)
        fig_bt.add_hline(y=0.5, line_dash="dot", line_color=colors["text_muted"],
                         annotation_text="50%", annotation_position="bottom right")

        # Overlay SPY close on secondary axis
        fig_bt.add_trace(go.Scatter(
            x=bt["date"], y=bt["close"],
            name="SPY Close", line=dict(color=colors["text_muted"], width=1),
            opacity=0.4,
        ), secondary_y=True)

        fig_bt.update_layout(**layout, height=400,
                             legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                         xanchor="right", x=1))
        fig_bt.update_yaxes(title_text="Accuracy", tickformat=".0%",
                            secondary_y=False)
        fig_bt.update_yaxes(title_text="SPY Close ($)", secondary_y=True)
        st.plotly_chart(fig_bt, use_container_width=True, key="bt_rolling_acc")

        # Prediction vs Actual direction chart
        _dir_map_bt = {
            "STRONG_BULLISH": 2, "WEAK_BULLISH": 1, "BULLISH": 1,
            "NEUTRAL": 0, "WEAK_BEARISH": -1, "BEARISH": -1,
            "STRONG_BEARISH": -2,
        }
        bt["pred_val"] = bt["predicted_direction"].map(_dir_map_bt).fillna(0)
        bt["actual_val"] = bt["actual_direction"].map(_dir_map_bt).fillna(0)

        fig_bt2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_bt2.add_trace(go.Scatter(
            x=bt["date"], y=bt["pred_val"], name="Predicted",
            mode="lines", line=dict(color=colors["blue"], width=1.5),
        ), secondary_y=False)
        fig_bt2.add_trace(go.Scatter(
            x=bt["date"], y=bt["actual_val"], name="Actual",
            mode="lines", line=dict(color=colors["orange"], width=1.5, dash="dash"),
        ), secondary_y=False)

        # Highlight misses
        bt_misses = bt[bt["correct"] == 0]
        if not bt_misses.empty:
            fig_bt2.add_trace(go.Scatter(
                x=bt_misses["date"], y=bt_misses["actual_val"], name="Miss",
                mode="markers",
                marker=dict(size=5, color=colors["red"], symbol="x", opacity=0.5),
            ), secondary_y=False)

        fig_bt2.add_trace(go.Scatter(
            x=bt["date"], y=bt["close"], name="SPY Close",
            line=dict(color=colors["text_muted"], width=1), opacity=0.3,
        ), secondary_y=True)

        fig_bt2.update_layout(**layout, height=400,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                          xanchor="right", x=1))
        fig_bt2.update_yaxes(
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=["Strong Bear", "Bearish", "Neutral", "Bullish", "Strong Bull"],
            title_text="Direction", secondary_y=False,
        )
        fig_bt2.update_yaxes(title_text="SPY Close ($)", secondary_y=True)
        st.plotly_chart(fig_bt2, use_container_width=True, key="bt_pred_vs_actual")

        # Confidence distribution
        with st.expander("📊 Backtest Details", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                fig_conf = go.Figure(go.Histogram(
                    x=bt["predicted_confidence"], nbinsx=20,
                    marker_color=colors["blue"], opacity=0.7,
                ))
                fig_conf.update_layout(**layout, title=dict(text="Confidence Distribution",
                                       font=get_title_font()), height=280,
                                       xaxis_title="Confidence %", yaxis_title="Count")
                st.plotly_chart(fig_conf, use_container_width=True, key="bt_conf_dist")

            with col_b:
                # Accuracy by confidence bucket
                bt["conf_bucket"] = pd.cut(bt["predicted_confidence"],
                                           bins=[0, 40, 50, 60, 70, 100],
                                           labels=["<40%", "40-50%", "50-60%", "60-70%", ">70%"])
                bucket_acc = bt.groupby("conf_bucket", observed=True).agg(
                    accuracy=("correct", "mean"), count=("correct", "count")
                )
                if not bucket_acc.empty:
                    fig_bkt = go.Figure(go.Bar(
                        x=bucket_acc.index.astype(str), y=bucket_acc["accuracy"],
                        marker_color=colors["blue"],
                        text=[f"{v:.0%}<br>n={c}" for v, c in
                              zip(bucket_acc["accuracy"], bucket_acc["count"])],
                        textposition="outside",
                    ))
                    fig_bkt.update_layout(**layout, title=dict(text="Accuracy by Confidence",
                                         font=get_title_font()), height=280,
                                         yaxis_tickformat=".0%",
                                         yaxis_range=[0, min(1.0, bucket_acc["accuracy"].max() + 0.15)])
                    st.plotly_chart(fig_bkt, use_container_width=True, key="bt_acc_by_conf")

            # Raw data table
            st.dataframe(
                bt[["date", "predicted_direction", "predicted_confidence",
                    "actual_direction", "actual_return_pct", "correct",
                    "rolling_accuracy_20d"]].tail(50).iloc[::-1],
                use_container_width=True, hide_index=True,
            )
