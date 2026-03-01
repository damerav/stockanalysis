"""🔍 Single-Stock Analysis Page — Deep-dive into any ticker.

Features: KPIs, performance metrics, 4-panel technical chart, AI narrative,
news & sentiment tab. Inspired by ErikThiart/ai-stock-dashboard.
"""

import os
import sqlite3
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = "./data"

# Theme-aware colors and layout
from src.dashboard.theme import (
    get_colors as _get_theme_colors,
    get_plotly_layout as _get_theme_layout,
    metric_card as _theme_metric_card,
    page_header, is_dark,
)


def _refresh_theme():
    global DARK_LAYOUT, COLORS
    COLORS = _get_theme_colors()
    DARK_LAYOUT = _get_theme_layout()

COLORS = _get_theme_colors()
DARK_LAYOUT = _get_theme_layout()


def _metric_card(label: str, value: str, color: str = "white", sub: str = "") -> str:
    _refresh_theme()
    return _theme_metric_card(label, value, color, sub)


@st.cache_data(ttl=300)
def _fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch OHLCV data via yfinance (or DB for SPY)."""
    if ticker == "SPY":
        try:
            conn = sqlite3.connect(os.path.join(DATA_DIR, "spy.db"))
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM prices ORDER BY date", conn)
            conn.close()
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception:
            pass
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.DataFrame()
        data = data.reset_index()
        data.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in data.columns]
        return data[["date", "open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _fetch_company_info(ticker: str) -> dict:
    """Fetch company metadata from yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector") or ("ETF" if info.get("quoteType") == "ETF" else "—"),
            "industry": info.get("industry") or ("ETF" if info.get("quoteType") == "ETF" else "—"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "dividend_yield": info.get("dividendYield", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "avg_volume": info.get("averageVolume", 0),
            "description": info.get("longBusinessSummary", ""),
            "website": info.get("website", ""),
            "employees": info.get("fullTimeEmployees", 0),
        }
    except Exception:
        return {"name": ticker}


def _compute_performance(df: pd.DataFrame) -> dict:
    """Compute performance metrics from price data."""
    if df.empty or len(df) < 2:
        return {}
    close = df["close"].values
    returns = pd.Series(close).pct_change().dropna()
    total_return = (close[-1] / close[0] - 1) * 100
    ann_return = total_return * (252 / len(df)) if len(df) > 0 else 0
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    max_dd = ((pd.Series(close).cummax() - close) / pd.Series(close).cummax()).max() * 100
    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "best_day": returns.max() * 100,
        "worst_day": returns.min() * 100,
    }


def _compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technicals for the stock analysis chart."""
    from src.data.features import (compute_sma, compute_rsi, compute_macd,
                                    compute_bollinger, compute_stochastic)
    df = df.sort_values("date").copy()
    close = df["close"]
    df["sma_20"] = compute_sma(close, 20)
    df["sma_50"] = compute_sma(close, 50)
    df["sma_200"] = compute_sma(close, 200)
    df["rsi_14"] = compute_rsi(close, 14)
    macd, sig, hist = compute_macd(close)
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = hist
    bb_u, bb_m, bb_l = compute_bollinger(close)
    df["bb_upper"] = bb_u
    df["bb_lower"] = bb_l
    stoch_k, stoch_d = compute_stochastic(df["high"], df["low"], close)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    return df


def _build_technical_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Build 4-panel technical chart (Price+MAs+BB, Volume, MACD, RSI+Stoch)."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.15, 0.20, 0.20],
        subplot_titles=["Price & Indicators", "Volume", "MACD", "RSI & Stochastic"],
    )

    # Panel 1: Candlestick + MAs + Bollinger
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name=ticker,
        increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"],
    ), row=1, col=1)
    for col_name, color, label in [
        ("sma_20", COLORS["yellow"], "SMA 20"),
        ("sma_50", COLORS["orange"], "SMA 50"),
        ("sma_200", COLORS["purple"], "SMA 200"),
    ]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col_name], name=label,
                line=dict(color=color, width=1), opacity=0.8,
            ), row=1, col=1)
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], name="BB Upper",
                                  line=dict(color=COLORS["cyan"], width=1, dash="dot"), opacity=0.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], name="BB Lower",
                                  line=dict(color=COLORS["cyan"], width=1, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(0,188,212,0.05)", opacity=0.5), row=1, col=1)

    # Panel 2: Volume
    vol_colors = [COLORS["green"] if c >= o else COLORS["red"]
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Volume",
                          marker_color=vol_colors, opacity=0.6), row=2, col=1)

    # Panel 3: MACD
    if "macd" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd"], name="MACD",
                                  line=dict(color=COLORS["cyan"], width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"], name="Signal",
                                  line=dict(color=COLORS["orange"], width=1.5)), row=3, col=1)
        hist_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in df["macd_hist"]]
        fig.add_trace(go.Bar(x=df["date"], y=df["macd_hist"], name="Histogram",
                              marker_color=hist_colors, opacity=0.5), row=3, col=1)

    # Panel 4: RSI + Stochastic
    if "rsi_14" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"], name="RSI",
                                  line=dict(color=COLORS["purple"], width=1.5)), row=4, col=1)
    if "stoch_k" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["stoch_k"], name="%K",
                                  line=dict(color=COLORS["blue"], width=1)), row=4, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["stoch_d"], name="%D",
                                  line=dict(color=COLORS["yellow"], width=1, dash="dash")), row=4, col=1)
    fig.add_hline(y=70, row=4, col=1, line_dash="dash", line_color=COLORS["red"], opacity=0.4)
    fig.add_hline(y=30, row=4, col=1, line_dash="dash", line_color=COLORS["green"], opacity=0.4)

    fig.update_layout(
        **{k: v for k, v in DARK_LAYOUT.items() if k not in ("xaxis", "yaxis", "legend")},
        height=700, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=COLORS["grid"], row=i, col=1)
        fig.update_yaxes(gridcolor=COLORS["grid"], row=i, col=1)
    # Update subplot title colors
    for ann in fig.layout.annotations:
        ann.font = dict(color=COLORS["text_secondary"], size=11)

    return fig


def _generate_ai_narrative(ticker: str, df: pd.DataFrame, perf: dict) -> str:
    """Generate rule-based AI narrative from technicals and performance."""
    if df.empty:
        return "Insufficient data for analysis."
    last = df.iloc[-1]
    close = float(last["close"])
    rsi = float(last.get("rsi_14", 50))
    macd_val = float(last.get("macd", 0))
    macd_sig = float(last.get("macd_signal", 0))
    sma20 = float(last.get("sma_20", close))
    sma50 = float(last.get("sma_50", close))
    stoch_k = float(last.get("stoch_k", 50))

    signals = []
    # Trend
    if close > sma20 > sma50:
        signals.append(("🟢 Uptrend", f"{ticker} is trading above both SMA 20 (\\${sma20:,.2f}) and SMA 50 (\\${sma50:,.2f}), confirming a bullish trend."))
    elif close < sma20 < sma50:
        signals.append(("🔴 Downtrend", f"{ticker} is below both moving averages, indicating bearish momentum."))
    else:
        signals.append(("🟡 Mixed", f"Price is between key moving averages — trend is unclear."))

    # RSI
    if rsi > 70:
        signals.append(("⚠️ Overbought", f"RSI at {rsi:.1f} — stock may be overextended to the upside."))
    elif rsi < 30:
        signals.append(("💡 Oversold", f"RSI at {rsi:.1f} — potential bounce opportunity."))
    else:
        signals.append(("✅ RSI Neutral", f"RSI at {rsi:.1f} — within normal range."))

    # MACD
    if macd_val > macd_sig:
        signals.append(("📈 MACD Bullish", "MACD line is above signal line — bullish momentum."))
    else:
        signals.append(("📉 MACD Bearish", "MACD line is below signal line — bearish momentum."))

    # Stochastic
    if stoch_k > 80:
        signals.append(("⚡ Stoch Overbought", f"Stochastic %K at {stoch_k:.0f} — overbought territory."))
    elif stoch_k < 20:
        signals.append(("⚡ Stoch Oversold", f"Stochastic %K at {stoch_k:.0f} — oversold territory."))

    # Performance summary
    if perf:
        sharpe = perf.get("sharpe", 0)
        dd = perf.get("max_drawdown", 0)
        if sharpe > 1.5:
            signals.append(("🏆 Strong Risk-Adjusted", f"Sharpe ratio of {sharpe:.2f} indicates excellent risk-adjusted returns."))
        if dd > 20:
            signals.append(("⚠️ High Drawdown", f"Max drawdown of {dd:.1f}% — significant downside risk observed."))

    parts = []
    for title, desc in signals:
        parts.append(f"**{title}**: {desc}")
    return "\n\n".join(parts)


def page_single_stock():
    """Render the 🔍 Single-Stock Analysis page."""
    _refresh_theme()
    # ── Compact toolbar: title + controls on one line ──
    _t, _c1, _c2 = st.columns([4, 1, 1])
    with _t:
        st.markdown(page_header('🔍 Single-Stock Analysis'), unsafe_allow_html=True)
    with _c1:
        ticker = st.selectbox("Ticker", ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
                                          "JPM", "V", "UNH", "XOM", "JNJ"],
                              key="ss_ticker", label_visibility="collapsed")
    with _c2:
        period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2,
                              key="ss_period", label_visibility="collapsed")

    # Fetch data
    df = _fetch_stock_data(ticker, period)
    if df.empty:
        st.warning(f"No data available for {ticker}")
        return

    info = _fetch_company_info(ticker)
    perf = _compute_performance(df)
    df_tech = _compute_technicals(df)

    # ── KPI Row ──
    last_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
    day_change = last_close - prev_close
    day_pct = (day_change / prev_close * 100) if prev_close else 0
    high_52w = info.get("52w_high", 0)
    low_52w = info.get("52w_low", 0)
    mcap = info.get("market_cap", 0)
    mcap_str = f"${mcap / 1e9:.1f}B" if mcap > 1e9 else f"${mcap / 1e6:.0f}M" if mcap > 1e6 else "—"

    kpis = [
        ("Price", f"${last_close:,.2f}", "green" if day_change >= 0 else "red", f"{day_change:+.2f} ({day_pct:+.1f}%)"),
        ("Market Cap", mcap_str, "blue", ""),
        ("P/E Ratio", f"{info.get('pe_ratio', 0):.1f}" if info.get("pe_ratio") else "—", "cyan", ""),
        ("52W High", f"${high_52w:,.2f}" if high_52w else "—", "green", ""),
        ("52W Low", f"${low_52w:,.2f}" if low_52w else "—", "red", ""),
        ("Avg Volume", f"{info.get('avg_volume', 0):,.0f}" if info.get("avg_volume") else "—", "yellow", ""),
    ]
    # Split into two rows of 3 to avoid overflow on smaller screens
    for row_start in range(0, len(kpis), 3):
        row_kpis = kpis[row_start:row_start + 3]
        cols = st.columns(len(row_kpis))
        for col, (label, val, color, sub) in zip(cols, row_kpis):
            col.markdown(_metric_card(label, val, color, sub), unsafe_allow_html=True)

    # ── Performance Panel ──
    if perf:
        with st.expander("📊 Performance Metrics", expanded=True):
            metrics = [
                ("Total Return", f"{perf['total_return']:+.1f}%", "green" if perf["total_return"] > 0 else "red"),
                ("Sharpe Ratio", f"{perf['sharpe']:.2f}", "green" if perf["sharpe"] > 1 else "yellow"),
                ("Volatility", f"{perf['volatility']:.1f}%", "yellow"),
                ("Max Drawdown", f"-{perf['max_drawdown']:.1f}%", "red"),
                ("Best Day", f"{perf['best_day']:+.1f}%", "green"),
            ]
            for row_start in range(0, len(metrics), 3):
                row_metrics = metrics[row_start:row_start + 3]
                pc = st.columns(len(row_metrics))
                for col, (label, val, color) in zip(pc, row_metrics):
                    col.markdown(_metric_card(label, val, color), unsafe_allow_html=True)

    # ── Tabbed Interface ──
    tabs = st.tabs(["📋 Company Info", "📊 Raw Data", "🔧 Technical Chart",
                     "🤖 AI Analysis", "📰 News & Sentiment"])

    # Tab 1: Company Info
    with tabs[0]:
        if info.get("description"):
            sector_industry = f"{info.get('sector', '')} / {info.get('industry', '')}" if info.get('sector') and info.get('sector') != '—' else ""
            header = f"**{info.get('name', ticker)}**"
            if sector_industry:
                header += f" — {sector_industry}"
            st.markdown(header)
            st.markdown(info["description"][:600] + ("..." if len(info.get("description", "")) > 600 else ""))
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Sector", info.get("sector", "—"))
                st.metric("Employees", f"{info.get('employees', 0):,}" if info.get("employees") else "—")
            with c2:
                st.metric("Industry", info.get("industry", "—"))
                if info.get("website"):
                    st.markdown(f"[Website]({info['website']})")
        else:
            st.info("Company info not available")

    # Tab 2: Raw Data
    with tabs[1]:
        st.dataframe(df.tail(100).sort_values("date", ascending=False),
                      use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, f"{ticker}_data.csv", "text/csv")

    # Tab 3: Technical Chart
    with tabs[2]:
        fig = _build_technical_chart(df_tech.tail(200), ticker)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 4: AI Analysis
    with tabs[3]:
        narrative = _generate_ai_narrative(ticker, df_tech, perf)
        st.markdown(narrative)

        # Feature importance from SPY model (if available and ticker is SPY)
        if ticker == "SPY":
            try:
                import json
                model_files = sorted(
                    [f for f in os.listdir("./models") if f.startswith("xgb_spy") and f.endswith(".json")],
                    reverse=True,
                )
                if model_files:
                    import xgboost as xgb
                    model = xgb.XGBClassifier()
                    model.load_model(os.path.join("./models", model_files[0]))
                    importance = model.get_booster().get_score(importance_type="gain")
                    if importance:
                        imp_df = pd.DataFrame(
                            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15],
                            columns=["Feature", "Importance"],
                        )
                        fig_imp = go.Figure(go.Bar(
                            y=imp_df["Feature"], x=imp_df["Importance"],
                            orientation="h", marker_color=COLORS["blue"],
                        ))
                        fig_imp.update_layout(**DARK_LAYOUT, height=300,
                                              title=dict(text="Feature Importance (XGBoost)", font=dict(color=COLORS["text"], size=13)),
                                              yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_imp, use_container_width=True)
            except Exception:
                pass

    # Tab 5: News & Sentiment
    with tabs[4]:
        _render_news_tab(ticker)


def _render_news_tab(ticker: str):
    """Render the News & Sentiment sub-tab."""
    try:
        from src.data.news_fetcher import NewsFetcher
        from src.data.news_features import NewsFeatureProcessor
        import yaml

        with open("config.yaml") as f:
            config = yaml.safe_load(f) or {}

        fetcher = NewsFetcher(config)
        articles = fetcher.get_recent(days=7, ticker=ticker)
        if not articles:
            articles = fetcher.get_recent(days=7)  # fallback to all
        fetcher.close()

        if not articles:
            st.info("No recent news articles available.")
            if st.button("🔄 Run News Pipeline Now", key="run_news_pipeline"):
                with st.spinner("Fetching news and processing features..."):
                    try:
                        from src.pipeline.news_pipeline_run import run_news_pipeline
                        run_news_pipeline(config)
                        st.success("News pipeline complete. Refreshing...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
            return

        processor = NewsFeatureProcessor(config)
        article_df = processor.process_articles(articles)
        processor.close()

        if article_df.empty:
            st.info("No processed articles available")
            return

        # Sentiment gauge
        avg_sent = article_df["sentiment_compound"].mean()
        sent_label = "Bullish" if avg_sent > 0.15 else "Bearish" if avg_sent < -0.15 else "Neutral"
        sent_color = COLORS["green"] if avg_sent > 0.15 else COLORS["red"] if avg_sent < -0.15 else COLORS["yellow"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                f'<div style="background:{COLORS["card"]}; {"backdrop-filter:blur(12px);" if is_dark() else "box-shadow:0 1px 3px rgba(0,0,0,0.08);"} border:1px solid {COLORS["card_border"]}; border-radius:10px; padding:20px; text-align:center;">'
                f'<div style="color:{COLORS["text_secondary"]}; font-size:0.8em;">SENTIMENT</div>'
                f'<div style="color:{sent_color}; font-size:2em; font-weight:bold;">{sent_label}</div>'
                f'<div style="color:{COLORS["text"]}; font-size:1.1em;">{avg_sent:+.3f}</div>'
                f'<div style="color:{COLORS["text_secondary"]}; font-size:0.8em;">{len(articles)} articles</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c2:
            # Sentiment over time
            daily = article_df.groupby("date")["sentiment_compound"].mean().reset_index()
            if not daily.empty:
                fig = go.Figure()
                colors = [COLORS["green"] if v > 0 else COLORS["red"] for v in daily["sentiment_compound"]]
                fig.add_trace(go.Bar(x=daily["date"], y=daily["sentiment_compound"],
                                      marker_color=colors, name="Sentiment"))
                fig.add_hline(y=0, line_color=COLORS["border"], line_dash="dash")
                fig.update_layout(**DARK_LAYOUT, height=200,
                                  title=dict(text="Daily Sentiment", font=dict(color=COLORS["text"], size=12)))
                st.plotly_chart(fig, use_container_width=True)

        # News predictor output
        try:
            from src.model.news_predictor import NewsPredictor
            predictor = NewsPredictor()
            model_path = "./models/news_predictor.pkl"
            if os.path.exists(model_path):
                predictor.load(model_path)
                texts = article_df["headline"].tolist()
                processor2 = NewsFeatureProcessor(config)
                try:
                    processor2.load_vectorizer()
                except Exception:
                    processor2.fit_tfidf(texts)
                tfidf = processor2.transform_tfidf(texts)
                X = predictor.prepare_features(tfidf, article_df)
                preds = predictor.predict(X)
                processor2.close()

                if preds:
                    # Aggregate predictions
                    labels = [p["label"] for p in preds]
                    from collections import Counter
                    counts = Counter(labels)
                    dominant = counts.most_common(1)[0]
                    st.markdown(f"**News-Based Prediction**: {dominant[0].replace('_', ' ').title()} "
                                f"({dominant[1]}/{len(preds)} articles agree) — "
                                f"Model accuracy: {predictor.accuracy:.1%}")
        except Exception:
            pass

        # Recent headlines
        st.markdown("**Recent Headlines**")
        for a in articles[:15]:
            sent = article_df[article_df["headline"] == a.get("headline")]
            score = float(sent["sentiment_compound"].iloc[0]) if not sent.empty else 0
            emoji = "🟢" if score > 0.15 else "🔴" if score < -0.15 else "⚪"
            pub = a.get("published_at", "")[:16]
            st.markdown(
                f'<div style="padding:4px 0; border-bottom:1px solid {COLORS["border"]};">'
                f'<span style="color:{COLORS["text_secondary"]}; font-size:0.75em;">{pub} {emoji} {a.get("source", "")}</span><br>'
                f'<span style="color:{COLORS["text"]}; font-size:0.9em;">{a.get("headline", "")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    except ImportError:
        st.info("News pipeline not available. Ensure news_fetcher and news_features modules are installed.")
    except Exception as e:
        st.warning(f"News tab error: {e}")
