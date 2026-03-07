"""Report Generator — produces PDF and CSV reports from the platform database.

All reports are read-only. No data is written to the database.
PDF reports use a light theme with embedded Plotly charts exported as PNG images.
"""
import io
import json
import logging
import os
import tempfile
import yaml
from datetime import datetime, date
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import xgboost as xgb

from src.data.db_router import get_router

logger = logging.getLogger(__name__)

ReportFormat = Literal["pdf", "csv"]


class _LightPDF(FPDF):
    """FPDF subclass with a light-theme header and footer."""

    TITLE_COLOR = (30, 30, 30)
    HEADER_COLOR = (41, 98, 255)
    ROW_ALT = (245, 247, 250)

    def __init__(self, title: str, date_range: str):
        super().__init__(orientation="L", unit="mm", format="A4")
        self._report_title = title
        self._date_range = date_range
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*self.TITLE_COLOR)
        self.cell(0, 10, self._report_title, ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 6, f"Date range: {self._date_range}  |  Generated: {now_str}",
                  ln=True, align="C")
        self.ln(4)

    def model_performance(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Rolling accuracy by confidence tier and VIX regime."""
        df = self.router.query(
            "SELECT date, confidence_tier, vix_regime, cumulative_accuracy "
            "FROM performance WHERE date BETWEEN ? AND ? ORDER BY date DESC",
            (str(start), str(end)),
        )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Model Performance Report", _date_range_label(start, end))
        if not df.empty and "cumulative_accuracy" in df.columns:
            fig = go.Figure(go.Scatter(
                x=df["date"], y=df["cumulative_accuracy"],
                mode="lines", line=dict(color="#2962FF"),
                name="Cumulative Accuracy",
            ))
            fig.update_layout(title="Cumulative Accuracy",
                              paper_bgcolor="white", plot_bgcolor="white",
                              yaxis_tickformat=".0%")
            pdf.add_chart(fig)
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def feature_importance(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Top feature importances from the most recently trained model."""
        import xgboost as xgb
        try:
            model_dir = "./models"
            model_files = sorted([
                f for f in os.listdir(model_dir)
                if f.startswith("xgb_spy_") and f.endswith(".json")
                and not f.endswith("_meta.json") and "_binary_" not in f
            ], reverse=True)
            if not model_files:
                df = pd.DataFrame(columns=["feature", "importance"])
            else:
                model_path = os.path.join(model_dir, model_files[0])
                meta_path = model_path.replace(".json", "_meta.json")
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                importances = model.feature_importances_
                with open(meta_path) as mf:
                    meta = json.load(mf)
                feature_names = meta.get("feature_names", [])
                df = pd.DataFrame({"feature": feature_names, "importance": importances})
                df = df.sort_values("importance", ascending=False).head(50)
        except Exception as e:
            logger.warning("Could not load model for feature importance: %s", e)
            df = pd.DataFrame(columns=["feature", "importance"])

        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Feature Importance Report", _date_range_label(start, end))
        if not df.empty:
            top20 = df.head(20)
            fig = go.Figure(go.Bar(
                x=top20["importance"], y=top20["feature"],
                orientation="h", marker_color="#2962FF",
            ))
            fig.update_layout(title="Top 20 Features by Importance",
                              paper_bgcolor="white", plot_bgcolor="white",
                              yaxis=dict(autorange="reversed"))
            pdf.add_chart(fig)
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def es_strategy_pnl(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """ES strategy P&L from es_state.json (no DB table for trades yet)."""
        try:
            with open("data/es_state.json") as f:
                state = json.load(f)
            df = pd.DataFrame([{
                "metric": k, "value": str(v)
            } for k, v in state.items() if k in (
                "total_pnl", "win_rate", "total_trades", "position",
                "regime", "daily_pnl", "max_drawdown",
            )])
        except Exception:
            df = pd.DataFrame(columns=["metric", "value"])
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("ES Strategy P&L Summary", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def market_data_export(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Raw prices and technicals for the selected date range."""
        df = self.router.query(
            "SELECT p.date, p.open, p.high, p.low, p.close, p.volume, "
            "t.rsi_14, t.macd, t.bb_upper, t.bb_lower, t.atr_14 "
            "FROM prices p LEFT JOIN technicals t ON p.date = t.date "
            "WHERE p.date BETWEEN ? AND ? ORDER BY p.date DESC",
            (str(start), str(end)),
        )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Market Data Export", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def options_analytics_export(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """IV, skew, put/call ratio, GEX for the selected date range."""
        df = self.router.query(
            "SELECT date, put_call_ratio, max_pain, iv_skew, gex, "
            "vanna_exposure, charm_exposure, zero_dte_pcr "
            "FROM options_analytics WHERE date BETWEEN ? AND ? ORDER BY date DESC",
            (str(start), str(end)),
        )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Options Analytics Export", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def news_sentiment_export(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Headlines and sentiment scores for the selected date range."""
        df = self.router.query(
            "SELECT date, category, avg_sentiment, article_count "
            "FROM news_features WHERE date BETWEEN ? AND ? ORDER BY date DESC",
            (str(start), str(end)),
        )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("News & Sentiment Export", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def macro_indicators_export(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Key macro indicators for the selected date range."""
        df = self.router.query(
            "SELECT date, vix, us10y_yield, us3m_yield, dxy, gold, crude, "
            "fed_funds, cpi, pce, gdp, nfp, unemployment_rate "
            "FROM macro WHERE date BETWEEN ? AND ? ORDER BY date DESC",
            (str(start), str(end)),
        )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Macro Indicators Export", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def platform_health(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Pipeline status, data freshness, model metadata, and system config."""
        sections = {}
        try:
            with open("config.yaml") as f:
                sections["config"] = yaml.safe_load(f) or {}
        except Exception:
            sections["config"] = {}
        try:
            with open("data/spy_state.json") as f:
                sections["spy_state"] = json.load(f)
        except Exception:
            sections["spy_state"] = {}

        if fmt == "csv":
            rows = []
            for section, data in sections.items():
                if isinstance(data, dict):
                    for k, v in data.items():
                        rows.append({"section": section, "key": k, "value": str(v)})
            return pd.DataFrame(rows).to_csv(index=False).encode()

        pdf = _LightPDF("Platform Health Report", _date_range_label(start, end))
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Current SPY State", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for k, v in sections.get("spy_state", {}).items():
            pdf.cell(0, 6, f"  {k}: {v}", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Configuration Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for section, data in sections.get("config", {}).items():
            pdf.cell(0, 6, f"  [{section}]: {str(data)[:80]}", ln=True)
        return bytes(pdf.output())

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Stock Analysis Platform  -  Page {self.page_no()}", align="C")

    def add_dataframe(self, df: pd.DataFrame):
        """Render a DataFrame as a styled table."""
        if df.empty:
            self.set_font("Helvetica", "I", 10)
            self.cell(0, 10, "No data available for this date range.", ln=True)
            return

        col_w = (self.w - 20) / len(df.columns)

        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*self.HEADER_COLOR)
        self.set_text_color(255, 255, 255)
        for col in df.columns:
            self.cell(col_w, 8, str(col)[:20], border=1, fill=True)
        self.ln()

        self.set_font("Helvetica", "", 8)
        self.set_text_color(*self.TITLE_COLOR)
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 2 == 0:
                self.set_fill_color(*self.ROW_ALT)
            else:
                self.set_fill_color(255, 255, 255)
            for val in row:
                self.cell(col_w, 7, str(val)[:20], border=1, fill=True)
            self.ln()

    def add_chart(self, fig: go.Figure, width_mm: int = 260):
        """Embed a Plotly figure as a PNG image in the PDF."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.write_image(tmp.name, width=1200, height=500, scale=2)
                self.image(tmp.name, x=10, w=width_mm)
                self.ln(5)
            os.unlink(tmp.name)
        except Exception as e:
            logger.warning("Could not embed chart in PDF: %s", e)


def _date_range_label(start: date, end: date) -> str:
    return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"


class ReportGenerator:
    """Generates PDF and CSV reports from the platform database."""

    def __init__(self, config: dict = None):
        if config is None:
            try:
                with open("config.yaml") as f:
                    config = yaml.safe_load(f) or {}
            except Exception:
                config = {}
        self.config = config
        self.router = get_router(self.config)

    def prediction_history(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Prediction direction, confidence, and actual outcome for each day."""
        df = self.router.query(
            "SELECT date, predicted_direction, predicted_confidence, "
            "actual_direction FROM backtest_results "
            "WHERE date BETWEEN ? AND ? ORDER BY date DESC",
            (str(start), str(end)),
        )
        if not df.empty:
            df["result"] = df.apply(
                lambda r: "Correct" if r["predicted_direction"] == r["actual_direction"] else "Wrong",
                axis=1,
            )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Prediction History", _date_range_label(start, end))
        if not df.empty and "result" in df.columns:
            counts = df["result"].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index.tolist(), values=counts.values.tolist(),
                hole=0.4, marker_colors=["#2962FF", "#FF4444"],
            ))
            fig.update_layout(title="Prediction Accuracy",
                              paper_bgcolor="white", plot_bgcolor="white")
            pdf.add_chart(fig)
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def model_performance(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Rolling accuracy by confidence tier and VIX regime."""
        df = self.router.query(
            "SELECT date, confidence_tier, vix_regime, cumulative_accuracy "
            "FROM performance WHERE date BETWEEN ? AND ? ORDER BY date DESC",
            (str(start), str(end)),
        )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Model Performance Report", _date_range_label(start, end))
        if not df.empty and "cumulative_accuracy" in df.columns:
            fig = go.Figure(go.Scatter(
                x=df["date"], y=df["cumulative_accuracy"],
                mode="lines", line=dict(color="#2962FF"),
            ))
            fig.update_layout(title="Cumulative Accuracy",
                              paper_bgcolor="white", plot_bgcolor="white",
                              yaxis_tickformat=".0%")
            pdf.add_chart(fig)
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def feature_importance(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Top feature importances from the most recently trained model."""
        try:
            model_dir = "./models"
            model_files = sorted([
                f for f in os.listdir(model_dir)
                if f.startswith("xgb_spy_") and f.endswith(".json")
                and not f.endswith("_meta.json") and "_binary_" not in f
            ], reverse=True)
            if not model_files:
                df = pd.DataFrame(columns=["feature", "importance"])
            else:
                model_path = os.path.join(model_dir, model_files[0])
                meta_path = model_path.replace(".json", "_meta.json")
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                importances = model.feature_importances_
                with open(meta_path) as mf:
                    meta = json.load(mf)
                feature_names = meta.get("feature_names", [])
                df = pd.DataFrame({"feature": feature_names, "importance": importances})
                df = df.sort_values("importance", ascending=False).head(50)
        except Exception as e:
            logger.warning("Could not load model for feature importance: %s", e)
            df = pd.DataFrame(columns=["feature", "importance"])

        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Feature Importance Report", _date_range_label(start, end))
        if not df.empty:
            top20 = df.head(20)
            fig = go.Figure(go.Bar(
                x=top20["importance"], y=top20["feature"],
                orientation="h", marker_color="#2962FF",
            ))
            fig.update_layout(title="Top 20 Features by Importance",
                              paper_bgcolor="white", plot_bgcolor="white",
                              yaxis=dict(autorange="reversed"))
            pdf.add_chart(fig)
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def es_strategy_pnl(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """ES strategy trade log with P&L summary (stub — no es_trades table yet)."""
        df = pd.DataFrame()
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("ES Strategy P&L Summary", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def market_data_export(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Raw prices and technicals for the selected date range."""
        df = self.router.query(
            "SELECT p.date, p.open, p.high, p.low, p.close, p.volume, "
            "t.rsi_14, t.macd, t.bb_upper, t.bb_lower, t.atr_14 "
            "FROM prices p LEFT JOIN technicals t ON p.date = t.date "
            "WHERE p.date BETWEEN ? AND ? ORDER BY p.date DESC",
            (str(start), str(end)),
        )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Market Data Export", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def options_analytics_export(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """IV, skew, put/call ratio, GEX for the selected date range."""
        df = self.router.query(
            "SELECT date, put_call_ratio, max_pain, iv_skew, gex, "
            "vanna_exposure, charm_exposure, zero_dte_pcr "
            "FROM options_analytics WHERE date BETWEEN ? AND ? ORDER BY date DESC",
            (str(start), str(end)),
        )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Options Analytics Export", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def news_sentiment_export(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """News sentiment summary (stub — raw_articles in news.db, not main DB)."""
        df = pd.DataFrame()
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("News & Sentiment Export", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def macro_indicators_export(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Key macro indicators for the selected date range."""
        df = self.router.query(
            "SELECT date, vix, us10y_yield, us3m_yield, dxy, gold, crude, "
            "fed_funds, cpi, pce, gdp, nfp, unemployment_rate "
            "FROM macro WHERE date BETWEEN ? AND ? ORDER BY date DESC",
            (str(start), str(end)),
        )
        if fmt == "csv":
            return df.to_csv(index=False).encode()
        pdf = _LightPDF("Macro Indicators Export", _date_range_label(start, end))
        pdf.add_dataframe(df)
        return bytes(pdf.output())

    def platform_health(self, start: date, end: date, fmt: ReportFormat) -> bytes:
        """Pipeline status, data freshness, model metadata, and system config."""
        sections = {}
        try:
            with open("config.yaml") as f:
                sections["config"] = yaml.safe_load(f) or {}
        except Exception:
            sections["config"] = {}
        try:
            with open("./data/spy_state.json") as f:
                sections["spy_state"] = json.load(f)
        except Exception:
            sections["spy_state"] = {}

        if fmt == "csv":
            rows = []
            for section, data in sections.items():
                if isinstance(data, dict):
                    for k, v in data.items():
                        rows.append({"section": section, "key": k, "value": str(v)})
            return pd.DataFrame(rows).to_csv(index=False).encode()

        pdf = _LightPDF("Platform Health Report", _date_range_label(start, end))
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Current SPY State", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for k, v in sections.get("spy_state", {}).items():
            pdf.cell(0, 6, f"  {k}: {v}", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Configuration Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for section, data in sections.get("config", {}).items():
            pdf.cell(0, 6, f"  [{section}]: {str(data)[:80]}", ln=True)
        return bytes(pdf.output())
