"""7A. Daily Pipeline Orchestrator — 13-step sequential pipeline at 4:30 PM ET.

Usage:
    python -m src.pipeline.daily_run
    python -m src.pipeline.daily_run --config config.yaml --skip-llm
"""

import argparse
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.data.init_db import get_connection, load_config
from src.data.polygon_fetcher import PolygonFetcher
from src.data.fetcher import FallbackFetcher
from src.data.daily_pull import run_daily_pull
from src.data.features import (
    compute_all_technicals, store_technicals,
    build_feature_vector, get_feature_columns, get_target,
)
from src.llm.analyzer import LLMAnalyzer
from src.llm.reporter import DailyReporter
from src.model.trainer import SPYPredictor, evaluate_past_prediction
from src.realtime.dashboard_bridge import write_spy_state
from src.data.feature_store import FeatureStore
from src.model.regime import HMMRegimeDetector
from src.data.earnings_calendar import fetch_earnings_yf, store_earnings
from src.data.fed_comms import update_fed_communications

logger = logging.getLogger(__name__)


class DailyPipeline:
    """13-step daily pipeline for SPY prediction and data refresh."""

    def __init__(self, config: dict):
        self.config = config
        self.conn: Optional[sqlite3.Connection] = None
        self.polygon: Optional[PolygonFetcher] = None
        self.fallback: Optional[FallbackFetcher] = None
        self.llm: Optional[LLMAnalyzer] = None
        self.predictor: Optional[SPYPredictor] = None
        self.reporter: Optional[DailyReporter] = None
        self.skip_llm = False
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.results: dict = {}

    def _init_components(self):
        """Initialise all pipeline components."""
        self.conn = get_connection(self.config)
        api_key = self.config.get("polygon", {}).get("api_key", "")
        if api_key and api_key != "YOUR_POLYGON_KEY":
            self.polygon = PolygonFetcher(api_key)
        self.fallback = FallbackFetcher()
        self.llm = LLMAnalyzer(self.config)
        self.predictor = SPYPredictor(self.config)
        self.reporter = DailyReporter(self.config)
        # P2 components
        self.feature_store = FeatureStore(self.config)
        self.regime_detector = HMMRegimeDetector()

    def run(self, skip_llm: bool = False) -> dict:
        """Execute the full 13-step pipeline.

        Returns dict with step results and overall status.
        """
        self.skip_llm = skip_llm
        start = time.time()
        logger.info(f"{'='*50}")
        logger.info(f"DAILY PIPELINE START — {self.today}")
        logger.info(f"{'='*50}")

        self._init_components()

        steps = [
            (0,   "LLM Health Check",         self._step0_llm_check),
            (0.5, "Daily Data Pull",           self._step05_data_pull),
            (1,   "Evaluate Past Predictions", self._step1_evaluate),
            (2,   "Fetch Daily Prices",        self._step2_prices),
            (3,   "Fetch News",                self._step3_news),
            (4,   "LLM Sentiment Analysis",    self._step4_sentiment),
            (5,   "Fetch Macro Data",          self._step5_macro),
            (6,   "Fetch Options Chain",       self._step6_options_chain),
            (7,   "Compute Options Analytics", self._step7_options_analytics),
            (8,   "Compute Technicals",        self._step8_technicals),
            (9,   "Build Intraday Features",   self._step9_intraday),
            (9.5, "Earnings Calendar",          self._step95_earnings),
            (9.6, "Fed Communications",         self._step96_fed_comms),
            (10,  "Retrain XGBoost",           self._step10_retrain),
            (11,  "Generate Prediction",        self._step11_predict),
            (12,  "Generate LLM Report",       self._step12_report),
            (13,  "Send Alerts",               self._step13_alerts),
        ]

        for step_num, name, func in steps:
            step_start = time.time()
            logger.info(f"\n--- Step {step_num}: {name} ---")
            try:
                result = func()
                elapsed = time.time() - step_start
                self.results[f"step_{step_num}"] = {
                    "status": "ok", "elapsed": round(elapsed, 1), "result": result,
                }
                logger.info(f"Step {step_num} complete ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - step_start
                logger.error(f"Step {step_num} FAILED ({elapsed:.1f}s): {e}")
                self.results[f"step_{step_num}"] = {
                    "status": "error", "elapsed": round(elapsed, 1), "error": str(e),
                }

        total = time.time() - start
        logger.info(f"\n{'='*50}")
        logger.info(f"PIPELINE COMPLETE — {total:.0f}s total")
        logger.info(f"{'='*50}")

        if self.conn:
            self.conn.close()

        self.results["total_elapsed"] = round(total, 1)
        self.results["date"] = self.today
        return self.results


    # --- Individual Steps ---

    def _step0_llm_check(self) -> dict:
        """Step 0: Check LLM model availability."""
        if self.skip_llm:
            logger.info("LLM check skipped (--skip-llm)")
            self.llm.llm_available = False
            return {"skipped": True}
        ok = self.llm.check_health()
        return {"llm_available": ok}

    def _step05_data_pull(self) -> dict:
        """Step 0.5: Gap detection and backfill."""
        counts = run_daily_pull(self.config)
        return {"table_counts": counts}

    def _step1_evaluate(self) -> dict:
        """Step 1: Evaluate yesterday's prediction accuracy."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        result = evaluate_past_prediction(self.conn, yesterday)
        if result:
            logger.info(f"Yesterday's prediction: {result['predicted']} → "
                        f"Actual: {result['actual']} — "
                        f"{'✓' if result['correct'] else '✗'} "
                        f"(cumulative: {result['cumulative_accuracy']:.1%})")
        else:
            logger.info("No prediction to evaluate for yesterday")
        return result or {"no_prediction": True}

    def _step2_prices(self) -> dict:
        """Step 2: Fetch today's daily prices."""
        if self.polygon:
            try:
                df = self.polygon.get_daily_bars("SPY", self.today, self.today)
                if not df.empty:
                    for _, row in df.iterrows():
                        self.conn.execute(
                            """INSERT OR REPLACE INTO prices
                               (date, open, high, low, close, volume)
                               VALUES (?,?,?,?,?,?)""",
                            (row["date"], row["open"], row["high"],
                             row["low"], row["close"], row["volume"]),
                        )
                    self.conn.commit()
                    return {"source": "polygon", "rows": len(df)}
            except Exception as e:
                logger.warning(f"Polygon price fetch failed: {e}")

        # Fallback to yfinance
        df = self.fallback.get_daily_bars_yf("SPY", days=5)
        if not df.empty:
            today_row = df[df["date"] == self.today]
            if not today_row.empty:
                for _, row in today_row.iterrows():
                    self.conn.execute(
                        """INSERT OR REPLACE INTO prices
                           (date, open, high, low, close, volume)
                           VALUES (?,?,?,?,?,?)""",
                        (row["date"], row["open"], row["high"],
                         row["low"], row["close"], row["volume"]),
                    )
                self.conn.commit()
                return {"source": "yfinance", "rows": len(today_row)}
        return {"source": "none", "rows": 0}

    def _step3_news(self) -> dict:
        """Step 3: Fetch news from Finnhub + RSS."""
        articles = []
        # Finnhub
        finnhub_articles = self.fallback.get_news_finnhub()
        articles.extend(finnhub_articles)
        # RSS
        rss_articles = self.fallback.get_news_rss()
        articles.extend(rss_articles)

        # Store in DB
        for a in articles:
            self.conn.execute(
                """INSERT INTO news (date, source, headline, summary, url, fetched_at)
                   VALUES (?,?,?,?,?,?)""",
                (a["date"], a["source"], a["headline"],
                 a["summary"], a.get("url", ""), a.get("fetched_at", "")),
            )
        self.conn.commit()
        logger.info(f"Fetched {len(articles)} news articles")
        return {"articles": len(articles)}

    def _step4_sentiment(self) -> dict:
        """Step 4: LLM sentiment analysis on today's news."""
        if not self.llm.llm_available:
            logger.info("LLM unavailable — using neutral sentiment")
            neutral = self.llm.get_neutral_sentiment()
            self._store_sentiment(neutral)
            return {"skipped": True, "sentiment": neutral}

        # Get today's articles
        rows = self.conn.execute(
            "SELECT headline, summary FROM news WHERE date = ? LIMIT 50",
            (self.today,),
        ).fetchall()
        articles = [{"headline": r[0], "summary": r[1]} for r in rows]

        if not articles:
            logger.info("No articles to analyse")
            neutral = self.llm.get_neutral_sentiment()
            self._store_sentiment(neutral)
            return {"articles": 0, "sentiment": neutral}

        logger.info(f"Analysing sentiment for {len(articles)} articles...")
        results = self.llm.analyze_sentiment(articles)
        daily = self.llm.aggregate_daily_sentiment(results)
        self._store_sentiment(daily)
        return {"articles": len(articles), "sentiment": daily}

    def _store_sentiment(self, sentiment: dict):
        self.conn.execute(
            """INSERT OR REPLACE INTO daily_sentiment
               (date, score, confidence, article_count,
                positive_ratio, negative_ratio, neutral_ratio,
                macro_sentiment, earnings_sentiment, geopolitical_sentiment,
                technical_sentiment, sentiment_dispersion, sentiment_velocity)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.today, sentiment["score"], sentiment.get("confidence", 0),
             sentiment.get("article_count", 0), sentiment.get("positive_ratio", 0),
             sentiment.get("negative_ratio", 0), sentiment.get("neutral_ratio", 1),
             sentiment.get("macro_sentiment", 0), sentiment.get("earnings_sentiment", 0),
             sentiment.get("geopolitical_sentiment", 0), sentiment.get("technical_sentiment", 0),
             sentiment.get("sentiment_dispersion", 0), sentiment.get("sentiment_velocity", 0)),
        )
        self.conn.commit()

    def _step5_macro(self) -> dict:
        """Step 5: Fetch macro data (VIX, yields, DXY, fed funds, gold, crude)
        + VIX term structure + cross-asset signals (P1)."""
        macro = self.fallback.get_macro_fred()

        # P1: VIX term structure
        vix_ts = self.fallback.get_vix_term_structure()
        macro.update({
            "vix9d": vix_ts.get("vix9d"),
            "vix3m": vix_ts.get("vix3m"),
            "vix6m": vix_ts.get("vix6m"),
            "vvix": vix_ts.get("vvix"),
            "skew_index": vix_ts.get("skew"),
        })

        # P1: Cross-asset signals
        cross = self.fallback.get_cross_asset_signals()
        macro.update({
            "hy_spread": cross.get("hy_spread"),
            "tlt_spy_ratio": cross.get("tlt_spy_ratio"),
            "eem_spy_ratio": cross.get("eem_spy_ratio"),
            "copper_gold_ratio": cross.get("copper_gold_ratio"),
            "xlk_xlf_ratio": cross.get("xlk_xlf_ratio"),
            "xlk_xle_ratio": cross.get("xlk_xle_ratio"),
        })

        self.conn.execute(
            """INSERT OR REPLACE INTO macro
               (date, vix, vix_change, us10y_yield, dxy, fed_funds, gold, crude,
                vix9d, vix3m, vix6m, vvix, skew_index,
                hy_spread, tlt_spy_ratio, eem_spy_ratio,
                copper_gold_ratio, xlk_xlf_ratio, xlk_xle_ratio)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.today, macro.get("vix"), macro.get("vix_change"),
             macro.get("us10y_yield"), macro.get("dxy"),
             macro.get("fed_funds"), macro.get("gold"), macro.get("crude"),
             macro.get("vix9d"), macro.get("vix3m"), macro.get("vix6m"),
             macro.get("vvix"), macro.get("skew_index"),
             macro.get("hy_spread"), macro.get("tlt_spy_ratio"),
             macro.get("eem_spy_ratio"), macro.get("copper_gold_ratio"),
             macro.get("xlk_xlf_ratio"), macro.get("xlk_xle_ratio")),
        )
        self.conn.commit()
        return macro


    def _step6_options_chain(self) -> dict:
        """Step 6: Fetch options chain snapshot."""
        if not self.polygon:
            return {"skipped": True, "reason": "no polygon key"}
        try:
            chain = self.polygon.get_options_chain("SPY")
            if chain.empty:
                return {"rows": 0}
            for _, row in chain.iterrows():
                self.conn.execute(
                    """INSERT OR REPLACE INTO options_chain
                       (date, contract_symbol, strike, expiry, option_type,
                        last_price, bid, ask, volume, open_interest,
                        iv, delta, gamma, theta, vega)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (self.today, row["contract_symbol"], row["strike"],
                     row["expiry"], row["option_type"], row["last_price"],
                     row["bid"], row["ask"], row["volume"],
                     row["open_interest"], row["iv"], row["delta"],
                     row["gamma"], row["theta"], row["vega"]),
                )
            self.conn.commit()
            return {"rows": len(chain)}
        except Exception as e:
            logger.warning(f"Options chain fetch failed: {e}")
            return {"error": str(e)}

    def _step7_options_analytics(self) -> dict:
        """Step 7: Compute options analytics (P/C ratio, max pain, IV skew, GEX,
        P3: vanna, charm, 0DTE PCR)."""
        if not self.polygon:
            return {"skipped": True}
        try:
            analytics = self.polygon.get_options_analytics("SPY")
            self.conn.execute(
                """INSERT OR REPLACE INTO options_analytics
                   (date, put_call_ratio, max_pain, iv_skew, gex,
                    vanna_exposure, charm_exposure, zero_dte_pcr)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (self.today, analytics.get("put_call_ratio"),
                 analytics.get("max_pain"), analytics.get("iv_skew"),
                 analytics.get("gex"), analytics.get("vanna_exposure"),
                 analytics.get("charm_exposure"), analytics.get("zero_dte_pcr")),
            )
            self.conn.commit()
            return analytics
        except Exception as e:
            logger.warning(f"Options analytics failed: {e}")
            return {"error": str(e)}

    def _step8_technicals(self) -> dict:
        """Step 8: Compute daily technicals (SMAs, RSI, MACD, BB, ATR)."""
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM prices ORDER BY date",
            self.conn,
        )
        if df.empty:
            return {"rows": 0}
        tech_df = compute_all_technicals(df, self.config)
        store_technicals(self.conn, tech_df)
        return {"rows": len(tech_df)}

    def _step9_intraday(self) -> dict:
        """Step 9: Build intraday features (VWAP spread, momentum, range)."""
        # Check if we have intraday bars for today
        row = self.conn.execute(
            "SELECT COUNT(*) FROM intraday_bars WHERE timestamp LIKE ?",
            (f"{self.today}%",),
        ).fetchone()
        bar_count = row[0] if row else 0

        if bar_count == 0:
            # No intraday data — store zeros
            self.conn.execute(
                """INSERT OR REPLACE INTO intraday_features
                   (date, vwap_spread, intraday_momentum, intraday_range, volume_ratio)
                   VALUES (?,?,?,?,?)""",
                (self.today, 0, 0, 0, 1.0),
            )
            self.conn.commit()
            return {"bars": 0, "features": "default"}

        # Compute from intraday bars
        bars = pd.read_sql_query(
            "SELECT * FROM intraday_bars WHERE timestamp LIKE ? ORDER BY timestamp",
            self.conn, params=(f"{self.today}%",),
        )
        if bars.empty:
            return {"bars": 0}

        vwap_val = (bars["close"] * bars["volume"]).sum() / bars["volume"].sum() if bars["volume"].sum() > 0 else bars["close"].mean()
        last_close = bars["close"].iloc[-1]
        vwap_spread = (last_close - vwap_val) / vwap_val if vwap_val else 0
        momentum = (bars["close"].iloc[-1] - bars["close"].iloc[0]) / bars["close"].iloc[0] if len(bars) > 1 else 0
        intraday_range = (bars["high"].max() - bars["low"].min()) / bars["close"].mean() if bars["close"].mean() else 0
        avg_vol = bars["volume"].mean() or 1
        volume_ratio = bars["volume"].iloc[-10:].mean() / avg_vol if len(bars) >= 10 else 1.0

        self.conn.execute(
            """INSERT OR REPLACE INTO intraday_features
               (date, vwap_spread, intraday_momentum, intraday_range, volume_ratio)
               VALUES (?,?,?,?,?)""",
            (self.today, round(vwap_spread, 6), round(momentum, 6),
             round(intraday_range, 6), round(volume_ratio, 4)),
        )
        self.conn.commit()
        return {"bars": bar_count, "features": "computed"}

    def _step95_earnings(self) -> dict:
        """Step 9.5: Fetch and store earnings calendar for mega-caps."""
        try:
            earnings = fetch_earnings_yf()
            if earnings:
                store_earnings(self.conn, earnings)
                logger.info(f"Stored {len(earnings)} earnings dates")
            return {"earnings_fetched": len(earnings)}
        except Exception as e:
            logger.warning(f"Earnings calendar fetch failed (non-fatal): {e}")
            return {"error": str(e)}

    def _step96_fed_comms(self) -> dict:
        """Step 9.6: Fetch and score latest Fed communications."""
        try:
            llm_analyzer = self.llm if (self.llm and self.llm.llm_available) else None
            results = update_fed_communications(self.conn, llm_analyzer)
            logger.info(f"Fed comms update: {results}")
            return results
        except Exception as e:
            logger.warning(f"Fed communications update failed (non-fatal): {e}")
            return {"error": str(e)}

    def _step10_retrain(self) -> dict:
        """Step 10: Build feature vector + retrain XGBoost (P2: with feature store, regime)."""
        feature_cols = get_feature_columns()

        # P2: Try feature store for cached features
        fv = None
        try:
            all_dates = [r[0] for r in self.conn.execute(
                "SELECT date FROM prices ORDER BY date").fetchall()]
            fv = self.feature_store.get_features(
                feature_cols, all_dates=all_dates,
                build_fn=lambda missing: build_feature_vector(self.conn),
            )
        except Exception as e:
            logger.warning(f"Feature store failed, falling back to direct build: {e}")

        if fv is None or fv.empty:
            fv = build_feature_vector(self.conn)

        if fv is None or len(fv) < 50:
            logger.warning("Insufficient data for training")
            self.predictor.load_latest_model()
            return {"skipped": True, "reason": "insufficient data"}

        # P2: HMM regime detection
        regime = "low_vol_range"
        regime_info = {}
        try:
            # Train/update regime detector
            price_macro = fv[["close", "volume"]].copy()
            if "vix" in fv.columns:
                price_macro["vix"] = fv["vix"]
            regime_info = self.regime_detector.fit(price_macro)
            regime = self.regime_detector.predict(price_macro.tail(60))
            logger.info(f"Current regime: {regime}")
        except Exception as e:
            logger.warning(f"Regime detection failed (non-fatal): {e}")

        available = [c for c in feature_cols if c in fv.columns]
        X = fv[available]
        y = get_target(fv)

        metrics = self.predictor.train(X, y, use_gpu=True, feature_names=available)
        metrics["regime"] = regime
        metrics["regime_info"] = regime_info
        return metrics

    def _step11_predict(self) -> dict:
        """Step 11: Generate prediction for next trading day."""
        if self.predictor.model is None:
            if not self.predictor.load_latest_model():
                return {"error": "no model available"}

        fv = build_feature_vector(self.conn, date=self.today)
        if fv is None or fv.empty:
            return {"error": "no features for today"}

        feature_cols = get_feature_columns()
        available = [c for c in feature_cols if c in fv.columns]
        features = fv[available].iloc[0].values
        prediction = self.predictor.predict(features, feature_names=available)

        # P2: Add regime info to prediction for dashboard
        try:
            regime = self.regime_detector.predict(
                pd.read_sql_query(
                    "SELECT close, volume FROM prices ORDER BY date DESC LIMIT 60",
                    self.conn,
                ).assign(vix=lambda df: pd.read_sql_query(
                    "SELECT vix FROM macro ORDER BY date DESC LIMIT 60", self.conn
                ).get("vix", 18.0))
            )
            prediction["regime"] = regime
        except Exception:
            prediction["regime"] = self.results.get("step_10", {}).get("result", {}).get("regime", "")

        # P2: Flag if ensemble was used
        prediction["ensemble_used"] = self.predictor.ensemble is not None and self.predictor.use_ensemble

        # Store prediction
        self.conn.execute(
            """INSERT OR REPLACE INTO predictions
               (date, direction, confidence, factors, predicted_at)
               VALUES (?,?,?,?,?)""",
            (self.today, prediction["scale_label"], prediction["confidence"],
             json.dumps(prediction["probabilities"]),
             datetime.now().isoformat()),
        )
        self.conn.commit()

        # Update dashboard state
        macro = self.fallback.get_macro_fred() if self.fallback else {}
        tech_row = self.conn.execute(
            "SELECT rsi_14, macd, sma_20, sma_50 FROM technicals WHERE date = ?",
            (self.today,),
        ).fetchone()
        indicators = {}
        if tech_row:
            indicators = {"rsi_14": tech_row[0], "macd": tech_row[1],
                          "sma_20": tech_row[2], "sma_50": tech_row[3]}
        indicators["vix"] = macro.get("vix")

        write_spy_state(prediction=prediction, indicators=indicators)
        logger.info(f"Prediction: {prediction['scale_label']} "
                    f"({prediction['confidence']:.0f}%)"
                    f"{' [LOW CONVICTION]' if prediction.get('is_low_conviction') else ''}")
        return prediction

    def _step12_report(self) -> dict:
        """Step 12: Generate LLM daily report."""
        # Gather context
        tech_row = self.conn.execute(
            "SELECT * FROM technicals WHERE date = ?", (self.today,),
        ).fetchone()
        sent_row = self.conn.execute(
            "SELECT * FROM daily_sentiment WHERE date = ?", (self.today,),
        ).fetchone()
        macro_row = self.conn.execute(
            "SELECT * FROM macro WHERE date = ?", (self.today,),
        ).fetchone()
        pred_row = self.conn.execute(
            "SELECT direction, confidence FROM predictions WHERE date = ?",
            (self.today,),
        ).fetchone()

        context = {
            "prediction": {"direction": pred_row[0] if pred_row else "N/A",
                           "confidence": pred_row[1] if pred_row else 0},
            "technicals": dict(tech_row) if tech_row else {},
            "sentiment": dict(sent_row) if sent_row else {},
            "macro": dict(macro_row) if macro_row else {},
        }

        report = self.reporter.generate_report(
            context, llm_available=self.llm.llm_available,
        )

        # Store report text
        self.conn.execute(
            "UPDATE predictions SET report_text = ? WHERE date = ?",
            (report, self.today),
        )
        self.conn.commit()
        logger.info(f"Report generated ({len(report)} chars)")
        return {"length": len(report)}

    def _step13_alerts(self) -> dict:
        """Step 13: Send alerts (Telegram + email)."""
        try:
            from src.pipeline.alerts import send_alerts
            pred_row = self.conn.execute(
                "SELECT direction, confidence, report_text FROM predictions WHERE date = ?",
                (self.today,),
            ).fetchone()
            if not pred_row:
                return {"sent": False, "reason": "no prediction"}

            alert_data = {
                "date": self.today,
                "direction": pred_row[0],
                "confidence": pred_row[1],
                "report": pred_row[2] or "",
            }
            result = send_alerts(self.config, alert_data)
            return result
        except Exception as e:
            logger.warning(f"Alert sending failed: {e}")
            return {"sent": False, "error": str(e)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Daily Pipeline Orchestrator")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM steps")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        config = {}

    pipeline = DailyPipeline(config)
    results = pipeline.run(skip_llm=args.skip_llm)

    # Save results
    out_path = f"data/pipeline_results_{pipeline.today}.json"
    try:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {out_path}")
    except Exception as e:
        logger.warning(f"Could not save results: {e}")


if __name__ == "__main__":
    main()
