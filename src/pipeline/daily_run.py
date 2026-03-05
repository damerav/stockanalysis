"""7A. Daily Pipeline Orchestrator — 13-step sequential pipeline at 4:30 PM ET.

Usage:
    python -m src.pipeline.daily_run
    python -m src.pipeline.daily_run --config config.yaml --skip-llm
"""

import argparse
import json
import logging
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
from src.data.db_router import get_router
from src.data.news_fetcher import NewsFetcher
from src.data.news_features import NewsFeatureProcessor
from src.data.geopolitical_features import compute_daily_finbert_features

logger = logging.getLogger(__name__)


class DailyPipeline:
    """13-step daily pipeline for SPY prediction and data refresh."""

    def __init__(self, config: dict):
        self.config = config
        self.conn = None
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
        # Create a FRESH DbRouter for this pipeline run (thread-safe).
        # Don't use the singleton get_router() — it may hold connections
        # from a different thread (e.g. when triggered from Streamlit UI).
        from src.data.db_router import DbRouter
        try:
            self.router = DbRouter(self.config)
            logger.info(f"Pipeline DbRouter: {'PostgreSQL' if self.router.using_postgres else 'SQLite'}")
        except Exception as e:
            logger.warning(f"Database router unavailable: {e}")
            self.router = None

        # Legacy self.conn — only used as fallback if router is None
        self.conn = get_connection(self.config)

        api_key = self.config.get("polygon", {}).get("api_key", "")
        if api_key and api_key != "YOUR_POLYGON_KEY":
            self.polygon = PolygonFetcher(api_key)
        self.fallback = FallbackFetcher(config=self.config)
        self.llm = LLMAnalyzer(self.config)
        self.predictor = SPYPredictor(self.config)
        self.reporter = DailyReporter(self.config)
        # P2 components
        self.feature_store = FeatureStore(self.config)
        self.regime_detector = HMMRegimeDetector()

    # ── Thread-safe DB helpers ────────────────────────────────────────
    def _db_execute(self, sql: str, params: tuple = None):
        """Execute a write query via router (PostgreSQL primary, SQLite fallback)."""
        if self.router:
            self.router.execute(sql, params)
        elif self.conn:
            self.conn.execute(sql, params or ())
            self.conn.commit()

    def _db_query(self, sql: str, params: tuple = None) -> pd.DataFrame:
        """Execute a read query via router, return DataFrame."""
        if self.router:
            return self.router.query(sql, params)
        elif self.conn:
            return pd.read_sql_query(sql, self.conn, params=params or ())
        return pd.DataFrame()

    def _db_fetchone(self, sql: str, params: tuple = None):
        """Fetch a single row via router, return tuple or None."""
        df = self._db_query(sql, params)
        if not df.empty:
            return tuple(df.iloc[0])
        return None

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

        if self.router:
            self.router.close()
        elif self.conn:
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

    def _get_conn(self):
        """Return the DbRouter for external functions.
        External functions (evaluate_past_prediction, store_earnings, etc.)
        now accept DbRouter instances and route queries to PostgreSQL.
        Functions like build_feature_vector/store_technicals already call
        get_router() internally for PostgreSQL routing."""
        if self.router:
            return self.router
        return self.conn

    def _step1_evaluate(self) -> dict:
        """Step 1: Evaluate yesterday's prediction accuracy."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        result = evaluate_past_prediction(self._get_conn(), yesterday)
        if result:
            logger.info(f"Yesterday's prediction: {result['predicted']} → "
                        f"Actual: {result['actual']} — "
                        f"{'✓' if result['correct'] else '✗'} "
                        f"(cumulative: {result['cumulative_accuracy']:.1%})")
        else:
            logger.info("No prediction to evaluate for yesterday")
        return result or {"no_prediction": True}

    def _step2_prices(self) -> dict:
        """Step 2: Fetch today's daily prices. Enhancement 26: Writes to DuckDB."""
        def _write_price(row):
            if self.router:
                self.router.write_analytics(
                    """INSERT OR REPLACE INTO prices
                       (date, open, high, low, close, volume)
                       VALUES (?,?,?,?,?,?)""",
                    (row["date"], row["open"], row["high"],
                     row["low"], row["close"], row["volume"]),
                )
            else:
                self._db_execute(
                    """INSERT OR REPLACE INTO prices
                       (date, open, high, low, close, volume)
                       VALUES (?,?,?,?,?,?)""",
                    (row["date"], row["open"], row["high"],
                     row["low"], row["close"], row["volume"]),
                )

        if self.polygon:
            try:
                df = self.polygon.get_daily_bars("SPY", self.today, self.today)
                if not df.empty:
                    for _, row in df.iterrows():
                        _write_price(row)
                    return {"source": "polygon", "rows": len(df)}
            except Exception as e:
                logger.warning(f"Polygon price fetch failed: {e}")

        # Fallback to yfinance
        df = self.fallback.get_daily_bars_yf("SPY", days=5)
        if not df.empty:
            today_row = df[df["date"] == self.today]
            if not today_row.empty:
                for _, row in today_row.iterrows():
                    _write_price(row)
                return {"source": "yfinance", "rows": len(today_row)}
        return {"source": "none", "rows": 0}

    def _step3_news(self) -> dict:
        """Step 3: Fetch news from expanded sources (Finnhub + company news + 45+ categorized RSS feeds)."""
        # Use expanded NewsFetcher (news.db) for broad coverage
        expanded_count = 0
        category_stats = {}
        try:
            nf = NewsFetcher(self.config)
            expanded_count = nf.fetch_all()
            category_stats = nf.get_category_sentiment_summary(days=1)

            # Bridge today's articles from news.db → PostgreSQL news table
            recent = nf.get_recent(days=1)
            bridged = 0
            for a in recent:
                pub = a.get("published_at", "")
                date_str = pub[:10] if pub else self.today
                try:
                    self._db_execute(
                        """INSERT OR IGNORE INTO news
                           (date, source, headline, summary, url, fetched_at)
                           VALUES (?,?,?,?,?,?)""",
                        (date_str, a.get("source", ""), a.get("headline", ""),
                         a.get("summary", ""), a.get("url", ""),
                         a.get("fetched_at", "")),
                    )
                    bridged += 1
                except Exception:
                    pass
            # Sync sentiment to PostgreSQL for Quant Agent queries
            nf.backfill_sentiment()
            nf.sync_sentiment_to_postgres()
            nf.close()
            logger.info(f"Expanded news: {expanded_count} new, {bridged} bridged to DB")
            if category_stats:
                cats = ", ".join(f"{k}={v['count']}" for k, v in category_stats.items())
                logger.info(f"Category breakdown: {cats}")
        except Exception as e:
            logger.warning(f"Expanded news fetch failed: {e}")

        # Also fetch via legacy path as fallback
        articles = []
        finnhub_articles = self.fallback.get_news_finnhub()
        articles.extend(finnhub_articles)
        rss_articles = self.fallback.get_news_rss()
        articles.extend(rss_articles)

        for a in articles:
            try:
                self._db_execute(
                    """INSERT OR IGNORE INTO news
                       (date, source, headline, summary, url, fetched_at)
                       VALUES (?,?,?,?,?,?)""",
                    (a["date"], a["source"], a["headline"],
                     a["summary"], a.get("url", ""), a.get("fetched_at", "")),
                )
            except Exception:
                pass
        logger.info(f"Legacy news: {len(articles)} articles")
        return {"articles": len(articles), "expanded_articles": expanded_count,
                "categories": category_stats}

    def _step4_sentiment(self) -> dict:
        """Step 4: Sentiment analysis — combines LLM analysis with expanded news.db corpus."""
        # --- Part A: Process expanded news.db articles via VADER/FinBERT ---
        expanded_sentiment = {}
        try:
            processor = NewsFeatureProcessor(self.config)
            article_df = processor.process_articles()
            if not article_df.empty:
                # Filter to today's articles
                today_articles = article_df[article_df["date"] == self.today]
                if not today_articles.empty:
                    expanded_sentiment = {
                        "article_count": len(today_articles),
                        "avg_sentiment": float(today_articles["sentiment_compound"].mean()),
                        "max_sentiment": float(today_articles["sentiment_compound"].max()),
                        "min_sentiment": float(today_articles["sentiment_compound"].min()),
                        "sentiment_std": float(today_articles["sentiment_compound"].std()) if len(today_articles) > 1 else 0,
                        "positive_ratio": float((today_articles["sentiment_compound"] > 0.05).mean()),
                        "negative_ratio": float((today_articles["sentiment_compound"] < -0.05).mean()),
                    }
                    # Compute per-category sentiment from ticker matching
                    macro_arts = today_articles[today_articles["ticker"].isin(["MACRO", "MARKET", "SPY", "VIX"])]
                    tech_arts = today_articles[today_articles["ticker"].isin(["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AMD", "INTC"])]
                    expanded_sentiment["macro_sentiment_expanded"] = float(macro_arts["sentiment_compound"].mean()) if not macro_arts.empty else 0
                    expanded_sentiment["tech_sentiment_expanded"] = float(tech_arts["sentiment_compound"].mean()) if not tech_arts.empty else 0
                    logger.info(f"Expanded news sentiment: {len(today_articles)} articles, "
                                f"avg={expanded_sentiment['avg_sentiment']:.3f}")

                # Also build and store daily features for the news_features table
                daily_df = processor.build_daily_features()
                processor.store_features(daily_df)
            processor.close()
        except Exception as e:
            logger.warning(f"Expanded news processing failed (non-fatal): {e}")

        # --- Part B: LLM sentiment analysis on news table (original path) ---
        if not self.llm.llm_available:
            logger.info("LLM unavailable — using expanded news sentiment")
            if expanded_sentiment:
                # Use expanded corpus sentiment as primary
                sentiment = {
                    "score": expanded_sentiment.get("avg_sentiment", 0),
                    "confidence": min(expanded_sentiment.get("article_count", 0) / 50, 1.0),
                    "article_count": expanded_sentiment.get("article_count", 0),
                    "positive_ratio": expanded_sentiment.get("positive_ratio", 0),
                    "negative_ratio": expanded_sentiment.get("negative_ratio", 0),
                    "neutral_ratio": 1 - expanded_sentiment.get("positive_ratio", 0) - expanded_sentiment.get("negative_ratio", 0),
                    "macro_sentiment": expanded_sentiment.get("macro_sentiment_expanded", 0),
                    "earnings_sentiment": 0,
                    "geopolitical_sentiment": 0,
                    "technical_sentiment": expanded_sentiment.get("tech_sentiment_expanded", 0),
                    "sentiment_dispersion": expanded_sentiment.get("sentiment_std", 0),
                    "sentiment_velocity": 0,
                }
            else:
                sentiment = self.llm.get_neutral_sentiment()
            self._store_sentiment(sentiment)
            return {"skipped_llm": True, "expanded_articles": expanded_sentiment.get("article_count", 0), "sentiment": sentiment}

        # Get today's articles from DB
        news_df = self._db_query(
            "SELECT headline, summary FROM news WHERE date = ? LIMIT 200",
            (self.today,),
        )
        articles = [{"headline": r["headline"], "summary": r["summary"]} for _, r in news_df.iterrows()]

        if not articles:
            logger.info("No articles in DB — using expanded sentiment")
            if expanded_sentiment:
                sentiment = {
                    "score": expanded_sentiment.get("avg_sentiment", 0),
                    "confidence": min(expanded_sentiment.get("article_count", 0) / 50, 1.0),
                    "article_count": expanded_sentiment.get("article_count", 0),
                    "positive_ratio": expanded_sentiment.get("positive_ratio", 0),
                    "negative_ratio": expanded_sentiment.get("negative_ratio", 0),
                    "neutral_ratio": 1 - expanded_sentiment.get("positive_ratio", 0) - expanded_sentiment.get("negative_ratio", 0),
                    "macro_sentiment": expanded_sentiment.get("macro_sentiment_expanded", 0),
                    "earnings_sentiment": 0,
                    "geopolitical_sentiment": 0,
                    "technical_sentiment": expanded_sentiment.get("tech_sentiment_expanded", 0),
                    "sentiment_dispersion": expanded_sentiment.get("sentiment_std", 0),
                    "sentiment_velocity": 0,
                }
            else:
                sentiment = self.llm.get_neutral_sentiment()
            self._store_sentiment(sentiment)
            return {"articles": 0, "expanded_articles": expanded_sentiment.get("article_count", 0), "sentiment": sentiment}

        logger.info(f"Analysing sentiment for {len(articles)} articles...")
        results = self.llm.analyze_sentiment(articles)
        daily = self.llm.aggregate_daily_sentiment(results)

        # --- Part C: Blend LLM sentiment with expanded corpus sentiment ---
        if expanded_sentiment and expanded_sentiment.get("article_count", 0) > 10:
            # Weighted blend: LLM is deeper analysis, expanded is broader coverage
            llm_weight = 0.6
            exp_weight = 0.4
            daily["score"] = daily["score"] * llm_weight + expanded_sentiment["avg_sentiment"] * exp_weight
            daily["article_count"] = daily.get("article_count", len(articles)) + expanded_sentiment["article_count"]
            daily["sentiment_dispersion"] = max(
                daily.get("sentiment_dispersion", 0),
                expanded_sentiment.get("sentiment_std", 0)
            )
            logger.info(f"Blended sentiment: LLM({llm_weight}) + expanded({exp_weight}) = {daily['score']:.3f}")

        # Compute sentiment velocity (change vs yesterday)
        try:
            prev = self._db_fetchone(
                "SELECT score FROM daily_sentiment WHERE date < ? ORDER BY date DESC LIMIT 1",
                (self.today,),
            )
            if prev and prev[0] is not None:
                daily["sentiment_velocity"] = round(daily["score"] - prev[0], 4)
            else:
                daily["sentiment_velocity"] = 0
        except Exception:
            daily["sentiment_velocity"] = daily.get("sentiment_velocity", 0)

        self._store_sentiment(daily)

        # --- Part D: FinBERT sentiment (runs on recent articles for higher accuracy) ---
        finbert_result = {}
        try:
            fb_daily = compute_daily_finbert_features(self.config, days_back=3)
            if not fb_daily.empty:
                today_fb = fb_daily[fb_daily["date"] == self.today]
                if not today_fb.empty:
                    finbert_result = today_fb.iloc[0].to_dict()
                    logger.info(f"FinBERT sentiment: score={finbert_result.get('finbert_score', 0):.3f}")
        except Exception as e:
            logger.warning(f"FinBERT processing failed (non-fatal): {e}")

        return {"articles": len(articles), "expanded_articles": expanded_sentiment.get("article_count", 0),
                "sentiment": daily, "finbert": finbert_result}

    def _store_sentiment(self, sentiment: dict):
        self._db_execute(
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

        if self.router:
            self.router.write_analytics(
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
        else:
            self._db_execute(
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
        return macro


    def _step6_options_chain(self) -> dict:
        """Step 6: Fetch options chain snapshot. Enhancement 26: Writes to DuckDB."""
        if not self.polygon:
            return {"skipped": True, "reason": "no polygon key"}
        try:
            chain = self.polygon.get_options_chain("SPY")
            if chain.empty:
                return {"rows": 0}
            if self.router:
                for _, row in chain.iterrows():
                    self.router.execute(
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
            else:
                for _, row in chain.iterrows():
                    self._db_execute(
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
            self._db_execute(
                """INSERT OR REPLACE INTO options_analytics
                   (date, put_call_ratio, max_pain, iv_skew, gex,
                    vanna_exposure, charm_exposure, zero_dte_pcr)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (self.today, analytics.get("put_call_ratio"),
                 analytics.get("max_pain"), analytics.get("iv_skew"),
                 analytics.get("gex"), analytics.get("vanna_exposure"),
                 analytics.get("charm_exposure"), analytics.get("zero_dte_pcr")),
            )
            return analytics
        except Exception as e:
            logger.warning(f"Options analytics failed: {e}")
            return {"error": str(e)}

    def _step8_technicals(self) -> dict:
        """Step 8: Compute daily technicals. Enhancement 26: Reads prices from DuckDB."""
        if self.router:
            df = self.router.read_analytics(
                "SELECT date, open, high, low, close, volume FROM prices ORDER BY date"
            )
        else:
            df = self._db_query(
                "SELECT date, open, high, low, close, volume FROM prices ORDER BY date"
            )
        if df.empty:
            return {"rows": 0}
        tech_df = compute_all_technicals(df, self.config)
        store_technicals(self._get_conn(), tech_df, self.config)
        return {"rows": len(tech_df)}

    def _step9_intraday(self) -> dict:
        """Step 9: Build intraday features. Enhancement 26: Reads intraday_bars from DuckDB."""
        # Check if we have intraday bars for today
        bar_count = 0
        if self.router:
            df_cnt = self.router.read_analytics(
                "SELECT COUNT(*) as cnt FROM intraday_bars WHERE timestamp LIKE ?",
                (f"{self.today}%",),
            )
            bar_count = int(df_cnt.iloc[0]["cnt"]) if not df_cnt.empty else 0
        else:
            row = self._db_fetchone(
                "SELECT COUNT(*) FROM intraday_bars WHERE timestamp LIKE ?",
                (f"{self.today}%",),
            )
            bar_count = row[0] if row else 0

        if bar_count == 0:
            # No intraday data — store zeros
            self._db_execute(
                """INSERT OR REPLACE INTO intraday_features
                   (date, vwap_spread, intraday_momentum, intraday_range, volume_ratio)
                   VALUES (?,?,?,?,?)""",
                (self.today, 0, 0, 0, 1.0),
            )
            return {"bars": 0, "features": "default"}

        # Compute from intraday bars
        if self.router:
            bars = self.router.read_analytics(
                "SELECT * FROM intraday_bars WHERE timestamp LIKE ? ORDER BY timestamp",
                (f"{self.today}%",),
            )
        else:
            bars = self._db_query(
                "SELECT * FROM intraday_bars WHERE timestamp LIKE ? ORDER BY timestamp",
                (f"{self.today}%",),
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

        self._db_execute(
            """INSERT OR REPLACE INTO intraday_features
               (date, vwap_spread, intraday_momentum, intraday_range, volume_ratio)
               VALUES (?,?,?,?,?)""",
            (self.today, round(vwap_spread, 6), round(momentum, 6),
             round(intraday_range, 6), round(volume_ratio, 4)),
        )
        return {"bars": bar_count, "features": "computed"}

    def _step95_earnings(self) -> dict:
        """Step 9.5: Fetch and store earnings calendar for mega-caps."""
        try:
            earnings = fetch_earnings_yf()
            if earnings:
                store_earnings(self._get_conn(), earnings)
                logger.info(f"Stored {len(earnings)} earnings dates")
            return {"earnings_fetched": len(earnings)}
        except Exception as e:
            logger.warning(f"Earnings calendar fetch failed (non-fatal): {e}")
            return {"error": str(e)}

    def _step96_fed_comms(self) -> dict:
        """Step 9.6: Fetch and score latest Fed communications."""
        try:
            llm_analyzer = self.llm if (self.llm and self.llm.llm_available) else None
            results = update_fed_communications(self._get_conn(), llm_analyzer)
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
            date_df = self._db_query("SELECT date FROM prices ORDER BY date")
            all_dates = date_df["date"].tolist() if not date_df.empty else []
            conn_for_build = self._get_conn()
            fv = self.feature_store.get_features(
                feature_cols, all_dates=all_dates,
                build_fn=lambda missing: build_feature_vector(conn_for_build, config=self.config),
            )
        except Exception as e:
            logger.warning(f"Feature store failed, falling back to direct build: {e}")

        if fv is None or fv.empty:
            fv = build_feature_vector(self._get_conn(), config=self.config)

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

        fv = build_feature_vector(self._get_conn(), date=self.today, config=self.config)
        if fv is None or fv.empty:
            return {"error": "no features for today"}

        feature_cols = get_feature_columns()
        available = [c for c in feature_cols if c in fv.columns]

        # Align features to model's trained feature set if metadata exists
        if self.predictor.trained_feature_names:
            available = [c for c in self.predictor.trained_feature_names if c in fv.columns]
        elif hasattr(self.predictor.model, 'n_features_in_') and self.predictor.model.n_features_in_ != len(available):
            # Auto-retrain with curated feature set
            logger.info(f"Feature mismatch ({self.predictor.model.n_features_in_} vs {len(available)}) — auto-retraining...")
            from src.data.features import get_target
            full_fv = build_feature_vector(self._get_conn(), config=self.config)
            if full_fv is not None and not full_fv.empty:
                train_cols = [c for c in feature_cols if c in full_fv.columns]
                target = get_target(full_fv)
                result = self.predictor.train(full_fv[train_cols], target,
                                              feature_names=train_cols, force_save=False)
                if result.get("error"):
                    return {"error": f"Auto-retrain failed: {result['error']}"}
                logger.info(f"Auto-retrained: accuracy={result.get('accuracy', 0):.3f}")
                # Use the filtered feature names from training (may be fewer than train_cols)
                available = self.predictor.trained_feature_names or train_cols

        features = fv[available].iloc[0].values
        prediction = self.predictor.predict(features, feature_names=available)

        # P2: Add regime info to prediction for dashboard
        try:
            if self.router:
                price_df = self.router.read_analytics(
                    "SELECT close, volume FROM prices ORDER BY date DESC LIMIT 60"
                )
                macro_df = self.router.read_analytics(
                    "SELECT vix FROM macro ORDER BY date DESC LIMIT 60"
                )
            else:
                price_df = self._db_query(
                    "SELECT close, volume FROM prices ORDER BY date DESC LIMIT 60"
                )
                macro_df = self._db_query(
                    "SELECT vix FROM macro ORDER BY date DESC LIMIT 60"
                )
            price_df["vix"] = macro_df.get("vix", 18.0)
            regime = self.regime_detector.predict(price_df)
            prediction["regime"] = regime
        except Exception:
            prediction["regime"] = self.results.get("step_10", {}).get("result", {}).get("regime", "")

        # P2: Flag if ensemble was used
        prediction["ensemble_used"] = self.predictor.ensemble is not None and self.predictor.use_ensemble

        # Store prediction
        self._db_execute(
            """INSERT OR REPLACE INTO predictions
               (date, direction, confidence, factors, predicted_at)
               VALUES (?,?,?,?,?)""",
            (self.today, prediction["scale_label"], prediction["confidence"],
             json.dumps(prediction["probabilities"]),
             datetime.now().isoformat()),
        )

        # Update dashboard state
        macro = self.fallback.get_macro_fred() if self.fallback else {}
        if self.router:
            tech_df = self.router.read_analytics(
                "SELECT rsi_14, macd, sma_20, sma_50 FROM technicals WHERE date = ?",
                (self.today,),
            )
            tech_row = tech_df.iloc[0] if not tech_df.empty else None
        else:
            tech_row = self._db_fetchone(
                "SELECT rsi_14, macd, sma_20, sma_50 FROM technicals WHERE date = ?",
                (self.today,),
            )
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
        tech_row = self._db_fetchone(
            "SELECT * FROM technicals WHERE date = ?", (self.today,),
        )
        sent_row = self._db_fetchone(
            "SELECT * FROM daily_sentiment WHERE date = ?", (self.today,),
        )
        macro_row = self._db_fetchone(
            "SELECT * FROM macro WHERE date = ?", (self.today,),
        )
        pred_row = self._db_fetchone(
            "SELECT direction, confidence FROM predictions WHERE date = ?",
            (self.today,),
        )

        context = {
            "prediction": {"direction": pred_row[0] if pred_row else "N/A",
                           "confidence": pred_row[1] if pred_row else 0},
            "technicals": {},
            "sentiment": {},
            "macro": {},
        }
        # _db_fetchone returns tuples, not dict — use _db_query for dict access
        for key, sql in [("technicals", "SELECT * FROM technicals WHERE date = ?"),
                         ("sentiment", "SELECT * FROM daily_sentiment WHERE date = ?"),
                         ("macro", "SELECT * FROM macro WHERE date = ?")]:
            df = self._db_query(sql, (self.today,))
            if not df.empty:
                context[key] = df.iloc[0].to_dict()

        report = self.reporter.generate_report(
            context, llm_available=self.llm.llm_available,
        )

        # Store report text
        self._db_execute(
            "UPDATE predictions SET report_text = ? WHERE date = ?",
            (report, self.today),
        )
        logger.info(f"Report generated ({len(report)} chars)")
        return {"length": len(report)}

    def _step13_alerts(self) -> dict:
        """Step 13: Send alerts (Telegram + email)."""
        try:
            from src.pipeline.alerts import send_alerts
            pred_row = self._db_fetchone(
                "SELECT direction, confidence, report_text FROM predictions WHERE date = ?",
                (self.today,),
            )
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
