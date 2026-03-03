"""Quant Agent — Conversational AI assistant for the Stock Analysis Platform.

Uses DeepSeek R1 70B via Ollama for natural language interaction with the
platform's databases, models, and research tools.

Admin-only. Can read data, run backtests, retrain models, and modify config.
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.data.init_db import get_connection, load_config
from src.data.db_router import get_router

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:70b"


class QuantAgent:
    """Conversational quant agent with tool-use capabilities."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        llm_cfg = self.config.get("llm", {})
        self.base_url = llm_cfg.get("base_url", OLLAMA_BASE)
        self.model = llm_cfg.get("model", OLLAMA_MODEL)
        self.data_dir = "./data"
        self.history: list[dict] = []

        # Tool registry
        self.tools = {
            "query_database": self._tool_query_database,
            "get_prediction_state": self._tool_get_prediction_state,
            "get_model_info": self._tool_get_model_info,
            "get_feature_importance": self._tool_get_feature_importance,
            "run_backtest": self._tool_run_backtest,
            "retrain_model": self._tool_retrain_model,
            "get_news_summary": self._tool_get_news_summary,
            "get_regime_history": self._tool_get_regime_history,
            "get_pipeline_status": self._tool_get_pipeline_status,
            "assess_news_risk": self._tool_assess_news_risk,
            "generate_alpha_hypothesis": self._tool_generate_alpha_hypothesis,
            "compare_strategies": self._tool_compare_strategies,
            "analyze_feature_correlations": self._tool_analyze_feature_correlations,
            "explain_regime": self._tool_explain_regime,
            "search_similar_news": self._tool_search_similar_news,
        }

    # ── LLM Communication ──────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        tool_descriptions = """You are a quantitative finance AI assistant for the SPY/SPX Predictor platform.
You have access to these tools (call them by responding with a JSON block):

TOOLS:
1. query_database(sql, db="spy") — Run read-only SQL on PostgreSQL (primary) or SQLite (fallback).
   All tables: prices, technicals, macro, daily_sentiment, predictions, options_analytics,
   options_chain, intraday_features, intraday_bars, earnings_calendar, fed_communications, news, users,
   raw_articles (headline, source, published_at, category, sentiment_compound), finbert_cache
   Use db="news" for raw_articles/finbert_cache queries.
   Example: {"tool": "query_database", "args": {"sql": "SELECT date, close FROM prices ORDER BY date DESC LIMIT 5"}}

2. get_prediction_state() — Get current prediction from spy_state.json (direction, confidence, probabilities, regime)
   Example: {"tool": "get_prediction_state", "args": {}}

3. get_model_info() — Get model metadata (feature names, accuracy, training date)
   Example: {"tool": "get_model_info", "args": {}}

4. get_feature_importance(top_n=15) — Get top feature importances from the current model
   Example: {"tool": "get_feature_importance", "args": {"top_n": 10}}

5. run_backtest(days=60) — Run walk-forward backtest on recent N days, returns accuracy metrics
   Example: {"tool": "run_backtest", "args": {"days": 60}}

6. retrain_model() — Retrain the XGBoost model with latest data (GPU). Returns accuracy metrics.
   Example: {"tool": "retrain_model", "args": {}}

7. get_news_summary(days=1, category=null) — Get news sentiment summary
   Categories: markets, forex, bonds, commodities, crypto, centralbanks, economic, ipo, derivatives, fintech, regulation, institutional, analysis
   Example: {"tool": "get_news_summary", "args": {"days": 3, "category": "markets"}}

8. get_regime_history(days=30) — Get HMM regime detection history
   Example: {"tool": "get_regime_history", "args": {"days": 30}}

9. get_pipeline_status() — Get latest pipeline run results
   Example: {"tool": "get_pipeline_status", "args": {}}

10. assess_news_risk(days=1, category=null) — Use DeepSeek to score news risk 1-5 (inspired by FinRL-DeepSeek).
    Returns per-article risk scores + aggregate risk level. Complements sentiment with a risk dimension.
    Example: {"tool": "assess_news_risk", "args": {"days": 1, "category": "markets"}}

11. generate_alpha_hypothesis(context=null) — Propose new alpha factor ideas based on current regime, model performance,
    and feature gaps. Inspired by RD-Agent's hypothesis-backtest loop. Returns hypotheses with rationale.
    Example: {"tool": "generate_alpha_hypothesis", "args": {"context": "model accuracy dropped to 48%"}}

12. compare_strategies(days=60) — Run multiple strategy variants (full ensemble, XGB-only, binary-only, regime-filtered)
    and compare Sharpe, max drawdown, win rate, and profit factor side-by-side.
    Example: {"tool": "compare_strategies", "args": {"days": 90}}

13. analyze_feature_correlations(threshold=0.8) — Compute feature correlation matrix, detect multicollinearity,
    and suggest features to drop. Returns top correlated pairs and VIF analysis.
    Example: {"tool": "analyze_feature_correlations", "args": {"threshold": 0.7}}

14. explain_regime() — Use DeepSeek to explain WHY we're in the current HMM regime based on indicators,
    recent price action, VIX, volume, and news sentiment. Goes beyond just showing the state label.
    Example: {"tool": "explain_regime", "args": {}}

15. search_similar_news(query, limit=10, category=null, days_back=30) — Semantic vector search for similar articles
    using pgvector FinBERT embeddings. Finds historically similar news patterns.
    Example: {"tool": "search_similar_news", "args": {"query": "Fed rate hike inflation", "limit": 5}}

RULES:
- To use a tool, respond with EXACTLY one JSON block: {"tool": "name", "args": {...}}
- After receiving tool results, analyze them and give a clear answer.
- For SQL queries, use SELECT only (no INSERT/UPDATE/DELETE).
- When showing numbers, round to 2-3 decimal places.
- Be concise and quantitative. Use specific numbers, not vague language.
- If asked to retrain, confirm the results and compare with previous accuracy.
- You can chain multiple tool calls across turns to answer complex questions.
"""
        return tool_descriptions

    def chat(self, user_message: str) -> tuple[str, Optional[dict]]:
        """Process a user message. Returns (response_text, chart_data_or_none).

        May make multiple LLM calls if tool use is needed.
        """
        self.history.append({"role": "user", "content": user_message})

        # Build messages for Ollama
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        # Keep last 20 messages for context
        messages.extend(self.history[-20:])

        # LLM call loop (max 3 tool calls per turn)
        chart_data = None
        for _ in range(4):
            response_text = self._call_llm(messages)
            if response_text is None:
                err = "LLM is unavailable. Make sure Ollama is running with DeepSeek R1."
                self.history.append({"role": "assistant", "content": err})
                return err, None

            # Check if response contains a tool call
            tool_call = self._extract_tool_call(response_text)
            if tool_call:
                tool_name = tool_call.get("tool", "")
                tool_args = tool_call.get("args", {})

                if tool_name in self.tools:
                    # Execute tool
                    tool_result = self.tools[tool_name](**tool_args)

                    # Check if tool returned chart data
                    if isinstance(tool_result, dict) and "chart" in tool_result:
                        chart_data = tool_result.pop("chart")

                    result_str = json.dumps(tool_result, default=str, indent=2)
                    # Add tool interaction to history
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"[Tool Result for {tool_name}]:\n{result_str}\n\nNow analyze this result and respond to the user."})
                    continue  # Loop back for LLM to interpret results
                else:
                    response_text = f"Unknown tool: {tool_name}. Available: {', '.join(self.tools.keys())}"

            # No tool call — this is the final response
            # Strip <think>...</think> blocks from DeepSeek R1
            response_text = self._strip_thinking(response_text)
            self.history.append({"role": "assistant", "content": response_text})
            return response_text, chart_data

        # Exceeded tool call limit
        fallback = "I've made several tool calls but couldn't fully resolve your question. Could you rephrase?"
        self.history.append({"role": "assistant", "content": fallback})
        return fallback, chart_data

    def _call_llm(self, messages: list[dict]) -> Optional[str]:
        """Call Ollama chat API. Auto-starts Ollama if not running."""
        # Try the call first
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 2048},
                },
                timeout=300,
            )
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "")
            logger.warning(f"Ollama returned {resp.status_code}")
            return None
        except requests.ConnectionError:
            # Ollama not running — try to start it
            logger.info("Ollama not responding, attempting to start...")
            if self._start_ollama():
                try:
                    resp = requests.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "stream": False,
                            "options": {"temperature": 0.3, "num_predict": 2048},
                        },
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        return resp.json().get("message", {}).get("content", "")
                except Exception as e:
                    logger.warning(f"Ollama call failed after start: {e}")
            return None
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return None

    def _start_ollama(self) -> bool:
        """Attempt to start Ollama if not running."""
        import subprocess
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.error("Ollama binary not found")
            return False
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False

        # Wait up to 15s for Ollama to become responsive
        for _ in range(15):
            time.sleep(1)
            try:
                resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
                if resp.status_code == 200:
                    logger.info("Ollama started successfully")
                    return True
            except Exception:
                pass
        logger.error("Ollama did not respond within 15s")
        return False

    def _extract_tool_call(self, text: str) -> Optional[dict]:
        """Extract a tool call JSON from LLM response."""
        import re
        # Look for JSON blocks with "tool" key
        patterns = [
            r'```json\s*(\{[^`]*?"tool"[^`]*?\})\s*```',
            r'```\s*(\{[^`]*?"tool"[^`]*?\})\s*```',
            r'(\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}\s*\})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        return None

    def _strip_thinking(self, text: str) -> str:
        """Remove <think>...</think> blocks from DeepSeek R1 output."""
        import re
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # ── Tool Implementations ──────────────────────────────────────

    def _tool_query_database(self, sql: str, db: str = "spy") -> dict:
        """Run read-only SQL query against PostgreSQL (primary) or SQLite (fallback)."""
        sql_clean = sql.strip().upper()
        if not sql_clean.startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed."}

        try:
            if db == "news":
                # Try PostgreSQL first for news (raw_articles table)
                try:
                    from src.data.db_router import get_router
                    router = get_router(self.config)
                    if router.using_postgres:
                        df = router.query(sql)
                        if not df.empty:
                            if len(df) > 50:
                                return {
                                    "rows": len(df), "columns": list(df.columns),
                                    "data": df.head(50).to_dict(orient="records"),
                                    "truncated": True, "note": f"Showing first 50 of {len(df)} rows",
                                }
                            return {"rows": len(df), "columns": list(df.columns),
                                    "data": df.to_dict(orient="records")}
                except Exception:
                    pass
                # Fallback to news.db SQLite
                db_path = self.config.get("news_pipeline", {}).get("db_path", "./data/news.db")
                conn = sqlite3.connect(db_path)
            else:
                # Try PostgreSQL first
                try:
                    from src.data.db_router import get_router
                    router = get_router(self.config)
                    if router.using_postgres:
                        df = router.query(sql)
                        if len(df) > 50:
                            return {
                                "rows": len(df), "columns": list(df.columns),
                                "data": df.head(50).to_dict(orient="records"),
                                "truncated": True, "note": f"Showing first 50 of {len(df)} rows",
                            }
                        return {"rows": len(df), "columns": list(df.columns),
                                "data": df.to_dict(orient="records")}
                except Exception:
                    pass
                conn = get_connection(self.config)

            df = pd.read_sql_query(sql, conn)
            conn.close()

            if len(df) > 50:
                return {
                    "rows": len(df), "columns": list(df.columns),
                    "data": df.head(50).to_dict(orient="records"),
                    "truncated": True, "note": f"Showing first 50 of {len(df)} rows",
                }
            return {"rows": len(df), "columns": list(df.columns),
                    "data": df.to_dict(orient="records")}
        except Exception as e:
            return {"error": str(e)}

    def _tool_get_prediction_state(self) -> dict:
        """Get current prediction from spy_state.json."""
        try:
            state_path = os.path.join(self.data_dir, "spy_state.json")
            with open(state_path) as f:
                state = json.load(f)
            return {
                "prediction": state.get("prediction", {}),
                "indicators": state.get("indicators", {}),
                "updated_at": state.get("updated_at", ""),
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_get_model_info(self) -> dict:
        """Get model metadata from the latest model file."""
        try:
            import glob
            meta_files = sorted(glob.glob("./models/xgb_spy_*_meta.json"))
            if not meta_files:
                return {"error": "No model metadata found"}
            latest = meta_files[-1]
            with open(latest) as f:
                meta = json.load(f)
            # Also get model file stats
            model_file = latest.replace("_meta.json", ".json")
            model_size = os.path.getsize(model_file) if os.path.exists(model_file) else 0
            return {
                "meta_file": os.path.basename(latest),
                "feature_count": len(meta.get("feature_names", [])),
                "feature_names": meta.get("feature_names", []),
                "conformal_quantile": meta.get("conformal_quantile"),
                "conformal_significance": meta.get("conformal_significance"),
                "model_size_kb": round(model_size / 1024, 1),
                "training_date": os.path.basename(latest).split("_")[2].replace("meta.json", ""),
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_get_feature_importance(self, top_n: int = 15) -> dict:
        """Get feature importances from the current model."""
        try:
            import xgboost as xgb
            import glob

            model_files = sorted(glob.glob("./models/xgb_spy_*.json"))
            # Filter out meta/binary/conformal files
            model_files = [f for f in model_files if "_meta" not in f
                          and "_binary" not in f and "_conformal" not in f]
            if not model_files:
                return {"error": "No model found"}

            model = xgb.XGBClassifier()
            model.load_model(model_files[-1])

            # Load feature names
            meta_file = model_files[-1].replace(".json", "_meta.json")
            feature_names = [f"f{i}" for i in range(model.n_features_in_)]
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    meta = json.load(f)
                    feature_names = meta.get("feature_names", feature_names)

            importances = model.feature_importances_
            pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
            top = pairs[:top_n]

            return {
                "total_features": len(feature_names),
                "top_features": [{"name": n, "importance": round(float(v), 4)} for n, v in top],
                "top_feature_pct": round(sum(v for _, v in top) / max(sum(importances), 1e-10) * 100, 1),
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_run_backtest(self, days: int = 60) -> dict:
        """Run walk-forward backtest on recent N days."""
        try:
            from src.data.features import build_feature_vector, get_feature_columns, get_target
            from src.model.trainer import SPYPredictor

            router = get_router(self.config)
            conn = router.get_pg() or router.get_sqlite()
            fv = build_feature_vector(conn, config=self.config)
            if fv is None or fv.empty:
                return {"error": "No feature data available"}

            feature_cols = get_feature_columns()
            available = [c for c in feature_cols if c in fv.columns]
            target = get_target(fv)

            # Use last N days as test set
            days = min(days, len(fv) - 100)  # need at least 100 for training
            test_start = len(fv) - days

            predictor = SPYPredictor(self.config)
            # Train on data before test period
            train_fv = fv.iloc[:test_start]
            train_target = target.iloc[:test_start]
            result = predictor.train(train_fv[available], train_target,
                                     feature_names=available, force_save=False)

            if result.get("error"):
                return {"error": f"Training failed: {result['error']}"}

            # Predict on test period
            test_fv = fv.iloc[test_start:]
            test_target = target.iloc[test_start:]
            correct = 0
            total = 0
            predictions = []

            for i in range(len(test_fv)):
                if pd.isna(test_target.iloc[i]):
                    continue
                features = test_fv[available].iloc[i].values
                pred = predictor.predict(features, feature_names=available)
                actual_dir = int(test_target.iloc[i])
                pred_dir = 1 if "BULLISH" in pred.get("direction", "") else (-1 if "BEARISH" in pred.get("direction", "") else 0)
                is_correct = pred_dir == actual_dir
                correct += int(is_correct)
                total += 1
                predictions.append({
                    "date": test_fv.iloc[i]["date"],
                    "predicted": pred.get("direction"),
                    "actual": {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}.get(actual_dir),
                    "correct": is_correct,
                    "confidence": pred.get("confidence", 0),
                })

            accuracy = correct / max(total, 1)

            return {
                "days_tested": total,
                "accuracy": round(accuracy, 3),
                "correct": correct,
                "total": total,
                "train_accuracy": result.get("accuracy", 0),
                "recent_10": predictions[-10:] if predictions else [],
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_retrain_model(self) -> dict:
        """Retrain the XGBoost model with latest data."""
        try:
            from src.data.features import build_feature_vector, get_feature_columns, get_target
            from src.model.trainer import SPYPredictor

            router = get_router(self.config)
            conn = router.get_pg() or router.get_sqlite()
            fv = build_feature_vector(conn, config=self.config)
            if fv is None or fv.empty:
                return {"error": "No feature data"}

            feature_cols = get_feature_columns()
            available = [c for c in feature_cols if c in fv.columns]
            target = get_target(fv)

            predictor = SPYPredictor(self.config)
            result = predictor.train(fv[available], target,
                                     feature_names=available, force_save=False)

            return {
                "val_accuracy": result.get("accuracy", 0),
                "test_accuracy": result.get("test_accuracy", 0),
                "binary_up_accuracy": result.get("binary_up_accuracy", 0),
                "features_kept": result.get("n_features_kept", 0),
                "gated": result.get("gated", False),
                "gate_reason": result.get("gate_reason", ""),
                "model_path": result.get("model_path", ""),
                "adaptive_window": result.get("adaptive_window", 0),
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_get_news_summary(self, days: int = 1, category: str = None) -> dict:
        """Get news sentiment summary via PostgreSQL router."""
        try:
            router = get_router(self.config)
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            where = f"WHERE published_at >= '{cutoff}'"
            if category:
                where += f" AND category = '{category}'"

            # Overall stats
            stats = router.query(
                f"SELECT COUNT(*) as count, "
                f"AVG(sentiment_compound) as avg_sentiment, "
                f"COUNT(DISTINCT source) as sources "
                f"FROM raw_articles {where}"
            )

            # Top headlines by absolute sentiment
            top = router.query(
                f"SELECT headline, source, category, sentiment_compound, published_at "
                f"FROM raw_articles {where} "
                f"ORDER BY ABS(sentiment_compound) DESC LIMIT 10"
            )

            # Category breakdown
            cats = router.query(
                f"SELECT category, COUNT(*) as count, AVG(sentiment_compound) as avg_sent "
                f"FROM raw_articles {where} AND category IS NOT NULL "
                f"GROUP BY category ORDER BY count DESC"
            )

            return {
                "period_days": days,
                "filter_category": category,
                "total_articles": int(stats.iloc[0]["count"]) if not stats.empty else 0,
                "avg_sentiment": round(float(stats.iloc[0]["avg_sentiment"] or 0), 4),
                "unique_sources": int(stats.iloc[0]["sources"]) if not stats.empty else 0,
                "top_headlines": top.to_dict(orient="records") if not top.empty else [],
                "category_breakdown": cats.to_dict(orient="records") if not cats.empty else [],
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_get_regime_history(self, days: int = 30) -> dict:
        """Get HMM regime detection history."""
        try:
            router = get_router(self.config)
            # Fetch extra days to account for weekends/holidays + HMM needs history
            fetch_days = max(days * 2, 90)
            cutoff = (datetime.now() - timedelta(days=fetch_days)).strftime("%Y-%m-%d")

            # Get price + macro data for regime detection
            df = router.query(
                f"SELECT p.date, p.close, p.volume, m.vix "
                f"FROM prices p LEFT JOIN macro m ON p.date = m.date "
                f"WHERE p.date >= '{cutoff}' ORDER BY p.date"
            )

            if df.empty or len(df) < 5:
                return {"error": f"Not enough data for regime detection (got {len(df)} rows)"}

            from src.model.regime import HMMRegimeDetector
            detector = HMMRegimeDetector()

            # Detect regime for each day (rolling window)
            regimes = []
            for i in range(max(0, len(df) - 30), len(df)):
                window = df.iloc[max(0, i - 59):i + 1]
                if len(window) >= 10:
                    regime = detector.predict(window)
                    regimes.append({"date": df.iloc[i]["date"], "regime": regime})

            # Count regime distribution
            regime_counts = {}
            for r in regimes:
                regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1

            return {
                "days": days,
                "regime_history": regimes[-15:],  # last 15 entries
                "regime_distribution": regime_counts,
                "current_regime": regimes[-1]["regime"] if regimes else "unknown",
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_get_pipeline_status(self) -> dict:
        """Get latest pipeline run results."""
        try:
            import glob
            result_files = sorted(glob.glob("./data/pipeline_results_*.json"))
            if not result_files:
                return {"error": "No pipeline results found"}

            latest = result_files[-1]
            with open(latest) as f:
                results = json.load(f)

            # Summarize step statuses
            steps = {}
            for key, val in results.items():
                if key.startswith("step_"):
                    steps[key] = {
                        "status": val.get("status", "unknown"),
                        "elapsed": val.get("elapsed", 0),
                    }

            return {
                "date": results.get("date", ""),
                "total_elapsed": results.get("total_elapsed", 0),
                "steps": steps,
                "file": os.path.basename(latest),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── New Tools (Alpha-Agent / FinRL-DeepSeek inspired) ─────────

    def _tool_assess_news_risk(self, days: int = 1, category: str = None) -> dict:
        """Use DeepSeek to score news risk 1-5 per article (FinRL-DeepSeek pattern)."""
        try:
            router = get_router(self.config)
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            where = f"WHERE published_at >= '{cutoff}'"
            if category:
                where += f" AND category = '{category}'"

            articles = router.query(
                f"SELECT headline, source, category, sentiment_compound, published_at "
                f"FROM raw_articles {where} "
                f"ORDER BY published_at DESC LIMIT 20"
            )

            if articles.empty:
                return {"error": "No articles found for the given period"}

            # Build few-shot risk assessment prompt for DeepSeek
            headlines = articles["headline"].tolist()
            numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))

            risk_prompt = [
                {"role": "system", "content": (
                    "You are a financial risk assessment expert. "
                    "Score each news headline for market risk on a scale of 1-5:\n"
                    "1=very low risk, 2=low risk, 3=moderate risk, "
                    "4=high risk, 5=very high risk.\n"
                    "Consider: geopolitical impact, market volatility potential, "
                    "systemic risk, sector contagion, and urgency.\n"
                    "Respond with ONLY comma-separated integers, one per headline."
                )},
                {"role": "user", "content": (
                    "1. Apple reports record Q4 earnings beating estimates\n"
                    "2. Federal Reserve signals emergency rate hike amid inflation surge\n"
                    "3. Microsoft announces new AI partnership"
                )},
                {"role": "assistant", "content": "1, 5, 2"},
                {"role": "user", "content": numbered},
            ]

            risk_response = self._call_llm(risk_prompt)
            risk_response = self._strip_thinking(risk_response or "")

            # Parse risk scores
            risk_scores = []
            for s in risk_response.replace("\n", ",").split(","):
                s = s.strip()
                try:
                    val = int(s)
                    if 1 <= val <= 5:
                        risk_scores.append(val)
                except ValueError:
                    continue

            # Pad or truncate to match article count
            while len(risk_scores) < len(headlines):
                risk_scores.append(3)  # default moderate
            risk_scores = risk_scores[:len(headlines)]

            # Build results
            assessed = []
            for i, row in articles.iterrows():
                idx = articles.index.get_loc(i)
                assessed.append({
                    "headline": row["headline"],
                    "source": row["source"],
                    "category": row["category"],
                    "sentiment": round(float(row["sentiment_compound"] or 0), 3),
                    "risk_score": risk_scores[idx],
                    "risk_label": {1: "very_low", 2: "low", 3: "moderate",
                                   4: "high", 5: "very_high"}.get(risk_scores[idx], "moderate"),
                })

            avg_risk = round(np.mean(risk_scores), 2)
            high_risk_count = sum(1 for r in risk_scores if r >= 4)

            return {
                "period_days": days,
                "articles_assessed": len(assessed),
                "avg_risk_score": avg_risk,
                "risk_level": "HIGH" if avg_risk >= 3.5 else ("MODERATE" if avg_risk >= 2.5 else "LOW"),
                "high_risk_articles": high_risk_count,
                "assessed_articles": assessed[:15],
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_generate_alpha_hypothesis(self, context: str = None) -> dict:
        """Generate alpha factor hypotheses using DeepSeek (RD-Agent pattern)."""
        try:
            # Gather current system state for context
            state_info = self._tool_get_prediction_state()
            model_info = self._tool_get_model_info()
            regime_info = self._tool_get_regime_history(days=14)

            current_regime = regime_info.get("current_regime", "unknown")
            feature_names = model_info.get("feature_names", [])
            prediction = state_info.get("prediction", {})
            indicators = state_info.get("indicators", {})

            system_ctx = (
                "You are a quantitative researcher proposing new alpha factors for a "
                "SPY/SPX prediction model. The model uses XGBoost + BiLSTM + LightGBM stacking ensemble.\n\n"
                f"Current features ({len(feature_names)}): {', '.join(feature_names[:20])}{'...' if len(feature_names) > 20 else ''}\n"
                f"Current regime: {current_regime}\n"
                f"Current prediction: {prediction.get('direction', 'N/A')} "
                f"(confidence: {prediction.get('confidence', 'N/A')})\n"
                f"VIX: {indicators.get('vix', 'N/A')}, RSI: {indicators.get('rsi_14', 'N/A')}\n"
            )
            if context:
                system_ctx += f"Additional context: {context}\n"

            hypothesis_prompt = [
                {"role": "system", "content": system_ctx},
                {"role": "user", "content": (
                    "Propose 3-5 new alpha factor hypotheses that could improve prediction accuracy. "
                    "For each hypothesis, provide:\n"
                    "1. Factor name (snake_case)\n"
                    "2. Formula or computation description\n"
                    "3. Rationale (why it should have predictive power)\n"
                    "4. Data source (what data is needed)\n"
                    "5. Expected signal (how it relates to SPY direction)\n\n"
                    "Focus on factors NOT already in the model. Consider:\n"
                    "- Cross-asset signals (bonds, gold, dollar index)\n"
                    "- Microstructure (bid-ask spread dynamics, order flow imbalance)\n"
                    "- Volatility surface features (skew, term structure)\n"
                    "- Intermarket momentum divergences\n"
                    "- Sentiment regime shifts (not just level, but rate of change)\n\n"
                    "Format as JSON array: [{\"name\": ..., \"formula\": ..., \"rationale\": ..., "
                    "\"data_source\": ..., \"expected_signal\": ...}, ...]"
                )},
            ]

            response = self._call_llm(hypothesis_prompt)
            response = self._strip_thinking(response or "")

            # Try to parse JSON from response
            hypotheses = []
            try:
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    hypotheses = json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError):
                # If JSON parsing fails, return raw text
                return {
                    "current_regime": current_regime,
                    "current_features": len(feature_names),
                    "raw_hypotheses": response,
                    "parse_note": "Could not parse structured JSON — raw LLM output included",
                }

            return {
                "current_regime": current_regime,
                "current_features": len(feature_names),
                "hypotheses_count": len(hypotheses),
                "hypotheses": hypotheses,
                "note": "These are AI-generated hypotheses. Validate via backtest before adding to production.",
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_compare_strategies(self, days: int = 60) -> dict:
        """Compare multiple strategy variants side-by-side."""
        try:
            from src.data.features import build_feature_vector, get_feature_columns, get_target

            router = get_router(self.config)
            conn = router.get_pg() or router.get_sqlite()
            fv = build_feature_vector(conn, config=self.config)
            if fv is None or fv.empty:
                return {"error": "No feature data available"}

            feature_cols = get_feature_columns()
            available = [c for c in feature_cols if c in fv.columns]
            target = get_target(fv)

            days = min(days, len(fv) - 120)
            test_start = len(fv) - days

            test_fv = fv.iloc[test_start:]
            test_target = target.iloc[test_start:]

            # Get price returns for P&L calculation
            test_prices = test_fv[["date", "close"]].copy() if "close" in test_fv.columns else None

            # Strategy 1: Full model predictions
            from src.model.trainer import SPYPredictor
            predictor = SPYPredictor(self.config)
            predictor.load_latest()

            strategies = {}

            # --- Strategy A: Full ensemble (current model) ---
            preds_a, actuals_a = [], []
            for i in range(len(test_fv)):
                if pd.isna(test_target.iloc[i]):
                    continue
                features = test_fv[available].iloc[i].values
                pred = predictor.predict(features, feature_names=available)
                pred_dir = 1 if "BULLISH" in pred.get("direction", "") else (-1 if "BEARISH" in pred.get("direction", "") else 0)
                actual = int(test_target.iloc[i])
                preds_a.append(pred_dir)
                actuals_a.append(actual)

            strategies["full_model"] = self._calc_strategy_metrics(preds_a, actuals_a, "Full Ensemble")

            # --- Strategy B: Always bullish baseline ---
            preds_b = [1] * len(actuals_a)
            strategies["always_bullish"] = self._calc_strategy_metrics(preds_b, actuals_a, "Always Bullish")

            # --- Strategy C: Regime-filtered (only trade in bull/low_vol regimes) ---
            try:
                from src.model.regime import HMMRegimeDetector
                detector = HMMRegimeDetector()
                preds_c, actuals_c = [], []
                for i in range(len(test_fv)):
                    if pd.isna(test_target.iloc[i]):
                        continue
                    # Simple regime check: use VIX as proxy
                    vix_val = test_fv.iloc[i].get("vix", 20)
                    if vix_val and float(vix_val) > 30:
                        preds_c.append(0)  # sit out in high vol
                    else:
                        features = test_fv[available].iloc[i].values
                        pred = predictor.predict(features, feature_names=available)
                        pred_dir = 1 if "BULLISH" in pred.get("direction", "") else (-1 if "BEARISH" in pred.get("direction", "") else 0)
                        preds_c.append(pred_dir)
                    actuals_c.append(int(test_target.iloc[i]))
                strategies["regime_filtered"] = self._calc_strategy_metrics(preds_c, actuals_c, "Regime-Filtered (VIX<30)")
            except Exception:
                pass

            # --- Strategy D: High-confidence only (>60%) ---
            preds_d, actuals_d = [], []
            for i in range(len(test_fv)):
                if pd.isna(test_target.iloc[i]):
                    continue
                features = test_fv[available].iloc[i].values
                pred = predictor.predict(features, feature_names=available)
                conf = pred.get("confidence", 0)
                if conf >= 0.6:
                    pred_dir = 1 if "BULLISH" in pred.get("direction", "") else (-1 if "BEARISH" in pred.get("direction", "") else 0)
                    preds_d.append(pred_dir)
                else:
                    preds_d.append(0)  # sit out
                actuals_d.append(int(test_target.iloc[i]))
            strategies["high_confidence"] = self._calc_strategy_metrics(preds_d, actuals_d, "High Confidence (>60%)")

            # Build chart data for comparison
            chart_data = {
                "data": [
                    {
                        "type": "bar",
                        "x": list(strategies.keys()),
                        "y": [s["accuracy"] for s in strategies.values()],
                        "name": "Accuracy",
                        "marker": {"color": "#2962FF"},
                    }
                ],
                "layout": {"title": "Strategy Comparison — Accuracy", "yaxis": {"title": "Accuracy", "tickformat": ".1%"}},
            }

            return {
                "days_tested": days,
                "strategies": strategies,
                "chart": chart_data,
                "best_strategy": max(strategies, key=lambda k: strategies[k]["accuracy"]),
            }
        except Exception as e:
            return {"error": str(e)}

    def _calc_strategy_metrics(self, predictions: list, actuals: list, name: str) -> dict:
        """Calculate strategy performance metrics."""
        if not predictions or not actuals:
            return {"name": name, "accuracy": 0, "trades": 0}

        total = len(predictions)
        trades = sum(1 for p in predictions if p != 0)
        correct = sum(1 for p, a in zip(predictions, actuals) if p == a and p != 0)
        wrong = sum(1 for p, a in zip(predictions, actuals) if p != a and p != 0)

        accuracy = correct / max(trades, 1)
        win_rate = correct / max(trades, 1)

        # Simulated P&L (1 unit per trade)
        pnl = sum(p * a for p, a in zip(predictions, actuals))

        # Max drawdown (cumulative)
        cum = np.cumsum([p * a for p, a in zip(predictions, actuals)])
        peak = np.maximum.accumulate(cum) if len(cum) > 0 else np.array([0])
        drawdown = peak - cum
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        # Profit factor
        gross_profit = sum(p * a for p, a in zip(predictions, actuals) if p * a > 0)
        gross_loss = abs(sum(p * a for p, a in zip(predictions, actuals) if p * a < 0))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        return {
            "name": name,
            "accuracy": round(accuracy, 3),
            "win_rate": round(win_rate, 3),
            "trades": trades,
            "skipped": total - trades,
            "pnl_units": round(float(pnl), 2),
            "max_drawdown": round(max_dd, 2),
            "profit_factor": round(profit_factor, 2),
        }

    def _tool_analyze_feature_correlations(self, threshold: float = 0.8) -> dict:
        """Analyze feature correlations and detect multicollinearity."""
        try:
            from src.data.features import build_feature_vector, get_feature_columns

            router = get_router(self.config)
            conn = router.get_pg() or router.get_sqlite()
            fv = build_feature_vector(conn, config=self.config)

            if fv is None or fv.empty:
                return {"error": "No feature data available"}

            feature_cols = get_feature_columns()
            available = [c for c in feature_cols if c in fv.columns]
            feat_df = fv[available].copy()
            # Drop columns that are entirely NaN (e.g. microstructure, options)
            feat_df = feat_df.dropna(axis=1, how="all")
            # Drop columns with >50% NaN
            thresh = int(len(feat_df) * 0.5)
            feat_df = feat_df.dropna(axis=1, thresh=thresh)
            available = list(feat_df.columns)
            feat_df = feat_df.dropna()

            if feat_df.empty or len(feat_df) < 30:
                return {"error": "Not enough data for correlation analysis"}

            # Correlation matrix
            corr = feat_df.corr()

            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(available)):
                for j in range(i + 1, len(available)):
                    c = abs(corr.iloc[i, j])
                    if c >= threshold:
                        high_corr_pairs.append({
                            "feature_1": available[i],
                            "feature_2": available[j],
                            "correlation": round(float(corr.iloc[i, j]), 3),
                            "abs_correlation": round(float(c), 3),
                        })

            high_corr_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

            # VIF (Variance Inflation Factor) for top features
            vif_results = []
            try:
                from numpy.linalg import LinAlgError
                X = feat_df.values
                # Standardize
                X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
                corr_matrix = np.corrcoef(X_std.T)
                try:
                    inv_corr = np.linalg.inv(corr_matrix)
                    vifs = np.diag(inv_corr)
                    for idx, (feat, vif) in enumerate(zip(available, vifs)):
                        vif_results.append({"feature": feat, "vif": round(float(vif), 2)})
                    vif_results.sort(key=lambda x: x["vif"], reverse=True)
                except LinAlgError:
                    vif_results = [{"note": "Correlation matrix is singular — severe multicollinearity detected"}]
            except Exception:
                vif_results = [{"note": "VIF calculation failed"}]

            # Suggest drops: for each high-corr pair, suggest dropping the one with lower importance
            drop_suggestions = []
            seen = set()
            for pair in high_corr_pairs[:10]:
                f1, f2 = pair["feature_1"], pair["feature_2"]
                if f1 not in seen and f2 not in seen:
                    # Suggest dropping the one that appears more often in high-corr pairs
                    count_f1 = sum(1 for p in high_corr_pairs if f1 in (p["feature_1"], p["feature_2"]))
                    count_f2 = sum(1 for p in high_corr_pairs if f2 in (p["feature_1"], p["feature_2"]))
                    drop = f1 if count_f1 > count_f2 else f2
                    drop_suggestions.append(drop)
                    seen.add(drop)

            return {
                "total_features": len(available),
                "threshold": threshold,
                "high_corr_pairs": high_corr_pairs[:15],
                "high_corr_count": len(high_corr_pairs),
                "vif_top10": vif_results[:10],
                "drop_suggestions": drop_suggestions,
                "note": "Features with VIF > 10 indicate severe multicollinearity. Consider dropping suggested features.",
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_explain_regime(self) -> dict:
        """Use DeepSeek to explain the current market regime with full context."""
        try:
            # Gather all relevant data
            state = self._tool_get_prediction_state()
            regime_info = self._tool_get_regime_history(days=14)
            news = self._tool_get_news_summary(days=2)

            indicators = state.get("indicators", {})
            prediction = state.get("prediction", {})
            current_regime = regime_info.get("current_regime", "unknown")
            regime_dist = regime_info.get("regime_distribution", {})
            avg_sentiment = news.get("avg_sentiment", 0)
            article_count = news.get("total_articles", 0)

            # Get recent price action
            try:
                router_local = get_router(self.config)
                prices = router_local.query(
                    "SELECT date, close, volume FROM prices ORDER BY date DESC LIMIT 10"
                )
                recent_prices = prices.to_dict(orient="records") if not prices.empty else []
                if len(recent_prices) >= 2:
                    pct_change_1d = round((recent_prices[0]["close"] / recent_prices[1]["close"] - 1) * 100, 2)
                else:
                    pct_change_1d = 0
                if len(recent_prices) >= 5:
                    pct_change_5d = round((recent_prices[0]["close"] / recent_prices[4]["close"] - 1) * 100, 2)
                else:
                    pct_change_5d = 0
            except Exception:
                recent_prices = []
                pct_change_1d = 0
                pct_change_5d = 0

            # Build explanation prompt
            explain_prompt = [
                {"role": "system", "content": (
                    "You are a senior market strategist. Explain the current market regime "
                    "in 3-5 concise paragraphs. Be specific with numbers. "
                    "Cover: 1) What regime we're in and why, 2) Key drivers, "
                    "3) What to watch for regime change, 4) Trading implications."
                )},
                {"role": "user", "content": (
                    f"Current HMM Regime: {current_regime}\n"
                    f"Regime distribution (14d): {json.dumps(regime_dist)}\n"
                    f"SPY 1-day change: {pct_change_1d}%, 5-day change: {pct_change_5d}%\n"
                    f"VIX: {indicators.get('vix', 'N/A')}, VIX change: {indicators.get('vix_change', 'N/A')}\n"
                    f"RSI(14): {indicators.get('rsi_14', 'N/A')}\n"
                    f"MACD: {indicators.get('macd', 'N/A')}\n"
                    f"ATR(14): {indicators.get('atr_14', 'N/A')}\n"
                    f"Volume ratio: {indicators.get('volume_ratio', 'N/A')}\n"
                    f"News sentiment (2d avg): {avg_sentiment} ({article_count} articles)\n"
                    f"Model prediction: {prediction.get('direction', 'N/A')} "
                    f"(confidence: {prediction.get('confidence', 'N/A')})\n"
                    f"Probabilities: {json.dumps(prediction.get('probabilities', {}))}\n\n"
                    f"Explain this regime and its implications."
                )},
            ]

            explanation = self._call_llm(explain_prompt)
            explanation = self._strip_thinking(explanation or "No explanation available.")

            return {
                "current_regime": current_regime,
                "regime_distribution_14d": regime_dist,
                "key_indicators": {
                    "vix": indicators.get("vix"),
                    "rsi_14": indicators.get("rsi_14"),
                    "macd": indicators.get("macd"),
                    "volume_ratio": indicators.get("volume_ratio"),
                    "spy_1d_pct": pct_change_1d,
                    "spy_5d_pct": pct_change_5d,
                    "news_sentiment": avg_sentiment,
                },
                "explanation": explanation,
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_search_similar_news(self, query: str, limit: int = 10,
                                   category: str = None, days_back: int = 30) -> dict:
        """Search for semantically similar news articles using pgvector embeddings."""
        try:
            from src.data.db_router import get_router
            router = get_router(self.config)

            if not router.using_postgres:
                return {"error": "Vector search requires PostgreSQL+pgvector (not available)"}

            # Check if embeddings exist
            count_df = router.query(
                "SELECT COUNT(*) as cnt FROM raw_articles WHERE embedding IS NOT NULL"
            )
            embed_count = int(count_df.iloc[0]["cnt"]) if not count_df.empty else 0
            if embed_count == 0:
                return {"error": "No embeddings stored yet. Run embedding pipeline first."}

            # Generate embedding for the query using FinBERT
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch

                tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                model = AutoModel.from_pretrained("ProsusAI/finbert")
                inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
            except Exception as e:
                return {"error": f"FinBERT embedding failed: {e}"}

            # Vector similarity search
            results = router.vector_search(
                embedding, limit=limit, category=category, days_back=days_back
            )

            if results.empty:
                return {"query": query, "results": [], "total": 0}

            return {
                "query": query,
                "total_embeddings": embed_count,
                "results": results.to_dict(orient="records"),
                "total": len(results),
            }
        except Exception as e:
            return {"error": str(e)}
