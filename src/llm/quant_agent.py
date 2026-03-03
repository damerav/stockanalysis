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
        }

    # ── LLM Communication ──────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        tool_descriptions = """You are a quantitative finance AI assistant for the SPY/SPX Predictor platform.
You have access to these tools (call them by responding with a JSON block):

TOOLS:
1. query_database(sql, db="spy") — Run read-only SQL on spy.db, analytics.duckdb, or news.db
   Tables in spy.db: prices, technicals, macro, daily_sentiment, predictions, options_analytics,
   options_chain, intraday_features, intraday_bars, earnings_calendar, fed_communications, news, users
   Tables in news.db: raw_articles (headline, source, published_at, category, sentiment_compound, finbert_score)
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
        """Call Ollama chat API."""
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
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return None

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
        """Run read-only SQL query against spy.db, news.db, or analytics.duckdb."""
        sql_clean = sql.strip().upper()
        if not sql_clean.startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed."}

        try:
            if db == "news":
                db_path = self.config.get("news_pipeline", {}).get("db_path", "./data/news.db")
                conn = sqlite3.connect(db_path)
            elif db == "duckdb":
                try:
                    import duckdb
                    conn = duckdb.connect("./data/analytics.duckdb", read_only=True)
                except Exception as e:
                    return {"error": f"DuckDB unavailable: {e}"}
            else:
                conn = get_connection(self.config)

            df = pd.read_sql_query(sql, conn)
            conn.close()

            # Limit output size
            if len(df) > 50:
                return {
                    "rows": len(df),
                    "columns": list(df.columns),
                    "data": df.head(50).to_dict(orient="records"),
                    "truncated": True,
                    "note": f"Showing first 50 of {len(df)} rows",
                }
            return {
                "rows": len(df),
                "columns": list(df.columns),
                "data": df.to_dict(orient="records"),
            }
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

            conn = get_connection(self.config)
            fv = build_feature_vector(conn, config=self.config)
            if fv is None or fv.empty:
                conn.close()
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
                conn.close()
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

            conn.close()
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

            conn = get_connection(self.config)
            fv = build_feature_vector(conn, config=self.config)
            if fv is None or fv.empty:
                conn.close()
                return {"error": "No feature data"}

            feature_cols = get_feature_columns()
            available = [c for c in feature_cols if c in fv.columns]
            target = get_target(fv)

            predictor = SPYPredictor(self.config)
            result = predictor.train(fv[available], target,
                                     feature_names=available, force_save=False)
            conn.close()

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
        """Get news sentiment summary from news.db."""
        try:
            db_path = self.config.get("news_pipeline", {}).get("db_path", "./data/news.db")
            if not os.path.exists(db_path):
                return {"error": "news.db not found"}

            conn = sqlite3.connect(db_path)
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            where = f"WHERE published_at >= '{cutoff}'"
            if category:
                where += f" AND category = '{category}'"

            # Overall stats
            stats = pd.read_sql_query(
                f"SELECT COUNT(*) as count, "
                f"AVG(sentiment_compound) as avg_sentiment, "
                f"COUNT(DISTINCT source) as sources "
                f"FROM raw_articles {where}", conn
            )

            # Top headlines by absolute sentiment
            top = pd.read_sql_query(
                f"SELECT headline, source, category, sentiment_compound, published_at "
                f"FROM raw_articles {where} "
                f"ORDER BY ABS(sentiment_compound) DESC LIMIT 10", conn
            )

            # Category breakdown
            cats = pd.read_sql_query(
                f"SELECT category, COUNT(*) as count, AVG(sentiment_compound) as avg_sent "
                f"FROM raw_articles {where} AND category IS NOT NULL "
                f"GROUP BY category ORDER BY count DESC", conn
            )

            conn.close()

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
            conn = get_connection(self.config)
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            # Get price + macro data for regime detection
            df = pd.read_sql_query(
                f"SELECT p.date, p.close, p.volume, m.vix "
                f"FROM prices p LEFT JOIN macro m ON p.date = m.date "
                f"WHERE p.date >= '{cutoff}' ORDER BY p.date", conn
            )
            conn.close()

            if df.empty or len(df) < 10:
                return {"error": "Not enough data for regime detection"}

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
