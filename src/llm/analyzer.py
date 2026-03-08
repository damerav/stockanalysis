"""Phase 2: LLM Health Check + Sentiment Analyzer.

Handles Ollama lifecycle: check → start → model availability → auto-download → validate.
Graceful degradation: pipeline NEVER aborts due to LLM unavailability.

P1 enhancement: FinBERT fast-path for all articles, DeepSeek deep-path for top 5.
"""

import time
import json
import logging
import subprocess
import requests
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
OLLAMA_STARTUP_TIMEOUT = 15  # seconds
INFERENCE_TIMEOUT = 120  # seconds (first load can take 60s+ for 70B model)
DOWNLOAD_LOG_INTERVAL = 5  # log every 5% progress


class LLMAnalyzer:
    """Manages Ollama LLM lifecycle and provides sentiment analysis."""

    def __init__(self, config: dict = None):
        config = config or {}
        llm_config = config.get("llm", {})
        self.base_url = llm_config.get("base_url", DEFAULT_BASE_URL)
        self.model = llm_config.get("model", DEFAULT_MODEL)
        self.temperature = llm_config.get("temperature", 0.3)
        self.llm_available = False
        self.llm_degraded = False
        # P1: FinBERT fast-path
        self.finbert = None
        self.finbert_available = False
        self._init_finbert()
        # Cache-aware FinBERT: initialize router for cache reads/writes
        self._router = None
        try:
            from src.data.db_router import get_router
            self._router = get_router(config)
        except Exception:
            pass  # Router unavailable — will fall back to uncached scoring
        # Cache-aware FinBERT: initialize router for cache reads/writes
        self._router = None
        try:
            from src.data.db_router import get_router
            self._router = get_router(config)
        except Exception:
            pass  # Router unavailable — will fall back to uncached scoring

    # --- P1: FinBERT fast-path ---

    def _init_finbert(self):
        """Load FinBERT model for fast financial sentiment scoring."""
        try:
            from transformers import pipeline as hf_pipeline
            self.finbert = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0,  # GPU
                truncation=True,
                max_length=512,
            )
            self.finbert_available = True
            logger.info("FinBERT loaded on GPU for fast sentiment")
        except Exception as e:
            logger.info(f"FinBERT not available (will use LLM only): {e}")
            self.finbert_available = False

    def _finbert_score(self, text: str) -> dict:
        """Score a single text with FinBERT. Returns {score, confidence, label}."""
        if not self.finbert_available or not self.finbert:
            return {"score": 0.0, "confidence": 0, "label": "neutral"}
        try:
            result = self.finbert(text[:512])[0]
            label = result["label"].lower()
            conf = int(result["score"] * 100)
            score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            raw_score = score_map.get(label, 0.0) * result["score"]
            return {"score": round(raw_score, 4), "confidence": conf, "label": label}
        except Exception:
            return {"score": 0.0, "confidence": 0, "label": "neutral"}

    # --- 2A: Ollama Health Check ---

    def _check_ollama(self) -> bool:
        """Check if Ollama process is running and responsive."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                logger.info("Ollama is running and responsive")
                return True
        except requests.ConnectionError:
            pass
        except Exception as e:
            logger.warning(f"Ollama health check error: {e}")
        return False

    def _start_ollama(self) -> bool:
        """Attempt to start Ollama if not running."""
        logger.info("Attempting to start Ollama...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.error("Ollama binary not found. Is it installed?")
            return False
        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            return False

        # Wait for Ollama to become responsive
        start = time.time()
        while time.time() - start < OLLAMA_STARTUP_TIMEOUT:
            if self._check_ollama():
                logger.info("Ollama started successfully")
                return True
            time.sleep(1)

        logger.error(f"Ollama did not respond within {OLLAMA_STARTUP_TIMEOUT}s")
        return False

    # --- 2B: Model Availability Check ---

    def _model_available(self) -> bool:
        """Check if the configured model is downloaded."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if resp.status_code != 200:
                return False
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Check exact match and partial match (e.g. "deepseek-r1:70b" in "deepseek-r1:70b")
            for m in models:
                if self.model in m or m in self.model:
                    logger.info(f"Model '{self.model}' is available")
                    return True
            logger.info(f"Model '{self.model}' not found. Available: {models}")
            return False
        except Exception as e:
            logger.warning(f"Model check failed: {e}")
            return False

    # --- 2C: Auto-Download with Progress ---

    def _pull_model(self) -> bool:
        """Download the model with streaming progress logging."""
        logger.info(f"Downloading model '{self.model}' (this may take a while)...")
        try:
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                stream=True,
                timeout=7200,  # 2 hour timeout for large models
            )
            if resp.status_code != 200:
                logger.error(f"Model pull failed with status {resp.status_code}")
                return False

            last_pct = -1
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = data.get("status", "")
                    total = data.get("total", 0)
                    completed = data.get("completed", 0)

                    if total > 0:
                        pct = int((completed / total) * 100)
                        # Log every DOWNLOAD_LOG_INTERVAL percent
                        if pct >= last_pct + DOWNLOAD_LOG_INTERVAL:
                            last_pct = pct
                            total_gb = total / (1024 ** 3)
                            done_gb = completed / (1024 ** 3)
                            logger.info(
                                f"Downloading {self.model}: {pct}% "
                                f"({done_gb:.1f}GB / {total_gb:.1f}GB)"
                            )
                    elif status:
                        logger.info(f"Pull status: {status}")

                    if status == "success":
                        logger.info(f"Model '{self.model}' downloaded successfully")
                        return True
                except json.JSONDecodeError:
                    continue

            # Check if model is now available
            return self._model_available()

        except requests.Timeout:
            logger.error("Model download timed out after 2 hours")
            return False
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False

    # --- 2D: Inference Validation ---

    def _validate_inference(self) -> bool:
        """Run a quick test inference to verify the model works."""
        logger.info("Running inference validation...")
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Reply OK"}],
                    "stream": False,
                },
                timeout=INFERENCE_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                if content.strip():
                    logger.info(f"Inference validation passed: '{content[:50]}'")
                    return True
                else:
                    logger.warning("Inference returned empty response")
                    return False
            else:
                logger.warning(f"Inference validation failed: HTTP {resp.status_code}")
                return False
        except requests.Timeout:
            logger.warning(f"Inference timed out after {INFERENCE_TIMEOUT}s")
            return False
        except Exception as e:
            logger.warning(f"Inference validation error: {e}")
            return False

    # --- 2E: Full Health Check (called from launcher, pipeline, realtime) ---

    def check_health(self) -> bool:
        """Full LLM health check sequence. Returns True if LLM is usable.

        Sequence:
        1. Check if Ollama is running → if not, try to start
        2. Check if model is downloaded → if not, auto-download
        3. Validate inference works
        4. On any failure → set llm_available=False, continue gracefully
        """
        logger.info("=== LLM Health Check ===")
        start = time.time()

        # Step 1: Check/start Ollama
        if not self._check_ollama():
            if not self._start_ollama():
                logger.warning("LLM unavailable: Ollama not running")
                self.llm_available = False
                return False

        # Step 2: Check/download model
        if not self._model_available():
            if not self._pull_model():
                logger.warning("LLM unavailable: model download failed")
                self.llm_available = False
                return False

        # Step 3: Validate inference
        if not self._validate_inference():
            logger.warning("LLM degraded: inference validation failed")
            self.llm_available = False
            self.llm_degraded = True
            return False

        elapsed = time.time() - start
        logger.info(f"LLM health check passed in {elapsed:.1f}s")
        self.llm_available = True
        self.llm_degraded = False
        return True

    # --- Sentiment Analysis (used by Phase 3/7 pipeline) ---

    def analyze_sentiment(self, articles: list[dict]) -> list[dict]:
        """Analyze sentiment using two-tier pipeline with FinBERT cache (P1 enhanced):
        1. Fast path: FinBERT on articles NOT in cache (cache-hit articles are free)
        2. Deep path: DeepSeek on top 5 highest-impact articles

        Falls back to LLM-only or FinBERT-only if either is unavailable.
        Cache key: SHA-256 of (headline + summary[:300]) — works across both
        raw_articles and news tables.
        """
        if not self.finbert_available and not self.llm_available:
            logger.info("No sentiment models available, returning neutral")
            return [{"score": 0.0, "confidence": 0, "topics": []} for _ in articles]

        # --- Fast path: FinBERT with cache ---
        fast_results = []
        if self.finbert_available:
            if self._router is not None:
                # Cache-aware path: only run inference on uncached articles
                from src.data.finbert_cache_utils import score_articles_with_cache
                fb_results = score_articles_with_cache(
                    router=self._router,
                    articles=articles,
                    finbert_pipeline=self.finbert,
                    batch_size=32,
                )
                for r in fb_results:
                    raw_score = r["fb_score"]
                    conf = int(max(r["fb_positive"], r["fb_negative"], r["fb_neutral"]) * 100)
                    fast_results.append({
                        "score": round(raw_score, 4),
                        "confidence": conf,
                        "topics": [],
                    })
            else:
                # Fallback: no router available, run uncached (original behavior)
                logger.info(f"FinBERT fast-path (uncached): scoring {len(articles)} articles...")
                for a in articles:
                    text = f"{a.get('headline', '')} {a.get('summary', '')[:300]}"
                    fb = self._finbert_score(text)
                    fast_results.append({
                        "score": fb["score"],
                        "confidence": fb["confidence"],
                        "topics": [],
                    })
                logger.info("FinBERT fast-path complete")
        else:
            fast_results = [{"score": 0.0, "confidence": 0, "topics": []} for _ in articles]

        if not self.llm_available:
            return fast_results

        # --- Deep path: DeepSeek on top 5 by absolute FinBERT score ---
        abs_scores = [abs(r["score"]) for r in fast_results]
        top_indices = sorted(range(len(abs_scores)),
                             key=lambda i: abs_scores[i], reverse=True)[:5]
        top_articles = [articles[i] for i in top_indices]

        logger.info(f"DeepSeek deep-path: analysing top {len(top_articles)} articles...")
        deep_results = self._analyze_batch(top_articles)

        # Merge: replace fast scores with deep scores for top articles
        results = list(fast_results)
        for rank, orig_idx in enumerate(top_indices):
            if rank < len(deep_results):
                results[orig_idx] = deep_results[rank]

        return results

    def _analyze_batch(self, articles: list[dict]) -> list[dict]:
        """Analyze a batch of articles with structured sentiment decomposition (P2)."""
        article_text = ""
        for idx, a in enumerate(articles):
            article_text += f"\n[{idx+1}] {a.get('headline', '')}\n{a.get('summary', '')[:300]}\n"

        prompt = f"""Analyze the market sentiment of these financial news articles.
For each article, provide a JSON object with:
- "score": float from -1.0 (very bearish) to 1.0 (very bullish)
- "confidence": integer 0-100
- "topics": list of key topics
- "category": one of "macro", "earnings", "geopolitical", "technical", "other"

Return ONLY a JSON array of objects, one per article. No other text.

Articles:{article_text}"""

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=60,  # qwen3 is fast
            )
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "")
                return self._parse_sentiment_response(content, len(articles))
            else:
                logger.warning(f"Sentiment API error: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")

        return [{"score": 0.0, "confidence": 0, "topics": []} for _ in articles]

    def _parse_sentiment_response(self, content: str, expected_count: int) -> list[dict]:
        """Parse LLM JSON response, with fallback for malformed output."""
        # Try to extract JSON array from response
        try:
            # Find JSON array in response
            start = content.find("[")
            end = content.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(content[start:end])
                if isinstance(data, list):
                    results = []
                    for item in data:
                        results.append({
                            "score": max(-1.0, min(1.0, float(item.get("score", 0)))),
                            "confidence": max(0, min(100, int(item.get("confidence", 0)))),
                            "topics": item.get("topics", []),
                            "category": item.get("category", "other"),
                        })
                    # Pad if fewer results than expected
                    while len(results) < expected_count:
                        results.append({"score": 0.0, "confidence": 0, "topics": []})
                    return results[:expected_count]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse sentiment JSON: {e}")

        return [{"score": 0.0, "confidence": 0, "topics": []} for _ in range(expected_count)]

    def get_neutral_sentiment(self) -> dict:
        """Return neutral sentiment when LLM is unavailable."""
        return {
            "score": 0.0,
            "confidence": 0,
            "article_count": 0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 1.0,
        }

    def aggregate_daily_sentiment(self, results: list[dict]) -> dict:
        """Aggregate individual article sentiments into daily summary.

        P2: Includes structured decomposition by category.
        """
        if not results:
            return self.get_neutral_sentiment()

        scores = [r["score"] for r in results]
        confidences = [r["confidence"] for r in results]

        # Weighted average by confidence
        total_weight = sum(confidences) or 1
        weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_weight

        positive = sum(1 for s in scores if s > 0.1)
        negative = sum(1 for s in scores if s < -0.1)
        neutral = len(scores) - positive - negative
        total = len(scores) or 1

        # P2: Decomposed sentiment by category
        categories = {"macro": [], "earnings": [], "geopolitical": [], "technical": []}
        for r in results:
            cat = r.get("category", "other")
            if cat in categories:
                categories[cat].append(r["score"])

        macro_sent = float(np.mean(categories["macro"])) if categories["macro"] else 0.0
        earnings_sent = float(np.mean(categories["earnings"])) if categories["earnings"] else 0.0
        geo_sent = float(np.mean(categories["geopolitical"])) if categories["geopolitical"] else 0.0
        tech_sent = float(np.mean(categories["technical"])) if categories["technical"] else 0.0

        # Dispersion: std dev of all scores
        dispersion = float(np.std(scores)) if len(scores) > 1 else 0.0

        return {
            "score": round(weighted_score, 4),
            "confidence": round(sum(confidences) / len(confidences), 1),
            "article_count": len(results),
            "positive_ratio": round(positive / total, 3),
            "negative_ratio": round(negative / total, 3),
            "neutral_ratio": round(neutral / total, 3),
            # P2: Structured decomposition
            "macro_sentiment": round(macro_sent, 4),
            "earnings_sentiment": round(earnings_sent, 4),
            "geopolitical_sentiment": round(geo_sent, 4),
            "technical_sentiment": round(tech_sent, 4),
            "sentiment_dispersion": round(dispersion, 4),
        }
