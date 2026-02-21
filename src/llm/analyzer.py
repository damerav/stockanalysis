"""Phase 2: LLM Health Check + Sentiment Analyzer.

Handles Ollama lifecycle: check → start → model availability → auto-download → validate.
Graceful degradation: pipeline NEVER aborts due to LLM unavailability.
"""

import time
import json
import logging
import subprocess
import requests
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-r1:70b"
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
        """Analyze sentiment of news articles using the LLM.

        Args:
            articles: List of dicts with 'headline' and 'summary' keys

        Returns:
            List of dicts with 'score' (-1.0 to 1.0), 'confidence' (0-100), 'topics' []
            Returns neutral scores if LLM is unavailable.
        """
        if not self.llm_available:
            logger.info("LLM unavailable, returning neutral sentiment")
            return [{"score": 0.0, "confidence": 0, "topics": []} for _ in articles]

        results = []
        batch_size = 5  # process 5 at a time to avoid context overflow
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            batch_results = self._analyze_batch(batch)
            results.extend(batch_results)
            if i + batch_size < len(articles):
                logger.info(f"Sentiment progress: {min(i + batch_size, len(articles))}/{len(articles)}")

        return results

    def _analyze_batch(self, articles: list[dict]) -> list[dict]:
        """Analyze a batch of articles."""
        article_text = ""
        for idx, a in enumerate(articles):
            article_text += f"\n[{idx+1}] {a.get('headline', '')}\n{a.get('summary', '')[:300]}\n"

        prompt = f"""Analyze the market sentiment of these financial news articles.
For each article, provide a JSON object with:
- "score": float from -1.0 (very bearish) to 1.0 (very bullish)
- "confidence": integer 0-100
- "topics": list of key topics

Return ONLY a JSON array of objects, one per article. No other text.

Articles:{article_text}"""

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=300,  # 5 min per batch
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
        """Aggregate individual article sentiments into daily summary."""
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

        return {
            "score": round(weighted_score, 4),
            "confidence": round(sum(confidences) / len(confidences), 1),
            "article_count": len(results),
            "positive_ratio": round(positive / total, 3),
            "negative_ratio": round(negative / total, 3),
            "neutral_ratio": round(neutral / total, 3),
        }
