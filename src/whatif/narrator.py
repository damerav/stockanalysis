"""8D. LLM-Narrated Explanations — Ask DeepSeek to explain What-If results."""

import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class WhatIfNarrator:
    """Generates plain-English narratives for What-If analysis results."""

    def __init__(self, config: dict = None):
        config = config or {}
        llm_cfg = config.get("llm", {})
        wi_cfg = config.get("whatif", {})
        self.base_url = llm_cfg.get("base_url", "http://localhost:11434")
        self.model = llm_cfg.get("model", "deepseek-r1:70b")
        self.temperature = wi_cfg.get("narrator_temperature", 0.5)
        self.max_tokens = wi_cfg.get("narrator_max_tokens", 800)

    def narrate_es_result(self, result: dict, llm_available: bool = True) -> str:
        """Narrate an ES strategy what-if result."""
        if not llm_available:
            return self._fallback_es(result)

        prompt = self._build_es_prompt(result)
        return self._call_llm(prompt) or self._fallback_es(result)

    def narrate_spy_result(self, result: dict, llm_available: bool = True) -> str:
        """Narrate a SPY predictor what-if result."""
        if not llm_available:
            return self._fallback_spy(result)

        prompt = self._build_spy_prompt(result)
        return self._call_llm(prompt) or self._fallback_spy(result)

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call Ollama for narrative generation."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=300,
            )
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "")
                if content.strip():
                    return content.strip()
        except Exception as e:
            logger.warning(f"Narrator LLM call failed: {e}")
        return None

    def _build_es_prompt(self, result: dict) -> str:
        scenario = result.get("scenario", "parameter change")
        original = result.get("original", {})
        modified = result.get("modified", {})

        return (
            f"You are a quantitative trading analyst. Explain these ES futures "
            f"strategy what-if results in 2-3 concise paragraphs.\n\n"
            f"Scenario: {scenario}\n"
            f"Original — P&L: ${original.get('total_pnl', 0):+,.0f}, "
            f"Trades: {original.get('trades', 0)}\n"
            f"Modified — P&L: ${modified.get('total_pnl', 0):+,.0f}, "
            f"Trades: {modified.get('trades', 0)}\n\n"
            f"Parameters changed: {result.get('params_changed', {})}\n\n"
            f"Explain why this change affected results and what risks it introduces. "
            f"Be specific about the trading mechanics."
        )

    def _build_spy_prompt(self, result: dict) -> str:
        scenario = result.get("scenario", "feature override")
        original = result.get("original", {})
        modified = result.get("modified", {})
        overrides = result.get("overrides", {})

        return (
            f"You are a quantitative market analyst. Explain these SPY prediction "
            f"what-if results in 2-3 concise paragraphs.\n\n"
            f"Scenario: {scenario}\n"
            f"Original prediction: {original.get('direction', 'N/A')} "
            f"({original.get('confidence', 0):.0f}%)\n"
            f"Modified prediction: {modified.get('direction', 'N/A')} "
            f"({modified.get('confidence', 0):.0f}%)\n"
            f"Features overridden: {overrides}\n\n"
            f"Explain why these feature changes shifted the prediction and "
            f"what market conditions this scenario represents."
        )

    def _fallback_es(self, result: dict) -> str:
        orig = result.get("original", {})
        mod = result.get("modified", {})
        delta_pnl = mod.get("total_pnl", 0) - orig.get("total_pnl", 0)
        return (
            f"Scenario: {result.get('scenario', 'N/A')}\n"
            f"P&L change: ${delta_pnl:+,.0f} "
            f"(${orig.get('total_pnl', 0):+,.0f} → ${mod.get('total_pnl', 0):+,.0f})\n"
            f"Trade count: {orig.get('trades', 0)} → {mod.get('trades', 0)}\n"
            f"(LLM narrative unavailable)"
        )

    def _fallback_spy(self, result: dict) -> str:
        orig = result.get("original", {})
        mod = result.get("modified", {})
        return (
            f"Scenario: {result.get('scenario', 'N/A')}\n"
            f"Prediction: {orig.get('direction', '?')} → {mod.get('direction', '?')}\n"
            f"Confidence: {orig.get('confidence', 0):.0f}% → {mod.get('confidence', 0):.0f}%\n"
            f"(LLM narrative unavailable)"
        )
