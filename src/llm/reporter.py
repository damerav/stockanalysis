"""3H. Daily Report Generator — LLM-narrated market brief."""

import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class DailyReporter:
    """Generates ~400-word daily market brief using LLM."""

    def __init__(self, config: dict = None):
        config = config or {}
        llm_cfg = config.get("llm", {})
        self.base_url = llm_cfg.get("base_url", "http://localhost:11434")
        self.model = llm_cfg.get("model", "deepseek-r1:70b")
        self.temperature = 0.4  # slightly more creative than sentiment

    def generate_report(self, context: dict, llm_available: bool = True) -> str:
        """Generate daily market report.

        Args:
            context: Dict with keys: prediction, technicals, sentiment, signals
            llm_available: Whether LLM is available

        Returns:
            ~400-word market brief string
        """
        if not llm_available:
            return self._fallback_report(context)

        prompt = self._build_prompt(context)

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=300,
            )
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "")
                if content.strip():
                    return content.strip()
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

        return self._fallback_report(context)

    def _build_prompt(self, ctx: dict) -> str:
        pred = ctx.get("prediction", {})
        tech = ctx.get("technicals", {})
        sent = ctx.get("sentiment", {})
        macro = ctx.get("macro", {})

        return f"""Write a concise ~400-word daily market analysis report for SPY/SPX.
Use a professional but accessible tone. Include:
1. Today's prediction and confidence level
2. Key technical signals
3. Market sentiment summary
4. Notable macro factors
5. Key risks to watch

Data:
- Prediction: {pred.get('direction', 'N/A')} ({pred.get('confidence', 0):.0f}% confidence)
- Scale: {pred.get('scale_label', 'N/A')}
- RSI(14): {tech.get('rsi_14', 'N/A')}
- MACD: {tech.get('macd', 'N/A')}, Signal: {tech.get('macd_signal', 'N/A')}
- SMA20: {tech.get('sma_20', 'N/A')}, SMA50: {tech.get('sma_50', 'N/A')}
- ATR(14): {tech.get('atr_14', 'N/A')}
- VIX: {macro.get('vix', 'N/A')}, Change: {macro.get('vix_change', 'N/A')}
- 10Y Yield: {macro.get('us10y_yield', 'N/A')}
- DXY: {macro.get('dxy', 'N/A')}
- Sentiment Score: {sent.get('score', 0):.2f} (from {sent.get('article_count', 0)} articles)
- Positive/Negative ratio: {sent.get('positive_ratio', 0):.0%} / {sent.get('negative_ratio', 0):.0%}

Write the report now. No preamble, just the report."""

    def _fallback_report(self, ctx: dict) -> str:
        """Generate a basic report without LLM."""
        pred = ctx.get("prediction", {})
        tech = ctx.get("technicals", {})
        macro = ctx.get("macro", {})

        direction = pred.get("direction", "NEUTRAL")
        confidence = pred.get("confidence", 0)
        rsi = tech.get("rsi_14", "N/A")
        vix = macro.get("vix", "N/A")

        return (
            f"Daily Market Summary\n\n"
            f"Prediction: {direction} ({confidence:.0f}% confidence)\n"
            f"RSI(14): {rsi}\n"
            f"VIX: {vix}\n\n"
            f"Note: Full LLM-generated report unavailable. "
            f"Showing raw data summary only."
        )
