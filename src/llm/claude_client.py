"""Claude Opus 4.6 client for market analysis via Anthropic API.

Uses encrypted secrets DB for API key storage.
Provides a simple interface for sending market analysis prompts
and receiving structured predictions.
"""

import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-opus-4-6"
MAX_TOKENS = 1024


def _get_api_key() -> Optional[str]:
    """Retrieve Claude API key from encrypted secrets DB."""
    try:
        from src.data.secrets_manager import get_secret
        key = get_secret("claude_api_key")
        if key:
            return key
    except Exception as e:
        logger.warning("Failed to get Claude API key from secrets: %s", e)
    import os
    return os.environ.get("ANTHROPIC_API_KEY")


def analyze_market(prompt: str, temperature: float = 0.3) -> dict:
    """Send a market analysis prompt to Claude Opus 4.6.

    Returns dict with keys: direction, confidence, reasoning, raw_response,
    latency_s, model, error (if any).
    """
    api_key = _get_api_key()
    if not api_key:
        return {"error": "Claude API key not configured. Set via Admin > System Management or set_secret('claude_api_key', '...')"}

    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic package not installed. Run: pip install anthropic"}

    system_prompt = (
        "You are a quantitative market analyst. Analyze the provided market data "
        "and return a JSON object with exactly these keys:\n"
        '- "direction": one of "BULLISH", "BEARISH", or "NEUTRAL"\n'
        '- "confidence": integer 0-100\n'
        '- "reasoning": string with 2-3 sentence explanation\n'
        '- "key_factors": list of up to 5 short factor strings\n'
        '- "risk_level": one of "LOW", "MEDIUM", "HIGH"\n'
        "Return ONLY valid JSON, no markdown fences, no extra text."
    )

    start = time.time()
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - start
        raw = message.content[0].text.strip()

        # Parse JSON response
        parsed = _parse_response(raw)
        parsed["raw_response"] = raw
        parsed["latency_s"] = round(latency, 2)
        parsed["model"] = CLAUDE_MODEL
        return parsed

    except Exception as e:
        latency = time.time() - start
        logger.error("Claude API call failed: %s", e)
        return {
            "error": str(e),
            "latency_s": round(latency, 2),
            "model": CLAUDE_MODEL,
        }


def _parse_response(raw: str) -> dict:
    """Parse Claude's JSON response with fallback for malformed output."""
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
        return {
            "direction": data.get("direction", "NEUTRAL"),
            "confidence": max(0, min(100, int(data.get("confidence", 50)))),
            "reasoning": data.get("reasoning", ""),
            "key_factors": data.get("key_factors", []),
            "risk_level": data.get("risk_level", "MEDIUM"),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning("Failed to parse Claude response: %s", e)
        return {
            "direction": "NEUTRAL",
            "confidence": 0,
            "reasoning": raw[:500],
            "key_factors": [],
            "risk_level": "MEDIUM",
            "parse_error": str(e),
        }


def analyze_market_qwq(prompt: str, base_url: str = "http://localhost:11434",
                       model: str = "qwq:32b", temperature: float = 0.3) -> dict:
    """Send the same market analysis prompt to QwQ:32b via Ollama.

    Returns dict with same structure as analyze_market() for comparison.
    """
    import re
    import requests

    system_prompt = (
        "You are a quantitative market analyst. Analyze the provided market data "
        "and return a JSON object with exactly these keys:\n"
        '- "direction": one of "BULLISH", "BEARISH", or "NEUTRAL"\n'
        '- "confidence": integer 0-100\n'
        '- "reasoning": string with 2-3 sentence explanation\n'
        '- "key_factors": list of up to 5 short factor strings\n'
        '- "risk_level": one of "LOW", "MEDIUM", "HIGH"\n'
        "Return ONLY valid JSON, no markdown fences, no extra text."
    )

    start = time.time()
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": temperature, "num_predict": 1024},
            },
            timeout=120,
        )
        resp.raise_for_status()
        latency = time.time() - start
        raw = resp.json().get("message", {}).get("content", "")
        # Strip <think> blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        parsed = _parse_response(raw)
        parsed["raw_response"] = raw
        parsed["latency_s"] = round(latency, 2)
        parsed["model"] = model
        return parsed

    except Exception as e:
        latency = time.time() - start
        logger.error("QwQ API call failed: %s", e)
        return {
            "error": str(e),
            "latency_s": round(latency, 2),
            "model": model,
        }


def build_market_prompt(state: dict, macro: dict = None) -> str:
    """Build a standardized market analysis prompt from spy_state and macro data.

    Both QwQ and Claude receive the identical prompt for fair comparison.
    """
    pred = state.get("prediction", {})
    indicators = state.get("indicators", {})
    regime = state.get("regime", "unknown")

    parts = ["Analyze the current SPY/S&P 500 market conditions:\n"]

    # Current prediction context
    parts.append(f"Current ML Model Prediction: {pred.get('direction', 'N/A')} "
                 f"(confidence: {pred.get('confidence', 'N/A')}%)")
    probs = pred.get("probabilities", {})
    if probs:
        parts.append(f"Probabilities — UP: {probs.get('UP', 0):.1%}, "
                     f"DOWN: {probs.get('DOWN', 0):.1%}, "
                     f"NEUTRAL: {probs.get('NEUTRAL', 0):.1%}")

    # Technical indicators
    parts.append(f"\nTechnical Indicators:")
    parts.append(f"  RSI(14): {indicators.get('rsi_14', 'N/A')}")
    parts.append(f"  MACD: {indicators.get('macd', 'N/A')}")
    parts.append(f"  ATR(14): {indicators.get('atr_14', 'N/A')}")
    parts.append(f"  Volume Ratio: {indicators.get('volume_ratio', 'N/A')}")

    # Market context
    parts.append(f"\nMarket Context:")
    parts.append(f"  VIX: {indicators.get('vix', 'N/A')}")
    parts.append(f"  VIX Change: {indicators.get('vix_change', 'N/A')}")
    parts.append(f"  Sentiment Score: {indicators.get('sentiment_score', 'N/A')}")
    parts.append(f"  HMM Regime: {regime}")

    # Macro data if available
    if macro:
        parts.append(f"\nMacro Data:")
        for key in ["dxy", "us10y", "gold", "fear_greed", "trin"]:
            val = macro.get(key)
            if val is not None:
                parts.append(f"  {key}: {val}")

    # Vigilance alerts
    alerts = state.get("vigilance_alerts", [])
    if alerts:
        parts.append(f"\nActive Vigilance Alerts: {len(alerts)}")
        for a in alerts[:3]:
            parts.append(f"  - {a.get('type', 'unknown')}: {a.get('message', '')}")

    parts.append("\nProvide your independent market assessment as JSON.")
    return "\n".join(parts)
