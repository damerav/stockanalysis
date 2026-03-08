"""LLM-Driven Formulaic Alpha Generation.

Uses DeepSeek R1 (via Ollama) to generate mathematical alpha expressions
combining price, volume, sentiment, and macro data. Evaluates each alpha
historically and keeps those with predictive power (IC > threshold).

Alpha expressions are Python-evaluable formulas operating on a DataFrame row.
"""

import logging
import json
import os
import time
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
ALPHA_CACHE_PATH = "./models/alpha_formulas.json"

# Columns available for alpha construction (subset of feature vector)
ALPHA_BUILDING_BLOCKS = [
    "close", "volume", "return_1d", "return_2d", "return_3d",
    "momentum_5d", "momentum_10d", "momentum_20d",
    "rsi_14", "rsi_2", "rsi_9", "macd", "macd_hist",
    "atr_14", "bb_upper_dist", "bb_lower_dist",
    "vix", "vix_change", "vix_mean_reversion",
    "volume_ratio", "volume_spike",
    "sentiment_score", "finbert_score",
    "put_call_ratio", "iv_skew", "gex_normalized",
    "sma20_slope", "sma50_slope",
    "price_vs_sma20_pct", "price_vs_sma50_pct",
    "advance_decline_ratio", "breadth_thrust",
    "cot_commercial_change", "cot_leveraged_net",
    "safe_haven_flow", "flow_momentum_5d",
    "fear_greed_index", "trin",
    "yield_curve_10y3m", "earnings_yield_gap",
    "geo_risk_score", "crude_shock",
    "consecutive_up_days", "consecutive_down_days",
    "pct_from_52w_high", "pct_from_52w_low",
    "daily_range_pct", "overnight_gap",
]


ALPHA_GENERATION_PROMPT = """You are a quantitative researcher generating formulaic alpha signals for predicting next-day SPY direction.

Available columns (all numeric, daily frequency):
{columns}

Generate {count} unique alpha formulas. Each must be:
1. A valid Python expression using only the column names above, numpy (as np), and basic math
2. Designed to capture a specific market insight (momentum reversal, sentiment divergence, volatility regime, etc.)
3. Different from standard technical indicators — combine multiple signals creatively

Rules:
- Use np.sign(), np.log1p(), np.abs(), np.clip(), np.where(), np.tanh() for transforms
- Reference columns as row["column_name"]
- Each formula should fit on one line
- Avoid division by zero: use np.where(denom != 0, num/denom, 0) pattern
- Output ONLY valid JSON array of objects with "name", "formula", "rationale" keys
- Names should be descriptive like "alpha_momentum_sentiment_divergence"

Example output:
[
  {{"name": "alpha_rsi_vix_cross", "formula": "np.tanh(row['rsi_14'] / 50 - 1) * np.sign(row['vix_change'])", "rationale": "RSI overbought/oversold signal weighted by VIX direction change"}},
  {{"name": "alpha_flow_momentum_gap", "formula": "row['flow_momentum_5d'] * np.sign(row['overnight_gap'])", "rationale": "Institutional flow momentum confirmed by overnight gap direction"}}
]

Generate exactly {count} alphas. Output ONLY the JSON array, no other text."""


def _call_ollama(prompt: str, model: str = "qwen3:8b",
                 timeout: int = 300) -> Optional[str]:
    """Call Ollama API via chat endpoint with thinking disabled for speed."""
    import requests
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False,
                  "think": False,
                  "options": {"temperature": 0.7, "num_predict": 4096}},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "")
        logger.warning(f"Ollama returned {resp.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}")
        return None


def _parse_alpha_response(text: str) -> list[dict]:
    """Extract JSON array from LLM response, handling markdown fences."""
    if not text:
        return []
    # Strip thinking tags if present
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Find JSON array
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        logger.warning("Failed to parse alpha JSON from LLM response")
        return []


def generate_alpha_candidates(count: int = 50,
                              model: str = "qwen3:8b") -> list[dict]:
    """Use DeepSeek to generate formulaic alpha candidates."""
    cols_str = ", ".join(ALPHA_BUILDING_BLOCKS)
    prompt = ALPHA_GENERATION_PROMPT.format(columns=cols_str, count=count)

    logger.info(f"Generating {count} alpha candidates via {model}...")
    # Generate in batches of 15 to get better quality
    all_alphas = []
    batch_size = 15
    for batch_num in range(0, count, batch_size):
        remaining = min(batch_size, count - batch_num)
        batch_prompt = prompt.replace(f"Generate exactly {count}",
                                      f"Generate exactly {remaining}")
        response = _call_ollama(batch_prompt, model=model)
        alphas = _parse_alpha_response(response)
        all_alphas.extend(alphas)
        logger.info(f"Batch {batch_num // batch_size + 1}: got {len(alphas)} alphas")
        if batch_num + batch_size < count:
            time.sleep(2)  # Rate limit between batches

    # Deduplicate by name
    seen = set()
    unique = []
    for a in all_alphas:
        name = a.get("name", "")
        if name and name not in seen:
            seen.add(name)
            unique.append(a)

    logger.info(f"Generated {len(unique)} unique alpha candidates")
    return unique


def evaluate_alpha(formula: str, df: pd.DataFrame,
                   forward_col: str = "forward_return") -> dict:
    """Evaluate a single alpha formula on historical data.

    Returns dict with IC (information coefficient), IC_abs, hit_rate, and validity.
    """
    try:
        # Compute alpha values for each row
        alpha_values = []
        for _, row in df.iterrows():
            try:
                val = eval(formula, {"np": np, "row": row, "__builtins__": {}})
                alpha_values.append(float(val) if np.isfinite(val) else 0.0)
            except Exception:
                alpha_values.append(0.0)

        alpha_series = pd.Series(alpha_values, index=df.index)

        # Skip if all zeros or all same value
        if alpha_series.std() < 1e-10:
            return {"valid": False, "reason": "zero_variance"}

        # Information Coefficient: rank correlation with forward returns
        ic = alpha_series.corr(df[forward_col], method="spearman")
        if np.isnan(ic):
            return {"valid": False, "reason": "nan_ic"}

        # Hit rate: does sign of alpha predict sign of forward return?
        signs_match = (np.sign(alpha_series) == np.sign(df[forward_col]))
        hit_rate = signs_match.mean()

        # Turnover: how much does the alpha change day-to-day
        turnover = alpha_series.diff().abs().mean() / (alpha_series.abs().mean() + 1e-10)

        return {
            "valid": True,
            "ic": float(ic),
            "ic_abs": float(abs(ic)),
            "hit_rate": float(hit_rate),
            "turnover": float(turnover),
            "mean": float(alpha_series.mean()),
            "std": float(alpha_series.std()),
        }
    except Exception as e:
        return {"valid": False, "reason": str(e)[:100]}


def backtest_alphas(candidates: list[dict], df: pd.DataFrame,
                    ic_threshold: float = 0.02,
                    min_hit_rate: float = 0.51) -> list[dict]:
    """Evaluate all alpha candidates and return those that pass quality filters.

    Args:
        candidates: List of dicts with 'name', 'formula', 'rationale'
        df: Historical DataFrame with feature columns + 'forward_return'
        ic_threshold: Minimum absolute IC to keep
        min_hit_rate: Minimum directional hit rate

    Returns:
        List of winning alphas with evaluation metrics
    """
    winners = []
    for alpha in candidates:
        name = alpha.get("name", "unknown")
        formula = alpha.get("formula", "")
        if not formula:
            continue

        metrics = evaluate_alpha(formula, df)
        if not metrics.get("valid"):
            logger.debug(f"Alpha {name}: invalid — {metrics.get('reason')}")
            continue

        if metrics["ic_abs"] >= ic_threshold and metrics["hit_rate"] >= min_hit_rate:
            alpha_result = {**alpha, **metrics}
            winners.append(alpha_result)
            logger.info(f"Alpha {name}: IC={metrics['ic']:.4f} "
                        f"hit={metrics['hit_rate']:.1%} ✓")
        else:
            logger.debug(f"Alpha {name}: IC={metrics['ic']:.4f} "
                         f"hit={metrics['hit_rate']:.1%} — below threshold")

    # Sort by absolute IC
    winners.sort(key=lambda x: x["ic_abs"], reverse=True)
    logger.info(f"Backtest complete: {len(winners)}/{len(candidates)} alphas passed")
    return winners


def compute_alpha_features(df: pd.DataFrame,
                           alphas: list[dict] = None) -> pd.DataFrame:
    """Compute alpha feature columns for a DataFrame.

    Loads saved alphas from disk if not provided.
    Returns DataFrame with alpha columns added.
    """
    if alphas is None:
        alphas = load_alphas()
    if not alphas:
        return df

    result = df.copy()
    for alpha in alphas:
        name = alpha["name"]
        formula = alpha["formula"]
        try:
            values = []
            for _, row in result.iterrows():
                try:
                    val = eval(formula, {"np": np, "row": row, "__builtins__": {}})
                    values.append(float(val) if np.isfinite(val) else 0.0)
                except Exception:
                    values.append(0.0)
            result[name] = values
        except Exception as e:
            logger.warning(f"Failed to compute alpha {name}: {e}")
            result[name] = 0.0

    logger.info(f"Computed {len(alphas)} alpha features")
    return result


def save_alphas(alphas: list[dict], path: str = ALPHA_CACHE_PATH):
    """Save winning alphas to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save only the essential fields
    to_save = []
    for a in alphas:
        to_save.append({
            "name": a["name"],
            "formula": a["formula"],
            "rationale": a.get("rationale", ""),
            "ic": a.get("ic", 0),
            "hit_rate": a.get("hit_rate", 0),
        })
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)
    logger.info(f"Saved {len(to_save)} alphas to {path}")


def load_alphas(path: str = ALPHA_CACHE_PATH) -> list[dict]:
    """Load saved alphas from disk."""
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load alphas: {e}")
        return []


def run_alpha_pipeline(config: dict = None, count: int = 50,
                       model: str = "qwen3:8b") -> list[dict]:
    """Full pipeline: generate → evaluate → save winning alphas.

    Requires historical feature data with forward returns.
    """
    from src.data.db_router import get_router
    from src.data.features import build_feature_vector

    router = get_router(config)
    conn = router.get_sqlite() if hasattr(router, 'get_sqlite') else None

    # Build feature vector with prices for forward return calculation
    fv = build_feature_vector(conn, config=config)
    if fv is None or fv.empty:
        logger.error("No feature data available for alpha pipeline")
        return []

    # Get prices for forward return
    prices = router.query("SELECT date, close FROM prices ORDER BY date")
    if prices.empty:
        logger.error("No price data for forward returns")
        return []

    price_map = dict(zip(prices["date"].astype(str), prices["close"]))
    dates_sorted = sorted(price_map.keys())
    date_idx = {d: i for i, d in enumerate(dates_sorted)}

    # Compute forward returns
    fv["_date_str"] = fv["date"].astype(str)
    forward_returns = []
    for _, row in fv.iterrows():
        d = str(row["_date_str"])
        if d in date_idx:
            idx = date_idx[d]
            if idx < len(dates_sorted) - 1:
                next_d = dates_sorted[idx + 1]
                ret = (price_map[next_d] - price_map[d]) / price_map[d]
                forward_returns.append(ret)
            else:
                forward_returns.append(0.0)
        else:
            forward_returns.append(0.0)
    fv["forward_return"] = forward_returns

    # Drop rows with no forward return
    fv = fv[fv["forward_return"] != 0.0].copy()

    # Fill NaN in building block columns
    for col in ALPHA_BUILDING_BLOCKS:
        if col in fv.columns:
            fv[col] = fv[col].fillna(0.0)

    logger.info(f"Alpha pipeline: {len(fv)} rows with forward returns")

    # Generate candidates
    candidates = generate_alpha_candidates(count=count, model=model)
    if not candidates:
        logger.warning("No alpha candidates generated")
        return []

    # Evaluate
    winners = backtest_alphas(candidates, fv, ic_threshold=0.02, min_hit_rate=0.505)

    # Merge with any existing alphas (keep best by IC)
    existing = load_alphas()
    existing_names = {a["name"] for a in existing}
    for w in winners:
        if w["name"] not in existing_names:
            existing.append(w)
    # Re-sort and keep top 30
    existing.sort(key=lambda x: abs(x.get("ic", 0)), reverse=True)
    final = existing[:30]

    save_alphas(final)
    router.close()
    return final
