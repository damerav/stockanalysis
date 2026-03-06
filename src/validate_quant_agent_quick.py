"""Quick validation of QuantAgent tools — skip feature-vector-heavy tools."""
import json
import sys
import time
import traceback

sys.path.insert(0, ".")

from src.data.init_db import load_config
from src.llm.quant_agent import QuantAgent

config = load_config()
agent = QuantAgent(config)

TOOLS_TO_TEST = [
    ("query_database",              {"sql": "SELECT date, close FROM prices ORDER BY date DESC LIMIT 3"}),
    ("query_database",              {"sql": "SELECT COUNT(*) as cnt FROM raw_articles", "db": "news"}),
    ("get_prediction_state",        {}),
    ("get_model_info",              {}),
    ("get_feature_importance",      {"top_n": 5}),
    ("get_news_summary",            {"days": 2}),
    ("get_news_summary",            {"days": 1, "category": "markets"}),
    ("get_regime_history",          {"days": 14}),
    ("get_pipeline_status",         {}),
    ("search_similar_news",         {"query": "Fed rate hike inflation", "limit": 5}),
    # LLM-dependent tools
    ("assess_news_risk",            {"days": 1}),
    ("explain_regime",              {}),
    ("generate_alpha_hypothesis",   {}),
]

print("=" * 70)
print("QUANT AGENT TOOL VALIDATION (QUICK — no feature vector tools)")
print(f"Tools to test: {len(TOOLS_TO_TEST)}")
print("=" * 70)

passed = 0
failed = 0

for tool_name, args in TOOLS_TO_TEST:
    label = f"{tool_name}({json.dumps(args, default=str)[:60]})"
    print(f"\n  TEST  {label}")
    t0 = time.time()
    try:
        fn = agent.tools[tool_name]
        result = fn(**args)
        elapsed = round(time.time() - t0, 2)

        has_error = isinstance(result, dict) and "error" in result
        if has_error:
            err_msg = result["error"][:120]
            # Expected failures
            if "No pipeline results" in err_msg or "No embeddings" in err_msg:
                print(f"  XFAIL {label} ({elapsed}s) — expected: {err_msg}")
                passed += 1
            else:
                print(f"  FAIL  {label} ({elapsed}s)")
                print(f"        Error: {err_msg}")
                failed += 1
        else:
            summary = ""
            if isinstance(result, dict):
                keys = list(result.keys())[:6]
                summary = f"keys={keys}"
                for k in ["rows", "total_articles", "accuracy", "hypotheses_count",
                           "articles_assessed", "avg_risk_score", "current_regime"]:
                    if k in result:
                        summary += f", {k}={result[k]}"
                if "explanation" in result:
                    summary += f", explanation_len={len(result['explanation'])}"
            else:
                summary = str(result)[:100]
            print(f"  PASS  {label} ({elapsed}s)")
            print(f"        {summary}")
            passed += 1
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        print(f"  FAIL  {label} ({elapsed}s)")
        print(f"        Exception: {e}")
        traceback.print_exc()
        failed += 1

print(f"\n{'='*70}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*70}")
sys.exit(1 if failed > 0 else 0)
