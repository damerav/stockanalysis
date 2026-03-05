"""Validate all QuantAgent tools — run on DGX."""
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
    ("analyze_feature_correlations",{"threshold": 0.85}),
    # Vector search (needs embeddings — will gracefully error if none)
    ("search_similar_news",         {"query": "Fed rate hike inflation", "limit": 5}),
    # LLM-dependent tools (need Ollama + DeepSeek R1, slow)
    ("assess_news_risk",            {"days": 2}),
    ("explain_regime",              {}),
    ("generate_alpha_hypothesis",   {}),
    # Heavy tools
    ("compare_strategies",          {"days": 30}),
    ("run_backtest",                {"days": 30}),
    ("retrain_model",              {}),
]

# Skip these heavy/LLM tools unless --full flag
HEAVY = {"compare_strategies", "run_backtest", "retrain_model",
         "assess_news_risk", "explain_regime", "generate_alpha_hypothesis"}
run_full = "--full" in sys.argv

print("=" * 70)
print("QUANT AGENT TOOL VALIDATION")
print(f"Total tools registered: {len(agent.tools)}")
print(f"Tools to test: {len(TOOLS_TO_TEST)}")
print(f"Mode: {'FULL (including heavy tools)' if run_full else 'QUICK (skipping heavy tools)'}")
print("=" * 70)

passed = 0
failed = 0
skipped = 0
results = []

for tool_name, args in TOOLS_TO_TEST:
    label = f"{tool_name}({json.dumps(args, default=str)[:60]})"

    if tool_name in HEAVY and not run_full:
        print(f"  SKIP  {label}")
        skipped += 1
        results.append({"tool": tool_name, "status": "SKIPPED"})
        continue

    print(f"\n  TEST  {label}")
    t0 = time.time()
    try:
        fn = agent.tools[tool_name]
        result = fn(**args)
        elapsed = round(time.time() - t0, 2)

        has_error = isinstance(result, dict) and "error" in result
        if has_error:
            print(f"  FAIL  {label} ({elapsed}s)")
            print(f"        Error: {result['error'][:120]}")
            failed += 1
            results.append({"tool": tool_name, "status": "FAIL", "error": result["error"][:200], "elapsed": elapsed})
        else:
            # Summarize result
            if isinstance(result, dict):
                keys = list(result.keys())
                summary = f"keys={keys[:6]}"
                if "rows" in result:
                    summary += f", rows={result['rows']}"
                if "total_articles" in result:
                    summary += f", articles={result['total_articles']}"
                if "accuracy" in result:
                    summary += f", accuracy={result['accuracy']}"
                if "hypotheses_count" in result:
                    summary += f", hypotheses={result['hypotheses_count']}"
                if "articles_assessed" in result:
                    summary += f", assessed={result['articles_assessed']}, avg_risk={result.get('avg_risk_score')}"
                if "explanation" in result:
                    summary += f", explanation_len={len(result['explanation'])}"
                if "strategies" in result:
                    summary += f", strategies={list(result['strategies'].keys())}"
                if "high_corr_count" in result:
                    summary += f", high_corr_pairs={result['high_corr_count']}"
            else:
                summary = str(result)[:100]

            print(f"  PASS  {label} ({elapsed}s)")
            print(f"        {summary}")
            passed += 1
            results.append({"tool": tool_name, "status": "PASS", "elapsed": elapsed})
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        print(f"  FAIL  {label} ({elapsed}s)")
        print(f"        Exception: {e}")
        traceback.print_exc()
        failed += 1
        results.append({"tool": tool_name, "status": "FAIL", "error": str(e)[:200], "elapsed": elapsed})

# Also verify all registered tools are covered
registered = set(agent.tools.keys())
tested = set(t[0] for t in TOOLS_TO_TEST)
untested = registered - tested
if untested:
    print(f"\n  WARN  Untested tools: {untested}")

print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
print(f"Registered tools: {len(registered)}, Tested: {len(tested)}, Untested: {len(untested)}")
print("=" * 70)

sys.exit(1 if failed > 0 else 0)
