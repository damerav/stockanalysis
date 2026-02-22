#!/usr/bin/env python3
"""Validate FRED and Finnhub API keys work."""
import sys, os, traceback, time
sys.path.insert(0, ".")

results = []
def check(name, fn):
    try:
        start = time.time()
        ok = fn()
        elapsed = time.time() - start
        status = f"PASS ({elapsed:.1f}s)" if ok else "FAIL"
    except Exception as e:
        ok = False
        status = f"FAIL: {e}"
        traceback.print_exc()
    results.append((name, "PASS" in status))
    print(f"  {'✓' if ok else '✗'} {name}: {status}")

# 1. Config has both keys
def t_config():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    fred_key = cfg.get("fred", {}).get("api_key", "")
    finnhub_key = cfg.get("finnhub", {}).get("api_key", "")
    assert fred_key and len(fred_key) > 10, f"FRED key missing or too short: {fred_key[:5]}..."
    assert finnhub_key and len(finnhub_key) > 10, f"Finnhub key missing or too short"
    print(f"    FRED key: {fred_key[:8]}...")
    print(f"    Finnhub key: {finnhub_key[:8]}...")
    return True
check("Config has API keys", t_config)

# 2. FRED API with key
def t_fred_api():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher(config=cfg)
    assert f.fred_key, "FRED key not loaded into fetcher"
    macro = f.get_macro_fred()
    filled = {k: v for k, v in macro.items() if v is not None}
    print(f"    FRED API response: {filled}")
    assert "vix" in filled, "VIX not returned from FRED API"
    assert "us10y_yield" in filled, "10Y yield not returned"
    return True
check("FRED API with key", t_fred_api)

# 3. Finnhub API with key
def t_finnhub():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher(config=cfg)
    assert f.finnhub_key, "Finnhub key not loaded into fetcher"
    articles = f.get_news_finnhub()
    print(f"    Finnhub returned {len(articles)} articles")
    if articles:
        print(f"    Sample: {articles[0]['headline'][:80]}...")
    assert len(articles) > 0, "Finnhub returned no articles with key"
    return True
check("Finnhub API with key", t_finnhub)

# 4. Combined news (Finnhub + RSS)
def t_combined_news():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher(config=cfg)
    articles = f.get_news()
    sources = set(a["source"] for a in articles)
    print(f"    Combined: {len(articles)} articles from {sources}")
    assert "finnhub" in sources or len(sources) >= 2, "Expected Finnhub in sources"
    return True
check("Combined news (Finnhub + RSS)", t_combined_news)

# 5. FallbackFetcher from pipeline context
def t_pipeline_fetcher():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher(config=cfg)
    assert f.fred_key == cfg["fred"]["api_key"]
    assert f.finnhub_key == cfg["finnhub"]["api_key"]
    return True
check("Pipeline FallbackFetcher gets keys from config", t_pipeline_fetcher)

# Summary
print(f"\n{'='*60}")
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"Results: {passed}/{total} passed")
for name, ok in results:
    print(f"  {'✓' if ok else '✗'} {name}")
if passed < total:
    sys.exit(1)
