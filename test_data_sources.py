#!/usr/bin/env python3
"""Validate all non-Polygon data sources work independently."""
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
    return ok

# 1. yfinance — SPY daily prices
def t_yf_prices():
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher()
    df = f.get_daily_bars_yf("SPY", days=10)
    print(f"    yfinance SPY: {len(df)} rows")
    assert len(df) >= 5, f"Expected >=5 rows, got {len(df)}"
    assert "close" in df.columns
    print(f"    Latest: {df.iloc[-1]['date']} close={df.iloc[-1]['close']:.2f}")
    return True
check("yfinance SPY daily prices", t_yf_prices)

# 2. yfinance — VIX term structure
def t_vix_term():
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher()
    vts = f.get_vix_term_structure()
    filled = {k: v for k, v in vts.items() if v is not None}
    print(f"    VIX term: {filled}")
    assert len(filled) >= 3, f"Expected >=3 VIX term values, got {len(filled)}"
    return True
check("yfinance VIX term structure", t_vix_term)

# 3. yfinance — Cross-asset signals
def t_cross_asset():
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher()
    signals = f.get_cross_asset_signals()
    filled = {k: v for k, v in signals.items() if v is not None}
    print(f"    Cross-asset: {list(filled.keys())}")
    assert len(filled) >= 4, f"Expected >=4 signals, got {len(filled)}"
    return True
check("yfinance cross-asset signals", t_cross_asset)

# 4. FRED — Macro data
def t_fred():
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher()
    macro = f.get_macro_fred()
    filled = {k: v for k, v in macro.items() if v is not None}
    print(f"    FRED macro: {filled}")
    assert "vix" in filled or "us10y_yield" in filled, "No FRED data returned"
    return True
check("FRED macro data", t_fred)

# 5. RSS feeds — Yahoo, CNBC, MarketWatch
def t_rss():
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher()
    articles = f.get_news_rss()
    sources = set(a["source"] for a in articles)
    print(f"    RSS: {len(articles)} articles from {sources}")
    assert len(articles) >= 1, "No RSS articles fetched"
    return True
check("RSS news feeds (Yahoo/CNBC/MarketWatch)", t_rss)

# 6. Finnhub — news (may fail without API key, that's OK)
def t_finnhub():
    from src.data.fetcher import FallbackFetcher
    f = FallbackFetcher(finnhub_key="")  # No key
    articles = f.get_news_finnhub()
    print(f"    Finnhub (no key): {len(articles)} articles — expected 0 without key")
    # Without a key, returns empty — that's correct behavior
    return True
check("Finnhub news (no key = graceful empty)", t_finnhub)

# 7. Ollama / DeepSeek R1 — LLM availability
def t_ollama():
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            has_deepseek = any("deepseek" in m for m in models)
            print(f"    Ollama models: {models}")
            print(f"    DeepSeek R1 available: {has_deepseek}")
            return True
        print(f"    Ollama returned {resp.status_code}")
        return False
    except Exception as e:
        print(f"    Ollama offline: {e}")
        return False
check("Ollama / DeepSeek R1 70B", t_ollama)

# 8. LLM Analyzer — neutral fallback
def t_llm_neutral():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    from src.llm.analyzer import LLMAnalyzer
    analyzer = LLMAnalyzer(cfg)
    neutral = analyzer.get_neutral_sentiment()
    print(f"    Neutral fallback: {neutral}")
    assert neutral["score"] == 0.0
    return True
check("LLM neutral sentiment fallback", t_llm_neutral)

# 9. Earnings calendar (yfinance)
def t_earnings():
    from src.data.earnings_calendar import fetch_earnings_yf
    earnings = fetch_earnings_yf()
    print(f"    Earnings: {len(earnings)} entries")
    # May be empty if no earnings this week — that's OK
    return True
check("Earnings calendar (yfinance)", t_earnings)

# 10. Fed communications
def t_fed():
    from src.data.fed_comms import get_fed_features
    import sqlite3, yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    conn = sqlite3.connect(cfg["database"]["path"])
    conn.row_factory = sqlite3.Row
    features = get_fed_features(conn, "2026-02-22")
    conn.close()
    print(f"    Fed features: {features}")
    assert "fomc_hawkish_score" in features
    return True
check("Fed communications features", t_fed)

# 11. Full pipeline data pull (non-Polygon path)
def t_daily_pull():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Force no Polygon
    cfg["polygon"]["api_key"] = "YOUR_POLYGON_KEY"
    from src.data.daily_pull import run_daily_pull
    counts = run_daily_pull(cfg)
    print(f"    Daily pull counts: {counts}")
    assert counts.get("prices", 0) > 0, "No prices after daily pull"
    return True
check("Full daily pull (no Polygon)", t_daily_pull)

# Summary
print(f"\n{'='*60}")
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"Results: {passed}/{total} passed")
for name, ok in results:
    print(f"  {'✓' if ok else '✗'} {name}")
