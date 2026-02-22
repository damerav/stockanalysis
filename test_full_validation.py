"""Full System Validation — Tests all P1, P2, P3 enhancements + core functionality."""
import sys, os, traceback
sys.path.insert(0, os.path.abspath("."))

results = []

def check(name, fn):
    try:
        ok = fn()
        status = "PASS" if ok else "FAIL"
    except Exception as e:
        status = f"FAIL ({e})"
        ok = False
    results.append((name, status))
    print(f"  {'✓' if ok else '✗'} {name}: {status}")
    return ok

print("=" * 70)
print("FULL SYSTEM VALIDATION — Core + P1 + P2 + P3")
print("=" * 70)

# =====================================================================
# 0. DB Migration
# =====================================================================
print("\n--- 0. Database Migration ---")
def t_migration():
    from src.data.init_db import get_connection
    conn = get_connection()
    conn.close()
    return True
check("DB migration runs cleanly", t_migration)

# =====================================================================
# 1. CORE: Database Schema
# =====================================================================
print("\n--- 1. Database Schema ---")
def t_schema():
    import sqlite3
    conn = sqlite3.connect("./data/spy.db")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    # Core tables
    core = ["prices", "technicals", "news", "daily_sentiment", "macro",
            "predictions", "intraday_bars", "options_chain",
            "options_analytics", "intraday_features", "performance"]
    # P2 tables
    p2 = ["model_registry", "feature_cache", "feature_store_meta"]
    # P3 tables
    p3 = ["earnings_calendar", "fed_communications"]
    missing = [t for t in core + p2 + p3 if t not in tables]
    assert not missing, f"Missing tables: {missing}"
    conn.close()
    return True
check("All tables exist (core + P2 + P3)", t_schema)

def t_schema_cols():
    import sqlite3
    conn = sqlite3.connect("./data/spy.db")
    # P1: macro VIX term structure
    macro_cols = [r[1] for r in conn.execute("PRAGMA table_info(macro)").fetchall()]
    for c in ["vix9d", "vix3m", "vix6m", "vvix", "skew_index",
              "hy_spread", "tlt_spy_ratio", "eem_spy_ratio",
              "copper_gold_ratio", "xlk_xlf_ratio", "xlk_xle_ratio"]:
        assert c in macro_cols, f"P1 macro col {c} missing"
    # P2: sentiment decomposition
    sent_cols = [r[1] for r in conn.execute("PRAGMA table_info(daily_sentiment)").fetchall()]
    for c in ["macro_sentiment", "earnings_sentiment", "geopolitical_sentiment",
              "technical_sentiment", "sentiment_dispersion", "sentiment_velocity"]:
        assert c in sent_cols, f"P2 sentiment col {c} missing"
    # P3: options analytics extended
    opt_cols = [r[1] for r in conn.execute("PRAGMA table_info(options_analytics)").fetchall()]
    for c in ["vanna_exposure", "charm_exposure", "zero_dte_pcr"]:
        assert c in opt_cols, f"P3 options col {c} missing"
    # P1: performance stratified
    perf_cols = [r[1] for r in conn.execute("PRAGMA table_info(performance)").fetchall()]
    for c in ["confidence_tier", "vix_regime", "day_of_week", "event_proximity"]:
        assert c in perf_cols, f"P1 performance col {c} missing"
    conn.close()
    return True
check("All P1/P2/P3 columns exist", t_schema_cols)

# =====================================================================
# 2. CORE: Feature Engineering
# =====================================================================
print("\n--- 2. Feature Engineering ---")
def t_features_count():
    from src.data.features import get_feature_columns
    cols = get_feature_columns()
    print(f"    Feature count: {len(cols)}")
    assert len(cols) == 85, f"Expected 85 features, got {len(cols)}"
    return True
check("85 features in get_feature_columns()", t_features_count)

def t_features_categories():
    from src.data.features import get_feature_columns
    cols = get_feature_columns()
    # P1 features
    p1 = ["vix9d", "vix3m", "vix6m", "vvix", "skew_index",
           "vix_term_slope", "vix_term_curve", "vix_realised_ratio",
           "hy_spread", "tlt_spy_ratio", "eem_spy_ratio",
           "copper_gold_ratio", "xlk_xlf_ratio", "xlk_xle_ratio",
           "days_to_fomc", "is_fomc_week", "is_fomc_day",
           "days_to_cpi", "days_to_nfp", "days_to_opex",
           "is_triple_witching", "is_quarter_end",
           "day_of_week", "week_of_month",
           "vix_percentile", "spy_es_zscore", "rth_flag",
           "minutes_to_close", "event_proximity"]
    # P2 features
    p2 = ["macro_sentiment", "earnings_sentiment",
           "geopolitical_sentiment", "technical_sentiment",
           "sentiment_dispersion", "sentiment_velocity"]
    # P3 features
    p3 = ["vanna_exposure", "charm_exposure", "zero_dte_pcr",
           "gex_sign_change", "max_pain_velocity",
           "vanna_normalized", "charm_normalized",
           "earnings_density", "days_to_next_mega", "earnings_week",
           "fomc_hawkish_score", "beige_book_score", "fed_sentiment_avg"]
    for cat_name, cat_list in [("P1", p1), ("P2", p2), ("P3", p3)]:
        missing = [c for c in cat_list if c not in cols]
        assert not missing, f"{cat_name} features missing: {missing}"
    return True
check("All P1/P2/P3 features present", t_features_categories)

def t_build_fv():
    from src.data.features import build_feature_vector
    import sqlite3
    conn = sqlite3.connect("./data/spy.db")
    fv = build_feature_vector(conn)
    conn.close()
    if fv is None or fv.empty:
        print("    (No data — skipped)")
        return True
    print(f"    Feature vector shape: {fv.shape}")
    assert fv.shape[1] > 90, f"Expected >90 columns, got {fv.shape[1]}"
    return True
check("build_feature_vector() runs", t_build_fv)

# =====================================================================
# 3. P1: VIX Term Structure + Cross-Asset + Calendar
# =====================================================================
print("\n--- 3. P1: VIX Term Structure + Cross-Asset + Calendar ---")
def t_p1_fetcher():
    from src.data.fetcher import FallbackFetcher
    ff = FallbackFetcher()
    assert hasattr(ff, 'get_vix_term_structure')
    assert hasattr(ff, 'get_cross_asset_signals')
    return True
check("FallbackFetcher has P1 methods", t_p1_fetcher)

def t_p1_calendar():
    from src.data.calendar import get_event_features
    from datetime import date
    feats = get_event_features(date(2026, 2, 20))
    assert "days_to_fomc" in feats
    assert "day_of_week" in feats
    return True
check("Calendar event features", t_p1_calendar)

def t_p1_drift():
    from src.data.drift_monitor import compute_psi, monitor_features
    assert callable(compute_psi)
    assert callable(monitor_features)
    return True
check("DriftMonitor functions exist", t_p1_drift)

# =====================================================================
# 4. P2: Ensemble + Conformal + Regime + Registry
# =====================================================================
print("\n--- 4. P2: Ensemble + Conformal + Regime + Registry ---")

def t_p2_bilstm():
    from src.model.bilstm_model import BiLSTMClassifier
    import torch
    model = BiLSTMClassifier(input_dim=10, hidden_dim=64)
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')
    return True
check("BiLSTM classifier class", t_p2_bilstm)

def t_p2_ensemble():
    from src.model.ensemble import StackingEnsemble
    se = StackingEnsemble({})
    assert hasattr(se, 'fit')
    assert hasattr(se, 'predict')
    return True
check("StackingEnsemble class", t_p2_ensemble)

def t_p2_conformal():
    from src.model.conformal import ConformalPredictor
    cp = ConformalPredictor()
    assert hasattr(cp, 'calibrate')
    assert hasattr(cp, 'predict_set')
    return True
check("ConformalPredictor class", t_p2_conformal)

def t_p2_regime():
    from src.model.regime import HMMRegimeDetector
    rd = HMMRegimeDetector()
    assert hasattr(rd, 'fit')
    assert hasattr(rd, 'predict')
    return True
check("HMMRegimeDetector class", t_p2_regime)

def t_p2_purged_cv():
    from src.model.purged_cv import PurgedWalkForwardCV
    cv = PurgedWalkForwardCV()
    assert hasattr(cv, 'split')
    return True
check("PurgedWalkForwardCV class", t_p2_purged_cv)

def t_p2_adaptive():
    from src.model.adaptive_window import select_optimal_window
    assert callable(select_optimal_window)
    return True
check("AdaptiveWindow function", t_p2_adaptive)

def t_p2_registry():
    from src.model.registry import ModelRegistry
    mr = ModelRegistry({})
    assert hasattr(mr, 'register')
    assert hasattr(mr, 'get_active')
    return True
check("ModelRegistry class", t_p2_registry)

def t_p2_feature_store():
    from src.data.feature_store import FeatureStore
    fs = FeatureStore({})
    assert hasattr(fs, 'get_features')
    return True
check("FeatureStore class", t_p2_feature_store)

# =====================================================================
# 5. P3: Extended Options + Earnings + Fed Comms
# =====================================================================
print("\n--- 5. P3: Extended Options + Earnings + Fed Comms ---")

def t_p3_polygon():
    from src.data.polygon_fetcher import PolygonFetcher
    pf = PolygonFetcher("dummy")
    assert hasattr(pf, '_calc_vanna')
    assert hasattr(pf, '_calc_charm')
    assert hasattr(pf, '_calc_zero_dte_pcr')
    return True
check("PolygonFetcher P3 methods", t_p3_polygon)

def t_p3_earnings():
    from src.data.earnings_calendar import fetch_earnings_yf, store_earnings, get_earnings_features
    import sqlite3
    conn = sqlite3.connect("./data/spy.db")
    feats = get_earnings_features(conn, "2026-02-20")
    assert "earnings_density" in feats
    assert "days_to_next_mega" in feats
    assert "earnings_week" in feats
    conn.close()
    return True
check("Earnings calendar features", t_p3_earnings)

def t_p3_fed():
    from src.data.fed_comms import get_fed_features, _keyword_score
    import sqlite3
    conn = sqlite3.connect("./data/spy.db")
    feats = get_fed_features(conn, "2026-02-20")
    assert "fomc_hawkish_score" in feats
    assert "beige_book_score" in feats
    assert "fed_sentiment_avg" in feats
    # Keyword scoring
    hawk = _keyword_score("inflation is persistent and elevated, we must tighten")
    assert hawk > 0, f"Expected hawkish > 0, got {hawk}"
    dove = _keyword_score("easing conditions, rate cut likely, patient approach")
    assert dove < 0, f"Expected dovish < 0, got {dove}"
    conn.close()
    return True
check("Fed comms features + keyword scoring", t_p3_fed)

# =====================================================================
# 6. Pipeline
# =====================================================================
print("\n--- 6. Pipeline ---")

def t_pipeline_steps():
    from src.pipeline.daily_run import DailyPipeline
    dp = DailyPipeline({})
    required = [
        '_step0_llm_check', '_step05_data_pull', '_step1_evaluate',
        '_step2_prices', '_step3_news', '_step4_sentiment',
        '_step5_macro', '_step6_options_chain', '_step7_options_analytics',
        '_step8_technicals', '_step9_intraday',
        '_step95_earnings', '_step96_fed_comms',
        '_step10_retrain', '_step11_predict', '_step12_report', '_step13_alerts',
    ]
    missing = [s for s in required if not hasattr(dp, s)]
    assert not missing, f"Missing pipeline steps: {missing}"
    print(f"    Pipeline steps: {len(required)}")
    return True
check("All 17 pipeline steps exist", t_pipeline_steps)

# =====================================================================
# 7. Model Layer
# =====================================================================
print("\n--- 7. Model Layer ---")

def t_trainer():
    from src.model.trainer import SPYPredictor
    sp = SPYPredictor({})
    assert hasattr(sp, 'train')
    assert hasattr(sp, 'predict')
    assert hasattr(sp, 'load_latest_model')
    return True
check("SPYPredictor class", t_trainer)

# =====================================================================
# 8. Dashboard
# =====================================================================
print("\n--- 8. Dashboard ---")

def t_dashboard_imports():
    from src.dashboard.monitoring import page_monitoring
    assert callable(page_monitoring)
    return True
check("Monitoring dashboard importable", t_dashboard_imports)

# =====================================================================
# 9. Config
# =====================================================================
print("\n--- 9. Configuration ---")

def t_config():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    required_sections = [
        "polygon", "llm", "xgboost", "technicals", "es_strategy",
        "alerts", "whatif", "database", "auth", "confidence_api",
        "ensemble", "conformal", "regime", "adaptive_window",
        "earnings", "fed_comms",
    ]
    missing = [s for s in required_sections if s not in cfg]
    assert not missing, f"Missing config sections: {missing}"
    return True
check("config.yaml has all sections", t_config)

# =====================================================================
# 10. Requirements
# =====================================================================
print("\n--- 10. Requirements ---")

def t_requirements():
    with open("requirements.txt") as f:
        content = f.read()
    required = ["xgboost", "torch", "streamlit", "plotly", "fastapi",
                "hmmlearn", "lightgbm", "feedparser", "shap"]
    missing = [r for r in required if r not in content]
    assert not missing, f"Missing requirements: {missing}"
    return True
check("requirements.txt complete", t_requirements)

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
passed = sum(1 for _, s in results if s == "PASS")
total = len(results)
print(f"FULL VALIDATION: {passed}/{total} passed")
if passed == total:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ✗")
    for name, status in results:
        if status != "PASS":
            print(f"  FAILED: {name} — {status}")
print("=" * 70)
sys.exit(0 if passed == total else 1)
