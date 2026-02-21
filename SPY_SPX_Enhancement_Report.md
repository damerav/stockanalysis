# SPY/SPX Prediction System — Detailed Review & Enhancement Report

**Author:** Vamshi Damera  
**Date:** February 21, 2026  
**Document Version:** 1.0  
**Scope:** Stock Analysis Platform — SPY/SPX Predictor Subsystem

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Solution Assessment](#2-current-solution-assessment)
   - 2.1 [Architecture Overview](#21-architecture-overview)
   - 2.2 [Strengths](#22-strengths)
   - 2.3 [Identified Gaps and Weaknesses](#23-identified-gaps-and-weaknesses)
3. [Enhancement Recommendations](#3-enhancement-recommendations)
   - 3.1 [Model Architecture Enhancements](#31-model-architecture-enhancements)
   - 3.2 [Feature Engineering Enhancements](#32-feature-engineering-enhancements)
   - 3.3 [Data Source Enhancements](#33-data-source-enhancements)
   - 3.4 [LLM and Sentiment Pipeline Enhancements](#34-llm-and-sentiment-pipeline-enhancements)
   - 3.5 [Training and Validation Enhancements](#35-training-and-validation-enhancements)
   - 3.6 [Prediction Output and Calibration Enhancements](#36-prediction-output-and-calibration-enhancements)
   - 3.7 [Pipeline and Operational Enhancements](#37-pipeline-and-operational-enhancements)
   - 3.8 [Infrastructure and Architecture Enhancements](#38-infrastructure-and-architecture-enhancements)
4. [Enhancement Prioritisation Matrix](#4-enhancement-prioritisation-matrix)
5. [Proposed Enhanced Architecture](#5-proposed-enhanced-architecture)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Expected Impact Summary](#7-expected-impact-summary)

---

## 1. Executive Summary

The Stock Analysis Platform represents a well-engineered, production-grade signal generation system that combines quantitative feature engineering, GPU-accelerated XGBoost classification, and large language model sentiment analysis to produce daily SPY/SPX direction predictions. The solution demonstrates strong engineering discipline, with robust fallback logic, a clean 13-step pipeline, and a well-structured multi-page dashboard.

However, a detailed review against the current state of academic research and practitioner best practices in ML-based equity prediction reveals a number of material opportunities to improve predictive accuracy, signal reliability, and operational resilience. These enhancements span seven domains: model architecture, feature engineering, data sourcing, LLM integration, training methodology, prediction calibration, and infrastructure.

This report provides a systematic, prioritised set of enhancement recommendations, each grounded in specific gaps identified in the current solution and supported by evidence from the research literature. The recommendations are categorised by impact and implementation complexity to allow incremental delivery without disrupting the live system.

---

## 2. Current Solution Assessment

### 2.1 Architecture Overview

The SPY/SPX Predictor subsystem operates as a daily batch pipeline that ingests data from five categories — technical indicators, macro variables, LLM-scored news sentiment, intraday microstructure, and options analytics — engineers a 37+ feature vector, and trains an XGBoost multi-class classifier (UP / NEUTRAL / DOWN) on a 252-day rolling window. The model outputs a 5-level directional prediction (STRONG BULLISH to STRONG BEARISH) with a 0–100% confidence score.

Key architectural characteristics are summarised below.

| Dimension | Current Implementation |
|-----------|----------------------|
| Primary model | XGBoost 2.0, `multi:softprob`, 3-class |
| Training window | 252 trading days rolling |
| Validation | Walk-forward 80/20 split, no shuffle |
| Feature count | 37+ across 5 categories |
| Sentiment | DeepSeek R1 70B via Ollama, 50 articles/day |
| Prediction granularity | 5-level ordinal scale |
| Retraining cadence | Daily at 4:30 PM ET |
| GPU acceleration | NVIDIA GB10 (DGX Spark), `gpu_hist` |
| Neutral threshold | ±0.3% daily return |
| Data storage | SQLite 3.45, WAL mode, 11 tables |

### 2.2 Strengths

The current solution exhibits several commendable design decisions that form a solid foundation for enhancement.

**Robust fallback and resilience design.** Every data source has a fallback path (Polygon → yfinance, LLM → neutral score), and the pipeline never aborts on step failure. This is a production-grade design choice that prevents silent data gaps.

**Multi-category feature engineering.** The combination of technical, macro, sentiment, intraday microstructure, and options analytics features is broader than most academic implementations, which typically rely on price-derived technicals alone. The inclusion of GEX, IV skew, and put/call ratio is particularly noteworthy.

**Daily retraining with walk-forward validation.** Retraining on the most recent 252 days with a proper time-series split (no shuffle, no data leakage) is methodologically sound and ensures the model adapts to evolving market regimes.

**GPU-accelerated training.** Leveraging the DGX Spark GB10 for both XGBoost training and LLM inference eliminates the latency bottleneck that would otherwise make daily retraining impractical with a 70B parameter model.

**What-If and stress-testing infrastructure.** The built-in Monte Carlo, feature ablation, and pre-built stress scenarios (VIX spike, March 2020 crash, melt-up) provide a meaningful framework for understanding model sensitivity — a capability absent from most comparable systems.

**Operational completeness.** The Admin Console, Telegram/email alerts, cloud relay, and shell script management layer represent a mature operational posture rarely seen in research-grade systems.

### 2.3 Identified Gaps and Weaknesses

The following gaps were identified through systematic review of the four documents against current best practices.

**Single-model dependency.** The prediction pipeline relies exclusively on a single XGBoost model. There is no ensemble or stacking layer that combines multiple model architectures. Research consistently demonstrates that heterogeneous ensembles (combining tree-based models with sequence models) reduce variance and improve directional accuracy by 3–8 percentage points over single-model approaches.

**Shallow temporal modelling.** XGBoost treats each feature vector as an independent observation. It has no native capacity to model sequential dependencies across days — for example, the persistence of a bearish sentiment trend over 5 days, or the pattern of VIX term structure inversion preceding corrections. LSTM, GRU, and Transformer architectures are specifically designed to capture such temporal patterns.

**Coarse sentiment pipeline.** The current sentiment pipeline processes up to 50 news articles per day through a general-purpose 70B LLM and produces a single aggregate score. This approach has two limitations: (a) DeepSeek R1, while powerful, is not fine-tuned on financial language, and (b) the aggregation discards the intra-day temporal distribution of news — a factor shown to have predictive value independent of the aggregate score.

**Limited options analytics depth.** The current options features (put/call ratio, max pain distance, IV skew, GEX) are computed from a daily snapshot. The system does not capture: (a) the VIX term structure (VIX9D / VIX3M / VIX6M ratios), (b) VVIX (volatility-of-volatility), (c) dealer vanna and charm exposures, or (d) 0DTE options flow — all of which have documented predictive value for next-day SPX direction.

**Absence of cross-asset signals.** The macro feature set (VIX, 10Y yield, DXY, fed funds, gold, crude) does not include: credit spreads (HYG/LQD ratio), equity breadth indicators (advance/decline, new highs/lows), sector rotation signals (XLK/XLF/XLE relative strength), or international equity correlations (EEM, EFA). These cross-asset signals are standard inputs in institutional equity prediction models.

**No probability calibration.** The XGBoost `softprob` output is used directly as a confidence score without post-hoc calibration. XGBoost classifiers are known to produce poorly calibrated probabilities — particularly in imbalanced or non-stationary financial datasets — which means the stated confidence percentages may not reflect true empirical probabilities.

**Fixed neutral threshold.** The ±0.3% neutral zone threshold is static. In a low-volatility regime (VIX ~12), ±0.3% represents a relatively large move, while in a high-volatility regime (VIX ~35), it is trivial. A regime-adaptive neutral threshold would produce more meaningful and consistent class labels across market conditions.

**Single training window.** The 252-day rolling window is fixed. There is no mechanism to test whether a shorter window (e.g., 126 days) performs better in trending regimes or a longer window (e.g., 504 days) performs better in mean-reverting regimes. Adaptive window selection based on detected market regime is a documented improvement.

**No feature importance drift monitoring.** The PSI (Population Stability Index) drift monitor exists in the ES strategy's AI models but is not applied to the SPY predictor's feature set. Feature distributions can shift materially after macro regime changes (e.g., post-Fed pivot), degrading model performance without triggering any alert.

**Absence of earnings and event calendar integration.** The pipeline has no awareness of scheduled macro events beyond the FOMC/CPI/NFP session guards in the ES strategy. The SPY predictor does not incorporate: days-to-earnings for major S&P 500 constituents, FOMC meeting proximity, or options expiration cycle effects (monthly OpEx, quarterly OpEx) — all of which materially affect next-day volatility and direction.

**SQLite scalability ceiling.** The current SQLite database with WAL mode is adequate for a single-user local deployment but introduces a hard ceiling on concurrent write throughput. As the intraday_bars table grows (5-second bars, 6.5 trading hours/day = ~4,680 bars/day), database size and query latency will increase, and the 5-second busy timeout may become a bottleneck.

**No backtesting accuracy stratification.** The performance table tracks cumulative accuracy but does not stratify accuracy by: confidence tier (high vs. low confidence predictions), volatility regime, day of week, or proximity to macro events. This stratification is essential for understanding when the model is reliable and when it should be discounted.

---

## 3. Enhancement Recommendations

### 3.1 Model Architecture Enhancements

#### 3.1.1 Heterogeneous Stacking Ensemble

**Current state:** Single XGBoost classifier.

**Enhancement:** Implement a two-layer stacking ensemble where the first layer consists of three diverse base learners — the existing XGBoost model, a Bidirectional LSTM (BiLSTM) operating on the last 20 days of features as a sequence, and a LightGBM model with different hyperparameters — and the second layer is a logistic regression meta-learner that combines the base learner probability outputs.

The BiLSTM is particularly valuable because it captures temporal dependencies that XGBoost cannot model. For example, a sequence of five consecutive days with declining breadth, rising VIX, and negative sentiment has a different predictive implication than a single day with those same values. Research by Kehinde et al. (2023) and the DLSTM paper (2025) demonstrate that LSTM-based models achieve 75–82% directional accuracy on daily SPY forecasts when combined with technical features, compared to 60–68% for standalone XGBoost.

```python
# Proposed stacking architecture (pseudocode)
class SPYStackingEnsemble:
    base_learners = [
        XGBoostClassifier(gpu_hist=True, ...),          # existing model
        BiLSTMClassifier(seq_len=20, hidden=128, ...),  # new: temporal
        LightGBMClassifier(num_leaves=63, ...)          # new: diversity
    ]
    meta_learner = CalibratedLogisticRegression()       # new: calibrated
    
    def fit(self, X_seq, y):
        # Train base learners on non-overlapping folds (purged CV)
        # Train meta-learner on out-of-fold predictions
        
    def predict_proba(self, X_seq):
        # Stack base learner probabilities → meta-learner → calibrated output
```

**Implementation complexity:** Medium-High. Requires adding PyTorch BiLSTM training to the pipeline, which is feasible given the existing PyTorch dependency for the CNN exit controller.

**Expected impact:** +4–8% directional accuracy improvement; reduced variance in confidence scores.

---

#### 3.1.2 Hidden Markov Model Regime Pre-filter

**Current state:** No market regime detection in the SPY predictor (regime detection exists only in the ES strategy's `RegimeDetector`).

**Enhancement:** Add a Hidden Markov Model (HMM) with 3–4 latent states (Bull Trending, Bear Trending, High-Volatility Choppy, Low-Volatility Ranging) trained on VIX, realised volatility, and rolling return features. The detected regime is then used to: (a) select the appropriate sub-model from a regime-specific model bank, (b) adjust the neutral threshold dynamically, and (c) weight the ensemble members differently per regime.

Research by Gupta et al. (2025) demonstrates that an ensemble-HMM voting framework improves regime-adjusted directional accuracy by 6–12% over a single-regime model, particularly in transition periods between bull and bear markets.

```python
# Regime-adaptive model selection (pseudocode)
class RegimeAdaptiveSPYPredictor:
    hmm = GaussianHMM(n_components=4, covariance_type='full')
    regime_models = {
        'bull_trend': StackingEnsemble(neutral_threshold=0.002),
        'bear_trend': StackingEnsemble(neutral_threshold=0.002),
        'high_vol':   StackingEnsemble(neutral_threshold=0.005),
        'low_vol':    StackingEnsemble(neutral_threshold=0.001),
    }
    
    def predict(self, features):
        regime = self.hmm.predict(features[-20:])[-1]
        return self.regime_models[regime].predict(features)
```

**Implementation complexity:** Medium. The `hmmlearn` library integrates cleanly with the existing pandas/numpy stack.

**Expected impact:** Improved accuracy in regime transitions; more meaningful confidence scores per market environment.

---

#### 3.1.3 Temporal Fusion Transformer (Optional Advanced Enhancement)

**Current state:** No attention-based sequence modelling.

**Enhancement:** For a longer-term advanced enhancement, implement a Temporal Fusion Transformer (TFT) that natively handles: multi-horizon forecasting, variable selection networks (automatically learning which features matter at each time step), and interpretable attention weights that show which historical days most influenced the current prediction.

The TFT architecture is particularly well-suited to this problem because it can simultaneously process static covariates (e.g., current macro regime), known future inputs (e.g., FOMC meeting in 2 days), and historical time series. The `pytorch-forecasting` library provides a production-ready TFT implementation compatible with the existing PyTorch environment.

**Implementation complexity:** High. Recommended as a Phase 3 enhancement after the stacking ensemble is validated.

**Expected impact:** Potential +3–5% additional accuracy; strong interpretability for understanding prediction drivers.

---

### 3.2 Feature Engineering Enhancements

#### 3.2.1 Extended Options Analytics: Dealer Greek Exposures

**Current state:** put/call ratio, max pain distance, IV skew, GEX (normalised).

**Enhancement:** Expand the options analytics feature set to include the following dealer Greek exposure metrics, which have strong documented predictive value for next-day SPX direction:

| New Feature | Description | Predictive Mechanism |
|-------------|-------------|---------------------|
| `vanna_exposure` | ΔDelta/ΔIV across all strikes | Dealer vanna hedging drives directional flows when IV changes |
| `charm_exposure` | ΔDelta/Δtime across all strikes | Charm hedging creates predictable end-of-day flows |
| `vix9d_vix_ratio` | VIX9D / VIX (short-term fear premium) | Ratio > 1 signals acute near-term fear; ratio < 0.8 signals complacency |
| `vix_vix3m_ratio` | VIX / VIX3M (term structure slope) | Backwardation (ratio > 1) historically precedes mean-reversion rallies |
| `vvix` | CBOE VVIX (volatility of VIX) | Elevated VVIX signals unstable volatility regime |
| `gex_sign_change` | Binary: GEX crossed zero in last 3 days | GEX sign changes mark transitions between pinning and trending regimes |
| `zero_dte_pcr` | Put/call ratio for 0DTE SPX options | 0DTE flow has become the dominant intraday directional signal since 2022 |
| `max_pain_velocity` | Rate of change of max pain level | Accelerating max pain shift signals upcoming pinning behaviour |

These features can be computed from the existing `options_chain` table data that is already being fetched from Polygon. The vanna and charm exposures require summing `gamma × delta × open_interest × spot_price` and `theta × delta × open_interest` across all strikes respectively.

**Implementation complexity:** Low-Medium. Requires extending `features.py` and `polygon_fetcher.py`'s `get_options_analytics()` method.

**Expected impact:** Options flow features are among the highest-information signals for next-day SPX direction; estimated +2–4% accuracy improvement.

---

#### 3.2.2 VIX Term Structure and Volatility Surface Features

**Current state:** Single `vix_level` and `vix_change` features.

**Enhancement:** Replace the single VIX scalar with a richer volatility surface representation:

```python
# New volatility surface features (to add to features.py)
volatility_features = {
    'vix_spot':           vix,                          # existing
    'vix_change':         vix - vix_prev,               # existing
    'vix9d':              fetch_vix9d(),                 # new
    'vix3m':              fetch_vix3m(),                 # new
    'vix6m':              fetch_vix6m(),                 # new
    'vvix':               fetch_vvix(),                  # new
    'vix_term_slope':     (vix3m - vix9d) / vix9d,      # new: contango/backwardation
    'vix_term_curve':     vix6m - 2*vix3m + vix9d,      # new: curvature
    'vix_realised_ratio': vix / realised_vol_20d,        # new: fear premium
    'skew_index':         fetch_cboe_skew(),             # new: tail risk
}
```

VIX9D, VIX3M, VIX6M, and VVIX are all available as free CBOE data via `yfinance` (tickers: `^VIX9D`, `^VIX3M`, `^VIX6M`, `^VVIX`). The CBOE SKEW index (`^SKEW`) measures the perceived tail risk in S&P 500 returns and has documented predictive value for large drawdowns.

**Implementation complexity:** Low. All tickers are available via the existing `yfinance` fallback fetcher.

**Expected impact:** Volatility surface features are among the most information-rich signals for equity direction; estimated +1–3% accuracy improvement.

---

#### 3.2.3 Cross-Asset and Market Breadth Features

**Current state:** Macro features limited to VIX, 10Y yield, DXY, fed funds, gold, crude.

**Enhancement:** Add the following cross-asset and breadth features:

| Feature | Ticker/Source | Rationale |
|---------|--------------|-----------|
| `hy_spread` | HYG/LQD ratio | Credit stress precedes equity stress by 1–3 days |
| `ig_spread` | LQD price change | Investment-grade credit as risk-off signal |
| `advance_decline_ratio` | NYSE A/D line | Breadth divergence signals distribution/accumulation |
| `new_highs_lows` | NYSE new 52W H/L | Breadth thrust signals; divergence from price is predictive |
| `spx_above_200ma_pct` | % of S&P 500 stocks above 200-day MA | Internal market health; below 50% historically bearish |
| `sector_rotation_score` | XLK vs XLF vs XLE relative strength | Risk-on/risk-off sector rotation signal |
| `eem_spy_ratio` | EEM / SPY | Emerging market risk appetite as leading indicator |
| `tlt_spy_ratio` | TLT / SPY | Flight-to-safety signal |
| `copper_gold_ratio` | CPER / GLD | Industrial demand vs. safe-haven demand |

All of these are available via `yfinance` at no additional cost and can be added to the existing `fetcher.py` FRED/macro fetch routine.

**Implementation complexity:** Low. Straightforward extension of `fetcher.py` and the `macro` database table.

**Expected impact:** Cross-asset signals capture systemic risk dynamics that price-only technicals miss; estimated +1–2% accuracy improvement.

---

#### 3.2.4 Calendar and Event-Aware Features

**Current state:** No awareness of scheduled events in the SPY predictor feature vector.

**Enhancement:** Add event calendar features that encode the proximity and type of scheduled macro events:

```python
# Event calendar features (new)
event_features = {
    'days_to_fomc':          calendar.days_until_next_fomc(),
    'is_fomc_week':          int(days_to_fomc <= 5),
    'is_fomc_day':           int(days_to_fomc == 0),
    'days_to_cpi':           calendar.days_until_next_cpi(),
    'days_to_nfp':           calendar.days_until_next_nfp(),
    'days_to_opex':          calendar.days_until_monthly_opex(),
    'is_triple_witching':    int(calendar.is_triple_witching_week()),
    'is_quarter_end':        int(calendar.is_quarter_end_week()),
    'sp500_earnings_pct':    calendar.pct_sp500_reporting_this_week(),
    'day_of_week':           date.weekday(),  # Monday effect, Friday effect
    'week_of_month':         calendar.week_of_month(),
}
```

The FOMC calendar is publicly available and deterministic. CPI and NFP release dates are published months in advance by the BLS. Monthly options expiration (third Friday) and quarterly triple witching are fully deterministic. These features encode the well-documented "event premium" effect where markets behave differently in the days surrounding scheduled catalysts.

**Implementation complexity:** Low-Medium. Requires building a lightweight economic calendar module.

**Expected impact:** Event proximity features are among the most consistent sources of edge in daily equity prediction; estimated +1–3% accuracy improvement, particularly in reducing false signals around FOMC weeks.

---

#### 3.2.5 Intraday Microstructure Enhancement

**Current state:** VWAP spread, intraday momentum, intraday range, volume ratio.

**Enhancement:** Extend the intraday feature set with higher-resolution microstructure signals:

| New Feature | Description |
|-------------|-------------|
| `opening_gap_pct` | Overnight gap as % of prior close (pre-market signal) |
| `opening_range_breakout` | Whether price broke above/below first 30-min range |
| `close_vs_high_pct` | Close relative to intraday high (distribution signal) |
| `close_vs_low_pct` | Close relative to intraday low (accumulation signal) |
| `afternoon_reversal` | Whether price reversed direction in last 90 minutes |
| `institutional_hour_vol` | Volume in 9:30–11:00 vs. 14:00–16:00 (institutional vs. retail) |
| `tick_divergence` | NYSE TICK extreme readings during the day |
| `vwap_reclaim_count` | Number of times price reclaimed VWAP after losing it |

The `opening_gap_pct` feature is particularly valuable: academic research consistently shows that overnight gaps have a mean-reversion tendency that is predictable at the daily level, and the direction of the gap combined with the first-hour price action is one of the strongest single-day directional signals available.

**Implementation complexity:** Low. All features can be computed from the existing `intraday_bars` table (5-second bars).

**Expected impact:** Estimated +1–2% accuracy improvement; particularly valuable for reducing false signals on gap days.

---

### 3.3 Data Source Enhancements

#### 3.3.1 Dark Pool and Institutional Flow Data

**Current state:** Options sweeps and block trades detected from Polygon WebSocket.

**Enhancement:** Integrate dark pool print data to capture institutional equity flow that does not appear in lit market options activity. Recommended sources:

- **Unusual Whales API** (paid): Provides dark pool prints, unusual options activity, congressional trading disclosures, and institutional 13F flow data in near-real-time.
- **FINRA ATS data** (free, T+1): Weekly dark pool volume by security, available from FINRA's OTC Transparency portal.
- **Cboe Global Indices** (free): Provides daily VVIX, SKEW, and term structure data.

Dark pool volume as a percentage of total volume is a documented leading indicator of institutional accumulation/distribution. A sustained increase in dark pool volume at a price level suggests institutional positioning that often precedes directional moves.

**Implementation complexity:** Medium. Requires a new API integration module, but the data pipeline architecture is already designed for extensibility.

---

#### 3.3.2 Earnings Calendar and SEC Filing Integration

**Current state:** No earnings or SEC filing data.

**Enhancement:** Integrate earnings calendar data to capture the "earnings season effect" on SPX direction:

- **Earnings Whispers API** or **Alpha Vantage Earnings** (free tier): Provides S&P 500 constituent earnings dates, EPS estimates, and surprise history.
- **SEC EDGAR RSS** (free): Real-time feed of 8-K filings (material events), 10-Q/10-K filings, and Form 4 (insider transactions).

The percentage of S&P 500 market cap reporting earnings in a given week is a strong predictor of realised volatility and can be used as a feature to adjust the model's confidence threshold. Weeks where >20% of S&P 500 market cap is reporting should trigger wider neutral zones and lower confidence scores.

**Implementation complexity:** Low-Medium. EDGAR RSS is free and straightforward to parse with the existing `feedparser` dependency.

---

#### 3.3.3 Federal Reserve Communication Data

**Current state:** Fed funds rate as a scalar macro feature.

**Enhancement:** Add structured features derived from Federal Reserve communications:

- **FOMC Statement Sentiment Score**: Run the LLM over the most recent FOMC statement to extract a hawkish/dovish score (-1 to +1). This is a one-time computation per FOMC meeting (8 times per year) and can be cached.
- **Fed Funds Futures Implied Rate**: The CME FedWatch tool provides the market-implied probability of rate changes at upcoming FOMC meetings. This forward-looking signal is more informative than the current backward-looking fed funds rate scalar.
- **Beige Book Sentiment**: The Fed's Beige Book is published 8 times per year and contains qualitative regional economic assessments. LLM sentiment scoring of the Beige Book has documented predictive value for equity direction over the following 2–4 weeks.

**Implementation complexity:** Medium. FOMC statements and Beige Books are publicly available from the Federal Reserve website. CME FedWatch data requires a CME API key or web scraping.

---

### 3.4 LLM and Sentiment Pipeline Enhancements

#### 3.4.1 Domain-Specific Sentiment Model: FinBERT as Fast Path

**Current state:** All 50 articles processed through DeepSeek R1 70B (~60–90 minutes per day).

**Enhancement:** Implement a two-tier sentiment pipeline:

1. **Fast path (FinBERT):** Run all articles through FinBERT (a BERT model fine-tuned on financial text from ProsusAI), which runs in seconds on GPU and produces financial-domain-calibrated sentiment scores. FinBERT is specifically trained on financial news and earnings call transcripts, making it more accurate than a general-purpose LLM for financial sentiment classification.

2. **Deep path (DeepSeek R1 70B):** Reserve the 70B LLM for: (a) the top 5 highest-impact articles identified by FinBERT, (b) FOMC statements and Fed communications, and (c) earnings call transcripts for major S&P 500 constituents reporting that week.

This hybrid approach reduces the daily sentiment pipeline from 60–90 minutes to approximately 5–10 minutes for the fast path, with the deep path running asynchronously and updating the sentiment score when complete.

```python
# Two-tier sentiment pipeline (pseudocode)
class HybridSentimentAnalyzer:
    finbert = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    deepseek = OllamaClient(model='deepseek-r1:70b')
    
    def analyze_daily(self, articles):
        # Fast path: all articles through FinBERT (< 60 seconds)
        fast_scores = [self.finbert(a.headline + ' ' + a.summary) 
                       for a in articles]
        
        # Identify top-impact articles for deep analysis
        top_articles = self.select_high_impact(articles, fast_scores, n=5)
        
        # Deep path: top articles + structured documents through DeepSeek
        deep_scores = self.deepseek.analyze_batch(top_articles)
        
        # Weighted aggregate: FinBERT base + DeepSeek premium for top articles
        return self.weighted_aggregate(fast_scores, deep_scores)
```

**Implementation complexity:** Low-Medium. FinBERT is available via HuggingFace `transformers` and runs efficiently on the DGX Spark GPU.

**Expected impact:** Reduces pipeline latency from 60–90 minutes to 5–10 minutes; improves sentiment accuracy for financial-specific language; enables more frequent intraday sentiment updates.

---

#### 3.4.2 Structured Sentiment Decomposition

**Current state:** Single aggregate sentiment score per day (-1.0 to 1.0).

**Enhancement:** Decompose the daily sentiment into structured sub-dimensions that carry independent predictive information:

| Sentiment Dimension | Description | Predictive Value |
|--------------------|-------------|-----------------|
| `macro_sentiment` | Sentiment of macro/Fed-related articles | Captures policy uncertainty |
| `earnings_sentiment` | Sentiment of earnings-related articles | Captures fundamental momentum |
| `geopolitical_sentiment` | Sentiment of geopolitical/risk articles | Captures tail risk perception |
| `technical_sentiment` | Sentiment of technical analysis articles | Captures crowd positioning |
| `sentiment_dispersion` | Std dev of article scores | High dispersion = uncertainty |
| `sentiment_velocity` | Change in sentiment vs. prior 3 days | Momentum of narrative shift |
| `source_credibility_score` | Weighted by source tier (WSJ > RSS) | Quality-adjusted signal |

This decomposition allows the model to distinguish between, for example, a day where macro sentiment is strongly negative but earnings sentiment is positive — a nuanced signal that a single aggregate score would obscure.

**Implementation complexity:** Medium. Requires modifying the LLM prompt to return structured JSON with category labels, and extending the `daily_sentiment` table schema.

---

#### 3.4.3 Social Media and Alternative Sentiment

**Current state:** News headlines from Finnhub and RSS feeds only.

**Enhancement:** Add structured social media sentiment from:

- **Reddit WallStreetBets and r/investing**: The Reddit API (free) provides post and comment data. Aggregate bullish/bearish mentions of SPY/SPX with volume weighting.
- **StockTwits API** (free tier): Provides real-time bullish/bearish sentiment counts for SPY, with historical data available.
- **Twitter/X Financial Sentiment**: Via RapidAPI or direct API, aggregate sentiment from financial accounts.

Social media sentiment, while noisier than news sentiment, captures retail investor positioning that has become increasingly relevant to SPX direction since the 2021 meme stock era. The key is to use it as a contrarian indicator at extremes (extreme retail bullishness is often a bearish signal) rather than a directional signal.

**Implementation complexity:** Medium. Requires new API integrations but fits cleanly into the existing sentiment pipeline architecture.

---

### 3.5 Training and Validation Enhancements

#### 3.5.1 Purged and Embargoed Walk-Forward Cross-Validation

**Current state:** Simple 80/20 walk-forward split with no embargo period.

**Enhancement:** Implement Combinatorial Purged Cross-Validation (CPCV) as described by Marcos López de Prado in *Advances in Financial Machine Learning*. This approach:

1. **Purges** observations from the training set that overlap with the validation set in terms of label formation (e.g., if the label for day T uses data from T to T+1, then day T-1 must be purged from training when validating on T).
2. **Embargoes** a gap period (typically 5 trading days) between the end of the training set and the start of the validation set to prevent information leakage through autocorrelated features.
3. **Combines** multiple non-overlapping validation paths to produce a more statistically robust accuracy estimate.

The current 80/20 split without embargo is susceptible to information leakage through autocorrelated features (e.g., a 14-day RSI computed on day 200 shares 13 days of data with the RSI computed on day 201). This can produce optimistically biased accuracy estimates.

```python
# Purged walk-forward CV (pseudocode)
class PurgedWalkForwardCV:
    def __init__(self, n_splits=5, embargo_days=5, purge_days=1):
        self.n_splits = n_splits
        self.embargo = embargo_days
        self.purge = purge_days
    
    def split(self, X, y, dates):
        # Generate non-overlapping train/test splits
        # Apply purge: remove training samples within purge_days of test start
        # Apply embargo: skip embargo_days between train end and test start
        # Yield purged train indices, test indices
```

**Implementation complexity:** Medium. The `mlfinlab` library provides a reference implementation of CPCV.

**Expected impact:** More reliable accuracy estimates; reduced risk of overfitting to in-sample patterns.

---

#### 3.5.2 Regime-Adaptive Neutral Threshold

**Current state:** Fixed ±0.3% neutral zone threshold.

**Enhancement:** Make the neutral threshold a function of the current volatility regime:

```python
# Regime-adaptive neutral threshold
def get_neutral_threshold(vix_level: float) -> float:
    """
    Scale neutral zone with VIX to maintain consistent signal quality
    across volatility regimes.
    """
    # ATR-based scaling: threshold = base_threshold × (VIX / VIX_baseline)
    vix_baseline = 18.0  # long-run VIX average
    base_threshold = 0.003  # current fixed threshold
    
    # Clamp to reasonable range [0.001, 0.008]
    adaptive_threshold = base_threshold * (vix_level / vix_baseline)
    return max(0.001, min(0.008, adaptive_threshold))
```

When VIX is at 12 (low volatility), the neutral zone narrows to ~±0.2%, correctly classifying small moves as directional. When VIX is at 35, the neutral zone widens to ~±0.58%, correctly classifying moderate moves as neutral in the context of elevated volatility.

**Implementation complexity:** Low. A one-line change to `trainer.py`'s `get_target()` function.

**Expected impact:** More consistent class label quality across market regimes; reduces label noise that degrades model performance.

---

#### 3.5.3 Adaptive Training Window Selection

**Current state:** Fixed 252-day rolling window.

**Enhancement:** Implement a dynamic window selector that tests multiple window lengths (63, 126, 252, 504 days) and selects the window with the best recent walk-forward accuracy:

```python
# Adaptive window selection (pseudocode)
class AdaptiveWindowTrainer:
    candidate_windows = [63, 126, 252, 504]
    
    def select_optimal_window(self, features_df, target_series):
        scores = {}
        for window in self.candidate_windows:
            # Train on window, validate on last 21 days
            score = self.walk_forward_score(features_df, target_series, window)
            scores[window] = score
        
        # Select window with best recent validation accuracy
        optimal = max(scores, key=scores.get)
        return optimal
```

In trending bull markets, longer windows (252–504 days) tend to perform better as they capture the persistent upward bias. In choppy or transitioning markets, shorter windows (63–126 days) adapt faster to the new regime. Automatically selecting the best window eliminates the need to manually tune this parameter.

**Implementation complexity:** Low-Medium. Adds ~4× training time but is well within the DGX Spark's GPU capacity.

---

### 3.6 Prediction Output and Calibration Enhancements

#### 3.6.1 Post-Hoc Probability Calibration

**Current state:** Raw `softprob` output used directly as confidence percentage.

**Enhancement:** Apply isotonic regression calibration to the XGBoost probability outputs. Isotonic regression is a non-parametric calibration method that maps the raw model probabilities to empirical probabilities using a monotone step function fitted on a held-out calibration set.

Research by the scikit-learn calibration module and Calibration Meets Reality (2025) demonstrates that isotonic regression outperforms Platt scaling (logistic calibration) in 18 of 20 tested scenarios for tree-based classifiers, particularly when the raw probabilities are bimodally distributed (as is common with XGBoost's `softprob` output).

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrated XGBoost wrapper
calibrated_model = CalibratedClassifierCV(
    xgb_model, 
    method='isotonic',
    cv='prefit'  # use separate calibration set
)
calibrated_model.fit(X_calibration, y_calibration)
```

The calibration set should be a recent 42-day (2-month) hold-out period that is not used in training, updated monthly.

**Implementation complexity:** Low. Requires adding a calibration step after model training in `trainer.py`.

**Expected impact:** Confidence scores become empirically meaningful (a 70% confidence prediction should be correct ~70% of the time); improves decision-making quality for the user.

---

#### 3.6.2 Prediction Uncertainty Quantification

**Current state:** Single point prediction with a single confidence score.

**Enhancement:** Add uncertainty quantification to the prediction output using conformal prediction intervals:

```python
# Conformal prediction for uncertainty quantification
class ConformaPredictionWrapper:
    def predict_with_uncertainty(self, X):
        # Get base model probabilities
        probs = self.model.predict_proba(X)
        
        # Compute conformal prediction set
        # Returns a set of classes that are consistent with the data
        # at the specified significance level (e.g., 90%)
        prediction_set = self.conformal_predictor.predict(X, significance=0.10)
        
        return {
            'point_prediction': probs.argmax(),
            'confidence': probs.max(),
            'prediction_set': prediction_set,  # e.g., {UP, NEUTRAL} at 90% coverage
            'uncertainty': 1 - probs.max(),
            'is_ambiguous': len(prediction_set) > 1
        }
```

When the conformal prediction set contains more than one class (e.g., {UP, NEUTRAL}), the dashboard should display the prediction as "LOW CONVICTION" rather than presenting a single direction with potentially misleading confidence. This prevents the user from acting on predictions where the model is genuinely uncertain.

**Implementation complexity:** Medium. The `nonconformist` or `MAPIE` library provides conformal prediction implementations.

**Expected impact:** Prevents overconfident signals; improves user trust calibration; reduces false high-confidence predictions.

---

#### 3.6.3 Stratified Accuracy Reporting

**Current state:** Cumulative accuracy tracked in the `performance` table without stratification.

**Enhancement:** Extend the `performance` table and the SPY Predictor dashboard page to report accuracy stratified by:

| Stratification Dimension | Why It Matters |
|--------------------------|---------------|
| Confidence tier (>70%, 50–70%, <50%) | Validates that high-confidence predictions are actually more accurate |
| Volatility regime (Low/Med/High VIX) | Identifies regimes where the model is reliable vs. unreliable |
| Day of week | Captures day-of-week effects (e.g., Monday reversals, Friday momentum) |
| FOMC proximity (within 5 days vs. not) | Quantifies model degradation around Fed events |
| Prediction direction (UP vs. DOWN) | Identifies directional bias in the model |
| Trailing 21-day accuracy | Rolling accuracy to detect model drift early |

This stratification should be displayed as a breakdown table on the SPY Predictor dashboard page and should trigger an alert if the trailing 21-day accuracy drops below a configurable threshold (e.g., 52%).

**Implementation complexity:** Low-Medium. Requires schema extension and dashboard update.

---

### 3.7 Pipeline and Operational Enhancements

#### 3.7.1 Feature Drift Monitoring for SPY Predictor

**Current state:** PSI drift monitoring exists in the ES strategy's AI models but not in the SPY predictor.

**Enhancement:** Apply Population Stability Index (PSI) monitoring to all 37+ SPY predictor features, with alerts triggered when PSI > 0.2 for any feature. Additionally, implement Kolmogorov-Smirnov (KS) tests comparing the current 21-day feature distribution against the training distribution.

```python
# Feature drift monitor (pseudocode)
class SPYFeatureDriftMonitor:
    def __init__(self, training_distributions: dict):
        self.baseline = training_distributions  # {feature: (mean, std, percentiles)}
    
    def check_drift(self, current_features: pd.DataFrame) -> dict:
        drift_report = {}
        for feature in current_features.columns:
            psi = self.compute_psi(
                self.baseline[feature], 
                current_features[feature].values
            )
            drift_report[feature] = {
                'psi': psi,
                'status': 'DRIFT' if psi > 0.2 else 'WARN' if psi > 0.1 else 'OK'
            }
        return drift_report
```

When drift is detected in high-importance features (as determined by XGBoost feature importance), the system should: (a) trigger an immediate retraining with a shorter window to adapt faster, (b) reduce the confidence score by a drift penalty factor, and (c) send an alert to the user.

**Implementation complexity:** Low-Medium. Extends the existing PSI infrastructure from `ai_models.py`.

---

#### 3.7.2 Intraday Prediction Updates

**Current state:** Single prediction generated at 4:30 PM ET, valid for the next trading day.

**Enhancement:** Add an intraday prediction update mechanism that refreshes the prediction at two additional times:

1. **Pre-market update (8:30 AM ET):** After overnight news, pre-market futures movement, and any 8:30 AM economic releases (CPI, NFP) are available. This update uses the same model but with refreshed sentiment and macro features.

2. **Mid-day update (12:00 PM ET):** After the morning session's price action, options flow, and any Fed communications are incorporated. The intraday features (VWAP spread, momentum, volume ratio) are updated with the morning session data.

Each update should be clearly timestamped and displayed on the dashboard with a "freshness" indicator. The original 4:30 PM prediction should be retained as the "baseline" prediction.

**Implementation complexity:** Medium. Requires scheduling two additional pipeline runs (Steps 3–12 only, skipping the full LLM sentiment) and updating the dashboard to display multiple predictions.

---

#### 3.7.3 Automated Model Performance Gating

**Current state:** Model is retrained and deployed daily without performance validation.

**Enhancement:** Implement a performance gate that prevents a newly trained model from being deployed if its walk-forward validation accuracy falls below a minimum threshold:

```python
# Model performance gate (pseudocode)
class ModelPerformanceGate:
    MIN_ACCURACY = 0.52      # minimum directional accuracy
    MIN_IMPROVEMENT = -0.02  # max allowed degradation vs. prior model
    
    def should_deploy(self, new_model, prior_model, X_val, y_val):
        new_acc = accuracy_score(y_val, new_model.predict(X_val))
        prior_acc = accuracy_score(y_val, prior_model.predict(X_val))
        
        if new_acc < self.MIN_ACCURACY:
            logger.warning(f"Model rejected: accuracy {new_acc:.3f} < minimum {self.MIN_ACCURACY}")
            return False
        
        if new_acc < prior_acc + self.MIN_IMPROVEMENT:
            logger.warning(f"Model rejected: accuracy degraded {prior_acc:.3f} → {new_acc:.3f}")
            return False
        
        return True
```

If the gate rejects the new model, the system continues using the prior day's model and sends an alert. This prevents a bad training run (e.g., due to corrupted data or a data source outage) from degrading the live prediction.

**Implementation complexity:** Low. A straightforward addition to `trainer.py`.

---

#### 3.7.4 Prediction Explanation Layer

**Current state:** LLM generates a ~400-word narrative report, but there is no structured explanation of which features drove the prediction.

**Enhancement:** Add SHAP (SHapley Additive exPlanations) values to the prediction output to provide a structured, quantitative explanation of the top contributing features:

```python
import shap

# SHAP explanation for each prediction
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_today)

# Top 5 features driving the prediction
top_drivers = pd.Series(
    shap_values[predicted_class], 
    index=feature_names
).abs().nlargest(5)
```

The SHAP values should be displayed on the SPY Predictor dashboard as a horizontal bar chart showing the top 5 positive and negative contributors to the current prediction. This provides the user with an interpretable, quantitative basis for the prediction rather than relying solely on the LLM narrative.

**Implementation complexity:** Low. The `shap` library has native XGBoost support and runs in milliseconds.

**Expected impact:** Dramatically improves user trust and decision-making quality; enables the user to validate that the prediction is driven by sensible features rather than spurious correlations.

---

### 3.8 Infrastructure and Architecture Enhancements

#### 3.8.1 Database Migration to TimescaleDB or DuckDB

**Current state:** SQLite 3.45 with WAL mode.

**Enhancement:** Migrate the time-series tables (`prices`, `technicals`, `macro`, `intraday_bars`, `options_chain`) to either:

- **TimescaleDB** (PostgreSQL extension): Provides automatic time-series partitioning, compression, and continuous aggregates. Particularly valuable for the `intraday_bars` table, which will grow to millions of rows over time. TimescaleDB can compress time-series data by 90–95% and provides 10–100× faster range queries.

- **DuckDB** (analytical in-process database): An excellent alternative for read-heavy analytical workloads. DuckDB's columnar storage and vectorised execution engine can execute the feature engineering queries 10–50× faster than SQLite, which is particularly valuable as the training window grows.

The `daily_sentiment`, `predictions`, and `performance` tables can remain in SQLite as they are small and write-heavy.

**Implementation complexity:** High. Requires schema migration and updating all database access code. Recommended as a Phase 2 enhancement.

---

#### 3.8.2 Feature Store Implementation

**Current state:** Features are recomputed from scratch at each pipeline run.

**Enhancement:** Implement a lightweight feature store that caches computed features and only recomputes features for new dates:

```python
# Feature store (pseudocode)
class FeatureStore:
    def get_features(self, date_range: tuple) -> pd.DataFrame:
        cached = self.load_from_cache(date_range)
        missing_dates = self.find_missing(date_range, cached)
        
        if missing_dates:
            new_features = self.compute_features(missing_dates)
            self.save_to_cache(new_features)
            return pd.concat([cached, new_features]).sort_index()
        
        return cached
```

This eliminates the redundant recomputation of features for the 251 historical days that have not changed, reducing the feature engineering step from O(N) to O(1) for the common case.

**Implementation complexity:** Low. Can be implemented as a simple SQLite-backed cache within the existing `features.py` module.

---

#### 3.8.3 Model Registry and Versioning

**Current state:** Models saved as `xgb_spy_{date}.json` with no structured versioning or metadata.

**Enhancement:** Implement a lightweight model registry that stores, for each trained model:

| Metadata Field | Description |
|---------------|-------------|
| `model_id` | UUID |
| `training_date` | Date model was trained |
| `training_window` | Number of days used |
| `feature_set_version` | Hash of feature column list |
| `validation_accuracy` | Walk-forward accuracy |
| `calibration_score` | Brier score on calibration set |
| `deployment_status` | active / shadow / retired |
| `shap_importance` | Top 10 features and their importance |

This registry enables: (a) A/B testing between model versions (shadow deployment), (b) rollback to a prior model if performance degrades, and (c) tracking of how model accuracy and feature importance evolve over time.

**Implementation complexity:** Low-Medium. Can be implemented as a new SQLite table or a JSON file in the `models/` directory.

---

#### 3.8.4 Real-Time Confidence API Enhancement

**Current state:** FastAPI confidence API on port 8100 with three endpoints.

**Enhancement:** Add the following endpoints to the confidence API to support richer integration:

| New Endpoint | Method | Description |
|-------------|--------|-------------|
| `GET /prediction/current` | GET | Current SPY prediction with SHAP explanation |
| `GET /prediction/history` | GET | Last N predictions with accuracy |
| `GET /features/current` | GET | Current feature vector with drift status |
| `GET /model/metadata` | GET | Current model metadata from registry |
| `POST /prediction/override` | POST | Manual override for special events (admin only) |
| `GET /calibration/curve` | GET | Reliability diagram data for confidence calibration |

These endpoints enable external systems (trading platforms, risk management tools, custom dashboards) to consume the prediction data in a structured, programmatic way.

**Implementation complexity:** Low. Straightforward FastAPI endpoint additions.

---

## 4. Enhancement Prioritisation Matrix

The following matrix prioritises all enhancements by expected accuracy impact and implementation complexity, enabling a phased delivery approach.

| Enhancement | Category | Accuracy Impact | Complexity | Priority |
|-------------|----------|----------------|------------|----------|
| Regime-adaptive neutral threshold | Training | High | Low | **P1** |
| Post-hoc probability calibration | Output | High | Low | **P1** |
| SHAP prediction explanation | Output | Medium | Low | **P1** |
| Feature drift monitoring | Operations | High | Low-Medium | **P1** |
| Extended options analytics (vanna, charm, VIX term structure) | Features | High | Low-Medium | **P1** |
| Calendar/event-aware features | Features | High | Low-Medium | **P1** |
| Cross-asset and breadth features | Features | Medium | Low | **P1** |
| Model performance gating | Operations | Medium | Low | **P1** |
| FinBERT fast-path sentiment | LLM | Medium | Low-Medium | **P1** |
| Stratified accuracy reporting | Output | Medium | Low-Medium | **P1** |
| Purged walk-forward CV | Training | Medium | Medium | **P2** |
| Adaptive training window | Training | Medium | Low-Medium | **P2** |
| Structured sentiment decomposition | LLM | Medium | Medium | **P2** |
| Intraday microstructure enhancement | Features | Medium | Low | **P2** |
| Intraday prediction updates | Pipeline | Medium | Medium | **P2** |
| Conformal prediction uncertainty | Output | Medium | Medium | **P2** |
| BiLSTM stacking ensemble | Model | High | Medium-High | **P2** |
| HMM regime pre-filter | Model | High | Medium | **P2** |
| Feature store implementation | Infrastructure | Low | Low | **P2** |
| Model registry and versioning | Infrastructure | Low | Low-Medium | **P2** |
| Social media sentiment | Data | Low-Medium | Medium | **P3** |
| Dark pool flow integration | Data | Medium | Medium | **P3** |
| Earnings calendar integration | Data | Medium | Low-Medium | **P3** |
| Fed communication NLP | Data | Medium | Medium | **P3** |
| TimescaleDB/DuckDB migration | Infrastructure | Low | High | **P3** |
| Temporal Fusion Transformer | Model | High | High | **P3** |
| Real-time API enhancement | Infrastructure | Low | Low | **P3** |

---

## 5. Proposed Enhanced Architecture

The following diagram describes the enhanced SPY/SPX prediction architecture after implementing all Priority 1 and Priority 2 enhancements.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENHANCED SPY/SPX PREDICTOR                           │
│                                                                         │
│  ┌──────────────────────── DATA LAYER (ENHANCED) ──────────────────┐   │
│  │                                                                  │   │
│  │  Polygon.io  yfinance  Finnhub  FRED  CBOE  Unusual Whales      │   │
│  │  (existing)  (fallback) (news)  (macro) (VIX term) (dark pool)  │   │
│  │       │          │        │       │        │           │         │   │
│  │       └──────────┴────────┴───────┴────────┴───────────┘         │   │
│  │                              │                                   │   │
│  │                    ┌─────────▼─────────┐                         │   │
│  │                    │  Feature Store    │  (new: cached features) │   │
│  │                    └─────────┬─────────┘                         │   │
│  └──────────────────────────────┼──────────────────────────────────┘   │
│                                 │                                       │
│  ┌──────────────────── FEATURE ENGINEERING (ENHANCED) ─────────────┐   │
│  │                             │                                    │   │
│  │  Technical (existing) + Options Greeks (vanna/charm/vix term)   │   │
│  │  Macro (extended: breadth, credit spreads, cross-asset)          │   │
│  │  Calendar (new: FOMC proximity, OpEx, earnings season)           │   │
│  │  Intraday (enhanced: gap, opening range, close vs. high/low)     │   │
│  │  Sentiment (decomposed: macro/earnings/geo/technical dims)       │   │
│  │                             │                                    │   │
│  │  Feature Drift Monitor ─────┤  (new: PSI + KS test per feature) │   │
│  └──────────────────────────────┼──────────────────────────────────┘   │
│                                 │                                       │
│  ┌──────────────────── SENTIMENT PIPELINE (ENHANCED) ──────────────┐   │
│  │                                                                  │   │
│  │  FinBERT (fast path, all articles, <60s) ──────────────────────► │   │
│  │  DeepSeek R1 70B (deep path, top 5 articles + FOMC/earnings)    │   │
│  │                             │                                    │   │
│  │  Structured output: {macro, earnings, geo, technical, velocity}  │   │
│  └──────────────────────────────┼──────────────────────────────────┘   │
│                                 │                                       │
│  ┌──────────────────── COMPUTE LAYER (ENHANCED) ────────────────────┐  │
│  │                             │                                    │  │
│  │  ┌──────────────────────────┼──────────────────────────────┐    │  │
│  │  │           HMM Regime Detector (new)                      │    │  │
│  │  │  Bull Trend │ Bear Trend │ High-Vol Choppy │ Low-Vol Rng │    │  │
│  │  └──────────────────────────┬──────────────────────────────┘    │  │
│  │                             │ regime signal                     │  │
│  │                             ▼                                   │  │
│  │  ┌──────────────────────────────────────────────────────────┐   │  │
│  │  │           Stacking Ensemble (new)                         │   │  │
│  │  │                                                           │   │  │
│  │  │  XGBoost (GPU)  +  BiLSTM (seq=20)  +  LightGBM          │   │  │
│  │  │  (existing,         (new: temporal    (new: diversity)    │   │  │
│  │  │   enhanced)          dependencies)                        │   │  │
│  │  │                             │                             │   │  │
│  │  │  Logistic Regression Meta-Learner (calibrated)            │   │  │
│  │  └──────────────────────────────────────────────────────────┘   │  │
│  │                             │                                   │  │
│  │  Adaptive Window Selector ──┤  (new: 63/126/252/504 day test)  │  │
│  │  Purged Walk-Forward CV ────┤  (new: embargo + purge)          │  │
│  │  Model Performance Gate ────┤  (new: min accuracy threshold)   │  │
│  │  Isotonic Calibration ──────┤  (new: empirical probabilities)  │  │
│  │  Conformal Prediction ──────┤  (new: prediction sets)          │  │
│  └──────────────────────────────┼──────────────────────────────────┘  │
│                                 │                                      │
│  ┌──────────────────── OUTPUT LAYER (ENHANCED) ─────────────────────┐ │
│  │                             │                                    │ │
│  │  Prediction: 5-level scale + calibrated confidence               │ │
│  │  SHAP explanation: top 5 feature drivers (new)                   │ │
│  │  Conformal prediction set: {UP} or {UP, NEUTRAL} (new)           │ │
│  │  Stratified accuracy: by regime, confidence, day-of-week (new)   │ │
│  │  Drift status: feature drift alerts (new)                        │ │
│  │  Model registry: version, accuracy, calibration score (new)      │ │
│  └──────────────────────────────┼──────────────────────────────────┘ │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │
                     ┌────────────▼────────────┐
                     │  Enhanced Dashboard      │
                     │  + SHAP waterfall chart  │
                     │  + Stratified accuracy   │
                     │  + Drift status panel    │
                     │  + Calibration curve     │
                     └─────────────────────────┘
```

---

## 6. Implementation Roadmap

### Phase 1 — Quick Wins (Weeks 1–4)

Phase 1 focuses on enhancements that deliver immediate accuracy and reliability improvements with low implementation risk. All Phase 1 items require changes only to `features.py`, `trainer.py`, and `app.py` — no new dependencies or architectural changes.

| Week | Deliverable | Files Changed |
|------|-------------|--------------|
| 1 | Regime-adaptive neutral threshold | `trainer.py` |
| 1 | Post-hoc isotonic calibration | `trainer.py` |
| 1 | SHAP explanation in predictions | `trainer.py`, `app.py` |
| 2 | VIX term structure features (VIX9D, VIX3M, VVIX) | `fetcher.py`, `features.py` |
| 2 | Cross-asset features (HYG/LQD, breadth, TLT/SPY) | `fetcher.py`, `features.py` |
| 3 | Calendar/event features (FOMC, CPI, OpEx proximity) | new `calendar.py`, `features.py` |
| 3 | Model performance gating | `trainer.py` |
| 4 | Feature drift monitoring (PSI + KS) | new `drift_monitor.py` |
| 4 | Stratified accuracy reporting | `init_db.py`, `app.py` |

### Phase 2 — Core Enhancements (Weeks 5–12)

Phase 2 delivers the most impactful structural improvements: the stacking ensemble, HMM regime detection, improved validation, and the FinBERT sentiment fast path.

| Week | Deliverable | Files Changed |
|------|-------------|--------------|
| 5–6 | FinBERT fast-path sentiment pipeline | `analyzer.py`, `daily_run.py` |
| 5–6 | Structured sentiment decomposition | `analyzer.py`, `init_db.py` |
| 7–8 | HMM regime detector | new `regime.py`, `trainer.py` |
| 7–8 | Adaptive training window selection | `trainer.py` |
| 9–10 | BiLSTM base learner | new `bilstm_model.py`, `trainer.py` |
| 9–10 | Stacking ensemble with meta-learner | `trainer.py` |
| 11 | Purged walk-forward cross-validation | `trainer.py` |
| 11 | Conformal prediction uncertainty | `trainer.py`, `app.py` |
| 12 | Intraday prediction updates (8:30 AM, 12:00 PM) | `daily_run.py`, `launcher.py` |

### Phase 3 — Advanced Enhancements (Weeks 13–20)

Phase 3 delivers the advanced data integrations, infrastructure improvements, and the Temporal Fusion Transformer.

| Week | Deliverable | Files Changed |
|------|-------------|--------------|
| 13–14 | Extended options analytics (vanna, charm, 0DTE PCR) | `polygon_fetcher.py`, `features.py` |
| 13–14 | Earnings calendar integration | new `earnings_calendar.py`, `features.py` |
| 15–16 | Dark pool flow integration (Unusual Whales API) | new `darkpool_fetcher.py`, `features.py` |
| 15–16 | Fed communication NLP (FOMC statements, Beige Book) | `analyzer.py` |
| 17–18 | Feature store implementation | new `feature_store.py` |
| 17–18 | Model registry and versioning | new `model_registry.py` |
| 19–20 | Temporal Fusion Transformer | new `tft_model.py`, `trainer.py` |
| 19–20 | TimescaleDB/DuckDB migration (optional) | `init_db.py`, all data modules |

---

## 7. Expected Impact Summary

The following table summarises the expected cumulative impact of each phase on the key performance metrics of the SPY/SPX predictor.

| Metric | Current Baseline | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|-----------------|---------------|---------------|---------------|
| Directional accuracy (estimated) | ~58–62% | ~62–65% | ~65–70% | ~68–74% |
| Confidence calibration (Brier score) | Uncalibrated | Calibrated | Well-calibrated | Excellent |
| Sentiment pipeline latency | 60–90 min | 60–90 min | 5–10 min | 5–10 min |
| Feature count | 37+ | 55+ | 65+ | 80+ |
| Regime awareness | None | None | HMM (4 states) | HMM + TFT |
| Prediction freshness | 1×/day | 1×/day | 3×/day | 3×/day |
| Explainability | LLM narrative | SHAP + narrative | SHAP + narrative | Full SHAP + attention |
| Drift detection | None | PSI + KS | PSI + KS + alerts | Full monitoring |
| Model validation | Simple 80/20 | Simple 80/20 | Purged CPCV | Purged CPCV |

> **Important caveat:** Directional accuracy estimates are based on published academic research for comparable feature sets and model architectures on daily SPX/SPY data. Actual performance will depend on the specific data quality, market regime during the evaluation period, and implementation details. No accuracy improvement is guaranteed in live trading conditions. The system remains signal-only; all trade execution is manual.

---

*This report was prepared based on a detailed review of the SOLUTION_DOCUMENT.md, USER_GUIDE.md, ADMIN_GUIDE.md, and ARCHITECTURE.md documents, cross-referenced against current academic literature and practitioner best practices in ML-based equity direction prediction as of February 2026.*
