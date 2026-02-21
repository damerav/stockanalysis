# Stock Analysis Platform — Architecture Reference

## 1. System Overview

The platform runs on an NVIDIA DGX Spark local GPU server with optional AWS cloud mirroring. It combines quantitative analysis, machine learning (XGBoost on GPU), and large language model inference (DeepSeek R1 70B via Ollama) to produce daily SPY/SPX direction predictions and intraday ES futures trading signals.

Signal-only system — all trade execution is manual.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA DGX Spark (192.168.1.211)                         │
│                                                                             │
│  ┌─────────────────────── DATA SOURCES ───────────────────────┐            │
│  │                                                             │            │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │            │
│  │  │ Polygon   │  │ yfinance │  │ Finnhub  │  │ FRED      │  │            │
│  │  │ REST +    │  │ (fallback│  │ + RSS    │  │ Macro     │  │            │
│  │  │ WebSocket │  │  prices) │  │ (news)   │  │ (VIX,etc) │  │            │
│  │  └─────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │            │
│  │        └──────────────┴─────────────┴──────────────┘        │            │
│  └─────────────────────────────┬───────────────────────────────┘            │
│                                │                                            │
│  ┌─────────────────────── DATA LAYER ─────────────────────────┐            │
│  │                             │                               │            │
│  │                      ┌──────▼──────┐                        │            │
│  │                      │  SQLite DB  │  ./data/spy.db         │            │
│  │                      │  11 tables  │  WAL mode              │            │
│  │                      │  5s timeout │                        │            │
│  │                      └──────┬──────┘                        │            │
│  │                             │                               │            │
│  │  ┌──────────────────────────┼──────────────────────────┐   │            │
│  │  │         Feature Engineering (37+ features)           │   │            │
│  │  │  Technical │ Macro │ Sentiment │ Intraday │ Options  │   │            │
│  │  └──────────────────────────┬──────────────────────────┘   │            │
│  └─────────────────────────────┼───────────────────────────────┘            │
│                                │                                            │
│  ┌──────────── COMPUTE LAYER ──┼──────────────────────────────┐            │
│  │                             │                               │            │
│  │  ┌───────────┐  ┌──────────▼─────────┐  ┌──────────────┐  │            │
│  │  │ Ollama    │  │ XGBoost Classifier  │  │ ES Strategy  │  │            │
│  │  │ DeepSeek  │  │ GPU: gpu_hist       │  │ Engine       │  │            │
│  │  │ R1 70B    │  │ 3-class softprob    │  │ Keltner +    │  │            │
│  │  │ (~42GB)   │  │ 252-day rolling     │  │ 3-lot tiered │  │            │
│  │  └─────┬─────┘  └──────────┬─────────┘  └──────┬───────┘  │            │
│  │        │                   │                    │           │            │
│  │        │  ┌────────────────┼────────────────────┤           │            │
│  │        │  │                │                    │           │            │
│  │  ┌─────▼──▼────┐  ┌───────▼────────┐  ┌───────▼────────┐ │            │
│  │  │ What-If     │  │ Dashboard      │  │ Confidence API │ │            │
│  │  │ Engine      │  │ Bridge         │  │ FastAPI :8100  │ │            │
│  │  │ (scenarios) │  │ (JSON state)   │  │ /confidence    │ │            │
│  │  └─────────────┘  └───────┬────────┘  │ /exit /spread  │ │            │
│  │                           │           └────────┬───────┘  │            │
│  └───────────────────────────┼────────────────────┼──────────┘            │
│                               │                                             │
│  ┌──────────── PRESENTATION LAYER ────────────────────────────┐            │
│  │                            │                                │            │
│  │              ┌─────────────▼─────────────┐                  │            │
│  │              │   Unified Dashboard       │                  │            │
│  │              │   Streamlit :8501          │                  │            │
│  │              │                           │                  │            │
│  │              │  ┌─────┐ ┌────┐ ┌──────┐ │                  │            │
│  │              │  │ SPY │ │ ES │ │What- │ │  ┌───────────┐   │            │
│  │              │  │Pred.│ │Str.│ │ If   │ │  │  Admin    │   │            │
│  │              │  └─────┘ └────┘ └──────┘ │  │  Console  │   │            │
│  │              │                           │  │  (5 tabs) │   │            │
│  │              └─────────────┬─────────────┘  └───────────┘   │            │
│  └────────────────────────────┼────────────────────────────────┘            │
│                               │                                             │
│  ┌──────────── CLOUD SYNC (OPTIONAL) ─────────────────────────┐            │
│  │              ┌─────────────▼─────────────┐                  │            │
│  │              │  Cloud Publisher           │                  │            │
│  │              │  HTTPS POST → AWS Relay    │                  │            │
│  │              └─────────────┬─────────────┘                  │            │
│  └────────────────────────────┼────────────────────────────────┘            │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │ HTTPS
                    ┌───────────▼───────────┐
                    │  AWS EC2 t3.micro      │
                    │  FastAPI Relay :8000   │
                    │  Cloud Dashboard :8501 │
                    │  (in-memory, no DB)    │
                    └───────────────────────┘
```

---

## 3. Technology Stack

```
┌────────────────────────────────────────────────────────────┐
│                    APPLICATION STACK                        │
├──────────────┬─────────────────┬───────────────────────────┤
│ Layer        │ Technology      │ Purpose                   │
├──────────────┼─────────────────┼───────────────────────────┤
│ Runtime      │ Python 3.12     │ Application language      │
│ Database     │ SQLite 3.45     │ Local store, WAL mode     │
│ ML           │ XGBoost 2.0     │ SPY classifier (GPU)      │
│ Deep Learn   │ PyTorch 2.1     │ CNN exit controller       │
│ LLM          │ Ollama+DeepSeek │ Sentiment, reports        │
│ Dashboard    │ Streamlit 1.30  │ Unified web UI            │
│ Charts       │ Plotly 5.18     │ Interactive visuals       │
│ Data Sources │ Polygon, yf,   │ Market data, news, macro  │
│              │ Finnhub, FRED  │                           │
│ Cloud Relay  │ FastAPI+Uvicorn│ Stateless AWS relay       │
│ Containers   │ Docker Compose │ Cloud deployment          │
│ GPU          │ NVIDIA GB10    │ Training, inference, LLM  │
└──────────────┴─────────────────┴───────────────────────────┘
```

---

## 4. Component Interaction Map

```
┌──────────────────────────────────────────────────────────────────────┐
│                        COMPONENT DEPENDENCIES                        │
│                                                                      │
│  polygon_fetcher.py ──┐                                              │
│  fetcher.py ──────────┼──► init_db.py (get_connection)               │
│  daily_pull.py ───────┘        │                                     │
│       │                        ▼                                     │
│       └──────────────► features.py ──► trainer.py                    │
│                            │               │                         │
│                            │               ▼                         │
│  analyzer.py ──────────────┤        SPYPredictor                     │
│  (LLM sentiment)           │          │    │                         │
│       │                    │          │    └──► dashboard_bridge.py   │
│       ▼                    │          │              │                │
│  reporter.py               │          │              ▼                │
│  (LLM reports)             │          │        spy_state.json        │
│                            │          │        es_state.json         │
│                            ▼          ▼              │                │
│                      daily_run.py ◄───┘              │                │
│                      (13-step pipeline)              │                │
│                            │                         │                │
│                            ▼                         ▼                │
│                      alerts.py              app.py (dashboard)       │
│                      (Telegram/email)        │  │  │  │              │
│                                              │  │  │  │              │
│  indicators.py ──┐                    SPY ───┘  │  │  └─── Admin     │
│  position.py ────┼──► engine.py       ES ───────┘  │                 │
│  ai_models.py ───┤    (ES strategy)   What-If ─────┘                 │
│  rl_trail.py ────┘    (+ RL trail)                                   │
│                            │                                         │
│                            ▼                                         │
│                      runner.py                                       │
│                      (live/paper/backtest)                            │
│                                                                      │
│  confidence_server.py ──► ai_models.py + engine.py                   │
│  (FastAPI :8100)          /confidence, /exit, /spread                │
│                                                                      │
│  labeling.py ──────────► trainer.py (ES entry/exit training)         │
│                                                                      │
│  engine.py (whatif) ──► presets.py                                    │
│       │                                                              │
│       └──► narrator.py (LLM explanations)                            │
│                                                                      │
│  publisher.py ──────► relay_server.py (AWS)                          │
│  (cloud sync)          (FastAPI, in-memory)                          │
│                                                                      │
│  launcher.py ──► manages all processes + scheduler                   │
│  scripts/*.sh ──► start/stop/status/restart                          │
└──────────────────────────────────────────────────────────────────────┘
```

### Module Dependency Summary

| Module | Depends On | Depended By |
|--------|-----------|-------------|
| `init_db.py` | yaml | All data modules, pipeline, dashboard |
| `polygon_fetcher.py` | requests, pandas | daily_pull, daily_run |
| `fetcher.py` | requests, feedparser, pandas | daily_pull, daily_run, dashboard |
| `daily_pull.py` | polygon_fetcher, fetcher, init_db | daily_run, dashboard (admin) |
| `features.py` | numpy, pandas, init_db | trainer, daily_run, whatif engine, dashboard |
| `trainer.py` | xgboost, numpy, pandas | daily_run, whatif engine, dashboard |
| `analyzer.py` | requests (Ollama API) | daily_run, dashboard (admin) |
| `reporter.py` | requests (Ollama API) | daily_run, dashboard (admin) |
| `indicators.py` | numpy, pandas | engine (ES), CuMLRegimeClassifier |
| `position.py` | — | engine (ES) |
| `engine.py` (ES) | indicators, position, ai_models, rl_trail | runner |
| `ai_models.py` | numpy, torch | engine (ES), confidence_server |
| `rl_trail.py` | numpy, json | engine (ES) |
| `labeling.py` | numpy, pandas | trainer |
| `runner.py` | engine (ES), yaml | launcher |
| `confidence_server.py` | fastapi, ai_models, engine | start.sh (port 8100) |
| `streamer.py` | websockets, aiohttp | launcher |
| `dashboard_bridge.py` | json | daily_run, streamer |
| `app.py` | streamlit, plotly, all modules | — (top-level UI) |
| `engine.py` (whatif) | features, trainer, presets, narrator | dashboard |
| `daily_run.py` | all data, llm, model, bridge | launcher, scheduler |
| `alerts.py` | requests, smtplib | daily_run |
| `publisher.py` | requests | launcher |
| `relay_server.py` | fastapi, uvicorn | cloud deployment |
| `launcher.py` | yaml, subprocess | scripts, user CLI |

---

## 4b. AI Confidence API Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  AI Confidence API (port 8100)                    │
│                                                                  │
│  MT5/FxDreema ──POST──► /confidence ──► ESEntryGate.predict()   │
│                         │                    │                   │
│                         │  entry_conf ◄──────┘                   │
│                         │  vol_regime                            │
│                         │  advice: allow|block                   │
│                         │                                        │
│  MT5/FxDreema ──POST──► /exit ──► ESExitController.predict()    │
│                         │              │                         │
│                         │  exit_conf ◄─┘                         │
│                         │  tp2_trail_atr                         │
│                         │  runner_trail_atr                      │
│                         │                                        │
│  Broker/Manual ─POST──► /spread ──► ESStrategyEngine             │
│                         │           .update_spread()             │
│                         │                                        │
│  All endpoints ────────► Audit Log (JSONL)                       │
│                          ./logs/trade_audit.jsonl                │
│                                                                  │
│  RL Trail Agent ◄──────── engine.py (per-bar Q-learning update) │
│  CuML Regime ◄─────────── indicators.py (ATR percentile input)  │
│  Triple-Barrier Labels ◄── labeling.py (training pipeline)      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Data Flow Diagrams

### 5.1 Daily Pipeline Flow (4:30 PM ET, Mon-Fri)

```
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  Polygon.io │     │  Finnhub    │     │  FRED       │     │  RSS Feeds  │
  │  REST API   │     │  News API   │     │  Macro API  │     │  Yahoo/CNBC │
  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
         │                   │                   │                   │
         │  Step 2           │  Step 3           │  Step 5           │  Step 3
         ▼                   ▼                   ▼                   ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                         SQLite Database (spy.db)                         │
  │                                                                          │
  │  prices ◄── Step 2      news ◄── Step 3         macro ◄── Step 5        │
  │  options_chain ◄── Step 6                options_analytics ◄── Step 7    │
  │  technicals ◄── Step 8                   intraday_features ◄── Step 9   │
  │  daily_sentiment ◄── Step 4              performance ◄── Step 1         │
  └──────────────────────────────┬───────────────────────────────────────────┘
                                 │
                          Step 10│  Feature Engineering
                                 │  (37+ features from 5 categories)
                                 ▼
                    ┌────────────────────────┐
                    │  XGBoost Training      │
                    │  GPU: gpu_hist         │
                    │  252-day rolling       │
                    │  3-class: UP/DOWN/NEUT │
                    │  Walk-forward split    │
                    └───────────┬────────────┘
                                │
                         Step 11│  Inference
                                ▼
                    ┌────────────────────────┐
                    │  Prediction            │
                    │  5-level scale         │
                    │  0-100% confidence     │
                    │  → predictions table   │
                    └───────────┬────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
             Step 12▼    Step 13▼           ▼
        ┌──────────────┐ ┌──────────┐ ┌──────────────┐
        │ LLM Report   │ │ Telegram │ │ Email Alert  │
        │ DeepSeek R1  │ │ Bot API  │ │ SMTP         │
        │ → report_text│ └──────────┘ └──────────────┘
        └──────────────┘

  Step 0:   LLM health check (auto-start Ollama if needed)
  Step 0.5: Gap detection + backfill missing dates
  Step 1:   Evaluate yesterday's prediction vs actual
  Step 4:   LLM sentiment analysis (~60-90 min for 50 articles)
            Falls back to neutral (0.0) if LLM unavailable
```

### 5.2 Realtime Data Flow (Market Hours)

```
  ┌──────────────────────────────────────────────────────────────┐
  │                  Polygon.io WebSocket                        │
  │                                                              │
  │  Stocks Channel          Options Channel                     │
  │  T.SPY (trades)          T.O:SPX* (all SPX options)         │
  └──────┬───────────────────────────┬───────────────────────────┘
         │                           │
         ▼                           ▼
  ┌──────────────┐          ┌──────────────────┐
  │ 5-sec OHLCV  │          │ Options Flow     │
  │ Aggregation  │          │ Detection        │
  │              │          │                  │
  │ → intraday   │          │ Sweeps: >$50K    │
  │   _bars      │          │   same option    │
  │              │          │   multi-exchange  │
  │              │          │   <2 sec window   │
  │              │          │                  │
  │              │          │ Blocks: >$100K   │
  │              │          │   single fill    │
  └──────┬───────┘          └────────┬─────────┘
         │                           │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Dashboard Bridge     │
         │                       │
         │  spy_state.json       │  Atomic write (tmp → rename)
         │  es_state.json        │  Prevents partial reads
         └───────────┬───────────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
   ┌──────────┐ ┌────────┐ ┌──────────────┐
   │ SPY Page │ │ES Page │ │ Cloud Pub.   │
   │ 15s poll │ │ 5s poll│ │ HTTPS POST   │
   └──────────┘ └────────┘ │ 30s heartbeat│
                           └──────┬───────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │ AWS Relay Server │
                        │ FastAPI :8000    │
                        │ In-memory state  │
                        │ SSE stream       │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Cloud Dashboard  │
                        │ Streamlit :8501  │
                        └──────────────────┘
```

### 5.3 ES Strategy Signal Flow

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    1-Minute Bar Input                        │
  └──────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    Indicator Computation                      │
  │                                                              │
  │  ATR(14) → Keltner Channel (EMA20 ± 1.5×ATR)               │
  │  EMA(9), VWAP, RSI(14), ROC(3)                              │
  │                                                              │
  │  Regime Detector: ATR percentile over 10,080 bars            │
  │    Low (<33rd) │ Med (33-66th) │ High (>66th)               │
  │    3-bar hysteresis to prevent rapid switching                │
  └──────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    Entry Signal Detection                     │
  │                                                              │
  │  Phase 1 (Pure Edge):                                        │
  │    Price touches K ± C (Keltner band ± credit width)         │
  │                                                              │
  │  Phase 2 (Confluence Reload):                                │
  │    K ± C touch PLUS 2 of 3 filters:                          │
  │      • ROC confirmation                                      │
  │      • ATR expansion                                         │
  │      • VWAP alignment                                        │
  └──────────────────────────┬─────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  AI Gate?       │  (optional, --ai flag)
                    │  XGBoost: 17    │
                    │  features,      │
                    │  p_enter > thr  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  Lot 0   │  │  Lot 1   │  │  Lot 2   │
        │  TP1     │  │  TP2     │  │  Runner  │
        │  Tight   │  │  Medium  │  │  Wide    │
        │  trail   │  │  trail   │  │  trail   │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    Risk Management                           │
  │                                                              │
  │  Emergency Stop: 20% × C (always active)                    │
  │  Jump Exit: 5 pts adverse during 1-min hold                 │
  │  Session Flatten: before 15:55 CT                           │
  │  Event Flatten: before FOMC/CPI/NFP                         │
  │  Circuit Breaker: daily P&L ≤ -$2,000                       │
  │    → flatten all + disable until 17:00 CT reset             │
  └──────────────────────────────────────────────────────────────┘
```

---

## 6. Sequence Diagrams

### 6.1 Daily Pipeline Sequence

```
  User/Scheduler          Launcher           Pipeline          Data Layer        LLM            Model
       │                     │                  │                  │              │                │
       │  trigger (4:30 PM)  │                  │                  │              │                │
       │────────────────────►│                  │                  │              │                │
       │                     │  run()           │                  │              │                │
       │                     │─────────────────►│                  │              │                │
       │                     │                  │                  │              │                │
       │                     │                  │  Step 0: check   │              │                │
       │                     │                  │─────────────────────────────────►  health()      │
       │                     │                  │                  │              │◄───────────────│
       │                     │                  │                  │              │                │
       │                     │                  │  Step 0.5: gap detect + backfill                │
       │                     │                  │─────────────────►│              │                │
       │                     │                  │                  │              │                │
       │                     │                  │  Step 1: evaluate yesterday     │                │
       │                     │                  │─────────────────►│              │                │
       │                     │                  │                  │              │                │
       │                     │                  │  Steps 2-3: fetch prices + news │                │
       │                     │                  │─────────────────►│              │                │
       │                     │                  │                  │              │                │
       │                     │                  │  Step 4: sentiment analysis     │                │
       │                     │                  │─────────────────────────────────►  analyze()     │
       │                     │                  │                  │              │  (~60-90 min)  │
       │                     │                  │◄─────────────────────────────────  scores        │
       │                     │                  │                  │              │                │
       │                     │                  │  Steps 5-9: macro, options, technicals          │
       │                     │                  │─────────────────►│              │                │
       │                     │                  │                  │              │                │
       │                     │                  │  Step 10: build features + train│                │
       │                     │                  │─────────────────►│              │                │
       │                     │                  │─────────────────────────────────────────────────►│
       │                     │                  │                  │              │    train(GPU)  │
       │                     │                  │◄─────────────────────────────────────────────────│
       │                     │                  │                  │              │                │
       │                     │                  │  Step 11: predict│              │                │
       │                     │                  │─────────────────────────────────────────────────►│
       │                     │                  │◄─────────────────────────────────────────────────│
       │                     │                  │                  │              │                │
       │                     │                  │  Step 12: report │              │                │
       │                     │                  │─────────────────────────────────►  generate()   │
       │                     │                  │◄─────────────────────────────────               │
       │                     │                  │                  │              │                │
       │                     │                  │  Step 13: alerts │              │                │
       │                     │                  │──► Telegram + Email             │                │
       │                     │                  │                  │              │                │
       │                     │◄─────────────────│  results         │              │                │
       │◄────────────────────│  complete         │                  │              │                │
       │                     │                  │                  │              │                │
```

### 6.2 Dashboard Request Sequence

```
  Browser              Streamlit (app.py)       State Files          Database
     │                       │                      │                   │
     │  HTTP GET :8501       │                      │                   │
     │──────────────────────►│                      │                   │
     │                       │                      │                   │
     │                       │  read spy_state.json │                   │
     │                       │─────────────────────►│                   │
     │                       │◄─────────────────────│                   │
     │                       │                      │                   │
     │                       │  read es_state.json  │                   │
     │                       │─────────────────────►│                   │
     │                       │◄─────────────────────│                   │
     │                       │                      │                   │
     │                       │  query predictions   │                   │
     │                       │──────────────────────────────────────────►│
     │                       │◄──────────────────────────────────────────│
     │                       │                      │                   │
     │  HTML + Plotly charts │                      │                   │
     │◄──────────────────────│                      │                   │
     │                       │                      │                   │
     │  (auto-refresh 5-15s) │                      │                   │
     │──────────────────────►│                      │                   │
     │         ...           │                      │                   │
```

### 6.3 Cloud Sync Sequence

```
  DGX Publisher          AWS Relay (FastAPI)       Cloud Dashboard
       │                       │                         │
       │  POST /push/prediction│                         │
       │──────────────────────►│                         │
       │  X-API-Key: <key>     │  store in memory        │
       │◄──────────────────────│                         │
       │                       │                         │
       │  POST /push/es_state  │                         │
       │──────────────────────►│                         │
       │◄──────────────────────│                         │
       │                       │                         │
       │  POST /push/heartbeat │                         │
       │──────────────────────►│  (every 30s)            │
       │◄──────────────────────│                         │
       │                       │                         │
       │                       │  GET /state             │
       │                       │◄────────────────────────│
       │                       │────────────────────────►│
       │                       │                         │
       │                       │  GET /stream (SSE)      │
       │                       │◄────────────────────────│
       │                       │──── event: prediction ─►│
       │                       │──── event: es_state ───►│
       │                       │──── event: heartbeat ──►│
       │                       │                         │
       │                       │  (no heartbeat 90s)     │
       │                       │  mark source OFFLINE    │
       │                       │                         │
```

### 6.4 Admin Console Action Sequence (Example: Retrain)

```
  Browser              Admin Console (app.py)     Features         Trainer          Database
     │                       │                      │                │                │
     │  click "Retrain"      │                      │                │                │
     │──────────────────────►│                      │                │                │
     │                       │                      │                │                │
     │  spinner: "Training"  │                      │                │                │
     │◄──────────────────────│                      │                │                │
     │                       │  build_feature_vector │                │                │
     │                       │─────────────────────►│                │                │
     │                       │                      │  SELECT *      │                │
     │                       │                      │────────────────────────────────►│
     │                       │                      │◄────────────────────────────────│
     │                       │◄─────────────────────│  DataFrame     │                │
     │                       │                      │                │                │
     │                       │  train(X, y)         │                │                │
     │                       │──────────────────────────────────────►│                │
     │                       │                      │                │  XGBoost GPU   │
     │                       │                      │                │  gpu_hist      │
     │                       │◄──────────────────────────────────────│  model saved   │
     │                       │                      │                │                │
     │  success + accuracy   │                      │                │                │
     │◄──────────────────────│                      │                │                │
     │                       │                      │                │                │
```

---

## 7. Database Schema (ERD)

```
  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │     prices      │     │   technicals    │     │      macro      │
  ├─────────────────┤     ├─────────────────┤     ├─────────────────┤
  │ PK date         │◄───►│ PK date         │◄───►│ PK date         │
  │    open         │     │    sma_20       │     │    vix          │
  │    high         │     │    sma_50       │     │    vix_change   │
  │    low          │     │    rsi_14       │     │    us10y_yield  │
  │    close        │     │    macd         │     │    dxy          │
  │    volume       │     │    macd_signal  │     │    fed_funds    │
  │    adj_close    │     │    macd_hist    │     │    gold         │
  └────────┬────────┘     │    bb_upper     │     │    crude        │
           │              │    bb_lower     │     └─────────────────┘
           │              │    atr_14       │
           │              └─────────────────┘
           │
           │         ┌─────────────────┐     ┌─────────────────────┐
           │         │      news       │     │  daily_sentiment    │
           │         ├─────────────────┤     ├─────────────────────┤
           │         │ PK id (auto)    │     │ PK date             │
           │         │    date ────────┼────►│    sentiment_score  │
           │         │    source       │     │    news_count       │
           │         │    headline     │     │    positive_ratio   │
           │         │    summary      │     │    negative_ratio   │
           │         │    url          │     │    avg_confidence   │
           │         │    fetched_at   │     └─────────────────────┘
           │         └─────────────────┘
           │
           │         ┌─────────────────────┐  ┌─────────────────────┐
           │         │   options_chain     │  │ options_analytics   │
           │         ├─────────────────────┤  ├─────────────────────┤
           │         │ PK (date,           │  │ PK date             │
           │         │    contract_symbol) │  │    put_call_ratio   │
           │         │    strike           │  │    max_pain         │
           │         │    type (C/P)       │  │    max_pain_dist    │
           │         │    expiration       │  │    iv_skew          │
           │         │    bid/ask          │  │    gex              │
           │         │    volume/oi        │  └─────────────────────┘
           │         │    iv, delta        │
           │         │    gamma,theta,vega │
           │         └─────────────────────┘
           │
           │         ┌─────────────────────┐  ┌─────────────────────┐
           │         │  intraday_bars     │  │ intraday_features   │
           │         ├─────────────────────┤  ├─────────────────────┤
           │         │ PK (timestamp,      │  │ PK date             │
           │         │     ticker)         │  │    vwap_spread      │
           │         │    open,high        │  │    intraday_momentum│
           │         │    low,close        │  │    intraday_range   │
           │         │    volume           │  │    volume_ratio     │
           │         └─────────────────────┘  └─────────────────────┘
           │
           ▼
  ┌─────────────────────┐     ┌─────────────────┐
  │    predictions      │     │   performance   │
  ├─────────────────────┤     ├─────────────────┤
  │ PK date             │────►│ PK date         │
  │    direction        │     │    predicted     │
  │    confidence       │     │    actual        │
  │    probabilities    │     │    correct       │
  │    factors          │     │    cumulative_acc│
  │    report_text      │     └─────────────────┘
  │    predicted_at     │
  └─────────────────────┘

  All tables joined on `date` (except intraday_bars on timestamp+ticker, news on id).
  Database: SQLite 3.45, WAL journal mode, 5-second busy timeout.
```

---

## 8. Deployment Architecture

### 8.1 Local Deployment (Primary)

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Windows Dev Machine (F:\websites\stockanalysis)            │
  │                                                             │
  │  Code editing in Kiro IDE                                   │
  │                                                             │
  └──────────────────────┬──────────────────────────────────────┘
                         │ Mutagen Sync
                         │ (bidirectional, real-time)
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  NVIDIA DGX Spark (192.168.1.211)                           │
  │                                                             │
  │  ~/stockanalysis/                                           │
  │  ├── .venv/          Python 3.12 virtual environment        │
  │  ├── src/            Application code                       │
  │  ├── data/spy.db     SQLite database                        │
  │  ├── models/         Trained XGBoost models                 │
  │  ├── config.yaml     All configuration                      │
  │  ├── scripts/        Operations scripts                     │
  │  ├── logs/           Runtime logs                           │
  │  └── .pids/          PID tracking files                     │
  │                                                             │
  │  Processes:                                                 │
  │  ├── Ollama (:11434)        DeepSeek R1 70B (~42GB VRAM)   │
  │  ├── Streamlit (:8501)      Unified dashboard               │
  │  ├── ES Runner              Paper/live/backtest             │
  │  └── Scheduler              Daily pipeline at 4:30 PM ET   │
  │                                                             │
  │  GPU: NVIDIA GB10                                           │
  │  RAM: 128GB+                                                │
  └─────────────────────────────────────────────────────────────┘
```

### 8.2 Cloud Deployment (Optional Mirror)

```
  ┌─────────────────────────────────────────────────────────────┐
  │  AWS EC2 t3.micro (~$8-10/mo)                               │
  │                                                             │
  │  Docker Compose:                                            │
  │  ┌─────────────────────┐  ┌─────────────────────┐          │
  │  │  relay              │  │  dashboards          │          │
  │  │  FastAPI + Uvicorn  │  │  Streamlit           │          │
  │  │  Port 8000          │  │  Port 8501           │          │
  │  │  In-memory state    │  │  Reads from relay    │          │
  │  │  API key auth       │  │  RELAY_URL env var   │          │
  │  │  SSE streaming      │  │  No GPU, no DB       │          │
  │  └─────────────────────┘  └─────────────────────┘          │
  │                                                             │
  │  No model inference, no database — display only             │
  └─────────────────────────────────────────────────────────────┘
```

---

## 9. Failure Modes and Resilience

```
  ┌──────────────────────────────────────────────────────────────────────┐
  │                     FAILURE HANDLING MATRIX                          │
  ├──────────────────────┬──────────────────────────────────────────────┤
  │ Failure              │ System Response                              │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ Polygon WS disconnect│ Auto-reconnect, exponential backoff 1→30s   │
  │ Polygon REST down    │ Fallback to yfinance for price data         │
  │ Ollama not running   │ Auto-start; if fails → neutral sentiment    │
  │ Model not downloaded │ Auto-download with progress; if fails → 0.0 │
  │ LLM inference timeout│ Skip sentiment, use score=0.0               │
  │ SQLite locked        │ 5s busy timeout + WAL for concurrent reads  │
  │ Dashboard crash      │ Launcher auto-restarts within 30s           │
  │ Cloud relay down     │ Publisher retries 3x then skips; local OK   │
  │ Circuit breaker      │ Flatten all ES positions, pause til 17:00   │
  │ Pipeline step fails  │ Log error, continue next step; never aborts │
  │ No heartbeat 90s     │ Relay marks source OFFLINE                  │
  └──────────────────────┴──────────────────────────────────────────────┘
```

---

## 10. Network Ports

| Port | Service | Protocol | Access |
|------|---------|----------|--------|
| 8501 | Unified Dashboard (Streamlit) | HTTP | LAN / Cloud |
| 8100 | AI Confidence API (FastAPI) | HTTP | LAN (MT5/FxDreema) |
| 9190 | Prometheus Metrics Exporter | HTTP | LAN (Prometheus) |
| 3001 | Grafana Dashboard | HTTP | LAN |
| 9092 | Prometheus (Docker) | HTTP | Docker internal |
| 11434 | Ollama LLM API | HTTP | localhost only |
| 8000 | Cloud Relay (FastAPI) | HTTPS | Public (AWS) |

---

## 11. File Structure

```
stockanalysis/
├── src/
│   ├── data/                  # Data ingestion and feature engineering
│   │   ├── init_db.py         # Schema manager, get_connection()
│   │   ├── polygon_fetcher.py # Polygon.io REST client
│   │   ├── fetcher.py         # Fallback: yfinance, Finnhub, RSS, FRED
│   │   ├── daily_pull.py      # Gap detection + backfill
│   │   ├── backfill.py        # Initial bulk load (252 days)
│   │   └── features.py        # 37+ feature engineering
│   ├── llm/                   # LLM integration
│   │   ├── analyzer.py        # Sentiment analysis via Ollama
│   │   └── reporter.py        # Daily report generation
│   ├── model/                 # Machine learning
│   │   └── trainer.py         # XGBoost SPY predictor (GPU)
│   ├── realtime/              # Real-time data
│   │   ├── streamer.py        # Polygon WebSocket client
│   │   └── dashboard_bridge.py# JSON state file writer
│   ├── es_strategy/           # ES futures strategy
│   │   ├── indicators.py      # ATR, Keltner, EMA, VWAP, RSI, ROC, CuMLRegimeClassifier
│   │   ├── position.py        # 3-lot position manager
│   │   ├── engine.py          # Signal generation logic + RL trail integration
│   │   ├── ai_models.py       # XGBoost gate + CNN exit controller
│   │   ├── rl_trail.py        # Q-learning trailing stop agent
│   │   ├── labeling.py        # Triple-barrier entry + reversal exit labels
│   │   └── runner.py          # Live/paper/backtest runner
│   ├── dashboard/             # Web UI
│   │   ├── app.py             # Unified dashboard (4 pages + admin)
│   │   ├── realtime_app.py    # Legacy standalone SPY
│   │   ├── es_dashboard.py    # Legacy standalone ES
│   │   └── whatif_app.py      # Legacy standalone What-If
│   ├── whatif/                 # Scenario analysis
│   │   ├── engine.py          # What-If compute engine
│   │   ├── presets.py         # 5 stress test scenarios
│   │   └── narrator.py        # LLM what-if explanations
│   ├── pipeline/              # Automation
│   │   ├── daily_run.py       # 13-step pipeline orchestrator
│   │   └── alerts.py          # Telegram + email notifications
│   ├── sync/                  # Cloud sync
│   │   ├── publisher.py       # DGX → AWS state publisher
│   │   └── relay_server.py    # FastAPI cloud relay
│   ├── api/                   # Real-time AI API
│   │   ├── confidence_server.py # /confidence, /exit, /spread endpoints (port 8100)
│   │   └── metrics_exporter.py  # Prometheus metrics exporter (port 9190)
│   └── launcher.py            # Process manager + scheduler
├── grafana/                   # Grafana monitoring
│   ├── grafana.ini            # Grafana config (Google OAuth + RBAC)
│   ├── provisioning/          # Auto-provisioning
│   │   ├── datasources/       # Prometheus datasource
│   │   └── dashboards/        # Dashboard provider config
│   └── dashboards/            # Dashboard JSON files
│       ├── spy-predictor.json
│       ├── es-strategy.json
│       ├── system-health.json
│       ├── confidence-api.json
│       └── pipeline-status.json
├── prometheus/                # Prometheus config
│   └── prometheus.yml         # Scrape targets
├── scripts/                   # Operations
│   ├── start.sh               # Start all components
│   ├── stop.sh                # Graceful shutdown
│   ├── status.sh              # Health check
│   └── restart.sh             # Stop + start
├── cloud/                     # Cloud deployment
│   ├── Dockerfile*            # Container images
│   ├── docker-compose.yml     # Service orchestration
│   ├── deploy_aws.sh          # ECR + EC2 deployment
│   └── start*.sh              # Container entrypoints
├── data/                      # Runtime data
│   ├── spy.db                 # SQLite database
│   ├── spy_state.json         # SPY dashboard state
│   └── es_state.json          # ES dashboard state
├── models/                    # Trained models
│   ├── xgb_spy_YYYYMMDD.json # XGBoost model files
│   └── rl_trail_qtable.json   # RL trailing agent Q-table
├── logs/                      # Runtime logs
│   └── trade_audit.jsonl      # Trade audit trail (JSONL)
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # This file
│   ├── SOLUTION_DOCUMENT.md   # Full component inventory
│   ├── ADMIN_GUIDE.md         # Operations and configuration
│   └── USER_GUIDE.md          # End-user dashboard guide
├── config.yaml                # All configuration
├── requirements.txt           # Python dependencies
└── PROMPT.md                  # Build specification
```
