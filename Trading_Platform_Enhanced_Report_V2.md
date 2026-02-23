# SPY/SPX Trading Platform: Comprehensive Enhancement Report

**Platform:** AI Agentic International Trading Platform — `trading.aiagenticinternational.org`
**Date:** February 22, 2026
**Author:** Manus AI
**Classification:** Technical Enhancement Specification

---

## Executive Summary

This report delivers a thorough, research-backed enhancement plan for the AI Agentic International trading platform. It combines findings from a live site review across all six menu sections with deep research into state-of-the-art financial machine learning, trading dashboard design, and financial platform security and observability.

The live review uncovered three categories of findings. First, there are **critical system failures** that render the platform non-functional: the data pipeline is broken, most charts are empty, and API keys are exposed directly in the frontend configuration — a severe security vulnerability. Second, there are **significant UX and layout deficiencies** that would impede professional use even if the data were flowing correctly. Third, there are **substantial functional gaps** relative to the current state of the art in institutional-grade trading analytics systems.

The enhancement roadmap is organized into three sequential phases. **Phase 1** (Weeks 1–2) is a critical triage to restore baseline functionality and eliminate the security vulnerability. **Phase 2** (Weeks 3–10) introduces research-backed predictive model upgrades, advanced feature engineering, and new analytical capabilities. **Phase 3** (Weeks 11–20) proposes a full platform modernization, migrating from Streamlit to a production-grade FastAPI and React architecture with real-time WebSocket data streaming and an LLM-powered AI assistant.

---

## Part 1: Live Site Review — Findings by Page

### 1.1. SPY Predictor Page

The SPY Predictor is the flagship page of the platform, yet it is in a largely broken state. The prediction card at the top of the page displays a direction signal and confidence score, but these values appear static and are not updating in real time. The "Last Updated" timestamp is stale, confirming that the data pipeline is not running. The feature importance chart is empty, and the news sentiment section shows no articles. The historical performance chart displays only a flat line, indicating that no historical prediction data has been stored in the database.

The layout itself has several UX issues. The prediction confidence gauge is visually prominent but provides no contextual interpretation — a trader seeing "62% confidence" has no way to know whether that is a strong signal or a weak one without historical calibration data. The page lacks any indicator of the current market regime (trending vs. mean-reverting, high vs. low volatility), which is essential context for interpreting any directional prediction.

### 1.2. ES Strategy Page

The ES Futures Strategy page is almost entirely non-functional. The primary chart area is blank, and the position state panel shows no active or historical trades. The strategy performance metrics (Sharpe ratio, win rate, average P&L) all display zero or null values. The "Run Strategy" button does not appear to trigger any visible response.

This page suffers from the same root cause as the SPY Predictor: the absence of live data flowing from the pipeline. However, even with data, the page design is sparse. There is no visualization of the strategy's entry and exit logic overlaid on a price chart, which is the standard presentation in professional backtesting platforms.

### 1.3. What-If Analysis Page

The What-If Analysis page has two tabs: ES Strategy and SPY Predictor. The ES Strategy tab contains a parameter sweep interface (the "K/C Sweep") that, when executed, ran for several minutes and ultimately returned no results. The SPY Predictor tab offers scenario testing with options for "Scenario Type" (including a Stress Test option), but the results panel remains empty after running a scenario.

The concept behind this page is sound — scenario analysis is a valuable tool for understanding model sensitivity. However, the execution is incomplete. The parameter sweep has no progress indicator, no partial results display, and no error message when it fails. The stress test scenarios are not clearly defined; a user has no way to know what specific market conditions the "Stress Test" scenario simulates.

### 1.4. Monitoring Page

The Monitoring page is intended to display system health and model performance metrics. In its current state, it shows a series of empty charts and metric cards with placeholder values. The Grafana comparison sub-page, accessible from the sidebar, appears to attempt to embed an external Grafana dashboard but fails to load, displaying only a blank iframe.

This page represents a missed opportunity. A well-designed monitoring page would be one of the most valuable tools for an administrator, providing at-a-glance visibility into data pipeline health, model drift, prediction accuracy over time, and system resource utilization.

### 1.5. Admin Console

The Admin Console has five tabs: System Status, Actions, Users, Database, and Configuration.

The **System Status** tab shows a table of service statuses. Several services are marked as "Error" or "Unknown," confirming the pipeline failures observed elsewhere. The **Actions** tab contains buttons for manual pipeline triggers (e.g., "Fetch Data," "Retrain Model") but these buttons produce no visible feedback when clicked. The **Users** tab shows a user management interface, but the user table is empty. The **Database** tab shows database statistics, but the row counts are all zero, confirming that no data has been persisted. The **Configuration** tab is the source of the critical security vulnerability: it displays the full Polygon.io API key, the database credentials, and other sensitive secrets in plain text within the UI, accessible to any logged-in user.

---

## Part 2: Critical Fixes — Phase 1 (Weeks 1–2)

The following issues must be resolved before any enhancement work begins. The platform is not safe or functional in its current state.

### 2.1. Security Vulnerability: API Key Exposure (P0 — Immediate)

The most urgent issue is the exposure of API keys and database credentials in the Admin Console Configuration tab. This violates the most fundamental principle of secrets management: secrets must never be rendered in a user interface or stored in client-accessible configuration files [1].

**Required Action:** All secrets must be immediately removed from `config.toml` and any other file accessible to the Streamlit frontend. The correct architecture is to store secrets in environment variables on the server, or in a dedicated secrets management service such as HashiCorp Vault or AWS Secrets Manager. The Admin Console Configuration tab should display only masked values (e.g., `POLY_*****_XXXX`) and provide a mechanism to rotate keys, not to view them. Every access to a secret should be logged with a timestamp and user identity.

### 2.2. Data Pipeline Restoration (P1)

The data pipeline is the foundation of the entire platform. Without it, no page functions correctly.

**Required Action:** A systematic debugging session is needed to identify the point of failure in the pipeline. The `data_fetcher.py`, `polygon_fetcher.py`, and `streamer.py` services must be examined for connection errors, authentication failures, and data parsing bugs. A health check endpoint (e.g., `/api/health`) should be implemented that returns the status of each data source — Polygon.io REST, Polygon.io WebSocket, yfinance, Finnhub, and the database connection — so that failures are immediately visible without requiring a developer to inspect logs.

### 2.3. UI State Truthfulness (P1)

The UI must accurately reflect the true state of the system. Displaying a "Connected" status badge when the data pipeline is broken is misleading and dangerous in a trading context.

**Required Action:** All status indicators must be derived from live backend health checks, not from static configuration values. Charts must display a clear "No Data Available" empty state with a timestamp of the last successful data fetch, rather than rendering blank axes or placeholder images. Admin action buttons must provide immediate visual feedback (e.g., a spinner, a success/error toast notification) when clicked.

---

## Part 3: Predictive Model Enhancements — Phase 2 (Weeks 3–10)

With the platform stabilized, the focus shifts to dramatically improving its predictive power. The current single XGBoost model is a reasonable baseline, but the academic literature and industry practice have advanced significantly. The following enhancements are grounded in peer-reviewed research published in 2024 and 2025.

### 3.1. Model Architecture: Stacking Ensemble with Temporal Fusion Transformer

The most impactful single improvement is to replace the single-model architecture with a multi-layer stacking ensemble. Research published in 2025 demonstrates that stacking heterogeneous models — combining the strengths of different algorithmic families — consistently outperforms any individual model on financial time series prediction tasks [2][3].

**Proposed Architecture:**

The ensemble operates in two levels. At **Level 0**, three specialist models are trained independently:

| Model | Role | Key Strength |
|---|---|---|
| **Temporal Fusion Transformer (TFT)** | Primary temporal forecaster | Captures long-range dependencies, handles mixed-frequency inputs, provides interpretable attention weights [4] |
| **Hidden Markov Model (HMM)** | Market regime classifier | Identifies the current market state (e.g., Bull Trending, Bear Volatile, Sideways Compressing) and outputs a regime probability vector [5] |
| **FinBERT-LSTM** | Sentiment forecaster | Processes news headlines and social media text through a financial domain-specific BERT model, producing a sentiment score that feeds an LSTM for temporal smoothing [6] |

At **Level 1**, a meta-learner XGBoost model is trained on the out-of-fold predictions of the three Level 0 models. This meta-learner learns the optimal combination weights for each regime and market condition, producing a final calibrated probability of an upward move.

The TFT-GNN hybrid, as demonstrated by Lynch et al. [4], achieved the highest accuracy of all models tested for SPY in 2024, outperforming the standalone TFT in 11 of 12 evaluated periods. The key insight from that research is that incorporating relational signals between correlated assets (e.g., QQQ, VIX, DXY, gold) through a Graph Attention Network provides meaningful additional predictive power beyond technical indicators alone.

### 3.2. Feature Engineering: Options Flow, Microstructure, and Macro Signals

The predictive power of any model is bounded by the quality of its input features. The current feature set is limited to price-derived technical indicators and a small set of macro variables. The following new feature categories represent significant, research-validated sources of alpha.

**Options-Based Features:**

Gamma Exposure (GEX) is one of the most powerful short-term predictive signals available for SPY and SPX. GEX measures the aggregate change in dealer delta exposure for a 1% move in the underlying. When GEX is strongly positive, market makers are long gamma and will sell into rallies and buy dips, suppressing volatility and creating mean-reversion conditions. When GEX is strongly negative, market makers are short gamma and must hedge in the same direction as price moves, amplifying volatility and creating trending conditions [7]. This single feature can fundamentally change the character of a trading day and is directly computable from the Polygon.io options chain data already being ingested.

Vanna and Charm exposures provide second-order signals. Vanna (the rate of change of delta with respect to implied volatility) is particularly important on days when the VIX is moving significantly, as it creates predictable dealer hedging flows. Charm (the rate of change of delta with respect to time) drives predictable end-of-day and end-of-week hedging flows as options approach expiration.

**Market Microstructure Features:**

Order Flow Imbalance (OFI), defined as the difference between buyer-initiated and seller-initiated volume normalized by total volume, is a well-established predictor of short-term price direction. Research published in 2025 demonstrates that OFI, combined with bid-ask spread dynamics, significantly improves intraday volatility prediction [8]. The Polygon.io 5-second bar data already being collected is sufficient to compute OFI at a meaningful resolution.

Volume Profile analysis identifies the Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL) for each trading session. These price levels represent areas of high trading interest and act as strong support and resistance. Encoding the current price's position relative to these levels as a feature provides the model with structural market context that is invisible to pure price-based indicators.

**Dark Pool Activity:**

The Dark Pool Index (DPI), computed as the ratio of off-exchange volume to total volume, is a leading indicator of institutional positioning. Research by Kaabar [9] demonstrates that elevated dark pool buying activity precedes upward moves in the S&P 500 with statistically significant predictive power. The Polygon.io Advanced subscription already provides access to off-exchange trade data, making this feature directly computable without additional data costs.

**Macro and Cross-Asset Features:**

The current feature set includes VIX, DXY, gold, and crude oil. The following additions would significantly enrich the macro context:

| Feature | Source | Predictive Signal |
|---|---|---|
| **2Y–10Y Treasury Yield Spread** | FRED | Recession indicator; inversion historically precedes equity weakness |
| **Credit Spread (HYG–IEF)** | yfinance | Risk appetite proxy; widening spreads precede equity sell-offs |
| **AAII Sentiment Survey** | AAII (weekly) | Contrarian indicator; extreme bullishness is bearish for forward returns |
| **CFTC Commitment of Traders (CoT)** | CFTC (weekly) | Net speculative positioning in ES futures; extreme positioning is a mean-reversion signal |
| **Sector Rotation (XLF, XLK, XLE relative strength)** | yfinance | Identifies which sectors are leading or lagging, providing context for index direction |

### 3.3. Intraday Prediction Updates

The current system produces a single daily prediction. Adding intraday prediction updates at key market intervals (9:45 AM, 11:00 AM, 1:00 PM, 3:00 PM ET) would dramatically increase the platform's utility for active traders. Each intraday update would incorporate the most recent VWAP deviation, OFI, GEX, and options flow data to produce an updated directional probability for the remainder of the session.

### 3.4. Probability Calibration

The current model outputs a raw probability score that is not calibrated. A model that outputs 70% confidence should be correct approximately 70% of the time; without calibration, this relationship does not hold. Implementing Platt Scaling or Isotonic Regression as a post-processing step on the model's output will produce well-calibrated probabilities that are meaningful for position sizing and risk management.

### 3.5. Model Explainability: SHAP Integration

Every prediction should be accompanied by a SHAP (SHapley Additive exPlanations) waterfall chart that shows which features drove the prediction and by how much. Research demonstrates that SHAP integration in trading dashboards significantly improves user trust and decision quality [10]. The SHAP chart should be displayed directly on the SPY Predictor page, allowing a trader to immediately understand why the model is predicting an upward or downward move on any given day.

---

## Part 4: Platform Modernization — Phase 3 (Weeks 11–20)

### 4.1. Architecture: Migration from Streamlit to FastAPI + React

Streamlit is an excellent prototyping tool but has fundamental limitations that make it unsuitable for a production trading platform. Its global re-render model, lack of true WebSocket support, and limited layout control are already causing problems in the current system. The research literature on production financial dashboards consistently recommends a decoupled architecture with a dedicated API backend and a modern JavaScript frontend [11].

**Proposed Target Architecture:**

The new architecture separates concerns cleanly into four layers. The **Data Layer** consists of the existing Polygon.io, yfinance, Finnhub, and FRED data sources, with the addition of an Alpaca Markets fallback for intraday data resilience. The **Backend Layer** is rebuilt in FastAPI, providing RESTful endpoints for historical data and predictions, WebSocket endpoints for real-time streaming, and a secure secrets management integration. The **ML Layer** runs as a separate microservice, exposing a `/predict` endpoint that the backend calls; this allows the model to be updated and redeployed independently of the UI. The **Frontend Layer** is a React application that communicates with the backend via REST for page loads and WebSockets for real-time updates.

This architecture is horizontally scalable, testable, and maintainable in a way that the current monolithic Streamlit application is not.

### 4.2. Real-Time Data Architecture

The current system uses Streamlit's `st.rerun()` polling mechanism to simulate real-time updates. This is inefficient and introduces latency. The new architecture should use a proper real-time data flow:

The Polygon.io WebSocket stream feeds into a Redis Pub/Sub channel. The FastAPI backend subscribes to this channel and pushes updates to connected frontend clients via WebSocket. The frontend React components subscribe to these WebSocket messages and update their state directly, triggering targeted re-renders of only the affected chart or metric card. This approach reduces latency from seconds to milliseconds and eliminates the CPU overhead of full-page re-renders.

### 4.3. LLM-Powered AI Financial Assistant

The most transformative enhancement is the addition of a context-bound AI assistant that allows users to interact with the platform using natural language. Research published in 2025 demonstrates that voice-enabled AI assistants for stock market analysis, combining LSTM predictions with NLP, represent a significant advancement in human-computer interaction for financial platforms [12].

**Capabilities:**

The assistant will be able to answer questions about the current prediction and its drivers (e.g., *"Why is the model predicting a down day?"*), run on-demand scenario analyses (e.g., *"What would the prediction be if VIX were at 30?"*), explain complex metrics in plain language (e.g., *"What does the current GEX reading mean for today's trading?"*), and summarize the day's news sentiment and its impact on the prediction.

A critical design constraint is that the assistant must be **strictly context-bound**. It should only answer questions about the platform's data and analytics. It must not provide general financial advice, discuss securities not tracked by the platform, or engage in off-topic conversation. This is both a safety requirement and a regulatory consideration.

**Implementation:** The assistant will use a Retrieval-Augmented Generation (RAG) architecture. The LLM (GPT-4 or a fine-tuned open-source equivalent) will be given access to a context window containing the current prediction, the SHAP feature importance values, the latest news headlines, and the current market regime state. All responses will be grounded in this context, preventing hallucination.

The interface will support both text and voice input, with voice transcription handled by a Whisper-based speech-to-text service running locally to avoid sending audio to external APIs.

### 4.4. UX Redesign: Institutional-Grade Dashboard

The new React frontend will implement the following UX improvements, informed by best practices for real-time financial data dashboards [13]:

**Information Hierarchy:** The most important signal — the current prediction with its confidence and direction — should dominate the visual space at the top of the page. Supporting evidence (SHAP chart, regime indicator, sentiment score) should be arranged in descending order of importance below it.

**Customizable Layout:** Users should be able to rearrange dashboard panels using drag-and-drop, save multiple layout configurations (e.g., "Morning Setup," "Intraday Monitoring"), and choose which metrics appear on their primary view.

**Consistent Color Semantics:** A strict color system should be applied consistently across all pages: green for bullish signals, red for bearish signals, amber for neutral or uncertain conditions, and blue for informational elements. This consistency is currently absent, making the UI cognitively demanding.

**Data Freshness Indicators:** Every data-driven element on the page should display a timestamp showing when it was last updated. Stale data (older than a configurable threshold) should be visually flagged.

**Contextual Tooltips:** Every metric, chart, and indicator should have a tooltip explaining what it measures, how it is calculated, and how to interpret it. This is essential for onboarding new users and for ensuring that all users are interpreting signals correctly.

---

## Part 5: MLOps, Security, and Observability Enhancements

### 5.1. Model Monitoring and Drift Detection

A production ML model requires continuous monitoring. The current system has no mechanism to detect when the model's predictive performance is degrading due to changes in market conditions (concept drift) or changes in the statistical properties of the input features (data drift).

The recommended approach is to implement **Evidently AI**, the leading open-source framework for ML monitoring in financial systems [14]. Evidently will generate daily reports comparing the current feature distributions against the training distribution (data drift) and comparing the model's recent prediction accuracy against its historical baseline (concept drift). These reports should be displayed on the Monitoring page and should trigger automated alerts when drift exceeds a configurable threshold.

### 5.2. Automated Model Retraining Pipeline

When drift is detected, the model should be automatically retrained on a rolling window of recent data. The retraining pipeline should be implemented as a scheduled job (e.g., weekly on Sundays) using a workflow orchestration tool like Apache Airflow or Prefect. The pipeline should include automated backtesting of the new model against the old model, and the new model should only be promoted to production if it demonstrates a statistically significant improvement in out-of-sample accuracy.

### 5.3. Comprehensive Secrets Management

As described in Phase 1, all secrets must be removed from the frontend and managed through a dedicated secrets management system. In addition to the immediate fix, the following practices should be implemented:

All API keys should be rotated on a 90-day schedule. Each service (data fetcher, ML service, frontend backend) should have its own API key with the minimum permissions required for its function. All access to secrets should be logged with a timestamp, user identity, and the service that requested the secret. The Admin Console should provide a key rotation interface that allows an administrator to generate a new key and update the secret store without any downtime.

### 5.4. Structured Logging and Alerting

The current system uses unstructured log output that is difficult to parse and analyze. All log messages should be converted to structured JSON format with consistent fields: `timestamp`, `level`, `component`, `event`, `context`. This enables log aggregation tools like Elasticsearch or Loki to index and query logs efficiently.

Critical events — data pipeline failures, model prediction errors, authentication failures, and security events — should trigger immediate alerts via Slack or PagerDuty, ensuring that the operations team is notified within minutes of any issue.

---

## Implementation Roadmap Summary

The following table provides a consolidated view of all enhancements, organized by phase, priority, and estimated effort.

| Phase | Enhancement | Priority | Effort | Expected Impact |
|---|---|---|---|---|
| **1** | Remove API keys from frontend; implement secrets management | P0 | 1 day | Eliminates critical security vulnerability |
| **1** | Debug and restore data pipeline | P1 | 3–5 days | Restores all page functionality |
| **1** | Fix UI state truthfulness (status indicators, empty states) | P1 | 2 days | Eliminates misleading information |
| **1** | Admin action button feedback | P2 | 1 day | Improves operator UX |
| **2** | Probability calibration (Platt Scaling) | P1 | 2 days | Immediately improves signal reliability |
| **2** | SHAP explainability integration | P1 | 3 days | Dramatically improves user trust |
| **2** | GEX, Vanna, Charm feature engineering | P1 | 5 days | High alpha signal for short-term prediction |
| **2** | Dark Pool Index and OFI features | P2 | 5 days | Institutional-grade leading indicators |
| **2** | HMM regime detection | P2 | 5 days | Improves model accuracy in all regimes |
| **2** | FinBERT-LSTM sentiment model | P2 | 7 days | Adds news-driven alpha signal |
| **2** | TFT model development | P2 | 10 days | State-of-the-art temporal forecasting |
| **2** | Stacking ensemble meta-learner | P2 | 5 days | Combines all models for best accuracy |
| **2** | Intraday prediction updates (4x daily) | P3 | 5 days | Increases utility for active traders |
| **2** | Macro feature expansion (CoT, credit spreads) | P3 | 3 days | Richer macro context for predictions |
| **2** | Evidently AI drift monitoring | P2 | 3 days | Detects model degradation automatically |
| **3** | FastAPI backend rebuild | P2 | 15 days | Production-grade, scalable architecture |
| **3** | React frontend rebuild | P2 | 20 days | Full layout control, WebSocket support |
| **3** | WebSocket real-time data streaming | P2 | 5 days | True real-time updates, sub-second latency |
| **3** | LLM AI Assistant (text + voice) | P3 | 15 days | Natural language interface to all platform data |
| **3** | Customizable dashboard layouts | P3 | 7 days | Personalized user experience |
| **3** | Automated model retraining pipeline | P3 | 7 days | Self-maintaining model performance |
| **3** | Structured logging and Slack/PagerDuty alerting | P2 | 3 days | Operational visibility and rapid incident response |

---

## References

[1] OWASP Cheat Sheet Series. "Secrets Management Cheat Sheet." *OWASP*, 2025. https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html

[2] Jiang, M., et al. "An improved Stacking framework for stock index prediction by leveraging tree-based ensemble models and deep learning algorithms." *Physica A*, 2020. https://www.sciencedirect.com/science/article/abs/pii/S0378437119313093

[3] Parker, M., et al. "Stock Price Prediction Using a Stacked Heterogeneous Ensemble." *MDPI IJFS*, 2025. https://www.mdpi.com/2227-7072/13/4/201

[4] Lynch, S., et al. "A Novel Hybrid Temporal Fusion Transformer Graph Neural Network Model for Stock Market Prediction." *MDPI AppliedMath*, 2025. https://www.mdpi.com/2673-9909/5/4/176

[5] "Market Regime Detection using Hidden Markov Models in QSTrader." *QuantStart*, n.d. https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/

[6] Ruan, L., et al. "Stock Price Prediction Using FinBERT-Enhanced..." *MDPI Mathematics*, 2025. https://www.mdpi.com/2227-7390/13/17/2747

[7] "What is GEX? The Complete Guide to Gamma Exposure." *Options Trading IQ*, 2023. https://optionstradingiq.com/what-is-gex/

[8] "Incorporating Liquidity and Order Flow Imbalances for Intraday Stock Volatility Prediction." *IEEE*, 2025.

[9] Kaabar, S. "Cracking The Dark Pool: Forecasting S&P 500 Using Machine Learning." *Medium*, 2024. https://kaabar-sofien.medium.com/cracking-the-dark-pool-forecasting-s-p-500-using-machine-learning-ce6fd1cc2055

[10] Jagannathan, J., et al. "Integration of a hybrid model with XAI for stock price forecasting: A dashboard-driven approach." *IEEE*, 2025.

[11] "Best Streamlit Alternatives for Production-Grade Data Apps." *Plotly Blog*, 2025. https://plotly.com/blog/best-streamlit-alternatives-production-data-apps/

[12] Shilaskar, S., et al. "Voice-Enabled AI Assistant for Real-Time Stock Market Analysis and Prediction Using LSTM and NLP." *Springer*, 2025.

[13] "From Data To Decisions: UX Strategies For Real-Time Dashboards." *Smashing Magazine*, 2025. https://www.smashingmagazine.com/2025/09/ux-strategies-real-time-dashboards/

[14] "Real-Time Machine Learning Model Monitoring for Banking Fraud Detection: A Micro-Batch Approach with Evidently AI." *DiVA Portal*, 2025.
