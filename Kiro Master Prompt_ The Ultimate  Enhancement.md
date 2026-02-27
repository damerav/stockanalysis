# Kiro Master Prompt: The Ultimate `stockanalysis` Enhancement

> **How to use this prompt:** This is a consolidated master prompt. Paste the entire content below the horizontal rule into Kiro's agent chat. Kiro will read all four specified repositories and implement the full, combined feature set in a single, optimized workflow.

---

## 1. Task Overview

Your task is to perform a comprehensive enhancement of the `damerav/stockanalysis` repository by integrating the best features and concepts from three other projects. You will add **two new pages** to the Streamlit UI, implement **two new ML models**, build a complete **news analysis pipeline**, and significantly expand the platform's feature engineering capabilities.

## 2. Source Repositories

| Repository | Purpose |
| :--- | :--- |
| `damerav/stockanalysis` | **Target Project:** The base platform you will be enhancing. |
| `ErikThiart/ai-stock-dashboard` | **UI/UX Blueprint:** Source for the single-stock analysis page layout, charts, and data storytelling. |
| `Finance-And-ML/US-Stock-Prediction-Using-ML-And-Spark` | **News Pipeline Concept:** Source for the news-driven prediction model, NLP pipeline, and feature engineering. |
| `SevilayMuni/stock-prediction-web-app` | **Price Forecaster Model:** Source for the multi-day LSTM price regression model and its UI. |

Clone the reference projects if needed:
`git clone https://github.com/ErikThiart/ai-stock-dashboard.git /tmp/ai-stock-dashboard`
`git clone https://github.com/Finance-And-ML/US-Stock-Prediction-Using-ML-And-Spark.git /tmp/spark-ml-repo`
`git clone https://github.com/SevilayMuni/stock-prediction-web-app.git /tmp/stock-prediction-web-app`

---

## 3. Implementation Steps (Consolidated)

### Step 3.1: Configuration (`config.yaml`)

Update the main configuration file with parameters for the new models and features.

```yaml
# In config.yaml

# Add 200 to sma_periods
technicals:
  sma_periods: [20, 50, 200]
  # ... (keep existing technicals)

# Add new section for the LSTM price forecaster
lstm_predictor:
  enabled: true
  n_past: 21
  epochs: 50
  batch_size: 32
  feature_list: ["close", "garman_klass_vol", "dollar_volume", "obv", "ma_3_days"]

# Add new section for the News/Sentiment pipeline
news_pipeline:
  enabled: true
  db_path: "./data/news.db"
  # ... (add other relevant configs as you build)
```

### Step 3.2: Feature Engineering (`src/data/features.py`)

Enhance the feature library with new technical indicators.

1.  **Add `compute_obv()`**: On-Balance Volume. Logic from `/tmp/stock-prediction-web-app/streamlit_app.py`.
2.  **Add `compute_garman_klass_volatility()`**: Garman-Klass Volatility. Logic from `/tmp/stock-prediction-web-app/streamlit_app.py`.
3.  **Add `compute_stochastic()`**: Stochastic Oscillator. Logic from `/tmp/ai-stock-dashboard/stock_dashboard.py`.
4.  **Update `compute_all_technicals()`**: Integrate the three new functions. Also, refactor the SMA computation to loop through `config["technicals"]["sma_periods"]` to dynamically create `sma_{period}` columns.
5.  **Update `store_technicals()`**: Update the `INSERT OR REPLACE` statement in DuckDB to include the new columns: `obv`, `garman_klass_vol`, `stoch_k`, `stoch_d`, and all dynamic SMAs.

### Step 3.3: News & Sentiment Pipeline (New Files)

Build the end-to-end news analysis pipeline inspired by the Spark repository.

1.  **`src/data/news_fetcher.py`**: Create `NewsFetcher` class. Implement methods to scrape Reuters/WSJ (using `playwright`) and fetch from Finnhub. Store results in a new SQLite DB at `./data/news.db`.
2.  **`src/data/news_features.py`**: Create `NewsFeatureProcessor` class. Implement text cleaning, ticker-alias mapping, N-gram generation (`nltk`), TF-IDF vectorization (`sklearn`), and VADER sentiment scoring. Store processed features in a new `news_features` table in `analytics.duckdb`.
3.  **`src/model/news_predictor.py`**: Create `NewsPredictor` class. Implement target variable creation (classify price change at +15m, +60m, +4h into 5 buckets). Train an `XGBClassifier` on TF-IDF vectors and sentiment scores. Serialize the trained model and vectorizer.
4.  **`src/pipeline/news_pipeline_run.py`**: Create an orchestration script that runs the fetcher, feature processor, and model trainer in sequence.

### Step 3.4: LSTM Price Forecaster (New File)

Build the dedicated Keras/TensorFlow LSTM model for multi-day price regression.

1.  **`src/model/lstm_predictor.py`**: Create `LSTMPredictor` class. It should be a scikit-learn compatible wrapper around a Keras `Sequential` model. It must have `fit`, `predict` (forecasting next 5 days), `save`, and `load` methods.

### Step 3.5: UI Implementation (New & Updated Files)

Create the two new pages and integrate them into the main application.

1.  **Create `src/dashboard/single_stock_app.py`**: Implement `page_single_stock()`.
    *   **UI**: Use a sidebar for ticker input and period selection. The main area should have a KPI metrics row, a performance panel (Total Return, Sharpe, etc.), and a tabbed interface.
    *   **Tabs**:
        *   `📋 Company Info`: Display company metadata from `yfinance`.
        *   `📊 Raw Data`: Display the raw OHLCV data with a CSV download button.
        *   `🔧 Technical Chart`: A 4-panel Plotly chart (Price+MAs+BB, Volume, MACD, RSI+Stoch), adapted from `ai-stock-dashboard`.
        *   `🤖 AI Analysis`: A rule-based narrative combining technicals and the HMM regime state. Also include the feature importance bar chart from the trained `SPYPredictor`.
        *   `📰 News & Sentiment`: The UI for the news pipeline. Show a recent news feed, a sentiment gauge, a sentiment-over-time chart, and the news-based price movement prediction from your `NewsPredictor` model.

2.  **Create `src/dashboard/forecast_app.py`**: Implement `page_forecast()`.
    *   **UI**: A simple page with a stock selector.
    *   **Functionality**: On selection, load the `LSTMPredictor` model, generate a 5-day forecast, and display:
        *   A 2-column layout with a 5-day prediction table and an "Insight" box (predicted vs. actual price + % change).
        *   A Plotly chart showing historical price + the 5-day forecast path.

3.  **Update `src/dashboard/app.py`**: Import the two new page functions and add them to the `PAGES` dictionary for sidebar navigation: `"🔮 Forecast": page_forecast` and `"🔍 Single-Stock Analysis": page_single_stock`.

---

## 4. Acceptance Criteria (Consolidated)

1.  **Configuration**: `config.yaml` is updated with `sma_periods`, `lstm_predictor`, and `news_pipeline` sections.
2.  **Features**: `features.py` computes and stores OBV, Garman-Klass Volatility, and Stochastic Oscillator.
3.  **News Pipeline**: The news pipeline can be executed via `python -m src.pipeline.news_pipeline_run` and populates `news.db` and the `news_features` table in `analytics.duckdb`.
4.  **LSTM Forecaster**: The `LSTMPredictor` can be trained, saved, loaded, and used to generate 5-day price forecasts.
5.  **UI - Forecast Page**: The "🔮 Forecast" page is in the sidebar and correctly displays the 5-day price forecast table, insight box, and chart.
6.  **UI - Single-Stock Page**: The "🔍 Single-Stock Analysis" page is in the sidebar and correctly displays all its components: KPIs, performance metrics, the 4-panel technical chart, the AI narrative, and the new "News & Sentiment" tab.
7.  **No Regressions**: All previously existing functionality in `stockanalysis` remains completely intact and operational.
