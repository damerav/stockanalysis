# SPY/SPX-Specific Gap Analysis & Kiro Prompt

**Authored by:** Manus AI
**Date:** March 4, 2026

## 1. Executive Summary

This report provides a precise, SPY/SPX-specific gap analysis of the `stockanalysis` platform. The audit confirms that the platform has an exceptionally deep and sophisticated feature set tailored specifically for SPY/SPX, but it has two key gaps that, if filled, would significantly enhance its predictive power:

1.  **Index-Level Fundamentals:** The platform does not track the S&P 500's aggregate P/E ratio, earnings yield, dividend yield, or buyback yield. These are critical drivers of long-term valuation and can provide a powerful macro context for the daily prediction model.
2.  **Market Internals / Breadth:** The platform does not compute advance/decline lines, new highs/lows, or the percentage of stocks above their moving averages. These are classic, high-alpha indicators of market health and risk appetite.

This report includes a detailed Kiro prompt to bridge this gap by fetching the necessary data from **YCharts** and **Finviz** and integrating it into the feature pipeline.

## 2. SPY/SPX Feature Gap Analysis

| Feature Category | Implemented? | Details & Gaps |
| :--- | :--- | :--- |
| **Technical Indicators** | **Partial** | Implements ~10 core indicators (SMA, RSI, MACD, etc.) but lacks a comprehensive library. **Gap:** No ADX, CCI, MFI, Parabolic SAR, Ichimoku, etc. |
| **Index Fundamentals** | **No** | **Critical Gap:** No S&P 500 aggregate P/E, earnings yield, dividend yield, or buyback yield. |
| **Market Internals** | **No** | **Critical Gap:** No Advance/Decline Line, New Highs/Lows, % Stocks > 50/200-day MA. |
| **Options Analytics** | **Yes** | Excellent coverage: GEX, Vanna, Charm, 0DTE PCR, Max Pain, IV Skew. |
| **Cross-Asset** | **Yes** | Excellent coverage: VIX, DXY, Gold, Oil, Bonds (TLT, HYG), Sector Ratios. |
| **Macro** | **Yes** | Good coverage: Fed Funds Rate, Beige Book sentiment, FOMC hawkishness, CPI/NFP event proximity. |
| **Sentiment** | **Yes** | Excellent coverage: FinBERT news sentiment, LLM-based analysis, Geopolitical risk. |

## 3. Kiro Implementation Prompt

This prompt will instruct Kiro to implement the missing index fundamentals and market internals features.

### Part 1: Fetch Index Fundamentals from YCharts

1.  **Create `ycharts_fetcher.py`:** Create a new file `src/data/ycharts_fetcher.py`.
2.  **Implement Web Scraper:** In the new file, create a function `get_sp500_fundamentals()` that uses `requests` and `BeautifulSoup` to scrape the following data points from the public YCharts S&P 500 page (`https://ycharts.com/indicators/sp_500_pe_ratio` and related pages):
    *   S&P 500 P/E Ratio
    *   S&P 500 Earnings Yield
    *   S&P 500 Dividend Yield
    *   S&P 500 Buyback Yield
3.  **Update `fetcher.py`:** In `fetcher.py`, import and call `get_sp500_fundamentals()` and add the results to the daily macro data pull.

### Part 2: Fetch Market Internals from Finviz

1.  **Create `finviz_fetcher.py`:** Create a new file `src/data/finviz_fetcher.py`.
2.  **Implement Web Scraper:** In the new file, create a function `get_market_internals()` that scrapes the following from the Finviz homepage (`https://finviz.com`):
    *   Advance/Decline Line (Stocks only)
    *   New Highs / New Lows
    *   % Stocks above 50-day MA
    *   % Stocks above 200-day MA
3.  **Update `fetcher.py`:** In `fetcher.py`, import and call `get_market_internals()`.

### Part 3: Integrate New Features into the Model

1.  **Update `features.py`:** In `build_feature_vector`, add the new index fundamental and market internal data to the feature vector.
2.  **Update `get_feature_columns()`:** Add the names of all the new features to the list returned by this function.

---
