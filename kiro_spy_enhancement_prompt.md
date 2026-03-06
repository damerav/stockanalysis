# Kiro Prompt: SPY/SPX Prediction Enhancements
## Based on Community Survey of Free Predictive Frameworks

**Authored by:** Manus AI | **Date:** March 4, 2026
**Target Repository:** `damerav/stockanalysis`

---

## Context & Codebase State

You are working on a production Streamlit + Python trading intelligence platform at `damerav/stockanalysis`. The platform predicts next-day SPY/SPX direction using an XGBoost ensemble with 136 features. The codebase uses PostgreSQL (via `DbRouter` with SQLAlchemy 2.0), `yfinance`, `Polygon.io`, FRED, and Finnhub as data sources.

**Do NOT modify:** `db_router.py`, `init_db.py` (schema migrations only via `_migrate_schema`), any existing feature columns in `get_feature_columns()` (only append new ones), or any ES strategy files.

**Key integration points:**
- `src/data/fetcher.py` — `FallbackFetcher.get_macro_fred()` fetches macro data
- `src/data/market_breadth.py` — `fetch_index_fundamentals()` and `fetch_market_breadth()` fetch index-level data
- `src/data/features.py` — `build_feature_vector()` assembles the full feature matrix; `get_feature_columns()` lists all model features
- `src/pipeline/daily_run.py` — orchestrates the daily data pull and feature computation
- `src/dashboard/app.py` — `page_spy()` is the main SPY prediction dashboard

---

## Gap Analysis: Document vs. Current Implementation

The `FreeSPYPredictionSites.md` document identifies the following analytical frameworks. This table maps each to the current implementation status:

| Framework from Document | Status | Gap |
| :--- | :--- | :--- |
| Technical indicators (SMA, EMA, RSI, MACD, BB, ATR, OBV, Stochastic) | **DONE** | None |
| VIX term structure (VIX9D, VIX3M, VIX6M, VVIX, SKEW) | **DONE** | None |
| Options analytics (GEX, Vanna, Charm, 0DTE PCR, Max Pain, IV Skew) | **DONE** | None |
| Market breadth (A/D ratio, % above 50/200MA, new highs/lows) | **DONE** | None |
| Index fundamentals (P/E, forward P/E, earnings yield, dividend yield) | **DONE** | None |
| Cross-asset signals (HYG, TLT, EEM, XLK, XLF, XLE, Gold, Crude) | **DONE** | None |
| FinBERT news sentiment | **DONE** | None |
| Geopolitical risk scoring | **DONE** | None |
| Fed communication sentiment (FOMC, Beige Book) | **DONE** | None |
| Calendar/event features (FOMC, CPI, NFP, OPEX, triple witching) | **DONE** | None |
| **Shiller CAPE Ratio** | **MISSING** | Not fetched or used as feature |
| **Buffett Indicator (Market Cap / GDP)** | **MISSING** | Not fetched or used as feature |
| **Sahm Rule (real-time recession indicator)** | **MISSING** | Not fetched from FRED |
| **Yield Curve (10Y-3M spread)** | **MISSING** | Only 10Y yield; no 3M or spread |
| **Consumer Confidence (U. of Michigan)** | **MISSING** | Not fetched from FRED |
| **ISM Manufacturing PMI** | **MISSING** | Not fetched from FRED |
| **Comprehensive technical indicators** (ADX, CCI, MFI, Williams %R, Parabolic SAR, Ichimoku, Donchian, Keltner multi-period, TRIX, DPO, CMF, Aroon, etc.) | **MISSING** | Only 8 indicators manually implemented |
| **Multi-timeframe technicals** (weekly RSI, monthly momentum) | **MISSING** | Only daily timeframe |
| **StockTwits social sentiment** (bullish/bearish ratio for SPY) | **MISSING** | No social media sentiment source |
| **Barchart Opinion score** (composite buy/sell rating) | **MISSING** | Not integrated |
| **DCF / Earnings Yield Gap** (vs. 10Y Treasury) | **MISSING** | Earnings yield exists but not compared to risk-free rate |
| **Relative Rotation** (SPY vs. sector ETFs) | **MISSING** | Only 2 sector ratios (XLK/XLF, XLK/XLE) |

---

## Implementation Plan

### Part 1: Extend FRED Macro Data — 5 New Series

**File:** `src/data/fetcher.py`
**Function:** `FallbackFetcher.get_macro_fred()`

Add the following 5 FRED series to the existing `series` dictionary. All use the same existing fetch pattern (API key if available, CSV fallback if not):

```python
# ADD these 5 entries to the existing series dict in get_macro_fred():
"us3m_yield":        "DTB3",          # 3-Month Treasury Bill rate
"yield_curve_10y3m": "T10Y3M",        # 10Y minus 3M spread (recession signal)
"sahm_rule":         "SAHMREALTIME",  # Sahm Rule real-time recession indicator
"consumer_conf":     "UMCSENT",       # U. of Michigan Consumer Sentiment
"ism_pmi":           "MANEMP",        # ISM Manufacturing Employment proxy (use NAPM for PMI)
```

**Note:** After fetching, compute the `earnings_yield_gap` as a derived field:
```python
# After the series loop, add this derived computation:
ey = result.get("sp500_earnings_yield")  # already fetched in market_breadth
ry = result.get("us10y_yield")
if ey is not None and ry is not None:
    result["earnings_yield_gap"] = round(ey - (ry / 100.0), 4)
else:
    result["earnings_yield_gap"] = None
```

---

### Part 2: Add Shiller CAPE and Buffett Indicator to `market_breadth.py`

**File:** `src/data/market_breadth.py`
**Function:** `fetch_index_fundamentals()` — extend the existing function

Add CAPE and Buffett Indicator fetching to the existing `fetch_index_fundamentals()` function. Both use free public data sources (no API key required):

```python
# ADD to fetch_index_fundamentals() after the existing yfinance block:

# --- Shiller CAPE Ratio (from datahub.io, free CSV) ---
try:
    cape_url = "https://datahub.io/core/s-and-p-500/r/data.csv"
    cape_df = pd.read_csv(cape_url)
    # Column names vary; find the CAPE column
    cape_col = next((c for c in cape_df.columns if "PE10" in c.upper() or "CAPE" in c.upper()), None)
    if cape_col:
        cape_df = cape_df.dropna(subset=[cape_col])
        result["sp500_cape"] = round(float(cape_df[cape_col].iloc[-1]), 2)
    else:
        result["sp500_cape"] = None
except Exception as e:
    logger.warning("CAPE fetch failed: %s", e)
    result["sp500_cape"] = None

# --- Buffett Indicator: Wilshire 5000 / GDP (from FRED, free CSV) ---
try:
    # Wilshire 5000 Total Market Index
    w5000_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WILL5000PR"
    w5000_df = pd.read_csv(w5000_url)
    w5000_df.columns = ["date", "value"]
    w5000_df["value"] = pd.to_numeric(w5000_df["value"], errors="coerce")
    w5000_df = w5000_df.dropna(subset=["value"])
    # Nominal GDP (quarterly, forward-filled)
    gdp_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP"
    gdp_df = pd.read_csv(gdp_url)
    gdp_df.columns = ["date", "gdp"]
    gdp_df["gdp"] = pd.to_numeric(gdp_df["gdp"], errors="coerce")
    gdp_df = gdp_df.dropna(subset=["gdp"])
    if not w5000_df.empty and not gdp_df.empty:
        latest_w5000 = float(w5000_df["value"].iloc[-1])
        latest_gdp = float(gdp_df["gdp"].iloc[-1])
        # Buffett Indicator = (Market Cap in billions) / (GDP in billions) * 100
        result["buffett_indicator"] = round((latest_w5000 / latest_gdp) * 100, 2)
    else:
        result["buffett_indicator"] = None
except Exception as e:
    logger.warning("Buffett Indicator fetch failed: %s", e)
    result["buffett_indicator"] = None
```

**Update `store_breadth_fundamentals()`:** Add `sp500_cape` and `buffett_indicator` to the `cols` list in `store_breadth_fundamentals()`.

**Update `init_db.py`:** Add `sp500_cape REAL` and `buffett_indicator REAL` columns to the `market_breadth` table in the `_migrate_schema()` function using `ALTER TABLE IF NOT EXISTS`.

---

### Part 3: Add Comprehensive Technical Indicators via `pandas-ta`

**File:** `src/data/features.py`
**Function:** `compute_all_technicals()`

1.  **Add to `requirements.txt`:** Add `pandas-ta>=0.3.14b` on a new line.

2.  **Add the following block** at the end of `compute_all_technicals()`, before the `return result` statement:

```python
# --- Comprehensive technicals via pandas-ta ---
try:
    import pandas_ta as ta
    # Build a clean OHLCV dataframe for pandas-ta
    pta_df = df[["open", "high", "low", "close", "volume"]].copy()
    pta_df.columns = ["Open", "High", "Low", "Close", "Volume"]

    # Compute a curated set of high-value indicators (not "all" to avoid 300+ columns)
    # Trend
    result["adx_14"]       = ta.adx(pta_df["High"], pta_df["Low"], pta_df["Close"], length=14)["ADX_14"]
    result["cci_20"]       = ta.cci(pta_df["High"], pta_df["Low"], pta_df["Close"], length=20)
    result["aroon_up"]     = ta.aroon(pta_df["High"], pta_df["Low"], length=14)["AROONU_14"]
    result["aroon_down"]   = ta.aroon(pta_df["High"], pta_df["Low"], length=14)["AROOND_14"]
    result["psar_long"]    = ta.psar(pta_df["High"], pta_df["Low"], pta_df["Close"])["PSARl_0.02_0.2"]
    result["psar_short"]   = ta.psar(pta_df["High"], pta_df["Low"], pta_df["Close"])["PSARs_0.02_0.2"]
    result["dpo_20"]       = ta.dpo(pta_df["Close"], length=20)
    result["trix_14"]      = ta.trix(pta_df["Close"], length=14)["TRIX_14_9"]
    result["vortex_pos"]   = ta.vortex(pta_df["High"], pta_df["Low"], pta_df["Close"], length=14)["VTXP_14"]
    result["vortex_neg"]   = ta.vortex(pta_df["High"], pta_df["Low"], pta_df["Close"], length=14)["VTXM_14"]
    # Momentum
    result["williams_r"]   = ta.willr(pta_df["High"], pta_df["Low"], pta_df["Close"], length=14)
    result["mfi_14"]       = ta.mfi(pta_df["High"], pta_df["Low"], pta_df["Close"], pta_df["Volume"], length=14)
    result["rsi_2"]        = ta.rsi(pta_df["Close"], length=2)    # Short-term RSI (mean reversion)
    result["rsi_9"]        = ta.rsi(pta_df["Close"], length=9)
    result["rsi_21"]       = ta.rsi(pta_df["Close"], length=21)
    result["cmo_14"]       = ta.cmo(pta_df["Close"], length=14)   # Chande Momentum Oscillator
    result["ppo"]          = ta.ppo(pta_df["Close"])["PPO_12_26_9"]  # Percentage Price Oscillator
    result["roc_5"]        = ta.roc(pta_df["Close"], length=5)
    result["roc_21"]       = ta.roc(pta_df["Close"], length=21)
    # Volatility
    result["kc_upper_20"]  = ta.kc(pta_df["High"], pta_df["Low"], pta_df["Close"], length=20)["KCUe_20_2"]
    result["kc_lower_20"]  = ta.kc(pta_df["High"], pta_df["Low"], pta_df["Close"], length=20)["KCLe_20_2"]
    result["atr_7"]        = ta.atr(pta_df["High"], pta_df["Low"], pta_df["Close"], length=7)
    result["atr_21"]       = ta.atr(pta_df["High"], pta_df["Low"], pta_df["Close"], length=21)
    result["donchian_high"]= ta.donchian(pta_df["High"], pta_df["Low"], length=20)["DCH_20_20.0"]
    result["donchian_low"] = ta.donchian(pta_df["High"], pta_df["Low"], length=20)["DCL_20_20.0"]
    result["ulcer_14"]     = ta.ui(pta_df["Close"], length=14)    # Ulcer Index (drawdown risk)
    # Volume
    result["cmf_20"]       = ta.cmf(pta_df["High"], pta_df["Low"], pta_df["Close"], pta_df["Volume"], length=20)
    result["vwma_20"]      = ta.vwma(pta_df["Close"], pta_df["Volume"], length=20)
    result["pvol"]         = ta.pvol(pta_df["Close"], pta_df["Volume"])  # Price-Volume
    result["eom_14"]       = ta.eom(pta_df["High"], pta_df["Low"], pta_df["Close"], pta_df["Volume"], length=14)
    # Moving Averages (additional)
    result["ema_9"]        = ta.ema(pta_df["Close"], length=9)
    result["ema_21"]       = ta.ema(pta_df["Close"], length=21)
    result["ema_200"]      = ta.ema(pta_df["Close"], length=200)
    result["hma_20"]       = ta.hma(pta_df["Close"], length=20)   # Hull MA (fast, low-lag)
    result["wma_20"]       = ta.wma(pta_df["Close"], length=20)
    result["dema_20"]      = ta.dema(pta_df["Close"], length=20)  # Double EMA
    result["tema_20"]      = ta.tema(pta_df["Close"], length=20)  # Triple EMA
    result["kama_10"]      = ta.kama(pta_df["Close"], length=10)  # Kaufman Adaptive MA
    # Ichimoku Cloud (key levels only)
    ichimoku = ta.ichimoku(pta_df["High"], pta_df["Low"], pta_df["Close"])
    if ichimoku is not None and len(ichimoku) == 2:
        ichi_df = ichimoku[0]
        result["ichi_tenkan"]  = ichi_df.get("ITS_9")
        result["ichi_kijun"]   = ichi_df.get("IKS_26")
        result["ichi_senkou_a"]= ichi_df.get("ISA_9")
        result["ichi_senkou_b"]= ichi_df.get("ISB_26")
    logger.debug("pandas-ta indicators computed: %d new columns", 40)
except ImportError:
    logger.warning("pandas-ta not installed — skipping comprehensive technicals. Run: pip install pandas-ta")
except Exception as e:
    logger.warning("pandas-ta computation failed: %s", e)
```

---

### Part 4: Add Multi-Timeframe Technical Features

**File:** `src/data/features.py`
**Function:** `build_feature_vector()` — add a new section after the existing technical indicators section

Multi-timeframe analysis is a key technique used by professional platforms. Add weekly and monthly RSI and momentum features by resampling the daily price data:

```python
# --- Multi-Timeframe Technical Features ---
try:
    # Resample daily prices to weekly and monthly
    price_df_indexed = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    price_df_indexed.index = pd.to_datetime(price_df_indexed.index)

    # Weekly OHLCV
    weekly = price_df_indexed.resample("W").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    # Monthly OHLCV
    monthly = price_df_indexed.resample("ME").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    if len(weekly) >= 14:
        # Weekly RSI
        weekly_rsi = compute_rsi(weekly["close"], 14)
        # Map back to daily: forward-fill weekly value onto each day of that week
        df["weekly_rsi"] = df["date"].map(
            pd.Series(weekly_rsi.values, index=weekly.index.strftime("%Y-%m-%d"))
        ).ffill()
        # Weekly momentum (5-week)
        weekly_mom = weekly["close"].pct_change(5)
        df["weekly_momentum_5w"] = df["date"].map(
            pd.Series(weekly_mom.values, index=weekly.index.strftime("%Y-%m-%d"))
        ).ffill()
        # Weekly MACD histogram
        _, _, weekly_macd_hist = compute_macd(weekly["close"])
        df["weekly_macd_hist"] = df["date"].map(
            pd.Series(weekly_macd_hist.values, index=weekly.index.strftime("%Y-%m-%d"))
        ).ffill()

    if len(monthly) >= 12:
        # Monthly RSI
        monthly_rsi = compute_rsi(monthly["close"], 14)
        df["monthly_rsi"] = df["date"].map(
            pd.Series(monthly_rsi.values, index=monthly.index.strftime("%Y-%m-%d"))
        ).ffill()
        # Monthly momentum (3-month)
        monthly_mom = monthly["close"].pct_change(3)
        df["monthly_momentum_3m"] = df["date"].map(
            pd.Series(monthly_mom.values, index=monthly.index.strftime("%Y-%m-%d"))
        ).ffill()

    # Fill any remaining NaN from forward-fill
    for col in ["weekly_rsi", "weekly_momentum_5w", "weekly_macd_hist",
                "monthly_rsi", "monthly_momentum_3m"]:
        if col in df.columns:
            df[col] = df[col].fillna(method="bfill").fillna(0)
        else:
            df[col] = 0.0

    logger.debug("Multi-timeframe features computed.")
except Exception as e:
    logger.warning("Multi-timeframe features failed: %s", e)
    for col in ["weekly_rsi", "weekly_momentum_5w", "weekly_macd_hist",
                "monthly_rsi", "monthly_momentum_3m"]:
        df[col] = 0.0
```

---

### Part 5: Add StockTwits Social Sentiment

**File:** Create `src/data/social_fetcher.py`

Create a new file with the following content:

```python
"""Social Sentiment Fetcher — StockTwits SPY sentiment (free, no API key).

Scrapes the StockTwits symbol page for SPY to get bullish/bearish message counts.
Falls back to a neutral score (0.5) if scraping fails.
"""
import logging
import time
from typing import Optional
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_CACHE: dict = {"data": None, "updated": 0}
_CACHE_TTL = 3600  # 1 hour


def get_stocktwits_sentiment(ticker: str = "SPY") -> dict:
    """Fetch StockTwits bullish/bearish sentiment for a ticker.

    Returns:
        dict with keys:
          st_bullish_pct: float 0-1, fraction of messages that are bullish
          st_bearish_pct: float 0-1, fraction of messages that are bearish
          st_bull_bear_ratio: float, bullish / max(bearish, 1)
          st_message_volume: int, approximate message count in last 24h
    """
    now = time.time()
    if _CACHE["data"] and (now - _CACHE["updated"]) < _CACHE_TTL:
        return _CACHE["data"]

    result = {
        "st_bullish_pct": 0.5,
        "st_bearish_pct": 0.5,
        "st_bull_bear_ratio": 1.0,
        "st_message_volume": 0,
    }

    try:
        url = f"https://stocktwits.com/symbol/{ticker.upper()}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # StockTwits embeds sentiment data in a <script> tag as JSON
        import json, re
        scripts = soup.find_all("script", type="application/json")
        for script in scripts:
            try:
                data = json.loads(script.string or "")
                # Navigate the JSON to find sentiment counts
                # Structure varies; search recursively for bullish/bearish keys
                text = json.dumps(data)
                bull_match = re.search(r'"bullish"\s*:\s*(\d+)', text)
                bear_match = re.search(r'"bearish"\s*:\s*(\d+)', text)
                if bull_match and bear_match:
                    bull = int(bull_match.group(1))
                    bear = int(bear_match.group(1))
                    total = bull + bear
                    if total > 0:
                        result["st_bullish_pct"] = round(bull / total, 4)
                        result["st_bearish_pct"] = round(bear / total, 4)
                        result["st_bull_bear_ratio"] = round(bull / max(bear, 1), 3)
                        result["st_message_volume"] = total
                        logger.info("StockTwits %s: bull=%.1f%%, bear=%.1f%%, vol=%d",
                                    ticker, result["st_bullish_pct"] * 100,
                                    result["st_bearish_pct"] * 100, total)
                        break
            except (json.JSONDecodeError, AttributeError):
                continue

        _CACHE["data"] = result
        _CACHE["updated"] = now

    except Exception as e:
        logger.warning("StockTwits sentiment fetch failed for %s: %s", ticker, e)

    return result
```

**Update `src/data/features.py`:** Add the following import at the top of `features.py`:
```python
from src.data.social_fetcher import get_stocktwits_sentiment
```

Then add the following block inside `build_feature_vector()`, after the FinBERT sentiment section:

```python
# --- StockTwits Social Sentiment ---
try:
    st_data = get_stocktwits_sentiment("SPY")
    for col in ["st_bullish_pct", "st_bearish_pct", "st_bull_bear_ratio", "st_message_volume"]:
        df[col] = st_data.get(col, 0.0)
    logger.debug("StockTwits sentiment added: bull=%.2f", st_data.get("st_bullish_pct", 0))
except Exception as e:
    logger.warning("StockTwits features failed: %s", e)
    for col in ["st_bullish_pct", "st_bearish_pct", "st_bull_bear_ratio", "st_message_volume"]:
        df[col] = 0.0
```

---

### Part 6: Add Earnings Yield Gap Feature

**File:** `src/data/features.py`
**Function:** `build_feature_vector()`

The document highlights the **Earnings Yield Gap** (S&P 500 earnings yield minus 10Y Treasury yield) as a key valuation signal. Add this derived feature after the macro data is merged into the feature vector:

```python
# --- Earnings Yield Gap (SPY earnings yield vs. 10Y Treasury) ---
# This is the "Fed Model" — a positive gap means equities are cheap vs. bonds
try:
    if "sp500_earnings_yield" in df.columns and "us10y_yield" in df.columns:
        df["earnings_yield_gap"] = (
            df["sp500_earnings_yield"] - (df["us10y_yield"] / 100.0)
        ).round(4)
    else:
        df["earnings_yield_gap"] = 0.0
except Exception as e:
    logger.warning("Earnings yield gap failed: %s", e)
    df["earnings_yield_gap"] = 0.0
```

---

### Part 7: Add Relative Sector Rotation Features

**File:** `src/data/fetcher.py`
**Function:** `FallbackFetcher.get_cross_asset_signals()`

The document highlights Relative Rotation Graphs (RRG) as a key tool. Add 8 more sector ETF ratios to the existing cross-asset signals to capture sector rotation dynamics:

```python
# ADD these tickers to the existing tickers dict in get_cross_asset_signals():
"xlv": "XLV",    # Healthcare
"xli": "XLI",    # Industrials
"xlu": "XLU",    # Utilities (defensive)
"xlb": "XLB",    # Materials
"xlp": "XLP",    # Consumer Staples (defensive)
"xly": "XLY",    # Consumer Discretionary
"xlre": "XLRE",  # Real Estate
"qqq": "QQQ",    # Nasdaq-100 (growth vs. value)
"iwm": "IWM",    # Russell 2000 (small cap risk appetite)
"dia": "DIA",    # Dow Jones (value vs. growth)
```

Then in `src/data/features.py`, inside `build_feature_vector()`, add the following derived ratio features after the existing sector ratio section:

```python
# --- Extended Sector Rotation Ratios ---
# Defensive vs. Offensive ratio: (XLU + XLP) / (XLY + XLK)
if all(c in df.columns for c in ["xlu", "xlp", "xly", "xlk"]):
    df["defensive_offensive_ratio"] = (
        (df["xlu"] + df["xlp"]) / (df["xly"] + df["xlk"] + 1e-9)
    ).round(4)
else:
    df["defensive_offensive_ratio"] = 0.0

# QQQ vs. IWM ratio (growth vs. small cap risk)
if "qqq" in df.columns and "iwm" in df.columns:
    df["qqq_iwm_ratio"] = (df["qqq"] / df["iwm"].replace(0, np.nan)).round(4).fillna(1.0)
else:
    df["qqq_iwm_ratio"] = 1.0

# Healthcare vs. Energy (defensive vs. cyclical)
if "xlv" in df.columns and "xle" in df.columns:
    df["xlv_xle_ratio"] = (df["xlv"] / df["xle"].replace(0, np.nan)).round(4).fillna(1.0)
else:
    df["xlv_xle_ratio"] = 1.0
```

---

### Part 8: Update `get_feature_columns()` and DB Schema

**File:** `src/data/features.py`
**Function:** `get_feature_columns()`

Append all new feature names to the list returned by `get_feature_columns()`. Add them in a clearly labeled section at the end:

```python
# --- Part 8: New features from FreeSPYPredictionSites enhancements ---
# Macro / Valuation
"sp500_cape",
"buffett_indicator",
"sahm_rule",
"yield_curve_10y3m",
"us3m_yield",
"consumer_conf",
"ism_pmi",
"earnings_yield_gap",
# Comprehensive Technicals (pandas-ta)
"adx_14", "cci_20", "aroon_up", "aroon_down",
"psar_long", "psar_short", "dpo_20", "trix_14",
"vortex_pos", "vortex_neg", "williams_r", "mfi_14",
"rsi_2", "rsi_9", "rsi_21", "cmo_14", "ppo",
"roc_5", "roc_21",
"kc_upper_20", "kc_lower_20", "atr_7", "atr_21",
"donchian_high", "donchian_low", "ulcer_14",
"cmf_20", "vwma_20", "pvol", "eom_14",
"ema_9", "ema_21", "ema_200",
"hma_20", "wma_20", "dema_20", "tema_20", "kama_10",
"ichi_tenkan", "ichi_kijun", "ichi_senkou_a", "ichi_senkou_b",
# Multi-Timeframe
"weekly_rsi", "weekly_momentum_5w", "weekly_macd_hist",
"monthly_rsi", "monthly_momentum_3m",
# Social Sentiment
"st_bullish_pct", "st_bearish_pct", "st_bull_bear_ratio", "st_message_volume",
# Sector Rotation
"defensive_offensive_ratio", "qqq_iwm_ratio", "xlv_xle_ratio",
"xlv", "xli", "xlu", "xlb", "xlp", "xly", "xlre", "qqq", "iwm", "dia",
```

**File:** `src/data/init_db.py`
**Function:** `_migrate_schema()`

Add `ALTER TABLE IF NOT EXISTS` statements for the new columns in the `market_breadth` table:

```python
# ADD these to _migrate_schema() in the market_breadth section:
_safe_alter(conn, "ALTER TABLE market_breadth ADD COLUMN IF NOT EXISTS sp500_cape REAL")
_safe_alter(conn, "ALTER TABLE market_breadth ADD COLUMN IF NOT EXISTS buffett_indicator REAL")
```

---

### Part 9: Add Valuation Context Panel to SPY Dashboard

**File:** `src/dashboard/app.py`
**Function:** `page_spy()`

Add a new expandable "Market Valuation Context" section to the SPY Predictor page. This implements the document's recommendation for a **Contextual Layer** that shows whether the market is overvalued or undervalued. Insert this block after the main prediction metrics row:

```python
# --- Market Valuation Context Panel ---
with st.expander("📊 Market Valuation Context", expanded=False):
    try:
        router = get_router(config)
        breadth_df = router.query(
            "SELECT * FROM market_breadth ORDER BY date DESC LIMIT 1"
        )
        if not breadth_df.empty:
            row = breadth_df.iloc[0]
            col1, col2, col3, col4 = st.columns(4)

            # Shiller CAPE
            cape = row.get("sp500_cape")
            cape_signal = "🔴 Overvalued" if cape and cape > 30 else ("🟡 Elevated" if cape and cape > 20 else "🟢 Fair")
            col1.metric("Shiller CAPE", f"{cape:.1f}" if cape else "N/A",
                        help="Cyclically Adjusted P/E. Historical avg ~17. >30 = historically overvalued.")
            col1.caption(cape_signal)

            # Buffett Indicator
            buffett = row.get("buffett_indicator")
            buffett_signal = "🔴 Strongly OV" if buffett and buffett > 150 else ("🟡 Overvalued" if buffett and buffett > 100 else "🟢 Fair")
            col2.metric("Buffett Indicator", f"{buffett:.0f}%" if buffett else "N/A",
                        help="Market Cap / GDP. >100% = overvalued. >150% = strongly overvalued.")
            col2.caption(buffett_signal)

            # Earnings Yield Gap
            ey_gap = row.get("earnings_yield_gap")
            ey_signal = "🟢 Equities Cheap" if ey_gap and ey_gap > 0 else "🔴 Bonds Better"
            col3.metric("Earnings Yield Gap", f"{ey_gap:.2%}" if ey_gap else "N/A",
                        help="S&P 500 Earnings Yield minus 10Y Treasury Yield. Positive = equities attractive vs. bonds.")
            col3.caption(ey_signal)

            # Yield Curve
            yc = row.get("yield_curve_10y3m")
            yc_signal = "🔴 Inverted (Recession Risk)" if yc and yc < 0 else "🟢 Normal"
            col4.metric("Yield Curve (10Y-3M)", f"{yc:.2f}%" if yc else "N/A",
                        help="10-Year minus 3-Month Treasury spread. Negative = inverted = recession signal.")
            col4.caption(yc_signal)
        else:
            st.info("Run the daily pipeline to populate valuation data.")
    except Exception as e:
        st.warning(f"Valuation context unavailable: {e}")
```

---

## Summary of Changes

| File | Action | New Features Added |
| :--- | :--- | :--- |
| `requirements.txt` | Add dependency | `pandas-ta>=0.3.14b` |
| `src/data/fetcher.py` | Extend `get_macro_fred()` | CAPE, Buffett, Sahm, Yield Curve, Consumer Conf., ISM PMI |
| `src/data/market_breadth.py` | Extend `fetch_index_fundamentals()` | Shiller CAPE, Buffett Indicator |
| `src/data/features.py` | Extend `compute_all_technicals()` | 40+ pandas-ta indicators |
| `src/data/features.py` | Extend `build_feature_vector()` | Multi-timeframe, social sentiment, earnings yield gap, sector rotation |
| `src/data/features.py` | Extend `get_feature_columns()` | All 60+ new feature names |
| `src/data/social_fetcher.py` | **Create new file** | StockTwits bullish/bearish sentiment |
| `src/data/init_db.py` | Extend `_migrate_schema()` | `sp500_cape`, `buffett_indicator` columns |
| `src/dashboard/app.py` | Extend `page_spy()` | Valuation Context panel (CAPE, Buffett, EY Gap, Yield Curve) |

**Total new features:** ~65 (40 pandas-ta technicals, 5 multi-timeframe, 4 social sentiment, 8 macro/valuation, 8 sector rotation)
