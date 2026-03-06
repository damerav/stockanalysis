---

**Kiro Prompt: Sentiment & Demand/Supply Enhancement**

**Goal:** Enhance the platform's sentiment and demand/supply analysis capabilities by adding the CNN Fear & Greed Index, the Arms Index (TRIN), and a proxy for Cumulative Volume Delta (CVD), then representing all new metrics on the main dashboard.

---

### Part 1: Data Layer — `market_breadth.py` & `init_db.py`

**1.1. Add `fear_greed_index` to `market_breadth` table:**

- **File:** `src/data/init_db.py`
- **Action:** In the `CREATE TABLE IF NOT EXISTS market_breadth` statement, add a new column:

```sql
-- Add this line after breadth_thrust REAL
fear_greed_index INTEGER
```

**1.2. Create `fear_greed_fetcher.py`:**

- **File:** `src/data/fear_greed_fetcher.py` (new file)
- **Action:** Create a new fetcher to scrape the Fear & Greed Index from `alternative.me`.

```python
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

def fetch_fear_greed_index() -> dict:
    """Fetch the Fear & Greed Index from alternative.me."""
    result = {}
    try:
        url = "https://alternative.me/crypto/fear-and-greed-index/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        # Find the main gauge div
        fng_div = soup.find('div', class_='fng-circle')
        if fng_div:
            result['fear_greed_index'] = int(fng_div.text)
            logger.info(f"Fetched Fear & Greed Index: {result['fear_greed_index']}")
    except Exception as e:
        logger.warning(f"fetch_fear_greed_index failed: {e}")
    return result
```

**1.3. Add `trin` and `obv` to `market_breadth.py`:**

- **File:** `src/data/market_breadth.py`
- **Action:** In the `fetch_market_breadth` function, add the calculation for TRIN and OBV.

```python
# Inside fetch_market_breadth, after advance/decline calculation

# TRIN (Arms Index)
adv_vol = volumes[latest > prev].sum()
decl_vol = volumes[latest < prev].sum()
if declines > 0 and decl_vol > 0:
    result["trin"] = (advances / declines) / (adv_vol / decl_vol)

# OBV is already in features.py, but we can add a market-wide version here
# This requires historical data, so we'll do it in features.py instead.
```

**1.4. Update `store_breadth_fundamentals`:**

- **File:** `src/data/market_breadth.py`
- **Action:** Add `fear_greed_index` and `trin` to the `cols` list and the `INSERT` statement.

```python
# In store_breadth_fundamentals
cols = [..., "buffett_indicator", "fear_greed_index", "trin"]
```

### Part 2: Pipeline Integration — `daily_run.py`

- **File:** `src/pipeline/daily_run.py`
- **Action:** In `_step97_market_breadth`, import and call the new fetcher.

```python
# In _step97_market_breadth
from src.data.fear_greed_fetcher import fetch_fear_greed_index

# ... inside the try block
fundamentals = fetch_index_fundamentals()
breadth = fetch_market_breadth()
fear_greed = fetch_fear_greed_index()

# Merge all data before storing
merged_data = {**fundamentals, **breadth, **fear_greed}

store_breadth_fundamentals(self.router, self.today, merged_data)

return {"market_data": merged_data}
```

### Part 3: Feature Engineering — `features.py`

- **File:** `src/data/features.py`
- **Action:** Add `fear_greed_index` and `trin` to the feature vector.

```python
# In build_feature_vector, inside the breadth_df merge section

breadth_cols = [..., "breadth_thrust", "fear_greed_index", "trin"]

# In the final get_feature_columns() list
"fear_greed_index",
"trin",
```

### Part 4: Dashboard Representation — `app.py`

- **File:** `src/dashboard/app.py`
- **Action:** Add the Fear & Greed Index and TRIN to the Market Valuation Context panel.

```python
# In the Market Valuation Context expander

# Change the layout to 5 columns
vc1, vc2, vc3, vc4, vc5 = st.columns(5)

# Add a new column for Fear & Greed
with vc5:
    fg = row.get("fear_greed_index")
    if fg is not None:
        fg_sig = "🔥 Extreme Greed" if fg > 75 else ("Greed" if fg > 55 else ("😐 Neutral" if fg > 45 else ("😨 Fear" if fg > 25 else "🥶 Extreme Fear")))
        st.metric("Fear & Greed Index", f"{fg:.0f}",
                  help="Composite index of market sentiment (0-100). Extreme fear can be a contrarian buy signal.")
        st.caption(fg_sig)
    else:
        st.metric("Fear & Greed Index", "N/A")

# You can replace one of the existing metrics or add TRIN similarly
# For example, replacing the Yield Curve with TRIN:
with vc4:
    trin = row.get("trin")
    if trin is not None:
        trin_sig = "🟢 Buying Pressure" if trin < 0.8 else ("🔴 Selling Pressure" if trin > 1.2 else "😐 Neutral")
        st.metric("TRIN Arms Index", f"{trin:.2f}",
                  help="Volume-weighted breadth. < 1.0 suggests buying pressure; > 1.0 suggests selling pressure.")
        st.caption(trin_sig)
    else:
        st.metric("TRIN Arms Index", "N/A")
```
