# Kiro Prompt: Enhanced Prediction with Institutional Flow — Historical Tracking

## Context

This platform has two independent signals for next-day SPY direction:

1. **Model Prediction** — an XGBoost ensemble trained on 214 features (macro, technicals, options, sentiment, microstructure). Stored daily in the `predictions` table. Evaluated against actuals in the `performance` table. Displayed as a hero banner on the SPY Predictor page.

2. **Institutional Flow** — real-time Polygon.io WebSocket data capturing large options sweeps and block trades on SPX. Stored as a rolling list of alerts in `spy_state.json` under the key `flow_alerts`. Currently displayed as a raw log in a collapsed expander on the SPY Predictor page. **Never stored historically. Never evaluated against actuals.**

The goal of this implementation is to:
- Compute a new standalone metric called **"Prediction + Institutional Flow"** (short name: `enhanced_prediction`) each day.
- Store it historically in the database alongside the model prediction.
- Evaluate it against actuals every day (same way the model prediction is evaluated).
- Display it as a **separate, clearly labelled signal** in the existing Prediction History chart — so the user can visually compare Model Prediction, Enhanced Prediction, and Actual outcome on the same chart over time.
- Display a standalone **Enhanced Prediction card** on the SPY Predictor page, positioned directly below the existing hero prediction banner.

---

## Part 1: Database Schema Changes

### File: `src/data/init_db.py`

**Step 1a — Add two new columns to the `predictions` table.**

In the `SCHEMA` string, find the `predictions` table definition:

```sql
CREATE TABLE IF NOT EXISTS predictions (
    date TEXT PRIMARY KEY,
    direction TEXT, confidence REAL,
    factors TEXT,
    report_text TEXT,
    predicted_at TEXT
);
```

Replace it with:

```sql
CREATE TABLE IF NOT EXISTS predictions (
    date TEXT PRIMARY KEY,
    direction TEXT, confidence REAL,
    factors TEXT,
    report_text TEXT,
    predicted_at TEXT,
    -- Enhanced Prediction: model + institutional flow fusion
    enhanced_direction TEXT,        -- 'BULLISH', 'BEARISH', 'NEUTRAL', 'CONFLICTED'
    enhanced_score REAL,            -- -100 to +100 continuous score
    flow_score REAL,                -- raw institutional flow component (-100 to +100)
    flow_alert_count INTEGER        -- number of flow alerts used in computation
);
```

**Step 1b — Add two new columns to the `performance` table.**

In the `SCHEMA` string, find the `performance` table definition:

```sql
CREATE TABLE IF NOT EXISTS performance (
    date TEXT PRIMARY KEY,
    predicted TEXT, actual TEXT,
    correct INTEGER,
    cumulative_accuracy REAL,
    confidence_tier TEXT,
    vix_regime TEXT,
    day_of_week INTEGER,
    event_proximity INTEGER
);
```

Replace it with:

```sql
CREATE TABLE IF NOT EXISTS performance (
    date TEXT PRIMARY KEY,
    predicted TEXT, actual TEXT,
    correct INTEGER,
    cumulative_accuracy REAL,
    confidence_tier TEXT,
    vix_regime TEXT,
    day_of_week INTEGER,
    event_proximity INTEGER,
    -- Enhanced prediction tracking
    enhanced_predicted TEXT,        -- enhanced_direction for this date
    enhanced_correct INTEGER,       -- 1 if enhanced_predicted matched actual, 0 otherwise
    enhanced_cumulative_accuracy REAL  -- running cumulative accuracy of enhanced signal
);
```

**Step 1c — Add schema migration for existing databases.**

In the `_migrate_schema(conn)` function, add the following migration block alongside the existing `ALTER TABLE` migrations:

```python
# Enhanced prediction columns on predictions table
for col, col_type in [
    ("enhanced_direction", "TEXT"),
    ("enhanced_score", "REAL"),
    ("flow_score", "REAL"),
    ("flow_alert_count", "INTEGER"),
]:
    try:
        conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")
    except Exception:
        pass  # Column already exists

# Enhanced prediction tracking columns on performance table
for col, col_type in [
    ("enhanced_predicted", "TEXT"),
    ("enhanced_correct", "INTEGER"),
    ("enhanced_cumulative_accuracy", "REAL"),
]:
    try:
        conn.execute(f"ALTER TABLE performance ADD COLUMN {col} {col_type}")
    except Exception:
        pass  # Column already exists

conn.commit()
```

---

## Part 2: Enhanced Prediction Computation Logic

### File: `src/realtime/dashboard_bridge.py`

**Step 2a — Update `write_spy_state` to accept and store the enhanced prediction.**

Find the current `write_spy_state` function signature:

```python
def write_spy_state(prediction: dict = None, indicators: dict = None,
                    flow_alerts: list = None):
```

Replace the entire function with:

```python
def write_spy_state(prediction: dict = None, indicators: dict = None,
                    flow_alerts: list = None, enhanced_prediction: dict = None):
    """Write SPY predictor state for dashboard consumption."""
    state = {
        "updated_at": datetime.now().isoformat(),
        "prediction": prediction or {},
        "indicators": indicators or {},
        "flow_alerts": flow_alerts or [],
        "enhanced_prediction": enhanced_prediction or {},
    }
    _atomic_write(os.path.join(DATA_DIR, "spy_state.json"), state)
```

**Step 2b — Add the `compute_enhanced_prediction` function.**

Add the following new function to `dashboard_bridge.py` after `write_spy_state`:

```python
def compute_enhanced_prediction(prediction: dict, flow_alerts: list) -> dict:
    """Compute the Enhanced Prediction signal by fusing model prediction with institutional flow.

    The enhanced prediction is a SEPARATE metric from the model prediction.
    It answers: "When smart money (institutional flow) and the model agree, how strong is the signal?"

    Scoring:
    - Model Score = confidence × direction_sign
      where direction_sign = +1 for BULLISH variants, -1 for BEARISH variants, 0 for NEUTRAL
    - Flow Score = (CALL notional - PUT notional) / (CALL notional + PUT notional) × 100
      computed from the last 20 flow alerts; 0.0 if no alerts present
    - Enhanced Score = (Model Score × 0.65) + (Flow Score × 0.35)
      Weights: model carries 65% (higher historical reliability), flow carries 35%

    Direction labels:
    - enhanced_score > +40  → 'BULLISH'
    - enhanced_score > +10  → 'LEAN BULLISH'
    - enhanced_score < -40  → 'BEARISH'
    - enhanced_score < -10  → 'LEAN BEARISH'
    - abs(enhanced_score) <= 10 → 'NEUTRAL'
    - model and flow point opposite directions with |score| > 10 → 'CONFLICTED'

    Returns dict with keys: enhanced_direction, enhanced_score, flow_score,
    flow_alert_count, model_score, alignment.
    """
    if not prediction:
        return {
            "enhanced_direction": "NEUTRAL",
            "enhanced_score": 0.0,
            "flow_score": 0.0,
            "flow_alert_count": 0,
            "model_score": 0.0,
            "alignment": "NO_DATA",
        }

    # --- Model score ---
    scale_label = prediction.get("scale_label", "NEUTRAL")
    confidence = float(prediction.get("confidence", 0))
    if "BULLISH" in scale_label:
        direction_sign = 1.0
    elif "BEARISH" in scale_label:
        direction_sign = -1.0
    else:
        direction_sign = 0.0
    model_score = confidence * direction_sign  # range: -100 to +100

    # --- Flow score ---
    recent_alerts = (flow_alerts or [])[-20:]  # last 20 alerts
    call_notional = sum(
        float(a.get("notional", 0))
        for a in recent_alerts
        if a.get("direction", "").upper() == "CALL"
    )
    put_notional = sum(
        float(a.get("notional", 0))
        for a in recent_alerts
        if a.get("direction", "").upper() == "PUT"
    )
    total_notional = call_notional + put_notional
    if total_notional > 0:
        flow_score = ((call_notional - put_notional) / total_notional) * 100.0
    else:
        flow_score = 0.0  # no flow data — flow component is neutral

    # --- Enhanced score (weighted fusion) ---
    enhanced_score = (model_score * 0.65) + (flow_score * 0.35)

    # --- Direction label ---
    model_bull = direction_sign > 0
    model_bear = direction_sign < 0
    flow_bull = flow_score > 10
    flow_bear = flow_score < -10
    signals_conflict = (model_bull and flow_bear) or (model_bear and flow_bull)

    if signals_conflict and abs(enhanced_score) > 10:
        enhanced_direction = "CONFLICTED"
        alignment = "CONFLICTED"
    elif enhanced_score > 40:
        enhanced_direction = "BULLISH"
        alignment = "ALIGNED_BULLISH"
    elif enhanced_score > 10:
        enhanced_direction = "LEAN BULLISH"
        alignment = "LEAN_BULLISH"
    elif enhanced_score < -40:
        enhanced_direction = "BEARISH"
        alignment = "ALIGNED_BEARISH"
    elif enhanced_score < -10:
        enhanced_direction = "LEAN BEARISH"
        alignment = "LEAN_BEARISH"
    else:
        enhanced_direction = "NEUTRAL"
        alignment = "NEUTRAL"

    return {
        "enhanced_direction": enhanced_direction,
        "enhanced_score": round(enhanced_score, 2),
        "flow_score": round(flow_score, 2),
        "flow_alert_count": len(recent_alerts),
        "model_score": round(model_score, 2),
        "alignment": alignment,
    }
```

---

## Part 3: Pipeline Integration — Compute and Store Daily

### File: `src/pipeline/daily_run.py`

**Step 3a — Import the new function.**

Find the existing import at the top of the file:

```python
from src.realtime.dashboard_bridge import write_spy_state
```

Replace with:

```python
from src.realtime.dashboard_bridge import write_spy_state, compute_enhanced_prediction
```

**Step 3b — Compute and store the enhanced prediction in `_step11_predict`.**

Find the section in `_step11_predict` that ends with the `write_spy_state` call (around line 882):

```python
        write_spy_state(prediction=prediction, indicators=indicators)
        logger.info(f"Prediction: {prediction['scale_label']} "
                    f"({prediction['confidence']:.0f}%)"
                    f"{' [LOW CONVICTION]' if prediction.get('is_low_conviction') else ''}")
        return prediction
```

Replace it with:

```python
        # Compute enhanced prediction (model + institutional flow fusion)
        current_state = {}
        try:
            from src.realtime.dashboard_bridge import read_state
            current_state = read_state("spy_state.json")
        except Exception:
            pass
        flow_alerts = current_state.get("flow_alerts", [])
        enhanced = compute_enhanced_prediction(prediction, flow_alerts)

        # Store enhanced prediction columns in predictions table
        if self.router:
            self.router.write_analytics(
                """UPDATE predictions SET
                   enhanced_direction = ?,
                   enhanced_score = ?,
                   flow_score = ?,
                   flow_alert_count = ?
                   WHERE date = ?""",
                (enhanced["enhanced_direction"], enhanced["enhanced_score"],
                 enhanced["flow_score"], enhanced["flow_alert_count"],
                 self.today),
            )
        else:
            self._db_execute(
                """UPDATE predictions SET
                   enhanced_direction = ?,
                   enhanced_score = ?,
                   flow_score = ?,
                   flow_alert_count = ?
                   WHERE date = ?""",
                (enhanced["enhanced_direction"], enhanced["enhanced_score"],
                 enhanced["flow_score"], enhanced["flow_alert_count"],
                 self.today),
            )

        # Write full state including enhanced prediction
        write_spy_state(
            prediction=prediction,
            indicators=indicators,
            flow_alerts=flow_alerts,
            enhanced_prediction=enhanced,
        )
        logger.info(
            f"Prediction: {prediction['scale_label']} ({prediction['confidence']:.0f}%) | "
            f"Enhanced: {enhanced['enhanced_direction']} (score={enhanced['enhanced_score']:+.1f}, "
            f"flow={enhanced['flow_score']:+.1f}, alerts={enhanced['flow_alert_count']})"
            f"{' [LOW CONVICTION]' if prediction.get('is_low_conviction') else ''}"
        )
        return prediction
```

---

## Part 4: Evaluate Enhanced Prediction Against Actuals

### File: `src/model/trainer.py`

**Step 4a — Update `evaluate_past_prediction` to also evaluate the enhanced signal.**

Find the section in `evaluate_past_prediction` that reads the prediction from the database:

```python
    if router:
        pred_df = router.query(
            "SELECT direction, confidence FROM predictions WHERE date = ?", (date_str,)
        )
        if pred_df.empty:
            return None
        predicted = pred_df.iloc[0]["direction"]
        pred_confidence = pred_df.iloc[0]["confidence"] or 0
```

Replace with:

```python
    if router:
        pred_df = router.query(
            "SELECT direction, confidence, enhanced_direction FROM predictions WHERE date = ?",
            (date_str,)
        )
        if pred_df.empty:
            return None
        predicted = pred_df.iloc[0]["direction"]
        pred_confidence = pred_df.iloc[0]["confidence"] or 0
        enhanced_predicted = pred_df.iloc[0]["enhanced_direction"] if "enhanced_direction" in pred_df.columns else None
```

And in the `else` (raw connection) branch:

```python
    else:
        row = conn_or_router.execute(
            "SELECT direction, confidence FROM predictions WHERE date = ?", (date_str,)
        ).fetchone()
        if not row:
            return None
        predicted = row[0]
        pred_confidence = row[1] or 0
```

Replace with:

```python
    else:
        row = conn_or_router.execute(
            "SELECT direction, confidence, enhanced_direction FROM predictions WHERE date = ?",
            (date_str,)
        ).fetchone()
        if not row:
            return None
        predicted = row[0]
        pred_confidence = row[1] or 0
        enhanced_predicted = row[2] if len(row) > 2 else None
```

**Step 4b — Compute enhanced accuracy and write it to the performance table.**

Find the section that computes `cum_accuracy` and writes to the `performance` table:

```python
    if router:
        perf_df = router.query("SELECT COUNT(*) as cnt, SUM(correct) as s FROM performance")
        total = (int(perf_df.iloc[0]["cnt"]) if not perf_df.empty else 0) + 1
        correct_total = (int(perf_df.iloc[0]["s"] or 0) if not perf_df.empty else 0) + correct
    else:
        perf_rows = conn_or_router.execute("SELECT COUNT(*), SUM(correct) FROM performance").fetchone()
        total = (perf_rows[0] or 0) + 1
        correct_total = (perf_rows[1] or 0) + correct
    cum_accuracy = correct_total / total
```

Replace with:

```python
    if router:
        perf_df = router.query(
            "SELECT COUNT(*) as cnt, SUM(correct) as s, "
            "SUM(CASE WHEN enhanced_correct IS NOT NULL THEN enhanced_correct ELSE 0 END) as es, "
            "COUNT(CASE WHEN enhanced_correct IS NOT NULL THEN 1 END) as ec "
            "FROM performance"
        )
        total = (int(perf_df.iloc[0]["cnt"]) if not perf_df.empty else 0) + 1
        correct_total = (int(perf_df.iloc[0]["s"] or 0) if not perf_df.empty else 0) + correct
        enhanced_total = (int(perf_df.iloc[0]["ec"] or 0) if not perf_df.empty else 0)
        enhanced_correct_total = (int(perf_df.iloc[0]["es"] or 0) if not perf_df.empty else 0)
    else:
        perf_rows = conn_or_router.execute(
            "SELECT COUNT(*), SUM(correct), "
            "SUM(CASE WHEN enhanced_correct IS NOT NULL THEN enhanced_correct ELSE 0 END), "
            "COUNT(CASE WHEN enhanced_correct IS NOT NULL THEN 1 END) "
            "FROM performance"
        ).fetchone()
        total = (perf_rows[0] or 0) + 1
        correct_total = (perf_rows[1] or 0) + correct
        enhanced_total = (perf_rows[3] or 0)
        enhanced_correct_total = (perf_rows[2] or 0)
    cum_accuracy = correct_total / total

    # Enhanced accuracy computation
    enhanced_correct_val = None
    enhanced_cum_accuracy = None
    if enhanced_predicted and enhanced_predicted not in ("CONFLICTED", "NEUTRAL"):
        enhanced_correct_val = 1 if (
            ("BULLISH" in enhanced_predicted and actual == "BULLISH") or
            ("BEARISH" in enhanced_predicted and actual == "BEARISH")
        ) else 0
        enhanced_total += 1
        enhanced_correct_total += enhanced_correct_val
        enhanced_cum_accuracy = enhanced_correct_total / enhanced_total
    elif enhanced_predicted in ("NEUTRAL",):
        # NEUTRAL enhanced predictions: count them but only correct if actual is NEUTRAL
        enhanced_correct_val = 1 if actual == "NEUTRAL" else 0
        enhanced_total += 1
        enhanced_correct_total += enhanced_correct_val
        enhanced_cum_accuracy = enhanced_correct_total / enhanced_total
    # CONFLICTED enhanced predictions are NOT counted in accuracy — they are abstentions
```

Then find the `INSERT OR REPLACE INTO performance` statements and update both (router and raw connection) to include the new columns:

```python
    if router:
        router.execute(
            """INSERT OR REPLACE INTO performance
               (date, predicted, actual, correct, cumulative_accuracy,
                confidence_tier, vix_regime, day_of_week, event_proximity,
                enhanced_predicted, enhanced_correct, enhanced_cumulative_accuracy)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date_str, predicted, actual, correct, cum_accuracy,
             conf_tier, vix_regime, dow, event_prox,
             enhanced_predicted, enhanced_correct_val, enhanced_cum_accuracy)
        )
    else:
        conn_or_router.execute(
            """INSERT OR REPLACE INTO performance
               (date, predicted, actual, correct, cumulative_accuracy,
                confidence_tier, vix_regime, day_of_week, event_proximity,
                enhanced_predicted, enhanced_correct, enhanced_cumulative_accuracy)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date_str, predicted, actual, correct, cum_accuracy,
             conf_tier, vix_regime, dow, event_prox,
             enhanced_predicted, enhanced_correct_val, enhanced_cum_accuracy)
        )
        conn_or_router.commit()
```

Also update the `return` dict at the end of `evaluate_past_prediction` to include the enhanced fields:

```python
    return {
        "date": date_str, "predicted": predicted, "actual": actual,
        "correct": bool(correct), "cumulative_accuracy": round(cum_accuracy, 3),
        "confidence_tier": conf_tier, "vix_regime": vix_regime,
        "enhanced_predicted": enhanced_predicted,
        "enhanced_correct": bool(enhanced_correct_val) if enhanced_correct_val is not None else None,
        "enhanced_cumulative_accuracy": round(enhanced_cum_accuracy, 3) if enhanced_cum_accuracy else None,
    }
```

---

## Part 5: Dashboard Changes

### File: `src/dashboard/app.py`

#### 5a — Add the Enhanced Prediction card below the hero banner

Find the section in `page_spy()` that ends the hero prediction card block:

```python
    else:
        st.info("Waiting for prediction data...")
```

Insert the following block immediately after it (before the `# --- P3: Earnings + Fed + Options` section):

```python
    # --- Enhanced Prediction card (model + institutional flow) ---
    enhanced = state.get("enhanced_prediction", {})
    enh_direction = enhanced.get("enhanced_direction", "")
    enh_score = enhanced.get("enhanced_score", None)
    enh_flow_score = enhanced.get("flow_score", None)
    enh_model_score = enhanced.get("model_score", None)
    enh_alert_count = enhanced.get("flow_alert_count", 0)
    enh_alignment = enhanced.get("alignment", "")

    enh_color_map = {
        "BULLISH": c["green"], "LEAN BULLISH": c["green"],
        "NEUTRAL": c["yellow"],
        "LEAN BEARISH": c["red"], "BEARISH": c["red"],
        "CONFLICTED": c["yellow"],
    }
    enh_color = enh_color_map.get(enh_direction, c["text_secondary"])

    if enh_direction:
        alignment_badge = {
            "ALIGNED_BULLISH": ("✅ ALIGNED BULLISH", c["green"]),
            "ALIGNED_BEARISH": ("✅ ALIGNED BEARISH", c["red"]),
            "LEAN_BULLISH": ("↗ LEAN BULLISH", c["green"]),
            "LEAN_BEARISH": ("↘ LEAN BEARISH", c["red"]),
            "CONFLICTED": ("⚠️ CONFLICTED", c["yellow"]),
            "NEUTRAL": ("◆ NEUTRAL", c["yellow"]),
        }.get(enh_alignment, ("—", c["text_secondary"]))

        st.markdown(
            f"""<div style="border:2px solid {enh_color}44; border-radius:12px;
                padding:16px 20px; margin-bottom:16px;
                background: linear-gradient(135deg, {enh_color}11 0%, {enh_color}22 100%);">
                <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
                    <div>
                        <div style="font-size:0.72rem; color:{c['text_secondary']}; text-transform:uppercase;
                                    letter-spacing:0.12em; font-weight:600; margin-bottom:4px;">
                            PREDICTION + INSTITUTIONAL FLOW
                        </div>
                        <div style="font-size:1.5rem; font-weight:800; color:{enh_color};">
                            {enh_direction}
                        </div>
                        <div style="font-size:0.8rem; color:{alignment_badge[1]}; margin-top:2px; font-weight:600;">
                            {alignment_badge[0]}
                        </div>
                    </div>
                    <div style="display:flex; gap:20px; flex-wrap:wrap;">
                        <div style="text-align:center;">
                            <div style="font-size:1.4rem; font-weight:800; color:{enh_color};">
                                {f'{enh_score:+.0f}' if enh_score is not None else '—'}
                            </div>
                            <div style="font-size:0.68rem; color:{c['text_secondary']}; text-transform:uppercase;">
                                Enhanced Score
                            </div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:1.4rem; font-weight:700; color:{c['text']};">
                                {f'{enh_model_score:+.0f}' if enh_model_score is not None else '—'}
                            </div>
                            <div style="font-size:0.68rem; color:{c['text_secondary']}; text-transform:uppercase;">
                                Model (65%)
                            </div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:1.4rem; font-weight:700; color:{c['text']};">
                                {f'{enh_flow_score:+.0f}' if enh_flow_score is not None else '—'}
                            </div>
                            <div style="font-size:0.68rem; color:{c['text_secondary']}; text-transform:uppercase;">
                                Flow (35%) · {enh_alert_count} alerts
                            </div>
                        </div>
                    </div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Enhanced prediction not yet available (requires live flow data)")
```

#### 5b — Update the Prediction History chart to show both signals and actuals

Find the `load_prediction_history` function:

```python
@st.cache_data(ttl=300, show_spinner=False)
def load_prediction_history(days: int = 30) -> pd.DataFrame:
```

Replace the entire function with:

```python
@st.cache_data(ttl=300, show_spinner=False)
def load_prediction_history(days: int = 30) -> pd.DataFrame:
    """Load prediction history joined with performance actuals and enhanced prediction."""
    try:
        router = get_router(_load_config())
        df = router.query(
            """SELECT
                p.date,
                p.direction,
                p.confidence,
                p.enhanced_direction,
                p.enhanced_score,
                p.flow_score,
                p.flow_alert_count,
                perf.actual,
                perf.correct,
                perf.enhanced_correct,
                perf.enhanced_cumulative_accuracy,
                perf.cumulative_accuracy
               FROM predictions p
               LEFT JOIN performance perf ON p.date = perf.date
               ORDER BY p.date DESC LIMIT ?""",
            (days,),
        )
        return df.sort_values("date") if not df.empty else df
    except Exception:
        return pd.DataFrame()
```

#### 5c — Update the Prediction History chart rendering

Find the chart rendering block inside `page_spy()` that starts with:

```python
        hist_df = load_prediction_history(30)
        if not hist_df.empty:
            colors = hist_df["direction"].map({
```

Replace the entire chart block (from `hist_df = load_prediction_history(30)` through `st.plotly_chart(fig, use_container_width=True)`) with:

```python
        hist_df = load_prediction_history(30)
        if not hist_df.empty:
            # Color bars by model prediction direction
            bar_colors = hist_df["direction"].map({
                "BULLISH": c["green"], "STRONG_BULLISH": c["green"],
                "BEARISH": c["red"], "STRONG_BEARISH": c["red"],
                "NEUTRAL": c["text_secondary"],
            }).fillna(c["text_muted"])

            fig = go.Figure()

            # Trace 1: Model prediction confidence bars (background)
            fig.add_trace(go.Bar(
                name="Model Prediction",
                x=hist_df["date"],
                y=hist_df["confidence"],
                marker_color=bar_colors.tolist(),
                opacity=0.55,
                text=hist_df["direction"].str.replace("_", " "),
                textposition="outside",
                textfont=dict(color=c["text"], size=8),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Model: %{text}<br>"
                    "Confidence: %{y:.0f}%<extra></extra>"
                ),
            ))

            # Trace 2: Enhanced prediction score line (secondary axis)
            if "enhanced_score" in hist_df.columns and hist_df["enhanced_score"].notna().any():
                enh_line_colors = hist_df["enhanced_direction"].map({
                    "BULLISH": c["green"], "LEAN BULLISH": c["green"],
                    "BEARISH": c["red"], "LEAN BEARISH": c["red"],
                    "NEUTRAL": c["yellow"], "CONFLICTED": c["yellow"],
                }).fillna(c["text_secondary"])

                fig.add_trace(go.Scatter(
                    name="Enhanced Prediction Score",
                    x=hist_df["date"],
                    y=hist_df["enhanced_score"],
                    mode="lines+markers",
                    line=dict(color="#00bcd4", width=2, dash="dot"),
                    marker=dict(
                        size=7,
                        color=enh_line_colors.tolist(),
                        line=dict(color="#00bcd4", width=1),
                    ),
                    yaxis="y2",
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Enhanced: %{customdata}<br>"
                        "Score: %{y:+.1f}<extra></extra>"
                    ),
                    customdata=hist_df["enhanced_direction"].fillna("N/A"),
                ))

            # Trace 3: Actual outcome markers (triangles)
            if "actual" in hist_df.columns and hist_df["actual"].notna().any():
                actual_colors = hist_df["actual"].map({
                    "BULLISH": c["green"],
                    "BEARISH": c["red"],
                    "NEUTRAL": c["yellow"],
                }).fillna(c["text_secondary"])
                actual_symbols = hist_df["actual"].map({
                    "BULLISH": "triangle-up",
                    "BEARISH": "triangle-down",
                    "NEUTRAL": "diamond",
                }).fillna("circle")

                fig.add_trace(go.Scatter(
                    name="Actual Outcome",
                    x=hist_df["date"],
                    y=[95] * len(hist_df),  # fixed position at top of chart
                    mode="markers",
                    marker=dict(
                        symbol=actual_symbols.tolist(),
                        size=10,
                        color=actual_colors.tolist(),
                        line=dict(color=c["text"], width=1),
                    ),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Actual: %{customdata}<extra></extra>"
                    ),
                    customdata=hist_df["actual"].fillna("N/A"),
                ))

            _hist_bg = "rgba(0,0,0,0)" if is_dark() else c["surface"]
            fig.update_layout(
                height=260,
                margin=dict(l=10, r=50, t=5, b=25),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=_hist_bg,
                font=dict(color=c["text_secondary"], size=10),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=9),
                ),
                xaxis=dict(
                    gridcolor=c["grid"], tickfont=dict(size=9),
                    tickformat="%b %d", dtick="D1",
                ),
                yaxis=dict(
                    title="Confidence %", range=[0, 105],
                    gridcolor=c["grid"],
                ),
                yaxis2=dict(
                    title="Enhanced Score",
                    overlaying="y", side="right",
                    range=[-110, 110],
                    showgrid=False,
                    zeroline=True, zerolinecolor=c["grid"],
                    tickfont=dict(size=9),
                ),
                barmode="overlay",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Chart legend explanation
            st.markdown(
                f'<p style="color:{c["text_secondary"]}; font-size:0.72rem; margin-top:-8px;">'
                f'<span style="color:{c["green"]};">▌</span> Bullish bar · '
                f'<span style="color:{c["red"]};">▌</span> Bearish bar = Model prediction confidence · '
                f'<span style="color:#00bcd4;">---</span> Enhanced score (right axis) · '
                f'▲▼◆ = Actual outcome</p>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No prediction history yet")
```

#### 5d — Update the Accuracy Tracking expander to show both accuracies side by side

Find the `📊 Accuracy Tracking` expander block:

```python
        perf_df = load_performance()
        if not perf_df.empty:
            with st.expander("📊 Accuracy Tracking", expanded=False):
                latest_acc = perf_df.iloc[0]["cumulative_accuracy"] if len(perf_df) > 0 else 0
                st.metric("Cumulative Accuracy", f"{latest_acc:.1%}")
```

Replace the first two lines inside the expander with:

```python
        perf_df = load_performance()
        if not perf_df.empty:
            with st.expander("📊 Accuracy Tracking", expanded=False):
                latest_acc = perf_df.iloc[0]["cumulative_accuracy"] if len(perf_df) > 0 else 0
                latest_enh_acc = None
                if "enhanced_cumulative_accuracy" in perf_df.columns:
                    enh_series = perf_df["enhanced_cumulative_accuracy"].dropna()
                    if not enh_series.empty:
                        latest_enh_acc = float(enh_series.iloc[0])

                acc_col1, acc_col2 = st.columns(2)
                with acc_col1:
                    st.metric(
                        "Model Accuracy",
                        f"{latest_acc:.1%}",
                        help="Cumulative directional accuracy of the XGBoost model prediction.",
                    )
                with acc_col2:
                    st.metric(
                        "Enhanced Accuracy",
                        f"{latest_enh_acc:.1%}" if latest_enh_acc is not None else "N/A",
                        delta=f"{latest_enh_acc - latest_acc:+.1%}" if latest_enh_acc is not None else None,
                        delta_color="normal",
                        help="Cumulative accuracy of the Enhanced Prediction (model + institutional flow). "
                             "CONFLICTED days are excluded from this count.",
                    )
```

---

## Part 6: Market Overview Page Update

### File: `src/dashboard/market_overview_app.py`

**Step 6a — Add the Enhanced Prediction metric to the Key Indicators section.**

Find the section in `market_overview_app.py` that renders the Key Indicators row. It will contain `st.columns` calls for VIX, Fear & Greed, TRIN, etc. Locate the `spy_state.json` read at the top of the function:

```python
    state = load_spy_state()
    prediction = state.get("prediction", {})
```

Add the following line immediately after:

```python
    enhanced = state.get("enhanced_prediction", {})
```

Then find the first row of Key Indicator metrics (the row containing VIX, 10Y Yield, DXY, Gold) and add the Enhanced Prediction as an additional metric in the second row (the row containing Fear & Greed, TRIN, Buffett Indicator, Shiller CAPE). Change that row from 4 columns to 5 columns:

```python
    # Row 2: Breadth / Valuation / Enhanced
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
```

And add the Enhanced Prediction metric in `r2c5`:

```python
    with r2c5:
        enh_dir = enhanced.get("enhanced_direction", "")
        enh_score = enhanced.get("enhanced_score")
        enh_color_map = {
            "BULLISH": "normal", "LEAN BULLISH": "normal",
            "BEARISH": "inverse", "LEAN BEARISH": "inverse",
            "NEUTRAL": "off", "CONFLICTED": "off",
        }
        st.metric(
            "Enhanced Signal",
            enh_dir if enh_dir else "N/A",
            delta=f"Score: {enh_score:+.0f}" if enh_score is not None else None,
            delta_color=enh_color_map.get(enh_dir, "off"),
            help="Prediction + Institutional Flow fusion signal. "
                 "Combines model confidence (65%) with live options flow direction (35%).",
        )
```

---

## Part 7: Validation Checklist

After implementation, verify the following:

| # | Check | Expected Result |
| :--- | :--- | :--- |
| 1 | Run `python -m src.data.init_db` | No errors; `predictions` and `performance` tables have new columns |
| 2 | Run `python -c "from src.realtime.dashboard_bridge import compute_enhanced_prediction; print(compute_enhanced_prediction({'scale_label': 'BULLISH', 'confidence': 72}, [{'direction': 'CALL', 'notional': 500000}, {'direction': 'PUT', 'notional': 100000}]))"` | Returns dict with `enhanced_direction='BULLISH'`, positive `enhanced_score`, positive `flow_score` |
| 3 | Run `python -c "from src.realtime.dashboard_bridge import compute_enhanced_prediction; print(compute_enhanced_prediction({'scale_label': 'BULLISH', 'confidence': 72}, [{'direction': 'PUT', 'notional': 900000}]))"` | Returns `enhanced_direction='CONFLICTED'` or `'LEAN BEARISH'` due to opposing flow |
| 4 | Run `python -c "from src.realtime.dashboard_bridge import compute_enhanced_prediction; print(compute_enhanced_prediction({'scale_label': 'BULLISH', 'confidence': 72}, []))"` | Returns `flow_score=0.0`, `flow_alert_count=0`; enhanced score equals model score × 0.65 |
| 5 | Run the daily pipeline manually: `python -m src.pipeline.daily_run` | Log shows both `Prediction: ...` and `Enhanced: ...` lines; `predictions` table row for today has non-null `enhanced_direction` |
| 6 | Open SPY Predictor page | New "PREDICTION + INSTITUTIONAL FLOW" card appears below the hero banner |
| 7 | Open SPY Predictor page | Prediction History chart shows 3 traces: bars (model), dotted line (enhanced score), triangle markers (actuals) |
| 8 | Open SPY Predictor page → Accuracy Tracking expander | Two side-by-side metrics: "Model Accuracy" and "Enhanced Accuracy" |
| 9 | Open Market Overview page | "Enhanced Signal" metric card visible in the second row of Key Indicators |
| 10 | After several days of data accumulate | `performance` table rows have non-null `enhanced_predicted`, `enhanced_correct`, `enhanced_cumulative_accuracy` |

---

## Design Notes for Kiro

- The `compute_enhanced_prediction` function in `dashboard_bridge.py` is the single source of truth for the computation logic. It is called both by the daily pipeline (for historical storage) and by the WebSocket streamer's `on_flow_alert_handler` (for live intraday updates to `spy_state.json`).
- When the WebSocket streamer is live (after the Polygon fix is deployed), the `on_flow_alert_handler` should also call `compute_enhanced_prediction` and update `spy_state.json` with the refreshed enhanced prediction on every new flow alert — giving the dashboard near-real-time updates to the enhanced signal throughout the trading day.
- The `CONFLICTED` direction is intentionally excluded from the accuracy denominator. A CONFLICTED signal is an abstention — it tells the user "do not trade this signal today." Counting abstentions as wrong would penalize the system for correctly identifying uncertainty.
- The `flow_score = 0.0` case (no alerts) is handled gracefully: the enhanced score simply equals `model_score × 0.65`, which will always be lower magnitude than the raw model confidence. This is correct behavior — without flow confirmation, the enhanced signal is more conservative than the model alone.
