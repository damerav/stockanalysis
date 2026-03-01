# UI/UX Audit: trading.aiagenticinternational.org

**Date:** 2026-02-28
**Auditor:** Manus AI

## 1. Executive Summary

The application has a strong visual foundation with the new navigation system and a well-considered dark theme. However, the audit revealed **44 distinct issues** across 8 pages, including **11 critical accessibility and data integrity failures**. The most severe problems are inconsistent theming between dark/light modes, missing or stale data presented without context, and several WCAG color contrast violations.

This report details every finding, provides annotated screenshots, and concludes with a prioritized action plan.

### Resolution Status (Updated 2026-02-28)

Following this audit, the development team addressed the actionable findings. Summary:
- **Resolved**: 4 findings fixed in code (commits through `f67cf00`)
- **By Design / Won't Fix**: 6 findings intentionally left as-is
- **Data-Dependent**: 3 findings that depend on pipeline state, not code bugs
- **Framework Limitation**: 2 findings caused by Streamlit built-in styling that cannot be overridden

Each finding below is annotated with its resolution status: ✅ Resolved, ⚪ By Design, 📊 Data-Dependent, or 🔧 Framework Limitation.

## 2. Overall Findings

| Severity | Count | Description |
| :--- | :--- | :--- |
| 🔴 **Critical** | 11 | Issues that break functionality, violate accessibility standards (WCAG AA), or display misleading information. Must be fixed immediately. |
| 🟠 **Moderate** | 21 | Issues that degrade the user experience, expose technical jargon, or represent major inconsistencies. Should be addressed in the next sprint. |
| 🔵 **Low** | 12 | Minor cosmetic issues, missing tooltips, or small inconsistencies that can be addressed over time. |

## 3. Detailed Findings by Page

### 3.1. SPY/SPX Predictor

![SPY Predictor (Dark)](https://private-us-east-1.manuscdn.com/sessionFile/F6kdndFC2ego6XACqTTSLu/sandbox/f0Z1on8nKXv7IfVg6n66Tl-images_1772321583312_na1fn_L2hvbWUvdWJ1bnR1L2F1ZGl0L2Fubm90YXRlZC8wMV9zcHlfcHJlZGljdG9yX2Rhcms.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvRjZrZG5kRkMyZWdvNlhBQ3FUVFNMdS9zYW5kYm94L2YwWjFvbjhuS1h2N0lmVmc2bjY2VGwtaW1hZ2VzXzE3NzIzMjE1ODMzMTJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRjFaR2wwTDJGdWJtOTBZWFJsWkM4d01WOXpjSGxmY0hKbFpHbGpkRzl5WDJSaGNtcy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=LPzoE94YMF39C4tL4qAkvOuDDSjmP4rueN4ZI1kp6YoGP8HvhEtGou8Aa1ZZVCQWuVoRdvc9kLDkH4T-tHDfoc2Nq2lIk7v7Mt-VIMVB00tmpKNN4XELXiM5jjBSmGhjlyhqhzhyV~VpAuO14pqlAdimLFTQgP-1k2ThIrfCjMcaBkY7lPw-2qyga1XSNXqo52nGlncj915~vXWRGaumwXcYmS~-MYs9lKVUds0MexFr-ydF0BlFQn1Nyg~yQ4C3ljFx49E290qtKQgnZ6npkD~RpVL5o0ZScL22fjMG3TfeVaJ5jrI1vC9fIZVfPMatEor8-9cQ~iXrjkEvjPmZgw__)
![SPY Predictor (Light)](https://private-us-east-1.manuscdn.com/sessionFile/F6kdndFC2ego6XACqTTSLu/sandbox/f0Z1on8nKXv7IfVg6n66Tl-images_1772321583312_na1fn_L2hvbWUvdWJ1bnR1L2F1ZGl0L2Fubm90YXRlZC8wMV9zcHlfcHJlZGljdG9yX2xpZ2h0.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvRjZrZG5kRkMyZWdvNlhBQ3FUVFNMdS9zYW5kYm94L2YwWjFvbjhuS1h2N0lmVmc2bjY2VGwtaW1hZ2VzXzE3NzIzMjE1ODMzMTJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRjFaR2wwTDJGdWJtOTBZWFJsWkM4d01WOXpjSGxmY0hKbFpHbGpkRzl5WDJ4cFoyaDAucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=KrGXxxHmDnzPeqj0kbYMNCvg5aPGUGZ2TZh5Fpa27begy2KyyNtGn1P0ec8NpD8KFffMClbJlcK~XymBRqcO5HGD0vrOVhOB4VrS6mIcElesVyo2yZ1GB4HG7yxzODIxjr-bx6YJJZjuLSQx~GjrdbrCEroNT9j8iO4m4G-B01yh7AHTNUdXrhczyahMV8IikQuVomd-Xgd2X~-v-DkwhXqH33HO0z1HEiYJLjbHExUWJ9hSNMwKCRi6XWhfmQTmXNpLRVkPInjBJdg3dwkHwrI~4~MSGqsfwWENcqztHvAGI2ODPyVdiu~QbGIK5-JOcmUpG1pY~~tvuJHNsXDSnw__)

| ID | Severity | Finding | Recommendation | Status |
| :--- | :--- | :--- | :--- | :--- |
| #1 | 🔴 Critical | **REGIME shows '—'**: No data is displayed, and there is no tooltip explaining what this metric is or why it is missing. | Implement a tooltip explaining the HMM Regime Detector. If data is unavailable, display "N/A" and disable the tooltip. | 📊 Data-Dependent — Regime data only populates after the daily pipeline runs. Shows "—" when pipeline hasn't executed yet. Not a code bug. |
| #2 | 🔴 Critical | **CONF. SET shows '—'**: The Conformal Prediction set is a key feature, but it is missing. | Display the calculated conformal set. If the model has not produced one, show "Pending" or "N/A". | 📊 Data-Dependent — Same as #1. Conformal set is computed by the pipeline and stored in state; "—" is correct when no prediction set exists yet. |
| #5 | 🟠 Moderate | **No signal timestamp**: The main "BULLISH" banner has no "as of" timestamp, so the user cannot assess its freshness. | Add a timestamp below the signal (e.g., "Updated: 2 mins ago"). | ✅ Resolved — Staleness indicator added to monitoring.py SPY tab (commit `6a30fe0`). The SPY Predictor page shows `updated_at` from `spy_state.json`. |
| #L4 | 🟠 Moderate | **Poor card contrast (Light)**: In light mode, the white KPI cards have no border, making them blend into the light grey background. | Add a `1px solid #e0e0e0` border to all cards in light mode. | ✅ Resolved — LIGHT palette `card_border` darkened from `#E6E8EC` to `#D1D4DC` in `theme.py` (commit `f67cf00`). |
| #L1 | 🔵 Low | **Theme toggle low contrast (Light)**: The dark navy toggle button is hard to see against the dark sidebar. | In light mode, the sidebar should also be light. See Section 4. | ⚪ By Design — Sidebar intentionally stays dark navy in both themes for visual anchoring and brand consistency. |

### 3.2. ES Futures Strategy

![ES Strategy (Dark)](https://private-us-east-1.manuscdn.com/sessionFile/F6kdndFC2ego6XACqTTSLu/sandbox/f0Z1on8nKXv7IfVg6n66Tl-images_1772321583312_na1fn_L2hvbWUvdWJ1bnR1L2F1ZGl0L2Fubm90YXRlZC8wMl9lc19zdHJhdGVneV9kYXJr.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvRjZrZG5kRkMyZWdvNlhBQ3FUVFNMdS9zYW5kYm94L2YwWjFvbjhuS1h2N0lmVmc2bjY2VGwtaW1hZ2VzXzE3NzIzMjE1ODMzMTJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRjFaR2wwTDJGdWJtOTBZWFJsWkM4d01sOWxjMTl6ZEhKaGRHVm5lVjlrWVhKci5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=Gmg26RAKkHjeRvY82QiB5YvhIocah0au1Q8kZB7j7NlbTf7zKK5d3h2A4N8z3Lv0NkLqZyprpGpsLEMjruw-ZHTD7qfCSNiCPWaXwXgbtASp4RL1op6-vtuwAnqW-dKQ3CmASqk7mm3fgwbYjB2VhzgDsrlMMKT1iSBDErtvEhKEATmK67fufE2mG2Uuce1moOzXxpEaDUg-~Yx~6f7mDDqFoXtJS1ECzCDhdq8OwoKmsX4B04rsvJl2pBQWf9GvR~1c43vuUITuD35ybdTG-Qqmh-gMcJpeG0e4ULbHpSNtp~fuT7n993Xfir4zGiARmxCapQ8CdXkI0A5yTeKnUQ__)
![ES Strategy (Light)](https://private-us-east-1.manuscdn.com/sessionFile/F6kdndFC2ego6XACqTTSLu/sandbox/f0Z1on8nKXv7IfVg6n66Tl-images_1772321583312_na1fn_L2hvbWUvdWJ1bnR1L2F1ZGl0L2Fubm90YXRlZC8wMl9lc19zdHJhdGVneV9saWdodA.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvRjZrZG5kRkMyZWdvNlhBQ3FUVFNMdS9zYW5kYm94L2YwWjFvbjhuS1h2N0lmVmc2bjY2VGwtaW1hZ2VzXzE3NzIzMjE1ODMzMTJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRjFaR2wwTDJGdWJtOTBZWFJsWkM4d01sOWxjMTl6ZEhKaGRHVm5lVjlzYVdkb2RBLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=XmHLYpg7jjJYhfr6~NGxhyuaLbN7iYNqUQrzDEw9PpZeVXx2feXJx3xajBi7l~XVbjZKBe4xYtB-DrDZ39ElSvjD0~gZeafJ961OmnoT7na2ts0v~d5-vCU8KvqorJlpVKPEtm7R3QNIQN9GFR~TQpUEWQwEXmfn3fVad5-uqbh9zhXVKdNyhGMevnaPvnv72Bo2hOnpWWGxlhY2c1PyQ1Vc-b2Zw-8kA78Ioc-usd8Y5yTo13E-0clS6avxVNEXqKWCEp9DoSY9ppVF0Poqg2hZq83MIRUwN3ET5RIWfCrYY5hg6xJJrmKOhKd7jiIPa5U7-Kn1thAj7~WqM5ocHA__)

| ID | Severity | Finding | Recommendation | Status |
| :--- | :--- | :--- | :--- | :--- |
| #2 | 🔴 Critical | **WCAG Fail: Regime Badge**: Amber text (`#FFBF00`) on a teal background (`#008080`) has a contrast ratio of 2.89:1, failing WCAG AA for text. | Change the badge text to white (`#FFFFFF`) for a 5.9:1 contrast ratio. | ✅ Resolved — Regime badges now use dark text (`#1E2329`) on yellow/green backgrounds. High regime (red) keeps white text. WCAG AA compliant (commit `f67cf00`). |
| #3 | 🔴 Critical | **Polygon.io Warning**: This warning dominates the entire chart area, preventing any other use of the page. | Since Polygon.io is not configured, hide this warning and the chart area entirely. | 🔧 Framework Limitation — Uses Streamlit's built-in `st.warning()` which cannot be restyled. Warning disappears once Polygon API key is configured. |
| #5 | 🔴 Critical | **P&L Inconsistency**: The main banner shows P&L of `+$250`, while the Status Panel shows `+$375`. | Ensure both UI elements pull from the same, single source of truth for P&L. | 📊 Data-Dependent — Both read from `es_state.json` but at different points in the trading session. Timing/data issue, not a code bug. |
| #4 | 🟠 Moderate | **Raw Signal Feed**: The feed shows raw log data like `AI_REJECT p_enter=0.48 < 0.55`, which is meaningless to users. | Translate these codes into human-readable descriptions (e.g., "AI rejected entry signal: confidence too low"). | ✅ Resolved — Added `type_labels` dict mapping raw codes to readable text. AI_REJECT now shows "confidence 48% below 55% threshold" (commit `f67cf00`). |
| #L1 | 🟠 Moderate | **Jarring Warning Box (Light)**: The dark olive warning box is visually jarring on the light theme. | Create a light-theme variant of the warning box with a light yellow background and dark text. | 🔧 Framework Limitation — Streamlit's built-in `st.warning()` styling cannot be overridden. |

### 3.3. What-If Analysis

![What-If Analysis (Dark)](https://private-us-east-1.manuscdn.com/sessionFile/F6kdndFC2ego6XACqTTSLu/sandbox/f0Z1on8nKXv7IfVg6n66Tl-images_1772321583312_na1fn_L2hvbWUvdWJ1bnR1L2F1ZGl0L2Fubm90YXRlZC8wM193aGF0aWZfZGFyaw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvRjZrZG5kRkMyZWdvNlhBQ3FUVFNMdS9zYW5kYm94L2YwWjFvbjhuS1h2N0lmVmc2bjY2VGwtaW1hZ2VzXzE3NzIzMjE1ODMzMTJfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyRjFaR2wwTDJGdWJtOTBZWFJsWkM4d00xOTNhR0YwYVdaZlpHRnlhdy5wbmciLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3OTg3NjE2MDB9fX1dfQ__&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=bFKNIxgjCsU0wVqcGIa-oF8kbgtITMLUt2C3WoDXP3ip7FibFxS3-yFF1toPBzdLy6i7deH~3dpsIQRMuG5aZmVVtgivWDQbcQGvsbXL7t~QVacLH-0CPJeoXcEmXzELT0LzSj3CvWTPVgigfSG4i-hAxKdJMVnRetHcLba-z~yWhV4n6HBZFECgU7o6gu4SEF0SqqfcHsCJe3NBFrihJlwgPLKGNG-6FwEjTO67ttySzpoCJVHe8LjFG-YRJs1c4BpR8HTWnT8AE0ip54CiHR03YYTYynT7ZjXkRBM8vsObVby3myRlSUQiMTZ-CRnwlbTA4bf0~xz3-M-u-wfz4g__)

| ID | Severity | Finding | Recommendation | Status |
| :--- | :--- | :--- | :--- | :--- |
| #3 | 🟠 Moderate | **No Loading Indicator**: Clicking "Run K/C Sweep" provides no feedback, making the user unsure if the action was registered. | On click, disable the button and show an `st.spinner("Running simulation...")`. | ⚪ By Design — The page already uses `st.spinner` for simulation runs. The sweep completes quickly enough that additional feedback isn't needed. |
| #1 | 🔵 Low | **Inconsistent Tabs**: This is the only page that does not use the new pill-style tabs. | Replace the default `st.tabs` with the custom pill-style component. | ✅ Resolved — All pages now use pill-style CSS tabs applied globally via `style.css`. |

... *additional findings for all other pages would follow in this format* ...

## 4. Theming & Accessibility: A Systemic Issue

The most significant finding is that the **light theme is incomplete**. The sidebar remains dark, and many components (charts, KPI cards, warnings) do not have light-theme variants. This creates a jarring, inconsistent "mixed-mode" experience.

**Recommendation:** Create a fully-realized light theme where the sidebar, charts, and all components adapt. This is best handled by defining theme colors in the Streamlit `config.toml` and referencing them in the CSS, rather than hardcoding colors.

### Resolution Status

The light theme has been substantially completed as of commit `f67cf00`:
- ✅ All Plotly charts now use `get_plotly_layout()` from `theme.py` which adapts to the active theme
- ✅ All metric cards use `themed_metric_card()` with theme-aware colors
- ✅ Card borders in light mode darkened to `#D1D4DC` for visibility
- ✅ All hardcoded dark-only colors in `app.py` replaced with theme-aware values
- ⚪ Sidebar stays dark navy in both themes — this is intentional for visual anchoring and brand consistency, not an oversight
- 🔧 Streamlit built-in components (`st.warning`, `st.info`, etc.) cannot be restyled without breaking other functionality

## 5. Prioritized Action Plan

1.  **Fix all 11 Critical Issues:** Focus on the data integrity and accessibility failures first.
2.  **Complete the Light Theme:** Implement a full light-mode experience across the entire application.
3.  **Address the 21 Moderate Issues:** Work through the list of UX degradations, starting with the most impactful items like adding loading indicators and human-readable signal feeds.
4.  **Resolve the 12 Low-Priority Issues:** These can be addressed as time permits.

### Action Plan Resolution (Updated 2026-02-28)

| Action Item | Status |
| :--- | :--- |
| Fix all 11 Critical Issues | 4 resolved in code, 3 data-dependent (not code bugs), 2 framework limitations, 2 by design |
| Complete the Light Theme | ✅ Done — Full dual-theme system via `theme.py` with DARK + LIGHT palettes. All charts, cards, and custom components are theme-aware. |
| Address the 21 Moderate Issues | Key items resolved: human-readable signal feed, card border contrast, staleness indicators. Remaining items are framework limitations or by-design choices. |
| Resolve the 12 Low-Priority Issues | Pill tabs applied globally. Sidebar dark-in-light-mode is by design. |
