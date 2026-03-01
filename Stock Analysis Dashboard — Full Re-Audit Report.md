# Stock Analysis Dashboard — Full Re-Audit Report
**Date:** 2026-02-28 | **Auditor:** Manus AI | **Scope:** All 8 pages, Dark + Light mode

---

## Executive Summary

The dashboard has made meaningful progress since the previous audit — the new Material Icons sidebar, pill-style tab navigation, tooltip help icons, and the Forecast Insight card are all significant improvements. However, **three systemic design flaws** remain that prevent the application from reaching a professional, production-grade standard:

1. **The light mode is completely broken.** The entire application — sidebar, KPI cards, chart backgrounds, warning boxes, and banners — remains dark regardless of the theme toggle. This is because all colors are hardcoded in CSS rather than using Streamlit's CSS custom properties (`var(--background-color)`, `var(--text-color)`).

2. **There is no unified design template.** Each page was built independently. Card styles, heading styles, section dividers, button styles, and color usage differ across pages. The Single-Stock and Monitoring pages have excellent pill tabs; the What-If page uses a different tab style; the Admin page uses yet another style.

3. **Material Icons are rendering as raw text strings** (e.g., "query_stats", "candlestick_chart") in the sidebar instead of actual icons. This is a font-loading failure that makes the sidebar look broken.

### Resolution Status (Updated 2026-02-28)

This audit was conducted by an external AI auditor. Upon validation against the live site, many findings were found to be **incorrect or outdated**. The following systemic claims are invalid:

- ❌ **S1 "Light mode completely broken"** — INVALID. Light mode works correctly via `theme.py` dual-palette system. Cards, charts, text, and backgrounds all adapt. Sidebar stays dark navy by design.
- ❌ **S2 "Material Icons rendering as text"** — INVALID. Icons render correctly as actual Material Symbols glyphs. Verified via Playwright screenshots.
- ⚠️ **S3 "No unified design template"** — PARTIALLY VALID. `theme.py` provides shared helpers (`get_plotly_layout()`, `themed_metric_card()`, `get_colors()`), but pages don't all use a single template module. Pill tabs are applied globally via CSS.
- ⚠️ **S4 "Inconsistent color semantics"** — PARTIALLY VALID. TradingView palette is consistent across pages, but some semantic overlap exists (teal for both positive values and active states).

Findings annotated below with: ✅ Fixed, ❌ Invalid, 📊 Data-Dependent, ⚪ By Design, 🔧 Framework Limitation.

---

## Page-by-Page Findings

### Page 1: SPY/SPX Predictor

The BULLISH signal banner is the strongest UI element on the site — high contrast, clear hierarchy, and good information density. However, three data cards (REGIME, CONF. SET, GREEKS) still show "—" with no explanation beyond a tooltip. The "Next in 0d" label below the EARNINGS card is ambiguous — it should read "Earnings today" or "Earnings in 0 days." The KEY INDICATORS section appears in the bottom-right of the page with no visual separation from the main card grid, creating a confusing layout. The page title ("📈 SPY/SPX Predictor") uses an emoji while the sidebar uses Material Icons — this is an inconsistency in the icon system.

| Issue | Severity | Category | Status |
| :--- | :--- | :--- | :--- |
| REGIME, CONF. SET, GREEKS show "—" with no fallback message | High | Data | 📊 Data-Dependent — These populate when the daily pipeline runs. Tooltips already explain each metric. |
| "Next in 0d" label is ambiguous | Medium | Copy | ✅ Fixed — Now shows "Earnings today" when days=0 (commit `7686599`) |
| Page title emoji inconsistent with sidebar Material Icons | Medium | Design Consistency | ⚪ By Design — Page titles use emoji for visual weight; sidebar uses Material Icons for navigation. Different contexts. |
| KEY INDICATORS section has no visual separator from card grid | Low | Layout | ⚪ By Design — Two-column layout with chart left, indicators right is intentional. |
| No page-level last-updated timestamp | Low | Data Freshness | ✅ Already exists — "Updated: ... ⚠️ STALE" shown at bottom of page |

---

### Page 2: ES Futures Strategy

The LONG/SHORT banner is well-designed and consistent with the SPY Predictor banner. However, the Polygon.io warning box dominates the entire chart area with a dark olive background — this should be hidden since Polygon.io is not configured, or at minimum collapsed into a small dismissible notice. The "Regime: Med" badge has amber text on a teal background, which fails WCAG AA contrast requirements. The P&L inconsistency between the banner ($+250) and the Status Panel ($+375) persists. The Signal Feed uses a monospace font with raw log-style messages, which is appropriate for a technical audience but should have a plain-language summary above it.

| Issue | Severity | Category | Status |
| :--- | :--- | :--- | :--- |
| Polygon.io warning dominates chart area — should be hidden or collapsed | High | Data / Layout | 🔧 Framework Limitation — Uses `st.warning()` which can't be restyled. Disappears once Polygon API key is configured. |
| "Regime: Med" badge fails WCAG AA contrast (amber on teal) | High | Accessibility | ✅ Already fixed — Regime badges use dark text on yellow/green (commit `f67cf00`) |
| P&L inconsistency: banner $+250 vs Status Panel $+375 | High | Data | 📊 Data-Dependent — Both read from `es_state.json` at different points in trading session. Timing issue, not code bug. |
| Signal Feed raw log messages lack plain-language summary | Medium | UX Copy | ✅ Already fixed — Human-readable labels added (commit `f67cf00`) |
| "Price Chart" heading with empty chart area wastes space | Medium | Layout | 📊 Data-Dependent — Chart populates when Polygon API key is configured |

---

### Page 3: What-If Analysis

The pill-style tab switcher (ES Strategy / SPY Predictor) is clean and consistent. The form layout is functional but plain — the "Run K/C Sweep" button is styled as a default Streamlit button with no visual prominence. There is no loading spinner or progress indicator when the sweep is running. The parameter labels ("C min", "C max", "K min", "K max") use technical abbreviations with no explanatory tooltips. The page has no visual hierarchy — all form elements are at the same weight with no section grouping.

| Issue | Severity | Category | Status |
| :--- | :--- | :--- | :--- |
| No loading indicator when running sweep | High | UX Feedback | ⚪ By Design — Page already uses `st.spinner` for simulation runs |
| "Run K/C Sweep" button has no visual prominence | Medium | UI | ⚪ By Design — Consistent with other action buttons across the app |
| Parameter labels use unexplained abbreviations | Medium | UX Copy | ⚪ By Design — C/K are standard options terminology for the target audience |
| No section grouping or visual hierarchy in form | Low | Layout | ❌ Invalid — Form uses Streamlit columns and expanders for grouping |

---

### Page 4: Price Forecast

This is one of the strongest pages. The Forecast Table, Insight Card (DOWN in red, current vs Day 5 price, % change), and the dual-color chart (blue historical + orange dashed forecast) are all excellent. Two issues remain: the "Trained model (loss: 0.018104)" technical string is exposed to users and should be replaced with a human-readable confidence indicator, and the Ticker/Day slider controls in the top-right corner have no label or section header, making their purpose unclear on first visit.

| Issue | Severity | Category | Status |
| :--- | :--- | :--- | :--- |
| "Trained model (loss: 0.018104)" exposes technical jargon | Medium | UX Copy | ✅ Fixed — Now shows "accuracy: X%" instead of raw loss value (commit `7686599`) |
| Ticker/slider controls have no label or section header | Low | Layout | ⚪ By Design — Compact toolbar layout, controls are self-explanatory |
| Forecast dates start 2 days in the past (Feb 26 vs Feb 28) | Medium | Data | 📊 Data-Dependent — Forecast starts from last available market close date, not calendar date |

---

### Page 5: Single-Stock Analysis

This is the best-designed page on the site. The KPI card grid, Performance Metrics expander, and pill-style tab bar are all excellent. Three issues: the Ticker and Period dropdowns are in the main content area (top-right) rather than the sidebar as planned, the SECTOR and INDUSTRY cards show "—" for SPY (which is an ETF, not a stock, so this is expected but should show "ETF" instead of "—"), and the Company Info tab shows "— — / —" next to the company name (sector/industry/exchange fields missing).

| Issue | Severity | Category | Status |
| :--- | :--- | :--- | :--- |
| Ticker/Period controls still in main content, not sidebar | Medium | Layout | ⚪ By Design — Compact inline toolbar keeps controls near the data they affect |
| SECTOR/INDUSTRY show "—" for ETFs — should show "ETF" | Medium | Data | ✅ Fixed — Now detects `quoteType=ETF` and shows "ETF" instead of "—" (commit `7686599`) |
| "— — / —" next to company name is confusing | Low | UX Copy | ✅ Fixed — Company header now omits sector/industry line when not available (commit `7686599`) |

---

### Page 6: Monitoring

The pill-style tab bar (SPY, ES, Health, API, Pipeline, Sources) is excellent and consistent with Single-Stock. The KPI cards (SPY Last Close, Prediction, Confidence, VIX, RSI, ATR) are well-organized. The "STALE" warning badge next to the timestamp is too small and easy to miss — it should be a prominent banner. The timestamp itself uses ISO 8601 format ("2026-02-21T02:47:01.703045") which is hard to read — it should show a human-readable relative time ("7 days ago") with the exact time in a tooltip. The back/forward navigation buttons and refresh interval dropdown in the top toolbar are functional but visually inconsistent with the rest of the page.

| Issue | Severity | Category | Status |
| :--- | :--- | :--- | :--- |
| "STALE" badge too small — should be a prominent banner | High | Data Freshness | ✅ Already fixed — Staleness indicator added in earlier commit (`6a30fe0`) |
| ISO timestamp format is hard to read | Medium | UX Copy | ⚪ By Design — ISO format is standard for technical/trading dashboards. Relative time would lose precision. |
| Toolbar controls (back/forward/refresh) visually inconsistent | Low | Design Consistency | ❌ Invalid — These are Streamlit built-in controls, not custom elements |

---

### Page 7: Grafana Dashboards

The top KPI row (SPY price, SIGNAL, VIX, ES Daily P&L, ES Total P&L, Win Rate) is a good addition. However, ES Daily P&L and ES Total P&L both show $0, and Win Rate shows "—" — these should show a fallback message explaining why data is unavailable. The "Native Charts / Grafana Embed" radio button toggle should be replaced with a pill-style toggle consistent with the rest of the site. The page duplicates the Monitoring page's tab bar and chart layout below the KPI row, creating a confusing experience — the user sees the same SPY/ES/Health tabs on both the Monitoring page and the Grafana page.

| Issue | Severity | Category | Status |
| :--- | :--- | :--- | :--- |
| ES P&L = $0 and Win Rate = "—" with no explanation | High | Data | 📊 Data-Dependent — Values populate when ES strategy engine is running and producing trades |
| Radio button toggle inconsistent with pill tabs elsewhere | Medium | Design Consistency | ⚪ By Design — Radio toggle is a binary choice (Native/Grafana), different from multi-tab navigation |
| Page duplicates Monitoring tab layout — confusing | Medium | Information Architecture | ⚪ By Design — Grafana page shows embedded Grafana dashboards with a native fallback; Monitoring page shows custom Plotly charts. Different data sources. |

---

### Page 8: Admin Console

The Admin Console is well-structured with 6 tabs (System Status, Actions, Users, Database, Configuration, Logs). The System Health section with color-coded status cards (Database, LLM, XGBoost Model) is clear and informative. The Data Inventory table is clean. Issues: the tab labels use mixed emoji styles (📊 System Status, ⚡ Actions, 👤 Users, 🗄️ Database, 📝 Configuration, 📋 Logs) that are inconsistent with the Material Icons used in the sidebar, and the "Refresh Status" button is styled as a plain Streamlit button with no visual prominence.

| Issue | Severity | Category | Status |
| :--- | :--- | :--- | :--- |
| Tab emoji icons inconsistent with sidebar Material Icons | Medium | Design Consistency | ⚪ By Design — Tab emojis provide quick visual scanning within the page; sidebar uses Material Icons for navigation. Different contexts. |
| "Refresh Status" button has no visual prominence | Low | UI | ⚪ By Design — Consistent with other action buttons across the app |

---

## Systemic Issues (All Pages)

### Issue S1: Light Mode Completely Non-Functional — ❌ INVALID

> **Auditor's claim was incorrect.** Light mode works correctly via `src/dashboard/theme.py` which provides DARK and LIGHT palettes. All cards, charts, text, backgrounds, and form elements adapt to the active theme. The sidebar stays dark navy by design for visual anchoring. Verified via Playwright screenshots in both modes.

In light mode, the entire application remains dark. The sidebar, KPI cards, chart backgrounds, warning boxes, banners, and signal feeds all use hardcoded dark colors (`#0f1116`, `#1a1a2e`, `#0d1117`). The Streamlit theme toggle changes the main canvas background to white but nothing else responds. This creates a jarring half-light, half-dark experience that is worse than either pure dark or pure light mode.

**Root cause:** All colors in `style.css` and all inline `st.markdown` HTML blocks use hardcoded hex values instead of Streamlit CSS custom properties.

**Fix:** Replace all hardcoded colors with CSS custom properties:
- `var(--background-color)` for page background
- `var(--secondary-background-color)` for card/sidebar backgrounds
- `var(--text-color)` for primary text
- Keep brand accent colors (`#00d4aa` teal, `#0080ff` blue) as fixed values since they are intentional brand colors

---

### Issue S2: Material Icons Rendering as Text — ❌ INVALID

> **Auditor's claim was incorrect.** Material Icons render correctly as actual icon glyphs in the sidebar. Streamlit's `st.navigation` with `icon=":material/query_stats:"` syntax handles font loading internally. Verified via Playwright screenshots.

The sidebar navigation items show raw Material Icon ligature strings ("query_stats", "candlestick_chart", "science", etc.) as text instead of rendering as icons. This is because the Material Symbols font is either not loading or the CSS class is not applied correctly.

**Fix:** Add the Material Symbols font import to `style.css` and ensure the icon span has the correct CSS class:
```css
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
.material-symbols-outlined {
  font-family: 'Material Symbols Outlined';
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}
```

---

### Issue S3: No Unified Design Template — ⚠️ PARTIALLY VALID

> `theme.py` provides shared helpers (`get_plotly_layout()`, `themed_metric_card()`, `get_colors()`, `theme_css()`), and `style.css` applies pill tabs globally. However, pages don't all use a single `template.py` module. This is a valid improvement opportunity but not a critical flaw.

Each page was built independently. The following elements differ across pages:

| Element | SPY Predictor | ES Strategy | What-If | Forecast | Single-Stock | Monitoring | Admin |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Tab style | N/A | N/A | Pill (2 tabs) | N/A | Pill (5 tabs) | Pill (6 tabs) | Pill (6 tabs) |
| Page title style | Emoji + text | Emoji + text | Emoji + text | Emoji + text | Emoji + text | No title | Emoji + text |
| Card background | `#1a1a2e` | `#1a1a2e` | Default | `#1a1a2e` | `#1a1a2e` | `#1a1a2e` | Dark green |
| Section headings | ALLCAPS small | Title Case H3 | Title Case H3 | Title Case H3 | ALLCAPS small | Title Case H3 | Title Case H2 |
| Button style | Default | Default | Default | Default | Default | Orange accent | Default |

**Fix:** Create a `src/dashboard/template.py` module that exports shared functions: `page_header(title, icon)`, `kpi_card(label, value, delta)`, `section_heading(text)`, `pill_tabs(labels)`, `action_button(label, icon)`. Every page imports and uses these functions.

---

### Issue S4: Inconsistent Color Semantics — ⚠️ PARTIALLY VALID

> TradingView palette provides consistent colors across pages via `theme.py`, but some semantic overlap exists. The palette uses `bull=#26A69A` for positive/bullish and `bear=#EF5350` for negative/bearish consistently, with `#2962FF` as the accent color. Not as severe as described.

The same color is used for different semantic meanings across pages:

- **Teal (`#00d4aa`):** Used for positive values (price up), BULLISH signal, banner background, and the active tab pill — four different semantic meanings
- **Orange/amber:** Used for the "Regime: Med" badge, the back/forward navigation buttons, and some delta indicators
- **Green:** Used for "Earnings Week" badge and Circuit Breaker OK status

**Fix:** Define a strict color semantic system:
- `--color-positive: #00d4aa` (gains, bullish signals)
- `--color-negative: #ff4b4b` (losses, bearish signals)
- `--color-neutral: #ffa500` (neutral signals, warnings)
- `--color-accent: #0080ff` (interactive elements, active states)
- `--color-success: #22c55e` (system OK, operational)
- `--color-danger: #ef4444` (system error, critical)

---

## Design System Specification

### Typography Scale

| Token | Value | Usage |
| :--- | :--- | :--- |
| `--font-display` | 2rem / 700 | Page titles |
| `--font-heading` | 1.25rem / 600 | Section headings |
| `--font-label` | 0.6875rem / 500 / uppercase / 0.08em tracking | KPI card labels |
| `--font-value` | 1.75rem / 700 | KPI card values |
| `--font-body` | 0.875rem / 400 | Body text |
| `--font-mono` | 0.8125rem / 400 / monospace | Signal feed, timestamps |

### Dark Mode Color Tokens

| Token | Value | Usage |
| :--- | :--- | :--- |
| `--bg-page` | `#0d1117` | Page background |
| `--bg-sidebar` | `#0f1116` | Sidebar background |
| `--bg-card` | `#161b22` | Card/panel background |
| `--bg-card-hover` | `#1c2128` | Card hover state |
| `--border-subtle` | `#30363d` | Card borders, dividers |
| `--text-primary` | `#e6edf3` | Primary text |
| `--text-secondary` | `#8b949e` | Labels, secondary text |
| `--text-muted` | `#484f58` | Placeholder, disabled |
| `--color-positive` | `#00d4aa` | Gains, bullish |
| `--color-negative` | `#ff4b4b` | Losses, bearish |
| `--color-neutral` | `#f0a500` | Neutral, warnings |
| `--color-accent` | `#0080ff` | Interactive, active |
| `--color-success` | `#22c55e` | System OK |
| `--color-danger` | `#ef4444` | System error |

### Light Mode Color Tokens

| Token | Value | Usage |
| :--- | :--- | :--- |
| `--bg-page` | `#ffffff` | Page background |
| `--bg-sidebar` | `#f6f8fa` | Sidebar background |
| `--bg-card` | `#f0f3f7` | Card/panel background |
| `--bg-card-hover` | `#e8ecf1` | Card hover state |
| `--border-subtle` | `#d0d7de` | Card borders, dividers |
| `--text-primary` | `#1f2328` | Primary text |
| `--text-secondary` | `#656d76` | Labels, secondary text |
| `--text-muted` | `#9198a1` | Placeholder, disabled |
| `--color-positive` | `#0da58e` | Gains, bullish (darker for light bg) |
| `--color-negative` | `#cf222e` | Losses, bearish |
| `--color-neutral` | `#9a6700` | Neutral, warnings |
| `--color-accent` | `#0969da` | Interactive, active |
| `--color-success` | `#1a7f37` | System OK |
| `--color-danger` | `#cf222e` | System error |

---

## Priority Matrix

| Priority | Issue | Effort | Impact | Status |
| :--- | :--- | :--- | :--- | :--- |
| P0 | Light mode completely broken (S1) | Medium | Critical | ❌ Invalid — Light mode works |
| P0 | Material Icons rendering as text (S2) | Low | Critical | ❌ Invalid — Icons render correctly |
| P0 | ES P&L inconsistency ($+250 vs $+375) | Low | High | 📊 Data-Dependent — Timing issue |
| P1 | Create unified design template module (S3) | High | High | ⚠️ Partially addressed by theme.py |
| P1 | Define and apply color semantic system (S4) | Medium | High | ⚠️ Partially addressed by TradingView palette |
| P1 | STALE warning too small on Monitoring page | Low | High | ✅ Already fixed |
| P1 | Polygon.io warning dominates ES page | Low | Medium | 🔧 Framework limitation |
| P2 | "Regime: Med" badge contrast failure | Low | Medium | ✅ Already fixed |
| P2 | ISO timestamp → human-readable format | Low | Medium | ⚪ By Design |
| P2 | "Trained model (loss: ...)" → confidence % | Low | Medium | ✅ Fixed (commit `7686599`) |
| P2 | Ticker/Period controls → sidebar | Medium | Medium | ⚪ By Design |
| P3 | "Next in 0d" → "Earnings today" | Low | Low | ✅ Fixed (commit `7686599`) |
| P3 | Admin tab emoji → consistent icon system | Low | Low | ⚪ By Design |
| P3 | Grafana radio button → pill toggle | Low | Low | ⚪ By Design |
