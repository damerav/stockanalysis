
# Kiro Prompt: Comprehensive Navigation & UI Redesign

## 1. High-Level Goal

Your primary mission is to **completely redesign the navigation system** of the `stockanalysis` Streamlit application. This includes the main sidebar menu and all in-page tab components. The new design must be modern, professional, consistent across all pages, and inspired by best-in-class financial platforms like TradingView and modern fintech applications.

## 2. Design Rationale & Core Principles

The current navigation uses default Streamlit components, which are functional but lack the polished, information-dense aesthetic of a dedicated financial analysis tool. Our research into platforms like TradingView, along with modern UX trends, points to several key principles that you must follow:

- **Maximize Content Real Estate:** The user's primary focus should be on charts and data. Navigation elements must be space-efficient.
- **Improve Information Scent:** Use clear, consistent icons to provide users with quick, scannable cues about the content of each page and tab.
- **Enhance Visual Hierarchy:** Guide the user's eye with deliberate use of color, spacing, and active states. Not all navigation items are equal; the currently active page/tab must be obvious.
- **Modernize Aesthetics:** Move away from default styles to a custom, professionally designed "pill" style for tabs and a cleaner, icon-driven sidebar.

## 3. Implementation Steps

### Step 3.1: Create a Centralized CSS File

Create a new file at `/home/ubuntu/stockanalysis/src/dashboard/style.css`. Paste the entire CSS block below into this file. This will be our single source of truth for all styling.

```css
/* --- HIDE STREAMLIT DEFAULTS --- */
/* Hide the default Streamlit hamburger menu */
#MainMenu {visibility: hidden;}
/* Hide the Streamlit footer */
footer {visibility: hidden;}
/* Hide the header that appears when running in Streamlit Cloud */
header {visibility: hidden;}

/* --- SIDEBAR STYLING --- */
/* Style for the main sidebar container */
[data-testid="stSidebar"] {
    background-color: #0f1116; /* A slightly darker shade for the sidebar */
}

/* Style for the individual sidebar navigation links */
[data-testid="stSidebarNav"] ul {
    padding-top: 1rem;
}

[data-testid="stSidebarNav"] ul a {
    display: flex;
    align-items: center;
    gap: 0.75rem; /* Space between icon and text */
    padding: 0.6rem 1rem;
    border-radius: 0.5rem; /* Rounded corners for the nav items */
    font-size: 0.95rem;
    font-weight: 500;
    transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out;
}

/* Style for the active/current page link in the sidebar */
[data-testid="stSidebarNav"] ul a[aria-current="page"] {
    background-color: rgba(0, 128, 255, 0.15); /* Use a subtle blue highlight */
    color: #ffffff;
    border-left: 3px solid #0080ff;
}

/* Hover effect for sidebar links */
[data-testid="stSidebarNav"] ul a:hover {
    background-color: rgba(255, 255, 255, 0.05);
}

/* --- PILL-STYLE TAB STYLING --- */
/* This targets the container for Streamlit's st.tabs */
[data-testid="stTabs"] {
    border: none;
    padding-top: 1rem;
}

/* Style for the tab bar (the container of the tab buttons) */
[data-testid="stTabs"] div[role="tablist"] {
    display: flex;
    gap: 10px; /* Space between tabs */
    border-bottom: none !important; /* Remove the default bottom border */
}

/* Style for each individual tab button */
[data-testid="stTabs"] button[role="tab"] {
    background-color: transparent;
    border: 1px solid #333;
    border-radius: 9999px; /* Creates the pill shape */
    padding: 8px 20px;
    color: #a0a0a0; /* Default text color for inactive tabs */
    font-weight: 500;
    transition: background-color 0.2s, color 0.2s, border-color 0.2s;
    display: flex;
    align-items: center;
    gap: 8px; /* Space between icon and text in tab */
}

/* Style for the selected/active tab button */
[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: #0080ff; /* Blue background for active tab */
    color: #ffffff; /* White text for active tab */
    border-color: #0080ff; /* Blue border for active tab */
}

/* Remove the orange line that appears on focus */
[data-testid="stTabs"] button:focus {
    box-shadow: none;
}
```

### Step 3.2: Load CSS and Redesign the Sidebar in `app.py`

Modify `/home/ubuntu/stockanalysis/src/dashboard/app.py`. You will inject the CSS and replace the `st.sidebar.radio` call with a modern `st.navigation` implementation.

```python
# At the top of app.py, add this function to load the CSS
import streamlit as st

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Inside the main part of your app, after st.set_page_config, call this function
load_css("src/dashboard/style.css")

# Find the existing sidebar navigation logic (st.sidebar.radio) and REPLACE it with this:

# --- User Info in Sidebar ---
user = get_user()

# --- Sidebar Navigation ---
st.sidebar.title("📊 Stock Analysis")
if user:
    st.sidebar.caption(f"👤 {user.get('name', user.get('email', ''))}")

# Define the pages with modern, consistent icons
pages = {
    "SPY Predictor": st.Page("src/dashboard/spy_predictor_app.py", title="SPY Predictor", icon="🔮"),
    "ES Strategy": st.Page("src/dashboard/es_strategy_app.py", title="ES Strategy", icon="📈"),
    "What-If Analysis": st.Page("src/dashboard/whatif_app.py", title="What-If Analysis", icon="🧪"),
    "Forecast": st.Page("src/dashboard/forecast_app.py", title="Forecast", icon="📊"),
    "Single-Stock Analysis": st.Page("src/dashboard/single_stock_app.py", title="Single-Stock", icon="🔍"),
    "Monitoring": st.Page("src/dashboard/monitoring.py", title="Monitoring", icon="🖥️"),
    "Grafana Dashboards": st.Page("src/dashboard/grafana_app.py", title="Grafana", icon="🔗"),
    "Admin": st.Page("src/dashboard/admin_app.py", title="Admin", icon="⚙️"),
}

pg = st.navigation(pages)

st.sidebar.divider()
mode_label = "☁️ Cloud" if IS_CLOUD else "🖥️ Local"
st.sidebar.caption(f"{mode_label} mode")
if user and user.get("email") != "anonymous":
    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        logout()
        st.rerun()

# Run the selected page
pg.run()

# Remove the old if/elif page routing block at the end of the file.
# st.navigation handles this automatically.
```

### Step 3.3: Update In-Page Tabs

Go through every file that uses `st.tabs`. You only need to change the text labels to include an icon at the beginning. The CSS you added in Step 3.1 will automatically handle the restyling.

**File: `/home/ubuntu/stockanalysis/src/dashboard/app.py`**
- **Find:** `tab_es, tab_spy = st.tabs(["ES Strategy", "SPY Predictor"])`
- **Replace with:** `tab_es, tab_spy = st.tabs(["📈 ES Strategy", "🔮 SPY Predictor"])`

- **Find:** `tab_status, tab_actions, tab_users, tab_db, tab_config, tab_logs = st.tabs(["System Status", "Actions", "👤 Users", "Database", "Configuration", "Logs"])`
- **Replace with:** `tab_status, tab_actions, tab_users, tab_db, tab_config, tab_logs = st.tabs(["ℹ️ System Status", "▶️ Actions", "👤 Users", "🗃️ Database", "📝 Configuration", "📜 Logs"])`

**File: `/home/ubuntu/stockanalysis/src/dashboard/monitoring.py`**
- **Find:** `src_tabs = st.tabs(["📰 Finnhub News", "📡 Yahoo RSS", "📡 CNBC RSS", "📡 MarketWatch RSS", "📊 FRED Macro", "💹 yfinance", "📅 Earnings", "🏛️ Fed Comms", "🗄️ All News (DB)"])`
- **Replace with:** `src_tabs = st.tabs(["📰 Finnhub", "📡 Yahoo", "📡 CNBC", "📡 MarketWatch", "📊 FRED", "💹 yfinance", "📅 Earnings", "🏛️ Fed Comms", "🗄️ All News"])`

- **Find:** `tabs = st.tabs(["📈 SPY Predictor", "📊 ES Strategy", "🖥️ System Health", "🤖 Confidence API", "⚙️ Pipeline Status", "📡 Data Sources"])`
- **Replace with:** `tabs = st.tabs(["📈 SPY", "📊 ES", "🖥️ Health", "🤖 API", "⚙️ Pipeline", "📡 Sources"])`

**File: `/home/ubuntu/stockanalysis/src/dashboard/whatif_app.py`**
- **Find:** `tab_es, tab_spy = st.tabs(["ES Strategy", "SPY Predictor"])`
- **Replace with:** `tab_es, tab_spy = st.tabs(["📈 ES Strategy", "🔮 SPY Predictor"])`

## 4. Acceptance Criteria

- [ ] The default Streamlit hamburger menu and footer are gone.
- [ ] The sidebar has a new, darker background color (`#0f1116`).
- [ ] All sidebar navigation items have a modern icon, a text label, and rounded corners.
- [ ] The currently active page in the sidebar is highlighted with a blue background and a left border.
- [ ] All in-page tabs across the entire application now appear as "pills".
- [ ] Inactive tabs have a greyish text color and a subtle border.
- [ ] The active tab has a solid blue background and white text.
- [ ] All tabs now have a relevant icon next to their text label.
- [ ] The application remains fully functional, and all pages are accessible.
- [ ] The old `st.sidebar.radio` and the `if/elif` page routing block in `app.py` have been completely removed and replaced by `st.navigation`.
