
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

Modify `/home/ubuntu/stockanalysis/src/dashboard/app.py`. You will inject the CSS and replace the `st.navigation` call with a custom loop to build the new sidebar.

```python
# At the top of app.py, add this function to load the CSS
import streamlit as st

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Inside the main part of your app, after st.set_page_config, call this function
load_css("src/dashboard/style.css")

# Find the existing `page = st.navigation(...)` block and REPLACE it with this:

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

# Render the custom sidebar using st.page_link
with st.sidebar:
    st.title("Stock Analysis")
    st.write("--- Administrator")
    st.divider()

    for page_key, page_value in pages.items():
        st.page_link(page_value, label=page_value.title, icon=page_value.icon)

    st.divider()
    st.info("Local mode") # Or whatever status you want to show
    if st.button("Sign Out"):
        st.query_params.clear()
        st.rerun()

# After the sidebar, determine which page to run
# This part requires you to manage the page state yourself.
# A simple way is to use session state.
if 'current_page' not in st.session_state:
    st.session_state.current_page = "SPY Predictor"

# You will need to update the st.session_state.current_page when a link is clicked.
# Streamlit's new st.page_link handles this navigation state for you, so you just need to call the page.
# The logic to run the selected page will depend on your exact setup, but st.page_link is the key.
# Find the part of your code that runs the selected page and ensure it's compatible with the new structure.
# The new `st.navigation` object is what you need to use.
pg = st.navigation(pages)
pg.run()

```

**Note:** The above snippet assumes you can replace the old navigation logic. You may need to adapt it slightly to fit your exact authentication and page-running logic, but the core idea is to use `st.page_link` inside a `with st.sidebar` block.

### Step 3.3: Update In-Page Tabs

Go through every file that uses `st.tabs`. You only need to change the text labels to include an icon at the beginning. The CSS you added in Step 3.1 will automatically handle the restyling.

**Example:** In `/home/ubuntu/stockanalysis/src/dashboard/single_stock_app.py`:

*   **Find this:**
    `tab_info, tab_raw, tab_chart, tab_ai, tab_news = st.tabs(["Company Info", "Raw Data", "Technical Chart", "AI Analysis", "News & Sentiment"])`

*   **Change it to this:**
    `tab_chart, tab_info, tab_raw, tab_ai, tab_news = st.tabs(["📈 Technical Chart", "ℹ️ Company Info", "🗃️ Raw Data", "🤖 AI Analysis", "📰 News & Sentiment"])`

Apply this pattern to **all** instances of `st.tabs` in the project (e.g., in `monitoring.py` as well).

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
