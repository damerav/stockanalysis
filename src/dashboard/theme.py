"""Shared theme system — dark/light mode for the Stock Analysis Platform.

Usage in any dashboard page:
    from src.dashboard.theme import get_theme, get_colors, get_plotly_layout, metric_card, badge_html, theme_css
"""

import os
import streamlit as st

# ══════════════════════════════════════════════════════════════════════
# THEME PALETTES
# ══════════════════════════════════════════════════════════════════════

DARK = {
    # Semantic
    "green": "#26A69A",
    "red": "#EF5350",
    "yellow": "#FFAB40",
    "blue": "#2962FF",
    "cyan": "#00BCD4",
    "orange": "#FF7043",
    "purple": "#AB47BC",
    # Layout
    "bg": "#131722",
    "surface": "#1E222D",
    "card": "rgba(30,34,45,0.6)",
    "card_bg": "#1E222D",          # backward-compat alias for surface
    "card_border": "rgba(255,255,255,0.05)",
    "card_hover": "rgba(41,98,255,0.3)",
    "border": "#2A2E39",
    # Text
    "text": "#D1D4DC",
    "text_secondary": "#787B86",
    "text_muted": "#363A45",
    "text_heading": "#D1D4DC",
    "white": "#D1D4DC",
    # Plotly
    "grid": "#1C1F2E",
    "zeroline": "#2A2E39",
    # Components
    "tab_bg": "rgba(30,34,45,0.5)",
    "tab_active": "#2962FF",
    "tab_text": "#787B86",
    "tab_hover": "rgba(255,255,255,0.04)",
    "input_bg": "#1E222D",
    "input_border": "#2A2E39",
    "form_bg": "rgba(30,34,45,0.8)",
    "btn_bg": "#2A2E39",
    "btn_border": "#4A4E59",
    "btn_text": "#D1D4DC",
    "btn_hover_bg": "#363A45",
    "btn_hover_border": "#5A5E69",
    "expander_bg": "rgba(30,34,45,0.4)",
    "scrollbar": "#2A2E39",
    "scrollbar_hover": "#363A45",
    "popover_bg": "#1E222D",
    "popover_hover": "#2A2E39",
    "df_border": "#2A2E39",
    "backdrop": "blur(12px)",
    # Sidebar
    "sidebar_bg": "linear-gradient(180deg, #0C0E14 0%, #0F1118 100%)",
    "sidebar_border": "#1C1F2E",
    "sidebar_text": "#D1D4DC",
    "sidebar_text_muted": "#787B86",
    "sidebar_btn_bg": "#1E222D",
    "sidebar_btn_border": "#363A45",
    "sidebar_btn_hover_bg": "#2A2E39",
    "sidebar_btn_hover_border": "#5A5E69",
    "sidebar_nav_hover": "rgba(255, 255, 255, 0.03)",
    "sidebar_nav_active_bg": "rgba(41, 98, 255, 0.1)",
    "sidebar_divider": "#1C1F2E",
}

LIGHT = {
    # Semantic — slightly deeper for readability on white
    "green": "#0ECB81",
    "red": "#F6465D",
    "yellow": "#F0B90B",
    "blue": "#2962FF",
    "cyan": "#0097A7",
    "orange": "#E65100",
    "purple": "#7B1FA2",
    # Layout
    "bg": "#F0F2F5",
    "surface": "#FFFFFF",
    "card": "#FFFFFF",
    "card_bg": "#FFFFFF",          # backward-compat alias for surface
    "card_border": "#D1D4DC",
    "card_hover": "rgba(41,98,255,0.15)",
    "border": "#E6E8EC",
    # Text
    "text": "#1E2329",
    "text_secondary": "#707A8A",
    "text_muted": "#B7BDC6",
    "text_heading": "#1E2329",
    "white": "#1E2329",
    # Plotly
    "grid": "#E6E8EC",
    "zeroline": "#B7BDC6",
    # Components
    "tab_bg": "#E6E8EC",
    "tab_active": "#2962FF",
    "tab_text": "#707A8A",
    "tab_hover": "rgba(0,0,0,0.04)",
    "input_bg": "#FFFFFF",
    "input_border": "#E6E8EC",
    "form_bg": "#FFFFFF",
    "btn_bg": "#FFFFFF",
    "btn_border": "#B7BDC6",
    "btn_text": "#1E2329",
    "btn_hover_bg": "#F0F2F5",
    "btn_hover_border": "#707A8A",
    "expander_bg": "#FFFFFF",
    "scrollbar": "#D1D4DC",
    "scrollbar_hover": "#B7BDC6",
    "popover_bg": "#FFFFFF",
    "popover_hover": "#F0F2F5",
    "df_border": "#E6E8EC",
    "backdrop": "none",
    # Sidebar
    "sidebar_bg": "linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%)",
    "sidebar_border": "#E6E8EC",
    "sidebar_text": "#1E2329",
    "sidebar_text_muted": "#707A8A",
    "sidebar_btn_bg": "#F0F2F5",
    "sidebar_btn_border": "#D1D4DC",
    "sidebar_btn_hover_bg": "#E6E8EC",
    "sidebar_btn_hover_border": "#B7BDC6",
    "sidebar_nav_hover": "rgba(0, 0, 0, 0.04)",
    "sidebar_nav_active_bg": "rgba(41, 98, 255, 0.08)",
    "sidebar_divider": "#E6E8EC",
}


# ══════════════════════════════════════════════════════════════════════
# THEME STATE
# ══════════════════════════════════════════════════════════════════════

def get_theme() -> str:
    """Return current theme name: 'dark' or 'light'."""
    return st.session_state.get("app_theme", "dark")


# ── Config.toml sync ────────────────────────────────────────────────
# Streamlit reads config.toml at startup AND on each rerun for theme
# variables.  By rewriting it before st.rerun(), the native theme
# engine renders widgets (buttons, inputs, selects) with the correct
# base colors — no CSS !important wars needed.

_TOML_TEMPLATE = """[theme]
base = "{base}"
primaryColor = "#2962FF"
backgroundColor = "{bg}"
secondaryBackgroundColor = "{surface}"
textColor = "{text}"
font = "sans serif"
"""

def _sync_config_toml(theme_name: str):
    """Rewrite .streamlit/config.toml to match the active theme.
    
    Only writes when content actually changes to avoid triggering
    Streamlit's file-watcher 'Rerun' banner on every page load.
    """
    palette = DARK if theme_name == "dark" else LIGHT
    content = _TOML_TEMPLATE.format(
        base=theme_name,
        bg=palette["bg"],
        surface=palette["surface"],
        text=palette["text"],
    )
    toml_dir = os.path.join(os.path.dirname(__file__), "..", "..", ".streamlit")
    toml_path = os.path.join(toml_dir, "config.toml")
    try:
        # Only write if content differs — avoids file-watcher churn
        existing = ""
        if os.path.exists(toml_path):
            with open(toml_path) as f:
                existing = f.read()
        if existing.strip() == content.strip():
            return
        os.makedirs(toml_dir, exist_ok=True)
        with open(toml_path, "w") as f:
            f.write(content)
    except OSError:
        pass  # read-only filesystem — CSS fallback still works


def set_theme(name: str):
    """Set theme to 'dark' or 'light' and update config.toml so Streamlit's
    native theme engine matches our palette."""
    st.session_state["app_theme"] = name
    _sync_config_toml(name)




def is_dark() -> bool:
    return get_theme() == "dark"


def get_colors() -> dict:
    """Return the active color palette dict."""
    return DARK if is_dark() else LIGHT


def render_theme_toggle():
    """Render a compact theme toggle in the sidebar (safe against duplicate calls)."""
    current = get_theme()
    icon = "🌙" if current == "dark" else "☀️"
    label = "Dark" if current == "dark" else "Light"
    try:
        if st.sidebar.button(f"{icon} {label}", key="theme_toggle", use_container_width=True):
            set_theme("light" if current == "dark" else "dark")
            st.rerun()
    except Exception:
        pass  # Already rendered in this script run (duplicate import path)


# ══════════════════════════════════════════════════════════════════════
# PLOTLY LAYOUT
# ══════════════════════════════════════════════════════════════════════

def get_plotly_layout() -> dict:
    """Return a Plotly layout dict matching the active theme."""
    c = get_colors()
    return dict(
        template="plotly_dark" if is_dark() else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)" if is_dark() else c["surface"],
        font=dict(color=c["text"], size=12),
        margin=dict(l=40, r=10, t=36, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=c["grid"], showgrid=True, zerolinecolor=c["zeroline"]),
        yaxis=dict(gridcolor=c["grid"], showgrid=True, zerolinecolor=c["zeroline"]),
    )


def get_title_font() -> dict:
    c = get_colors()
    return dict(color=c["text"], size=14)


# ══════════════════════════════════════════════════════════════════════
# HTML COMPONENTS
# ══════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, color: str = "white", sub: str = "") -> str:
    """Render a themed metric card as HTML."""
    c = get_colors()
    val_color = c.get(color, color)
    sub_html = f'<div style="color:{c["text_secondary"]}; font-size:0.65em; margin-top:1px;">{sub}</div>' if sub else ""
    shadow = "box-shadow:0 1px 3px rgba(0,0,0,0.08);" if not is_dark() else ""
    border = f"1px solid {c['card_border']}"
    bg = c["card"]
    blur = "backdrop-filter:blur(12px);" if is_dark() else ""
    return (
        f'<div style="background:{bg}; {blur} '
        f'border:{border}; '
        f'border-radius:6px; padding:8px 10px; text-align:center; {shadow} '
        f'transition: border-color 0.2s ease;">'
        f'<div style="color:{c["text_secondary"]}; font-size:0.6em; text-transform:uppercase; '
        f'letter-spacing:0.05em; margin-bottom:2px;">{label}</div>'
        f'<div style="color:{val_color}; font-size:1.1em; font-weight:600; '
        f'font-variant-numeric:tabular-nums;">{value}</div>'
        f'{sub_html}</div>'
    )


def badge_html(label: str, online: bool) -> str:
    """Render a themed status badge as HTML."""
    c = get_colors()
    color = c["green"] if online else c["red"]
    text = "ONLINE" if online else "OFFLINE"
    shadow = "box-shadow:0 1px 3px rgba(0,0,0,0.08);" if not is_dark() else ""
    blur = "backdrop-filter:blur(12px);" if is_dark() else ""
    return (
        f'<div style="background:{c["card"]}; {blur} '
        f'border:1px solid {c["card_border"]}; '
        f'border-radius:6px; padding:8px 10px; text-align:center; {shadow}">'
        f'<div style="color:{c["text_secondary"]}; font-size:0.6em; text-transform:uppercase; '
        f'letter-spacing:0.05em; margin-bottom:2px;">{label}</div>'
        f'<div style="color:{color}; font-size:1.1em; font-weight:600;">{text}</div>'
        f'</div>'
    )


def page_header(title: str) -> str:
    """Return HTML for a compact page header."""
    c = get_colors()
    return (
        f'<p style="margin:0;padding:4px 0;color:{c["text_heading"]};'
        f'font-size:1.1rem;font-weight:600;">{title}</p>'
    )


# ══════════════════════════════════════════════════════════════════════
# DYNAMIC CSS
# ══════════════════════════════════════════════════════════════════════

def theme_css() -> str:
    """Generate the full CSS string for the active theme.

    Sidebar follows the active theme (dark/light).
    """
    c = get_colors()
    dark = is_dark()

    # Card shadow for light mode
    card_shadow = "box-shadow: 0 1px 4px rgba(0,0,0,0.06);" if not dark else ""
    backdrop = "backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);" if dark else ""

    return f"""
    /* ===== Streamlit CSS variable overrides ===== */
    :root, .stApp {{
        --primary-color: #2962FF;
        --background-color: {c['bg']};
        --secondary-background-color: {c['surface']};
        --text-color: {c['text']};
        --font: "Source Sans Pro", sans-serif;
    }}

    /* ===== Base background ===== */
    .stApp {{ background-color: {c['bg']} !important; color: {c['text']} !important; }}

    /* ===== Sidebar — theme-aware ===== */
    .stSidebar, section[data-testid="stSidebar"] {{
        background: {c['sidebar_bg']} !important;
        border-right: 1px solid {c['sidebar_border']} !important;
    }}
    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stCaption {{
        color: {c['sidebar_text']} !important;
    }}
    section[data-testid="stSidebar"] h1 {{
        color: {c['sidebar_text']} !important;
    }}
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
        color: {c['sidebar_text']} !important;
        background-color: {c['sidebar_btn_bg']} !important;
        border-color: {c['sidebar_btn_border']} !important;
    }}
    section[data-testid="stSidebar"] button:hover,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{
        border-color: {c['sidebar_btn_hover_border']} !important;
        background-color: {c['sidebar_btn_hover_bg']} !important;
        color: {c['sidebar_text']} !important;
    }}
    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr {{
        border-color: {c['sidebar_divider']} !important;
    }}
    /* Sidebar nav section headers */
    [data-testid="stSidebarNav"] span[data-testid="stSidebarNavSectionHeader"] {{
        color: {c['sidebar_text_muted']} !important;
    }}
    /* Sidebar nav links */
    [data-testid="stSidebarNav"] ul li a {{
        color: {c['sidebar_text_muted']} !important;
    }}
    [data-testid="stSidebarNav"] ul li a[aria-current="page"] {{
        background-color: {c['sidebar_nav_active_bg']} !important;
        color: {c['sidebar_text']} !important;
        border-left: 2px solid #2962FF !important;
    }}
    [data-testid="stSidebarNav"] ul li a:hover {{
        background-color: {c['sidebar_nav_hover']} !important;
        color: {c['sidebar_text']} !important;
    }}
    [data-testid="stSidebarNav"] ul li a span[data-testid="stIconMaterial"] {{
        color: {c['sidebar_text_muted']} !important;
    }}
    [data-testid="stSidebarNav"] ul li a[aria-current="page"] span[data-testid="stIconMaterial"] {{
        color: #2962FF !important;
    }}

    /* Dividers */
    .stDivider, hr {{ border-color: {c['border']} !important; }}

    /* ===== Typography ===== */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {c['text_heading']} !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }}
    p, span, label, .stMarkdown, .stText,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {{
        color: {c['text']} !important;
    }}

    /* Metric values */
    [data-testid="stMetricValue"] {{
        color: {'#FFFFFF' if dark else '#1E2329'} !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        font-variant-numeric: tabular-nums;
    }}
    [data-testid="stMetricLabel"] {{
        color: {c['text_secondary']} !important;
        font-size: 0.65rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    [data-testid="stMetricDelta"] {{
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }}

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {c['text_secondary']} !important;
        font-size: 0.78rem !important;
    }}

    /* Section headers */
    .stMarkdown h2, .stSubheader {{
        border-bottom: 1px solid {c['border']};
        padding-bottom: 8px;
        margin-bottom: 16px !important;
    }}

    /* ===== Dropdowns ===== */
    [data-baseweb="popover"] {{ background-color: {c['popover_bg']} !important; }}
    [data-baseweb="popover"] li {{ color: {c['text']} !important; }}
    [data-baseweb="popover"] li:hover {{ background-color: {c['popover_hover']} !important; }}
    [role="listbox"] {{ background-color: {c['popover_bg']} !important; }}
    [role="option"] {{ color: {c['text']} !important; }}
    [role="option"]:hover {{ background-color: {c['popover_hover']} !important; }}

    /* Select boxes */
    [data-baseweb="select"] > div {{
        background-color: {c['input_bg']} !important;
        border-color: {c['input_border']} !important;
        color: {c['text']} !important;
    }}

    /* ===== Form containers ===== */
    [data-testid="stForm"] {{
        background: {c['form_bg']} !important;
        {backdrop}
        border: 1px solid {c['card_border']} !important;
        border-radius: 12px;
        padding: 32px !important;
        {card_shadow}
    }}
    [data-testid="stForm"] input {{
        background-color: {c['input_bg']} !important;
        border: 1px solid {c['input_border']} !important;
        border-radius: 6px !important;
        color: {c['text']} !important;
        padding: 10px 14px !important;
        font-size: 0.9rem !important;
    }}
    [data-testid="stForm"] input:focus {{
        border-color: #2962FF !important;
        box-shadow: 0 0 0 2px rgba(41,98,255,0.2) !important;
    }}

    /* ===== st.metric cards ===== */
    [data-testid="stMetric"] {{
        background: {c['card']} !important;
        {backdrop}
        border: 1px solid {c['card_border']} !important;
        border-radius: 8px !important;
        padding: 8px 10px !important;
        transition: border-color 0.2s ease;
        {card_shadow}
    }}
    [data-testid="stMetric"]:hover {{
        border-color: {c['card_hover']} !important;
    }}

    /* ===== Expanders ===== */
    [data-testid="stExpander"] {{
        background: {c['expander_bg']} !important;
        {backdrop}
        border: 1px solid {c['card_border']} !important;
        border-radius: 10px !important;
        {card_shadow}
    }}
    [data-testid="stExpander"] summary {{
        color: {c['text_secondary']} !important;
        background-color: transparent !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }}

    /* ===== Buttons — target data-testid to beat Emotion CSS ===== */
    /* Secondary buttons in main content */
    [data-testid="stBaseButton-secondary"] {{
        background-color: {c['btn_bg']} !important;
        color: {c['btn_text']} !important;
        border: 1px solid {c['btn_border']} !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.15s ease !important;
    }}
    [data-testid="stBaseButton-secondary"]:hover {{
        background-color: {c['btn_hover_bg']} !important;
        border-color: {c['btn_hover_border']} !important;
        color: {c['btn_text']} !important;
    }}
    [data-testid="stBaseButton-secondary"]:active,
    [data-testid="stBaseButton-secondary"]:focus {{
        background-color: {c['btn_hover_bg']} !important;
        color: {c['btn_text']} !important;
    }}
    /* Primary buttons — always blue with white text */
    [data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, #2962FF 0%, #1E88E5 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }}
    [data-testid="stBaseButton-primary"]:hover {{
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%) !important;
        color: #FFFFFF !important;
    }}

    /* ===== Form submit buttons — always white text on blue ===== */
    [data-testid="stBaseButton-secondaryFormSubmit"],
    [data-testid="stForm"] button[type="submit"] {{
        background: linear-gradient(135deg, #2962FF 0%, #1E88E5 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
    }}
    [data-testid="stBaseButton-secondaryFormSubmit"]:hover,
    [data-testid="stForm"] button[type="submit"]:hover {{
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%) !important;
        color: #FFFFFF !important;
    }}

    /* ===== Tooltips — theme-aware (data-testid beats Emotion CSS) ===== */
    [data-baseweb="tooltip"],
    [role="tooltip"] {{
        background-color: {c['surface']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 6px !important;
    }}
    [data-baseweb="tooltip"] div,
    [role="tooltip"] div {{
        background-color: {c['surface']} !important;
        color: {c['text']} !important;
    }}
    [data-testid="stTooltipContent"],
    [data-testid="stTooltipContent"] div,
    [data-testid="stTooltipContent"] p,
    [data-testid="stTooltipContent"] span,
    [data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"],
    [data-testid="stTooltipContent"] [data-testid="stMarkdownContainer"] p {{
        background-color: {c['surface']} !important;
        color: {c['text']} !important;
    }}
    /* Tooltip icon (?) button next to labels */
    [data-testid="stTooltipIcon"] {{
        color: {c['text_secondary']} !important;
    }}

    /* ===== Password toggle (eye icon) — theme-aware ===== */
    [data-testid="stTextInput"] button {{
        color: {c['text_secondary']} !important;
        background: transparent !important;
        border: none !important;
    }}
    [data-testid="stTextInput"] button:hover {{
        color: {c['text']} !important;
    }}

    /* ===== Tabs — pill style ===== */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {c['tab_bg']} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {c['tab_text']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {c['tab_active']} !important;
        color: #ffffff !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {c['tab_hover']} !important;
        color: {c['text']} !important;
    }}

    /* ===== DataFrames ===== */
    [data-testid="stDataFrame"] {{
        border: 1px solid {c['df_border']} !important;
        border-radius: 8px !important;
        overflow: hidden;
        {card_shadow}
    }}

    /* ===== Text inputs, number inputs, text areas ===== */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {{
        background-color: {c['input_bg']} !important;
        border-color: {c['input_border']} !important;
        color: {c['text']} !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus,
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {{
        border-color: #2962FF !important;
        box-shadow: 0 0 0 2px rgba(41,98,255,0.2) !important;
    }}

    /* ===== Multiselect ===== */
    [data-baseweb="tag"] {{
        background-color: {'rgba(41,98,255,0.2)' if dark else 'rgba(41,98,255,0.1)'} !important;
        color: {c['text']} !important;
    }}

    /* ===== Checkbox / Radio ===== */
    .stCheckbox label span, .stRadio label span {{
        color: {c['text']} !important;
    }}

    /* ===== Number input buttons ===== */
    [data-testid="stNumberInput"] button {{
        color: {c['text_secondary']} !important;
        background-color: {c['input_bg']} !important;
        border-color: {c['input_border']} !important;
    }}

    /* ===== Plotly ===== */
    .js-plotly-plot .plotly .main-svg {{ background: transparent !important; }}

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar-thumb {{ background: {c['scrollbar']}; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {c['scrollbar_hover']}; }}

    /* ===== Code blocks (st.code) ===== */
    [data-testid="stCode"],
    [data-testid="stCode"] > div {{
        background-color: {c['surface']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stCode"] code,
    [data-testid="stCode"] pre {{
        background-color: {c['surface']} !important;
        color: {c['text']} !important;
    }}
    .stCodeBlock, .stCodeBlock > div,
    pre[class*="language-"], code[class*="language-"] {{
        background-color: {c['surface']} !important;
        color: {c['text']} !important;
    }}
    /* Code copy button */
    [data-testid="stCode"] button {{
        color: {c['text_secondary']} !important;
    }}

    /* ===== Toast / Alerts ===== */
    [data-testid="stAlert"] {{
        background-color: {c['surface']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
    }}

    /* ===== Sidebar expand button — always visible ===== */
    button[data-testid="stSidebarCollapsedControl"] {{
        color: {'#D1D4DC' if dark else '#1E2329'} !important;
    }}
    """
