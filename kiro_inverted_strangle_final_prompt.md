# Kiro Prompt: Inverted Strangle with Defined Risk — Dashboard

**Authored by:** Manus AI | **Date:** March 5, 2026
**Target Repository:** `damerav/stockanalysis`
**New File:** `src/dashboard/strangle_app.py`
**Modified Files:** `src/data/init_db.py`, `src/dashboard/app.py`

---

## Strategy Reference

This dashboard implements the **Inverted Strangle with Defined Risk** strategy:

- **Core position:** Sell ITM Put at `spot + $5` and ITM Call at `spot - $5` (the $10 inversion) for 30–45 DTE.
- **Wings (defined risk):** Buy OTM Put at `short_put - $25` and OTM Call at `short_call + $25`.
- **Profit target:** Buy back the entire spread at 90% of the initial net credit (capture 10%).
- **Adjustment rule:** If price approaches a long wing, close and re-open at the new spot using the same 5-5 logic.

---

## Context & Codebase State

You are working on a production Streamlit + Python trading platform at `damerav/stockanalysis`. The platform uses:
- **PostgreSQL** via `DbRouter` (SQLAlchemy 2.0 engine for `pd.read_sql_query`, raw psycopg2 for writes).
- **`PolygonFetcher`** (`src/data/polygon_fetcher.py`) — `get_options_chain(underlying, expiry)` returns a DataFrame with columns: `contract_symbol`, `strike`, `expiry`, `option_type`, `last_price`, `bid`, `ask`, `volume`, `open_interest`, `iv`, `delta`, `gamma`, `theta`, `vega`.
- **`DbRouter`** (`src/data/db_router.py`) — `router.query(sql, params)` returns a DataFrame; `router.execute(sql, params)` writes; `router.close()` disposes the connection.
- **Theme system** — `from src.dashboard.theme import metric_card, badge_html, get_colors, get_plotly_layout` and `from src.dashboard.template import page_header`.
- **Navigation** — `src/dashboard/app.py` uses `st.navigation(_pages)` with a list of `st.Page(func, title, icon)` objects.

**Do NOT modify:** `db_router.py`, `polygon_fetcher.py`, or any existing dashboard pages. Only add to `init_db.py` and `app.py`.

---

## Part 1: Database Schema

### 1.1 — Add `inverted_strangle_positions` table

In `src/data/init_db.py`, inside the `_migrate_schema()` function, add the following `CREATE TABLE` statement immediately after the `market_breadth` table block. Use the same `IF NOT EXISTS` pattern as all other tables in the file:

```python
# src/data/init_db.py  →  inside _migrate_schema()

conn.execute("""
    CREATE TABLE IF NOT EXISTS inverted_strangle_positions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date      TEXT    NOT NULL,
        underlying      TEXT    NOT NULL DEFAULT 'SPY',
        spot_at_open    REAL    NOT NULL,
        expiry_date     TEXT    NOT NULL,
        dte_at_open     INTEGER NOT NULL,
        status          TEXT    NOT NULL DEFAULT 'OPEN',
        short_put       REAL    NOT NULL,
        short_call      REAL    NOT NULL,
        long_put        REAL    NOT NULL,
        long_call       REAL    NOT NULL,
        inversion_pts   REAL    NOT NULL DEFAULT 5.0,
        wing_pts        REAL    NOT NULL DEFAULT 25.0,
        initial_credit  REAL    NOT NULL,
        credit_per_leg  TEXT,
        profit_target   REAL    NOT NULL,
        current_value   REAL,
        current_pnl     REAL,
        close_date      TEXT,
        close_price     REAL,
        close_reason    TEXT,
        roll_count      INTEGER NOT NULL DEFAULT 0,
        notes           TEXT,
        vix_at_open     REAL
    );
""")

conn.execute("""
    CREATE TABLE IF NOT EXISTS inverted_strangle_adjustments (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        position_id     INTEGER NOT NULL,
        adj_date        TEXT    NOT NULL,
        adj_type        TEXT    NOT NULL,
        old_short_put   REAL,
        old_short_call  REAL,
        new_short_put   REAL,
        new_short_call  REAL,
        new_spot        REAL,
        debit_paid      REAL,
        notes           TEXT,
        FOREIGN KEY (position_id) REFERENCES inverted_strangle_positions(id)
    );
""")
```

For **PostgreSQL**, in the same `_migrate_schema()` function, add the equivalent `CREATE TABLE IF NOT EXISTS` blocks using `SERIAL PRIMARY KEY` instead of `INTEGER PRIMARY KEY AUTOINCREMENT`, following the exact same pattern used for the `strategy_rules` table already in the file.

---

## Part 2: Create `src/dashboard/strangle_app.py`

Create a new file `src/dashboard/strangle_app.py` with the complete implementation below. Do not split it into multiple files.

```python
"""
src/dashboard/strangle_app.py
Inverted Strangle with Defined Risk — Dashboard Page.

Strategy:
  - Sell ITM Put at spot + inversion_pts (default 5)
  - Sell ITM Call at spot - inversion_pts (default 5)
  - Buy OTM Put at short_put - wing_pts (default 25)
  - Buy OTM Call at short_call + wing_pts (default 25)
  - Profit target: buy back at 90% of initial credit (10% capture)
  - Adjustment: close and re-open at new spot if price approaches a wing
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

logger = logging.getLogger(__name__)

# ── Lazy imports ──────────────────────────────────────────────────────────────

def _router():
    from src.data.db_router import DbRouter
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    return DbRouter(cfg)


def _polygon():
    from src.data.polygon_fetcher import PolygonFetcher
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    api_key = cfg.get("polygon", {}).get("api_key", "")
    return PolygonFetcher(api_key)


# ── Theme helpers ─────────────────────────────────────────────────────────────

def _colors():
    try:
        from src.dashboard.theme import get_colors
        return get_colors()
    except Exception:
        return {}


def _plotly_layout():
    try:
        from src.dashboard.theme import get_plotly_layout
        return get_plotly_layout()
    except Exception:
        return {}


def _metric(label: str, value: str, color: str = "white", sub: str = "") -> str:
    try:
        from src.dashboard.theme import metric_card
        return metric_card(label, value, color, sub)
    except Exception:
        return f"**{label}:** {value}"


def _header(title: str) -> str:
    try:
        from src.dashboard.template import page_header
        return page_header(title)
    except Exception:
        return f"## {title}"


# ── Database helpers ──────────────────────────────────────────────────────────

def _load_positions(status: Optional[str] = None) -> pd.DataFrame:
    r = _router()
    if status:
        df = r.query(
            "SELECT * FROM inverted_strangle_positions WHERE status=? ORDER BY trade_date DESC",
            (status,),
        )
    else:
        df = r.query(
            "SELECT * FROM inverted_strangle_positions ORDER BY trade_date DESC"
        )
    r.close()
    return df


def _load_adjustments(position_id: int) -> pd.DataFrame:
    r = _router()
    df = r.query(
        "SELECT * FROM inverted_strangle_adjustments WHERE position_id=? ORDER BY adj_date",
        (position_id,),
    )
    r.close()
    return df


def _open_position(
    underlying: str,
    spot: float,
    expiry: str,
    dte: int,
    short_put: float,
    short_call: float,
    long_put: float,
    long_call: float,
    inversion_pts: float,
    wing_pts: float,
    initial_credit: float,
    credit_per_leg: str,
    profit_target_pct: float,
    vix_at_open: float,
) -> int:
    profit_target = round(initial_credit * (1 - profit_target_pct / 100), 2)
    r = _router()
    r.execute(
        """INSERT INTO inverted_strangle_positions
           (trade_date, underlying, spot_at_open, expiry_date, dte_at_open,
            status, short_put, short_call, long_put, long_call,
            inversion_pts, wing_pts, initial_credit, credit_per_leg,
            profit_target, current_value, current_pnl, roll_count, vix_at_open)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().strftime("%Y-%m-%d"),
            underlying, spot, expiry, dte,
            "OPEN", short_put, short_call, long_put, long_call,
            inversion_pts, wing_pts, initial_credit, credit_per_leg,
            profit_target, initial_credit, 0.0, 0, vix_at_open,
        ),
    )
    # Get the new row ID
    df = r.query(
        "SELECT id FROM inverted_strangle_positions ORDER BY id DESC LIMIT 1"
    )
    r.close()
    return int(df.iloc[0]["id"]) if not df.empty else -1


def _close_position(pos_id: int, close_price: float, reason: str):
    r = _router()
    df = r.query(
        "SELECT initial_credit FROM inverted_strangle_positions WHERE id=?", (pos_id,)
    )
    initial_credit = float(df.iloc[0]["initial_credit"]) if not df.empty else 0.0
    final_pnl = round(initial_credit - close_price, 2)
    r.execute(
        """UPDATE inverted_strangle_positions
           SET status='CLOSED', close_date=?, close_price=?,
               close_reason=?, current_pnl=?, current_value=?
           WHERE id=?""",
        (datetime.now().strftime("%Y-%m-%d"), close_price, reason, final_pnl, close_price, pos_id),
    )
    r.close()


def _log_adjustment(
    position_id: int,
    adj_type: str,
    old_short_put: float,
    old_short_call: float,
    new_short_put: float,
    new_short_call: float,
    new_spot: float,
    debit_paid: float,
    notes: str = "",
):
    r = _router()
    r.execute(
        """INSERT INTO inverted_strangle_adjustments
           (position_id, adj_date, adj_type, old_short_put, old_short_call,
            new_short_put, new_short_call, new_spot, debit_paid, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            position_id,
            datetime.now().strftime("%Y-%m-%d"),
            adj_type,
            old_short_put, old_short_call,
            new_short_put, new_short_call,
            new_spot, debit_paid, notes,
        ),
    )
    r.execute(
        "UPDATE inverted_strangle_positions SET roll_count = roll_count + 1 WHERE id=?",
        (position_id,),
    )
    r.close()


# ── Options pricing helpers ───────────────────────────────────────────────────

def _get_expiry_dates(dte_target: int = 45) -> list[str]:
    """Generate candidate expiry dates (Fridays) around the DTE target."""
    today = datetime.now().date()
    candidates = []
    for offset in range(dte_target - 10, dte_target + 15):
        d = today + timedelta(days=offset)
        if d.weekday() == 4:  # Friday
            candidates.append(d.strftime("%Y-%m-%d"))
    return candidates


def _fetch_chain_for_strike(
    underlying: str,
    expiry: str,
    strikes: list[float],
    option_types: list[str],
) -> pd.DataFrame:
    """Fetch the options chain and filter for specific strikes and types."""
    try:
        poly = _polygon()
        chain = poly.get_options_chain(underlying, expiry)
        if chain.empty:
            return pd.DataFrame()
        mask = chain["strike"].isin(strikes) & chain["option_type"].isin(option_types)
        return chain[mask].copy()
    except Exception as e:
        logger.warning("Options chain fetch failed: %s", e)
        return pd.DataFrame()


def _price_spread(chain: pd.DataFrame, short_put: float, short_call: float,
                  long_put: float, long_call: float) -> dict:
    """
    Calculate the net credit and per-leg prices from the options chain.
    Uses mid-price (bid+ask)/2 for each leg.
    Returns a dict with keys: short_put_price, short_call_price,
    long_put_price, long_call_price, net_credit, credit_per_leg_str.
    """
    def mid(row):
        return round((float(row["bid"]) + float(row["ask"])) / 2, 2)

    def get_price(strike, opt_type, fallback=0.0):
        row = chain[(chain["strike"] == strike) & (chain["option_type"] == opt_type)]
        return mid(row.iloc[0]) if not row.empty else fallback

    sp_price = get_price(short_put, "put")
    sc_price = get_price(short_call, "call")
    lp_price = get_price(long_put, "put")
    lc_price = get_price(long_call, "call")

    net_credit = round((sp_price + sc_price) - (lp_price + lc_price), 2)
    leg_str = (
        f"Short Put ${sp_price} | Short Call ${sc_price} | "
        f"Long Put ${lp_price} | Long Call ${lc_price}"
    )
    return {
        "short_put_price": sp_price,
        "short_call_price": sc_price,
        "long_put_price": lp_price,
        "long_call_price": lc_price,
        "net_credit": net_credit,
        "credit_per_leg_str": leg_str,
    }


def _get_greeks(chain: pd.DataFrame, short_put: float, short_call: float,
                long_put: float, long_call: float) -> dict:
    """Aggregate net Greeks for the four-leg spread."""
    def g(strike, opt_type, col, sign=1):
        row = chain[(chain["strike"] == strike) & (chain["option_type"] == opt_type)]
        return sign * float(row.iloc[0][col]) if not row.empty else 0.0

    net_delta = g(short_put, "put", "delta", -1) + g(short_call, "call", "delta", -1) \
              + g(long_put, "put", "delta", 1) + g(long_call, "call", "delta", 1)
    net_gamma = g(short_put, "put", "gamma", -1) + g(short_call, "call", "gamma", -1) \
              + g(long_put, "put", "gamma", 1) + g(long_call, "call", "gamma", 1)
    net_theta = g(short_put, "put", "theta", -1) + g(short_call, "call", "theta", -1) \
              + g(long_put, "put", "theta", 1) + g(long_call, "call", "theta", 1)
    net_vega  = g(short_put, "put", "vega", -1) + g(short_call, "call", "vega", -1) \
              + g(long_put, "put", "vega", 1) + g(long_call, "call", "vega", 1)
    return {
        "delta": round(net_delta, 4),
        "gamma": round(net_gamma, 4),
        "theta": round(net_theta, 4),
        "vega": round(net_vega, 4),
    }


# ── P&L Visualization ─────────────────────────────────────────────────────────

def _pnl_curve(pos: pd.Series) -> go.Figure:
    """
    Render the theoretical P&L at expiration for the four-leg spread.
    The inverted strangle has a profit zone between short_call and short_put.
    """
    lp = float(pos["long_put"])
    sp = float(pos["short_put"])
    sc = float(pos["short_call"])
    lc = float(pos["long_call"])
    credit = float(pos["initial_credit"])
    target = float(pos["profit_target"])

    prices = np.linspace(lp - 5, lc + 5, 400)
    pnl = []
    for price in prices:
        # Short put P&L: collected premium, loses if price < short_put
        short_put_pnl = -max(sp - price, 0)
        # Long put P&L: paid premium, gains if price < long_put
        long_put_pnl  = max(lp - price, 0)
        # Short call P&L: collected premium, loses if price > short_call
        short_call_pnl = -max(price - sc, 0)
        # Long call P&L: paid premium, gains if price > long_call
        long_call_pnl  = max(price - lc, 0)
        # Net P&L = credit received + all leg P&Ls
        net = credit + short_put_pnl + long_put_pnl + short_call_pnl + long_call_pnl
        pnl.append(net)

    layout = _plotly_layout()
    layout.update(
        title="P&L at Expiration",
        xaxis_title="Underlying Price at Expiration",
        yaxis_title="Profit / Loss ($)",
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )

    fig = go.Figure(layout=layout)

    # Shade profit zone (between short_call and short_put)
    fig.add_vrect(
        x0=sc, x1=sp,
        fillcolor="rgba(0,200,100,0.12)",
        layer="below", line_width=0,
        annotation_text="Max Profit Zone",
        annotation_position="top left",
    )

    # P&L curve — color by profit/loss
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in pnl]
    fig.add_trace(go.Scatter(
        x=prices, y=pnl,
        mode="lines",
        name="P&L",
        line=dict(color="#2962ff", width=2),
    ))

    # Mark key strikes
    for strike, label, color in [
        (lp, f"Long Put\n${lp}", "#ef5350"),
        (sp, f"Short Put\n${sp}", "#ff9800"),
        (sc, f"Short Call\n${sc}", "#ff9800"),
        (lc, f"Long Call\n${lc}", "#ef5350"),
    ]:
        fig.add_vline(x=strike, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="top right")

    # Profit target line
    fig.add_hline(y=credit - target, line_dash="dot", line_color="#26a69a",
                  annotation_text=f"Profit Target ${credit - target:.2f}",
                  annotation_position="right")

    # Zero line
    fig.add_hline(y=0, line_color="gray", line_width=1)

    return fig


def _greeks_gauge(greeks: dict) -> go.Figure:
    """Render a small bar chart of the net Greeks."""
    layout = _plotly_layout()
    layout.update(
        title="Net Position Greeks",
        height=200,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
    )
    fig = go.Figure(layout=layout)
    names = ["Delta", "Gamma", "Theta", "Vega"]
    vals = [greeks["delta"], greeks["gamma"], greeks["theta"], greeks["vega"]]
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in vals]
    fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors))
    return fig


def _performance_chart(closed: pd.DataFrame) -> go.Figure:
    """Render cumulative P&L of all closed positions."""
    if closed.empty:
        return go.Figure()
    df = closed.sort_values("close_date").copy()
    df["cumulative_pnl"] = df["current_pnl"].cumsum()
    layout = _plotly_layout()
    layout.update(
        title="Cumulative P&L — Closed Positions",
        xaxis_title="Close Date",
        yaxis_title="Cumulative P&L ($)",
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(
        x=df["close_date"], y=df["cumulative_pnl"],
        mode="lines+markers",
        line=dict(color="#2962ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(41,98,255,0.1)",
    ))
    return fig


# ── Main Page ─────────────────────────────────────────────────────────────────

def page_strangle():
    st.markdown(_header("🕷️ Inverted Strangle with Defined Risk Dashboard"), unsafe_allow_html=True)

    tab_open, tab_new, tab_history, tab_tracker, tab_guide = st.tabs([
        "📋 Open Positions",
        "➕ New Trade",
        "📊 History & Performance",
        "📈 Tracker",
        "📖 Strategy Guide",
    ])

    # ── Tab 1: Open Positions ─────────────────────────────────────────────────
    with tab_open:
        open_pos = _load_positions("OPEN")

        if open_pos.empty:
            st.info("No open positions. Use the **New Trade** tab to enter a position.")
        else:
            st.caption(f"{len(open_pos)} open position(s)")
            for _, pos in open_pos.iterrows():
                pos_id = int(pos["id"])
                underlying = pos["underlying"]
                expiry = pos["expiry_date"]
                credit = float(pos["initial_credit"])
                target = float(pos["profit_target"])
                current_pnl = float(pos["current_pnl"]) if pos["current_pnl"] is not None else 0.0
                pct_captured = round((current_pnl / credit) * 100, 1) if credit else 0

                with st.expander(
                    f"**{underlying}** | Opened {pos['trade_date']} | "
                    f"Expiry {expiry} | Credit ${credit:.2f} | "
                    f"P&L ${current_pnl:+.2f} ({pct_captured}%)",
                    expanded=True,
                ):
                    # ── Strike summary ────────────────────────────────────────
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(_metric("Long Put", f"${pos['long_put']:.2f}", "#ef5350"), unsafe_allow_html=True)
                    with c2:
                        st.markdown(_metric("Short Put (ITM)", f"${pos['short_put']:.2f}", "#ff9800"), unsafe_allow_html=True)
                    with c3:
                        st.markdown(_metric("Short Call (ITM)", f"${pos['short_call']:.2f}", "#ff9800"), unsafe_allow_html=True)
                    with c4:
                        st.markdown(_metric("Long Call", f"${pos['long_call']:.2f}", "#ef5350"), unsafe_allow_html=True)

                    # ── P&L metrics ───────────────────────────────────────────
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Initial Credit", f"${credit:.2f}")
                    with m2:
                        st.metric("Profit Target (Limit)", f"${target:.2f}")
                    with m3:
                        st.metric("Current P&L", f"${current_pnl:+.2f}")
                    with m4:
                        st.metric("% Captured", f"{pct_captured}%",
                                  delta=f"Target: 10%",
                                  delta_color="normal")

                    # ── P&L Curve ─────────────────────────────────────────────
                    st.plotly_chart(_pnl_curve(pos), use_container_width=True)

                    # ── Live Greeks (from Polygon) ────────────────────────────
                    with st.expander("📐 Live Greeks (fetches from Polygon)", expanded=False):
                        if st.button("Refresh Greeks", key=f"greeks_{pos_id}"):
                            with st.spinner("Fetching options chain..."):
                                chain = _fetch_chain_for_strike(
                                    underlying, expiry,
                                    [pos["short_put"], pos["short_call"], pos["long_put"], pos["long_call"]],
                                    ["put", "call"],
                                )
                                if not chain.empty:
                                    greeks = _get_greeks(
                                        chain,
                                        float(pos["short_put"]), float(pos["short_call"]),
                                        float(pos["long_put"]), float(pos["long_call"]),
                                    )
                                    g1, g2, g3, g4 = st.columns(4)
                                    with g1:
                                        st.metric("Net Delta", f"{greeks['delta']:+.4f}")
                                    with g2:
                                        st.metric("Net Gamma", f"{greeks['gamma']:+.4f}")
                                    with g3:
                                        st.metric("Net Theta", f"{greeks['theta']:+.4f}")
                                    with g4:
                                        st.metric("Net Vega", f"{greeks['vega']:+.4f}")
                                    st.plotly_chart(_greeks_gauge(greeks), use_container_width=True)
                                else:
                                    st.warning("Could not fetch options chain from Polygon.")

                    # ── Adjustment History ────────────────────────────────────
                    adjs = _load_adjustments(pos_id)
                    if not adjs.empty:
                        with st.expander(f"🔄 Adjustment History ({len(adjs)} rolls)", expanded=False):
                            st.dataframe(adjs[["adj_date", "adj_type", "old_short_put",
                                               "old_short_call", "new_short_put",
                                               "new_short_call", "debit_paid", "notes"]],
                                         use_container_width=True)

                    # ── Actions ───────────────────────────────────────────────
                    st.markdown("---")
                    a1, a2, a3 = st.columns(3)

                    with a1:
                        with st.form(key=f"close_form_{pos_id}"):
                            close_price = st.number_input(
                                "Close Price (debit to buy back)", 0.0, credit, target, 0.01,
                                key=f"cp_{pos_id}",
                            )
                            close_reason = st.selectbox(
                                "Reason", ["Profit Target Hit", "Manual Close", "Expiration", "Stop Loss"],
                                key=f"cr_{pos_id}",
                            )
                            if st.form_submit_button("✅ Close Position"):
                                _close_position(pos_id, close_price, close_reason)
                                st.success(f"Position #{pos_id} closed. P&L: ${credit - close_price:+.2f}")
                                st.rerun()

                    with a2:
                        with st.form(key=f"roll_form_{pos_id}"):
                            new_spot = st.number_input(
                                "New Spot Price (for roll)", 100.0, 1000.0,
                                float(pos["spot_at_open"]), 0.01,
                                key=f"ns_{pos_id}",
                            )
                            debit = st.number_input(
                                "Debit Paid to Close Old Legs ($)", 0.0, 100.0, 0.0, 0.01,
                                key=f"dp_{pos_id}",
                            )
                            roll_notes = st.text_input("Notes", key=f"rn_{pos_id}")
                            if st.form_submit_button("🔄 Roll Position"):
                                inv = float(pos["inversion_pts"])
                                wing = float(pos["wing_pts"])
                                new_sp = new_spot + inv
                                new_sc = new_spot - inv
                                _log_adjustment(
                                    pos_id, "ROLL",
                                    float(pos["short_put"]), float(pos["short_call"]),
                                    new_sp, new_sc, new_spot, debit, roll_notes,
                                )
                                # Update strikes in DB
                                r = _router()
                                r.execute(
                                    """UPDATE inverted_strangle_positions
                                       SET short_put=?, short_call=?,
                                           long_put=?, long_call=?,
                                           spot_at_open=?
                                       WHERE id=?""",
                                    (new_sp, new_sc, new_sp - wing, new_sc + wing, new_spot, pos_id),
                                )
                                r.close()
                                st.success(f"Position #{pos_id} rolled to new spot ${new_spot:.2f}.")
                                st.rerun()

                    with a3:
                        with st.form(key=f"update_pnl_{pos_id}"):
                            curr_val = st.number_input(
                                "Current Spread Value (debit to close now)", 0.0, credit * 2,
                                target, 0.01, key=f"cv_{pos_id}",
                            )
                            if st.form_submit_button("🔄 Update P&L"):
                                new_pnl = round(credit - curr_val, 2)
                                r = _router()
                                r.execute(
                                    "UPDATE inverted_strangle_positions SET current_value=?, current_pnl=? WHERE id=?",
                                    (curr_val, new_pnl, pos_id),
                                )
                                r.close()
                                st.success(f"P&L updated: ${new_pnl:+.2f}")
                                st.rerun()

    # ── Tab 2: New Trade ──────────────────────────────────────────────────────
    with tab_new:
        st.subheader("Build New Inverted Strangle with Defined Risk Trade")

        with st.form("new_trade_form"):
            c1, c2 = st.columns(2)
            with c1:
                underlying = st.text_input("Underlying", "SPY").upper()
                spot = st.number_input("Current Spot Price", 100.0, 1000.0, 540.0, 0.01)
                dte = st.slider("Target DTE", 25, 60, 45)
                inversion = st.number_input("Inversion Points ($)", 1.0, 20.0, 5.0, 0.5,
                                            help="Short Put = Spot + X | Short Call = Spot - X")
            with c2:
                wing = st.number_input("Wing Width ($)", 10.0, 60.0, 25.0, 1.0,
                                       help="Long Put = Short Put - W | Long Call = Short Call + W")
                profit_target_pct = st.slider("Profit Target (% of Credit to Capture)", 5, 50, 10)
                manual_credit = st.number_input(
                    "Manual Credit Override (leave 0 to fetch from Polygon)", 0.0, 200.0, 0.0, 0.01,
                )

            submitted = st.form_submit_button("📐 Price & Open Trade")

        if submitted:
            short_put = round(spot + inversion, 2)
            short_call = round(spot - inversion, 2)
            long_put = round(short_put - wing, 2)
            long_call = round(short_call + wing, 2)

            # Get expiry date
            expiry_candidates = _get_expiry_dates(dte)
            expiry = expiry_candidates[0] if expiry_candidates else (
                datetime.now() + timedelta(days=dte)
            ).strftime("%Y-%m-%d")

            st.markdown("### Proposed Trade Structure")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Long Put (Wing)", f"${long_put:.2f}")
            col2.metric("Short Put (ITM)", f"${short_put:.2f}")
            col3.metric("Short Call (ITM)", f"${short_call:.2f}")
            col4.metric("Long Call (Wing)", f"${long_call:.2f}")
            st.caption(f"Expiry: {expiry} ({dte} DTE) | Inversion: ${inversion} | Wing: ${wing}")

            # Price the spread
            credit_info = {}
            if manual_credit > 0:
                credit_info = {
                    "net_credit": manual_credit,
                    "credit_per_leg_str": "Manual override",
                }
            else:
                with st.spinner("Fetching options chain from Polygon..."):
                    chain = _fetch_chain_for_strike(
                        underlying, expiry,
                        [short_put, short_call, long_put, long_call],
                        ["put", "call"],
                    )
                    if not chain.empty:
                        credit_info = _price_spread(chain, short_put, short_call, long_put, long_call)
                    else:
                        st.warning(
                            "Could not fetch options chain from Polygon. "
                            "Enter the credit manually above and re-submit."
                        )

            if credit_info:
                net_credit = credit_info["net_credit"]
                profit_target_price = round(net_credit * (1 - profit_target_pct / 100), 2)

                st.markdown("### Pricing")
                p1, p2, p3 = st.columns(3)
                p1.metric("Net Credit Received", f"${net_credit:.2f}")
                p2.metric(f"Profit Target ({profit_target_pct}% capture)", f"${profit_target_price:.2f}")
                p3.metric("Max Risk", f"${wing - net_credit:.2f}")
                st.caption(credit_info.get("credit_per_leg_str", ""))

                # Preview P&L curve
                preview_pos = pd.Series({
                    "long_put": long_put, "short_put": short_put,
                    "short_call": short_call, "long_call": long_call,
                    "initial_credit": net_credit, "profit_target": profit_target_price,
                })
                st.plotly_chart(_pnl_curve(preview_pos), use_container_width=True)

                if st.button("✅ Confirm & Open Position"):
                    new_id = _open_position(
                        underlying, spot, expiry, dte,
                        short_put, short_call, long_put, long_call,
                        inversion, wing, net_credit,
                        credit_info.get("credit_per_leg_str", ""),
                        profit_target_pct,
                        0.0, # VIX at open
                    )
                    st.success(f"Position #{new_id} opened successfully!")
                    st.rerun()

    # ── Tab 3: History & Performance ──────────────────────────────────────────
    with tab_history:
        st.subheader("Closed Positions & Performance")
        closed = _load_positions("CLOSED")

        if closed.empty:
            st.info("No closed positions yet.")
        else:
            # Summary metrics
            total_pnl = closed["current_pnl"].sum()
            wins = (closed["current_pnl"] > 0).sum()
            win_rate = round(wins / len(closed) * 100, 1)
            avg_pnl = closed["current_pnl"].mean()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total P&L", f"${total_pnl:+.2f}")
            s2.metric("Win Rate", f"{win_rate}%")
            s3.metric("Avg P&L per Trade", f"${avg_pnl:+.2f}")
            s4.metric("Total Trades", len(closed))

            st.plotly_chart(_performance_chart(closed), use_container_width=True)

            # Closed positions table
            display_cols = [
                "trade_date", "underlying", "expiry_date", "initial_credit",
                "profit_target", "close_price", "current_pnl", "close_reason", "roll_count",
            ]
            st.dataframe(
                closed[display_cols].rename(columns={
                    "trade_date": "Open Date",
                    "underlying": "Ticker",
                    "expiry_date": "Expiry",
                    "initial_credit": "Credit",
                    "profit_target": "Target",
                    "close_price": "Close Price",
                    "current_pnl": "P&L",
                    "close_reason": "Reason",
                    "roll_count": "Rolls",
                }),
                use_container_width=True,
            )

    # ── Tab 4: Tracker ────────────────────────────────────────────────────────
    with tab_tracker:
        st.subheader("Performance Tracker")
        closed = _load_positions("CLOSED")

        if closed.empty:
            st.info("No closed positions yet.")
        else:
            # Time period filter
            period = st.selectbox("Time Period", ["All Time", "Last 90 Days", "Last 30 Days", "Year to Date"])
            if period == "Last 90 Days":
                closed = closed[pd.to_datetime(closed["close_date"]) > datetime.now() - timedelta(days=90)]
            elif period == "Last 30 Days":
                closed = closed[pd.to_datetime(closed["close_date"]) > datetime.now() - timedelta(days=30)]
            elif period == "Year to Date":
                closed = closed[pd.to_datetime(closed["close_date"]).dt.year == datetime.now().year]

            # Rolling metrics
            closed["rolling_pnl"] = closed["current_pnl"].rolling(window=10, min_periods=1).mean()
            closed["rolling_win_rate"] = (closed["current_pnl"] > 0).rolling(window=10, min_periods=1).mean() * 100

            # Win/loss streaks
            streaks = []
            current_streak = 0
            for pnl in closed["current_pnl"]:
                if pnl > 0:
                    if current_streak >= 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                else:
                    if current_streak <= 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                streaks.append(current_streak)
            closed["streak"] = streaks

            # Charts
            st.plotly_chart(_rolling_pnl_chart(closed), use_container_width=True)
            st.plotly_chart(_rolling_win_rate_chart(closed), use_container_width=True)
            st.plotly_chart(_streak_chart(closed), use_container_width=True)

            # Regime breakdown
            st.subheader("Performance by VIX Regime")
            closed["vix_regime"] = pd.cut(closed["vix_at_open"], bins=[0, 15, 25, 100], labels=["Low", "Medium", "High"])
            regime_perf = closed.groupby("vix_regime")["current_pnl"].agg(["sum", "mean", "count"])
            st.dataframe(regime_perf)

    # ── Tab 5: Strategy Guide ─────────────────────────────────────────────────
    with tab_guide:
        st.markdown("""
## Inverted Strangle with Defined Risk (Inverted Strangle with Defined Risk) — Quick Reference

### Trade Architecture

| Leg | Strike | Type | Action |
| :--- | :--- | :--- | :--- |
| Long Put (Wing) | Spot − Inversion − Wing | OTM Put | **Buy** |
| Short Put (Core) | Spot + Inversion | ITM Put | **Sell** |
| Short Call (Core) | Spot − Inversion | ITM Call | **Sell** |
| Long Call (Wing) | Spot − Inversion + Wing | OTM Call | **Buy** |

**Default parameters:** Inversion = $5 | Wing = $25 | DTE = 30–45 | Profit Target = 10%

### Execution Rules

**Entry:** Sell the spread at market open when IV Rank > 30% and VIX is stable or declining.

**Profit Exit:** Place a GTC limit order to buy back the entire four-leg spread at **90% of the initial credit** immediately after entry. This captures 10% of the premium.

**Adjustment (Roll):** If the underlying price moves within **$5 of a long wing**, close the entire position and re-open at the new spot price using the same 5-5 inversion logic. Log the debit paid and increment the roll counter.

**Stop Loss:** If the spread value exceeds **200% of the initial credit**, close the position to limit losses.

### Greek Characteristics

| Greek | Position | Implication |
| :--- | :--- | :--- |
| **Theta** | **Positive** | Time decay works in your favor. Profits accelerate near expiration. |
| **Vega** | **Negative** | Short volatility. Benefits from IV crush or stable prices. |
| **Delta** | **Near Zero** | Delta-neutral at entry. Requires re-balancing if the market trends. |
| **Gamma** | **Negative** | Risk accelerates near short strikes. Monitor closely inside 14 DTE. |

### Risk Profile

The maximum risk is defined by the wing width minus the net credit received:
```
Max Loss = Wing Width − Net Credit
         = $25 − $12.50 (example)
         = $12.50 per share ($1,250 per contract)
```

The maximum profit is the full net credit, achieved if the underlying expires between the two short strikes.
        """)

# --- Tracker Charts ---

def _rolling_pnl_chart(df):
    layout = _plotly_layout()
    layout.update(title="10-Trade Rolling Average P&L", xaxis_title="Trade Number", yaxis_title="Avg P&L ($)")
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(y=df["rolling_pnl"], mode="lines", line=dict(color="#2962ff")))
    return fig

def _rolling_win_rate_chart(df):
    layout = _plotly_layout()
    layout.update(title="10-Trade Rolling Win Rate", xaxis_title="Trade Number", yaxis_title="Win Rate (%)")
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(y=df["rolling_win_rate"], mode="lines", line=dict(color="#26a69a")))
    return fig

def _streak_chart(df):
    layout = _plotly_layout()
    layout.update(title="Win/Loss Streaks", xaxis_title="Trade Number", yaxis_title="Streak Length")
    fig = go.Figure(layout=layout)
    colors = ["#26a69a" if v > 0 else "#ef5350" for v in df["streak"]]
    fig.add_trace(go.Bar(y=df["streak"], marker_color=colors))
    return fig
```

---

## Part 3: Wire into `app.py` Navigation

In `src/dashboard/app.py`, add the following import near the top of the file with the other page imports:

```python
# src/dashboard/app.py  →  top of file, with other imports

from src.dashboard.strangle_app import page_strangle
```

Then, in the `_pages` dictionary inside the `st.navigation` block, add the new page. Place it in the **"Markets"** group, after the `page_es` entry:

```python
# src/dashboard/app.py  →  st.navigation block

_pages = {
    "Markets": [
        st.Page(page_spy,       title="SPY Predictor",     icon=":material/query_stats:", default=True),
        st.Page(page_performance, title="Performance",      icon=":material/verified:"),
        st.Page(page_es,        title="ES Strategy",        icon=":material/candlestick_chart:"),
        st.Page(page_strangle,  title="Inverted Strangle with Defined Risk",  icon=":material/mediation:"),   # ← ADD THIS LINE
        st.Page(page_tuning,    title="Tune & Backtest",    icon=":material/tune:"),
        st.Page(page_whatif,    title="What-If Analysis",   icon=":material/science:"),
        st.Page(page_single_stock, title="Single-Stock",    icon=":material/search:"),
        st.Page(page_quant_agent, title="Quant Agent",      icon=":material/smart_toy:"),
    ],
    "Admin": [
        st.Page(page_monitoring, title="Monitoring",        icon=":material/monitor_heart:"),
        st.Page(page_grafana,    title="Grafana Dashboards",icon=":material/dashboard:"),
        st.Page(page_rules,      title="Strategy Rules",    icon=":material/rule:"),
        st.Page(page_admin,      title="Admin",             icon=":material/settings:"),
    ],
}
```

---

## Part 4: Verification Checklist

After implementation, verify the following:

- [ ] `inverted_strangle_positions` and `inverted_strangle_adjustments` tables are created in the database on startup.
- [ ] "Inverted Strangle with Defined Risk" appears in the sidebar navigation under "Markets".
- [ ] The **New Trade** tab correctly calculates all four strikes from spot, inversion, and wing inputs.
- [ ] The P&L curve renders correctly, showing the profit zone between the two short strikes.
- [ ] The **Open Positions** tab loads positions from the database and displays them.
- [ ] The **Close Position** form correctly updates `status`, `close_date`, `close_price`, and `current_pnl`.
- [ ] The **Roll Position** form correctly logs the adjustment and updates the strikes in the database.
- [ ] The **Update P&L** form correctly updates `current_value` and `current_pnl`.
- [ ] The **History & Performance** tab shows cumulative P&L chart and win rate metrics.
- [ ] The **Tracker** tab shows rolling P&L, rolling win rate, win/loss streaks, and VIX regime breakdown.
- [ ] The **Strategy Guide** tab renders the reference table correctly.
- [ ] No existing pages or files are modified other than `init_db.py` and `app.py`.
