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


_polygon_instance = None

def _polygon():
    global _polygon_instance
    if _polygon_instance is not None:
        return _polygon_instance
    from src.data.polygon_fetcher import PolygonFetcher
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    api_key = cfg.get("polygon", {}).get("api_key", "")
    if not api_key or api_key in ("YOUR_POLYGON_KEY", "FROM_ENCRYPTED_DB"):
        try:
            from src.data.secrets_manager import get_secret
            api_key = get_secret("polygon_api_key", fallback=api_key or "")
        except Exception:
            pass
    _polygon_instance = PolygonFetcher(api_key)
    return _polygon_instance


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
        return f"<b>{label}:</b> {value}"


def _header(title: str):
    """Render page header directly via st.markdown."""
    try:
        from src.dashboard.template import page_header
        page_header(title)
    except Exception:
        st.markdown(f"## {title}")


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> float:
    """Black-Scholes option price for simulation mode.

    S=spot, K=strike, T=years to expiry, r=risk-free rate, sigma=IV (annualized).
    opt_type: 'call' or 'put'.
    """
    from math import log, sqrt, exp
    # Standard normal CDF approximation (Abramowitz & Stegun)
    def _norm_cdf(x):
        import math
        return (1.0 + math.erf(x / sqrt(2.0))) / 2.0

    if T <= 0:
        # At expiration
        if opt_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if opt_type == "call":
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    else:
        return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _theoretical_spread_price(
    spot: float, short_put: float, short_call: float,
    long_put: float, long_call: float, dte: int, vix: float = 20.0,
) -> dict:
    """Price the 4-leg spread using Black-Scholes with VIX as IV proxy.

    Returns same dict format as _price_spread() for seamless integration.
    """
    T = dte / 365.0
    r = 0.045  # approximate risk-free rate
    sigma = vix / 100.0  # VIX is annualized IV in percentage points

    sp_price = round(_bs_price(spot, short_put, T, r, sigma, "put"), 2)
    sc_price = round(_bs_price(spot, short_call, T, r, sigma, "call"), 2)
    lp_price = round(_bs_price(spot, long_put, T, r, sigma, "put"), 2)
    lc_price = round(_bs_price(spot, long_call, T, r, sigma, "call"), 2)

    net_credit = round((sp_price + sc_price) - (lp_price + lc_price), 2)
    leg_str = (
        f"Short Put ${sp_price} | Short Call ${sc_price} | "
        f"Long Put ${lp_price} | Long Call ${lc_price} (theoretical @ VIX={vix:.1f})"
    )
    return {
        "short_put_price": sp_price, "short_call_price": sc_price,
        "long_put_price": lp_price, "long_call_price": lc_price,
        "net_credit": net_credit, "credit_per_leg_str": leg_str,
    }

@st.cache_data(ttl=15)
def _live_spot(ticker: str = "SPY") -> float:
    """Fetch live spot price via yfinance with 15s cache."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        price = t.fast_info.get("lastPrice") or t.fast_info.get("previousClose")
        return round(float(price), 2) if price else 540.0
    except Exception:
        return 540.0


@st.cache_data(ttl=30)
def _yf_options_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """Fetch options chain from yfinance for a specific expiry.

    Returns DataFrame with columns: strike, option_type, bid, ask, lastPrice,
    volume, openInterest, impliedVolatility.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        avail = t.options
        if not avail:
            return pd.DataFrame()
        # Find closest expiry to requested
        target = datetime.strptime(expiry, "%Y-%m-%d").date()
        best = min(avail, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - target).days))
        chain = t.option_chain(best)
        calls = chain.calls.copy()
        calls["option_type"] = "call"
        puts = chain.puts.copy()
        puts["option_type"] = "put"
        df = pd.concat([calls, puts], ignore_index=True)
        # Normalize column names
        df = df.rename(columns={"impliedVolatility": "iv"})
        return df
    except Exception as e:
        logger.warning("yfinance options chain failed: %s", e)
        return pd.DataFrame()


def _yf_price_spread(ticker: str, expiry: str, short_put: float, short_call: float,
                     long_put: float, long_call: float) -> dict | None:
    """Price the 4-leg spread using live yfinance bid/ask mid-prices.

    Returns same dict format as _price_spread() or None if chain unavailable.
    """
    chain = _yf_options_chain(ticker, expiry)
    if chain.empty:
        return None

    def _mid(strike, opt_type):
        row = chain[(chain["strike"] == strike) & (chain["option_type"] == opt_type)]
        if row.empty:
            # Try nearest strike within $1
            nearby = chain[(chain["option_type"] == opt_type) &
                           (chain["strike"].between(strike - 1, strike + 1))]
            if nearby.empty:
                return None
            row = nearby.iloc[[nearby["strike"].sub(strike).abs().argmin()]]
        r = row.iloc[0]
        bid = float(r.get("bid", 0) or 0)
        ask = float(r.get("ask", 0) or 0)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 2)
        last = float(r.get("lastPrice", 0) or 0)
        return round(last, 2) if last > 0 else None

    sp_price = _mid(short_put, "put")
    sc_price = _mid(short_call, "call")
    lp_price = _mid(long_put, "put")
    lc_price = _mid(long_call, "call")

    if any(p is None for p in [sp_price, sc_price, lp_price, lc_price]):
        return None

    net_credit = round((sp_price + sc_price) - (lp_price + lc_price), 2)
    leg_str = (
        f"Short Put ${sp_price} | Short Call ${sc_price} | "
        f"Long Put ${lp_price} | Long Call ${lc_price} (yfinance live mid)"
    )
    return {
        "short_put_price": sp_price, "short_call_price": sc_price,
        "long_put_price": lp_price, "long_call_price": lc_price,
        "net_credit": net_credit, "credit_per_leg_str": leg_str,
    }


def _yf_refresh_position_pnl(pos: pd.Series) -> float | None:
    """Re-price an open position using live yfinance quotes.

    Returns current spread value (what it would cost to close) or None.
    """
    underlying = str(pos["underlying"]).replace("[PAPER] ", "")
    expiry = str(pos["expiry_date"])
    chain = _yf_options_chain(underlying, expiry)
    if chain.empty:
        return None

    def _mid(strike, opt_type):
        row = chain[(chain["strike"] == strike) & (chain["option_type"] == opt_type)]
        if row.empty:
            nearby = chain[(chain["option_type"] == opt_type) &
                           (chain["strike"].between(strike - 1, strike + 1))]
            if nearby.empty:
                return None
            row = nearby.iloc[[nearby["strike"].sub(strike).abs().argmin()]]
        r = row.iloc[0]
        bid = float(r.get("bid", 0) or 0)
        ask = float(r.get("ask", 0) or 0)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 2)
        last = float(r.get("lastPrice", 0) or 0)
        return round(last, 2) if last > 0 else None

    sp = _mid(float(pos["short_put"]), "put")
    sc = _mid(float(pos["short_call"]), "call")
    lp = _mid(float(pos["long_put"]), "put")
    lc = _mid(float(pos["long_call"]), "call")

    if any(p is None for p in [sp, sc, lp, lc]):
        return None

    # Current value = what you'd pay to close (buy back shorts, sell longs)
    return round((sp + sc) - (lp + lc), 2)


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
    underlying: str, spot: float, expiry: str, dte: int,
    short_put: float, short_call: float, long_put: float, long_call: float,
    inversion_pts: float, wing_pts: float, initial_credit: float,
    credit_per_leg: str, profit_target_pct: float, vix_at_open: float,
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
    df = r.query("SELECT id FROM inverted_strangle_positions ORDER BY id DESC LIMIT 1")
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
    position_id: int, adj_type: str,
    old_short_put: float, old_short_call: float,
    new_short_put: float, new_short_call: float,
    new_spot: float, debit_paid: float, notes: str = "",
):
    r = _router()
    r.execute(
        """INSERT INTO inverted_strangle_adjustments
           (position_id, adj_date, adj_type, old_short_put, old_short_call,
            new_short_put, new_short_call, new_spot, debit_paid, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            position_id, datetime.now().strftime("%Y-%m-%d"), adj_type,
            old_short_put, old_short_call, new_short_put, new_short_call,
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


def _snap_strike(available_strikes, target: float) -> float:
    """Snap a target strike to the nearest available strike in the chain."""
    if not len(available_strikes):
        return target
    idx = np.argmin(np.abs(np.array(available_strikes) - target))
    return float(available_strikes[idx])


def _fetch_chain_for_strike(
    underlying: str, expiry: str, strikes: list[float], option_types: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Fetch the options chain and filter for nearest matching strikes.

    Returns (filtered_chain, snap_map) where snap_map maps each original
    strike to the nearest available strike in the chain.
    """
    try:
        poly = _polygon()
        chain = poly.get_options_chain(underlying, expiry)
        if chain.empty:
            return pd.DataFrame(), {}
        available = sorted(chain["strike"].unique())
        snap_map = {s: _snap_strike(available, s) for s in strikes}
        snapped = list(set(snap_map.values()))
        mask = chain["strike"].isin(snapped) & chain["option_type"].isin(option_types)
        return chain[mask].copy(), snap_map
    except Exception as e:
        logger.warning("Options chain fetch failed: %s", e)
        return pd.DataFrame(), {}


def _price_spread(chain: pd.DataFrame, short_put: float, short_call: float,
                  long_put: float, long_call: float) -> dict:
    """Calculate net credit and per-leg prices from the options chain (mid-price)."""
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
        "short_put_price": sp_price, "short_call_price": sc_price,
        "long_put_price": lp_price, "long_call_price": lc_price,
        "net_credit": net_credit, "credit_per_leg_str": leg_str,
    }


def _get_greeks(chain: pd.DataFrame, short_put: float, short_call: float,
                long_put: float, long_call: float) -> dict:
    """Aggregate net Greeks for the four-leg spread."""
    def g(strike, opt_type, col, sign=1):
        row = chain[(chain["strike"] == strike) & (chain["option_type"] == opt_type)]
        return sign * float(row.iloc[0][col]) if not row.empty else 0.0

    return {
        "delta": round(
            g(short_put, "put", "delta", -1) + g(short_call, "call", "delta", -1)
            + g(long_put, "put", "delta", 1) + g(long_call, "call", "delta", 1), 4),
        "gamma": round(
            g(short_put, "put", "gamma", -1) + g(short_call, "call", "gamma", -1)
            + g(long_put, "put", "gamma", 1) + g(long_call, "call", "gamma", 1), 4),
        "theta": round(
            g(short_put, "put", "theta", -1) + g(short_call, "call", "theta", -1)
            + g(long_put, "put", "theta", 1) + g(long_call, "call", "theta", 1), 4),
        "vega": round(
            g(short_put, "put", "vega", -1) + g(short_call, "call", "vega", -1)
            + g(long_put, "put", "vega", 1) + g(long_call, "call", "vega", 1), 4),
    }


# ── P&L Visualization ─────────────────────────────────────────────────────────

def _pnl_curve(pos: pd.Series) -> go.Figure:
    """Render theoretical P&L at expiration for the four-leg spread."""
    lp = float(pos["long_put"])
    sp = float(pos["short_put"])
    sc = float(pos["short_call"])
    lc = float(pos["long_call"])
    credit = float(pos["initial_credit"])
    target = float(pos["profit_target"])

    prices = np.linspace(lp - 5, lc + 5, 400)
    pnl = []
    for price in prices:
        net = credit + (-max(sp - price, 0)) + max(lp - price, 0) + (-max(price - sc, 0)) + max(price - lc, 0)
        pnl.append(net)

    layout = _plotly_layout()
    layout.update(
        title="P&L at Expiration", xaxis_title="Underlying Price",
        yaxis_title="Profit / Loss ($)", height=350,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    fig = go.Figure(layout=layout)

    fig.add_vrect(x0=sc, x1=sp, fillcolor="rgba(0,200,100,0.12)",
                  layer="below", line_width=0,
                  annotation_text="Max Profit Zone", annotation_position="top left")

    fig.add_trace(go.Scatter(x=prices, y=pnl, mode="lines", name="P&L",
                             line=dict(color="#2962ff", width=2)))

    for strike, label, color in [
        (lp, f"Long Put ${lp}", "#ef5350"), (sp, f"Short Put ${sp}", "#ff9800"),
        (sc, f"Short Call ${sc}", "#ff9800"), (lc, f"Long Call ${lc}", "#ef5350"),
    ]:
        fig.add_vline(x=strike, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="top right")

    fig.add_hline(y=credit - target, line_dash="dot", line_color="#26a69a",
                  annotation_text=f"Profit Target ${credit - target:.2f}", annotation_position="right")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    return fig


def _greeks_gauge(greeks: dict) -> go.Figure:
    layout = _plotly_layout()
    layout.update(title="Net Position Greeks", height=200,
                  margin=dict(l=40, r=20, t=40, b=40), showlegend=False)
    fig = go.Figure(layout=layout)
    names = ["Delta", "Gamma", "Theta", "Vega"]
    vals = [greeks["delta"], greeks["gamma"], greeks["theta"], greeks["vega"]]
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in vals]
    fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors))
    return fig


def _performance_chart(closed: pd.DataFrame) -> go.Figure:
    if closed.empty:
        return go.Figure()
    df = closed.sort_values("close_date").copy()
    df["cumulative_pnl"] = df["current_pnl"].cumsum()
    layout = _plotly_layout()
    layout.update(title="Cumulative P&L — Closed Positions", xaxis_title="Close Date",
                  yaxis_title="Cumulative P&L ($)", height=300,
                  margin=dict(l=40, r=20, t=40, b=40))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=df["close_date"], y=df["cumulative_pnl"],
                             mode="lines+markers", line=dict(color="#2962ff", width=2),
                             fill="tozeroy", fillcolor="rgba(41,98,255,0.1)"))
    return fig


def _rolling_pnl_chart(df):
    layout = _plotly_layout()
    layout.update(title="10-Trade Rolling Average P&L", xaxis_title="Trade #", yaxis_title="Avg P&L ($)")
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(y=df["rolling_pnl"], mode="lines", line=dict(color="#2962ff")))
    return fig


def _rolling_win_rate_chart(df):
    layout = _plotly_layout()
    layout.update(title="10-Trade Rolling Win Rate", xaxis_title="Trade #", yaxis_title="Win Rate (%)")
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(y=df["rolling_win_rate"], mode="lines", line=dict(color="#26a69a")))
    return fig


def _streak_chart(df):
    layout = _plotly_layout()
    layout.update(title="Win/Loss Streaks", xaxis_title="Trade #", yaxis_title="Streak")
    fig = go.Figure(layout=layout)
    colors = ["#26a69a" if v > 0 else "#ef5350" for v in df["streak"]]
    fig.add_trace(go.Bar(y=df["streak"], marker_color=colors))
    return fig


# ── Predictive & Risk helpers ─────────────────────────────────────────────────

@st.cache_data(ttl=600)
def _fetch_iv_rank(underlying: str = "SPY") -> dict:
    """Compute IV Rank and IV Percentile using VIX history as proxy.

    IV Rank = (current_vix - 52w_low) / (52w_high - 52w_low) * 100
    IV Percentile = % of days in past year where VIX was below current level
    """
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="1y")
        if hist.empty or len(hist) < 20:
            return {"iv_rank": None, "iv_percentile": None, "current_iv": None}
        current = float(hist["Close"].iloc[-1])
        low_52w = float(hist["Close"].min())
        high_52w = float(hist["Close"].max())
        iv_rank = ((current - low_52w) / (high_52w - low_52w) * 100) if high_52w > low_52w else 50.0
        iv_pct = (hist["Close"] < current).sum() / len(hist) * 100
        return {
            "iv_rank": round(iv_rank, 1),
            "iv_percentile": round(iv_pct, 1),
            "current_iv": round(current, 2),
            "low_52w": round(low_52w, 2),
            "high_52w": round(high_52w, 2),
        }
    except Exception as e:
        logger.warning("IV Rank fetch failed: %s", e)
        return {"iv_rank": None, "iv_percentile": None, "current_iv": None}


@st.cache_data(ttl=600)
def _fetch_vix_term_structure() -> dict:
    """Fetch VIX term structure: VIX vs VIX3M for contango/backwardation."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period="5d")
        vix3m = yf.Ticker("^VIX3M").history(period="5d")
        if vix.empty or vix3m.empty:
            return {"term_structure_pct": None, "vix": None, "vix3m": None, "state": "Unknown"}
        v = float(vix["Close"].iloc[-1])
        v3m = float(vix3m["Close"].iloc[-1])
        spread_pct = round((v3m - v) / v * 100, 2) if v > 0 else 0
        state = "Contango" if spread_pct > 0 else "Backwardation"
        return {"term_structure_pct": spread_pct, "vix": round(v, 2), "vix3m": round(v3m, 2), "state": state}
    except Exception as e:
        logger.warning("VIX term structure fetch failed: %s", e)
        return {"term_structure_pct": None, "vix": None, "vix3m": None, "state": "Unknown"}


@st.cache_data(ttl=600)
def _fetch_market_skew() -> dict:
    """Fetch CBOE SKEW index and put/call ratio.

    PCR: DB options_analytics first (instant), Polygon API fallback, yfinance last resort.
    SKEW: yfinance ^SKEW ticker.
    """
    result = {"skew_index": None, "pcr": None}

    # SKEW index from yfinance (lightweight — just price history)
    try:
        import yfinance as yf
        skew = yf.Ticker("^SKEW").history(period="5d")
        if not skew.empty:
            result["skew_index"] = round(float(skew["Close"].iloc[-1]), 2)
    except Exception:
        pass

    # PCR: Try DB first (instant — populated by validation/pipeline)
    try:
        from src.data.db_router import get_router
        router = get_router()
        df = router.query(
            "SELECT put_call_ratio FROM options_analytics ORDER BY date DESC LIMIT 1"
        )
        router.close()
        if df is not None and not df.empty and df.iloc[0]["put_call_ratio"] is not None:
            result["pcr"] = round(float(df.iloc[0]["put_call_ratio"]), 3)
            return result
    except Exception:
        pass

    # PCR fallback: Polygon API (fetches full chain — slower)
    try:
        poly = _polygon()
        if poly and poly.api_key:
            analytics = poly.get_options_analytics("SPY")
            pcr = analytics.get("put_call_ratio")
            if pcr:
                result["pcr"] = round(pcr, 3)
                return result
    except Exception:
        pass

    # PCR last resort: yfinance SPY options chain (slowest)
    if result["pcr"] is None:
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            expirations = spy.options
            if expirations:
                chain = spy.option_chain(expirations[0])
                put_vol = chain.puts["volume"].sum()
                call_vol = chain.calls["volume"].sum()
                if call_vol > 0:
                    result["pcr"] = round(put_vol / call_vol, 3)
        except Exception:
            pass

    return result


def _estimate_vix_spike_prob(iv_data: dict = None, ts_data: dict = None,
                             skew_data: dict = None) -> dict:
    """Heuristic VIX spike probability based on current conditions.

    Accepts pre-fetched data to avoid redundant API calls.
    Uses a weighted score from: VIX level, term structure, SKEW, and IV Rank.
    """
    try:
        if iv_data is None:
            iv_data = _fetch_iv_rank()
        if ts_data is None:
            ts_data = _fetch_vix_term_structure()
        if skew_data is None:
            skew_data = _fetch_market_skew()

        score = 0.0  # 0-100 probability estimate
        factors = []

        # Factor 1: VIX level (higher = more spike risk)
        vix = ts_data.get("vix") or 0
        if vix > 30:
            score += 30
            factors.append(f"VIX elevated ({vix:.1f})")
        elif vix > 20:
            score += 15
            factors.append(f"VIX moderate ({vix:.1f})")
        else:
            score += 5
            factors.append(f"VIX low ({vix:.1f})")

        # Factor 2: Term structure (backwardation = danger)
        ts_pct = ts_data.get("term_structure_pct") or 0
        if ts_pct < -5:
            score += 30
            factors.append(f"Steep backwardation ({ts_pct:+.1f}%)")
        elif ts_pct < 0:
            score += 15
            factors.append(f"Mild backwardation ({ts_pct:+.1f}%)")
        else:
            score += 0
            factors.append(f"Contango ({ts_pct:+.1f}%)")

        # Factor 3: SKEW (>140 = elevated tail risk)
        skew = skew_data.get("skew_index") or 130
        if skew > 150:
            score += 20
            factors.append(f"SKEW very high ({skew:.0f})")
        elif skew > 140:
            score += 10
            factors.append(f"SKEW elevated ({skew:.0f})")
        else:
            score += 0
            factors.append(f"SKEW normal ({skew:.0f})")

        # Factor 4: IV Rank (high IV Rank = mean reversion likely, lower spike risk)
        iv_rank = iv_data.get("iv_rank") or 50
        if iv_rank > 80:
            score -= 5  # Already high, likely to revert down
            factors.append(f"IV Rank high ({iv_rank:.0f}) — mean reversion likely")
        elif iv_rank < 20:
            score += 10  # Low IV, complacency, spike risk
            factors.append(f"IV Rank low ({iv_rank:.0f}) — complacency risk")

        prob = max(0, min(100, score))
        return {"spike_prob": round(prob, 1), "factors": factors}
    except Exception as e:
        logger.warning("VIX spike estimation failed: %s", e)
        return {"spike_prob": None, "factors": []}


def _gauge_chart(value: float, title: str, max_val: float = 100,
                 green_range: tuple = (0, 30), yellow_range: tuple = (30, 60),
                 red_range: tuple = (60, 100)) -> go.Figure:
    """Render a gauge chart."""
    layout = _plotly_layout()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [0, max_val]},
            "bar": {"color": "#2962ff"},
            "steps": [
                {"range": list(green_range), "color": "rgba(38,166,154,0.3)"},
                {"range": list(yellow_range), "color": "rgba(255,152,0,0.3)"},
                {"range": list(red_range), "color": "rgba(239,83,80,0.3)"},
            ],
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=layout.get("font", {}).get("color", "white")))
    return fig


def _run_adjustment_alerts(open_pos: pd.DataFrame) -> list[dict]:
    """Check open positions for adjustment triggers. Returns list of alert dicts."""
    alerts = []
    today = datetime.now().date()
    for _, pos in open_pos.iterrows():
        pos_id = int(pos["id"])
        credit = float(pos["initial_credit"])
        current_pnl = float(pos["current_pnl"]) if pos["current_pnl"] is not None else 0.0

        # 21 DTE Rule
        try:
            expiry = datetime.strptime(str(pos["expiry_date"]), "%Y-%m-%d").date()
            dte_remaining = (expiry - today).days
            if dte_remaining <= 21:
                alerts.append({
                    "pos_id": pos_id, "type": "21 DTE",
                    "severity": "high" if dte_remaining <= 14 else "medium",
                    "msg": f"Position #{pos_id}: {dte_remaining} DTE remaining — close or roll to avoid gamma risk.",
                })
        except Exception:
            pass

        # 50% Profit Target
        if credit > 0 and current_pnl / credit >= 0.50:
            alerts.append({
                "pos_id": pos_id, "type": "50% Profit",
                "severity": "low",
                "msg": f"Position #{pos_id}: {current_pnl/credit*100:.0f}% of credit captured — consider taking profit.",
            })

        # Delta threshold (if position_delta is stored)
        pos_delta = pos.get("position_delta")
        if pos_delta is not None and abs(float(pos_delta)) > 0.30:
            alerts.append({
                "pos_id": pos_id, "type": "Delta Breach",
                "severity": "high",
                "msg": f"Position #{pos_id}: Delta = {float(pos_delta):+.2f} — roll untested side to re-center.",
            })

    return alerts


# ── Main Page ─────────────────────────────────────────────────────────────────

def page_strangle():
    from src.dashboard.chatbot_widget import render_chatbot_widget
    render_chatbot_widget(page_key="strangle", page_title="Inverted Strangle")
    _header("🕷️ Inverted Strangle — Defined Risk")

    tab_open, tab_new, tab_predict, tab_history, tab_tracker, tab_guardrails, tab_whatif, tab_guide, tab_deepdive = st.tabs([
        "📋 Open Positions", "➕ New Trade", "🔮 Prediction & Risk",
        "📊 History & Performance", "📈 Tracker", "🛡️ Guardrails & C2C",
        "🤔 What-If Engine", "📖 Strategy Guide", "📜 Strategy Deep-Dive",
    ])

    # ── Tab 1: Open Positions ─────────────────────────────────────────────────
    with tab_open:
        open_pos = _load_positions("OPEN")

        # Adjustment alerts
        if not open_pos.empty:
            alerts = _run_adjustment_alerts(open_pos)
            if alerts:
                st.markdown("#### ⚠️ Adjustment Alerts")
                for alert in alerts:
                    icon = "🔴" if alert["severity"] == "high" else ("🟡" if alert["severity"] == "medium" else "🟢")
                    st.warning(f"{icon} **{alert['type']}** — {alert['msg']}")
                st.markdown("---")

        if open_pos.empty:
            st.info("No open positions. Use the **New Trade** tab to enter a position.")
        else:
            # Bulk live P&L refresh
            if st.button("📡 Refresh All P&L (Live Quotes)", key="bulk_refresh_pnl"):
                updated = 0
                with st.spinner("Fetching live quotes for all positions..."):
                    for _, p in open_pos.iterrows():
                        new_val = _yf_refresh_position_pnl(p)
                        if new_val is not None:
                            pid = int(p["id"])
                            new_pnl = round(float(p["initial_credit"]) - new_val, 2)
                            r = _router()
                            r.execute(
                                "UPDATE inverted_strangle_positions SET current_value=?, current_pnl=? WHERE id=?",
                                (new_val, new_pnl, pid))
                            r.close()
                            updated += 1
                if updated:
                    st.success(f"Updated {updated}/{len(open_pos)} position(s) with live quotes.")
                    st.rerun()
                else:
                    st.warning("Could not fetch live quotes (market may be closed).")

            st.caption(f"{len(open_pos)} open position(s)")
            for _, pos in open_pos.iterrows():
                pos_id = int(pos["id"])
                credit = float(pos["initial_credit"])
                target = float(pos["profit_target"])
                current_pnl = float(pos["current_pnl"]) if pos["current_pnl"] is not None else 0.0
                pct_captured = round((current_pnl / credit) * 100, 1) if credit else 0

                with st.expander(
                    f"**{pos['underlying']}** | Opened {pos['trade_date']} | "
                    f"Expiry {pos['expiry_date']} | Credit ${credit:.2f} | "
                    f"P&L ${current_pnl:+.2f} ({pct_captured}%)",
                    expanded=True,
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(_metric("Long Put", f"${pos['long_put']:.2f}", "#ef5350"), unsafe_allow_html=True)
                    with c2:
                        st.markdown(_metric("Short Put (ITM)", f"${pos['short_put']:.2f}", "#ff9800"), unsafe_allow_html=True)
                    with c3:
                        st.markdown(_metric("Short Call (ITM)", f"${pos['short_call']:.2f}", "#ff9800"), unsafe_allow_html=True)
                    with c4:
                        st.markdown(_metric("Long Call", f"${pos['long_call']:.2f}", "#ef5350"), unsafe_allow_html=True)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Initial Credit", f"${credit:.2f}")
                    with m2:
                        st.metric("Profit Target", f"${target:.2f}")
                    with m3:
                        st.metric("Current P&L", f"${current_pnl:+.2f}")
                    with m4:
                        st.metric("% Captured", f"{pct_captured}%", delta="Target: 10%", delta_color="normal")

                    st.plotly_chart(_pnl_curve(pos), use_container_width=True, key=f"pnl_{pos_id}")

                    with st.expander("📐 Live Greeks", expanded=False):
                        if st.button("Refresh Greeks", key=f"greeks_{pos_id}"):
                            with st.spinner("Fetching options chain..."):
                                _ul = str(pos["underlying"]).replace("[PAPER] ", "")
                                chain, snap_map = _fetch_chain_for_strike(
                                    _ul, pos["expiry_date"],
                                    [pos["short_put"], pos["short_call"], pos["long_put"], pos["long_call"]],
                                    ["put", "call"],
                                )
                                if not chain.empty:
                                    # Use snapped strikes for Greeks lookup
                                    _sp = snap_map.get(float(pos["short_put"]), float(pos["short_put"]))
                                    _sc = snap_map.get(float(pos["short_call"]), float(pos["short_call"]))
                                    _lp = snap_map.get(float(pos["long_put"]), float(pos["long_put"]))
                                    _lc = snap_map.get(float(pos["long_call"]), float(pos["long_call"]))
                                    greeks = _get_greeks(chain, _sp, _sc, _lp, _lc)
                                    g1, g2, g3, g4 = st.columns(4)
                                    with g1: st.metric("Net Delta", f"{greeks['delta']:+.4f}")
                                    with g2: st.metric("Net Gamma", f"{greeks['gamma']:+.4f}")
                                    with g3: st.metric("Net Theta", f"{greeks['theta']:+.4f}")
                                    with g4: st.metric("Net Vega", f"{greeks['vega']:+.4f}")
                                    st.plotly_chart(_greeks_gauge(greeks), use_container_width=True, key=f"gg_{pos_id}")
                                    # Show strike snapping info if any strikes were adjusted
                                    snapped = {k: v for k, v in snap_map.items() if abs(k - v) > 0.01}
                                    if snapped:
                                        snap_str = ", ".join(f"${k:.2f}→${v:.0f}" for k, v in snapped.items())
                                        st.caption(f"ℹ️ Strikes snapped to nearest chain: {snap_str}")

                                    # Store position delta for adjustment alerts
                                    try:
                                        r = _router()
                                        r.execute("UPDATE inverted_strangle_positions SET position_delta=? WHERE id=?",
                                                  (greeks["delta"], pos_id))
                                        r.close()
                                    except Exception:
                                        pass
                                else:
                                    st.warning("Could not fetch options chain from Polygon.")

                    adjs = _load_adjustments(pos_id)
                    if not adjs.empty:
                        with st.expander(f"🔄 Adjustment History ({len(adjs)} rolls)", expanded=False):
                            st.dataframe(adjs[["adj_date", "adj_type", "old_short_put", "old_short_call",
                                               "new_short_put", "new_short_call", "debit_paid", "notes"]],
                                         use_container_width=True)

                    st.markdown("---")
                    a1, a2, a3 = st.columns(3)

                    with a1:
                        with st.form(key=f"close_form_{pos_id}"):
                            close_price = st.number_input("Close Price (debit)", 0.0, credit, target, 0.01, key=f"cp_{pos_id}")
                            close_reason = st.selectbox("Reason", ["Profit Target Hit", "Manual Close", "Expiration", "Stop Loss",
                                                                    "21 DTE Rule", "50% Target", "Delta Adjustment"], key=f"cr_{pos_id}")
                            if st.form_submit_button("✅ Close Position"):
                                _close_position(pos_id, close_price, close_reason)
                                st.success(f"Position #{pos_id} closed. P&L: ${credit - close_price:+.2f}")
                                st.rerun()

                    with a2:
                        with st.form(key=f"roll_form_{pos_id}"):
                            new_spot = st.number_input("New Spot Price", 100.0, 1000.0, float(pos["spot_at_open"]), 0.01, key=f"ns_{pos_id}")
                            debit = st.number_input("Debit Paid ($)", 0.0, 100.0, 0.0, 0.01, key=f"dp_{pos_id}")
                            roll_notes = st.text_input("Notes", key=f"rn_{pos_id}")
                            if st.form_submit_button("🔄 Roll Position"):
                                inv = float(pos["inversion_pts"])
                                wing = float(pos["wing_pts"])
                                new_sp = new_spot + inv
                                new_sc = new_spot - inv
                                _log_adjustment(pos_id, "ROLL", float(pos["short_put"]), float(pos["short_call"]),
                                                new_sp, new_sc, new_spot, debit, roll_notes)
                                r = _router()
                                r.execute(
                                    """UPDATE inverted_strangle_positions
                                       SET short_put=?, short_call=?, long_put=?, long_call=?, spot_at_open=?
                                       WHERE id=?""",
                                    (new_sp, new_sc, new_sp - wing, new_sc + wing, new_spot, pos_id),
                                )
                                r.close()
                                st.success(f"Position #{pos_id} rolled to spot ${new_spot:.2f}.")
                                st.rerun()

                    with a3:
                        with st.form(key=f"update_pnl_{pos_id}"):
                            curr_val = st.number_input("Current Spread Value", 0.0, credit * 2, target, 0.01, key=f"cv_{pos_id}")
                            if st.form_submit_button("🔄 Update P&L"):
                                new_pnl = round(credit - curr_val, 2)
                                r = _router()
                                r.execute("UPDATE inverted_strangle_positions SET current_value=?, current_pnl=? WHERE id=?",
                                          (curr_val, new_pnl, pos_id))
                                r.close()
                                st.success(f"P&L updated: ${new_pnl:+.2f}")
                                st.rerun()

    # ── Tab 2: New Trade ──────────────────────────────────────────────────────
    with tab_new:
        st.subheader("Build New Inverted Strangle")

        # Pre-trade checklist
        with st.expander("🔍 Pre-Trade Checklist (click to run)", expanded=False):
            if st.button("Run Pre-Trade Checklist", key="pretrade_check"):
                with st.spinner("Fetching market conditions..."):
                    iv_data = _fetch_iv_rank()
                    ts_data = _fetch_vix_term_structure()
                    spike_data = _estimate_vix_spike_prob(iv_data, ts_data)

                    checks = []
                    iv_rank = iv_data.get("iv_rank")
                    if iv_rank is not None:
                        ok = iv_rank > 50
                        checks.append(("IV Rank > 50", ok, f"IV Rank = {iv_rank:.0f}"))
                    else:
                        checks.append(("IV Rank > 50", False, "Data unavailable"))

                    ts_state = ts_data.get("state", "Unknown")
                    ok_ts = ts_state == "Contango"
                    checks.append(("VIX Term Structure in Contango", ok_ts,
                                   f"{ts_state} ({ts_data.get('term_structure_pct', 0):+.1f}%)"))

                    spike_prob = spike_data.get("spike_prob")
                    if spike_prob is not None:
                        ok_spike = spike_prob < 15
                        checks.append(("VIX Spike Probability < 15%", ok_spike, f"{spike_prob:.0f}%"))
                    else:
                        checks.append(("VIX Spike Probability < 15%", False, "Data unavailable"))

                    all_pass = all(c[1] for c in checks)
                    for label, passed, detail in checks:
                        icon = "✅" if passed else "❌"
                        st.markdown(f"{icon} **{label}** — {detail}")

                    if all_pass:
                        st.success("All checks passed. Conditions are favorable for entry.")
                    else:
                        st.warning("Some checks failed. Consider waiting for better conditions or adjusting parameters.")

        # Mode toggle (outside form so it persists)
        sim_mode = st.toggle("🧪 Paper Trading Mode (live yfinance quotes, no real execution)",
                             value=True, key="sim_mode_toggle")
        if sim_mode:
            st.caption("Uses live yfinance bid/ask mid-prices for realistic paper trades. Falls back to Black-Scholes when market is closed.")

        with st.form("new_trade_form"):
            c1, c2 = st.columns(2)
            with c1:
                underlying = st.text_input("Underlying", "SPY").upper()
                live_price = _live_spot(underlying)
                spot = st.number_input("Current Spot Price", 100.0, 1000.0, live_price, 0.01)
                dte = st.slider("Target DTE", 25, 60, 45)
                inversion = st.number_input("Inversion Points ($)", 1.0, 20.0, 5.0, 0.5,
                                            help="Short Put = Spot + X | Short Call = Spot - X")
            with c2:
                wing = st.number_input("Wing Width ($)", 10.0, 60.0, 25.0, 1.0,
                                       help="Long Put = Short Put - W | Long Call = Short Call + W")
                profit_target_pct = st.slider("Profit Target (% of Credit)", 5, 50, 10)
                manual_credit = st.number_input("Manual Credit Override (0 = auto-price)", 0.0, 200.0, 0.0, 0.01)
            submitted = st.form_submit_button("📐 Price & Open Trade")

        if submitted:
            short_put = round(spot + inversion, 2)
            short_call = round(spot - inversion, 2)
            long_put = round(short_put - wing, 2)
            long_call = round(short_call + wing, 2)

            expiry_candidates = _get_expiry_dates(dte)
            expiry = expiry_candidates[0] if expiry_candidates else (datetime.now() + timedelta(days=dte)).strftime("%Y-%m-%d")

            credit_info = {}
            if manual_credit > 0:
                credit_info = {"net_credit": manual_credit, "credit_per_leg_str": "Manual override"}
            elif sim_mode:
                credit_info = _yf_price_spread(underlying, expiry, short_put, short_call, long_put, long_call)
                if not credit_info:
                    try:
                        ts_data = _fetch_vix_term_structure()
                        vix_now = ts_data.get("vix") or 20.0
                    except Exception:
                        vix_now = 20.0
                    credit_info = _theoretical_spread_price(spot, short_put, short_call, long_put, long_call, dte, vix_now)
            else:
                chain, snap_map = _fetch_chain_for_strike(underlying, expiry,
                                                [short_put, short_call, long_put, long_call], ["put", "call"])
                if not chain.empty:
                    _sp = snap_map.get(short_put, short_put)
                    _sc = snap_map.get(short_call, short_call)
                    _lp = snap_map.get(long_put, long_put)
                    _lc = snap_map.get(long_call, long_call)
                    credit_info = _price_spread(chain, _sp, _sc, _lp, _lc)
                else:
                    credit_info = _yf_price_spread(underlying, expiry, short_put, short_call, long_put, long_call)
                    if not credit_info:
                        try:
                            ts_data = _fetch_vix_term_structure()
                            vix_now = ts_data.get("vix") or 20.0
                        except Exception:
                            vix_now = 20.0
                        credit_info = _theoretical_spread_price(spot, short_put, short_call, long_put, long_call, dte, vix_now)

            if credit_info:
                net_credit = credit_info["net_credit"]
                # Capture entry conditions
                entry_iv_rank = None
                entry_ts = None
                entry_vix = None
                try:
                    iv_data = _fetch_iv_rank()
                    ts_data2 = _fetch_vix_term_structure()
                    entry_iv_rank = iv_data.get("iv_rank")
                    entry_ts = ts_data2.get("term_structure_pct")
                    entry_vix = ts_data2.get("vix")
                except Exception:
                    pass

                # Store trade data in session_state so confirm button survives rerun
                st.session_state["_pending_trade"] = {
                    "underlying": underlying, "spot": spot, "expiry": expiry, "dte": dte,
                    "short_put": short_put, "short_call": short_call,
                    "long_put": long_put, "long_call": long_call,
                    "inversion": inversion, "wing": wing,
                    "net_credit": net_credit,
                    "credit_per_leg_str": credit_info.get("credit_per_leg_str", ""),
                    "profit_target_pct": profit_target_pct,
                    "entry_vix": entry_vix or 0.0,
                    "entry_iv_rank": entry_iv_rank,
                    "entry_ts": entry_ts,
                    "sim_mode": sim_mode,
                }
            else:
                st.session_state.pop("_pending_trade", None)
                st.error("Could not price the spread. Check market data availability.")

        # Show pending trade preview and confirm button (persists across reruns)
        pending = st.session_state.get("_pending_trade")
        if pending:
            st.markdown("### Proposed Trade Structure")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Long Put (Wing)", f"${pending['long_put']:.2f}")
            col2.metric("Short Put (ITM)", f"${pending['short_put']:.2f}")
            col3.metric("Short Call (ITM)", f"${pending['short_call']:.2f}")
            col4.metric("Long Call (Wing)", f"${pending['long_call']:.2f}")
            st.caption(f"Expiry: {pending['expiry']} ({pending['dte']} DTE) | "
                       f"Inversion: ${pending['inversion']} | Wing: ${pending['wing']}")

            if pending.get("sim_mode"):
                st.info("📡 Live yfinance mid-prices (paper trade)")

            net_credit = pending["net_credit"]
            ptp = pending["profit_target_pct"]
            profit_target_price = round(net_credit * (1 - ptp / 100), 2)

            st.markdown("### Pricing")
            p1, p2, p3 = st.columns(3)
            p1.metric("Net Credit Received", f"${net_credit:.2f}")
            p2.metric(f"Profit Target ({ptp}%)", f"${profit_target_price:.2f}")
            p3.metric("Max Risk", f"${pending['wing'] - net_credit:.2f}")
            st.caption(pending.get("credit_per_leg_str", ""))

            preview_pos = pd.Series({
                "long_put": pending["long_put"], "short_put": pending["short_put"],
                "short_call": pending["short_call"], "long_call": pending["long_call"],
                "initial_credit": net_credit, "profit_target": profit_target_price,
            })
            st.plotly_chart(_pnl_curve(preview_pos), use_container_width=True, key="pnl_preview")

            if st.button("✅ Confirm & Open Position"):
                trade_underlying = f"[PAPER] {pending['underlying']}" if pending.get("sim_mode") else pending["underlying"]
                new_id = _open_position(
                    trade_underlying, pending["spot"], pending["expiry"], pending["dte"],
                    pending["short_put"], pending["short_call"], pending["long_put"], pending["long_call"],
                    pending["inversion"], pending["wing"], pending["net_credit"],
                    pending.get("credit_per_leg_str", ""),
                    pending["profit_target_pct"], pending.get("entry_vix", 0.0),
                )
                # Store entry conditions
                eir = pending.get("entry_iv_rank")
                ets = pending.get("entry_ts")
                if eir is not None or ets is not None:
                    try:
                        r = _router()
                        r.execute(
                            "UPDATE inverted_strangle_positions SET entry_iv_rank=?, entry_vix_term_structure=? WHERE id=?",
                            (eir, ets, new_id))
                        r.close()
                    except Exception:
                        pass
                st.session_state.pop("_pending_trade", None)
                st.success(f"Position #{new_id} opened!")
                st.rerun()

    # ── Tab 3: Prediction & Risk ──────────────────────────────────────────────
    with tab_predict:
        st.subheader("Market Conditions & Risk Assessment")

        if st.button("🔄 Refresh All Data", key="refresh_predict"):
            st.cache_data.clear()

        with st.spinner("Loading market data..."):
            iv_data = _fetch_iv_rank()
            ts_data = _fetch_vix_term_structure()
            skew_data = _fetch_market_skew()
            spike_data = _estimate_vix_spike_prob(iv_data, ts_data, skew_data)

        # Row 1: Gauges
        g1, g2 = st.columns(2)
        with g1:
            iv_rank = iv_data.get("iv_rank")
            if iv_rank is not None:
                # For IV Rank: green > 50 (good for selling premium), red < 30
                st.plotly_chart(_gauge_chart(iv_rank, "IV Rank", 100,
                                            green_range=(50, 100), yellow_range=(30, 50), red_range=(0, 30)),
                               use_container_width=True, key="iv_gauge")
                st.caption(f"Current VIX: {iv_data.get('current_iv', 'N/A')} | "
                           f"52w Range: {iv_data.get('low_52w', 'N/A')} - {iv_data.get('high_52w', 'N/A')}")
            else:
                st.metric("IV Rank", "N/A")

        with g2:
            spike_prob = spike_data.get("spike_prob")
            if spike_prob is not None:
                st.plotly_chart(_gauge_chart(spike_prob, "VIX Spike Probability", 100,
                                            green_range=(0, 15), yellow_range=(15, 40), red_range=(40, 100)),
                               use_container_width=True, key="spike_gauge")
            else:
                st.metric("VIX Spike Probability", "N/A")

        # Row 2: Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            ts_pct = ts_data.get("term_structure_pct")
            ts_state = ts_data.get("state", "Unknown")
            color = "normal" if ts_state == "Contango" else "inverse"
            st.metric("VIX Term Structure", ts_state,
                      delta=f"{ts_pct:+.1f}%" if ts_pct is not None else None,
                      delta_color=color)
        with m2:
            st.metric("VIX", f"{ts_data.get('vix', 'N/A')}")
        with m3:
            skew = skew_data.get("skew_index")
            st.metric("CBOE SKEW", f"{skew:.0f}" if skew else "N/A",
                      help=">140 = elevated tail risk")
        with m4:
            pcr = skew_data.get("pcr")
            st.metric("Put/Call Ratio", f"{pcr:.3f}" if pcr else "N/A",
                      help=">1.0 = bearish sentiment")

        # Spike factors breakdown
        factors = spike_data.get("factors", [])
        if factors:
            with st.expander("📊 Spike Probability Factors", expanded=False):
                for f in factors:
                    st.markdown(f"- {f}")

        # Entry signal summary
        st.markdown("---")
        st.markdown("#### Entry Signal Summary")
        signals = []
        iv_rank = iv_data.get("iv_rank")
        if iv_rank is not None:
            signals.append(("IV Rank > 50", iv_rank > 50, f"{iv_rank:.0f}"))
        ts_state = ts_data.get("state", "Unknown")
        signals.append(("VIX in Contango", ts_state == "Contango", ts_state))
        spike_prob = spike_data.get("spike_prob")
        if spike_prob is not None:
            signals.append(("Spike Prob < 15%", spike_prob < 15, f"{spike_prob:.0f}%"))
        skew_val = skew_data.get("skew_index")
        if skew_val is not None:
            signals.append(("SKEW < 140", skew_val < 140, f"{skew_val:.0f}"))

        pass_count = sum(1 for _, ok, _ in signals if ok)
        for label, ok, detail in signals:
            icon = "✅" if ok else "❌"
            st.markdown(f"{icon} **{label}** — {detail}")

        if pass_count == len(signals):
            st.success("🟢 All conditions favorable — good entry window.")
        elif pass_count >= len(signals) - 1:
            st.info("🟡 Most conditions favorable — proceed with caution.")
        else:
            st.warning("🔴 Multiple conditions unfavorable — consider waiting.")

    # ── Tab 4: History & Performance ──────────────────────────────────────────
    with tab_history:
        st.subheader("Closed Positions & Performance")
        closed = _load_positions("CLOSED")
        if closed.empty:
            st.info("No closed positions yet.")
        else:
            total_pnl = closed["current_pnl"].sum()
            wins = (closed["current_pnl"] > 0).sum()
            win_rate = round(wins / len(closed) * 100, 1)
            avg_pnl = closed["current_pnl"].mean()

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total P&L", f"${total_pnl:+.2f}")
            s2.metric("Win Rate", f"{win_rate}%")
            s3.metric("Avg P&L/Trade", f"${avg_pnl:+.2f}")
            s4.metric("Total Trades", len(closed))

            st.plotly_chart(_performance_chart(closed), use_container_width=True, key="perf_chart")

            st.dataframe(
                closed[["trade_date", "underlying", "expiry_date", "initial_credit",
                         "profit_target", "close_price", "current_pnl", "close_reason", "roll_count"]].rename(columns={
                    "trade_date": "Open", "underlying": "Ticker", "expiry_date": "Expiry",
                    "initial_credit": "Credit", "profit_target": "Target", "close_price": "Close",
                    "current_pnl": "P&L", "close_reason": "Reason", "roll_count": "Rolls",
                }),
                use_container_width=True,
            )

    # ── Tab 5: Tracker ────────────────────────────────────────────────────────
    with tab_tracker:
        st.subheader("Performance Tracker")
        closed = _load_positions("CLOSED")
        if closed.empty:
            st.info("No closed positions yet.")
        else:
            period = st.selectbox("Time Period", ["All Time", "Last 90 Days", "Last 30 Days", "Year to Date"])
            if period == "Last 90 Days":
                closed = closed[pd.to_datetime(closed["close_date"]) > datetime.now() - timedelta(days=90)]
            elif period == "Last 30 Days":
                closed = closed[pd.to_datetime(closed["close_date"]) > datetime.now() - timedelta(days=30)]
            elif period == "Year to Date":
                closed = closed[pd.to_datetime(closed["close_date"]).dt.year == datetime.now().year]

            if not closed.empty:
                closed = closed.sort_values("close_date").reset_index(drop=True)
                closed["rolling_pnl"] = closed["current_pnl"].rolling(window=10, min_periods=1).mean()
                closed["rolling_win_rate"] = (closed["current_pnl"] > 0).rolling(window=10, min_periods=1).mean() * 100

                streaks = []
                current_streak = 0
                for pnl_val in closed["current_pnl"]:
                    if pnl_val > 0:
                        current_streak = current_streak + 1 if current_streak >= 0 else 1
                    else:
                        current_streak = current_streak - 1 if current_streak <= 0 else -1
                    streaks.append(current_streak)
                closed["streak"] = streaks

                st.plotly_chart(_rolling_pnl_chart(closed), use_container_width=True, key="roll_pnl")
                st.plotly_chart(_rolling_win_rate_chart(closed), use_container_width=True, key="roll_wr")
                st.plotly_chart(_streak_chart(closed), use_container_width=True, key="streaks")

                st.subheader("Performance by VIX Regime")
                if "vix_at_open" in closed.columns and closed["vix_at_open"].notna().any():
                    closed["vix_regime"] = pd.cut(closed["vix_at_open"].fillna(20),
                                                  bins=[0, 15, 25, 100], labels=["Low", "Medium", "High"])
                    regime_perf = closed.groupby("vix_regime", observed=True)["current_pnl"].agg(["sum", "mean", "count"])
                    st.dataframe(regime_perf)
                else:
                    st.caption("VIX data not available for regime breakdown.")
            else:
                st.info("No positions match the selected time period.")

    # ── Tab 6: Guardrails & C2C ──────────────────────────────────────────────
    with tab_guardrails:
        st.subheader("🛡️ Position Guardrails & Cost-to-Close Monitoring")
        open_pos = _load_positions("OPEN")

        if open_pos.empty:
            st.info("No open positions to monitor.")
        else:
            if st.button("🔄 Refresh C2C for All Positions", key="refresh_c2c"):
                with st.spinner("Computing cost-to-close..."):
                    for _, pos in open_pos.iterrows():
                        pos_id = int(pos["id"])
                        spot_now = _live_spot(pos["underlying"].replace("[PAPER] ", ""))
                        expiry_str = str(pos["expiry_date"])
                        try:
                            expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                            dte_now = max((expiry_dt - datetime.now().date()).days, 1)
                        except Exception:
                            dte_now = 30

                        # Try Polygon first, fall back to theoretical
                        c2c = None
                        try:
                            chain, snap_map = _fetch_chain_for_strike(
                                pos["underlying"].replace("[PAPER] ", ""), expiry_str,
                                [pos["short_put"], pos["short_call"], pos["long_put"], pos["long_call"]],
                                ["put", "call"])
                            if not chain.empty:
                                _sp = snap_map.get(float(pos["short_put"]), float(pos["short_put"]))
                                _sc = snap_map.get(float(pos["short_call"]), float(pos["short_call"]))
                                _lp = snap_map.get(float(pos["long_put"]), float(pos["long_put"]))
                                _lc = snap_map.get(float(pos["long_call"]), float(pos["long_call"]))
                                spread = _price_spread(chain, _sp, _sc, _lp, _lc)
                                c2c = spread["net_credit"]
                        except Exception:
                            pass

                        if c2c is None:
                            # Theoretical fallback
                            try:
                                ts = _fetch_vix_term_structure()
                                vix_now = ts.get("vix") or 20.0
                            except Exception:
                                vix_now = 20.0
                            theo = _theoretical_spread_price(
                                spot_now, float(pos["short_put"]), float(pos["short_call"]),
                                float(pos["long_put"]), float(pos["long_call"]), dte_now, vix_now)
                            c2c = theo["net_credit"]

                        r = _router()
                        # Compute derived guardrail columns
                        inversion_w = float(pos["inversion_pts"]) * 2
                        c2c_intrinsic = inversion_w
                        c2c_extrinsic = round(c2c - c2c_intrinsic, 4) if c2c is not None else None
                        credit_vs_w = round(float(pos["initial_credit"]) - inversion_w, 4)
                        breached_flag = 1 if (c2c is not None and c2c > float(pos["initial_credit"]) * 2) else 0

                        r.execute(
                            "UPDATE inverted_strangle_positions SET cost_to_close=?, c2c_updated_at=?, "
                            "c2c_extrinsic=?, c2c_intrinsic=?, credit_vs_width=?, loss_rule_2_1_breached=? WHERE id=?",
                            (c2c, datetime.now().strftime("%Y-%m-%d %H:%M"),
                             c2c_extrinsic, c2c_intrinsic, credit_vs_w, breached_flag, pos_id))
                        r.close()
                    st.rerun()

            # Display guardrails for each position
            for _, pos in open_pos.iterrows():
                pos_id = int(pos["id"])
                initial_credit = float(pos["initial_credit"])
                inversion_pts = float(pos["inversion_pts"])
                c2c = float(pos["cost_to_close"]) if pos.get("cost_to_close") is not None and pd.notna(pos.get("cost_to_close")) else None

                with st.expander(
                    f"**Position #{pos_id}** — {pos['underlying']} @ ${pos['spot_at_open']:.2f} | "
                    f"Credit ${initial_credit:.2f} | Expiry {pos['expiry_date']}",
                    expanded=True,
                ):
                    c1, c2_col, c3 = st.columns(3)

                    with c1:
                        if c2c is not None:
                            c2c_ext = float(pos.get("c2c_extrinsic") or 0) if pd.notna(pos.get("c2c_extrinsic")) else None
                            c2c_int = float(pos.get("c2c_intrinsic") or 0) if pd.notna(pos.get("c2c_intrinsic")) else None
                            st.metric("Cost to Close (C2C)", f"${c2c:.2f}")
                            if c2c_ext is not None and c2c_int is not None:
                                st.caption(f"Intrinsic: ${c2c_int:.2f} | Extrinsic: ${c2c_ext:.2f}")
                        else:
                            st.metric("Cost to Close (C2C)", "—")
                            st.caption("Click Refresh C2C above")

                    with c2_col:
                        loss_limit = round(initial_credit * 2, 2)
                        breached_db = int(pos.get("loss_rule_2_1_breached") or 0) if pd.notna(pos.get("loss_rule_2_1_breached")) else 0
                        if c2c is not None:
                            breached = c2c > loss_limit or breached_db == 1
                            delta_val = round(c2c - loss_limit, 2)
                            st.metric("2:1 Loss Rule", f"${loss_limit:.2f}",
                                      delta=f"${delta_val:+.2f}",
                                      delta_color="inverse" if breached else "normal")
                            if breached:
                                st.error("🚨 2:1 LOSS RULE BREACHED — close position!")
                        else:
                            st.metric("2:1 Loss Rule", f"${loss_limit:.2f}")

                    with c3:
                        # Credit vs Inversion Width: if credit < inversion width, guaranteed loss at expiry
                        inversion_width = inversion_pts * 2  # total inversion width in dollars
                        cvw_db = float(pos.get("credit_vs_width") or 0) if pd.notna(pos.get("credit_vs_width")) else None
                        credit_vs_width = cvw_db if cvw_db is not None else round(initial_credit - inversion_width, 2)
                        st.metric("Credit vs. Inversion Width", f"${credit_vs_width:+.2f}",
                                  help="Negative = guaranteed loss if underlying stays between short strikes at expiry")
                        if credit_vs_width < 0:
                            st.warning("⚠️ Credit < inversion width — guaranteed loss at expiry if untested.")

                    # P&L status bar
                    if c2c is not None:
                        pnl_now = round(initial_credit - c2c, 2)
                        pct_of_credit = round(pnl_now / initial_credit * 100, 1) if initial_credit > 0 else 0
                        p1, p2, p3 = st.columns(3)
                        with p1:
                            st.metric("Unrealized P&L", f"${pnl_now:+.2f}")
                        with p2:
                            st.metric("% of Credit", f"{pct_of_credit:+.1f}%")
                        with p3:
                            updated = pos.get("c2c_updated_at", "—")
                            st.caption(f"Last updated: {updated}")

                    st.markdown("---")

    # ── Tab 7: What-If Engine ─────────────────────────────────────────────────
    with tab_whatif:
        st.subheader("🤔 Defensive Adjustment What-If Engine")
        open_pos = _load_positions("OPEN")

        if open_pos.empty:
            st.info("No open positions to simulate adjustments on.")
        else:
            pos_options = {int(p["id"]): f"#{p['id']} — {p['underlying']} @ ${p['spot_at_open']:.2f} (Expiry {p['expiry_date']})"
                          for _, p in open_pos.iterrows()}
            selected_id = st.selectbox("Select Position", list(pos_options.keys()),
                                       format_func=lambda x: pos_options[x], key="whatif_pos")
            pos = open_pos[open_pos["id"] == selected_id].iloc[0]

            inv_pts = float(pos["inversion_pts"])
            wing_pts = float(pos["wing_pts"])
            orig_credit = float(pos["initial_credit"])

            # Current position summary
            st.markdown("**Current Position:**")
            cc1, cc2, cc3, cc4, cc5 = st.columns(5)
            cc1.metric("Long Put", f"${pos['long_put']:.2f}")
            cc2.metric("Short Put", f"${pos['short_put']:.2f}")
            cc3.metric("Short Call", f"${pos['short_call']:.2f}")
            cc4.metric("Long Call", f"${pos['long_call']:.2f}")
            cc5.metric("Credit", f"${orig_credit:.2f}")

            st.markdown("---")

            scenario = st.radio("Scenario", ["Roll to New Spot", "Adjust Wing Width", "Un-invert (go OTM)"],
                                horizontal=True, key="whatif_scenario")

            # Get current VIX for theoretical pricing
            try:
                ts_data = _fetch_vix_term_structure()
                vix_now = ts_data.get("vix") or 20.0
            except Exception:
                vix_now = 20.0

            if scenario == "Roll to New Spot":
                with st.form("whatif_roll"):
                    st.markdown("Simulate rolling the entire position to a new spot price.")
                    live_price = _live_spot(pos["underlying"].replace("[PAPER] ", ""))
                    new_spot = st.number_input("New Spot Price", 100.0, 1000.0, live_price, 0.01, key="wi_spot")
                    new_dte = st.slider("New DTE", 25, 60, 45, key="wi_dte")
                    run_roll = st.form_submit_button("▶️ Simulate Roll")

                if run_roll:
                    new_sp = round(new_spot + inv_pts, 2)
                    new_sc = round(new_spot - inv_pts, 2)
                    new_lp = round(new_sp - wing_pts, 2)
                    new_lc = round(new_sc + wing_pts, 2)

                    new_spread = _theoretical_spread_price(new_spot, new_sp, new_sc, new_lp, new_lc, new_dte, vix_now)
                    new_credit = new_spread["net_credit"]

                    # Estimate cost to close current (theoretical)
                    try:
                        expiry_dt = datetime.strptime(str(pos["expiry_date"]), "%Y-%m-%d").date()
                        dte_remaining = max((expiry_dt - datetime.now().date()).days, 1)
                    except Exception:
                        dte_remaining = 30
                    old_spread = _theoretical_spread_price(
                        new_spot, float(pos["short_put"]), float(pos["short_call"]),
                        float(pos["long_put"]), float(pos["long_call"]), dte_remaining, vix_now)
                    close_cost = old_spread["net_credit"]
                    net_roll = round(new_credit - close_cost, 2)

                    st.markdown("### Roll Simulation Results")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("New Credit", f"${new_credit:.2f}")
                    r2.metric("Close Cost (current)", f"${close_cost:.2f}")
                    r3.metric("Net Roll Credit/Debit", f"${net_roll:+.2f}",
                              delta="Credit" if net_roll > 0 else "Debit",
                              delta_color="normal" if net_roll > 0 else "inverse")

                    st.markdown("**New Strikes:**")
                    n1, n2, n3, n4 = st.columns(4)
                    n1.metric("Long Put", f"${new_lp:.2f}")
                    n2.metric("Short Put", f"${new_sp:.2f}")
                    n3.metric("Short Call", f"${new_sc:.2f}")
                    n4.metric("Long Call", f"${new_lc:.2f}")

                    preview = pd.Series({
                        "long_put": new_lp, "short_put": new_sp,
                        "short_call": new_sc, "long_call": new_lc,
                        "initial_credit": new_credit,
                        "profit_target": round(new_credit * 0.9, 2),
                    })
                    st.plotly_chart(_pnl_curve(preview), use_container_width=True, key="whatif_pnl_roll")
                    st.caption(f"Theoretical pricing @ VIX={vix_now:.1f}")

            elif scenario == "Adjust Wing Width":
                with st.form("whatif_wing"):
                    st.markdown("Simulate widening or narrowing the protective wings.")
                    new_wing = st.number_input("New Wing Width ($)", 10.0, 60.0, wing_pts, 1.0, key="wi_wing")
                    run_wing = st.form_submit_button("▶️ Simulate Wing Adjustment")

                if run_wing:
                    sp = float(pos["short_put"])
                    sc = float(pos["short_call"])
                    new_lp = round(sp - new_wing, 2)
                    new_lc = round(sc + new_wing, 2)

                    try:
                        expiry_dt = datetime.strptime(str(pos["expiry_date"]), "%Y-%m-%d").date()
                        dte_remaining = max((expiry_dt - datetime.now().date()).days, 1)
                    except Exception:
                        dte_remaining = 30

                    spot_now = _live_spot(pos["underlying"].replace("[PAPER] ", ""))
                    new_spread = _theoretical_spread_price(spot_now, sp, sc, new_lp, new_lc, dte_remaining, vix_now)
                    old_spread = _theoretical_spread_price(
                        spot_now, sp, sc, float(pos["long_put"]), float(pos["long_call"]), dte_remaining, vix_now)

                    st.markdown("### Wing Adjustment Results")
                    w1, w2, w3 = st.columns(3)
                    w1.metric("Original Wing", f"${wing_pts:.0f}", help=f"LP ${pos['long_put']:.2f} / LC ${pos['long_call']:.2f}")
                    w2.metric("New Wing", f"${new_wing:.0f}", help=f"LP ${new_lp:.2f} / LC ${new_lc:.2f}")
                    credit_diff = round(new_spread["net_credit"] - old_spread["net_credit"], 2)
                    w3.metric("Credit Change", f"${credit_diff:+.2f}",
                              help="Wider wings = more credit but more risk")

                    max_risk_old = round(wing_pts - orig_credit, 2)
                    max_risk_new = round(new_wing - new_spread["net_credit"], 2)
                    rr1, rr2 = st.columns(2)
                    rr1.metric("Max Risk (current)", f"${max_risk_old:.2f}")
                    rr2.metric("Max Risk (new)", f"${max_risk_new:.2f}")

                    preview = pd.Series({
                        "long_put": new_lp, "short_put": sp,
                        "short_call": sc, "long_call": new_lc,
                        "initial_credit": new_spread["net_credit"],
                        "profit_target": round(new_spread["net_credit"] * 0.9, 2),
                    })
                    st.plotly_chart(_pnl_curve(preview), use_container_width=True, key="whatif_pnl_wing")

            elif scenario == "Un-invert (go OTM)":
                with st.form("whatif_uninvert"):
                    st.markdown(
                        "Simulate closing the inverted (ITM) short strikes and selling a new OTM strangle. "
                        "This restores extrinsic value and reduces assignment risk."
                    )
                    otm_offset = st.number_input("OTM Offset from Spot ($)", 5.0, 50.0, 20.0, 1.0, key="wi_otm",
                                                 help="New short strikes will be Spot ± this value (OTM)")
                    new_dte = st.slider("New Strangle DTE", 25, 60, 45, key="wi_uninvert_dte")
                    run_uninvert = st.form_submit_button("▶️ Simulate Un-inversion")

                if run_uninvert:
                    spot_now = _live_spot(pos["underlying"].replace("[PAPER] ", ""))
                    new_sp_otm = round(spot_now - otm_offset, 2)  # OTM put below spot
                    new_sc_otm = round(spot_now + otm_offset, 2)  # OTM call above spot

                    # Cost to close current inverted position
                    try:
                        expiry_dt = datetime.strptime(str(pos["expiry_date"]), "%Y-%m-%d").date()
                        dte_remaining = max((expiry_dt - datetime.now().date()).days, 1)
                    except Exception:
                        dte_remaining = 30
                    close_spread = _theoretical_spread_price(
                        spot_now, float(pos["short_put"]), float(pos["short_call"]),
                        float(pos["long_put"]), float(pos["long_call"]), dte_remaining, vix_now)
                    close_cost = close_spread["net_credit"]

                    # New OTM strangle credit (no wings — naked or with wide wings)
                    new_lp_otm = round(new_sp_otm - wing_pts, 2)
                    new_lc_otm = round(new_sc_otm + wing_pts, 2)
                    new_spread = _theoretical_spread_price(
                        spot_now, new_sp_otm, new_sc_otm, new_lp_otm, new_lc_otm, new_dte, vix_now)
                    new_credit = new_spread["net_credit"]
                    net_cost = round(close_cost - new_credit, 2)

                    st.markdown("### Un-inversion Results")
                    u1, u2, u3 = st.columns(3)
                    u1.metric("Close Current (debit)", f"${close_cost:.2f}")
                    u2.metric("New OTM Credit", f"${new_credit:.2f}")
                    u3.metric("Net Cost", f"${net_cost:+.2f}",
                              delta="Debit" if net_cost > 0 else "Credit",
                              delta_color="inverse" if net_cost > 0 else "normal")

                    st.markdown("**New OTM Strikes:**")
                    o1, o2, o3, o4 = st.columns(4)
                    o1.metric("Long Put", f"${new_lp_otm:.2f}")
                    o2.metric("Short Put (OTM)", f"${new_sp_otm:.2f}")
                    o3.metric("Short Call (OTM)", f"${new_sc_otm:.2f}")
                    o4.metric("Long Call", f"${new_lc_otm:.2f}")

                    preview = pd.Series({
                        "long_put": new_lp_otm, "short_put": new_sp_otm,
                        "short_call": new_sc_otm, "long_call": new_lc_otm,
                        "initial_credit": new_credit,
                        "profit_target": round(new_credit * 0.9, 2),
                    })
                    st.plotly_chart(_pnl_curve(preview), use_container_width=True, key="whatif_pnl_uninvert")
                    st.caption(f"Theoretical pricing @ VIX={vix_now:.1f}")

    # ── Tab 8: Strategy Guide ─────────────────────────────────────────────────
    with tab_guide:
        st.markdown("""
## Inverted Strangle with Defined Risk — Quick Reference

### Trade Architecture

| Leg | Strike | Type | Action |
| :--- | :--- | :--- | :--- |
| Long Put (Wing) | Spot - Inversion - Wing | OTM Put | Buy |
| Short Put (Core) | Spot + Inversion | ITM Put | Sell |
| Short Call (Core) | Spot - Inversion | ITM Call | Sell |
| Long Call (Wing) | Spot - Inversion + Wing | OTM Call | Buy |

Default parameters: Inversion = $5 | Wing = $25 | DTE = 30-45 | Profit Target = 10%

### Execution Rules

**Entry:** Sell the spread when IV Rank > 30% and VIX is stable or declining.

**Profit Exit:** GTC limit order to buy back at 90% of initial credit (capture 10%).

**Adjustment (Roll):** If price moves within $5 of a long wing, close and re-open at new spot.

**Stop Loss:** If spread value exceeds 200% of initial credit, close to limit losses.

### Greek Characteristics

| Greek | Position | Implication |
| :--- | :--- | :--- |
| Theta | Positive | Time decay works in your favor |
| Vega | Negative | Benefits from IV crush or stable prices |
| Delta | Near Zero | Delta-neutral at entry |
| Gamma | Negative | Risk accelerates near short strikes inside 14 DTE |

### Risk Profile

Max Loss = Wing Width - Net Credit (e.g. $25 - $12.50 = $12.50/share = $1,250/contract)

Max Profit = Full net credit (underlying expires between short strikes)

---

## Guardrails & Risk Rules

### 2:1 Loss Rule
If the cost to close (C2C) exceeds 2× the initial credit received, close the position immediately. This caps your maximum realized loss at 2× credit.

### Credit vs. Inversion Width
At entry, compare your net credit to the inversion width (2 × inversion points). If credit < inversion width, the position has a guaranteed loss if the underlying stays between the short strikes at expiration. This is acceptable only if you expect the underlying to move beyond one of the short strikes.

### Cost-to-Close (C2C) Monitoring
C2C = current market price to buy back all four legs. Track this daily:
- C2C < 90% of credit → take profit (10% capture target)
- C2C > 150% of credit → consider rolling or adjusting
- C2C > 200% of credit → 2:1 loss rule triggered, close immediately

### Defensive Adjustments
1. **Roll**: Close current position, re-open at new spot with same inversion/wing parameters
2. **Wing Adjustment**: Widen wings for more protection (costs credit) or narrow for more credit (more risk)
3. **Un-invert**: Close the ITM inverted strikes, sell new OTM strangle — restores extrinsic value, reduces assignment risk
        """)

    # ── Tab 9: Strategy Deep-Dive ─────────────────────────────────────────────
    with tab_deepdive:
        st.markdown("""
## 📜 Defined-Risk Inverted Strangle — Complete Strategy Deep-Dive

---

### 1. Strategy Overview & Philosophy

The **Inverted Strangle with Defined Risk** is a premium-selling strategy that profits from time decay (theta)
while maintaining a defined maximum loss through protective wings. Unlike a traditional short strangle where
short strikes are OTM, this strategy *inverts* the short strikes ITM, creating a position that collects
significantly more premium at the cost of a guaranteed intrinsic loss component.

**Core Thesis:** The large upfront credit from selling ITM options more than compensates for the intrinsic
value that must be returned, provided the underlying moves enough to test one side or time decay erodes the
extrinsic value sufficiently.

**When to Use:**
- High implied volatility environments (IV Rank > 30%)
- Expectation of mean reversion or range-bound behavior
- When you want defined risk (unlike naked strangles)
- When you want higher probability of profit than directional trades

---

### 2. Position Construction

#### Leg-by-Leg Breakdown

Given a spot price S, inversion width I (default $5), and wing width W (default $25):

| Leg | Strike Formula | Type | Action | Purpose |
|-----|---------------|------|--------|---------|
| Long Put | S + I - W | OTM Put | Buy | Downside protection (wing) |
| Short Put | S + I | ITM Put | Sell | Premium collection |
| Short Call | S - I | ITM Call | Sell | Premium collection |
| Long Call | S - I + W | OTM Call | Buy | Upside protection (wing) |

**Example** (SPY @ $570):
- Long Put: $550 (buy) — Short Put: $575 (sell, ITM by $5)
- Short Call: $565 (sell, ITM by $5) — Long Call: $590 (buy)

#### Credit Composition

The initial credit has two components:
1. **Intrinsic Value** = 2 × Inversion Points (guaranteed to be returned if underlying stays between short strikes)
2. **Extrinsic Value** = Total Credit - Intrinsic Value (this is your *real* edge — pure time value)

**Key Insight:** Your true profit potential is the extrinsic portion. The intrinsic portion is "borrowed"
and must be returned unless the underlying moves beyond a short strike.

---

### 3. P&L Profile

- **Max Profit** = Net Credit Received (underlying expires beyond either short strike)
- **Max Loss** = Wing Width - Net Credit (underlying expires exactly between short strikes)
- **Breakeven Points**: Short Put - Credit and Short Call + Credit

---

### 4. Greek Analysis

| Greek | Position | Implication |
|-------|----------|-------------|
| **Theta** | Positive | Time decay works in your favor. Accelerates inside 21 DTE. |
| **Vega** | Negative | Benefits from IV crush. Enter when IV Rank > 30%. |
| **Delta** | Near Zero | Delta-neutral at entry. Adjust if delta exceeds ±0.30. |
| **Gamma** | Negative | Risk accelerates near short strikes inside 14 DTE. Be cautious. |

**Rules:**
- Enter with 30-45 DTE to capture the theta acceleration curve
- A VIX spike increases C2C — monitor VIX term structure
- Be extra cautious inside 14 DTE — consider closing early if P&L target not met

---

### 5. Entry Criteria & Pre-Trade Checklist

#### Mandatory Checks (all must pass)
1. IV Rank > 30% — ensures sufficient premium
2. VIX Term Structure in Contango — VIX < VIX3M (no near-term panic)
3. VIX Spike Probability < 15% — heuristic check for imminent volatility events
4. DTE: 30-45 days — optimal theta decay window
5. Credit > Inversion Width — ensures positive expected value at entry

#### Preferred Conditions
- IV Rank > 50% (ideal)
- No major earnings/FOMC within 7 days
- SPY not at 52-week extremes
- Put/Call ratio < 1.5 (no extreme fear)

---

### 6. Position Sizing Framework

| Level | Max Portfolio Risk | Notes |
|-------|-------------------|-------|
| Conservative | 2% per position | Start here |
| Moderate | 5% per position | After 10+ trades |
| Aggressive | 10% per position | Experienced only |

- Max risk per trade = Wing Width - Net Credit
- Start with 1 contract until you have 10+ trades of history
- Never exceed 5 contracts on a single underlying

---

### 7. Exit Rules & Profit Targets

| Exit Type | Trigger | Action |
|-----------|---------|--------|
| Profit Target | C2C < 90% of credit | Close (capture 10%) |
| Time-Based | 21 DTE, P&L positive | Close for whatever profit exists |
| Gamma Risk | 14 DTE, P&L negative | Close to avoid gamma acceleration |
| Stop Loss | C2C > 2× credit | Close immediately (2:1 rule) |
| **Never** hold to expiration | — | Assignment risk and pin risk |

---

### 8. Adjustment Decision Tree

**Position Open → Check in order:**

1. P&L > 10% target? → **CLOSE** (take profit)
2. DTE < 21? → **CLOSE** (time-based exit)
3. C2C > 2× credit? → **CLOSE** (stop loss)
4. Delta > ±0.30? → **ROLL** to new spot or un-invert to OTM
5. Price within $5 of wing? → **ROLL** or widen wings
6. None of the above → **HOLD** (let theta work)

#### Roll Mechanics
1. Buy back current 4-leg spread (debit)
2. Sell new 4-leg spread at current spot (credit)
3. Net cost = Close Debit - New Credit
4. **Max 2 rolls per position** — after that, take the loss

#### Wing Adjustment
- Widen wings: More protection, less credit (use when vol rising)
- Narrow wings: Less protection, more credit (use when vol falling)
- **Never narrow wings below $15**

#### Un-Inversion (Emergency)
- Close ITM inverted strikes, sell new OTM strangle at current spot
- Restores extrinsic value, eliminates intrinsic drag
- Use when position is deeply underwater and you want to reset

---

### 9. Market Regime Considerations

| Regime | IV Rank | Action |
|--------|---------|--------|
| Low Vol Range | < 20% | **Avoid** — insufficient premium |
| Normal | 20-40% | **Selective** — strong setup only |
| Elevated | 40-60% | **Ideal** — best risk/reward |
| High Vol | 60-80% | **Aggressive** — wide wings, smaller size |
| Crisis | > 80% | **Avoid** — spreads too wide, assignment risk |

**VIX Term Structure:**
- Contango (VIX < VIX3M): Normal — safe to enter
- Flat: Caution — uncertainty rising
- Backwardation (VIX > VIX3M): Danger — avoid new entries

---

### 10. Cost-to-Close (C2C) Deep Dive

| C2C Level | Signal | Action |
|-----------|--------|--------|
| < 90% of credit | 🟢 Profit target | Close for profit |
| 90-120% of credit | 🟡 Neutral | Hold, let theta work |
| 120-150% of credit | 🟠 Warning | Monitor closely |
| 150-200% of credit | 🔴 Danger | Roll or adjust |
| > 200% of credit | 🚨 Stop loss | Close immediately |

**C2C Components:**
- **C2C Intrinsic** = Inversion Width × 2 (guaranteed loss portion)
- **C2C Extrinsic** = C2C - Intrinsic (time value — what theta erodes)
- **Credit vs. Width** = Initial Credit - (Inversion Pts × 2). Positive = edge even if untested.

---

### 11. Paper Trading Protocol

| Phase | Duration | Method | Focus |
|-------|----------|--------|-------|
| Phase 1 | Weeks 1-4 | Black-Scholes theoretical | Entry/exit discipline |
| Phase 2 | Weeks 5-8 | Live yfinance quotes | Real spreads & slippage |
| Phase 3 | Week 9+ | Live with 1 contract | Strict guardrail adherence |

**Go-Live Criteria:** Win rate > 60% over 20+ paper trades, average winner > average loser, no 2:1 breaches.

---

### 12. Common Mistakes to Avoid

1. **Holding through expiration** — assignment risk, pin risk, gamma explosion
2. **Ignoring the 2:1 rule** — "it'll come back" is how accounts blow up
3. **Trading in low IV** — insufficient premium makes the math unfavorable
4. **Over-sizing** — one bad trade shouldn't impact more than 5% of portfolio
5. **Rolling too many times** — max 2 rolls; after that, take the loss
6. **Entering before earnings/FOMC** — IV crush is your friend *after* the event, not before
7. **Neglecting C2C monitoring** — check daily, not weekly
8. **Narrowing wings under pressure** — reduces protection when you need it most
        """)
