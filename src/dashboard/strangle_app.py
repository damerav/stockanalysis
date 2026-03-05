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
        return f"<b>{label}:</b> {value}"


def _header(title: str):
    """Render page header directly via st.markdown."""
    try:
        from src.dashboard.template import page_header
        page_header(title)
    except Exception:
        st.markdown(f"## {title}")


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


def _fetch_chain_for_strike(
    underlying: str, expiry: str, strikes: list[float], option_types: list[str],
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

@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
def _fetch_market_skew() -> dict:
    """Fetch CBOE SKEW index and compute put/call ratio from Polygon."""
    result = {"skew_index": None, "pcr": None}
    try:
        import yfinance as yf
        skew = yf.Ticker("^SKEW").history(period="5d")
        if not skew.empty:
            result["skew_index"] = round(float(skew["Close"].iloc[-1]), 2)
    except Exception:
        pass
    try:
        poly = _polygon()
        analytics = poly.get_options_analytics("SPY")
        result["pcr"] = round(analytics.get("put_call_ratio") or 0, 3)
    except Exception:
        pass
    return result


def _estimate_vix_spike_prob() -> dict:
    """Heuristic VIX spike probability based on current conditions.

    Uses a weighted score from: VIX level, term structure, SKEW, and VIX velocity.
    Not an ML model — a rules-based approximation that captures the same signals.
    """
    try:
        iv_data = _fetch_iv_rank()
        ts_data = _fetch_vix_term_structure()
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
    _header("🕷️ Inverted Strangle — Defined Risk")

    tab_open, tab_new, tab_predict, tab_history, tab_tracker, tab_guide = st.tabs([
        "📋 Open Positions", "➕ New Trade", "🔮 Prediction & Risk",
        "📊 History & Performance", "📈 Tracker", "📖 Strategy Guide",
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

                    with st.expander("📐 Live Greeks (fetches from Polygon)", expanded=False):
                        if st.button("Refresh Greeks", key=f"greeks_{pos_id}"):
                            with st.spinner("Fetching options chain..."):
                                chain = _fetch_chain_for_strike(
                                    pos["underlying"], pos["expiry_date"],
                                    [pos["short_put"], pos["short_call"], pos["long_put"], pos["long_call"]],
                                    ["put", "call"],
                                )
                                if not chain.empty:
                                    greeks = _get_greeks(chain, float(pos["short_put"]), float(pos["short_call"]),
                                                         float(pos["long_put"]), float(pos["long_call"]))
                                    g1, g2, g3, g4 = st.columns(4)
                                    with g1: st.metric("Net Delta", f"{greeks['delta']:+.4f}")
                                    with g2: st.metric("Net Gamma", f"{greeks['gamma']:+.4f}")
                                    with g3: st.metric("Net Theta", f"{greeks['theta']:+.4f}")
                                    with g4: st.metric("Net Vega", f"{greeks['vega']:+.4f}")
                                    st.plotly_chart(_greeks_gauge(greeks), use_container_width=True, key=f"gg_{pos_id}")

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
                    spike_data = _estimate_vix_spike_prob()

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
        sim_mode = st.toggle("🧪 Simulation Mode (theoretical pricing, no Polygon needed)",
                             value=True, key="sim_mode_toggle")
        if sim_mode:
            st.caption("Using Black-Scholes theoretical pricing with VIX as IV proxy. Trades are paper/simulated.")

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

            st.markdown("### Proposed Trade Structure")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Long Put (Wing)", f"${long_put:.2f}")
            col2.metric("Short Put (ITM)", f"${short_put:.2f}")
            col3.metric("Short Call (ITM)", f"${short_call:.2f}")
            col4.metric("Long Call (Wing)", f"${long_call:.2f}")
            st.caption(f"Expiry: {expiry} ({dte} DTE) | Inversion: ${inversion} | Wing: ${wing}")

            credit_info = {}
            if manual_credit > 0:
                credit_info = {"net_credit": manual_credit, "credit_per_leg_str": "Manual override"}
            elif sim_mode:
                # Simulation mode: use Black-Scholes theoretical pricing
                try:
                    ts_data = _fetch_vix_term_structure()
                    vix_now = ts_data.get("vix") or 20.0
                except Exception:
                    vix_now = 20.0
                credit_info = _theoretical_spread_price(spot, short_put, short_call, long_put, long_call, dte, vix_now)
                st.info(f"🧪 Theoretical pricing @ VIX={vix_now:.1f} (simulation mode)")
            else:
                with st.spinner("Fetching options chain from Polygon..."):
                    chain = _fetch_chain_for_strike(underlying, expiry,
                                                    [short_put, short_call, long_put, long_call], ["put", "call"])
                    if not chain.empty:
                        credit_info = _price_spread(chain, short_put, short_call, long_put, long_call)
                    else:
                        # Auto-fallback to theoretical pricing
                        st.warning("Polygon chain unavailable — falling back to theoretical pricing.")
                        try:
                            ts_data = _fetch_vix_term_structure()
                            vix_now = ts_data.get("vix") or 20.0
                        except Exception:
                            vix_now = 20.0
                        credit_info = _theoretical_spread_price(spot, short_put, short_call, long_put, long_call, dte, vix_now)

            if credit_info:
                net_credit = credit_info["net_credit"]
                profit_target_price = round(net_credit * (1 - profit_target_pct / 100), 2)

                st.markdown("### Pricing")
                p1, p2, p3 = st.columns(3)
                p1.metric("Net Credit Received", f"${net_credit:.2f}")
                p2.metric(f"Profit Target ({profit_target_pct}%)", f"${profit_target_price:.2f}")
                p3.metric("Max Risk", f"${wing - net_credit:.2f}")
                st.caption(credit_info.get("credit_per_leg_str", ""))

                preview_pos = pd.Series({
                    "long_put": long_put, "short_put": short_put,
                    "short_call": short_call, "long_call": long_call,
                    "initial_credit": net_credit, "profit_target": profit_target_price,
                })
                st.plotly_chart(_pnl_curve(preview_pos), use_container_width=True, key="pnl_preview")

                # Capture entry conditions for the position record
                entry_iv_rank = None
                entry_ts = None
                entry_vix = None
                try:
                    iv_data = _fetch_iv_rank()
                    ts_data = _fetch_vix_term_structure()
                    entry_iv_rank = iv_data.get("iv_rank")
                    entry_ts = ts_data.get("term_structure_pct")
                    entry_vix = ts_data.get("vix")
                except Exception:
                    pass

                if st.button("✅ Confirm & Open Position"):
                    # Tag paper trades with [PAPER] prefix
                    trade_underlying = f"[PAPER] {underlying}" if sim_mode else underlying
                    new_id = _open_position(
                        trade_underlying, spot, expiry, dte, short_put, short_call, long_put, long_call,
                        inversion, wing, net_credit, credit_info.get("credit_per_leg_str", ""),
                        profit_target_pct, entry_vix or 0.0,
                    )
                    # Store entry conditions
                    if entry_iv_rank is not None or entry_ts is not None:
                        try:
                            r = _router()
                            r.execute(
                                "UPDATE inverted_strangle_positions SET entry_iv_rank=?, entry_vix_term_structure=? WHERE id=?",
                                (entry_iv_rank, entry_ts, new_id))
                            r.close()
                        except Exception:
                            pass
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
            spike_data = _estimate_vix_spike_prob()

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

    # ── Tab 6: Strategy Guide ─────────────────────────────────────────────────
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
        """)
