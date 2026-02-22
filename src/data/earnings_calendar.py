"""P3: Earnings Calendar Integration — S&P 500 earnings dates and features.

Fetches earnings dates from Alpha Vantage (free tier) and yfinance.
Computes features: earnings_density (% of SPX market cap reporting this week),
days_to_next_mega_earnings, earnings_surprise_momentum.
"""

import logging
import sqlite3
from datetime import datetime, timedelta, date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Top 20 SPX constituents by weight (mega-caps that move the index)
MEGA_CAPS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "AVGO", "LLY",
    "MRK", "COST",
]


def fetch_earnings_yf(tickers: list[str] = None) -> list[dict]:
    """Fetch upcoming earnings dates from yfinance."""
    tickers = tickers or MEGA_CAPS
    results = []
    try:
        import yfinance as yf
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                cal = t.calendar
                if cal is not None and not cal.empty if hasattr(cal, 'empty') else cal:
                    if isinstance(cal, dict):
                        earn_date = cal.get("Earnings Date", [None])[0]
                        eps_est = cal.get("Earnings Average", None)
                    elif isinstance(cal, pd.DataFrame):
                        earn_date = cal.iloc[0, 0] if len(cal.columns) > 0 else None
                        eps_est = None
                    else:
                        continue
                    if earn_date:
                        if hasattr(earn_date, 'strftime'):
                            earn_date = earn_date.strftime("%Y-%m-%d")
                        results.append({
                            "ticker": ticker,
                            "date": str(earn_date),
                            "eps_estimate": eps_est,
                        })
            except Exception:
                continue
    except ImportError:
        logger.warning("yfinance not available for earnings calendar")
    return results


def store_earnings(conn: sqlite3.Connection, earnings: list[dict]):
    """Store earnings calendar data."""
    for e in earnings:
        conn.execute(
            """INSERT OR REPLACE INTO earnings_calendar
               (date, ticker, eps_estimate)
               VALUES (?, ?, ?)""",
            (e["date"], e["ticker"], e.get("eps_estimate")),
        )
    conn.commit()


def get_earnings_features(conn: sqlite3.Connection, target_date: str) -> dict:
    """Compute earnings-related features for a given date.

    Returns:
        earnings_density: Number of mega-caps reporting within ±3 days
        days_to_next_mega: Days until next mega-cap earnings
        earnings_week: 1 if any mega-cap reports this week
    """
    try:
        d = datetime.strptime(target_date, "%Y-%m-%d").date()
    except Exception:
        return {"earnings_density": 0, "days_to_next_mega": 30, "earnings_week": 0}

    # Count mega-caps reporting within ±3 days
    window_start = (d - timedelta(days=3)).strftime("%Y-%m-%d")
    window_end = (d + timedelta(days=3)).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT COUNT(*) FROM earnings_calendar WHERE date BETWEEN ? AND ?",
        (window_start, window_end),
    ).fetchone()
    density = row[0] if row else 0

    # Days to next mega-cap earnings
    next_row = conn.execute(
        "SELECT MIN(date) FROM earnings_calendar WHERE date >= ?",
        (target_date,),
    ).fetchone()
    if next_row and next_row[0]:
        try:
            next_date = datetime.strptime(next_row[0], "%Y-%m-%d").date()
            days_to_next = (next_date - d).days
        except Exception:
            days_to_next = 30
    else:
        days_to_next = 30

    # Is this an earnings week?
    week_start = (d - timedelta(days=d.weekday())).strftime("%Y-%m-%d")
    week_end = (d + timedelta(days=4 - d.weekday())).strftime("%Y-%m-%d")
    week_row = conn.execute(
        "SELECT COUNT(*) FROM earnings_calendar WHERE date BETWEEN ? AND ?",
        (week_start, week_end),
    ).fetchone()
    earnings_week = 1 if (week_row and week_row[0] > 0) else 0

    return {
        "earnings_density": density,
        "days_to_next_mega": min(days_to_next, 30),
        "earnings_week": earnings_week,
    }
