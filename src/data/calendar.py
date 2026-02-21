"""Economic calendar — FOMC, CPI, NFP, OpEx dates for event-aware features."""

import logging
from datetime import date, timedelta
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Third Friday of each month (options expiration)
def _third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given month/year."""
    d = date(year, month, 15)
    w = d.weekday()
    if w <= 4:  # Mon-Fri
        d += timedelta(days=(4 - w))
    else:  # Sat or Sun
        d += timedelta(days=(11 - w))
    return d

# FOMC meeting dates (2025-2026, last day of each 2-day meeting)
# Source: federalreserve.gov — update annually
FOMC_DATES = [
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

# CPI release dates (typically 2nd or 3rd Tuesday/Wednesday of month)
# Source: bls.gov — update annually
CPI_DATES = [
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
    "2025-05-13", "2025-06-11", "2025-07-11", "2025-08-12",
    "2025-09-10", "2025-10-14", "2025-11-12", "2025-12-10",
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-14",
    "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-10", "2026-10-13", "2026-11-10", "2026-12-10",
]

# NFP release dates (typically 1st Friday of month)
NFP_DATES = [
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04",
    "2025-05-02", "2025-06-06", "2025-07-03", "2025-08-01",
    "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]

# Pre-parsed date objects for fast lookup
_FOMC = [date.fromisoformat(d) for d in FOMC_DATES]
_CPI = [date.fromisoformat(d) for d in CPI_DATES]
_NFP = [date.fromisoformat(d) for d in NFP_DATES]


def days_until_next(event_dates: list[date], from_date: date) -> int:
    """Days until the next event on or after from_date. Returns 999 if none found."""
    for d in event_dates:
        if d >= from_date:
            return (d - from_date).days
    return 999


def _opex_date(from_date: date) -> date:
    """Next monthly options expiration (3rd Friday) on or after from_date."""
    y, m = from_date.year, from_date.month
    opex = _third_friday(y, m)
    if opex < from_date:
        m += 1
        if m > 12:
            m, y = 1, y + 1
        opex = _third_friday(y, m)
    return opex


def _is_triple_witching(from_date: date) -> bool:
    """True if from_date falls in a triple-witching week (3rd Friday of Mar/Jun/Sep/Dec)."""
    if from_date.month not in (3, 6, 9, 12):
        return False
    tw = _third_friday(from_date.year, from_date.month)
    # Within the same week (Mon-Fri containing the 3rd Friday)
    week_start = tw - timedelta(days=tw.weekday())
    return week_start <= from_date <= tw


def get_event_features(from_date: Union[date, None] = None) -> dict:
    """Compute calendar/event features for a given date.

    Returns dict with:
        days_to_fomc, is_fomc_week, is_fomc_day,
        days_to_cpi, days_to_nfp, days_to_opex,
        is_triple_witching, is_quarter_end,
        day_of_week (0=Mon..4=Fri), week_of_month (1-5)
    """
    if from_date is None:
        from_date = date.today()

    days_fomc = days_until_next(_FOMC, from_date)
    days_cpi = days_until_next(_CPI, from_date)
    days_nfp = days_until_next(_NFP, from_date)
    opex = _opex_date(from_date)
    days_opex = (opex - from_date).days

    return {
        "days_to_fomc": days_fomc,
        "is_fomc_week": int(days_fomc <= 5),
        "is_fomc_day": int(days_fomc == 0),
        "days_to_cpi": days_cpi,
        "days_to_nfp": days_nfp,
        "days_to_opex": days_opex,
        "is_triple_witching": int(_is_triple_witching(from_date)),
        "is_quarter_end": int(from_date.month in (3, 6, 9, 12)
                              and from_date.day >= 25),
        "day_of_week": from_date.weekday(),
        "week_of_month": (from_date.day - 1) // 7 + 1,
    }


def has_nearby_event(from_date: date, window: int = 2) -> bool:
    """True if FOMC, CPI, or NFP is within `window` trading days."""
    return (days_until_next(_FOMC, from_date) <= window
            or days_until_next(_CPI, from_date) <= window
            or days_until_next(_NFP, from_date) <= window)
