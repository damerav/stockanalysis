"""P3: Federal Reserve Communication NLP — FOMC + Beige Book sentiment.

Fetches FOMC statements and Beige Book from the Fed website,
scores hawkish/dovish sentiment using the LLM, and caches results.
Only runs ~8 times per year (after each FOMC meeting).
"""

import logging
import sqlite3
import requests
import feedparser
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

FED_RSS_URL = "https://www.federalreserve.gov/feeds/press_monetary.xml"
BEIGE_BOOK_RSS = "https://www.federalreserve.gov/feeds/beigebook.xml"


def fetch_latest_fomc_statement() -> Optional[dict]:
    """Fetch the most recent FOMC statement from the Fed RSS feed."""
    try:
        feed = feedparser.parse(FED_RSS_URL)
        for entry in feed.entries[:5]:
            title = entry.get("title", "").lower()
            if "statement" in title or "federal open market" in title:
                summary = entry.get("summary", entry.get("description", ""))
                pub_date = entry.get("published", "")
                try:
                    dt = datetime(*entry.published_parsed[:6])
                    date_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    date_str = datetime.now().strftime("%Y-%m-%d")
                return {
                    "date": date_str,
                    "type": "fomc_statement",
                    "title": entry.get("title", ""),
                    "text": summary[:2000],
                    "url": entry.get("link", ""),
                }
    except Exception as e:
        logger.warning(f"Failed to fetch FOMC statement: {e}")
    return None


def fetch_latest_beige_book() -> Optional[dict]:
    """Fetch the most recent Beige Book summary from the Fed RSS feed."""
    try:
        feed = feedparser.parse(BEIGE_BOOK_RSS)
        if feed.entries:
            entry = feed.entries[0]
            summary = entry.get("summary", entry.get("description", ""))
            try:
                dt = datetime(*entry.published_parsed[:6])
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = datetime.now().strftime("%Y-%m-%d")
            return {
                "date": date_str,
                "type": "beige_book",
                "title": entry.get("title", ""),
                "text": summary[:2000],
                "url": entry.get("link", ""),
            }
    except Exception as e:
        logger.warning(f"Failed to fetch Beige Book: {e}")
    return None


def score_fed_communication(text: str, llm_analyzer=None) -> float:
    """Score a Fed communication as hawkish (+1) to dovish (-1).

    Uses LLM if available, otherwise falls back to keyword scoring.
    """
    if llm_analyzer and llm_analyzer.llm_available:
        try:
            prompt = (
                "Score the following Federal Reserve communication on a scale from "
                "-1.0 (very dovish/accommodative) to +1.0 (very hawkish/restrictive). "
                "Return ONLY a single number.\n\n"
                f"Text: {text[:1500]}"
            )
            resp = requests.post(
                f"{llm_analyzer.base_url}/api/chat",
                json={
                    "model": llm_analyzer.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "")
                # Extract number from response
                import re
                nums = re.findall(r'-?\d+\.?\d*', content)
                if nums:
                    score = float(nums[0])
                    return max(-1.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM Fed scoring failed: {e}")

    # Keyword fallback
    return _keyword_score(text)


def _keyword_score(text: str) -> float:
    """Simple keyword-based hawkish/dovish scoring."""
    text_lower = text.lower()
    hawkish = ["inflation", "tighten", "restrictive", "rate increase",
               "price stability", "overheating", "elevated", "persistent"]
    dovish = ["accommodative", "easing", "rate cut", "slowdown",
              "labor market softening", "downside risk", "patient", "gradual"]

    hawk_count = sum(1 for w in hawkish if w in text_lower)
    dove_count = sum(1 for w in dovish if w in text_lower)
    total = hawk_count + dove_count
    if total == 0:
        return 0.0
    return round((hawk_count - dove_count) / total, 3)


def update_fed_communications(conn: sqlite3.Connection,
                               llm_analyzer=None) -> dict:
    """Fetch and score latest Fed communications. Returns summary."""
    results = {"fomc": None, "beige_book": None}

    # Check if we already have recent data
    latest = conn.execute(
        "SELECT date, type FROM fed_communications ORDER BY date DESC LIMIT 1"
    ).fetchone()
    latest_date = latest[0] if latest else "2000-01-01"

    # FOMC statement
    fomc = fetch_latest_fomc_statement()
    if fomc and fomc["date"] > latest_date:
        score = score_fed_communication(fomc["text"], llm_analyzer)
        conn.execute(
            """INSERT OR REPLACE INTO fed_communications
               (date, type, hawkish_score, summary, scored_at)
               VALUES (?, ?, ?, ?, ?)""",
            (fomc["date"], "fomc_statement", score,
             fomc["text"][:500], datetime.now().isoformat()),
        )
        conn.commit()
        results["fomc"] = {"date": fomc["date"], "score": score}
        logger.info(f"FOMC statement scored: {score:+.2f} ({fomc['date']})")

    # Beige Book
    bb = fetch_latest_beige_book()
    if bb and bb["date"] > latest_date:
        score = score_fed_communication(bb["text"], llm_analyzer)
        conn.execute(
            """INSERT OR REPLACE INTO fed_communications
               (date, type, hawkish_score, summary, scored_at)
               VALUES (?, ?, ?, ?, ?)""",
            (bb["date"], "beige_book", score,
             bb["text"][:500], datetime.now().isoformat()),
        )
        conn.commit()
        results["beige_book"] = {"date": bb["date"], "score": score}
        logger.info(f"Beige Book scored: {score:+.2f} ({bb['date']})")

    return results


def get_fed_features(conn: sqlite3.Connection, target_date: str) -> dict:
    """Get Fed communication features for a given date.

    Returns:
        fomc_hawkish_score: Most recent FOMC statement score (-1 to +1)
        beige_book_score: Most recent Beige Book score (-1 to +1)
        fed_sentiment_avg: Average of both scores
    """
    fomc_row = conn.execute(
        "SELECT hawkish_score FROM fed_communications "
        "WHERE type = 'fomc_statement' AND date <= ? ORDER BY date DESC LIMIT 1",
        (target_date,),
    ).fetchone()
    fomc_score = fomc_row[0] if fomc_row else 0.0

    bb_row = conn.execute(
        "SELECT hawkish_score FROM fed_communications "
        "WHERE type = 'beige_book' AND date <= ? ORDER BY date DESC LIMIT 1",
        (target_date,),
    ).fetchone()
    bb_score = bb_row[0] if bb_row else 0.0

    avg = (fomc_score + bb_score) / 2 if (fomc_score or bb_score) else 0.0

    return {
        "fomc_hawkish_score": round(fomc_score, 3),
        "beige_book_score": round(bb_score, 3),
        "fed_sentiment_avg": round(avg, 3),
    }
