"""Web search via DuckDuckGo — no API key required.

Uses the ddgs library which handles browser impersonation
properly. Only the user's question is sent externally.
No internal context, code, or documentation is ever transmitted.
"""
import logging

logger = logging.getLogger(__name__)


def search_web(query: str, count: int = 3) -> list[dict]:
    """Search the web using DuckDuckGo (no API key needed).

    Returns a list of dicts with 'name', 'url', and 'snippet'.
    """
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=count))

        results = []
        for r in raw:
            results.append({
                "name": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
        if results:
            logger.info("DuckDuckGo returned %d results for: %s", len(results), query[:60])
        return results

    except Exception as e:
        logger.warning("DuckDuckGo search failed: %s", e)
        return []


# Backward compatibility alias
search_bing = search_web
