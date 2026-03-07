"""Secure web search via Bing Search API.

Only the user's question is sent to Bing. No internal context, code,
or documentation is ever transmitted.
"""
import logging
import requests
from src.data.secrets_manager import get_secret

logger = logging.getLogger(__name__)

BING_API_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"


def search_bing(query: str, count: int = 5) -> list[dict]:
    """Perform a web search using the Bing Search API.

    Returns a list of dicts with 'name', 'url', and 'snippet'.
    Returns an empty list if the API key is not configured or the call fails.
    """
    api_key = get_secret("bing_api_key")
    if not api_key:
        return []

    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": count, "responseFilter": "Webpages"}

    try:
        resp = requests.get(BING_API_ENDPOINT, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("webPages", {}).get("value", [])
        return [
            {"name": r.get("name"), "url": r.get("url"), "snippet": r.get("snippet")}
            for r in results
        ]
    except Exception as e:
        logger.warning("Bing Search API call failed: %s", e)
        return []
