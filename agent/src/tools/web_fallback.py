"""A second way to search, for when Tavily cannot answer.

Tavily is the search backend and SerpAPI the other credential a deployment
usually holds, for product cards. A spent Tavily plan or a slow minute used to
end the search there, with the user told it had failed while a working search
sat one key away.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..adapters.http import shared_client
from ..settings import get_settings
from ..utils.logging import get_logger

logger = get_logger("tools.web_fallback")

SERPAPI_URL = "https://serpapi.com/search"


def safe_http_error(error: Exception) -> str:
    """What to log about a failed vendor call.

    Not `str(error)`: httpx puts the full URL in its message, and SerpAPI takes
    its key as a query parameter, so that wrote the key to the logs.
    """
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


async def serpapi_web_search(
    query: str, max_results: int = 5, usage_tracker: Any = None
) -> dict[str, Any] | None:
    """Google results in Tavily's shape, or None when this cannot help either."""
    api_key = get_settings().serpapi_key
    if not api_key:
        return None
    try:
        r = await shared_client().get(
            SERPAPI_URL,
            params={"engine": "google", "q": query, "api_key": api_key, "num": max_results},
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning("SerpApi web search failed: %s", safe_http_error(e))
        return None

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "content": item.get("snippet", ""),
        }
        for item in (data.get("organic_results") or [])[:max_results]
        if isinstance(item, dict)
    ]
    if not results:
        return None
    if usage_tracker:
        usage_tracker.record_external_usage(
            "tool", "serpapi/google", units_used=1.0, request_count=1
        )
    box = data.get("answer_box")
    answer = (box.get("answer") or box.get("snippet") or "") if isinstance(box, dict) else ""
    logger.info("Search answered by SerpApi after Tavily failed")
    return {"results": results, "answer": answer}
