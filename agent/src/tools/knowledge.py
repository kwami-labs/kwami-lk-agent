"""Research and market-data tools.

`web_search` answers one query with a handful of links, which is the right
shape for "what's the weather" and the wrong shape for "research X for me" or
"how is Tesla doing". Two gaps followed from that, and both showed up as the
agent reaching for the browser and narrating a page it had to read aloud:

* **Depth.** A research question is several searches, not one. Asked for one
  anyway, the model either settles for the first five links or burns a dozen
  conversational turns issuing follow-up searches by hand, out loud.
* **Prices.** Market questions have an exact answer that no web snippet
  reliably carries, and a stale number stated confidently is worse than no
  number at all.

Both tools are read-only on purpose. See `TRADING_GUIDANCE` for why there is a
quote tool here and no order tool.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
from livekit.agents import RunContext, function_tool

from ..adapters.http import shared_client
from ..settings import get_settings
from ..utils.logging import get_logger

logger = get_logger("knowledge")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

#: Yahoo's chart endpoint. No key, no registration, and it covers equities,
#: ETFs, FX and crypto (`BTC-USD`) through one symbol space. It is also an
#: undocumented endpoint, so every field is read defensively and a failure
#: degrades to "I couldn't get a price" rather than to a guess.
QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
QUOTE_TIMEOUT_SECONDS = 8.0
QUOTE_USER_AGENT = "Mozilla/5.0 (compatible; KwamiAgent/1.0)"

RESEARCH_TIMEOUT_SECONDS = 20.0
#: How many angles a research pass takes. Above about four the returns fall off
#: sharply while the latency -- which the user is sitting through -- does not.
MAX_RESEARCH_ANGLES = 4
MAX_SOURCES_PER_ANGLE = 4
#: A research briefing goes verbatim into the LLM context and is then spoken.
MAX_BRIEFING_CHARS = 6000
MAX_SNIPPET_CHARS = 420

#: The angles a research pass takes on a topic. Deliberately generic: a
#: topic-specific decomposition would need its own LLM round trip, which is
#: latency the user hears, for a gain the breadth here already delivers.
RESEARCH_ANGLES: tuple[tuple[str, str], ...] = (
    ("overview", "{topic}"),
    # `{year}` rather than a literal. This read "latest news 2026", which was
    # correct when it was written and would have quietly started biasing every
    # research pass towards a stale year on 1 January -- the kind of bug that
    # never fails a test, just slowly makes the answers worse.
    ("recent", "{topic} latest news {year}"),
    ("analysis", "{topic} analysis explained in depth"),
    ("critique", "{topic} criticism problems risks limitations"),
)

#: Why this module has `get_market_quote` and no `place_order`.
#:
#: Placing a real order is irreversible, and a voice agent's failure modes --
#: a misheard ticker, a misheard quantity, a tool call fired on a half-finished
#: sentence -- map directly onto losing the user's money. Reading a price and
#: preparing an order are safe; executing one is not something to infer. So
#: execution stays a browser flow on the user's own broker, which the user
#: watches live in the panel and confirms themselves.
TRADING_GUIDANCE = (
    "\nFor markets and trading: use get_market_quote for prices -- it is exact and current, "
    "where a search snippet is neither. Use deep_research for the story behind a move. "
    "You can walk the user through their broker in the browser panel, and they watch every "
    "step. Never place, modify or cancel an order without the user having said the "
    "instrument, the side and the size out loud, and confirmed after you read it back. "
    "If you are not certain you heard a ticker or a quantity correctly, ask again. "
    "You give information, not financial advice; say so if you are asked to choose for them."
)


def _clean(text: Any, limit: int) -> str:
    """A bounded, single-line string from untrusted provider output."""
    if not isinstance(text, str):
        return ""
    collapsed = " ".join(text.split())
    return collapsed[:limit]


def research_query(template: str, topic: str, *, now: datetime | None = None) -> str:
    """Fill a research angle template.

    `{year}` resolves at call time rather than being baked in. The "recent"
    angle previously read `latest news 2026`: correct on the day it was written,
    and from the following 1 January a silent bias towards a year that is no
    longer recent. Nothing would have failed -- the searches would just have
    quietly got worse.
    """
    year = (now or datetime.now(UTC)).year
    return template.format(topic=topic, year=year)


async def _tavily(
    client: httpx.AsyncClient, api_key: str, query: str, max_results: int
) -> dict[str, Any]:
    """One search. Never raises: a dead angle must not kill the whole pass."""
    try:
        response = await client.post(
            TAVILY_SEARCH_URL,
            json={
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": True,
            },
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=RESEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Research angle failed (%s): %s", query[:60], e)
        return {}


def _domain(url: str) -> str:
    """Host of a URL, for deduplication. Never raises on a malformed one."""
    try:
        from urllib.parse import urlparse

        return (urlparse(url).netloc or "").lower().removeprefix("www.")
    except Exception:
        return ""


class KnowledgeToolsMixin:
    """Research and market-data function tools for KwamiAgent.

    `usage_tracker` is supplied by `KwamiAgent`; it is annotated rather than
    only described so that a rename on the agent shows up here as a type error
    instead of as metering that silently stops recording.
    """

    usage_tracker: Any

    @function_tool()
    async def deep_research(self, context: RunContext, topic: str) -> str:
        """Research a topic properly: several searches at once, then a briefing.

        Use this instead of web_search when the user asks you to research,
        investigate, compare, or explain something in depth -- anything where one
        search and five links would not be a real answer.

        Args:
            topic: What to research, as a full phrase rather than a keyword.
        """
        topic = (topic or "").strip()
        if not topic:
            return "What would you like me to research?"

        api_key = get_settings().tavily_api_key
        if not api_key:
            logger.warning("TAVILY_API_KEY not set; deep research disabled")
            return "Research is not configured (missing TAVILY_API_KEY)."

        angles = RESEARCH_ANGLES[:MAX_RESEARCH_ANGLES]
        try:
            client = shared_client()
            # Concurrent, not sequential: the angles are independent, and
            # four round trips in series is dead air the user sits through.
            payloads = await asyncio.gather(
                *(
                    _tavily(
                        client,
                        api_key,
                        research_query(template, topic),
                        MAX_SOURCES_PER_ANGLE,
                    )
                    for _, template in angles
                )
            )
        except Exception:
            logger.exception("Deep research failed")
            return f"I couldn't complete the research on {topic}."

        if getattr(self, "usage_tracker", None):
            try:
                self.usage_tracker.record_external_usage(
                    "tool",
                    "tavily/search",
                    units_used=float(len(angles)),
                    request_count=len(angles),
                )
            except Exception as e:  # metering must never break a tool
                logger.debug("Could not record research usage: %s", e)

        sections: list[str] = []
        sources: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        seen_domains: set[str] = set()

        for (label, _), payload in zip(angles, payloads, strict=True):
            answer = _clean(payload.get("answer"), MAX_SNIPPET_CHARS * 2)
            results = payload.get("results")
            lines: list[str] = []
            if answer:
                lines.append(answer)

            for result in results if isinstance(results, list) else []:
                if not isinstance(result, dict):
                    continue
                url = _clean(result.get("url"), 500)
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = _clean(result.get("title"), 180)
                content = _clean(result.get("content"), MAX_SNIPPET_CHARS)
                if title or content:
                    lines.append(f"- {title}: {content}" if content else f"- {title}")

                # One card per publisher: four angles on one topic return the
                # same few outlets repeatedly, and a panel of duplicates reads
                # as breadth the research does not actually have.
                domain = _domain(url)
                if domain and domain not in seen_domains:
                    seen_domains.add(domain)
                    sources.append(
                        {
                            "title": title or domain,
                            "url": url,
                            "content": content,
                            "features": [domain, label],
                        }
                    )

            if lines:
                sections.append(f"## {label.title()}\n" + "\n".join(lines))

        if not sections:
            return f"I searched but found nothing usable on {topic}."

        self._publish_research(context, topic, sources)
        remember = getattr(self, "_remember_in_background", None)
        if callable(remember):
            remember([f"User asked for research on: {topic}"])

        briefing = "\n\n".join(sections)[:MAX_BRIEFING_CHARS]
        return (
            f"Research on '{topic}' from {len(seen_domains)} sources. "
            "Summarise this for the user in your own words, conversationally, and offer to go "
            "deeper on any part. Do not read it out verbatim.\n\n" + briefing
        )

    def _publish_research(
        self, context: RunContext, topic: str, sources: list[dict[str, Any]]
    ) -> None:
        """Put the sources on screen, without letting a publish failure kill the tool."""
        if not sources:
            return
        publisher = getattr(self, "_publisher", None)
        if not callable(publisher):
            return
        task = asyncio.create_task(
            publisher(context).publish(
                {
                    "type": "search_results",
                    "query": topic,
                    "results": sources,
                    "answer": f"Researched '{topic[:60]}' across {len(sources)} sources.",
                }
            )
        )
        tasks = getattr(self, "_background_tasks", None)
        if tasks is None:
            tasks = set()
            self._background_tasks = tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    @function_tool()
    async def get_market_quote(self, context: RunContext, symbol: str) -> dict[str, Any]:
        """Get the current price and day's move for a stock, ETF, currency or crypto.

        Use this rather than a web search whenever the user asks what something
        is trading at: a search snippet's price is whenever the page was
        indexed, and a stale number stated confidently is worse than none.

        Args:
            symbol: A ticker. Equities and ETFs as-is ('TSLA', 'VOO'), crypto
                    against a currency ('BTC-USD', 'ETH-EUR'), FX as a pair
                    ('EURUSD=X'), indices with a caret ('^GSPC').
        """
        ticker = (symbol or "").strip().upper()
        if not ticker:
            return {"error": "No symbol given."}

        try:
            client = shared_client()
            response = await client.get(
                QUOTE_URL.format(symbol=ticker),
                headers={"User-Agent": QUOTE_USER_AGENT},
                params={"range": "1d", "interval": "1d"},
                timeout=QUOTE_TIMEOUT_SECONDS,
            )
            # An unknown ticker 404s. That is a different answer from "the
            # feed is down", and the user can act on it -- they misheard
            # themselves, or the thing trades under another symbol.
            if response.status_code == 404:
                logger.info("Unknown market symbol: %s", ticker)
                return {
                    "error": f"I don't recognise the symbol {ticker}.",
                    "hint": "Crypto needs a currency, like BTC-USD. Indices start with ^.",
                }
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning("Quote lookup failed for %s: %s", ticker, e)
            return {"error": f"I couldn't get a price for {ticker} right now."}

        if getattr(self, "usage_tracker", None):
            try:
                self.usage_tracker.record_external_usage(
                    "tool", "market/quote", units_used=1.0, request_count=1
                )
            except Exception as e:
                logger.debug("Could not record quote usage: %s", e)

        meta = self._quote_meta(data)
        if meta is None:
            return {"error": f"I don't recognise the symbol {ticker}."}

        price = meta.get("regularMarketPrice")
        previous = meta.get("chartPreviousClose") or meta.get("previousClose")
        if not isinstance(price, int | float):
            return {"error": f"I couldn't get a price for {ticker} right now."}

        quote: dict[str, Any] = {
            "symbol": meta.get("symbol") or ticker,
            "name": meta.get("longName") or meta.get("shortName") or ticker,
            "price": round(float(price), 4),
            "currency": meta.get("currency") or "",
            "exchange": meta.get("fullExchangeName") or meta.get("exchangeName") or "",
        }
        if isinstance(previous, int | float) and previous:
            change = float(price) - float(previous)
            quote["change"] = round(change, 4)
            quote["change_percent"] = round(change / float(previous) * 100.0, 2)
            quote["direction"] = "up" if change > 0 else "down" if change < 0 else "flat"
        # Say it out loud, so the model does not present a delayed feed as live.
        quote["note"] = "Market data may be delayed. This is information, not advice."
        return quote

    @staticmethod
    def _quote_meta(data: Any) -> dict[str, Any] | None:
        """Dig the `meta` block out of a chart response, tolerating any shape."""
        if not isinstance(data, dict):
            return None
        chart = data.get("chart")
        if not isinstance(chart, dict):
            return None
        results = chart.get("result")
        if not isinstance(results, list) or not results:
            return None
        first = results[0]
        if not isinstance(first, dict):
            return None
        meta = first.get("meta")
        return meta if isinstance(meta, dict) else None
