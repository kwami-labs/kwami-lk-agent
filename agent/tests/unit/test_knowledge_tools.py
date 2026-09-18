"""Multi-angle research and market quotes.

Both tools talk to an outside service, so both are tested against `respx`
rather than a hand-written double: what matters is how real response shapes --
including the malformed and the missing ones -- are handled, and a double would
only ever return what this file already believes.

The market endpoint is undocumented, so "the shape changed" is a *when*, not an
*if*. Every field is read defensively and the tests say what each failure
degrades to, because a voice agent that states a wrong price confidently is
worse than one that says it could not get a price.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.tools.knowledge import QUOTE_URL, RESEARCH_ANGLES, TAVILY_SEARCH_URL


@pytest.fixture
def agent() -> KwamiAgent:
    return KwamiAgent(config=KwamiConfig())


def _quote_payload(**meta: Any) -> dict[str, Any]:
    base = {
        "symbol": "TSLA",
        "shortName": "Tesla",
        "longName": "Tesla, Inc.",
        "currency": "USD",
        "fullExchangeName": "NasdaqGS",
        "regularMarketPrice": 366.2,
        "chartPreviousClose": 358.08,
    }
    base.update(meta)
    return {"chart": {"result": [{"meta": base}], "error": None}}


def _tavily_payload(answer: str, urls: list[str]) -> dict[str, Any]:
    return {
        "answer": answer,
        "results": [
            {"title": f"Title {i}", "url": url, "content": f"Content about thing {i}"}
            for i, url in enumerate(urls)
        ],
    }


# -- market quotes ----------------------------------------------------------


@respx.mock
async def test_quote_reports_price_and_move(agent: KwamiAgent) -> None:
    respx.get(QUOTE_URL.format(symbol="TSLA")).mock(
        return_value=httpx.Response(200, json=_quote_payload())
    )

    quote = await agent.get_market_quote(None, "tsla")

    assert quote["symbol"] == "TSLA"
    assert quote["name"] == "Tesla, Inc."
    assert quote["price"] == 366.2
    assert quote["currency"] == "USD"
    assert quote["change"] == pytest.approx(8.12)
    assert quote["change_percent"] == pytest.approx(2.27, abs=0.01)
    assert quote["direction"] == "up"


@respx.mock
async def test_quote_says_the_data_may_be_delayed(agent: KwamiAgent) -> None:
    """The model will read this out; it must not present a delayed feed as live."""
    respx.get(QUOTE_URL.format(symbol="TSLA")).mock(
        return_value=httpx.Response(200, json=_quote_payload())
    )

    quote = await agent.get_market_quote(None, "TSLA")

    assert "delayed" in quote["note"].lower()
    assert "advice" in quote["note"].lower()


@respx.mock
async def test_a_downward_move_is_labelled(agent: KwamiAgent) -> None:
    respx.get(QUOTE_URL.format(symbol="TSLA")).mock(
        return_value=httpx.Response(
            200, json=_quote_payload(regularMarketPrice=350.0, chartPreviousClose=358.08)
        )
    )

    quote = await agent.get_market_quote(None, "TSLA")

    assert quote["direction"] == "down"
    assert quote["change"] < 0


@respx.mock
async def test_an_unknown_symbol_is_named_as_such(agent: KwamiAgent) -> None:
    """A 404 is "no such ticker", which the user can act on.

    Collapsing it into "couldn't get a price" hides the one thing they need to
    know: they said a symbol that does not exist.
    """
    respx.get(QUOTE_URL.format(symbol="NOSUCH")).mock(return_value=httpx.Response(404))

    quote = await agent.get_market_quote(None, "NOSUCH")

    assert "don't recognise" in quote["error"]
    assert "BTC-USD" in quote["hint"], "no nudge toward the right symbol format"


@respx.mock
async def test_a_dead_feed_does_not_produce_a_price(agent: KwamiAgent) -> None:
    respx.get(QUOTE_URL.format(symbol="TSLA")).mock(return_value=httpx.Response(503))

    quote = await agent.get_market_quote(None, "TSLA")

    assert "error" in quote
    assert "price" not in quote


@respx.mock
async def test_a_timeout_does_not_produce_a_price(agent: KwamiAgent) -> None:
    respx.get(QUOTE_URL.format(symbol="TSLA")).mock(side_effect=httpx.ConnectTimeout("slow"))

    quote = await agent.get_market_quote(None, "TSLA")

    assert "error" in quote
    assert "price" not in quote


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"chart": None},
        {"chart": {"result": []}},
        {"chart": {"result": [{}]}},
        {"chart": {"result": [{"meta": "not-a-dict"}]}},
        {"chart": {"result": [{"meta": {"symbol": "TSLA"}}]}},  # no price
        {"chart": {"result": [{"meta": {"regularMarketPrice": "not-a-number"}}]}},
    ],
    ids=[
        "empty",
        "null-chart",
        "no-results",
        "no-meta",
        "meta-not-a-dict",
        "no-price",
        "price-not-a-number",
    ],
)
@respx.mock
async def test_a_malformed_response_never_becomes_a_price(agent: KwamiAgent, payload) -> None:
    """The endpoint is undocumented; its shape will change without notice."""
    respx.get(QUOTE_URL.format(symbol="TSLA")).mock(return_value=httpx.Response(200, json=payload))

    quote = await agent.get_market_quote(None, "TSLA")

    assert "error" in quote
    assert "price" not in quote


async def test_an_empty_symbol_is_refused_without_a_request(agent: KwamiAgent) -> None:
    # No respx mock: reaching the network here would be the failure.
    quote = await agent.get_market_quote(None, "   ")
    assert "error" in quote


@respx.mock
async def test_a_flat_day_is_not_reported_as_a_move(agent: KwamiAgent) -> None:
    respx.get(QUOTE_URL.format(symbol="FLAT")).mock(
        return_value=httpx.Response(
            200, json=_quote_payload(regularMarketPrice=100.0, chartPreviousClose=100.0)
        )
    )

    quote = await agent.get_market_quote(None, "FLAT")

    assert quote["direction"] == "flat"
    assert quote["change"] == 0


@respx.mock
async def test_a_zero_previous_close_does_not_divide_by_zero(agent: KwamiAgent) -> None:
    respx.get(QUOTE_URL.format(symbol="NEW")).mock(
        return_value=httpx.Response(
            200, json=_quote_payload(regularMarketPrice=10.0, chartPreviousClose=0)
        )
    )

    quote = await agent.get_market_quote(None, "NEW")

    assert quote["price"] == 10.0
    assert "change_percent" not in quote, "reported a percentage move from a zero base"


# -- deep research ----------------------------------------------------------


async def test_research_without_a_key_says_so(agent: KwamiAgent) -> None:
    result = await agent.deep_research(None, "quantum computing")
    assert "not configured" in result


async def test_research_with_no_topic_asks_for_one(agent: KwamiAgent, fake_key) -> None:
    fake_key("TAVILY_API_KEY")
    result = await agent.deep_research(None, "  ")
    assert "?" in result


@respx.mock
async def test_research_runs_every_angle(agent: KwamiAgent, env_setting) -> None:
    """One search is not research; the point of the tool is the breadth."""
    env_setting("TAVILY_API_KEY", "tvly-test")
    route = respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(200, json=_tavily_payload("An answer.", ["https://a.com/1"]))
    )

    await agent.deep_research(None, "fusion power")

    assert route.call_count == len(RESEARCH_ANGLES)
    queries = [call.request.content.decode() for call in route.calls]
    assert any("fusion power" in q for q in queries)
    assert any("criticism" in q for q in queries), "the critical angle was not searched"


@respx.mock
async def test_research_briefing_carries_the_sources(agent: KwamiAgent, env_setting) -> None:
    env_setting("TAVILY_API_KEY", "tvly-test")
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json=_tavily_payload("Fusion is hard.", ["https://iaea.org/a", "https://nature.com/b"]),
        )
    )

    result = await agent.deep_research(None, "fusion power")

    assert "Fusion is hard." in result
    assert "Overview" in result
    assert "Critique" in result
    assert "in your own words" in result, "the model was not told to paraphrase"


@respx.mock
async def test_research_survives_a_dead_angle(agent: KwamiAgent, env_setting) -> None:
    """One failing search must not lose the other three."""
    env_setting("TAVILY_API_KEY", "tvly-test")
    responses = [
        httpx.Response(500),
        httpx.Response(200, json=_tavily_payload("Second angle.", ["https://b.com/1"])),
        httpx.Response(200, json=_tavily_payload("Third angle.", ["https://c.com/1"])),
        httpx.Response(200, json=_tavily_payload("Fourth angle.", ["https://d.com/1"])),
    ]
    respx.post(TAVILY_SEARCH_URL).mock(side_effect=responses)

    result = await agent.deep_research(None, "fusion power")

    assert "Second angle." in result
    assert "Fourth angle." in result


@respx.mock
async def test_research_with_nothing_found_says_so(agent: KwamiAgent, env_setting) -> None:
    env_setting("TAVILY_API_KEY", "tvly-test")
    respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json={"results": []}))

    result = await agent.deep_research(None, "asdkjhasdkjh")

    assert "found nothing" in result


@respx.mock
async def test_research_deduplicates_repeated_sources(agent: KwamiAgent, env_setting) -> None:
    """Four angles on one topic return the same outlets repeatedly.

    Left in, the briefing is padded with the same link four times and the
    on-screen cards read as breadth the research does not have.
    """
    env_setting("TAVILY_API_KEY", "tvly-test")
    respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(
            200, json=_tavily_payload("Same answer.", ["https://same.com/page"])
        )
    )

    result = await agent.deep_research(None, "fusion power")

    assert result.count("https://same.com/page") <= 1
    assert "from 1 sources" in result


@respx.mock
async def test_research_briefing_is_bounded(agent: KwamiAgent, env_setting) -> None:
    """The briefing goes verbatim into the LLM context on every research call."""
    from src.tools.knowledge import MAX_BRIEFING_CHARS

    env_setting("TAVILY_API_KEY", "tvly-test")
    huge = {
        "answer": "x" * 50_000,
        "results": [
            {"title": "t" * 5_000, "url": f"https://site{i}.com/", "content": "c" * 50_000}
            for i in range(20)
        ],
    }
    respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json=huge))

    result = await agent.deep_research(None, "fusion power")

    assert len(result) < MAX_BRIEFING_CHARS + 500
