"""Web and product search: the parsing, the failure modes, and the billing.

Search is the tool the user reaches most often, and almost all of its surface
is about what happens when the upstream is unhelpful rather than when it works:

* **Every provider error has to become a sentence.** A raised exception in a
  voice turn is silence; a leaked vendor body is worse, because Tavily's 432
  says "usage limit" when the credits are fine and the model would repeat that
  to the user as fact.
* **Results are trimmed before they are published.** The data channel has a
  hard limit, and an over-long payload is dropped whole -- no cards at all.
* **Every upstream call is billed.** These were the metered calls; a path that
  skips `record_external_usage` is invisible margin loss.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from src.domain import KwamiConfig
from src.settings import Settings, set_settings
from src.tools.builtin import (
    AgentToolsMixin,
    _extract_features,
    _extract_price,
    _product_name_from_title,
)

TAVILY = "https://api.tavily.com/search"
SERPAPI = "https://serpapi.com/search"


class FakePublisher:
    def __init__(self, ok: bool = True) -> None:
        self.published: list[dict] = []
        self.ok = ok

    async def publish(self, payload: dict, *, topic: str | None = None) -> bool:
        self.published.append(payload)
        return self.ok


class FakeTracker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def record_external_usage(self, model_type: str, model_id: str, **kwargs: Any) -> None:
        self.calls.append((model_type, model_id))


class FakeMemory:
    is_initialized = True

    def __init__(self) -> None:
        self.facts: list[str] = []

    async def add_fact(self, fact: str) -> None:
        self.facts.append(fact)


class Tools(AgentToolsMixin):
    def __init__(self, *, memory: Any = None, tracker: Any = None) -> None:
        self.kwami_config = KwamiConfig()
        self._current_voice_config = self.kwami_config.voice
        self._memory = memory
        self.session = None
        self.room = None
        self.usage_tracker = tracker
        self.publisher = FakePublisher()

    def _publisher(self, context: Any = None):
        return self.publisher


@pytest.fixture(autouse=True)
def _settings():
    set_settings(Settings(tavily_api_key="tv_key", serpapi_key="sa_key"))
    yield
    set_settings(None)


@pytest.fixture(autouse=True)
def _no_image_fetching(monkeypatch):
    """Image enrichment is a separate concern with its own upstreams."""
    import src.tools.builtin as builtin

    async def no_images(*args: Any, **kwargs: Any) -> dict:
        return {}

    async def no_image(*args: Any, **kwargs: Any):
        return None

    monkeypatch.setattr(builtin, "_tavily_extract_images", no_images)
    monkeypatch.setattr(builtin, "_fetch_image_for_url", no_image)


# -- Parsing helpers ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Only $49.99 today", "$49.99"),
        ("Costs €199 including delivery", "€199"),
        ("Yours for 1,200€", "€1.200"),
        ("Priced at 89.50 USD", "$89.50"),
        ("£50 flat", "£50"),
    ],
)
def test_prices_are_recognised_in_the_shapes_sites_write_them(text: str, expected: str) -> None:
    assert _extract_price(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "no numbers at all here"])
def test_text_without_a_price_yields_none(text: str) -> None:
    assert _extract_price(text) is None


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Leather Tote Bag | Acme Store", "Leather Tote Bag"),
        ("Wool Coat - Nordstrom", "Wool Coat"),
        ("Silk Scarf – Farfetch", "Silk Scarf"),
        ("Plain Title", "Plain Title"),
        ("", ""),
    ],
)
def test_the_site_name_is_stripped_from_a_product_title(title: str, expected: str) -> None:
    assert _product_name_from_title(title) == expected


def test_features_are_split_deduplicated_and_bounded() -> None:
    features = _extract_features("2 bedrooms, 2 bedrooms; balcony • sea view\nparking")

    assert features == ["2 bedrooms", "balcony", "sea view", "parking"]


def test_a_long_feature_is_truncated_rather_than_dropped() -> None:
    (feature,) = _extract_features("x" * 200)
    assert feature.endswith("...")
    assert len(feature) <= 72


def test_filler_words_are_not_features() -> None:
    assert _extract_features("and, or, the, with") == []


def test_no_content_yields_no_features() -> None:
    assert _extract_features("") == []


# -- web_search --------------------------------------------------------------


async def test_search_without_a_key_says_so_rather_than_failing() -> None:
    set_settings(Settings())
    assert "not configured" in await Tools().web_search(None, "kwami")


@respx.mock
async def test_a_successful_search_publishes_cards_and_answers() -> None:
    respx.post(TAVILY).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "Kwami is a 3D AI companion.",
                "results": [
                    {
                        "title": "Kwami — Docs",
                        "url": "https://kwami.io",
                        "content": "Voice, memory, tools. €199 a year.",
                    }
                ],
            },
        )
    )
    tools = Tools()

    answer = await tools.web_search(None, "kwami")

    assert answer == "Kwami is a 3D AI companion."
    (payload,) = tools.publisher.published
    assert payload["type"] == "search_results"
    assert payload["results"][0]["product_name"] == "Kwami"
    assert payload["results"][0]["price"] == "€199"


@respx.mock
async def test_the_search_is_billed() -> None:
    respx.post(TAVILY).mock(return_value=httpx.Response(200, json={"results": []}))
    tracker = FakeTracker()

    await Tools(tracker=tracker).web_search(None, "kwami")

    assert ("tool", "tavily/search") in tracker.calls


@respx.mock
async def test_results_fall_back_to_a_summary_when_there_is_no_answer() -> None:
    respx.post(TAVILY).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"title": "One", "url": "https://one.test", "content": "First result"}]
            },
        )
    )

    assert "One" in await Tools().web_search(None, "kwami")


@respx.mock
async def test_an_empty_result_set_says_so() -> None:
    respx.post(TAVILY).mock(return_value=httpx.Response(200, json={"results": []}))
    assert await Tools().web_search(None, "kwami") == "No results found."


@respx.mock
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (432, "temporarily unavailable"),
        (401, "not configured correctly"),
        (429, "rate limit"),
    ],
)
async def test_provider_errors_become_sentences(status: int, expected: str) -> None:
    respx.post(TAVILY).mock(return_value=httpx.Response(status, json={"detail": "usage limit"}))

    result = await Tools().web_search(None, "kwami")

    assert expected in result


@respx.mock
async def test_the_432_body_is_not_repeated_to_the_user() -> None:
    """Tavily says "usage limit" on 432 when the credits are fine.

    Echoing it makes the model state a falsehood about the user's account.
    """
    respx.post(TAVILY).mock(
        return_value=httpx.Response(432, json={"detail": "You have hit your usage limit"})
    )

    assert "usage limit" not in await Tools().web_search(None, "kwami")


@respx.mock
async def test_an_unexpected_status_reports_the_provider_message() -> None:
    respx.post(TAVILY).mock(return_value=httpx.Response(500, json={"message": "boom"}))
    assert "boom" in await Tools().web_search(None, "kwami")


@respx.mock
async def test_a_network_failure_is_a_sentence_not_an_exception() -> None:
    respx.post(TAVILY).mock(side_effect=httpx.ConnectError("dns"))
    assert "Search failed" in await Tools().web_search(None, "kwami")


@respx.mock
async def test_a_product_search_restricts_the_domains() -> None:
    route = respx.post(TAVILY).mock(return_value=httpx.Response(200, json={"results": []}))

    await Tools().web_search(None, "bags", search_for_products=True)

    body = __import__("json").loads(route.calls.last.request.content)
    assert "amazon.com" in body["include_domains"]


@respx.mock
@pytest.mark.parametrize(("requested", "expected"), [(0, 1), (3, 3), (50, 10)])
async def test_the_result_count_is_bounded(requested: int, expected: int) -> None:
    route = respx.post(TAVILY).mock(return_value=httpx.Response(200, json={"results": []}))

    await Tools().web_search(None, "kwami", max_results=requested)

    assert __import__("json").loads(route.calls.last.request.content)["max_results"] == expected


@respx.mock
async def test_the_query_is_remembered_for_later_conversations() -> None:
    respx.post(TAVILY).mock(return_value=httpx.Response(200, json={"results": []}))
    memory = FakeMemory()

    tools = Tools(memory=memory)
    await tools.web_search(None, "walking boots", search_for_products=True)
    for task in list(getattr(tools, "_background_tasks", ())):
        await task

    assert any("walking boots" in fact for fact in memory.facts)


# -- product_search ----------------------------------------------------------


async def test_product_search_without_a_key_points_at_the_alternative() -> None:
    set_settings(Settings(tavily_api_key="tv_key"))

    result = await Tools().product_search(None, "bags")

    assert "web_search" in result


@respx.mock
async def test_product_search_publishes_real_product_cards() -> None:
    respx.get(SERPAPI).mock(
        return_value=httpx.Response(
            200,
            json={
                "shopping_results": [
                    {
                        "title": "Leather Tote",
                        "price": "$120.00",
                        "product_link": "https://shop.test/tote",
                        "thumbnail": "https://img.test/tote.jpg",
                        "source": "Acme",
                        "snippet": "Full grain leather",
                    }
                ]
            },
        )
    )
    tools = Tools()

    answer = await tools.product_search(None, "leather tote")

    assert "Found 1 products" in answer
    card = tools.publisher.published[0]["results"][0]
    assert card["price"] == "$120.00"
    assert card["image"] == "https://img.test/tote.jpg"
    assert "Acme" in card["features"]


@respx.mock
async def test_product_search_is_billed() -> None:
    respx.get(SERPAPI).mock(return_value=httpx.Response(200, json={"shopping_results": [{}]}))
    tracker = FakeTracker()

    await Tools(tracker=tracker).product_search(None, "bags")

    assert ("tool", "serpapi/google_shopping") in tracker.calls


@respx.mock
async def test_no_products_points_back_at_web_search() -> None:
    respx.get(SERPAPI).mock(return_value=httpx.Response(200, json={"shopping_results": []}))
    assert "web_search" in await Tools().product_search(None, "bags")


@respx.mock
async def test_a_failing_product_search_falls_back_rather_than_raising() -> None:
    respx.get(SERPAPI).mock(side_effect=httpx.ConnectError("dns"))
    assert "web_search" in await Tools().product_search(None, "bags")


@respx.mock
@pytest.mark.parametrize(("requested", "expected"), [(0, 1), (4, 4), (99, 10)])
async def test_the_product_count_is_bounded(requested: int, expected: int) -> None:
    route = respx.get(SERPAPI).mock(return_value=httpx.Response(200, json={"shopping_results": []}))

    await Tools().product_search(None, "bags", max_results=requested)

    assert route.calls.last.request.url.params["num"] == str(expected)
