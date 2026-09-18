"""Memory tools, and the image enrichment behind search cards.

Both are enrichment: neither is allowed to take down the thing it decorates.

* **Memory** is optional per deployment and can fail mid-call. Every tool has
  to answer in a sentence, because the alternative on a voice turn is a raised
  exception and silence.
* **Images** come from two third-party services with no contract worth relying
  on. A product card without a picture is a worse card; a search that raises
  because a thumbnail 404'd is no card at all.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from src.domain import KwamiConfig
from src.tools.builtin import (
    AgentToolsMixin,
    _fetch_image_for_url,
    _tavily_extract_images,
)

EXTRACT = "https://api.tavily.com/extract"
MICROLINK = "https://api.microlink.io/"


class FakeTracker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def record_external_usage(self, model_type: str, model_id: str, **kwargs: Any) -> None:
        self.calls.append(model_id)


class FakeMemoryContext:
    def __init__(self) -> None:
        self.facts = ["likes tea"]
        self.recent_messages = ["hello"]
        self.summary = "a summary"


class FakeMemory:
    def __init__(self, *, initialized: bool = True, fail: Exception | None = None) -> None:
        self.is_initialized = initialized
        self.user_id = "user-1"
        self.session_id = "session-1"
        self.fail = fail
        self.facts: list[str] = []
        self.results: list[dict] = []

    async def add_fact(self, fact: str) -> None:
        if self.fail:
            raise self.fail
        self.facts.append(fact)

    async def search(self, topic: str, limit: int = 5) -> list[dict]:
        if self.fail:
            raise self.fail
        return self.results

    async def get_context(self) -> FakeMemoryContext:
        if self.fail:
            raise self.fail
        return FakeMemoryContext()


class Tools(AgentToolsMixin):
    def __init__(self, memory: Any = None) -> None:
        self.kwami_config = KwamiConfig()
        self._current_voice_config = self.kwami_config.voice
        self._memory = memory
        self.session = None
        self.room = None
        self.usage_tracker = None


# -- remember_fact -----------------------------------------------------------


async def test_a_fact_is_stored_and_confirmed() -> None:
    memory = FakeMemory()

    result = await Tools(memory).remember_fact(None, "prefers oat milk")

    assert memory.facts == ["prefers oat milk"]
    assert "prefers oat milk" in result


@pytest.mark.parametrize(
    "memory", [None, FakeMemory(initialized=False)], ids=["no-memory", "not-initialised"]
)
async def test_remembering_without_memory_says_so(memory: Any) -> None:
    assert "not available" in await Tools(memory).remember_fact(None, "anything")


async def test_a_memory_failure_while_remembering_is_a_sentence() -> None:
    memory = FakeMemory(fail=RuntimeError("zep down"))
    assert "couldn't save" in await Tools(memory).remember_fact(None, "anything")


# -- recall_memories ---------------------------------------------------------


async def test_recalled_memories_are_listed() -> None:
    memory = FakeMemory()
    memory.results = [{"content": "likes tea"}, {"content": "lives in Madrid"}]

    result = await Tools(memory).recall_memories(None, "preferences")

    assert "likes tea" in result and "lives in Madrid" in result


async def test_nothing_remembered_about_a_topic_says_so() -> None:
    memory = FakeMemory()
    memory.results = []

    assert "don't have any memories" in await Tools(memory).recall_memories(None, "sailing")


async def test_results_with_no_content_are_not_rendered_as_blanks() -> None:
    memory = FakeMemory()
    memory.results = [{"score": 0.9}, {"content": ""}]

    assert "don't have specific memories" in await Tools(memory).recall_memories(None, "sailing")


async def test_a_memory_failure_while_recalling_is_a_sentence() -> None:
    memory = FakeMemory(fail=RuntimeError("zep down"))
    assert "couldn't search" in await Tools(memory).recall_memories(None, "anything")


async def test_recalling_without_memory_says_so() -> None:
    assert "not available" in await Tools(None).recall_memories(None, "anything")


# -- get_memory_status -------------------------------------------------------


async def test_status_reports_what_memory_holds() -> None:
    status = await Tools(FakeMemory()).get_memory_status(None)

    assert status["enabled"] is True
    assert status["status"] == "Active"
    assert status["facts_count"] == 1
    assert status["has_summary"] is True


async def test_status_distinguishes_absent_from_uninitialised() -> None:
    absent = await Tools(None).get_memory_status(None)
    uninitialised = await Tools(FakeMemory(initialized=False)).get_memory_status(None)

    assert absent["enabled"] is False
    assert uninitialised["enabled"] is True
    assert "not initialized" in uninitialised["status"]


async def test_a_failing_status_reports_the_error_rather_than_raising() -> None:
    status = await Tools(FakeMemory(fail=RuntimeError("zep down"))).get_memory_status(None)

    assert status["enabled"] is True
    assert "Error" in status["status"]


# -- Background writes -------------------------------------------------------


async def test_background_facts_are_written_without_blocking_the_caller() -> None:
    """Awaiting a Zep round-trip mid-turn is audible dead air."""
    memory = FakeMemory()
    tools = Tools(memory)

    tools._remember_in_background(["one", "two"])
    for task in list(getattr(tools, "_background_tasks", ())):
        await task

    assert memory.facts == ["one", "two"]


async def test_a_failing_background_write_does_not_surface() -> None:
    memory = FakeMemory(fail=RuntimeError("zep down"))
    tools = Tools(memory)

    tools._remember_in_background(["one"])
    for task in list(getattr(tools, "_background_tasks", ())):
        await task  # must not raise


async def test_nothing_is_scheduled_without_memory_or_facts() -> None:
    tools = Tools(None)
    tools._remember_in_background(["one"])
    assert not getattr(tools, "_background_tasks", None)

    tools = Tools(FakeMemory())
    tools._remember_in_background([])
    assert not getattr(tools, "_background_tasks", None)


# -- Tavily Extract ----------------------------------------------------------


@respx.mock
async def test_extract_maps_each_url_to_its_images() -> None:
    respx.post(EXTRACT).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://a.test",
                        "images": ["https://img/1.jpg", "not-a-url", "https://img/2.jpg"],
                    }
                ]
            },
        )
    )

    images = await _tavily_extract_images("tv_key", ["https://a.test", "https://b.test"])

    assert images["https://a.test"] == ["https://img/1.jpg", "https://img/2.jpg"]
    assert images["https://b.test"] == []


@respx.mock
async def test_extract_is_billed() -> None:
    respx.post(EXTRACT).mock(return_value=httpx.Response(200, json={"results": []}))
    tracker = FakeTracker()

    await _tavily_extract_images("tv_key", ["https://a.test"], usage_tracker=tracker)

    assert "tavily/extract" in tracker.calls


@respx.mock
async def test_a_failing_extract_yields_empty_lists_not_an_exception() -> None:
    respx.post(EXTRACT).mock(side_effect=httpx.ConnectError("dns"))

    assert await _tavily_extract_images("tv_key", ["https://a.test"]) == {"https://a.test": []}


async def test_extract_without_a_key_or_urls_does_not_call_out() -> None:
    assert await _tavily_extract_images("", ["https://a.test"]) == {"https://a.test": []}
    assert await _tavily_extract_images("tv_key", []) == {}


# -- Microlink fallback ------------------------------------------------------


@respx.mock
@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"image": {"url": "https://img/hero.jpg"}}},
        {"data": {"image": "https://img/hero.jpg"}},
        {"data": {"logo": {"url": "https://img/hero.jpg"}}},
        {"data": {"logo": "https://img/hero.jpg"}},
    ],
    ids=["image-object", "image-string", "logo-object", "logo-string"],
)
async def test_an_image_is_found_in_every_shape_microlink_returns(payload: dict) -> None:
    respx.get(MICROLINK).mock(return_value=httpx.Response(200, json=payload))

    assert await _fetch_image_for_url("https://a.test") == "https://img/hero.jpg"


@respx.mock
async def test_the_image_is_preferred_over_the_logo() -> None:
    respx.get(MICROLINK).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "image": {"url": "https://img/hero.jpg"},
                    "logo": {"url": "https://img/logo.png"},
                }
            },
        )
    )

    assert await _fetch_image_for_url("https://a.test") == "https://img/hero.jpg"


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"data": {}}),
        httpx.Response(200, json={}),
        httpx.Response(429, json={}),
    ],
)
async def test_no_image_is_none_rather_than_an_error(response: httpx.Response) -> None:
    respx.get(MICROLINK).mock(return_value=response)

    assert await _fetch_image_for_url("https://a.test") is None


@respx.mock
async def test_a_microlink_outage_does_not_break_the_search_it_decorates() -> None:
    respx.get(MICROLINK).mock(side_effect=httpx.ConnectError("dns"))

    assert await _fetch_image_for_url("https://a.test") is None


@respx.mock
async def test_the_image_fetch_is_billed() -> None:
    respx.get(MICROLINK).mock(
        return_value=httpx.Response(200, json={"data": {"image": "https://img/x.jpg"}})
    )
    tracker = FakeTracker()

    await _fetch_image_for_url("https://a.test", usage_tracker=tracker)

    assert "microlink/fetch" in tracker.calls
