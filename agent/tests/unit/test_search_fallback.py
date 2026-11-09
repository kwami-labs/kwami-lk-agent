"""When Tavily cannot answer, the search moves to SerpAPI or says what happened."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from src.domain.prompt import CLAUDE_HEARING_NOTE
from src.settings import Settings, set_settings
from src.tools import web_fallback
from src.tools.knowledge import TAVILY_SEARCH_URL, KnowledgeToolsMixin
from src.tools.limits import trim_client_tools
from src.tools.web_fallback import safe_http_error, serpapi_web_search


@pytest.fixture(autouse=True)
def _restore_settings() -> Any:
    yield
    set_settings(None)


def _settings(**values: Any) -> MagicMock:
    return MagicMock(serpapi_key=values.get("serpapi_key", "secret-key"))


def _response(status: int, body: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json=body or {}, request=httpx.Request("GET", "https://x"))


def _client(response: httpx.Response) -> MagicMock:
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    return client


async def test_no_key_means_no_fallback() -> None:
    with patch.object(web_fallback, "get_settings", return_value=_settings(serpapi_key="")):
        assert await serpapi_web_search("q") is None


async def test_results_come_back_in_tavilys_shape_and_are_billed() -> None:
    body = {
        "organic_results": [{"title": "T", "link": "https://a", "snippet": "S"}, "junk"],
        "answer_box": {"answer": "42"},
    }
    tracker = MagicMock()
    with (
        patch.object(web_fallback, "get_settings", return_value=_settings()),
        patch.object(web_fallback, "shared_client", return_value=_client(_response(200, body))),
    ):
        found = await serpapi_web_search("q", usage_tracker=tracker)
    assert found == {
        "results": [{"title": "T", "url": "https://a", "content": "S"}],
        "answer": "42",
    }
    tracker.record_external_usage.assert_called_once()


async def test_nothing_found_or_a_failure_is_not_a_result() -> None:
    for response in (_response(200, {}), _response(429)):
        with (
            patch.object(web_fallback, "get_settings", return_value=_settings()),
            patch.object(web_fallback, "shared_client", return_value=_client(response)),
        ):
            assert await serpapi_web_search("q") is None


def test_the_key_in_serpapis_url_stays_out_of_the_logs() -> None:
    request = httpx.Request("GET", "https://serpapi.com/search?api_key=secret-key")
    error = httpx.HTTPStatusError(
        "failed for url https://serpapi.com/search?api_key=secret-key",
        request=request,
        response=httpx.Response(429, request=request),
    )
    assert safe_http_error(error) == "HTTP 429"
    assert safe_http_error(httpx.ReadTimeout("t")) == "ReadTimeout"


@respx.mock
async def test_research_with_the_service_down_does_not_claim_nothing_exists() -> None:
    respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(500))
    set_settings(Settings(tavily_api_key="tv"))
    agent = MagicMock(usage_tracker=MagicMock())

    result = await KnowledgeToolsMixin.deep_research(agent, None, "volcanoes")

    assert "not answering" in result
    assert "found nothing" not in result
    agent.usage_tracker.record_external_usage.assert_not_called()


@respx.mock
async def test_research_fills_only_the_failed_angles_from_serpapi() -> None:
    calls = {"n": 0}

    def tavily(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"answer": "from tavily", "results": []})

    respx.post(TAVILY_SEARCH_URL).mock(side_effect=tavily)
    serp = respx.get(web_fallback.SERPAPI_URL).mock(
        return_value=httpx.Response(
            200, json={"organic_results": [{"title": "T", "link": "https://a", "snippet": "S"}]}
        )
    )
    set_settings(Settings(tavily_api_key="tv", serpapi_key="sp"))
    agent = MagicMock(usage_tracker=MagicMock())
    agent._publish_research = MagicMock()

    result = await KnowledgeToolsMixin.deep_research(agent, None, "volcanoes")

    assert serp.call_count == 1
    assert "from tavily" in result
    assert "T: S" in result


def test_the_voice_is_told_cloud_can_be_claude() -> None:
    assert "Claude" in CLAUDE_HEARING_NOTE
    assert "delegate_task" in CLAUDE_HEARING_NOTE


def _tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.info.name = name
    return tool


def test_the_calendar_survives_a_trim_that_used_to_cut_it() -> None:
    tools = [_tool(f"ui_{i}") for i in range(5)] + [_tool("list_calendar_events")]
    kept = trim_client_tools(0, tools, limit=3)
    assert [t.info.name for t in kept] == ["list_calendar_events", "ui_0", "ui_1"]
