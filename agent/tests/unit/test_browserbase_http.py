"""The Browserbase REST client, against a faked transport.

Everything here is about the three places the vendor's API shape is load-bearing
and easy to get wrong, each of which fails in a way that costs money or logins
rather than raising something obvious:

* **Auth and project scoping.** The header is `X-BB-API-Key`, not a bearer
  token, and a key scoped to several projects needs `projectId` or the session
  lands in the wrong one.
* **The live view is a second call.** `POST /v1/sessions` returns no watchable
  URL; failing to fetch one must not take down a browser the agent can already
  drive.
* **Release is explicit.** Sessions bill until `REQUEST_RELEASE` or their own
  timeout, so a leaked session is a silent bill.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.browser.browserbase import (
    BB_API_BASE,
    SESSION_TIMEOUT_SECONDS,
    VIEWPORT,
    BrowserbaseClient,
)

API_KEY = "bb_test_key"
PROJECT = "proj_1"


def _client(project_id: str = PROJECT) -> BrowserbaseClient:
    return BrowserbaseClient(api_key=API_KEY, project_id=project_id)


# -- Construction ------------------------------------------------------------


def test_refuses_to_build_without_a_key() -> None:
    with pytest.raises(ValueError, match="BROWSERBASE_API_KEY"):
        BrowserbaseClient(api_key="", project_id=PROJECT)


# -- Contexts ----------------------------------------------------------------


@respx.mock
async def test_create_context_sends_the_project_and_returns_the_id() -> None:
    route = respx.post(f"{BB_API_BASE}/contexts").mock(
        return_value=httpx.Response(201, json={"id": "ctx-1"})
    )

    assert await _client().create_context("kwami-user-1") == "ctx-1"

    request = route.calls.last.request
    assert request.headers["X-BB-API-Key"] == API_KEY
    body = __import__("json").loads(request.content)
    assert body == {"name": "kwami-user-1", "projectId": PROJECT}


@respx.mock
async def test_create_context_omits_the_project_when_there_is_none() -> None:
    """The platform infers it from the key; sending null is not the same."""
    route = respx.post(f"{BB_API_BASE}/contexts").mock(
        return_value=httpx.Response(201, json={"id": "ctx-1"})
    )

    await _client(project_id="").create_context("kwami-user-1")

    assert "projectId" not in __import__("json").loads(route.calls.last.request.content)


@respx.mock
async def test_a_context_response_with_no_id_is_an_error() -> None:
    respx.post(f"{BB_API_BASE}/contexts").mock(return_value=httpx.Response(201, json={}))

    with pytest.raises(RuntimeError, match="context id"):
        await _client().create_context("kwami-user-1")


@respx.mock
async def test_context_exists_is_true_for_a_live_context() -> None:
    respx.get(f"{BB_API_BASE}/contexts/ctx-1").mock(
        return_value=httpx.Response(200, json={"id": "ctx-1"})
    )
    assert await _client().context_exists("ctx-1") is True


@respx.mock
async def test_context_exists_is_false_once_it_is_gone() -> None:
    """A deleted context makes the whole session-create fail, so it is checked."""
    respx.get(f"{BB_API_BASE}/contexts/ctx-1").mock(return_value=httpx.Response(404))
    assert await _client().context_exists("ctx-1") is False


@respx.mock
async def test_an_unreachable_api_does_not_declare_the_context_dead() -> None:
    """Unreachable is not gone. Discarding a live context signs the user out."""
    respx.get(f"{BB_API_BASE}/contexts/ctx-1").mock(side_effect=httpx.ConnectError("down"))
    assert await _client().context_exists("ctx-1") is True


async def test_a_blank_context_id_never_hits_the_network() -> None:
    assert await _client().context_exists("") is False


# -- Sessions ----------------------------------------------------------------


@respx.mock
async def test_create_session_persists_the_context() -> None:
    route = respx.post(f"{BB_API_BASE}/sessions").mock(
        return_value=httpx.Response(201, json={"id": "sess-1", "connectUrl": "wss://cdp"})
    )

    await _client().create_session("ctx-1")

    body = __import__("json").loads(route.calls.last.request.content)
    # persist=true is what saves the logins; without it the Context is read-only
    # and nothing the user signs into is kept.
    assert body["browserSettings"]["context"] == {"id": "ctx-1", "persist": True}
    assert body["projectId"] == PROJECT
    assert body["timeout"] == SESSION_TIMEOUT_SECONDS


@respx.mock
async def test_create_session_without_a_context_is_ephemeral_not_broken() -> None:
    route = respx.post(f"{BB_API_BASE}/sessions").mock(
        return_value=httpx.Response(201, json={"id": "sess-1", "connectUrl": "wss://cdp"})
    )

    await _client().create_session(None)

    assert (
        "context"
        not in __import__("json").loads(route.calls.last.request.content)["browserSettings"]
    )


@respx.mock
async def test_the_viewport_matches_the_one_the_agent_clicks_against() -> None:
    """page_info reports coordinates in this viewport; clicks use them."""
    route = respx.post(f"{BB_API_BASE}/sessions").mock(
        return_value=httpx.Response(201, json={"id": "sess-1", "connectUrl": "wss://cdp"})
    )

    await _client().create_session("ctx-1")

    body = __import__("json").loads(route.calls.last.request.content)
    assert body["browserSettings"]["viewport"] == VIEWPORT


@respx.mock
async def test_a_session_response_with_no_id_is_an_error() -> None:
    respx.post(f"{BB_API_BASE}/sessions").mock(return_value=httpx.Response(201, json={}))

    with pytest.raises(RuntimeError, match="session id"):
        await _client().create_session("ctx-1")


@respx.mock
async def test_an_http_error_creating_a_session_propagates() -> None:
    respx.post(f"{BB_API_BASE}/sessions").mock(return_value=httpx.Response(402, json={}))

    with pytest.raises(httpx.HTTPStatusError):
        await _client().create_session("ctx-1")


# -- Live view ---------------------------------------------------------------


@respx.mock
async def test_live_view_prefers_the_fullscreen_url() -> None:
    respx.get(f"{BB_API_BASE}/sessions/sess-1/debug").mock(
        return_value=httpx.Response(
            200,
            json={"debuggerFullscreenUrl": "https://full", "debuggerUrl": "https://framed"},
        )
    )
    assert await _client().live_view_url("sess-1") == "https://full"


@respx.mock
async def test_live_view_falls_back_to_the_framed_url() -> None:
    respx.get(f"{BB_API_BASE}/sessions/sess-1/debug").mock(
        return_value=httpx.Response(200, json={"debuggerUrl": "https://framed"})
    )
    assert await _client().live_view_url("sess-1") == "https://framed"


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={}),
        httpx.Response(200, json=["unexpected"]),
    ],
)
async def test_a_missing_live_view_does_not_take_the_browser_down(response) -> None:
    """The agent can already drive this browser; only the user's view is lost."""
    respx.get(f"{BB_API_BASE}/sessions/sess-1/debug").mock(return_value=response)
    assert await _client().live_view_url("sess-1") == ""


@respx.mock
async def test_an_unreachable_debug_endpoint_returns_no_url_rather_than_raising() -> None:
    respx.get(f"{BB_API_BASE}/sessions/sess-1/debug").mock(side_effect=httpx.ConnectError("down"))
    assert await _client().live_view_url("sess-1") == ""


# -- Release -----------------------------------------------------------------


@respx.mock
async def test_release_asks_the_platform_to_end_the_session_now() -> None:
    route = respx.post(f"{BB_API_BASE}/sessions/sess-1").mock(
        return_value=httpx.Response(200, json={"id": "sess-1", "status": "COMPLETED"})
    )

    await _client().release_session("sess-1")

    # Without REQUEST_RELEASE the session bills until its own timeout.
    assert __import__("json").loads(route.calls.last.request.content) == {
        "status": "REQUEST_RELEASE"
    }


@respx.mock
async def test_a_failed_release_is_visible_rather_than_swallowed() -> None:
    """Reporting a clean close over a still-running browser hides a live bill."""
    respx.post(f"{BB_API_BASE}/sessions/sess-1").mock(return_value=httpx.Response(500))

    with pytest.raises(httpx.HTTPStatusError):
        await _client().release_session("sess-1")
