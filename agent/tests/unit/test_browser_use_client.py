"""The Browser Use Cloud client and its provider adapter.

The alternative vendor to Browserbase. It differs in exactly two ways that
matter, and both are covered here: profiles are addressed by *name*, so nothing
has to be persisted on our side; and the CDP endpoint is an HTTP base to
discover a page target through, not a browser-level socket.

The invariant it shares with Browserbase is the one that matters most: profiles
carry cookies and logins, so no browser starts without a real tenant.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.browser.cloud_browser import BU_API_BASE, BrowserUseClient, BrowserUseProvider
from src.browser.providers import BROWSER_USE

API_KEY = "bu_test_key"


def _client() -> BrowserUseClient:
    return BrowserUseClient(api_key=API_KEY)


def test_refuses_to_build_without_a_key() -> None:
    from src.settings import Settings, set_settings

    set_settings(Settings())
    try:
        with pytest.raises(ValueError, match="BROWSER_USE_API_KEY"):
            BrowserUseClient()
    finally:
        set_settings(None)


# -- Browsers ----------------------------------------------------------------


@respx.mock
async def test_creating_a_browser_sends_the_profile_and_the_key() -> None:
    route = respx.post(f"{BU_API_BASE}/browsers").mock(
        return_value=httpx.Response(
            200, json={"id": "b1", "liveUrl": "https://live", "cdpUrl": "https://cdp"}
        )
    )

    await _client().create_browser(profile_id="p1", timeout_minutes=15)

    request = route.calls.last.request
    assert request.headers["X-Browser-Use-API-Key"] == API_KEY
    body = __import__("json").loads(request.content)
    assert body["profileId"] == "p1"
    assert body["timeout"] == 15


@respx.mock
async def test_a_browser_without_a_proxy_country_sends_null() -> None:
    """Omitting the key and sending null are different to this API."""
    route = respx.post(f"{BU_API_BASE}/browsers").mock(
        return_value=httpx.Response(200, json={"id": "b1", "cdpUrl": "https://cdp"})
    )

    await _client().create_browser(proxy_country=None)

    assert __import__("json").loads(route.calls.last.request.content)["proxyCountryCode"] is None


@respx.mock
async def test_stopping_a_browser_patches_its_status() -> None:
    route = respx.patch(f"{BU_API_BASE}/browsers/b1").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "stopped"})
    )

    await _client().stop_browser("b1")

    assert __import__("json").loads(route.calls.last.request.content) == {"status": "stopped"}


@respx.mock
async def test_reading_a_browser_returns_its_details() -> None:
    respx.get(f"{BU_API_BASE}/browsers/b1").mock(
        return_value=httpx.Response(200, json={"id": "b1", "status": "running"})
    )

    assert (await _client().get_browser("b1"))["status"] == "running"


# -- Profiles ----------------------------------------------------------------


@respx.mock
async def test_an_existing_profile_is_reused_rather_than_duplicated() -> None:
    """Creating a second profile for a user loses everything they signed into."""
    respx.get(f"{BU_API_BASE}/profiles").mock(
        return_value=httpx.Response(200, json={"items": [{"id": "p-existing", "name": "user-1"}]})
    )
    create = respx.post(f"{BU_API_BASE}/profiles")

    assert await _client().get_or_create_profile("user-1") == "p-existing"
    assert not create.called


@respx.mock
async def test_a_first_time_user_gets_a_profile_named_after_them() -> None:
    respx.get(f"{BU_API_BASE}/profiles").mock(return_value=httpx.Response(200, json={"items": []}))
    create = respx.post(f"{BU_API_BASE}/profiles").mock(
        return_value=httpx.Response(200, json={"id": "p-new"})
    )

    assert await _client().get_or_create_profile("user-1") == "p-new"
    assert __import__("json").loads(create.calls.last.request.content) == {"name": "user-1"}


@respx.mock
async def test_a_partial_name_match_is_not_treated_as_the_same_user() -> None:
    """The query is a substring search; "user-1" also matches "user-10"."""
    respx.get(f"{BU_API_BASE}/profiles").mock(
        return_value=httpx.Response(200, json={"items": [{"id": "p-10", "name": "user-10"}]})
    )
    respx.post(f"{BU_API_BASE}/profiles").mock(
        return_value=httpx.Response(200, json={"id": "p-new"})
    )

    assert await _client().get_or_create_profile("user-1") == "p-new"


# -- Provider adapter --------------------------------------------------------


class FakeClient:
    def __init__(self, *, browser: dict | None = None, profile_error: Exception | None = None):
        self.browser = browser or {
            "id": "b1",
            "liveUrl": "https://live",
            "cdpUrl": "https://cdp-1.browser-use.com",
        }
        self.profile_error = profile_error
        self.stopped: list[str] = []
        self.profiles: list[str] = []

    async def get_or_create_profile(self, user_id: str) -> str:
        if self.profile_error:
            raise self.profile_error
        self.profiles.append(user_id)
        return "p1"

    async def create_browser(self, profile_id=None, timeout_minutes=15, proxy_country="us"):
        self.last_profile = profile_id
        return self.browser

    async def stop_browser(self, browser_id: str) -> None:
        self.stopped.append(browser_id)


@pytest.mark.parametrize("user_id", ["", "   "])
async def test_the_provider_refuses_to_launch_without_a_tenant(user_id: str) -> None:
    client = FakeClient()

    with pytest.raises(ValueError):
        await BrowserUseProvider(client=client).launch(user_id)

    assert client.profiles == []


async def test_the_provider_reports_an_http_cdp_endpoint() -> None:
    """The session layer branches on this to skip page-target attachment."""
    launched = await BrowserUseProvider(client=FakeClient()).launch("user-1")

    assert launched.cdp_http_url == "https://cdp-1.browser-use.com"
    assert launched.cdp_ws_url == ""
    assert launched.vendor == BROWSER_USE
    assert launched.persistence_id == "p1"


async def test_a_profile_lookup_failure_degrades_to_an_ephemeral_browser() -> None:
    """Browsing without saved logins beats not browsing."""
    client = FakeClient(profile_error=RuntimeError("profiles api down"))

    launched = await BrowserUseProvider(client=client).launch("user-1")

    assert launched.persistence_id == ""
    assert client.last_profile is None


async def test_a_browser_with_no_id_is_an_error() -> None:
    client = FakeClient(browser={"liveUrl": "https://live", "cdpUrl": "https://cdp"})

    with pytest.raises(RuntimeError, match="browser id"):
        await BrowserUseProvider(client=client).launch("user-1")


async def test_a_browser_with_no_cdp_url_is_released_not_stranded() -> None:
    """It is already billing and nothing can drive it."""
    client = FakeClient(browser={"id": "b1", "liveUrl": "https://live"})

    with pytest.raises(RuntimeError, match="cdpUrl"):
        await BrowserUseProvider(client=client).launch("user-1")

    assert client.stopped == ["b1"]


async def test_releasing_nothing_is_a_no_op() -> None:
    client = FakeClient()
    await BrowserUseProvider(client=client).release("")
    assert client.stopped == []
