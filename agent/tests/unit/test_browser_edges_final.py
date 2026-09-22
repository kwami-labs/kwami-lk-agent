"""The last corners of the browser stack: metering, teardown, URLs and persistence.

Four small areas, and three of them cost real money or real privacy if they are
wrong:

* **Metering.** A browser that is rented and never billed is invisible margin
  loss, and every release path has to record its minutes -- including the ones
  that run because something failed.
* **Releasing a half-started browser.** If `start` fails after the rental but
  before the session is usable, nothing downstream considers it active, so
  nothing else will ever release it. It bills until the vendor's own timeout.
* **URL safety.** The profile carries the user's cookies and logins, so a
  hostname that resolves to a private address is a request made from inside
  their session.
* **Context persistence.** The Kwami API remembers which Browserbase context
  belongs to which user; without it every session starts logged out and the
  old context is orphaned but still billed for storage.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import respx

from src.browser.browser_session import CloudBrowserSession
from src.browser.context_store import KwamiApiContextStore
from src.browser.providers import BROWSER_USE
from src.browser.safety import UnsafeURLError, truncate_for_llm, validate_url_async


class Tracker:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, float]] = []

    def record_external_usage(
        self, kind: str, model: str, units_used: float = 1.0, **kwargs: Any
    ) -> None:
        self.records.append((kind, model, units_used))


class ClosableCDP:
    def __init__(self, fail: bool = False) -> None:
        self.closed = False
        self._fail = fail

    async def close(self) -> None:
        if self._fail:
            raise RuntimeError("socket already gone")
        self.closed = True


class Provider:
    def __init__(self, fail_release: bool = False) -> None:
        self.released: list[str] = []
        self._fail = fail_release

    async def release(self, browser_id: str) -> None:
        if self._fail:
            raise RuntimeError("vendor refused")
        self.released.append(browser_id)


# -- metering ---------------------------------------------------------------


def test_a_zero_length_rental_is_not_billed() -> None:
    """A browser released in the same instant owes nothing."""
    session = CloudBrowserSession(usage_tracker=Tracker())
    import time

    session._started_at = time.monotonic() + 10  # "started" in the future

    session._record_browser_minutes()

    assert session._usage_tracker.records == []  # type: ignore[union-attr]


def test_a_tracker_that_throws_does_not_break_teardown() -> None:
    """Metering is a side effect of releasing; it must not prevent the release."""

    class Exploding:
        def record_external_usage(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("metering down")

    import time

    session = CloudBrowserSession(usage_tracker=Exploding())
    session._started_at = time.monotonic() - 60

    session._record_browser_minutes()  # must not raise


# -- releasing a browser that never became usable ---------------------------


async def test_a_half_started_browser_is_released_and_billed() -> None:
    """Nothing downstream sees it as active, so this is its only release path."""
    provider, cdp, tracker = Provider(), ClosableCDP(), Tracker()
    import time

    session = CloudBrowserSession(usage_tracker=tracker)
    session._provider = provider  # type: ignore[assignment]
    session._cdp = cdp  # type: ignore[assignment]
    session._browser_id = "b-1"
    session._started_at = time.monotonic() - 120

    await session._release_unusable_browser()

    assert provider.released == ["b-1"], "a rented browser was left running"
    assert cdp.closed
    assert session._browser_id is None
    assert tracker.records, "the minutes it was held for were never billed"


async def test_a_half_open_socket_that_will_not_close_is_not_fatal() -> None:
    """The release below it is the part that costs money."""
    provider = Provider()
    session = CloudBrowserSession()
    session._provider = provider  # type: ignore[assignment]
    session._cdp = ClosableCDP(fail=True)  # type: ignore[assignment]
    session._browser_id = "b-1"

    await session._release_unusable_browser()

    assert provider.released == ["b-1"], "a failed socket close cost the release"


async def test_a_vendor_that_refuses_the_release_is_logged_not_raised() -> None:
    session = CloudBrowserSession()
    session._provider = Provider(fail_release=True)  # type: ignore[assignment]
    session._browser_id = "b-1"

    await session._release_unusable_browser()

    assert session._browser_id is None


async def test_a_provider_that_cannot_be_built_is_surfaced() -> None:
    """No credentials is a refusal the user should hear, not a silent no-op."""
    session = CloudBrowserSession()

    def explode() -> Any:
        raise ValueError("BROWSERBASE_API_KEY is not set")

    import src.browser.browser_session as module

    original = module.create_browser_provider
    module.create_browser_provider = explode  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="BROWSERBASE_API_KEY"):
            await session.start("tenant-1")
    finally:
        module.create_browser_provider = original  # type: ignore[assignment]


# -- the embeddable live URL ------------------------------------------------


def test_no_live_url_is_an_empty_string() -> None:
    assert CloudBrowserSession()._embeddable_live_url() == ""


def test_browser_use_urls_get_the_dark_chromeless_parameters() -> None:
    session = CloudBrowserSession()
    session._vendor = BROWSER_USE
    session._live_url = "https://live.example/view"

    assert session._embeddable_live_url() == "https://live.example/view?theme=dark&ui=false"


def test_an_existing_query_string_is_appended_to_not_replaced() -> None:
    session = CloudBrowserSession()
    session._vendor = BROWSER_USE
    session._live_url = "https://live.example/view?session=abc"

    result = session._embeddable_live_url()

    assert "session=abc" in result, "the vendor's own parameter was dropped"
    assert result.count("?") == 1


def test_another_vendors_url_is_left_alone() -> None:
    """Browserbase's fullscreen URL is already bare; these parameters do nothing.

    Sending them is harmless and silent, which is the kind of no-op that gets
    read as "the panel is broken".
    """
    session = CloudBrowserSession()
    session._vendor = "browserbase"
    session._live_url = "https://debugger.browserbase.com/s/1/fullscreen"

    assert session._embeddable_live_url() == "https://debugger.browserbase.com/s/1/fullscreen"


# -- publishing to the frontend ---------------------------------------------


async def test_a_session_event_carries_the_page_title(room) -> None:
    session = CloudBrowserSession(room=room)

    await session._publish_session_event("update", url="https://example.com", title="Example")

    assert room.published[-1]["title"] == "Example"


# -- the idle timeout -------------------------------------------------------


async def test_the_idle_timeout_closes_a_live_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timer is what stops an abandoned browser billing indefinitely."""
    monkeypatch.setattr("src.browser.browser_session.IDLE_TIMEOUT_SECONDS", 0)
    closed: list[str] = []

    session = CloudBrowserSession()
    session._browser_id = "b-1"
    session._cdp = type("C", (), {"is_connected": True})()  # type: ignore[assignment]

    async def fake_close() -> None:
        closed.append("closed")

    session.close = fake_close  # type: ignore[method-assign]

    await session._idle_timeout()

    assert closed == ["closed"], "an idle browser was left running"


async def test_the_idle_timeout_leaves_an_already_closed_browser_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.browser.browser_session.IDLE_TIMEOUT_SECONDS", 0)
    closed: list[str] = []
    session = CloudBrowserSession()  # never started, so not active

    async def fake_close() -> None:  # pragma: no cover - must not be reached
        closed.append("closed")

    session.close = fake_close  # type: ignore[method-assign]

    await session._idle_timeout()

    assert closed == []


async def test_a_cancelled_idle_timer_exits_quietly() -> None:
    session = CloudBrowserSession()
    task = asyncio.create_task(session._idle_timeout())
    await asyncio.sleep(0)
    task.cancel()

    await task  # the CancelledError is swallowed by design


# -- URL safety -------------------------------------------------------------


async def test_a_url_with_no_hostname_is_refused() -> None:
    """`http:///path` parses cleanly and names no host."""
    with pytest.raises(UnsafeURLError, match="no hostname"):
        await validate_url_async("http:///just-a-path")


async def test_a_hostname_that_does_not_resolve_is_allowed_through() -> None:
    """A DNS failure is not evidence of a private address.

    Refusing here would block every transient resolver hiccup; the request
    itself will fail honestly a moment later.
    """
    result = await validate_url_async("https://nx-does-not-exist-kwami-test.invalid/")

    assert result.startswith("https://nx-does-not-exist-kwami-test.invalid")


async def test_a_hostname_resolving_to_a_private_address_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS rebinding: a public name pointed at the metadata service."""
    import socket as socket_module

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> list:
        return [(2, 1, 6, "", ("169.254.169.254", 0))]

    monkeypatch.setattr(socket_module, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(UnsafeURLError, match="non-public address"):
        await validate_url_async("https://looks-fine.example/")


async def test_an_unparseable_resolved_address_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`getaddrinfo` can hand back something that is not an address at all."""
    import socket as socket_module

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any) -> list:
        return [(2, 1, 6, "", ("not-an-ip", 0)), (2, 1, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket_module, "getaddrinfo", fake_getaddrinfo)

    assert await validate_url_async("https://example.com/") == "https://example.com/"


def test_truncating_nothing_gives_an_empty_string() -> None:
    """Callers pass this straight from helpers whose return type is not guaranteed."""
    assert truncate_for_llm(None) == ""


# -- the context store ------------------------------------------------------


def test_the_store_builds_its_own_http_client_once(env_setting) -> None:
    """It owns the pool when nobody handed it one, and must not build two."""
    env_setting("KWAMI_API_KEY", "key")
    env_setting("KWAMI_API_URL", "https://api.example")

    store = KwamiApiContextStore()
    first = store._client()

    assert first is store._client(), "a second client was built for the same store"
    assert store._owns_http is True


async def test_closing_releases_a_pool_the_store_built(env_setting) -> None:
    env_setting("KWAMI_API_KEY", "key")
    env_setting("KWAMI_API_URL", "https://api.example")

    store = KwamiApiContextStore()
    store._client()

    await store.aclose()

    assert store._http is None


async def test_closing_does_not_release_a_pool_it_was_given() -> None:
    """Shared pools outlive the store; closing one would break its owner."""

    class Shared:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    shared = Shared()
    store = KwamiApiContextStore(http=shared)

    await store.aclose()

    assert shared.closed is False
    assert store._http is shared


@respx.mock
async def test_listing_profiles_can_be_filtered_by_name() -> None:
    """The query is how a per-user profile is found again on the next session."""
    import httpx as httpx_module

    from src.browser.cloud_browser import BU_API_BASE, BrowserUseClient

    route = respx.get(f"{BU_API_BASE}/profiles").mock(
        return_value=httpx_module.Response(200, json={"items": [{"id": "p-1"}]})
    )

    client = BrowserUseClient(api_key="bu-test")
    await client.list_profiles()
    unfiltered = dict(route.calls[-1].request.url.params)

    await client.list_profiles(query="tenant-1")
    filtered = dict(route.calls[-1].request.url.params)

    assert "query" not in unfiltered, "an empty filter was sent as a parameter"
    assert filtered["query"] == "tenant-1"


async def test_reusing_a_live_session_without_a_url_navigates_nowhere() -> None:
    """ "Open the browser" with nothing to open must not move the page."""

    class CDP:
        is_connected = True

        def __init__(self) -> None:
            self.navigated: list[str] = []

        async def navigate(self, url: str) -> None:  # pragma: no cover - must not run
            self.navigated.append(url)

    cdp = CDP()
    session = CloudBrowserSession()
    session._cdp = cdp  # type: ignore[assignment]
    session._browser_id = "b-1"
    session._live_url = "https://live.example/view"

    assert await session.start("tenant-1") == "https://live.example/view"
    assert cdp.navigated == []


async def test_releasing_a_browser_that_was_never_rented() -> None:
    """`start` can fail before the vendor call; there is nothing to release."""
    provider = Provider()
    session = CloudBrowserSession()
    session._provider = provider  # type: ignore[assignment]
    session._browser_id = None

    await session._release_unusable_browser()

    assert provider.released == []
