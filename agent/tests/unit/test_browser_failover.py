"""A vendor with a key and no balance must not leave the user without a browser.

Seen on a real call (2026-11-02): Browser Use answered 402 and the raw httpx
message, URL included, reached the model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.browser import browser_session as session_module
from src.browser.browser_session import CloudBrowserSession
from src.browser.providers import (
    BROWSER_USE,
    BROWSERBASE,
    LOCAL,
    FailoverBrowserProvider,
    LaunchedBrowser,
    ProviderUnavailableError,
    _resting,
    create_browser_provider,
)
from src.domain.tool_gating import BROWSER_TOOLS, unavailable_builtin_tools
from src.settings import Settings


@pytest.fixture(autouse=True)
def _clean_rest() -> Any:
    _resting.clear()
    yield
    _resting.clear()


def _status_error(status: int) -> httpx.HTTPStatusError:
    url = "https://api.browser-use.com/api/v3/browsers"
    request = httpx.Request("POST", url)
    return httpx.HTTPStatusError(
        f"Client error for url {url}",
        request=request,
        response=httpx.Response(status, request=request),
    )


def _launched(vendor: str) -> LaunchedBrowser:
    return LaunchedBrowser(browser_id=f"{vendor}-1", vendor=vendor, cdp_http_url="http://cdp")


def _vendor(vendor: str, *, fails_with: Exception | None = None) -> AsyncMock:
    provider = AsyncMock()
    provider.vendor = vendor
    provider.launch.side_effect = fails_with
    if fails_with is None:
        provider.launch.return_value = _launched(vendor)
    return provider


async def test_no_balance_moves_on_to_the_next_vendor() -> None:
    broke = _vendor(BROWSER_USE, fails_with=_status_error(402))
    working = _vendor(BROWSERBASE)
    failover = FailoverBrowserProvider(
        [(BROWSER_USE, lambda: broke), (BROWSERBASE, lambda: working)]
    )

    launched = await failover.launch("user-1")

    assert launched.vendor == BROWSERBASE
    assert failover.vendor == BROWSERBASE
    await failover.release(launched.browser_id)
    working.release.assert_awaited_once_with(launched.browser_id)
    broke.release.assert_not_awaited()


async def test_with_every_vendor_down_the_message_names_the_real_cause() -> None:
    broke = _vendor(BROWSER_USE, fails_with=_status_error(402))
    failover = FailoverBrowserProvider([(BROWSER_USE, lambda: broke)])

    with pytest.raises(ProviderUnavailableError) as caught:
        await failover.launch("user-1")

    text = str(caught.value)
    assert "Browser Use has no credit" in text
    assert "not the website" in text
    assert "api.browser-use.com" not in text


@pytest.mark.parametrize(
    ("status", "words"),
    [
        (401, "rejected our API key"),
        (403, "rejected our API key"),
        (429, "usage limit"),
        (500, "could not be reached"),
    ],
)
async def test_each_status_is_named(status: int, words: str) -> None:
    provider = _vendor(BROWSER_USE, fails_with=_status_error(status))
    with pytest.raises(ProviderUnavailableError, match=words):
        await FailoverBrowserProvider([(BROWSER_USE, lambda: provider)]).launch("user-1")


async def test_a_vendor_with_no_balance_is_not_asked_again_on_every_request() -> None:
    broke = _vendor(BROWSER_USE, fails_with=_status_error(402))
    for _ in range(2):
        with pytest.raises(ProviderUnavailableError, match="no credit"):
            await FailoverBrowserProvider([(BROWSER_USE, lambda: broke)]).launch("user-1")
    assert broke.launch.await_count == 1


async def test_an_outage_is_retried_next_time() -> None:
    down = _vendor(BROWSER_USE, fails_with=httpx.ConnectError("boom"))
    for _ in range(2):
        with pytest.raises(ProviderUnavailableError):
            await FailoverBrowserProvider([(BROWSER_USE, lambda: down)]).launch("user-1")
    assert down.launch.await_count == 2


async def test_a_bad_user_id_is_not_the_vendors_fault() -> None:
    provider = _vendor(BROWSER_USE, fails_with=ValueError("a browser session needs a real user id"))
    other = _vendor(BROWSERBASE)
    failover = FailoverBrowserProvider(
        [(BROWSER_USE, lambda: provider), (BROWSERBASE, lambda: other)]
    )
    with pytest.raises(ValueError):
        await failover.launch("")
    other.launch.assert_not_awaited()


async def test_a_vendor_is_built_once() -> None:
    provider = _vendor(BROWSER_USE)
    build = MagicMock(return_value=provider)
    failover = FailoverBrowserProvider([(BROWSER_USE, build)])
    await failover.launch("user-1")
    await failover.launch("user-1")
    build.assert_called_once()


async def test_releasing_a_browser_nobody_launched_does_nothing() -> None:
    provider = _vendor(BROWSER_USE)
    await FailoverBrowserProvider([(BROWSER_USE, lambda: provider)]).release("unknown")
    provider.release.assert_not_awaited()


async def test_quit_reaches_a_vendor_that_can() -> None:
    provider = _vendor(LOCAL)
    failover = FailoverBrowserProvider([(LOCAL, lambda: provider)])
    launched = await failover.launch("user-1")
    await failover.quit(launched.browser_id)
    provider.quit.assert_awaited_once_with(launched.browser_id)


def test_each_vendor_is_built_from_its_own_settings() -> None:
    settings = Settings(browserbase_api_key="bb", browser_use_api_key="bu")
    candidates = dict(create_browser_provider(settings)._candidates)
    assert candidates[BROWSER_USE]().vendor == BROWSER_USE
    with patch("src.browser.browserbase.create_context_store"):
        assert candidates[BROWSERBASE]().vendor == BROWSERBASE


async def test_quit_only_reaches_a_vendor_that_can() -> None:
    provider = _vendor(BROWSER_USE)
    del provider.quit
    failover = FailoverBrowserProvider([(BROWSER_USE, lambda: provider)])
    launched = await failover.launch("user-1")
    await failover.quit(launched.browser_id)
    await failover.quit("unknown")


def test_every_configured_vendor_stands_behind_the_chosen_one() -> None:
    provider = create_browser_provider(
        Settings(browserbase_api_key="bb", browser_use_api_key="bu", browser_provider=BROWSER_USE)
    )
    assert [vendor for vendor, _ in provider._candidates] == [BROWSER_USE, BROWSERBASE]


def test_local_is_only_used_when_asked_for() -> None:
    assert [
        vendor
        for vendor, _ in create_browser_provider(
            Settings(browser_use_api_key="bu", browser_provider="browserbase")
        )._candidates
    ] == [BROWSER_USE]
    assert [
        vendor
        for vendor, _ in create_browser_provider(
            Settings(browser_use_api_key="bu", browser_provider=LOCAL)
        )._candidates
    ] == [LOCAL, BROWSER_USE]


def test_local_needs_no_credential_to_offer_the_browser_tools() -> None:
    assert not (BROWSER_TOOLS & unavailable_builtin_tools(Settings(browser_provider=LOCAL)))
    assert unavailable_builtin_tools(Settings()) >= BROWSER_TOOLS


class TestALocalWindow:
    """It belongs to the user: not billed, not timed out, closed only when asked."""

    @pytest.fixture
    def session(self) -> Any:
        session = CloudBrowserSession()
        session._vendor = LOCAL
        session._provider = AsyncMock()
        session._browser_id = "local-1"
        session._cdp = AsyncMock()
        return session

    async def test_going_idle_sets_no_timer(self, session: Any) -> None:
        session._reset_idle_timer()
        assert session._idle_timer is None

    async def test_its_minutes_are_not_billed(self, session: Any) -> None:
        session._usage_tracker = MagicMock()
        session._started_at = 1.0
        session._record_browser_minutes()
        session._usage_tracker.record_external_usage.assert_not_called()

    async def test_the_session_ending_leaves_it_open(self, session: Any) -> None:
        provider = session._provider
        await session.close()
        provider.quit.assert_not_awaited()

    async def test_the_user_closing_it_closes_it(self, session: Any) -> None:
        provider = session._provider
        await session.close(quit_browser=True)
        provider.quit.assert_awaited_once_with("local-1")

    async def test_a_provider_with_nothing_to_quit_is_only_released(self, session: Any) -> None:
        provider = MagicMock(spec=["release"])
        provider.release = AsyncMock()
        session._provider = provider
        await session.close(quit_browser=True)
        provider.release.assert_awaited_once_with("local-1")

    async def test_starting_one_brings_it_forward_at_its_own_size(self) -> None:
        session = CloudBrowserSession()
        session._provider = AsyncMock()
        session._provider.launch.return_value = LaunchedBrowser(
            browser_id="local-1", vendor=LOCAL, cdp_http_url="http://127.0.0.1:1"
        )
        cdp = AsyncMock()
        with patch.object(session_module, "CDPConnection", return_value=cdp):
            await session.start("user-1")
        sent = [call.args[0] for call in cdp.send.await_args_list]
        assert "Page.bringToFront" in sent
        assert "Emulation.setDeviceMetricsOverride" not in sent

    async def test_a_cloud_browser_still_has_a_timer(self, session: Any) -> None:
        session._vendor = BROWSER_USE
        session._reset_idle_timer()
        assert session._idle_timer is not None
        session._cancel_idle_timer()
        assert session_module.IDLE_TIMEOUT_SECONDS > 0
