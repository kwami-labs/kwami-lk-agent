"""Browserbase is a second cloud-browser vendor, held to the first one's rules.

A new provider that does not re-prove the invariants is a regression wearing a
new name. The three that matter, all of them about a browser that carries the
user's real logins:

* **No session without a real user id.** A shared or blank profile hands one
  user's authenticated sessions to the next.
* **Nothing billed is left running.** A session that is created but never
  becomes usable must be released, not left to its own timeout.
* **Persistence is per user and durable.** The context id is the only handle
  Browserbase gives back, so losing it silently signs the user out of
  everything and orphans a paid context.
"""

from __future__ import annotations

import pytest

from src.browser.browserbase import BrowserbaseProvider, context_name_for
from src.browser.context_store import InMemoryContextStore
from src.browser.providers import BROWSERBASE


class FakeBrowserbaseClient:
    """Records what the provider asked the platform to do."""

    def __init__(
        self,
        *,
        existing_contexts: set[str] | None = None,
        create_context_error: Exception | None = None,
        connect_url: str | None = "wss://connect.browserbase.com?sessionId=s1",
    ) -> None:
        self.existing_contexts = existing_contexts or set()
        self.create_context_error = create_context_error
        self.connect_url = connect_url
        self.created_contexts: list[str] = []
        self.created_sessions: list[str | None] = []
        self.released: list[str] = []
        self._next_context = 0

    async def create_context(self, name: str) -> str:
        if self.create_context_error:
            raise self.create_context_error
        self._next_context += 1
        context_id = f"ctx-{self._next_context}"
        self.created_contexts.append(name)
        self.existing_contexts.add(context_id)
        return context_id

    async def context_exists(self, context_id: str) -> bool:
        return context_id in self.existing_contexts

    async def create_session(self, context_id=None, *, persist=True, timeout_seconds=900):
        self.created_sessions.append(context_id)
        session = {"id": "sess-1"}
        if self.connect_url:
            session["connectUrl"] = self.connect_url
        return session

    async def live_view_url(self, session_id: str) -> str:
        return f"https://browserbase.com/devtools/{session_id}"

    async def release_session(self, session_id: str) -> None:
        self.released.append(session_id)


def _provider(client: FakeBrowserbaseClient, store=None) -> BrowserbaseProvider:
    return BrowserbaseProvider(client=client, context_store=store or InMemoryContextStore())


# -- A browser always belongs to someone -------------------------------------


@pytest.mark.parametrize("user_id", ["", "   ", "\t\n"])
async def test_refuses_to_launch_without_a_real_user_id(user_id: str) -> None:
    client = FakeBrowserbaseClient()

    with pytest.raises(ValueError):
        await _provider(client).launch(user_id)

    assert client.created_sessions == [], "a browser was rented for nobody"
    assert client.created_contexts == []


# -- Persistence -------------------------------------------------------------


async def test_first_launch_creates_and_records_a_context() -> None:
    client = FakeBrowserbaseClient()
    store = InMemoryContextStore()

    launched = await _provider(client, store).launch("user-1")

    assert client.created_contexts == [context_name_for("user-1")]
    assert launched.persistence_id == "ctx-1"
    assert await store.get("user-1", BROWSERBASE) == "ctx-1"


async def test_second_launch_reuses_the_stored_context() -> None:
    """Otherwise every session starts signed out and orphans a paid context."""
    client = FakeBrowserbaseClient()
    store = InMemoryContextStore()
    provider = _provider(client, store)

    await provider.launch("user-1")
    await provider.launch("user-1")

    assert client.created_contexts == [context_name_for("user-1")], "created a second context"
    assert client.created_sessions == ["ctx-1", "ctx-1"]


async def test_two_users_never_share_a_context() -> None:
    client = FakeBrowserbaseClient()
    store = InMemoryContextStore()
    provider = _provider(client, store)

    first = await provider.launch("user-1")
    second = await provider.launch("user-2")

    assert first.persistence_id != second.persistence_id


async def test_a_context_deleted_upstream_is_replaced_not_reused() -> None:
    client = FakeBrowserbaseClient()
    store = InMemoryContextStore()
    await store.put("user-1", BROWSERBASE, "ctx-gone")

    launched = await _provider(client, store).launch("user-1")

    assert launched.persistence_id == "ctx-1"
    assert await store.get("user-1", BROWSERBASE) == "ctx-1"


async def test_browsing_still_works_when_a_context_cannot_be_created() -> None:
    """Degrade to an ephemeral session rather than refusing to browse at all."""
    client = FakeBrowserbaseClient(create_context_error=RuntimeError("quota exceeded"))

    launched = await _provider(client).launch("user-1")

    assert launched.persistence_id == ""
    assert client.created_sessions == [None]
    assert launched.browser_id == "sess-1"


# -- Nothing billed is left running ------------------------------------------


async def test_a_session_with_no_connect_url_is_released() -> None:
    """It is already billing and nothing can drive it."""
    client = FakeBrowserbaseClient(connect_url=None)

    with pytest.raises(RuntimeError, match="connectUrl"):
        await _provider(client).launch("user-1")

    assert client.released == ["sess-1"]


async def test_release_is_a_no_op_without_a_browser_id() -> None:
    client = FakeBrowserbaseClient()
    await _provider(client).release("")
    assert client.released == []


# -- What the session layer is handed ----------------------------------------


async def test_launch_reports_a_websocket_cdp_endpoint() -> None:
    """Browserbase gives a browser-level socket, not an HTTP CDP base.

    `CloudBrowserSession.start` branches on exactly this to decide whether it
    must attach to a page target, so getting it wrong means Page.enable fails
    and the panel never renders.
    """
    launched = await _provider(FakeBrowserbaseClient()).launch("user-1")

    assert launched.cdp_ws_url.startswith("wss://")
    assert launched.cdp_http_url == ""
    assert launched.vendor == BROWSERBASE
    assert launched.live_url


def test_context_names_fit_the_vendor_limit() -> None:
    assert len(context_name_for("u" * 500)) <= 128
