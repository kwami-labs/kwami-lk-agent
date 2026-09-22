"""Opening a search result the user clicked, and the two gates it must not skip.

`kwami-app` publishes `{type: "browser_open_request", url}` when the user picks
a search result. `DataMessageRouter` had no route for it, so the message was
decoded, matched nothing, and was dropped at a `debug` line -- clicking a result
did nothing, silently, which is the worst shape a failure can take because
neither side reports it.

Routing it is easy; routing it *through the same gate as the model's own
navigation* is the part worth testing. Two things must survive:

* URL validation, because the same envelope can carry a URL the model was
  reading on a page a moment earlier. "The frontend published it" is not the
  same as "a human asked for it".
* The blank-`kwami_id` refusal, because a browser profile holds the user's
  cookies and logins and this message can arrive before any browser exists.

Both live in `navigate_to`, so these tests assert the handler goes through it
rather than reimplementing the checks -- a second copy is a second thing to keep
in step, and the copy that drifts is the one nobody is watching.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.runtime.dispatch import DataMessageRouter
from src.session import SessionState


class FakeSession:
    def update_agent(self, agent: Any) -> None:  # pragma: no cover - unused here
        pass


def _router(agent: Any, room: Any = None) -> tuple[DataMessageRouter, SessionState]:
    state = SessionState(current_agent=agent)
    state.room = room
    return (
        DataMessageRouter(session=FakeSession(), state=state, room=room),
        state,
    )


async def _drain(state: SessionState) -> None:
    """Let the spawned handler finish."""
    tasks = list(state._background_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


class RecordingAgent(KwamiAgent):
    """A real agent whose navigation is recorded instead of performed.

    Subclassed rather than faked: the handler resolves `navigate_to` off the
    agent, and the point of the test is that it goes through *that* method.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.navigated: list[str] = []

    async def navigate_to(self, context: Any, url: str) -> str:  # type: ignore[override]
        self.navigated.append(url)
        return f"Opening {url}."


# -- the route exists -------------------------------------------------------


async def test_the_message_type_is_routed() -> None:
    """It used to fall through to "no handler" and be dropped."""
    agent = RecordingAgent(config=KwamiConfig(kwami_id="tenant-1"))
    router, state = _router(agent)

    handled = router.handle({"type": "browser_open_request", "url": "https://example.com"})

    assert handled == "browser_open_request", "the message is still unrouted"
    await _drain(state)


async def test_the_url_reaches_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = RecordingAgent(config=KwamiConfig(kwami_id="tenant-1"))
    router, state = _router(agent)

    router.handle({"type": "browser_open_request", "url": "https://example.com/article"})
    await _drain(state)

    assert agent.navigated == ["https://example.com/article"]


async def test_surrounding_whitespace_is_trimmed() -> None:
    agent = RecordingAgent(config=KwamiConfig(kwami_id="tenant-1"))
    router, state = _router(agent)

    router.handle({"type": "browser_open_request", "url": "  https://example.com  "})
    await _drain(state)

    assert agent.navigated == ["https://example.com"]


# -- malformed payloads -----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        {"type": "browser_open_request"},
        {"type": "browser_open_request", "url": ""},
        {"type": "browser_open_request", "url": "   "},
        {"type": "browser_open_request", "url": None},
        {"type": "browser_open_request", "url": 42},
        {"type": "browser_open_request", "url": ["https://example.com"]},
    ],
    ids=["missing", "empty", "blank", "null", "number", "list"],
)
async def test_a_payload_with_no_usable_url_navigates_nowhere(message: dict[str, Any]) -> None:
    """This is wire data; a malformed packet must not raise in the data handler."""
    agent = RecordingAgent(config=KwamiConfig(kwami_id="tenant-1"))
    router, state = _router(agent)

    router.handle(message)
    await _drain(state)

    assert agent.navigated == []


async def test_a_request_before_the_agent_exists_is_ignored() -> None:
    state = SessionState(current_agent=None)
    router = DataMessageRouter(session=FakeSession(), state=state, room=None)

    router.handle({"type": "browser_open_request", "url": "https://example.com"})

    assert not state._background_tasks


# -- the gates it must not skip ---------------------------------------------


async def test_a_safe_url_does_reach_the_browser() -> None:
    """The positive control for the two refusal tests below.

    Those assert that `_get_browser_session` is never called. Without this,
    they would pass just as well if the handler were broken, or if `navigate_to`
    bailed for some unrelated reason -- proving nothing about the gates at all.
    """
    agent = KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))
    started: list[str] = []

    class _Session:
        is_active = False

        async def start(self, user_id: str, url: str | None = None) -> str:
            started.append(url or "")
            return "https://live.example/view"

    async def _get() -> Any:
        return _Session()

    agent._get_browser_session = _get  # type: ignore[method-assign]
    router, state = _router(agent)

    router.handle({"type": "browser_open_request", "url": "https://example.com/article"})
    await _drain(state)

    assert started == ["https://example.com/article"], (
        "a safe URL never reached the browser, so the refusal tests below prove nothing"
    )


async def test_an_unsafe_url_is_refused() -> None:
    """The frontend is not a trusted source of URLs.

    The same envelope carries a link the model was reading on a page a moment
    earlier, so the address has to clear the same check the model's own
    navigation does -- loopback and link-local metadata would otherwise be
    reachable from a profile holding the user's cookies.
    """
    agent = KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))
    started: list[str] = []

    async def _never() -> Any:
        started.append("browser")
        raise AssertionError("a browser was started for an unsafe URL")

    agent._get_browser_session = _never  # type: ignore[method-assign]
    router, state = _router(agent)

    for unsafe in (
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/admin",
        "javascript:alert(1)",
        "file:///etc/passwd",
    ):
        router.handle({"type": "browser_open_request", "url": unsafe})
        await _drain(state)

    assert started == [], "an unsafe URL reached the browser"


async def test_a_session_with_no_tenant_does_not_start_a_browser() -> None:
    """A browser profile carries logins; a shared one leaks them between users.

    This message can arrive before any browser exists, which is precisely when
    that check matters -- and why the handler goes through `navigate_to` rather
    than reaching for the session itself.
    """
    agent = KwamiAgent(config=KwamiConfig(kwami_id=""))
    started: list[str] = []

    async def _never() -> Any:
        started.append("browser")
        raise AssertionError("a browser was started with no tenant id")

    agent._get_browser_session = _never  # type: ignore[method-assign]
    router, state = _router(agent)

    router.handle({"type": "browser_open_request", "url": "https://example.com"})
    await _drain(state)

    assert started == []


# -- the close route still works --------------------------------------------


async def test_open_and_close_are_different_routes() -> None:
    """Adding one must not shadow the other; they share a prefix."""
    agent = RecordingAgent(config=KwamiConfig(kwami_id="tenant-1"))
    router, state = _router(agent)

    assert router.handle({"type": "browser_close_request"}) == "browser_close_request"
    assert (
        router.handle({"type": "browser_open_request", "url": "https://example.com"})
        == "browser_open_request"
    )
    await _drain(state)

    assert agent.navigated == ["https://example.com"]
