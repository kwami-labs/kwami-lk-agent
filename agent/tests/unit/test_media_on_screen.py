"""A song or a video plays on Jarvis's own screen, not in a browser beside it (2026-11-17).

With the app's HUD there, `play_media` finds YouTube's first video and sends it to the
video window; the playback controls follow it there. Without it, or when YouTube cannot
be searched, the browser does it as before, through named actions where it has them.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.tools import media
from src.tools.media import YOUTUBE_RESULTS, first_youtube_video

RESULTS_PAGE = (
    'var ytInitialData = {"contents":[{"videoRenderer":{"videoId":"jfKfPfyJRdk",'
    '"thumbnail":{},"title":{"runs":[{"text":"lofi hip hop radio \\u0026 beats"}]}}}]};'
)


class Publisher:
    def __init__(self, answer: Any = True) -> None:
        self.answer = answer
        self.published: list[dict[str, Any]] = []

    async def publish(self, payload: dict[str, Any], *, topic: str | None = None) -> bool:
        if isinstance(self.answer, Exception):
            raise self.answer
        self.published.append(payload)
        return self.answer


class NamedSession:
    """A browser with its own named actions, as the user's Chrome has."""

    is_active = True

    def __init__(self, answers: dict[str, str] | None = None) -> None:
        self.answers = answers or {}
        self.ran: list[tuple[str, float]] = []

    async def run_named(self, name: str, level: float = 0.0) -> str:
        self.ran.append((name, level))
        return self.answers.get(name, "ok")

    async def evaluate_js(self, expression: str) -> str:
        raise AssertionError("a browser with named actions is never handed a script")


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media, "RESULTS_SETTLE_SECONDS", 0)
    monkeypatch.setattr(media, "PLAYBACK_SETTLE_SECONDS", 0)


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> KwamiAgent:
    instance = KwamiAgent(config=KwamiConfig())
    instance._navigated = []  # type: ignore[attr-defined]
    instance._app = Publisher()  # type: ignore[attr-defined]

    async def fake_navigate(context: Any, url: str) -> str:
        instance._navigated.append(url)  # type: ignore[attr-defined]
        return f"Navigating to {url}."

    monkeypatch.setattr(instance, "navigate_to", fake_navigate)
    monkeypatch.setattr(instance, "_publisher", lambda context=None: instance._app)
    monkeypatch.setattr(instance, "_registered_client_tool_names", lambda: {"show_on_screen"})
    return instance


def _attach(agent: KwamiAgent, session: Any) -> None:
    async def _get() -> Any:
        return session

    agent._get_browser_session = _get  # type: ignore[method-assign]


# -- finding the video ----------------------------------------------------------------------


@respx.mock
async def test_youtubes_first_video_is_found_with_its_title() -> None:
    route = respx.get(YOUTUBE_RESULTS).mock(return_value=httpx.Response(200, text=RESULTS_PAGE))
    assert await first_youtube_video("lofi") == (
        "https://www.youtube.com/watch?v=jfKfPfyJRdk",
        "lofi hip hop radio & beats",
    )
    request = route.calls.last.request
    assert request.url.params["search_query"] == "lofi"
    assert "SOCS=CAI" in request.headers["cookie"], "the EU consent page would stand in front"


@respx.mock
async def test_a_title_that_does_not_decode_is_kept_as_it_came() -> None:
    page = '"videoRenderer":{"videoId":"abcdefghijk","title":{"runs":[{"text":"bad \\x escape"}]}'
    respx.get(YOUTUBE_RESULTS).mock(return_value=httpx.Response(200, text=page))
    assert await first_youtube_video("x") == (
        "https://www.youtube.com/watch?v=abcdefghijk",
        "bad \\x escape",
    )


@respx.mock
async def test_a_video_without_a_title_still_plays() -> None:
    page = '"videoRenderer":{"videoId":"abcdefghijk"}'
    respx.get(YOUTUBE_RESULTS).mock(return_value=httpx.Response(200, text=page))
    assert await first_youtube_video("x") == ("https://www.youtube.com/watch?v=abcdefghijk", "")


@respx.mock
async def test_no_video_or_no_youtube_is_none() -> None:
    respx.get(YOUTUBE_RESULTS).mock(return_value=httpx.Response(200, text="<html></html>"))
    assert await first_youtube_video("x") is None
    respx.get(YOUTUBE_RESULTS).mock(return_value=httpx.Response(503))
    assert await first_youtube_video("x") is None


# -- playing on screen ------------------------------------------------------------------------


async def test_with_a_screen_it_plays_there_and_opens_no_browser(
    agent: KwamiAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def found(query: str) -> tuple[str, str]:
        return "https://www.youtube.com/watch?v=jfKfPfyJRdk", "lofi radio"

    monkeypatch.setattr(media, "first_youtube_video", found)
    said = await agent.play_media(None, "lofi", kind="video")
    assert "video window on Jarvis's screen" in said and "(lofi radio)" in said
    assert agent._navigated == []  # type: ignore[attr-defined]
    assert agent._app.published == [  # type: ignore[attr-defined]
        {
            "type": "hud_open",
            "kind": "video",
            "url": "https://www.youtube.com/watch?v=jfKfPfyJRdk",
            "title": "lofi radio",
        }
    ]
    assert agent._media_on_screen is True


async def test_an_untitled_video_is_named_by_what_was_asked(
    agent: KwamiAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def found(query: str) -> tuple[str, str]:
        return "https://www.youtube.com/watch?v=jfKfPfyJRdk", ""

    monkeypatch.setattr(media, "first_youtube_video", found)
    said = await agent.play_media(None, "lofi", kind="music")
    assert "()" not in said
    assert agent._app.published[0]["title"] == "lofi"  # type: ignore[attr-defined]


@pytest.mark.parametrize("why", ["no video", "the app did not take it", "no screen", "spotify"])
async def test_otherwise_the_browser_plays_it_as_before(
    agent: KwamiAgent, monkeypatch: pytest.MonkeyPatch, why: str
) -> None:
    async def found(query: str) -> tuple[str, str] | None:
        return None if why == "no video" else ("https://www.youtube.com/watch?v=x", "x")

    monkeypatch.setattr(media, "first_youtube_video", found)
    if why == "the app did not take it":
        agent._app = Publisher(False)  # type: ignore[attr-defined]
    if why == "no screen":
        monkeypatch.setattr(agent, "_registered_client_tool_names", lambda: {"other_tool"})
    session = NamedSession({"first_result": "clicked:a"})
    _attach(agent, session)
    said = await agent.play_media(None, "lofi", service="spotify" if why == "spotify" else "")
    assert said.startswith("Playing 'lofi' on")
    assert [name for name, _ in session.ran] == ["first_result", "resume"]
    assert agent._media_on_screen is False


async def test_a_screen_is_only_one_the_app_says_it_has(agent: KwamiAgent, monkeypatch) -> None:
    def broken() -> set[str]:
        raise RuntimeError("no tools yet")

    monkeypatch.setattr(agent, "_registered_client_tool_names", broken)
    assert agent._has_own_screen() is False
    monkeypatch.setattr(agent, "_registered_client_tool_names", None)
    assert agent._has_own_screen() is False


async def test_a_message_the_app_cannot_take_is_false_not_raised(
    agent: KwamiAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent._app = Publisher(RuntimeError("room gone"))  # type: ignore[attr-defined]
    assert await agent._publish_to_app(None, {"type": "hud_media"}) is False
    monkeypatch.setattr(agent, "_publisher", None)
    assert await agent._publish_to_app(None, {"type": "hud_media"}) is False


# -- controls follow it ----------------------------------------------------------------------


async def test_controls_go_to_the_screen_while_it_plays_there(agent: KwamiAgent) -> None:
    agent._media_on_screen = True
    assert await agent.control_playback(None, "pause") == "Done -- paused."
    assert await agent.control_playback(None, "replay") == "Done -- restartped."
    assert await agent.set_playback_volume(None, 0.4) == "Volume set to 40 percent."
    assert agent._app.published == [  # type: ignore[attr-defined]
        {"type": "hud_media", "action": "pause"},
        {"type": "hud_media", "action": "restart"},
        {"type": "hud_media", "action": "volume", "level": 0.4},
    ]
    now = await agent.get_now_playing(None)
    assert "Jarvis's screen" in now["where"]


async def test_controls_fall_back_to_the_browser_when_the_app_does_not_take_them(
    agent: KwamiAgent,
) -> None:
    agent._media_on_screen = True
    agent._app = Publisher(False)  # type: ignore[attr-defined]
    session = NamedSession({"pause": "paused", "volume": "volume:0.5"})
    _attach(agent, session)
    assert await agent.control_playback(None, "pause") == "Done -- paused."
    assert await agent.set_playback_volume(None, 0.5) == "Volume set to 50 percent."
    assert session.ran == [("pause", 0.0), ("volume", 0.5)]


async def test_their_chromes_player_is_read_by_name(agent: KwamiAgent) -> None:
    session = NamedSession(
        {"now_playing": '{"title": "Song", "paused": false, "position": 3, "duration": 9}'}
    )
    _attach(agent, session)
    now = await agent.get_now_playing(None)
    assert now["title"] == "Song" and now["playing"] is True
    assert session.ran == [("now_playing", 0.0)]
