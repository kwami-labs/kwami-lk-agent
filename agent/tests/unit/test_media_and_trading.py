"""Playing media by voice, and placing an order without being talked into it.

Media: the agent could reach a *results page* and stop there, because the step
that actually starts playback needs a click into a shadow DOM that
`read_navigation_page` cannot see. The tests here pin the two things that make
`play_media` safe to hand a model: the URL is built from a fixed template, and
the in-page script is a constant from our own source rather than anything the
model wrote.

Trading: the interlock is the whole feature. A confirmation the model can
supply for itself is not a confirmation, so the code is derived from the exact
order and re-derived on submit. These tests try to get an order sent without
the user, in every way the shape of the API allows.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.tools.media import SERVICES, resolve_service, search_url
from src.tools.trading import BROKER_URLS, PendingTrade, normalize_side


class FakeBrowserSession:
    """Records the scripts it was asked to run, and what it returns for each."""

    def __init__(self, *, active: bool = True, results: dict[str, str] | None = None) -> None:
        self.is_active = active
        self.evaluated: list[str] = []
        self._results = results or {}

    async def evaluate_js(self, expression: str) -> str:
        self.evaluated.append(expression)
        for needle, result in self._results.items():
            if needle in expression:
                return f"JavaScript executed successfully. Result: {result}"
        return "JavaScript executed successfully. Result: ok"


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the page-settle waits.

    They are real behaviour -- a results page needs a moment before its first
    item can be clicked -- but against a double there is nothing to wait for,
    and 4.5 real seconds per playback test is 14s of suite time bought with
    nothing.
    """
    from src.tools import media

    monkeypatch.setattr(media, "RESULTS_SETTLE_SECONDS", 0)
    monkeypatch.setattr(media, "PLAYBACK_SETTLE_SECONDS", 0)


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> KwamiAgent:
    """An agent whose browser is a double and whose navigation is recorded."""
    instance = KwamiAgent(config=KwamiConfig())
    instance._navigated: list[str] = []

    async def fake_navigate(context: Any, url: str) -> str:
        instance._navigated.append(url)
        return f"Navigating to {url}."

    monkeypatch.setattr(instance, "navigate_to", fake_navigate)
    return instance


def _attach(agent: KwamiAgent, session: Any) -> None:
    async def _get() -> Any:
        return session

    agent._get_browser_session = _get  # type: ignore[method-assign]


# -- media: URL construction ------------------------------------------------


def test_the_query_is_encoded_into_the_service_url() -> None:
    """The model never supplies a URL, so it cannot be steered into one."""
    url = search_url("youtube", "Bohemian Rhapsody & friends")

    assert url.startswith("https://www.youtube.com/results?search_query=")
    assert " " not in url
    assert "&friends" not in url, "an unencoded ampersand became a second query parameter"


def test_a_long_query_is_bounded() -> None:
    url = search_url("youtube", "x" * 5000)
    assert len(url) < 500


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_every_service_builds_an_https_url(service: str) -> None:
    url = search_url(service, "test song")
    assert url.startswith("https://")


def test_music_and_video_get_different_defaults() -> None:
    music, _ = resolve_service("music", None)
    video, _ = resolve_service("video", None)
    assert music != video


def test_an_unknown_service_is_refused_not_substituted() -> None:
    """Asked for Spotify, opening YouTube instead answers a different question."""
    assert resolve_service("music", "pandora") is None


def test_service_aliases_resolve() -> None:
    assert resolve_service("music", "YouTube Music")[0] == "youtube_music"
    assert resolve_service("video", "yt")[0] == "youtube"


# -- media: playback --------------------------------------------------------


async def test_play_media_opens_the_service_and_starts_it(agent: KwamiAgent) -> None:
    session = FakeBrowserSession(results={"sel": "clicked:ytd-video-renderer a#video-title"})
    _attach(agent, session)

    result = await agent.play_media(None, "Bohemian Rhapsody", kind="video")

    assert agent._navigated, "nothing was opened"
    assert "youtube.com/results" in agent._navigated[0]
    assert len(session.evaluated) >= 2, "opened the results page but never started playing"
    assert "Playing" in result


async def test_play_media_never_runs_a_script_the_model_wrote(agent: KwamiAgent) -> None:
    """Arbitrary in-page JS is off by default for good reason.

    The model chooses a named action; the script itself is a constant in our
    source. A query that looks like code must stay a query.
    """
    from src.tools import media

    session = FakeBrowserSession(results={"sel": "clicked:a"})
    _attach(agent, session)

    await agent.play_media(None, "');alert(document.cookie);//", kind="music")

    known = {media._FIRST_RESULT_JS, *media._MEDIA_ACTION_JS.values()}
    for script in session.evaluated:
        assert script in known, f"ran a script that is not one of our constants: {script[:80]}"


async def test_play_media_says_so_when_nothing_matched(agent: KwamiAgent) -> None:
    session = FakeBrowserSession(results={"sel": "no-result"})
    _attach(agent, session)

    result = await agent.play_media(None, "asdkjhasdkjh", kind="music")

    assert "couldn't find" in result


async def test_play_media_without_a_query_asks(agent: KwamiAgent) -> None:
    result = await agent.play_media(None, "   ")
    assert "?" in result
    assert agent._navigated == []


async def test_play_media_reports_a_panel_that_never_came_up(agent: KwamiAgent) -> None:
    _attach(agent, FakeBrowserSession(active=False))

    result = await agent.play_media(None, "something", kind="music")

    assert "didn't come up" in result


# -- media: control ---------------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [("pause", "pause"), ("play", "play"), ("continue", "play"), ("quiet", "muted")],
)
async def test_playback_controls_reach_the_page(
    agent: KwamiAgent, spoken: str, expected: str
) -> None:
    session = FakeBrowserSession()
    _attach(agent, session)

    await agent.control_playback(None, spoken)

    assert session.evaluated, f"'{spoken}' did nothing"
    assert expected in session.evaluated[0]


async def test_an_unknown_playback_action_is_refused(agent: KwamiAgent) -> None:
    session = FakeBrowserSession()
    _attach(agent, session)

    result = await agent.control_playback(None, "moonwalk")

    assert session.evaluated == []
    assert "don't know how" in result


async def test_controls_report_when_nothing_is_playing(agent: KwamiAgent) -> None:
    _attach(agent, FakeBrowserSession(active=False))
    assert "Nothing is playing" in await agent.control_playback(None, "pause")


async def test_volume_is_clamped(agent: KwamiAgent) -> None:
    session = FakeBrowserSession()
    _attach(agent, session)

    await agent.set_playback_volume(None, 9000)

    assert "m.volume=1.0" in session.evaluated[0].replace(" ", "")


async def test_now_playing_reads_the_player(agent: KwamiAgent) -> None:
    session = FakeBrowserSession(
        results={
            "JSON.stringify": '{"title":"A Song","paused":false,"muted":false,'
            '"volume":1,"position":30,"duration":210}'
        }
    )
    _attach(agent, session)

    info = await agent.get_now_playing(None)

    assert info["playing"] is True
    assert info["title"] == "A Song"
    assert info["duration"] == 210


async def test_now_playing_on_a_page_with_no_media(agent: KwamiAgent) -> None:
    session = FakeBrowserSession(results={"JSON.stringify": "no-media"})
    _attach(agent, session)

    info = await agent.get_now_playing(None)

    assert info["playing"] is False


# -- trading: the read-back -------------------------------------------------


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [("buy", "buy"), ("purchase", "buy"), ("long", "buy"), ("sell", "sell"), ("short", "sell")],
)
def test_spoken_sides_resolve(spoken: str, expected: str) -> None:
    assert normalize_side(spoken) == expected


def test_an_ambiguous_side_is_refused() -> None:
    """Guessing a direction would pick it for the user."""
    assert normalize_side("do something with") is None


def test_the_confirmation_code_is_derived_from_the_order() -> None:
    """Not random: it has to change when the order does."""
    one = PendingTrade("TSLA", "buy", 10, "market")
    same = PendingTrade("TSLA", "buy", 10, "market")
    more = PendingTrade("TSLA", "buy", 100, "market")
    other_side = PendingTrade("TSLA", "sell", 10, "market")

    assert one.confirmation_code == same.confirmation_code
    assert one.confirmation_code != more.confirmation_code
    assert one.confirmation_code != other_side.confirmation_code
    assert one.confirmation_code.isdigit()


async def test_prepare_trade_sends_nothing(agent: KwamiAgent, monkeypatch) -> None:
    async def fake_quote(context: Any, symbol: str) -> dict[str, Any]:
        return {"price": 100.0, "currency": "USD"}

    monkeypatch.setattr(agent, "get_market_quote", fake_quote)

    summary = await agent.prepare_trade(None, "tsla", "buy", 10)

    assert summary["submitted"] is False
    assert summary["order"] == "buy 10.0 TSLA at market"
    assert summary["estimated_value"] == 1000.0
    assert summary["confirmation_code"]
    assert agent._navigated == [], "a prepare step opened a broker"


async def test_prepare_trade_warns_when_it_cannot_price_the_order(
    agent: KwamiAgent, monkeypatch
) -> None:
    """Agreeing to a trade without knowing what it costs is the thing to avoid."""

    async def dead_feed(context: Any, symbol: str) -> dict[str, Any]:
        return {"error": "no price"}

    monkeypatch.setattr(agent, "get_market_quote", dead_feed)

    summary = await agent.prepare_trade(None, "TSLA", "buy", 10)

    assert "estimated_value" not in summary
    assert "warning" in summary


@pytest.mark.parametrize(
    ("symbol", "side", "quantity", "order_type", "limit"),
    [
        ("", "buy", 10, "market", 0),
        ("TSLA", "maybe", 10, "market", 0),
        ("TSLA", "buy", 0, "market", 0),
        ("TSLA", "buy", -5, "market", 0),
        ("TSLA", "buy", 10, "stop-loss", 0),
        ("TSLA", "buy", 10, "limit", 0),
    ],
    ids=["no-symbol", "ambiguous-side", "zero-qty", "negative-qty", "bad-type", "limit-no-price"],
)
async def test_a_malformed_order_is_refused(
    agent: KwamiAgent, symbol, side, quantity, order_type, limit
) -> None:
    result = await agent.prepare_trade(None, symbol, side, quantity, order_type, limit)

    assert "error" in result
    assert await agent.get_prepared_trade(None) == {"pending": False}


# -- trading: the interlock -------------------------------------------------


async def test_nothing_can_be_submitted_without_a_prepared_order(agent: KwamiAgent) -> None:
    result = await agent.submit_trade(None, "1234")
    assert "no order ready" in result.lower()


async def test_a_wrong_code_does_not_submit(agent: KwamiAgent, monkeypatch) -> None:
    monkeypatch.setattr(agent, "get_market_quote", _price(100.0))
    await agent.prepare_trade(None, "TSLA", "buy", 10)
    _attach(agent, FakeBrowserSession())

    result = await agent.submit_trade(None, "0000")

    assert "doesn't match" in result
    assert (await agent.get_prepared_trade(None))["pending"] is True, "the order was dropped"


async def test_the_code_from_a_different_order_does_not_submit(
    agent: KwamiAgent, monkeypatch
) -> None:
    """The code the user spoke has to belong to the order about to be sent."""
    monkeypatch.setattr(agent, "get_market_quote", _price(100.0))
    other = PendingTrade("NVDA", "sell", 999, "market")

    await agent.prepare_trade(None, "TSLA", "buy", 10)
    _attach(agent, FakeBrowserSession())

    result = await agent.submit_trade(None, other.confirmation_code)

    assert "doesn't match" in result


async def test_editing_the_order_invalidates_the_old_code(agent: KwamiAgent, monkeypatch) -> None:
    """Otherwise a code agreed for 10 shares would send 1000."""
    monkeypatch.setattr(agent, "get_market_quote", _price(100.0))
    first = await agent.prepare_trade(None, "TSLA", "buy", 10)
    await agent.prepare_trade(None, "TSLA", "buy", 1000)
    _attach(agent, FakeBrowserSession())

    result = await agent.submit_trade(None, first["confirmation_code"])

    assert "doesn't match" in result


async def test_the_right_code_submits(agent: KwamiAgent, monkeypatch) -> None:
    monkeypatch.setattr(agent, "get_market_quote", _price(100.0))
    summary = await agent.prepare_trade(None, "TSLA", "buy", 10)
    _attach(agent, FakeBrowserSession())

    result = await agent.submit_trade(None, summary["confirmation_code"])

    assert "Confirmed" in result
    assert (await agent.get_prepared_trade(None))["pending"] is False


async def test_a_spoken_code_with_padding_still_matches(agent: KwamiAgent, monkeypatch) -> None:
    """Transcripts arrive as "one two three four", not as a bare integer."""
    monkeypatch.setattr(agent, "get_market_quote", _price(100.0))
    summary = await agent.prepare_trade(None, "TSLA", "buy", 10)
    _attach(agent, FakeBrowserSession())

    spaced = " ".join(summary["confirmation_code"])
    result = await agent.submit_trade(None, spaced)

    assert "Confirmed" in result


async def test_submitting_without_a_broker_open_does_nothing(
    agent: KwamiAgent, monkeypatch
) -> None:
    monkeypatch.setattr(agent, "get_market_quote", _price(100.0))
    summary = await agent.prepare_trade(None, "TSLA", "buy", 10)
    _attach(agent, FakeBrowserSession(active=False))

    result = await agent.submit_trade(None, summary["confirmation_code"])

    assert "isn't open" in result
    assert (await agent.get_prepared_trade(None))["pending"] is True


async def test_cancelling_clears_the_order(agent: KwamiAgent, monkeypatch) -> None:
    monkeypatch.setattr(agent, "get_market_quote", _price(100.0))
    await agent.prepare_trade(None, "TSLA", "buy", 10)

    result = await agent.cancel_prepared_trade(None)

    assert "Cancelled" in result
    assert (await agent.get_prepared_trade(None))["pending"] is False


# -- trading: the broker page -----------------------------------------------


async def test_the_ticket_needs_an_order_first(agent: KwamiAgent) -> None:
    result = await agent.open_trade_ticket(None, broker="trading212")
    assert "work out the order first" in result
    assert agent._navigated == []


async def test_a_known_broker_opens(agent: KwamiAgent, monkeypatch) -> None:
    monkeypatch.setattr(agent, "get_market_quote", _price(100.0))
    await agent.prepare_trade(None, "TSLA", "buy", 10)

    await agent.open_trade_ticket(None, broker="Trading212")

    assert agent._navigated == [BROKER_URLS["trading212"]]


async def test_an_unknown_broker_is_named_not_guessed(agent: KwamiAgent, monkeypatch) -> None:
    monkeypatch.setattr(agent, "get_market_quote", _price(100.0))
    await agent.prepare_trade(None, "TSLA", "buy", 10)

    result = await agent.open_trade_ticket(None, broker="my-cousins-brokerage")

    assert agent._navigated == [], "navigated somewhere it could not identify"
    assert "trading212" in result


def _price(value: float):
    async def _quote(context: Any, symbol: str) -> dict[str, Any]:
        return {"price": value, "currency": "USD"}

    return _quote
