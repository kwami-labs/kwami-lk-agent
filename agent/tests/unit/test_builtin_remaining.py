"""The last uncovered branches in the built-in tools.

Almost all of these are failure paths around the cloud browser and the two
search enrichment helpers, and they share one property: the thing that failed
is *someone else's service*, reached mid-turn while the user is waiting. A
navigation tool that raises makes the agent go quiet, which the user cannot
distinguish from a dropped call, so every one of these has to come back as a
sentence instead.

The enrichment helpers matter for a second reason. Tavily and Microlink return
whatever they return, and their output is walked for URLs and image links; a
shape that is merely *unexpected* rather than absent must shrink the result set
rather than raise through the search that produced it.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.tools.builtin import (
    AgentToolsMixin,
    _extract_features,
    _extract_price,
    _fetch_image_for_url,
    _tavily_extract_images,
)

MICROLINK = "https://api.microlink.io/"
TAVILY_EXTRACT = "https://api.tavily.com/extract"


class BrokenSession:
    """A live browser whose every operation fails.

    `is_active` is True on purpose: the interesting branch is the one *after*
    the "no browser open" guard, where the browser exists and the call to it
    raises.
    """

    is_active = True

    async def navigate(self, url: str) -> str:
        raise RuntimeError("cdp socket closed")

    async def start(self, user_id: str, url: str | None = None) -> str:
        raise RuntimeError("cdp socket closed")

    async def go_back(self) -> str:
        raise RuntimeError("cdp socket closed")

    async def go_forward(self) -> str:
        raise RuntimeError("cdp socket closed")

    async def click(self, **kwargs: Any) -> str:
        raise RuntimeError("cdp socket closed")

    async def type_text(self, *args: Any, **kwargs: Any) -> str:
        raise RuntimeError("cdp socket closed")

    async def press_key(self, key: str) -> str:
        raise RuntimeError("cdp socket closed")

    async def scroll(self, direction: str = "down") -> str:
        raise RuntimeError("cdp socket closed")

    async def evaluate_js(self, expression: str) -> str:
        raise RuntimeError("cdp socket closed")

    async def read_page(self) -> str:
        raise RuntimeError("cdp socket closed")

    async def close(self) -> None:
        raise RuntimeError("cdp socket closed")


class FakeSession:
    def __init__(self, tts: Any = None, stt: Any = None) -> None:
        self.tts = tts
        self.stt = stt


class Tools(AgentToolsMixin):
    """The mixin alone, with only the attributes it documents as required.

    `Agent.session` is a read-only property that raises outside a running
    activity, so a real `KwamiAgent` cannot be handed a stand-in session. The
    voice tools only ever touch the mixin's own documented surface, so the
    mixin is exercised directly -- the same harness the other builtin voice
    tests use.
    """

    def __init__(self, *, session: Any = None) -> None:
        self.kwami_config = KwamiConfig()
        self._current_voice_config = self.kwami_config.voice
        self._memory = None
        self.session = session
        self.room = None
        self.usage_tracker = None


@pytest.fixture
def agent() -> KwamiAgent:
    return KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))


def _attach(agent: KwamiAgent, session: Any) -> None:
    async def _get() -> Any:
        return session

    agent._get_browser_session = _get  # type: ignore[method-assign]


# -- the small parsing helpers ----------------------------------------------


def test_a_price_with_no_number_is_not_a_price() -> None:
    """The pattern can match a currency symbol with nothing after it."""
    assert _extract_price("costs $") is None
    assert _extract_price("") is None
    assert _extract_price("no digits here") is None


def test_feature_extraction_drops_filler_and_duplicates() -> None:
    """Snippets are provider prose; the same phrase twice is not two features."""
    features = _extract_features("Leather, and, leather, the, Waterproof")

    assert features == ["Leather", "Waterproof"]


def test_feature_extraction_drops_fragments() -> None:
    """Splitting on commas and dashes leaves single characters behind."""
    assert _extract_features("a, b, , Waterproof") == ["Waterproof"]


def test_feature_extraction_truncates_a_long_phrase() -> None:
    features = _extract_features("x" * 200)

    assert len(features) == 1
    assert features[0].endswith("...")
    assert len(features[0]) <= 72


def test_feature_extraction_stops_at_the_limit() -> None:
    features = _extract_features(",".join(f"feature{i}" for i in range(50)), max_items=3)

    assert len(features) == 3


# -- image enrichment -------------------------------------------------------


@respx.mock
async def test_tavily_extract_skips_a_result_with_no_url() -> None:
    """A result without a URL cannot be attributed to a card."""
    respx.post(TAVILY_EXTRACT).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"images": ["https://img/1.png"]},
                    {"url": "https://a.com", "images": "not-a-list"},
                    {"url": "https://b.com", "images": ["https://img/2.png", "ftp://no"]},
                ]
            },
        )
    )

    images = await _tavily_extract_images("tvly-key", ["https://a.com", "https://b.com"])

    assert images["https://a.com"] == [], "a non-list `images` was walked anyway"
    assert images["https://b.com"] == ["https://img/2.png"], "a non-http image was kept"


async def test_tavily_extract_without_a_key_makes_no_request() -> None:
    # No respx mock: reaching the network here would be the failure.
    assert await _tavily_extract_images("", ["https://a.com"]) == {"https://a.com": []}
    assert await _tavily_extract_images("tvly-key", []) == {}


@respx.mock
async def test_microlink_falls_back_to_the_logo() -> None:
    """A page with no og:image still has a brand mark, which beats a blank card."""
    respx.get(MICROLINK).mock(
        return_value=httpx.Response(200, json={"data": {"logo": {"url": "https://a.com/logo.png"}}})
    )

    assert await _fetch_image_for_url("https://a.com") == "https://a.com/logo.png"


@respx.mock
async def test_microlink_accepts_a_logo_given_as_a_string() -> None:
    respx.get(MICROLINK).mock(
        return_value=httpx.Response(200, json={"data": {"logo": "https://a.com/logo.png"}})
    )

    assert await _fetch_image_for_url("https://a.com") == "https://a.com/logo.png"


@respx.mock
async def test_microlink_accepts_an_image_given_as_a_string() -> None:
    respx.get(MICROLINK).mock(
        return_value=httpx.Response(200, json={"data": {"image": "https://a.com/hero.png"}})
    )

    assert await _fetch_image_for_url("https://a.com") == "https://a.com/hero.png"


@respx.mock
async def test_microlink_with_a_logo_it_cannot_use() -> None:
    """A logo that is neither an object with a url nor an http string."""
    respx.get(MICROLINK).mock(return_value=httpx.Response(200, json={"data": {"logo": 12345}}))

    assert await _fetch_image_for_url("https://a.com") is None


@respx.mock
async def test_microlink_failing_yields_no_image_rather_than_raising() -> None:
    """Enrichment is decoration; it must not take the search down with it."""
    respx.get(MICROLINK).mock(return_value=httpx.Response(503))

    assert await _fetch_image_for_url("https://a.com") is None


# -- voice controls ---------------------------------------------------------


async def test_changing_speed_without_a_tts() -> None:
    tools = Tools(session=FakeSession(tts=None))

    assert "not available" in await tools.change_speaking_speed(None, 1.2)


async def test_a_tts_that_refuses_a_speed_change_is_reported() -> None:
    class Exploding:
        provider = "cartesia"

        def update_options(self, **kwargs: Any) -> None:
            raise RuntimeError("provider rejected the update")

    result = await Tools(session=FakeSession(tts=Exploding())).change_speaking_speed(None, 1.2)

    assert "couldn't change the speed" in result


async def test_changing_language_updates_stt_and_tolerates_tts() -> None:
    """Not every TTS takes a language; the STT half must still land."""
    applied: dict[str, Any] = {}

    class STT:
        def update_options(self, **kwargs: Any) -> None:
            applied.update(kwargs)

    class TTSWithoutLanguage:
        def update_options(self, **kwargs: Any) -> None:
            raise TypeError("unexpected keyword argument 'language'")

    result = await Tools(session=FakeSession(stt=STT(), tts=TTSWithoutLanguage())).change_language(
        None, "es"
    )

    assert applied == {"language": "es"}
    assert result, "a TTS that cannot take a language lost the whole change"


async def test_an_stt_that_refuses_a_language_change_is_reported() -> None:
    class Exploding:
        def update_options(self, **kwargs: Any) -> None:
            raise RuntimeError("stt is closed")

    tools = Tools(session=FakeSession(stt=Exploding()))

    assert "couldn't change the language" in await tools.change_language(None, "es")


# -- the browser session accessor -------------------------------------------


async def test_a_handed_over_browser_gets_the_room_and_tracker(agent: KwamiAgent) -> None:
    """A session transferred from a swapped-out agent carries neither."""

    class Handover:
        def __init__(self) -> None:
            self._room = None
            self._usage_tracker = None

        def set_room(self, room: Any) -> None:
            self._room = room

        def set_usage_tracker(self, tracker: Any) -> None:
            self._usage_tracker = tracker

    handover = Handover()
    agent._browser_session = handover
    agent.room = object()
    agent.usage_tracker = object()

    session = await agent._get_browser_session()

    assert session is handover
    assert handover._room is agent.room, "the carried-over browser could not publish"
    assert handover._usage_tracker is agent.usage_tracker, "its minutes would go unbilled"


# -- navigation failures ----------------------------------------------------


async def test_navigating_with_a_live_but_broken_browser(agent: KwamiAgent) -> None:
    _attach(agent, BrokenSession())

    result = await agent.navigate_to(None, "https://example.com")

    assert "Failed to open browser" in result


async def test_navigating_when_the_browser_cannot_be_created(agent: KwamiAgent) -> None:
    """`BrowserUseClient` raises ValueError when its key is missing."""

    class Unconfigured:
        is_active = False

        async def start(self, user_id: str, url: str | None = None) -> str:
            raise ValueError("BROWSER_USE_API_KEY is not set")

    _attach(agent, Unconfigured())

    result = await agent.navigate_to(None, "https://example.com")

    assert "Cannot open browser" in result


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("go_back_in_browser", (), "Failed to go back"),
        ("go_forward_in_browser", (), "Failed to go forward"),
        ("press_key_in_navigation", ("Enter",), "Failed to press key"),
        ("scroll_navigation", ("down",), "Failed to scroll"),
        ("read_navigation_page", (), "Failed to read page"),
    ],
)
async def test_a_failing_navigation_call_answers_in_words(
    agent: KwamiAgent, tool: str, args: tuple, expected: str
) -> None:
    """A tool that raises makes the agent go quiet mid-turn."""
    _attach(agent, BrokenSession())

    result = await getattr(agent, tool)(None, *args)

    assert expected in result


async def test_a_failing_click_answers_in_words(agent: KwamiAgent) -> None:
    _attach(agent, BrokenSession())

    result = await agent.click_in_navigation(None, element_id="el-1")

    assert "Failed to click" in result


async def test_a_failing_type_answers_in_words(agent: KwamiAgent) -> None:
    _attach(agent, BrokenSession())

    result = await agent.type_in_navigation(None, "hello", element_id="el-1")

    assert "Failed to type" in result


async def test_running_js_that_fails_answers_in_words(agent: KwamiAgent, env_setting) -> None:
    env_setting("KWAMI_ALLOW_BROWSER_JS", "1")
    _attach(agent, BrokenSession())

    result = await agent.run_js_in_navigation(None, "1 + 1")

    assert "Failed to run JS" in result


async def test_closing_a_browser_that_refuses_to_close(agent: KwamiAgent) -> None:
    _attach(agent, BrokenSession())

    result = await agent.close_navigation(None)

    assert isinstance(result, str) and result


# -- search failures and image enrichment, through the real tool -------------

TAVILY_SEARCH = "https://api.tavily.com/search"


def _searching_agent(env_setting) -> KwamiAgent:
    env_setting("TAVILY_API_KEY", "tvly-test")
    return KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))


@respx.mock
async def test_a_search_error_whose_body_cannot_be_read(env_setting) -> None:
    """Reading the body for the log can itself raise.

    The handler logs `e.response.text` before deciding what to tell the user.
    A response that cannot be decoded would otherwise turn a handled search
    failure into an unhandled one, mid-turn, and the agent goes quiet.
    """

    class Unreadable(httpx.Response):
        @property
        def text(self) -> str:
            raise RuntimeError("cannot decode body")

    request = httpx.Request("POST", TAVILY_SEARCH)
    response = Unreadable(500, request=request)
    respx.post(TAVILY_SEARCH).mock(
        side_effect=httpx.HTTPStatusError("boom", request=request, response=response)
    )

    result = await _searching_agent(env_setting).web_search(None, "anything")

    assert "Search failed" in result


@respx.mock
async def test_a_search_error_with_a_non_json_body(env_setting) -> None:
    """The detail is dug out of JSON; a plain-text body must not raise."""
    respx.post(TAVILY_SEARCH).mock(return_value=httpx.Response(500, text="upstream exploded"))

    result = await _searching_agent(env_setting).web_search(None, "anything")

    assert "Search failed" in result
    assert "upstream exploded" in result


@respx.mock
async def test_search_cards_take_their_image_from_tavily(env_setting) -> None:
    """The preferred source: a real page image rather than a link preview."""
    respx.post(TAVILY_SEARCH).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "An answer.",
                "results": [{"title": "T", "url": "https://a.com/1", "content": "c"}],
            },
        )
    )
    respx.post(TAVILY_EXTRACT).mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"url": "https://a.com/1", "images": ["https://img/real.png"]}]},
        )
    )
    # No Microlink mock: reaching it would mean the fallback ran needlessly.

    result = await _searching_agent(env_setting).web_search(None, "anything")

    assert "An answer." in result


@respx.mock
async def test_search_cards_fall_back_to_a_link_preview(env_setting) -> None:
    """Tavily found no image, so Microlink is asked for one."""
    respx.post(TAVILY_SEARCH).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "An answer.",
                "results": [{"title": "T", "url": "https://a.com/1", "content": "c"}],
            },
        )
    )
    respx.post(TAVILY_EXTRACT).mock(
        return_value=httpx.Response(200, json={"results": [{"url": "https://a.com/1"}]})
    )
    fallback = respx.get(MICROLINK).mock(
        return_value=httpx.Response(200, json={"data": {"image": {"url": "https://img/og.png"}}})
    )

    await _searching_agent(env_setting).web_search(None, "anything")

    assert fallback.called, "no image anywhere, and the fallback was never tried"


@respx.mock
async def test_a_product_search_whose_cards_cannot_be_published(env_setting) -> None:
    """Publishing is best-effort; the spoken answer must survive its failure."""
    env_setting("SERPAPI_KEY", "serp-test")
    respx.get("https://serpapi.com/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "shopping_results": [{"title": "A bag", "price": "$49", "link": "https://shop/1"}]
            },
        )
    )
    agent = KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))
    agent.room = None  # nothing to publish through

    result = await agent.product_search(None, "a bag")

    assert "Found 1 products" in result


# -- the remaining small branches -------------------------------------------


async def test_remembering_a_fact_reuses_the_existing_task_set() -> None:
    """The set is created once in `__init__`; a second call must not replace it."""

    class Memory:
        is_initialized = True

        def __init__(self) -> None:
            self.facts: list[str] = []

        async def add_fact(self, fact: str) -> None:
            self.facts.append(fact)

    import asyncio

    agent = KwamiAgent(config=KwamiConfig(), memory=Memory())
    original = agent._background_tasks

    agent._remember_in_background(["one"])
    agent._remember_in_background(["two"])

    assert agent._background_tasks is original, "the task set was replaced mid-session"
    await asyncio.gather(*list(agent._background_tasks), return_exceptions=True)


async def test_changing_language_without_an_stt_or_tts() -> None:
    """A realtime session has neither; the request must not raise."""
    result = await Tools(session=FakeSession(stt=None, tts=None)).change_language(None, "es")

    assert result, "a session with no STT or TTS lost the language change entirely"


async def test_a_browser_session_is_created_on_first_use() -> None:
    """The lazy path: no session yet, so one is constructed."""
    agent = KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))
    agent._browser_session = None

    session = await agent._get_browser_session()

    assert session is not None
    assert agent._browser_session is session, "a second call would build another one"


async def test_an_existing_browser_session_with_a_room_is_left_alone() -> None:
    """Already wired up: neither the room nor the tracker should be re-set."""

    class Existing:
        def __init__(self) -> None:
            self._room = object()
            self._usage_tracker = object()

        def set_room(self, room: Any) -> None:  # pragma: no cover - must not be called
            raise AssertionError("the room was re-set on a session that had one")

        def set_usage_tracker(self, tracker: Any) -> None:  # pragma: no cover
            raise AssertionError("the tracker was re-set on a session that had one")

    agent = KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))
    existing = Existing()
    agent._browser_session = existing

    assert await agent._get_browser_session() is existing


async def test_running_js_with_no_browser_open(env_setting) -> None:
    env_setting("KWAMI_ALLOW_BROWSER_JS", "1")

    class Closed:
        is_active = False

    agent = KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))

    async def _get() -> Any:
        return Closed()

    agent._get_browser_session = _get  # type: ignore[method-assign]

    assert "No browser is open" in await agent.run_js_in_navigation(None, "1 + 1")


async def test_a_handed_over_browser_when_the_agent_has_no_room_either() -> None:
    """Nothing to re-attach, so the tracker check must still be reached.

    A session transferred from a swapped-out agent carries neither room nor
    tracker. If the agent has no room to give it, the room is left alone -- but
    the tracker still has to be wired up, or the browser's minutes go unbilled.
    """

    class Handover:
        def __init__(self) -> None:
            self._room = None
            self._usage_tracker = None

        def set_room(self, room: Any) -> None:  # pragma: no cover - nothing to set
            raise AssertionError("set_room was called with no room available")

        def set_usage_tracker(self, tracker: Any) -> None:
            self._usage_tracker = tracker

    agent = KwamiAgent(config=KwamiConfig(kwami_id="tenant-1"))
    handover = Handover()
    agent._browser_session = handover
    agent.room = None
    agent.usage_tracker = object()

    session = await agent._get_browser_session()

    assert session is handover
    assert handover._usage_tracker is agent.usage_tracker, "its minutes would go unbilled"
