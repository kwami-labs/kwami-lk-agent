"""KwamiAgent: the hooks the framework actually dispatches, and what they do.

This class is where the two worst production defects lived. `on_enter` had a
`room` parameter the framework never passes, so the duplicate-agent guard never
ran and `self.room` was overwritten with None; and assistant turns were hung off
`on_agent_turn_completed`, a hook livekit-agents does not dispatch, so nothing
the assistant said was ever written to Zep.

The agent is real throughout -- it subclasses the installed `livekit.agents.Agent`.
Only `session`, which the framework populates from a running activity, is
supplied by a subclass here.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig, KwamiSoulConfig
from src.memory.context import MemoryContext


class FakeSession:
    def __init__(self, fail_first: bool = False) -> None:
        self.replies: list[dict[str, Any]] = []
        self.handlers: dict[str, list] = {}
        self._fail_first = fail_first

    def generate_reply(self, *, instructions: str, allow_interruptions: bool = True):
        if self._fail_first and not self.replies:
            self.replies.append({"failed": instructions})
            raise RuntimeError("tts unavailable")
        self.replies.append(
            {"instructions": instructions, "allow_interruptions": allow_interruptions}
        )

    def on(self, event: str, fn) -> None:
        self.handlers.setdefault(event, []).append(fn)


class AgentWithSession(KwamiAgent):
    """KwamiAgent with a supplied session.

    `Agent.session` is a read-only property backed by the running activity, so
    a unit test has no other way to reach the greeting path.
    """

    _fake_session: Any = None

    @property
    def session(self):  # type: ignore[override]
        if self._fake_session is not None:
            return self._fake_session
        # Delegate, so an agent with no fake behaves exactly as the framework
        # does: AttributeError while __init__ is still running (which
        # find_function_tools relies on getmembers swallowing), RuntimeError
        # afterwards, which is what _register_session_listeners catches.
        return KwamiAgent.session.fget(self)  # type: ignore[attr-defined]


class FakeRoom:
    def __init__(self, identity: str = "agent-1", participants: dict | None = None) -> None:
        self.local_participant = SimpleNamespace(identity=identity)
        self.remote_participants = participants or {}
        self.disconnected = False

    async def disconnect(self) -> None:
        self.disconnected = True


class FakeMemory:
    def __init__(
        self,
        *,
        initialized: bool = True,
        user_name: str | None = None,
        context: MemoryContext | None = None,
        get_context_error: Exception | None = None,
    ) -> None:
        self.is_initialized = initialized
        self._cached_user_name = user_name
        self._context = context or MemoryContext()
        self._get_context_error = get_context_error
        self.buffered: list[tuple[str, str | None]] = []
        self.exchanges: list[dict[str, Any]] = []
        self.set_names: list[str] = []

    async def get_user_name(self) -> str | None:
        return self._cached_user_name

    async def get_context(self) -> MemoryContext:
        if self._get_context_error is not None:
            raise self._get_context_error
        return self._context

    def set_user_name(self, name: str) -> None:
        self.set_names.append(name)
        self._cached_user_name = name

    async def buffer_user_message(self, content: str, name: str | None = None) -> None:
        self.buffered.append((content, name))

    async def add_exchange(self, *, assistant_content: str, assistant_name: str) -> None:
        self.exchanges.append({"content": assistant_content, "name": assistant_name})


class Unreadable:
    """A message the extractor cannot read.

    Its default repr is the `<module.Class object at 0x...>` form, which the
    last-resort branch filters out rather than handing to Zep as content.
    """

    def __init__(self, role: str | None = None) -> None:
        self.role = role


def agent_with(**kwargs: Any) -> AgentWithSession:
    config = kwargs.pop("config", None) or KwamiConfig()
    return AgentWithSession(config=config, **kwargs)


# =============================================================================
# Construction
# =============================================================================


def test_an_agent_builds_with_no_arguments() -> None:
    agent = KwamiAgent()

    assert agent.kwami_config is not None
    assert agent.room is None
    assert agent.usage_tracker is None


def test_the_soul_reaches_the_system_prompt() -> None:
    config = KwamiConfig(soul=KwamiSoulConfig(name="Ada", personality="dry and precise"))

    agent = KwamiAgent(config=config)

    assert "Ada" in agent.instructions
    assert "dry and precise" in agent.instructions


def test_client_tools_from_config_are_registered() -> None:
    config = KwamiConfig(
        tools=[{"name": "set_theme", "description": "change the theme", "parameters": {}}]
    )

    agent = KwamiAgent(config=config)

    assert "set_theme" in agent._registered_client_tool_names()


def test_builtins_are_not_duplicated_on_the_agent(all_tools_available) -> None:
    """The framework sets its tool list to `tools + find_function_tools(self)`,
    so passing the built-ins in as well would list each of them twice. The
    provider's 128-tool ceiling applies to the sum, and one tool over it is a
    400 on every turn, not a degraded feature.

    Asserted as a count against the framework's own discovery, so doubling
    shows up as 80 instead of 40 without reaching into a framework internal.
    """
    from livekit.agents.llm.tool_context import find_function_tools

    agent = KwamiAgent()

    assert len(agent.tools) == len(find_function_tools(KwamiAgent))


def test_registered_tool_names_tolerate_both_entry_shapes() -> None:
    agent = KwamiAgent()
    agent.client_tools.registered_tools = [  # type: ignore[attr-defined]
        {"name": "from_dict"},
        SimpleNamespace(name="from_object"),
        {"name": ""},
        {"no_name": 1},
        SimpleNamespace(name=None),
    ]

    assert agent._registered_client_tool_names() == {"from_dict", "from_object"}


def test_no_tool_manager_yields_no_names() -> None:
    agent = KwamiAgent()
    agent.client_tools = None  # type: ignore[assignment]

    assert agent._registered_client_tool_names() == set()


# =============================================================================
# _extract_message_content
# =============================================================================


def test_text_content_is_preferred() -> None:
    """livekit's ChatMessage stores content as a list; `text_content` joins the
    text parts. Without this Zep was fed a pydantic repr instead of speech."""
    agent = KwamiAgent()
    message = SimpleNamespace(text_content="  what the user said  ", content=["ignored"])

    assert agent._extract_message_content(message) == "what the user said"


def test_a_none_message_extracts_nothing() -> None:
    assert KwamiAgent()._extract_message_content(None) == ""


def test_a_plain_string_content_is_used() -> None:
    agent = KwamiAgent()

    assert agent._extract_message_content(SimpleNamespace(content="hello")) == "hello"


def test_a_list_content_is_joined() -> None:
    agent = KwamiAgent()

    result = agent._extract_message_content(SimpleNamespace(content=["a", "b"]))

    assert "a" in result and "b" in result


def test_a_bare_string_message_is_returned() -> None:
    assert KwamiAgent()._extract_message_content("just a string") == "just a string"


# =============================================================================
# on_enter
# =============================================================================


async def test_on_enter_takes_no_room_argument() -> None:
    """The defect this replaced: an extra `room` parameter meant the framework's
    zero-argument dispatch never matched, so the guard below never ran."""
    import inspect

    params = list(inspect.signature(KwamiAgent.on_enter).parameters)

    assert params == ["self"]


async def test_entering_registers_the_conversation_listener() -> None:
    agent = agent_with(skip_greeting=True)
    agent.room = FakeRoom()
    agent._fake_session = FakeSession()

    await agent.on_enter()

    assert "conversation_item_added" in agent._fake_session.handlers


async def test_a_duplicate_agent_disconnects(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    async def duplicate(room, identity) -> bool:
        return True

    monkeypatch.setattr("src.agent.should_disconnect_as_duplicate", duplicate)
    agent = agent_with(skip_greeting=True)
    room = FakeRoom()
    agent.room = room
    agent._fake_session = FakeSession()

    with caplog.at_level(logging.WARNING):
        await agent.on_enter()

    assert room.disconnected is True
    assert "duplicate detection" in caplog.text


async def test_entering_without_a_room_still_proceeds(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setattr("src.agent.room_from_context", lambda _: None)
    agent = agent_with(skip_greeting=True)
    agent._fake_session = FakeSession()

    with caplog.at_level(logging.WARNING):
        await agent.on_enter()

    assert "no room reference available" in caplog.text


async def test_the_room_falls_back_to_the_context(monkeypatch: pytest.MonkeyPatch) -> None:
    room = FakeRoom()
    monkeypatch.setattr("src.agent.room_from_context", lambda _: room)
    agent = agent_with(skip_greeting=True)
    agent._fake_session = FakeSession()

    await agent.on_enter()

    assert agent.room is room


async def test_a_reconfigured_agent_does_not_greet() -> None:
    agent = agent_with(skip_greeting=True)
    agent.room = FakeRoom()
    agent._fake_session = FakeSession()

    await agent.on_enter()

    assert agent._fake_session.replies == []


async def test_a_fresh_agent_greets() -> None:
    agent = agent_with(skip_greeting=False)
    agent.room = FakeRoom()
    agent._fake_session = FakeSession()

    await agent.on_enter()

    assert len(agent._fake_session.replies) == 1
    assert agent._fake_session.replies[0]["allow_interruptions"] is True


async def test_a_failing_greeting_falls_back_to_a_simple_one(caplog) -> None:
    """The agent must still speak; silence reads as a broken product."""
    agent = agent_with(skip_greeting=False)
    agent.room = FakeRoom()
    agent._fake_session = FakeSession(fail_first=True)

    with caplog.at_level(logging.ERROR):
        await agent.on_enter()

    assert "Greet the user casually" in agent._fake_session.replies[-1]["instructions"]


async def test_a_failing_fallback_greeting_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    class AlwaysFails(FakeSession):
        def generate_reply(self, **kwargs: Any):
            raise RuntimeError("no tts at all")

    agent = agent_with(skip_greeting=False)
    agent.room = FakeRoom()
    agent._fake_session = AlwaysFails()

    with caplog.at_level(logging.ERROR):
        await agent.on_enter()

    assert "Failed to generate fallback greeting" in caplog.text


async def test_slow_memory_does_not_hold_the_greeting(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Memory is an enhancement; the greeting is the product. A slow Zep used
    to hold the first utterance for as long as it took."""
    monkeypatch.setattr("src.agent.Timeouts.MEMORY_CONTEXT", 0.01)

    class SlowMemory(FakeMemory):
        async def get_context(self) -> MemoryContext:
            await asyncio.sleep(5)
            return MemoryContext()

    agent = agent_with(skip_greeting=False, memory=SlowMemory())
    agent.room = FakeRoom()
    agent._fake_session = FakeSession()

    with caplog.at_level(logging.WARNING):
        await agent.on_enter()

    assert "timed out" in caplog.text
    assert agent._fake_session.replies


async def test_a_memory_failure_does_not_stop_the_greeting(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    async def explode():
        raise RuntimeError("zep exploded")

    agent = agent_with(skip_greeting=False)
    agent.room = FakeRoom()
    agent._fake_session = FakeSession()
    monkeypatch.setattr(agent, "_inject_memory_context", explode)

    with caplog.at_level(logging.ERROR):
        await agent.on_enter()

    assert "greeting without it" in caplog.text
    assert agent._fake_session.replies


# =============================================================================
# _inject_memory_context
# =============================================================================


async def test_no_memory_injects_nothing() -> None:
    agent = agent_with()

    await agent._inject_memory_context()

    assert agent._last_memory_context is None


async def test_uninitialized_memory_injects_nothing() -> None:
    agent = agent_with(memory=FakeMemory(initialized=False))

    await agent._inject_memory_context()

    assert agent._last_memory_context is None


async def test_context_is_cached_for_the_greeting() -> None:
    """Avoids a second Zep round trip on the critical path."""
    context = MemoryContext(facts=["the user likes tea"])
    agent = agent_with(memory=FakeMemory(context=context))

    await agent._inject_memory_context()

    assert agent._last_memory_context is context


async def test_a_cached_user_name_is_logged_without_the_name(caplog) -> None:
    """The name is the user's, so the log records that one was found, not which.

    This test previously asserted the opposite -- that "Ada" appeared in the
    log line. It did, at INFO, along with every fact the user asked to be
    remembered, into whatever aggregator the deployment ships logs to.
    """
    agent = agent_with(memory=FakeMemory(user_name="Ada"))

    with caplog.at_level(logging.INFO):
        await agent._inject_memory_context()

    assert "Cached user name from memory" in caplog.text
    assert "A<3 chars>" in caplog.text
    assert "Ada" not in caplog.text


async def test_an_empty_context_does_not_rewrite_the_prompt() -> None:
    agent = agent_with(memory=FakeMemory(context=MemoryContext()))
    before = agent.instructions

    await agent._inject_memory_context()

    assert agent.instructions == before


async def test_an_injection_failure_is_logged_not_raised(caplog) -> None:
    agent = agent_with(memory=FakeMemory(get_context_error=RuntimeError("zep down")))

    with caplog.at_level(logging.ERROR):
        await agent._inject_memory_context()

    assert "Failed to inject memory context" in caplog.text


# =============================================================================
# _build_greeting_instructions
# =============================================================================


async def test_a_brand_new_user_gets_an_introduction() -> None:
    agent = agent_with(config=KwamiConfig(soul=KwamiSoulConfig(name="Ada")))

    instructions = await agent._build_greeting_instructions()

    assert "Introduce yourself" in instructions
    assert "Ada" in instructions


async def test_a_returning_user_without_a_name_is_asked_for_it() -> None:
    memory = FakeMemory(context=MemoryContext(facts=["likes tea"]))
    agent = agent_with(memory=memory)
    agent._last_memory_context = MemoryContext(facts=["likes tea"])

    instructions = await agent._build_greeting_instructions()

    assert "can't remember their name" in instructions


async def test_a_known_user_with_topics_is_greeted_with_one() -> None:
    context = MemoryContext(facts=["is learning the cello", "moved to Lisbon"])
    agent = agent_with(memory=FakeMemory(user_name="Ada", context=context))
    agent._last_memory_context = context

    instructions = await agent._build_greeting_instructions()

    assert "Ada" in instructions
    assert "cello" in instructions
    assert "Pick ONE topic" in instructions


async def test_name_facts_are_not_offered_as_conversation_topics() -> None:
    """ "your name is Ada" is not a thing to ask how it is going."""
    context = MemoryContext(facts=["the user's name is Ada"])
    agent = agent_with(memory=FakeMemory(user_name="Ada", context=context))
    agent._last_memory_context = context

    instructions = await agent._build_greeting_instructions()

    assert "name is Ada" not in instructions


async def test_a_known_user_with_only_a_summary_gets_a_check_in() -> None:
    context = MemoryContext(context_block="They have been busy with work.")
    agent = agent_with(memory=FakeMemory(user_name="Ada", context=context))
    agent._last_memory_context = context

    instructions = await agent._build_greeting_instructions()

    assert "summary of your past conversations" in instructions


async def test_a_summary_falls_back_to_the_thread_summary() -> None:
    context = MemoryContext(summary="Talked about trains.", recent_messages=[{"role": "user"}])
    agent = agent_with(memory=FakeMemory(user_name="Ada", context=context))
    agent._last_memory_context = context

    instructions = await agent._build_greeting_instructions()

    assert "Talked about trains" in instructions


async def test_a_known_user_with_no_history_gets_a_plain_hello() -> None:
    agent = agent_with(memory=FakeMemory(user_name="Ada", context=MemoryContext()))
    agent._last_memory_context = MemoryContext()

    instructions = await agent._build_greeting_instructions()

    assert "great to see you" in instructions


async def test_the_name_is_recovered_from_facts_when_not_cached() -> None:
    context = MemoryContext(facts=["the user's name is Grace"])
    memory = FakeMemory(user_name=None, context=context)
    agent = agent_with(memory=memory)
    agent._last_memory_context = context

    instructions = await agent._build_greeting_instructions()

    assert "Grace" in instructions
    assert memory.set_names == ["Grace"]


async def test_the_agents_own_name_is_never_adopted_as_the_users() -> None:
    context = MemoryContext(facts=["the assistant is called Ada"])
    memory = FakeMemory(user_name=None, context=context)
    agent = agent_with(config=KwamiConfig(soul=KwamiSoulConfig(name="Ada")), memory=memory)
    agent._last_memory_context = context

    await agent._build_greeting_instructions()

    assert memory.set_names == []


async def test_context_is_fetched_when_nothing_was_cached() -> None:
    context = MemoryContext(facts=["likes tea"])
    agent = agent_with(memory=FakeMemory(user_name="Ada", context=context))

    instructions = await agent._build_greeting_instructions()

    assert "tea" in instructions


async def test_a_memory_failure_still_yields_a_greeting(caplog) -> None:
    agent = agent_with(memory=FakeMemory(get_context_error=RuntimeError("zep down")))

    with caplog.at_level(logging.WARNING):
        instructions = await agent._build_greeting_instructions()

    assert "Could not extract user info" in caplog.text
    assert instructions


# =============================================================================
# Turn handling
# =============================================================================


async def test_a_user_turn_is_buffered() -> None:
    memory = FakeMemory(user_name="Ada")
    agent = agent_with(memory=memory)

    await agent.on_user_turn_completed(None, SimpleNamespace(text_content="hello there"))

    assert memory.buffered == [("hello there", "Ada")]


async def test_a_turn_without_memory_is_ignored() -> None:
    await agent_with().on_user_turn_completed(None, SimpleNamespace(text_content="hi"))


async def test_an_empty_turn_is_not_buffered() -> None:
    memory = FakeMemory()
    agent = agent_with(memory=memory)

    await agent.on_user_turn_completed(None, Unreadable())

    assert memory.buffered == []


async def test_a_whitespace_only_turn_still_falls_through_to_the_repr() -> None:
    """Pinning current behaviour, not endorsing it. `text_content` is only
    trusted when it strips to something non-empty, so a whitespace-only turn
    falls past every branch to `str(message)` -- which is exactly the pydantic
    repr that the text_content branch exists to prevent reaching Zep. Harmless
    today because such turns are rare, but it is the same defect in miniature.
    """
    memory = FakeMemory()
    agent = agent_with(memory=memory)

    await agent.on_user_turn_completed(None, SimpleNamespace(text_content="   "))

    assert memory.buffered
    assert "namespace(" in memory.buffered[0][0]


async def test_a_buffering_failure_is_warned(caplog) -> None:
    class Broken(FakeMemory):
        async def buffer_user_message(self, content: str, name: str | None = None) -> None:
            raise RuntimeError("zep down")

    agent = agent_with(memory=Broken())

    with caplog.at_level(logging.WARNING):
        await agent.on_user_turn_completed(None, SimpleNamespace(text_content="hi"))

    assert "Failed to buffer user message" in caplog.text


# =============================================================================
# Assistant turn persistence
# =============================================================================


def test_listeners_are_registered_once() -> None:
    agent = agent_with()
    agent._fake_session = FakeSession()

    agent._register_session_listeners()
    agent._register_session_listeners()

    assert len(agent._fake_session.handlers["conversation_item_added"]) == 1


def test_registration_is_deferred_when_there_is_no_activity() -> None:
    """on_enter calls this again once there is one."""
    agent = agent_with()

    agent._register_session_listeners()

    assert agent._session_listeners_registered is False


async def test_an_assistant_turn_is_persisted_as_an_exchange() -> None:
    """The hook this replaced -- on_agent_turn_completed -- is never dispatched
    by livekit-agents, so nothing the assistant said reached Zep at all."""
    memory = FakeMemory()
    agent = agent_with(config=KwamiConfig(soul=KwamiSoulConfig(name="Ada")), memory=memory)

    agent._on_conversation_item_added(
        SimpleNamespace(item=SimpleNamespace(role="assistant", text_content="the reply"))
    )
    await asyncio.gather(*agent._background_tasks)

    assert memory.exchanges == [{"content": "the reply", "name": "Ada"}]


async def test_the_persist_task_reference_is_retained() -> None:
    """A bare create_task can be garbage collected mid-flight and swallows its
    own exceptions."""
    agent = agent_with(memory=FakeMemory())

    agent._on_conversation_item_added(
        SimpleNamespace(item=SimpleNamespace(role="assistant", text_content="hi"))
    )

    assert len(agent._background_tasks) == 1


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(SimpleNamespace(item=None), id="no item"),
        pytest.param(
            SimpleNamespace(item=SimpleNamespace(role="user", text_content="hi")),
            id="user turn",
        ),
        pytest.param(
            SimpleNamespace(item=Unreadable("assistant")),
            id="unreadable content",
        ),
    ],
)
async def test_irrelevant_conversation_items_are_ignored(event: Any) -> None:
    agent = agent_with(memory=FakeMemory())

    agent._on_conversation_item_added(event)

    assert agent._background_tasks == set()


async def test_an_item_without_memory_is_ignored() -> None:
    agent = agent_with()

    agent._on_conversation_item_added(
        SimpleNamespace(item=SimpleNamespace(role="assistant", text_content="hi"))
    )

    assert agent._background_tasks == set()


async def test_a_failed_persist_is_warned_not_raised(caplog) -> None:
    class Broken(FakeMemory):
        async def add_exchange(self, **kwargs: Any) -> None:
            raise RuntimeError("zep down")

    agent = agent_with(memory=Broken())

    with caplog.at_level(logging.WARNING):
        await agent._persist_assistant_turn("the reply")

    assert "Failed to add exchange to memory" in caplog.text


def test_a_null_content_attribute_is_skipped() -> None:
    """`content=None` must fall through to `text`, not be read as absent."""
    agent = KwamiAgent()

    message = SimpleNamespace(content=None, text="the real text")

    assert agent._extract_message_content(message) == "the real text"


def test_an_empty_list_content_falls_through_to_the_next_attribute() -> None:
    agent = KwamiAgent()

    message = SimpleNamespace(content=[], text="the real text")

    assert agent._extract_message_content(message) == "the real text"


def test_a_list_of_non_strings_falls_through() -> None:
    """ChatMessage content can hold ImageContent alongside text; a list with no
    text parts carries nothing to store."""
    agent = KwamiAgent()

    message = SimpleNamespace(content=[object(), 42], text="the real text")

    assert agent._extract_message_content(message) == "the real text"


def test_a_whitespace_only_string_attribute_falls_through() -> None:
    agent = KwamiAgent()

    message = SimpleNamespace(content="   ", text="the real text")

    assert agent._extract_message_content(message) == "the real text"


def test_an_object_repr_is_filtered_rather_than_stored() -> None:
    """The last-resort branch: `<module.Class object at 0x...>` is not speech,
    and storing it is how Zep's graph filled with reprs."""
    agent = KwamiAgent()

    assert agent._extract_message_content(Unreadable()) == ""
