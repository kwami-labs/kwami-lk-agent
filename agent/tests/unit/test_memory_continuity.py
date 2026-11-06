"""A session that knows how the last one ended, and recall that follows the talk.

Each session is a new Zep thread. Context was read once, before the greeting,
relative to that brand-new (empty) thread, and the retrieval preset only changed
how many of its messages were read -- none. So the kwami opened every session
with no idea what had been said last time, and the preset made no difference.

Now: the end of the last session that had anything in it is carried in, Zep's
context is asked relative to it until this session has an exchange of its own,
the preset sets how much reaches the prompt, and on the standard pipeline the
block is read again as the conversation moves on.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.memory import context as context_mod
from src.memory import manager
from src.memory.context import (
    BALANCED_BUDGET,
    LEAN_BUDGET,
    RICH_BUDGET,
    LastConversation,
    MemoryContext,
    budget_for,
    get_context,
    load_last_conversation,
    setup_shared_template,
)
from tests.unit.test_memory_manager import FakeZepClient, ready_memory


@pytest.fixture(autouse=True)
def fresh_worker():
    manager._TEMPLATES_READY.clear()
    yield
    manager._TEMPLATES_READY.clear()


def m(content: str, role: str = "user", name: str | None = None, at: str | None = None):
    return SimpleNamespace(content=content, role=role, role_type=None, name=name, created_at=at)


class Threads:
    """`user.get_threads` + `thread.get(lastn=)` over scripted sessions."""

    def __init__(self, sessions: dict[str, tuple[str, list]], *, list_error=None, read_error=()):
        self.sessions = sessions
        self.list_error = list_error
        self.read_error = set(read_error)
        self.reads: list[dict] = []
        self.context_threads: list[str] = []

    async def get_threads(self, user_id: str):
        if self.list_error:
            raise self.list_error
        return [
            SimpleNamespace(thread_id=tid, created_at=created)
            for tid, (created, _) in self.sessions.items()
        ]

    async def get(self, *, thread_id: str, lastn: int | None = None, **kwargs: Any):
        self.reads.append({"thread_id": thread_id, "lastn": lastn})
        if thread_id in self.read_error:
            raise RuntimeError("thread unavailable")
        messages = self.sessions.get(thread_id, ("", []))[1]
        return SimpleNamespace(messages=messages[-lastn:] if lastn else messages)

    async def get_user_context(self, *, thread_id: str, **kwargs: Any):
        self.context_threads.append(thread_id)
        return SimpleNamespace(context=f"CONTEXT FOR {thread_id}")


def zep_with(threads: Threads) -> Any:
    client = SimpleNamespace(
        user=SimpleNamespace(get_threads=threads.get_threads),
        thread=threads,
        context=SimpleNamespace(),
    )
    return client


SESSIONS = {
    "t-current": ("2026-11-06T10:00:00Z", []),
    "t-empty": ("2026-11-05T20:00:00Z", []),
    "t-last": (
        "2026-11-05T18:00:00Z",
        [
            m("Ponme una alarma", at="2026-11-05T18:00:05Z"),
            m("Hecho, señor", "assistant", "Jarvis", "2026-11-05T18:00:07Z"),
            m("Y mañana revisamos el informe", at="2026-11-05T18:01:00Z"),
        ],
    ),
    "t-old": ("2026-10-01T09:00:00Z", [m("hace mucho")]),
}


# =============================================================================
# Budgets
# =============================================================================


@pytest.mark.parametrize(
    ("messages", "budget"),
    [
        (0, LEAN_BUDGET),
        (4, LEAN_BUDGET),
        (5, BALANCED_BUDGET),
        (10, BALANCED_BUDGET),
        (11, RICH_BUDGET),
        (16, RICH_BUDGET),
    ],
)
def test_each_preset_has_its_budget(messages: int, budget) -> None:
    assert budget_for(messages) is budget


def test_a_richer_preset_puts_more_in_front_of_the_model() -> None:
    assert LEAN_BUDGET.chars < BALANCED_BUDGET.chars < RICH_BUDGET.chars
    assert LEAN_BUDGET.edges < BALANCED_BUDGET.edges < RICH_BUDGET.edges


def test_a_budget_names_one_shared_template() -> None:
    assert RICH_BUDGET.template_id == "kwami-context-v2-e40-n20"
    assert "%{edges limit=40}" in RICH_BUDGET.template
    assert "%{entities limit=20}" in RICH_BUDGET.template
    assert "%{user_summary}" in RICH_BUDGET.template


def test_the_budget_caps_the_block() -> None:
    block = "x" * 10_000
    assert (
        len(MemoryContext(context_block=block, budget=LEAN_BUDGET).to_system_prompt_addition())
        == 1500
    )
    assert (
        len(MemoryContext(context_block=block, budget=RICH_BUDGET).to_system_prompt_addition())
        == 6000
    )


def test_the_fallback_keeps_as_many_facts_as_the_budget() -> None:
    facts = [f"fact {i}" for i in range(30)]
    rich = MemoryContext(facts=facts, budget=RICH_BUDGET).to_system_prompt_addition()
    lean = MemoryContext(facts=facts, budget=LEAN_BUDGET).to_system_prompt_addition()
    assert rich.count("\n- fact") == RICH_BUDGET.facts
    assert lean.count("\n- fact") == LEAN_BUDGET.facts


# =============================================================================
# The last conversation in the prompt
# =============================================================================


def carried(*lines: tuple[str, str, str | None], at: str | None = "2026-11-05T18:01:00Z"):
    return MemoryContext(
        context_block="FACTS",
        last_conversation=[{"role": r, "content": c, "name": n} for r, c, n in lines],
        last_conversation_at=at,
    )


def test_the_last_conversation_follows_the_facts() -> None:
    text = carried(
        ("user", "Ponme una alarma", None), ("assistant", "Hecho", "Jarvis")
    ).to_system_prompt_addition()
    assert text.startswith(
        "FACTS\n\n## How your last conversation ended (ended 2026-11-05T18:01:00Z)"
    )
    assert text.endswith("User: Ponme una alarma\nJarvis: Hecho")


def test_an_unnamed_assistant_and_an_undated_ending() -> None:
    text = carried(("assistant", "Hola", None), at=None).to_system_prompt_addition()
    assert "## How your last conversation ended\n" in text
    assert "Assistant: Hola" in text


def test_blank_lines_are_dropped_and_long_ones_cut() -> None:
    text = carried(("user", "   ", None), ("user", "x" * 1000, None)).to_system_prompt_addition()
    assert "User: " + "x" * context_mod.MAX_CARRIED_MESSAGE_CHARS in text
    assert "x" * (context_mod.MAX_CARRIED_MESSAGE_CHARS + 1) not in text


def test_a_conversation_of_only_blanks_adds_nothing() -> None:
    assert carried(("user", " ", None)).to_system_prompt_addition() == "FACTS"


def test_the_conversation_takes_at_most_a_third_and_keeps_its_end() -> None:
    lines = [("user", f"line {i} " + "y" * 200, None) for i in range(40)]
    ctx = carried(*lines)
    ctx.context_block = "F" * 5000
    text = ctx.to_system_prompt_addition()
    assert len(text) <= BALANCED_BUDGET.chars
    assert "line 39" in text
    assert "line 0 " not in text
    assert text.startswith("F" * 100)


def test_only_a_last_conversation_is_still_content() -> None:
    ctx = MemoryContext(last_conversation=[{"role": "user", "content": "hola"}])
    assert ctx.has_content()
    assert "User: hola" in ctx.to_system_prompt_addition()


# =============================================================================
# load_last_conversation
# =============================================================================


async def test_the_newest_earlier_session_with_words_is_found() -> None:
    threads = Threads(SESSIONS)
    last = await load_last_conversation(zep_with(threads), "u", "t-current", 2)

    assert last == LastConversation(
        thread_id="t-last",
        messages=[
            {
                "role": "assistant",
                "content": "Hecho, señor",
                "name": "Jarvis",
                "created_at": "2026-11-05T18:00:07Z",
            },
            {
                "role": "user",
                "content": "Y mañana revisamos el informe",
                "name": None,
                "created_at": "2026-11-05T18:01:00Z",
            },
        ],
        ended_at="2026-11-05T18:01:00Z",
    )
    assert [r["thread_id"] for r in threads.reads] == ["t-empty", "t-last"]
    assert threads.reads[0]["lastn"] == 2


async def test_the_scan_stops_after_a_few_empty_sessions() -> None:
    sessions = {f"t-{i}": (f"2026-11-0{i}", []) for i in range(1, 8)}
    threads = Threads(sessions)
    assert await load_last_conversation(zep_with(threads), "u", "t-now", 5) is None
    assert len(threads.reads) == context_mod.MAX_THREADS_TO_SCAN


async def test_an_unreadable_session_is_skipped() -> None:
    threads = Threads(SESSIONS, read_error={"t-last"})
    last = await load_last_conversation(zep_with(threads), "u", "t-current", 5)
    assert last is not None and last.thread_id == "t-old"
    assert last.ended_at is None


async def test_no_listing_no_last_conversation() -> None:
    threads = Threads(SESSIONS, list_error=RuntimeError("zep down"))
    assert await load_last_conversation(zep_with(threads), "u", "t-current", 5) is None


async def test_no_messages_asked_for_reads_nothing() -> None:
    threads = Threads(SESSIONS)
    assert await load_last_conversation(zep_with(threads), "u", "t-current", 0) is None
    assert threads.reads == []


async def test_an_enum_role_is_read_by_value() -> None:
    role = SimpleNamespace(value="user")
    threads = Threads(
        {"t-1": ("2026", [SimpleNamespace(content="hi", role=role, name=None, created_at=None)])}
    )
    last = await load_last_conversation(zep_with(threads), "u", "t-now", 5)
    assert last is not None and last.messages[0]["role"] == "user"


# =============================================================================
# get_context
# =============================================================================


async def test_preread_messages_are_not_read_again() -> None:
    threads = Threads(SESSIONS)
    ctx = await get_context(
        zep_with(threads), "u", "t-last", template_id="tpl", recent_messages=[{"content": "x"}]
    )
    assert ctx.recent_messages == [{"content": "x"}]
    assert threads.reads == []


async def test_facts_under_the_relevance_floor_are_left_out() -> None:
    edges = [
        SimpleNamespace(fact="relevant", relevance=0.9, invalid_at=None),
        SimpleNamespace(fact="noise", relevance=0.1, invalid_at=None),
        SimpleNamespace(fact="unscored", invalid_at=None),
    ]

    async def search(**kwargs: Any):
        return SimpleNamespace(edges=edges)

    async def no_context(**kwargs: Any):
        return SimpleNamespace(context=None)

    async def no_messages(**kwargs: Any):
        return SimpleNamespace(messages=[])

    client = SimpleNamespace(
        graph=SimpleNamespace(search=search),
        thread=SimpleNamespace(get_user_context=no_context, get=no_messages),
    )
    ctx = await get_context(client, "u", "s", min_relevance=0.5, budget=RICH_BUDGET)
    assert ctx.facts == ["relevant", "unscored"]
    assert ctx.budget is RICH_BUDGET


# =============================================================================
# setup_shared_template
# =============================================================================


class Templates:
    def __init__(self, update_error=None, create_error=None) -> None:
        self.update_error, self.create_error = update_error, create_error
        self.calls: list[tuple[str, dict]] = []

    async def update_context_template(self, **kwargs: Any) -> None:
        self.calls.append(("update", kwargs))
        if self.update_error:
            raise self.update_error

    async def create_context_template(self, **kwargs: Any) -> None:
        self.calls.append(("create", kwargs))
        if self.create_error:
            raise self.create_error


async def test_an_existing_shared_template_is_updated() -> None:
    templates = Templates()
    assert (
        await setup_shared_template(SimpleNamespace(context=templates), RICH_BUDGET)
        == RICH_BUDGET.template_id
    )
    assert templates.calls == [
        ("update", {"template_id": RICH_BUDGET.template_id, "template": RICH_BUDGET.template})
    ]


async def test_a_missing_shared_template_is_created() -> None:
    templates = Templates(update_error=RuntimeError("404"))
    assert (
        await setup_shared_template(SimpleNamespace(context=templates), LEAN_BUDGET)
        == LEAN_BUDGET.template_id
    )
    assert [c for c, _ in templates.calls] == ["update", "create"]


async def test_a_template_zep_refuses_is_none() -> None:
    templates = Templates(update_error=RuntimeError("404"), create_error=RuntimeError("plan"))
    assert await setup_shared_template(SimpleNamespace(context=templates), LEAN_BUDGET) is None


# =============================================================================
# KwamiMemory
# =============================================================================


class MemoryZep(FakeZepClient):
    def __init__(self, threads: Threads) -> None:
        super().__init__()
        self.threads = threads
        self.user.get_threads = threads.get_threads
        self.thread.get = threads.get
        self.thread.get_user_context = threads.get_user_context
        self.context = Templates()


def memory_on(threads: Threads, **config: Any):
    memory = ready_memory(MemoryZep(threads), **config)
    memory._session_id = "t-current"
    return memory


async def test_a_new_session_recalls_relative_to_the_last_one() -> None:
    threads = Threads(SESSIONS)
    memory = memory_on(threads)

    ctx = await memory.get_context()

    assert threads.context_threads == ["t-last"]
    assert ctx.context_block == "CONTEXT FOR t-last"
    assert ctx.last_conversation[-1]["content"] == "Y mañana revisamos el informe"
    assert ctx.recent_messages == ctx.last_conversation, "the greeting sees a returning user"
    assert ctx.last_conversation_at == "2026-11-05T18:01:00Z"


async def test_after_an_exchange_recall_follows_this_session() -> None:
    threads = Threads(SESSIONS)
    memory = memory_on(threads)
    memory._pending_user_message = ("hola", None)
    await memory.add_exchange("buenas", "Jarvis")

    ctx = await memory.get_context()

    assert threads.context_threads == ["t-current"]
    assert ctx.last_conversation, "how the last session ended stays in view"


async def test_the_last_conversation_is_read_once_per_session() -> None:
    threads = Threads(SESSIONS)
    memory = memory_on(threads)

    await memory.get_context()
    reads = len(threads.reads)
    await memory.get_context()

    assert len(threads.reads) == reads


async def test_a_first_session_has_no_last_conversation() -> None:
    threads = Threads({"t-current": ("2026", [])})
    memory = memory_on(threads)

    ctx = await memory.get_context()

    assert threads.context_threads == ["t-current"]
    assert ctx.last_conversation == []


async def test_the_preset_reaches_the_template_and_the_block() -> None:
    threads = Threads(SESSIONS)
    memory = memory_on(threads, max_context_messages=16)

    ctx = await memory.get_context()

    assert memory.budget is RICH_BUDGET
    assert ctx.budget is RICH_BUDGET
    assert memory._client.context.calls[0][1]["template_id"] == RICH_BUDGET.template_id


async def test_a_live_preset_change_moves_to_its_template() -> None:
    threads = Threads(SESSIONS)
    memory = memory_on(threads, max_context_messages=4)
    await memory.get_context()
    memory.config.max_context_messages = 16
    await memory.get_context()

    used = [kwargs["template_id"] for _, kwargs in memory._client.context.calls]
    assert used == [LEAN_BUDGET.template_id, RICH_BUDGET.template_id]


async def test_a_refresh_is_due_after_the_first_exchange_and_every_few() -> None:
    memory = memory_on(Threads(SESSIONS))
    due = []
    for _ in range(7):
        memory._pending_user_message = ("hola", None)
        await memory.add_exchange("buenas", "Jarvis")
        due.append(memory.take_refresh_due())

    assert due == [True, False, True, False, False, True, False]
    assert memory.take_refresh_due() is False


async def test_a_failed_write_is_not_an_exchange() -> None:
    memory = memory_on(Threads(SESSIONS))
    memory._client.thread.add_error = RuntimeError("zep down")
    memory._pending_user_message = ("hola", None)

    await memory.add_exchange("buenas", "Jarvis")

    assert memory._exchanges == 0
    assert memory.take_refresh_due() is False


async def test_without_a_client_there_is_no_last_conversation() -> None:
    memory = ready_memory()
    memory._client = None

    assert await memory.last_conversation() is None


# =============================================================================
# The agent refreshes its memory block as the talk moves on
# =============================================================================


class RefreshingMemory:
    is_initialized = True
    _cached_user_name = None

    def __init__(self, texts: list[str], due: bool = True, error: Exception | None = None):
        self.texts = texts
        self.due = due
        self.error = error
        self.contexts = 0
        self.exchanges = 0

    async def add_exchange(self, **kwargs: Any) -> None:
        self.exchanges += 1

    def take_refresh_due(self) -> bool:
        return self.due

    async def get_context(self) -> MemoryContext:
        if self.error:
            raise self.error
        text = self.texts[min(self.contexts, len(self.texts) - 1)]
        self.contexts += 1
        return MemoryContext(context_block=text)


class RecordingAgent(KwamiAgent):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.instruction_updates: list[str] = []

    async def update_instructions(self, instructions: str) -> None:  # type: ignore[override]
        self.instruction_updates.append(instructions)


def standard_agent(memory: Any, pipeline: str = "standard") -> RecordingAgent:
    config = KwamiConfig()
    config.voice.pipeline_type = pipeline
    return RecordingAgent(config=config, memory=memory)


async def test_a_due_refresh_rewrites_the_memory_block() -> None:
    memory = RefreshingMemory(["the talk moved to the trip"])
    agent = standard_agent(memory)

    await agent._persist_assistant_turn("reply")

    assert agent._memory_text == "the talk moved to the trip"
    assert "the talk moved to the trip" in agent.instruction_updates[-1]


async def test_an_unchanged_block_is_not_resent() -> None:
    memory = RefreshingMemory(["same"])
    agent = standard_agent(memory)
    agent._memory_text = "same"

    await agent._persist_assistant_turn("reply")

    assert agent.instruction_updates == []


async def test_an_empty_block_is_not_sent() -> None:
    agent = standard_agent(RefreshingMemory([""]))

    await agent._persist_assistant_turn("reply")

    assert agent.instruction_updates == []


async def test_no_refresh_when_none_is_due() -> None:
    memory = RefreshingMemory(["new"], due=False)
    agent = standard_agent(memory)

    await agent._persist_assistant_turn("reply")

    assert memory.contexts == 0


@pytest.mark.parametrize("pipeline", ["realtime", "hybrid"])
async def test_a_realtime_model_is_not_rewritten_mid_session(pipeline: str) -> None:
    """The Gemini API resends the whole prompt as a turn on every instruction update."""
    memory = RefreshingMemory(["new"])
    agent = standard_agent(memory, pipeline)

    await agent._persist_assistant_turn("reply")

    assert memory.contexts == 0
    assert agent.instruction_updates == []


async def test_a_failed_refresh_is_logged_not_raised(caplog) -> None:
    agent = standard_agent(RefreshingMemory([], error=RuntimeError("zep down")))

    with caplog.at_level(logging.ERROR):
        await agent._persist_assistant_turn("reply")

    assert "Failed to refresh memory context" in caplog.text


async def test_a_memory_without_refresh_support_is_left_alone() -> None:
    class Plain:
        is_initialized = True

        async def add_exchange(self, **kwargs: Any) -> None:
            return None

    agent = standard_agent(Plain())

    await agent._persist_assistant_turn("reply")

    assert agent.instruction_updates == []
