"""Memory context: what actually reaches the system prompt before the greeting.

Two things here have bitten production. Facts about the assistant were being
injected as facts about the user ("Kwami is an AI assistant" read back as
something true of the human), and both fallback accessors were named after
zep-cloud methods that do not exist, so `summary` and `recent_messages` were
unconditionally empty.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.memory.context import (
    DEFAULT_CONTEXT_TEMPLATE,
    MAX_CONTEXT_BLOCK_CHARS,
    MAX_ENTITIES,
    MAX_FACT_CHARS,
    MAX_FACTS,
    MAX_SUMMARY_CHARS,
    TEMPLATE_PREFIX,
    MemoryContext,
    _is_assistant_fact,
    get_context,
    setup_context_template,
)


def edge(fact: str | None = None, invalid_at: Any = None):
    return SimpleNamespace(fact=fact, invalid_at=invalid_at)


def message(role: str | None = "user", content: str = "hi", role_type: str | None = None):
    return SimpleNamespace(role=role, content=content, role_type=role_type)


class FakeContextApi:
    def __init__(self, update_error=None, create_error=None) -> None:
        self.update_error = update_error
        self.create_error = create_error
        self.updated: list[dict] = []
        self.created: list[dict] = []

    async def update_context_template(self, **kwargs: Any) -> None:
        self.updated.append(kwargs)
        if self.update_error is not None:
            raise self.update_error

    async def create_context_template(self, **kwargs: Any) -> None:
        self.created.append(kwargs)
        if self.create_error is not None:
            raise self.create_error


class FakeThreadApi:
    def __init__(
        self,
        user_contexts: Any = None,
        user_context_error: Exception | None = None,
        messages: Any = None,
        messages_error: Exception | None = None,
    ) -> None:
        self._user_contexts = user_contexts
        self._user_context_error = user_context_error
        self._messages = messages
        self._messages_error = messages_error
        self.context_calls: list[dict] = []
        self.get_calls: list[dict] = []

    async def get_user_context(self, **kwargs: Any):
        self.context_calls.append(kwargs)
        if self._user_context_error is not None:
            raise self._user_context_error
        if isinstance(self._user_contexts, list):
            return self._user_contexts.pop(0) if self._user_contexts else None
        return self._user_contexts

    async def get(self, **kwargs: Any):
        self.get_calls.append(kwargs)
        if self._messages_error is not None:
            raise self._messages_error
        return self._messages


class FakeGraph:
    def __init__(self, edges: Any = None, error: Exception | None = None) -> None:
        self._edges = edges
        self._error = error
        self.calls: list[dict] = []

    async def search(self, **kwargs: Any):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(edges=self._edges)


class FakeZep:
    def __init__(self, thread=None, graph=None, context=None) -> None:
        self.thread = thread or FakeThreadApi()
        self.graph = graph or FakeGraph()
        self.context = context or FakeContextApi()


# =============================================================================
# MemoryContext
# =============================================================================


def test_the_mutable_defaults_are_per_instance() -> None:
    """`facts: list = None` on a dataclass is a footgun; __post_init__ is what
    stops two contexts sharing one list."""
    a, b = MemoryContext(), MemoryContext()
    a.facts.append("mine")

    assert b.facts == []


def test_explicit_collections_survive_post_init() -> None:
    ctx = MemoryContext(facts=["a"], entities=[{"name": "n"}], recent_messages=[{"role": "user"}])

    assert ctx.facts == ["a"]
    assert ctx.entities == [{"name": "n"}]
    assert ctx.recent_messages == [{"role": "user"}]


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"context_block": "x"}, id="block"),
        pytest.param({"summary": "x"}, id="summary"),
        pytest.param({"facts": ["x"]}, id="facts"),
        pytest.param({"recent_messages": [{"role": "user"}]}, id="messages"),
    ],
)
def test_any_retrieved_content_counts_as_content(kwargs: dict) -> None:
    assert MemoryContext(**kwargs).has_content() is True


def test_an_empty_context_has_no_content() -> None:
    """Billing keys off this, so an empty result must not be charged."""
    assert MemoryContext().has_content() is False


def test_entities_alone_are_not_content() -> None:
    """Deliberate: entities without facts or a summary is not a useful recall
    and must not be billed as one."""
    assert MemoryContext(entities=[{"name": "n"}]).has_content() is False


# =============================================================================
# to_system_prompt_addition
# =============================================================================


def test_the_template_block_is_preferred_over_the_components() -> None:
    ctx = MemoryContext(context_block="FROM TEMPLATE", summary="ignored", facts=["ignored"])

    assert ctx.to_system_prompt_addition() == "FROM TEMPLATE"


def test_the_template_block_is_capped() -> None:
    """Anything entering the LLM is size-capped; an unbounded Zep block would
    push the persona out of the context window."""
    ctx = MemoryContext(context_block="x" * (MAX_CONTEXT_BLOCK_CHARS + 500))

    assert len(ctx.to_system_prompt_addition()) == MAX_CONTEXT_BLOCK_CHARS


def test_an_empty_context_renders_as_the_empty_string() -> None:
    assert MemoryContext().to_system_prompt_addition() == ""


def test_the_summary_is_rendered_and_capped() -> None:
    rendered = MemoryContext(summary="s" * (MAX_SUMMARY_CHARS + 50)).to_system_prompt_addition()

    assert "## Conversation Summary" in rendered
    assert "s" * MAX_SUMMARY_CHARS in rendered
    assert "s" * (MAX_SUMMARY_CHARS + 1) not in rendered


def test_facts_are_limited_in_number_and_length() -> None:
    ctx = MemoryContext(facts=[f"fact {i} " + "x" * 300 for i in range(MAX_FACTS + 5)])

    rendered = ctx.to_system_prompt_addition()

    assert rendered.count("\n- ") == MAX_FACTS
    for line in rendered.splitlines():
        if line.startswith("- "):
            assert len(line) <= MAX_FACT_CHARS + 2


def test_the_facts_block_says_they_are_about_the_human() -> None:
    """Without this the model repeatedly attributed user facts to itself."""
    rendered = MemoryContext(facts=["likes tea"]).to_system_prompt_addition()

    assert "NOT about you, the assistant" in rendered
    assert "no longer valid" in rendered


def test_entities_are_rendered_with_a_summary() -> None:
    ctx = MemoryContext(
        summary="s", entities=[{"name": "Ada", "summary": "a person the user knows"}]
    )

    rendered = ctx.to_system_prompt_addition()

    assert "## Relevant Entities" in rendered
    assert "- Ada: a person the user knows" in rendered


def test_an_entity_without_a_summary_falls_back_to_its_type() -> None:
    ctx = MemoryContext(summary="s", entities=[{"name": "Ada", "type": "Person"}])

    assert "- Ada: Person" in ctx.to_system_prompt_addition()


def test_an_entity_with_neither_summary_nor_type_falls_back_to_entity() -> None:
    ctx = MemoryContext(summary="s", entities=[{"name": "Ada"}])

    assert "- Ada: entity" in ctx.to_system_prompt_addition()


def test_a_nameless_entity_renders_as_unknown() -> None:
    ctx = MemoryContext(summary="s", entities=[{"summary": "mystery"}])

    assert "- Unknown: mystery" in ctx.to_system_prompt_addition()


def test_entities_are_limited_in_number() -> None:
    ctx = MemoryContext(
        summary="s", entities=[{"name": f"E{i}"} for i in range(MAX_ENTITIES + 4)]
    )

    rendered = ctx.to_system_prompt_addition().split("## Relevant Entities")[1]

    assert rendered.count("\n- ") == MAX_ENTITIES


def test_all_three_sections_are_joined() -> None:
    ctx = MemoryContext(summary="s", facts=["f"], entities=[{"name": "E"}])

    rendered = ctx.to_system_prompt_addition()

    assert "## Conversation Summary" in rendered
    assert "## Known Facts" in rendered
    assert "## Relevant Entities" in rendered


# =============================================================================
# _is_assistant_fact
# =============================================================================


@pytest.mark.parametrize(
    "fact",
    [
        "Kwami is an AI assistant",
        "Kwami was created recently",
        "Kwami can browse the web",
        "The assistant's name is kwami",
        "the bot called Kwami",
        "a helper named Kwami",
        "I'm Kwami",
        "I am Kwami",
        "Kwami helps the user",
    ],
)
def test_facts_about_the_assistant_are_recognised(fact: str) -> None:
    assert _is_assistant_fact(fact, "kwami") is True


@pytest.mark.parametrize(
    "fact",
    [
        "The user likes coffee",
        "Ada works at a university",
        "the user asked Kwami about trains",
    ],
)
def test_facts_about_the_user_are_not_filtered(fact: str) -> None:
    assert _is_assistant_fact(fact, "kwami") is False


# =============================================================================
# setup_context_template
# =============================================================================


async def test_an_existing_template_is_updated() -> None:
    client = FakeZep()

    template_id = await setup_context_template(client, "user-1")

    assert template_id == f"{TEMPLATE_PREFIX}-user-1"
    assert client.context.updated[0]["template"] == DEFAULT_CONTEXT_TEMPLATE
    assert client.context.created == []


async def test_a_missing_template_is_created(caplog) -> None:
    """First use: there is nothing to update, so the update failing is normal
    -- but it is logged rather than silently swallowed."""
    client = FakeZep(context=FakeContextApi(update_error=RuntimeError("404 not found")))

    with caplog.at_level(logging.DEBUG):
        template_id = await setup_context_template(client, "user-1")

    assert template_id == f"{TEMPLATE_PREFIX}-user-1"
    assert client.context.created[0]["template_id"] == f"{TEMPLATE_PREFIX}-user-1"
    assert "Could not update context template" in caplog.text


async def test_a_custom_template_is_used() -> None:
    client = FakeZep()

    await setup_context_template(client, "user-1", template="# CUSTOM")

    assert client.context.updated[0]["template"] == "# CUSTOM"


async def test_failing_to_create_returns_none(caplog) -> None:
    """Templates are optional; a session runs without one."""
    client = FakeZep(
        context=FakeContextApi(
            update_error=RuntimeError("no such template"),
            create_error=RuntimeError("not on your plan"),
        )
    )

    with caplog.at_level(logging.DEBUG):
        assert await setup_context_template(client, "user-1") is None

    assert "Could not set up context template" in caplog.text


# =============================================================================
# get_context
# =============================================================================


async def test_the_template_path_fills_the_context_block() -> None:
    client = FakeZep(thread=FakeThreadApi(user_contexts=SimpleNamespace(context="BLOCK")))

    context = await get_context(client, "u", "s", template_id="tpl")

    assert context.context_block == "BLOCK"
    assert client.thread.context_calls[0]["template_id"] == "tpl"


async def test_no_template_id_skips_straight_to_the_fallback() -> None:
    client = FakeZep(thread=FakeThreadApi(user_contexts=SimpleNamespace(context="SUMMARY")))

    context = await get_context(client, "u", "s")

    assert context.context_block is None
    assert context.summary == "SUMMARY"
    assert client.thread.context_calls[0]["mode"] == "summary"


async def test_a_failing_template_falls_back_to_the_summary(caplog) -> None:
    client = FakeZep(thread=FakeThreadApi(user_context_error=RuntimeError("no template")))

    with caplog.at_level(logging.DEBUG):
        context = await get_context(client, "u", "s", template_id="tpl")

    assert context.context_block is None
    assert "falling back" in caplog.text


async def test_an_empty_template_response_falls_back() -> None:
    client = FakeZep(
        thread=FakeThreadApi(
            user_contexts=[SimpleNamespace(context=""), SimpleNamespace(context="SUMMARY")]
        )
    )

    context = await get_context(client, "u", "s", template_id="tpl")

    assert context.summary == "SUMMARY"


async def test_a_successful_template_skips_the_fallback_entirely() -> None:
    client = FakeZep(
        thread=FakeThreadApi(user_contexts=[SimpleNamespace(context="BLOCK")]),
        graph=FakeGraph(edges=[edge("should not be fetched")]),
    )

    context = await get_context(client, "u", "s", template_id="tpl")

    assert context.facts == []
    assert client.graph.calls == []


async def test_a_thread_context_failure_is_warned_not_raised(caplog) -> None:
    client = FakeZep(thread=FakeThreadApi(user_context_error=RuntimeError("thread gone")))

    with caplog.at_level(logging.WARNING):
        context = await get_context(client, "u", "s")

    assert context.summary is None
    assert "Could not retrieve thread context" in caplog.text


async def test_facts_are_fetched_and_assistant_facts_filtered() -> None:
    client = FakeZep(
        thread=FakeThreadApi(user_contexts=None),
        graph=FakeGraph(
            edges=[
                edge("Kwami is an AI assistant"),
                edge("the user likes coffee"),
                edge(None),
                edge(""),
            ]
        ),
    )

    context = await get_context(client, "u", "s")

    assert context.facts == ["the user likes coffee"]


async def test_an_expired_fact_is_annotated_with_its_end_date() -> None:
    client = FakeZep(
        thread=FakeThreadApi(user_contexts=None),
        graph=FakeGraph(edges=[edge("the user lived in Paris", invalid_at="2025-01-01")]),
    )

    context = await get_context(client, "u", "s")

    assert context.facts == ["the user lived in Paris (no longer valid since 2025-01-01)"]


async def test_a_currently_valid_fact_is_not_annotated() -> None:
    client = FakeZep(
        thread=FakeThreadApi(user_contexts=None),
        graph=FakeGraph(edges=[edge("the user likes tea", invalid_at="present")]),
    )

    assert (await get_context(client, "u", "s")).facts == ["the user likes tea"]


async def test_facts_can_be_switched_off() -> None:
    client = FakeZep(thread=FakeThreadApi(user_contexts=None), graph=FakeGraph(edges=[edge("f")]))

    context = await get_context(client, "u", "s", include_facts=False)

    assert context.facts == []
    assert client.graph.calls == []


async def test_a_graph_failure_leaves_facts_empty(caplog) -> None:
    client = FakeZep(
        thread=FakeThreadApi(user_contexts=None), graph=FakeGraph(error=RuntimeError("graph down"))
    )

    with caplog.at_level(logging.DEBUG):
        context = await get_context(client, "u", "s")

    assert context.facts == []
    assert "Could not retrieve facts via graph" in caplog.text


async def test_an_empty_edge_response_leaves_facts_empty() -> None:
    client = FakeZep(thread=FakeThreadApi(user_contexts=None), graph=FakeGraph(edges=None))

    assert (await get_context(client, "u", "s")).facts == []


async def test_recent_messages_are_always_fetched() -> None:
    """`thread.get` with `lastn`; the old `thread.get_messages` does not exist,
    which left recent_messages unconditionally empty."""
    client = FakeZep(
        thread=FakeThreadApi(
            user_contexts=SimpleNamespace(context="BLOCK"),
            messages=SimpleNamespace(messages=[message("user", "hello")]),
        )
    )

    context = await get_context(client, "u", "s", template_id="tpl", max_messages=4)

    assert context.recent_messages == [{"role": "user", "content": "hello"}]
    assert client.thread.get_calls[0] == {"thread_id": "s", "lastn": 4}


async def test_a_message_without_a_role_falls_back_to_role_type() -> None:
    client = FakeZep(
        thread=FakeThreadApi(
            user_contexts=None,
            messages=SimpleNamespace(messages=[message(None, "hi", role_type="assistant")]),
        )
    )

    context = await get_context(client, "u", "s")

    assert context.recent_messages == [{"role": "assistant", "content": "hi"}]


async def test_a_message_fetch_failure_is_swallowed(caplog) -> None:
    client = FakeZep(
        thread=FakeThreadApi(user_contexts=None, messages_error=RuntimeError("no thread"))
    )

    with caplog.at_level(logging.DEBUG):
        context = await get_context(client, "u", "s")

    assert context.recent_messages == []
    assert "Could not retrieve thread messages" in caplog.text


async def test_an_empty_message_response_leaves_messages_empty() -> None:
    client = FakeZep(
        thread=FakeThreadApi(user_contexts=None, messages=SimpleNamespace(messages=[]))
    )

    assert (await get_context(client, "u", "s")).recent_messages == []
