"""`KwamiMemory` is the whole Zep surface the rest of the agent sees.

Three defects this class carries scar tissue from, all pinned below: messages
were stored as pydantic reprs instead of utterances, every reconfiguration
leaked a connection pool because `AsyncZep.close()` does not exist, and Zep
calls were billed even when they failed and returned empty.

The Zep client is a hand-written double with explicit keyword signatures, not
a MagicMock -- a wrong call shape still fails here. That the methods it stands
in for are real is pinned in tests/contract/test_zep_contract.py.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.domain import KwamiMemoryConfig, UsageTracker
from src.memory.context import MemoryContext
from src.memory.manager import KwamiMemory, create_memory


class RecordingThread:
    def __init__(
        self,
        *,
        exists: bool = True,
        add_error: Exception | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self.exists = exists
        self.add_error = add_error
        self.create_error = create_error
        self.added: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []

    async def get(self, *, thread_id: str, lastn: int | None = None):
        if not self.exists:
            raise RuntimeError("no such thread")
        return SimpleNamespace(messages=[])

    async def create(self, *, thread_id: str, user_id: str) -> None:
        self.created.append({"thread_id": thread_id, "user_id": user_id})
        if self.create_error is not None:
            raise self.create_error

    async def add_messages(
        self, *, thread_id: str, messages: list, ignore_roles: list | None = None
    ) -> None:
        self.added.append(
            {"thread_id": thread_id, "messages": messages, "ignore_roles": ignore_roles}
        )
        if self.add_error is not None:
            raise self.add_error

    async def get_user_context(self, **kwargs: Any):
        return SimpleNamespace(context=None)


class RecordingUser:
    def __init__(self, *, exists: bool = True, add_error: Exception | None = None) -> None:
        self.exists = exists
        self.add_error = add_error
        self.added: list[dict[str, Any]] = []

    async def get(self, user_id: str):
        if not self.exists:
            raise RuntimeError("no such user")
        return SimpleNamespace(user_id=user_id)

    async def add(self, *, user_id: str, metadata: dict) -> None:
        self.added.append({"user_id": user_id, "metadata": metadata})
        if self.add_error is not None:
            raise self.add_error


class RecordingGraph:
    def __init__(self, edges: list | None = None) -> None:
        self.edges = edges or []
        self.node = SimpleNamespace(get_by_user_id=self._nodes)

    async def search(self, **kwargs: Any):
        return SimpleNamespace(edges=self.edges, nodes=None)

    async def _nodes(self, **kwargs: Any):
        return []

    async def set_ontology(self, **kwargs: Any) -> None:
        return None


class FakeZepClient:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.user = RecordingUser()
        self.thread = RecordingThread()
        self.graph = RecordingGraph()
        self.context = SimpleNamespace(
            update_context_template=self._noop, create_context_template=self._noop
        )
        self.closed = False

    async def _noop(self, **kwargs: Any) -> None:
        return None

    async def aclose(self) -> None:
        self.closed = True


def memory_config(**overrides: Any) -> KwamiMemoryConfig:
    defaults: dict[str, Any] = {"enabled": True, "api_key": "zep-test-key"}
    defaults.update(overrides)
    return KwamiMemoryConfig(**defaults)


def ready_memory(client: Any = None, **config_overrides: Any) -> KwamiMemory:
    """A memory already past initialize(), with a client injected."""
    memory = KwamiMemory(memory_config(**config_overrides), "kwami-1", "Kwami")
    memory._client = client or FakeZepClient()
    memory._user_id = "kwami_kwami-1"
    memory._session_id = "session_1"
    memory._initialized = True
    return memory


# =============================================================================
# Construction and properties
# =============================================================================


def test_memory_is_enabled_only_with_a_key() -> None:
    assert KwamiMemory(memory_config(), "k").is_enabled is True
    assert KwamiMemory(memory_config(api_key=""), "k").is_enabled is False
    assert KwamiMemory(memory_config(enabled=False), "k").is_enabled is False


def test_a_fresh_memory_is_not_initialized() -> None:
    memory = KwamiMemory(memory_config(), "k")

    assert memory.is_initialized is False
    assert memory.user_id is None
    assert memory.session_id is None


def test_the_usage_tracker_can_be_attached_after_construction() -> None:
    memory = KwamiMemory(memory_config(), "k")
    tracker = UsageTracker()

    memory.set_usage_tracker(tracker)
    memory._record_usage("zep/thing", units_used=2.0)

    assert tracker.get_usage_summary()[0]["model_id"] == "zep/thing"


def test_recording_usage_without_a_tracker_is_harmless() -> None:
    KwamiMemory(memory_config(), "k")._record_usage("zep/thing")


# =============================================================================
# initialize
# =============================================================================


@pytest.fixture
def zep_factory(monkeypatch: pytest.MonkeyPatch):
    """Swap the module's lazy-import seam for a recording double."""
    created: list[FakeZepClient] = []

    def install(client_cls: Any = FakeZepClient, message_cls: Any = None) -> list:
        def fake_imports():
            return client_cls, message_cls or ZepMessageDouble, "RoleType"

        monkeypatch.setattr("src.memory.manager.get_zep_imports", fake_imports)
        return created

    return install


class ZepMessageDouble:
    """Stands in for zep_cloud Message, recording exactly what was stored."""

    def __init__(self, *, role: str, content: str, name: str, created_at: str) -> None:
        self.role = role
        self.content = content
        self.name = name
        self.created_at = created_at


async def test_initialize_builds_a_client_and_ids(zep_factory) -> None:
    zep_factory()
    memory = KwamiMemory(memory_config(configure_ontology=False), "kwami-1", "Ada")

    assert await memory.initialize() is True

    assert memory.is_initialized is True
    assert memory.user_id == "kwami_kwami-1"
    assert memory.session_id.startswith("session_kwami_kwami-1_")


async def test_initialize_passes_an_explicit_timeout(zep_factory) -> None:
    """Without one the SDK waits 60s per call, on the path before the greeting."""
    zep_factory()
    memory = KwamiMemory(memory_config(configure_ontology=False), "kwami-1")

    await memory.initialize()

    assert memory._client.init_kwargs["timeout"]


async def test_configured_ids_are_honoured(zep_factory) -> None:
    zep_factory()
    memory = KwamiMemory(
        memory_config(user_id="fixed-user", session_id="fixed-session", configure_ontology=False),
        "kwami-1",
    )

    await memory.initialize()

    assert memory.user_id == "fixed-user"
    assert memory.session_id == "fixed-session"


async def test_a_disabled_memory_refuses_to_initialize(caplog) -> None:
    memory = KwamiMemory(memory_config(api_key=""), "kwami-1")

    with caplog.at_level(logging.WARNING):
        assert await memory.initialize() is False

    assert "disabled or API key not configured" in caplog.text


async def test_a_missing_sdk_disables_memory(monkeypatch, caplog) -> None:
    monkeypatch.setattr("src.memory.manager.get_zep_imports", lambda: (None, None, None))
    memory = KwamiMemory(memory_config(), "kwami-1")

    with caplog.at_level(logging.ERROR):
        assert await memory.initialize() is False

    assert "zep_cloud not available" in caplog.text


async def test_initialize_failing_leaves_memory_uninitialized(zep_factory, caplog) -> None:
    class ExplodingClient(FakeZepClient):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("bad api key")

    zep_factory(ExplodingClient)
    memory = KwamiMemory(memory_config(), "kwami-1")

    with caplog.at_level(logging.ERROR):
        assert await memory.initialize() is False

    assert memory.is_initialized is False
    assert "Failed to initialize memory" in caplog.text


async def test_the_ontology_is_configured_when_asked(zep_factory) -> None:
    zep_factory()
    memory = KwamiMemory(memory_config(configure_ontology=True), "kwami-1")

    assert await memory.initialize() is True


# =============================================================================
# _ensure_user_exists / _ensure_session_exists
# =============================================================================


async def test_an_existing_user_is_not_recreated(zep_factory) -> None:
    zep_factory()
    memory = KwamiMemory(memory_config(configure_ontology=False), "kwami-1")

    await memory.initialize()

    assert memory._client.user.added == []


async def test_a_missing_user_is_created_and_billed(zep_factory) -> None:
    class MissingUserClient(FakeZepClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.user = RecordingUser(exists=False)

    zep_factory(MissingUserClient)
    tracker = UsageTracker()
    memory = KwamiMemory(
        memory_config(configure_ontology=False), "kwami-1", "Ada", usage_tracker=tracker
    )

    await memory.initialize()

    assert memory._client.user.added[0]["metadata"]["assistant_name"] == "Ada"
    assert any(e["model_id"] == "zep/create_user" for e in tracker.get_usage_summary())


async def test_a_creation_race_is_tolerated(zep_factory, caplog) -> None:
    """Two workers can start the same kwami at once; the loser must not fail."""

    class RacingClient(FakeZepClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.user = RecordingUser(
                exists=False, add_error=RuntimeError("400: user already exists")
            )

    zep_factory(RacingClient)
    memory = KwamiMemory(memory_config(configure_ontology=False), "kwami-1")

    with caplog.at_level(logging.INFO):
        assert await memory.initialize() is True

    assert "race condition" in caplog.text


async def test_a_real_user_creation_failure_aborts_initialize(zep_factory, caplog) -> None:
    class BrokenClient(FakeZepClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.user = RecordingUser(exists=False, add_error=RuntimeError("500 boom"))

    zep_factory(BrokenClient)
    memory = KwamiMemory(memory_config(configure_ontology=False), "kwami-1")

    with caplog.at_level(logging.ERROR):
        assert await memory.initialize() is False

    assert "Failed to create user" in caplog.text


async def test_a_missing_thread_is_created_and_billed(zep_factory) -> None:
    class MissingThreadClient(FakeZepClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.thread = RecordingThread(exists=False)

    zep_factory(MissingThreadClient)
    tracker = UsageTracker()
    memory = KwamiMemory(memory_config(configure_ontology=False), "kwami-1", usage_tracker=tracker)

    await memory.initialize()

    assert memory._client.thread.created
    assert any(e["model_id"] == "zep/create_thread" for e in tracker.get_usage_summary())


async def test_a_thread_creation_failure_aborts_initialize(zep_factory, caplog) -> None:
    class BrokenThreadClient(FakeZepClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.thread = RecordingThread(exists=False, create_error=RuntimeError("500 boom"))

    zep_factory(BrokenThreadClient)
    memory = KwamiMemory(memory_config(configure_ontology=False), "kwami-1")

    with caplog.at_level(logging.ERROR):
        assert await memory.initialize() is False

    assert "Failed to create thread" in caplog.text


# =============================================================================
# buffer_user_message / add_exchange
# =============================================================================


async def test_buffering_holds_the_message_for_the_next_exchange() -> None:
    memory = ready_memory()

    await memory.buffer_user_message("  hello  ", name="Ada")

    assert memory._pending_user_message == ("hello", "Ada")


@pytest.mark.parametrize("content", ["", "   "])
async def test_an_empty_user_message_is_not_buffered(content: str) -> None:
    memory = ready_memory()

    await memory.buffer_user_message(content)

    assert memory._pending_user_message is None


async def test_buffering_on_an_uninitialized_memory_is_a_no_op() -> None:
    memory = KwamiMemory(memory_config(), "k")

    await memory.buffer_user_message("hello")

    assert memory._pending_user_message is None


async def test_a_second_user_message_flushes_the_first(monkeypatch) -> None:
    """The user spoke twice without a reply; the first turn must not be lost."""
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    memory = ready_memory()

    await memory.buffer_user_message("first")
    await memory.buffer_user_message("second")

    assert [m.content for m in memory._client.thread.added[0]["messages"]] == ["first"]
    assert memory._pending_user_message == ("second", None)


async def test_an_exchange_stores_both_turns_as_utterances(monkeypatch) -> None:
    """The defect this replaced: messages were stored as pydantic reprs, so the
    graph was built from `ChatMessage(content=[...])` strings."""
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    memory = ready_memory()
    memory.set_user_name("Ada")

    await memory.buffer_user_message("what's the weather")
    await memory.add_exchange("It is sunny.")

    stored = memory._client.thread.added[0]["messages"]
    assert [(m.role, m.content) for m in stored] == [
        ("user", "what's the weather"),
        ("assistant", "It is sunny."),
    ]
    assert stored[0].name == "Ada"
    assert stored[1].name == "Kwami"


async def test_an_exchange_ignores_assistant_roles_for_graph_building(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    memory = ready_memory()

    await memory.add_exchange("hello")

    assert memory._client.thread.added[0]["ignore_roles"] == ["assistant"]


async def test_an_exchange_is_billed(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    tracker = UsageTracker()
    memory = ready_memory()
    memory.set_usage_tracker(tracker)

    await memory.add_exchange("hello")

    assert any(e["model_id"] == "zep/add_messages" for e in tracker.get_usage_summary())


async def test_an_exchange_with_nothing_to_say_sends_nothing(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    memory = ready_memory()

    await memory.add_exchange("   ")

    assert memory._client.thread.added == []


async def test_an_exchange_on_an_uninitialized_memory_is_a_no_op() -> None:
    memory = KwamiMemory(memory_config(), "k")

    await memory.add_exchange("hello")


async def test_an_exchange_without_the_sdk_is_a_no_op(monkeypatch) -> None:
    monkeypatch.setattr("src.memory.manager.get_zep_imports", lambda: (None, None, None))
    memory = ready_memory()

    await memory.add_exchange("hello")

    assert memory._client.thread.added == []


async def test_a_failed_exchange_is_logged_not_raised(monkeypatch, caplog) -> None:
    """Storing memory must never break the turn the user is having."""
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    client = FakeZepClient()
    client.thread = RecordingThread(add_error=RuntimeError("zep down"))
    memory = ready_memory(client)

    with caplog.at_level(logging.ERROR):
        await memory.add_exchange("hello")

    assert "Failed to add messages to memory" in caplog.text


async def test_an_unnamed_user_defaults_to_user(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    memory = ready_memory()

    await memory.buffer_user_message("hi")
    await memory.add_exchange("hello")

    assert memory._client.thread.added[0]["messages"][0].name == "User"


# =============================================================================
# add_message
# =============================================================================


@pytest.fixture
def messaging_memory(monkeypatch):
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    return ready_memory()


@pytest.mark.parametrize("role", ["user", "assistant", "system"])
async def test_each_known_role_is_stored(messaging_memory, role: str) -> None:
    await messaging_memory.add_message(role, "content")

    assert messaging_memory._client.thread.added[0]["messages"][0].role == role


async def test_an_unknown_role_falls_back_to_user(messaging_memory, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        await messaging_memory.add_message("narrator", "content")

    assert messaging_memory._client.thread.added[0]["messages"][0].role == "user"
    assert "Unknown role" in caplog.text


async def test_the_role_is_normalised(messaging_memory) -> None:
    await messaging_memory.add_message("  USER  ", "content")

    assert messaging_memory._client.thread.added[0]["messages"][0].role == "user"


@pytest.mark.parametrize(
    ("role", "expected"),
    [("user", "User"), ("assistant", "Kwami"), ("system", "System")],
)
async def test_the_default_name_follows_the_role(
    messaging_memory, role: str, expected: str
) -> None:
    await messaging_memory.add_message(role, "content")

    assert messaging_memory._client.thread.added[0]["messages"][0].name == expected


async def test_a_cached_user_name_is_used_for_user_messages(messaging_memory) -> None:
    messaging_memory.set_user_name("Ada")

    await messaging_memory.add_message("user", "content")

    assert messaging_memory._client.thread.added[0]["messages"][0].name == "Ada"


async def test_an_explicit_name_wins(messaging_memory) -> None:
    await messaging_memory.add_message("user", "content", name="Grace")

    assert messaging_memory._client.thread.added[0]["messages"][0].name == "Grace"


async def test_system_messages_are_not_ignored_for_graph_building(
    messaging_memory,
) -> None:
    """A fact added as a system message exists precisely to create graph
    entities, so it must not be in ignore_roles."""
    await messaging_memory.add_message("system", "content")

    assert messaging_memory._client.thread.added[0]["ignore_roles"] is None


@pytest.mark.parametrize("content", ["", "   "])
async def test_an_empty_message_is_not_sent(messaging_memory, content: str) -> None:
    await messaging_memory.add_message("user", content)

    assert messaging_memory._client.thread.added == []


async def test_add_message_on_an_uninitialized_memory_is_a_no_op() -> None:
    await KwamiMemory(memory_config(), "k").add_message("user", "hi")


async def test_add_message_without_the_sdk_is_a_no_op(monkeypatch) -> None:
    monkeypatch.setattr("src.memory.manager.get_zep_imports", lambda: (None, None, None))
    memory = ready_memory()

    await memory.add_message("user", "hi")

    assert memory._client.thread.added == []


async def test_a_failed_add_message_is_logged_not_raised(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    client = FakeZepClient()
    client.thread = RecordingThread(add_error=RuntimeError("zep down"))
    memory = ready_memory(client)

    with caplog.at_level(logging.ERROR):
        await memory.add_message("user", "hi")

    assert "Failed to add message to memory" in caplog.text


async def test_a_fact_is_stored_as_a_system_message(messaging_memory) -> None:
    await messaging_memory.add_fact("the user prefers tea")

    stored = messaging_memory._client.thread.added[0]["messages"][0]
    assert stored.role == "system"
    assert "the user prefers tea" in stored.content


# =============================================================================
# _flush_pending_message
# =============================================================================


async def test_flushing_with_nothing_buffered_is_a_no_op(messaging_memory) -> None:
    await messaging_memory._flush_pending_message()

    assert messaging_memory._client.thread.added == []


async def test_a_flush_without_the_sdk_still_clears_the_buffer(monkeypatch) -> None:
    memory = ready_memory()
    memory._pending_user_message = ("hello", None)
    monkeypatch.setattr("src.memory.manager.get_zep_imports", lambda: (None, None, None))

    await memory._flush_pending_message()

    assert memory._pending_user_message is None


async def test_a_failed_flush_is_warned_not_raised(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    client = FakeZepClient()
    client.thread = RecordingThread(add_error=RuntimeError("zep down"))
    memory = ready_memory(client)
    memory._pending_user_message = ("hello", None)

    with caplog.at_level(logging.WARNING):
        await memory._flush_pending_message()

    assert "Failed to flush pending message" in caplog.text


async def test_a_flush_prefers_the_buffered_name(messaging_memory) -> None:
    messaging_memory.set_user_name("Cached")
    messaging_memory._pending_user_message = ("hello", "Buffered")

    await messaging_memory._flush_pending_message()

    assert messaging_memory._client.thread.added[0]["messages"][0].name == "Buffered"


async def test_a_flush_falls_back_to_the_cached_name(messaging_memory) -> None:
    messaging_memory.set_user_name("Cached")
    messaging_memory._pending_user_message = ("hello", None)

    await messaging_memory._flush_pending_message()

    assert messaging_memory._client.thread.added[0]["messages"][0].name == "Cached"


# =============================================================================
# get_context / search / entities / user name
# =============================================================================


async def test_context_on_an_uninitialized_memory_is_empty() -> None:
    context = await KwamiMemory(memory_config(), "k").get_context()

    assert context.has_content() is False


async def test_a_context_with_content_is_billed(monkeypatch) -> None:
    async def fake_get_context(**kwargs: Any) -> MemoryContext:
        return MemoryContext(context_block="BLOCK")

    monkeypatch.setattr("src.memory.manager.get_context", fake_get_context)
    tracker = UsageTracker()
    memory = ready_memory()
    memory.set_usage_tracker(tracker)

    await memory.get_context()

    assert any(e["model_id"] == "zep/get_context" for e in tracker.get_usage_summary())


async def test_an_empty_context_is_not_billed(monkeypatch) -> None:
    """get_context swallows its own errors and returns empty, which is
    indistinguishable from success unless has_content() is checked."""

    async def fake_get_context(**kwargs: Any) -> MemoryContext:
        return MemoryContext()

    monkeypatch.setattr("src.memory.manager.get_context", fake_get_context)
    tracker = UsageTracker()
    memory = ready_memory()
    memory.set_usage_tracker(tracker)

    await memory.get_context()

    assert tracker.get_usage_summary() == []


async def test_a_context_failure_returns_an_empty_context(monkeypatch, caplog) -> None:
    async def boom(**kwargs: Any) -> MemoryContext:
        raise RuntimeError("zep down")

    monkeypatch.setattr("src.memory.manager.get_context", boom)
    memory = ready_memory()

    with caplog.at_level(logging.ERROR):
        context = await memory.get_context()

    assert context.has_content() is False
    assert "Failed to get memory context" in caplog.text


async def test_search_on_an_uninitialized_memory_is_empty() -> None:
    assert await KwamiMemory(memory_config(), "k").search("q") == []


async def test_a_search_with_results_is_billed() -> None:
    client = FakeZepClient()
    client.graph = RecordingGraph(edges=[SimpleNamespace(fact="a fact", score=1.0)])
    tracker = UsageTracker()
    memory = ready_memory(client)
    memory.set_usage_tracker(tracker)

    results = await memory.search("q")

    assert results
    assert any(e["model_id"] == "zep/graph_search" for e in tracker.get_usage_summary())


async def test_an_empty_search_is_not_billed() -> None:
    """The search helper swallows its exceptions and returns [], so an
    unconditional record charged for calls that never reached Zep."""
    tracker = UsageTracker()
    memory = ready_memory()
    memory.set_usage_tracker(tracker)

    assert await memory.search("q") == []
    assert tracker.get_usage_summary() == []


async def test_entities_on_an_uninitialized_memory_are_empty() -> None:
    assert await KwamiMemory(memory_config(), "k").get_entities_by_type("Person") == []


async def test_entities_are_delegated_to_the_graph() -> None:
    assert await ready_memory().get_entities_by_type("Person") == []


async def test_a_cached_user_name_short_circuits_the_lookup() -> None:
    memory = KwamiMemory(memory_config(), "k")
    memory.set_user_name("Ada")

    assert await memory.get_user_name() == "Ada"


async def test_a_user_name_lookup_on_an_uninitialized_memory_is_none() -> None:
    assert await KwamiMemory(memory_config(), "k").get_user_name() is None


async def test_a_found_user_name_is_cached_and_billed(monkeypatch) -> None:
    async def fake_lookup(*args: Any) -> str:
        return "Ada"

    monkeypatch.setattr("src.memory.manager.get_user_name", fake_lookup)
    tracker = UsageTracker()
    memory = ready_memory()
    memory.set_usage_tracker(tracker)

    assert await memory.get_user_name() == "Ada"
    assert memory._cached_user_name == "Ada"
    assert any(e["model_id"] == "zep/get_user_name" for e in tracker.get_usage_summary())


async def test_a_missing_user_name_is_not_cached(monkeypatch) -> None:
    async def fake_lookup(*args: Any) -> None:
        return None

    monkeypatch.setattr("src.memory.manager.get_user_name", fake_lookup)
    memory = ready_memory()

    assert await memory.get_user_name() is None
    assert memory._cached_user_name is None


async def test_a_failed_user_name_lookup_returns_none(monkeypatch, caplog) -> None:
    async def boom(*args: Any) -> None:
        raise RuntimeError("graph down")

    monkeypatch.setattr("src.memory.manager.get_user_name", boom)
    memory = ready_memory()

    with caplog.at_level(logging.DEBUG):
        assert await memory.get_user_name() is None

    assert "Could not get user name" in caplog.text


# =============================================================================
# close
# =============================================================================


async def test_close_releases_the_client_and_resets_state() -> None:
    client = FakeZepClient()
    memory = ready_memory(client)

    await memory.close()

    assert client.closed is True
    assert memory._client is None
    assert memory.is_initialized is False


async def test_close_flushes_a_pending_message(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.memory.manager.get_zep_imports", lambda: (None, ZepMessageDouble, None)
    )
    client = FakeZepClient()
    memory = ready_memory(client)
    memory._pending_user_message = ("last thing said", None)

    await memory.close()

    assert client.thread.added[0]["messages"][0].content == "last thing said"


async def test_a_failing_flush_does_not_stop_the_close(monkeypatch) -> None:
    def boom():
        raise RuntimeError("no sdk")

    monkeypatch.setattr("src.memory.manager.get_zep_imports", boom)
    client = FakeZepClient()
    memory = ready_memory(client)
    memory._pending_user_message = ("x", None)

    await memory.close()

    assert client.closed is True


async def test_closing_without_a_client_is_harmless() -> None:
    memory = KwamiMemory(memory_config(), "k")

    await memory.close()

    assert memory.is_initialized is False


async def test_a_client_exposing_only_sync_close_is_still_closed() -> None:
    class SyncCloseClient(FakeZepClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            del self.__dict__["closed"]
            self.closed = False

        aclose = None  # type: ignore[assignment]

        def close(self) -> None:
            self.closed = True

    client = SyncCloseClient()
    memory = ready_memory(client)

    await memory.close()

    assert client.closed is True


async def test_a_client_with_no_close_falls_back_to_the_httpx_pool() -> None:
    """AsyncZep exposes neither close() nor aclose(); the pool lives on the
    wrapped httpx client. Missing this leaked a pool per reconfiguration."""

    class Pool:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    pool = Pool()

    class WrapperOnlyClient:
        def __init__(self) -> None:
            self._client_wrapper = SimpleNamespace(httpx_client=pool)

    memory = ready_memory(WrapperOnlyClient())

    await memory.close()

    assert pool.closed is True


async def test_a_client_with_nothing_closeable_does_not_raise() -> None:
    class Opaque:
        pass

    memory = ready_memory(Opaque())

    await memory.close()

    assert memory._client is None


async def test_a_failing_close_is_warned_not_raised(caplog) -> None:
    class BadClose(FakeZepClient):
        async def aclose(self) -> None:
            raise RuntimeError("pool stuck")

    memory = ready_memory(BadClose())

    with caplog.at_level(logging.WARNING):
        await memory.close()

    assert "Failed to close Zep client cleanly" in caplog.text
    assert memory._client is None


# =============================================================================
# create_memory
# =============================================================================


async def test_create_memory_returns_an_initialized_instance(zep_factory) -> None:
    zep_factory()

    memory = await create_memory(memory_config(configure_ontology=False), "kwami-1", "Ada")

    assert memory is not None
    assert memory.is_initialized is True


async def test_create_memory_returns_none_when_disabled(caplog) -> None:
    with caplog.at_level(logging.INFO):
        memory = await create_memory(memory_config(api_key=""), "kwami-1", "Ada")

    assert memory is None
    assert "Memory disabled" in caplog.text


async def test_create_memory_returns_none_when_initialize_fails(monkeypatch) -> None:
    monkeypatch.setattr("src.memory.manager.get_zep_imports", lambda: (None, None, None))

    assert await create_memory(memory_config(), "kwami-1") is None


async def test_create_memory_passes_the_usage_tracker(zep_factory) -> None:
    zep_factory()
    tracker = UsageTracker()

    memory = await create_memory(
        memory_config(configure_ontology=False), "kwami-1", usage_tracker=tracker
    )

    assert memory._usage_tracker is tracker


async def test_releasing_the_pool_without_a_client_is_a_no_op() -> None:
    """`close()` guards on `self._client` before calling this, but the helper
    carries its own guard so a direct call during teardown cannot raise."""
    memory = KwamiMemory(memory_config(), "k")

    await memory._aclose_client()
