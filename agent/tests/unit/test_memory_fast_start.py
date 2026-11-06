"""What a session start costs in Zep round trips, and what it must not undo.

Every session start used to run `set_ontology` (a replace, which threw away any
ontology the user had set in the memory panel) and upsert the context template,
both before the greeting. The ontology is now applied once per version and
recorded on the user's Zep metadata; the template once per worker. The user's
name lives on their Zep record, so it is known without searching the graph.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.memory import manager
from src.memory.manager import (
    CUSTOM_ONTOLOGY,
    ONTOLOGY_METADATA_KEY,
    ONTOLOGY_VERSION,
    KwamiMemory,
)
from tests.unit.test_memory_manager import (
    FakeZepClient,
    RecordingGraph,
    RecordingUser,
    memory_config,
    ready_memory,
)


@pytest.fixture(autouse=True)
def fresh_worker():
    """Each test is a fresh worker process: nothing settled yet."""
    manager._ONTOLOGY_SETTLED.clear()
    manager._TEMPLATES_READY.clear()
    yield
    manager._ONTOLOGY_SETTLED.clear()
    manager._TEMPLATES_READY.clear()


class StampedUser(RecordingUser):
    """A Zep user record with metadata and a name, and an `update` that records."""

    def __init__(
        self,
        *,
        exists: bool = True,
        metadata: dict | None = None,
        first_name: str | None = None,
        update_error: Exception | None = None,
    ) -> None:
        super().__init__(exists=exists)
        self.metadata = metadata
        self.first_name = first_name
        self.update_error = update_error
        self.updates: list[dict[str, Any]] = []

    async def get(self, user_id: str):
        await super().get(user_id)
        return SimpleNamespace(user_id=user_id, metadata=self.metadata, first_name=self.first_name)

    async def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)
        if self.update_error is not None:
            raise self.update_error


class CountingGraph(RecordingGraph):
    def __init__(self) -> None:
        super().__init__()
        self.ontologies: list[dict[str, Any]] = []

    async def set_ontology(self, **kwargs: Any) -> None:
        self.ontologies.append(kwargs)


class CountingContext:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def update_context_template(self, **kwargs: Any) -> None:
        self.calls.append("update")

    async def create_context_template(self, **kwargs: Any) -> None:
        self.calls.append("create")


def install(monkeypatch, user: StampedUser) -> list:
    """Every client built in this test shares the same Zep user, graph and context."""
    graph = CountingGraph()
    context = CountingContext()
    clients: list = []

    class Client(FakeZepClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.user = user
            self.graph = graph
            self.context = context
            clients.append(self)

    monkeypatch.setattr("src.memory.manager.get_zep_imports", lambda: (Client, object, "RoleType"))
    return clients


async def start(**config: Any) -> KwamiMemory:
    memory = KwamiMemory(memory_config(**config), "kwami-1", "Jarvis")
    assert await memory.initialize() is True
    return memory


async def test_the_default_ontology_is_applied_once_and_stamped(monkeypatch) -> None:
    user = StampedUser(metadata={"kwami_id": "kwami-1"})
    clients = install(monkeypatch, user)

    await start(configure_ontology=True)

    assert len(clients[0].graph.ontologies) == 1
    assert user.updates == [
        {
            "user_id": "kwami_kwami-1",
            "metadata": {"kwami_id": "kwami-1", ONTOLOGY_METADATA_KEY: ONTOLOGY_VERSION},
        }
    ]


async def test_a_stamped_user_is_not_sent_the_ontology_again(monkeypatch) -> None:
    user = StampedUser(metadata={ONTOLOGY_METADATA_KEY: ONTOLOGY_VERSION})
    clients = install(monkeypatch, user)

    await start(configure_ontology=True)

    assert clients[0].graph.ontologies == []
    assert user.updates == []


async def test_a_custom_ontology_is_never_overwritten(monkeypatch) -> None:
    """The panel lets a user set their own types; a session start used to replace them."""
    user = StampedUser(metadata={ONTOLOGY_METADATA_KEY: CUSTOM_ONTOLOGY})
    clients = install(monkeypatch, user)

    await start(configure_ontology=True)

    assert clients[0].graph.ontologies == []


async def test_older_defaults_are_brought_up_to_date(monkeypatch) -> None:
    user = StampedUser(metadata={ONTOLOGY_METADATA_KEY: "2020-01"})
    clients = install(monkeypatch, user)

    await start(configure_ontology=True)

    assert len(clients[0].graph.ontologies) == 1


async def test_a_new_user_gets_the_ontology(monkeypatch) -> None:
    user = StampedUser(exists=False)
    clients = install(monkeypatch, user)

    await start(configure_ontology=True)

    assert len(clients[0].graph.ontologies) == 1
    assert user.updates[0]["metadata"] == {ONTOLOGY_METADATA_KEY: ONTOLOGY_VERSION}


async def test_the_second_session_on_a_worker_skips_the_check(monkeypatch) -> None:
    user = StampedUser()
    clients = install(monkeypatch, user)

    await start(configure_ontology=True)
    user.metadata = None  # a stale read must not trigger a second replace
    await start(configure_ontology=True)

    assert len(clients[0].graph.ontologies) == 1


async def test_an_unrecorded_stamp_is_retried_next_session(monkeypatch, caplog) -> None:
    user = StampedUser(update_error=RuntimeError("zep busy"))
    clients = install(monkeypatch, user)

    with caplog.at_level(logging.WARNING):
        await start(configure_ontology=True)
        await start(configure_ontology=True)

    assert len(clients[0].graph.ontologies) == 2
    assert "Could not record the ontology version" in caplog.text


async def test_a_failed_ontology_is_not_stamped(monkeypatch) -> None:
    user = StampedUser()
    install(monkeypatch, user)

    async def refuse(*args: Any, **kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(manager, "configure_ontology", refuse)
    await start(configure_ontology=True)

    assert user.updates == []
    assert "kwami_kwami-1" not in manager._ONTOLOGY_SETTLED


async def test_the_template_is_set_up_once_per_worker(monkeypatch) -> None:
    user = StampedUser()
    clients = install(monkeypatch, user)

    first = await start(configure_ontology=False)
    second = await start(configure_ontology=False)

    assert clients[0].context.calls == ["update"]
    assert first._template_id == second._template_id == "kwami-context-kwami_kwami-1"


async def test_a_template_that_could_not_be_set_up_is_tried_again(monkeypatch) -> None:
    install(monkeypatch, StampedUser())

    async def unavailable(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(manager, "setup_context_template", unavailable)
    memory = await start(configure_ontology=False)

    assert memory._template_id is None
    assert not manager._TEMPLATES_READY


async def test_the_name_on_the_zep_record_is_known_at_once(monkeypatch) -> None:
    install(monkeypatch, StampedUser(first_name="  Daniel "))

    memory = await start(configure_ontology=False)

    assert memory._cached_user_name == "Daniel"
    assert await memory.get_user_name() == "Daniel"


async def test_a_blank_name_on_the_record_is_ignored(monkeypatch) -> None:
    install(monkeypatch, StampedUser(first_name="   "))

    memory = await start(configure_ontology=False)

    assert memory._cached_user_name is None


async def test_a_name_found_in_the_graph_is_written_to_the_record(monkeypatch) -> None:
    user = StampedUser()
    install(monkeypatch, user)
    memory = await start(configure_ontology=False)

    async def found(*args: Any, **kwargs: Any) -> str:
        return "Ada"

    monkeypatch.setattr(manager, "get_user_name", found)
    assert await memory.get_user_name() == "Ada"
    await asyncio.gather(*memory._background)

    assert user.updates == [{"user_id": "kwami_kwami-1", "first_name": "Ada"}]


async def test_a_name_given_in_conversation_is_written_to_the_record() -> None:
    client = FakeZepClient()
    client.user = StampedUser()
    memory = ready_memory(client)

    memory.set_user_name("Ada")
    await asyncio.gather(*memory._background)

    assert client.user.updates == [{"user_id": "kwami_kwami-1", "first_name": "Ada"}]


async def test_a_failed_name_write_is_not_an_error() -> None:
    client = FakeZepClient()
    client.user = StampedUser(update_error=RuntimeError("zep busy"))
    memory = ready_memory(client)

    memory.set_user_name("Ada")
    await asyncio.gather(*memory._background)

    assert memory._cached_user_name == "Ada"


async def test_no_client_no_write() -> None:
    memory = ready_memory()
    memory._client = None

    await memory._store_user_name("Ada")


def test_a_name_set_before_initialize_is_only_cached() -> None:
    memory = KwamiMemory(memory_config(), "kwami-1")

    memory.set_user_name("Ada")

    assert memory._cached_user_name == "Ada"
    assert memory._background == set()


def test_no_running_loop_drops_the_write_cleanly() -> None:
    memory = ready_memory()

    async def write() -> None:  # closed by _in_background, never run
        raise AssertionError

    memory._in_background(write())

    assert memory._background == set()
