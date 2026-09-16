"""A config message must not mint a new Zep client every time.

`create_memory` builds a fresh `AsyncZep`, re-runs `set_ontology` (a
destructive project-level replace), re-upserts the context template and mints a
new `session_{user}_{uuid4}` thread. Running that per config message churned
one client per message -- none of which could be closed -- and scattered a
single conversation across several threads, so recall found nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.handlers.config_handler import _reuse_existing_memory
from src.session import SessionState


@dataclass
class FakeMemoryConfig:
    user_id: str = "kwami_abc_123"
    max_context_messages: int = 10
    include_facts: bool = True
    min_fact_relevance: float = 0.5


class FakeMemory:
    def __init__(self, user_id: str = "kwami_abc_123", initialized: bool = True) -> None:
        self.config = FakeMemoryConfig(user_id=user_id)
        self.is_initialized = initialized


class FakeAgent:
    def __init__(self, memory=None) -> None:
        self._memory = memory


def _state(memory=None) -> SessionState:
    return SessionState(current_agent=FakeAgent(memory))


def test_the_live_client_is_reused_for_the_same_user() -> None:
    memory = FakeMemory()
    reused = _reuse_existing_memory(_state(memory), FakeMemoryConfig())
    assert reused is memory


def test_a_different_user_gets_a_fresh_client() -> None:
    """Switching kwami must not inherit another kwami's memory."""
    memory = FakeMemory(user_id="kwami_abc_123")
    assert _reuse_existing_memory(_state(memory), FakeMemoryConfig(user_id="kwami_xyz_999")) is None


def test_an_uninitialized_memory_is_not_reused() -> None:
    memory = FakeMemory(initialized=False)
    assert _reuse_existing_memory(_state(memory), FakeMemoryConfig()) is None


def test_no_agent_or_no_memory_yields_nothing_to_reuse() -> None:
    assert _reuse_existing_memory(SessionState(), FakeMemoryConfig()) is None
    assert _reuse_existing_memory(_state(None), FakeMemoryConfig()) is None


def test_retrieval_settings_are_carried_onto_the_reused_client() -> None:
    """Reuse must not freeze the memory knobs at their first-seen values."""
    memory = FakeMemory()
    incoming = FakeMemoryConfig(
        max_context_messages=42,
        include_facts=False,
        min_fact_relevance=0.9,
    )

    reused = _reuse_existing_memory(_state(memory), incoming)

    assert reused is memory
    assert memory.config.max_context_messages == 42
    assert memory.config.include_facts is False
    assert memory.config.min_fact_relevance == 0.9
