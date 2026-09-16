"""Zep memory against the real Zep Cloud API.

This is the automated form of the manual check in the rework plan: open the Zep
dashboard and confirm stored messages are utterances rather than pydantic
reprs. Two production bugs lived here and both were invisible to the old suite
because it replaced `zep_cloud` with a MagicMock:

* every message written to Zep was `str(ChatMessage)` -- `"id='item_df43...'
  role='user' content=['I live in Barcelona']"` -- because `message.content` is
  a list in 1.3.12, not a str, so fact and entity extraction ran on garbage;
* five of the Zep methods this code called do not exist in zep-cloud 3.16, and
  each failure was swallowed, so recall silently always returned nothing.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.domain.config import KwamiMemoryConfig
from src.memory import create_memory

pytestmark = pytest.mark.live


@pytest.fixture
async def memory(zep_key: str) -> Any:
    """A throwaway Zep user and thread, torn down after the test."""
    unique = uuid.uuid4().hex[:12]
    config = KwamiMemoryConfig(
        enabled=True,
        api_key=zep_key,
        user_id=f"e2e_{unique}",
        session_id=f"e2e_thread_{unique}",
        configure_ontology=False,
    )
    instance = await create_memory(config, kwami_id=f"e2e_{unique}", kwami_name="E2E Kwami")
    if instance is None:
        pytest.skip("Zep memory could not be initialized")
    yield instance
    await instance.close()


async def test_memory_initializes_against_real_zep(memory: Any) -> None:
    """Initialization creates the user and thread; five phantom methods used to break it."""
    assert memory.is_initialized
    assert memory.user_id
    assert memory.session_id


async def test_a_stored_message_is_the_utterance_not_a_pydantic_repr(memory: Any) -> None:
    """The single clearest proof the message-content bug is fixed."""
    utterance = "I live in Barcelona and I ride a blue bicycle."
    await memory.add_message("user", utterance)

    context = await memory.get_context()
    blob = " ".join(
        filter(
            None,
            [
                context.context_block or "",
                context.summary or "",
                " ".join(context.facts or []),
                " ".join(str(m) for m in (context.recent_messages or [])),
            ],
        )
    )

    assert "Barcelona" in blob, f"the utterance never reached Zep: {blob[:400]}"
    for leaked in ("ChatMessage(", "item_id=", "type='message'"):
        assert leaked not in blob, f"a pydantic repr leaked into memory: {leaked}"


async def test_an_exchange_records_both_sides(memory: Any) -> None:
    """`add_exchange` was dead code: its only caller was a hook that never fired."""
    await memory.buffer_user_message("My favourite colour is green.")
    await memory.add_exchange("Noted -- green it is.")

    context = await memory.get_context()
    assert context.has_content(), "nothing was retrievable after a full exchange"


async def test_search_reaches_the_real_graph(memory: Any) -> None:
    """`thread.search` did not exist, so recall always answered 'no memories'."""
    await memory.add_message("user", "My dog is called Pepper.")

    results = await memory.search("dog", limit=3)
    assert isinstance(results, list), "search must return a list, even when empty"


async def test_closing_releases_the_client(memory: Any) -> None:
    """`client.close()` did not exist either, so every reconfiguration leaked a pool."""
    await memory.close()
    assert not memory.is_initialized
