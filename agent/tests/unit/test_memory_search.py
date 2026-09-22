"""Graph search and the three-strategy user-name extraction.

`search_thread` is what `recall_memories` actually calls, and it answered
"I don't have any memories about that yet" for months because the old code
called a `thread.search` that does not exist. The client is a hand-written
double here; that `graph.search` and `graph.node.get_by_user_id` are real is
pinned in tests/contract/test_zep_contract.py.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.memory.search import (
    _EXCLUDED_NAMES,
    _extract_name_from_fact,
    _is_valid_name,
    get_entities_by_type,
    get_user_name,
    search_thread,
)


def edge(fact: str | None = None, score: float | None = None, **extra: Any):
    return SimpleNamespace(fact=fact, score=score, **extra)


def node(**kwargs: Any):
    kwargs.setdefault("labels", None)
    kwargs.setdefault("name", "")
    kwargs.setdefault("summary", "")
    return SimpleNamespace(**kwargs)


class FakeNodeApi:
    def __init__(self, nodes: Any = None, error: Exception | None = None) -> None:
        self._nodes = nodes
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def get_by_user_id(self, **kwargs: Any):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._nodes


class FakeGraph:
    """`results` may be a single value or a list consumed one call at a time,
    so a multi-strategy path can be driven through its stages."""

    def __init__(
        self,
        results: Any = None,
        error: Exception | None = None,
        nodes: Any = None,
        node_error: Exception | None = None,
    ) -> None:
        self._results = results
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.node = FakeNodeApi(nodes, node_error)

    async def search(self, **kwargs: Any):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        if isinstance(self._results, list):
            return self._results.pop(0) if self._results else None
        return self._results


class FakeZep:
    def __init__(self, **kwargs: Any) -> None:
        self.graph = FakeGraph(**kwargs)


def results(edges=None, nodes=None):
    return SimpleNamespace(edges=edges, nodes=nodes)


# =============================================================================
# search_thread
# =============================================================================


async def test_a_search_without_a_user_id_is_refused(caplog) -> None:
    """The graph is per-user; searching without one would be a cross-user read."""
    client = FakeZep()

    with caplog.at_level(logging.WARNING):
        assert await search_thread(client, "session-1", "query") == []

    assert client.graph.calls == []
    assert "without a user_id" in caplog.text


async def test_facts_come_back_with_their_score_and_session() -> None:
    client = FakeZep(results=results(edges=[edge("likes coffee", 0.9)]))

    found = await search_thread(client, "session-1", "coffee", user_id="user-1")

    assert found == [{"content": "likes coffee", "score": 0.9, "thread_id": "session-1"}]


async def test_the_graph_is_searched_not_the_thread() -> None:
    """The bug this replaced: zep-cloud has no thread.search, so the old call
    raised AttributeError on every recall."""
    client = FakeZep(results=results(edges=[]))

    await search_thread(client, "session-1", "coffee", limit=3, user_id="user-1")

    assert client.graph.calls == [
        {"user_id": "user-1", "query": "coffee", "scope": "edges", "limit": 3}
    ]


async def test_edges_without_a_fact_are_dropped() -> None:
    client = FakeZep(results=results(edges=[edge(None), edge(""), edge("real fact")]))

    found = await search_thread(client, "s", "q", user_id="user-1")

    assert [f["content"] for f in found] == ["real fact"]


async def test_a_missing_score_becomes_zero() -> None:
    client = FakeZep(results=results(edges=[edge("fact", None)]))

    assert (await search_thread(client, "s", "q", user_id="u"))[0]["score"] == 0


async def test_a_response_without_edges_yields_nothing() -> None:
    client = FakeZep(results=results(edges=None))

    assert await search_thread(client, "s", "q", user_id="u") == []


async def test_a_search_failure_is_swallowed(caplog) -> None:
    """Recall failing must not end the turn."""
    client = FakeZep(error=RuntimeError("zep is down"))

    with caplog.at_level(logging.ERROR):
        assert await search_thread(client, "s", "q", user_id="u") == []

    assert "Failed to search memories" in caplog.text


# =============================================================================
# get_entities_by_type
# =============================================================================


async def test_entities_are_filtered_by_label() -> None:
    client = FakeZep(
        nodes=[
            node(name="Ada", labels=["Person"]),
            node(name="Widget", labels=["Product"]),
        ]
    )

    found = await get_entities_by_type(client, "u", "Person")

    assert [e["name"] for e in found] == ["Ada"]
    assert found[0]["type"] == "Person"
    assert found[0]["labels"] == ["Person"]


async def test_label_matching_is_case_insensitive() -> None:
    client = FakeZep(nodes=[node(name="Ada", labels=["person"])])

    assert len(await get_entities_by_type(client, "u", "PERSON")) == 1


async def test_more_nodes_are_fetched_than_requested_to_survive_filtering() -> None:
    client = FakeZep(nodes=[])

    await get_entities_by_type(client, "u", "Person", limit=5)

    assert client.graph.node.calls == [{"user_id": "u", "limit": 10}]


async def test_the_limit_stops_collection() -> None:
    client = FakeZep(nodes=[node(name=f"P{i}", labels=["Person"]) for i in range(10)])

    assert len(await get_entities_by_type(client, "u", "Person", limit=3)) == 3


async def test_a_node_without_labels_matches_nothing() -> None:
    client = FakeZep(nodes=[node(name="Mystery", labels=None)])

    assert await get_entities_by_type(client, "u", "Person") == []


async def test_created_at_is_stringified_when_present() -> None:
    client = FakeZep(nodes=[node(name="Ada", labels=["Person"], created_at="2026-01-01")])

    assert (await get_entities_by_type(client, "u", "Person"))[0]["created_at"] == "2026-01-01"


async def test_a_missing_created_at_is_none() -> None:
    client = FakeZep(nodes=[node(name="Ada", labels=["Person"], created_at=None)])

    assert (await get_entities_by_type(client, "u", "Person"))[0]["created_at"] is None


async def test_uuid_underscore_is_preferred_over_uuid() -> None:
    """zep-cloud returns `uuid_`; falling back to `uuid` keeps older shapes working."""
    client = FakeZep(nodes=[node(name="Ada", labels=["Person"], uuid_="A", uuid="B")])

    assert (await get_entities_by_type(client, "u", "Person"))[0]["uuid"] == "A"


async def test_an_empty_node_response_yields_nothing() -> None:
    assert await get_entities_by_type(FakeZep(nodes=None), "u", "Person") == []


async def test_a_node_lookup_failure_is_swallowed(caplog) -> None:
    client = FakeZep(node_error=RuntimeError("boom"))

    with caplog.at_level(logging.DEBUG):
        assert await get_entities_by_type(client, "u", "Person") == []

    assert "Failed to get entities by type" in caplog.text


# =============================================================================
# _is_valid_name
# =============================================================================


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("", id="empty"),
        pytest.param("A", id="single char"),
        pytest.param("ada", id="lowercase"),
        pytest.param("Ada2", id="not alpha"),
        pytest.param("Ada Lovelace", id="has a space"),
    ],
)
def test_implausible_names_are_rejected(name: str) -> None:
    assert _is_valid_name(name) is False


@pytest.mark.parametrize("excluded", sorted(_EXCLUDED_NAMES)[:8])
def test_excluded_words_are_rejected_whatever_their_case(excluded: str) -> None:
    assert _is_valid_name(excluded.capitalize()) is False


def test_a_plausible_name_is_accepted() -> None:
    assert _is_valid_name("Ada") is True


def test_extra_exclusions_are_honoured() -> None:
    """The assistant's own name must never be adopted as the user's."""
    assert _is_valid_name("Kwami", extra_excluded={"kwami"}) is False


# =============================================================================
# _extract_name_from_fact
# =============================================================================


@pytest.mark.parametrize(
    "fact",
    [
        "The user's name is Ada",
        "my name is Ada",
        "called Ada",
        "goes by Ada",
        "identified as Ada",
        "introduced themselves as Ada",
        "Ada is the user",
        "the user is Ada",
    ],
)
def test_every_name_pattern_extracts(fact: str) -> None:
    assert _extract_name_from_fact(fact) == "Ada"


def test_an_empty_fact_extracts_nothing() -> None:
    assert _extract_name_from_fact("") is None


def test_an_unrelated_fact_extracts_nothing() -> None:
    assert _extract_name_from_fact("likes coffee in the morning") is None


def test_an_extracted_word_still_has_to_be_a_plausible_name() -> None:
    """The patterns are case-insensitive, so "name is the" matches; validity is
    the second gate."""
    assert _extract_name_from_fact("name is the") is None


def test_the_assistant_name_is_never_extracted() -> None:
    assert _extract_name_from_fact("my name is Kwami", {"kwami"}) is None


# =============================================================================
# get_user_name
# =============================================================================


async def test_strategy_one_finds_a_name_in_an_explicit_fact(caplog) -> None:
    client = FakeZep(results=results(edges=[edge("my name is Ada")]))

    with caplog.at_level(logging.INFO):
        assert await get_user_name(client, "u") == "Ada"

    assert "from fact" in caplog.text


async def test_strategy_one_failing_does_not_stop_the_search(caplog) -> None:
    """Each strategy is individually wrapped, so one Zep hiccup does not cost
    the other two."""
    client = FakeZep(error=RuntimeError("search down"))

    with caplog.at_level(logging.DEBUG):
        assert await get_user_name(client, "u") is None

    assert "Graph search failed for query" in caplog.text


async def test_strategy_two_counts_name_verb_patterns(caplog) -> None:
    client = FakeZep(
        results=[
            results(edges=[]),  # strategy 1, query 1
            results(edges=[]),  # strategy 1, query 2
            results(edges=[]),  # strategy 1, query 3
            results(
                edges=[
                    edge("Bob likes coffee"),
                    edge("Ada wants a holiday"),
                    edge("Ada prefers tea"),
                ]
            ),
        ]
    )

    with caplog.at_level(logging.INFO):
        assert await get_user_name(client, "u") == "Ada"

    assert "from patterns" in caplog.text
    assert "appeared 2 times" in caplog.text


async def test_strategy_three_reads_a_person_node(caplog) -> None:
    client = FakeZep(
        results=results(edges=[]),
        nodes=[node(label="Ada", type="Person")],
    )

    with caplog.at_level(logging.INFO):
        assert await get_user_name(client, "u") == "Ada"

    assert "from graph node" in caplog.text


async def test_strategy_three_reads_an_identifying_summary(caplog) -> None:
    client = FakeZep(
        results=results(edges=[]),
        nodes=[node(label="Ada", type="Other", summary="The user is called this")],
    )

    with caplog.at_level(logging.INFO):
        assert await get_user_name(client, "u") == "Ada"

    assert "from node summary" in caplog.text


async def test_a_summary_missing_the_naming_words_is_not_enough() -> None:
    """ "user" alone is not evidence of a name; both halves of the guard matter."""
    client = FakeZep(
        results=results(edges=[]),
        nodes=[node(label="Ada", type="Other", summary="the user drinks coffee")],
    )

    assert await get_user_name(client, "u") is None


async def test_strategy_three_failing_returns_none(caplog) -> None:
    client = FakeZep(results=results(edges=[]), node_error=RuntimeError("nodes down"))

    with caplog.at_level(logging.DEBUG):
        assert await get_user_name(client, "u") is None

    assert "Could not get graph nodes" in caplog.text


async def test_nothing_anywhere_returns_none(caplog) -> None:
    client = FakeZep(results=results(edges=[]), nodes=[])

    with caplog.at_level(logging.DEBUG):
        assert await get_user_name(client, "u") is None

    assert "No user name found in memory" in caplog.text


async def test_the_kwami_name_is_excluded_from_every_strategy() -> None:
    """A Kwami called Ada must not decide the user is also called Ada."""
    client = FakeZep(results=results(edges=[edge("my name is Ada")]))

    assert await get_user_name(client, "u", kwami_name="Ada") is None


async def test_a_person_node_with_an_implausible_label_falls_through_to_the_summary() -> None:
    """A Person-typed node whose label is not a usable name must not end the
    search -- the summary check below it is the second chance."""
    client = FakeZep(
        results=results(edges=[]),
        nodes=[
            node(label="the", type="Person", summary="the user is called this"),
            node(label="Ada", type="Person", summary=""),
        ],
    )

    assert await get_user_name(client, "u") == "Ada"


async def test_a_person_node_with_no_usable_label_or_summary_is_skipped() -> None:
    client = FakeZep(
        results=results(edges=[]),
        nodes=[node(label="", type="Person", summary="")],
    )

    assert await get_user_name(client, "u") is None
