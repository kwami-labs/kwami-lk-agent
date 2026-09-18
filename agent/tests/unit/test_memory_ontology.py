"""The ontology is what makes Zep extract *useful* facts rather than a pile of
untyped nodes.

The models are built against the real zep_cloud ontology classes -- a fake
would defeat the point, since the whole job of this module is to produce
objects the SDK accepts. `configure_ontology` takes its client as an argument,
so it gets a hand-written double; that `graph.set_ontology` exists on the real
client is pinned separately in tests/contract/test_zep_contract.py.
"""

from __future__ import annotations

import builtins
import logging
from typing import Any

import pytest

from src.memory.ontology import (
    DEFAULT_EDGE_TYPES,
    DEFAULT_ENTITY_TYPES,
    _build_edge_models,
    _build_entity_models,
    configure_ontology,
)


class FakeGraph:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._error = error

    async def set_ontology(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error


class FakeZep:
    def __init__(self, error: Exception | None = None) -> None:
        self.graph = FakeGraph(error)


def refuse_imports(monkeypatch: pytest.MonkeyPatch, *prefixes: str) -> None:
    """Make the lazy SDK imports fail, to reach the degradation branches."""
    real_import = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith(prefixes):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", guarded)


# =============================================================================
# The shipped definitions
# =============================================================================


def test_every_default_entity_type_has_a_name_and_description() -> None:
    for entity in DEFAULT_ENTITY_TYPES:
        assert entity["name"]
        assert entity["description"]


def test_every_default_edge_type_declares_a_source() -> None:
    """Edges without a source constraint are how orphan nodes get created --
    the defect this module's docstring says it exists to prevent."""
    for edge in DEFAULT_EDGE_TYPES:
        assert edge.get("source"), f"{edge['name']} has no source"


def test_the_defaults_do_not_redefine_zep_builtins() -> None:
    """Zep already ships User, Assistant, Preference, Location, Event, Object,
    Topic, Organization and Document. Redefining one silently replaces it."""
    builtin_types = {
        "User",
        "Assistant",
        "Preference",
        "Location",
        "Event",
        "Object",
        "Topic",
        "Organization",
        "Document",
    }
    names = {e["name"] for e in DEFAULT_ENTITY_TYPES}

    assert not (names & builtin_types)


# =============================================================================
# _build_entity_models
# =============================================================================


def test_each_entity_definition_becomes_a_named_model() -> None:
    models = _build_entity_models(DEFAULT_ENTITY_TYPES)

    assert set(models) == {e["name"] for e in DEFAULT_ENTITY_TYPES}
    assert models["Person"].__name__ == "Person"


def test_the_description_becomes_the_docstring_zep_reads() -> None:
    models = _build_entity_models([{"name": "Thing", "description": "a thing"}])

    assert models["Thing"].__doc__ == "a thing"


def test_declared_fields_become_model_annotations() -> None:
    models = _build_entity_models(
        [{"name": "Thing", "description": "d", "fields": {"colour": "its colour"}}]
    )

    assert "colour" in models["Thing"].__annotations__


def test_a_fieldless_entity_gets_a_generic_detail_field() -> None:
    """Zep requires at least one custom property per type; a bare model is
    rejected at set_ontology time."""
    models = _build_entity_models([{"name": "Thing", "description": "d"}])

    assert "detail" in models["Thing"].__annotations__


def test_a_missing_description_falls_back_to_the_name() -> None:
    models = _build_entity_models([{"name": "Thing"}])

    assert models["Thing"].__doc__ == "Thing"


def test_an_empty_definition_list_builds_nothing() -> None:
    assert _build_entity_models([]) == {}


def test_entity_models_degrade_to_empty_without_the_sdk(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    refuse_imports(monkeypatch, "zep_cloud", "pydantic")

    with caplog.at_level(logging.WARNING):
        assert _build_entity_models(DEFAULT_ENTITY_TYPES) == {}

    assert "not available" in caplog.text


# =============================================================================
# _build_edge_models
# =============================================================================


def test_each_edge_definition_becomes_a_model_and_a_constraint() -> None:
    edges = _build_edge_models(DEFAULT_EDGE_TYPES)

    assert set(edges) == {e["name"] for e in DEFAULT_EDGE_TYPES}
    model, constraints = edges["KNOWS"]
    assert model.__name__ == "KNOWS"
    assert len(constraints) == 1


def test_the_source_and_target_reach_the_constraint() -> None:
    edges = _build_edge_models(
        [{"name": "KNOWS", "description": "d", "source": "User", "target": "Person"}]
    )
    _, (constraint,) = edges["KNOWS"]

    assert constraint.source == "User"
    assert constraint.target == "Person"


def test_an_edge_without_a_target_is_unconstrained_on_that_side() -> None:
    edges = _build_edge_models([{"name": "REL", "description": "d", "source": "User"}])
    _, (constraint,) = edges["REL"]

    assert constraint.source == "User"
    assert constraint.target is None


def test_an_edge_without_a_source_defaults_to_user() -> None:
    edges = _build_edge_models([{"name": "REL", "description": "d"}])
    _, (constraint,) = edges["REL"]

    assert constraint.source == "User"


def test_a_fieldless_edge_also_gets_a_detail_field() -> None:
    edges = _build_edge_models([{"name": "REL", "description": "d"}])
    model, _ = edges["REL"]

    assert "detail" in model.__annotations__


def test_declared_edge_fields_become_annotations() -> None:
    edges = _build_edge_models(
        [{"name": "REL", "description": "d", "fields": {"context": "how"}}]
    )
    model, _ = edges["REL"]

    assert "context" in model.__annotations__


def test_an_edge_without_a_description_falls_back_to_its_name() -> None:
    edges = _build_edge_models([{"name": "REL"}])
    model, _ = edges["REL"]

    assert model.__doc__ == "REL"


def test_an_empty_edge_list_builds_nothing() -> None:
    assert _build_edge_models([]) == {}


def test_edge_models_degrade_to_empty_without_the_sdk(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    refuse_imports(monkeypatch, "zep_cloud", "pydantic")

    with caplog.at_level(logging.WARNING):
        assert _build_edge_models(DEFAULT_EDGE_TYPES) == {}

    assert "not available" in caplog.text


# =============================================================================
# configure_ontology
# =============================================================================


async def test_the_defaults_are_applied_for_the_given_user() -> None:
    client = FakeZep()

    assert await configure_ontology(client, "user-1") is True

    call = client.graph.calls[0]
    assert call["user_ids"] == ["user-1"]
    assert set(call["entities"]) == {e["name"] for e in DEFAULT_ENTITY_TYPES}
    assert set(call["edges"]) == {e["name"] for e in DEFAULT_EDGE_TYPES}


async def test_custom_types_override_the_defaults() -> None:
    client = FakeZep()

    await configure_ontology(
        client,
        "user-1",
        entity_types=[{"name": "Custom", "description": "d"}],
        edge_types=[{"name": "CUSTOM_EDGE", "description": "d", "source": "User"}],
    )

    call = client.graph.calls[0]
    assert set(call["entities"]) == {"Custom"}
    assert set(call["edges"]) == {"CUSTOM_EDGE"}


async def test_nothing_is_sent_when_no_models_could_be_built(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    refuse_imports(monkeypatch, "zep_cloud", "pydantic")
    client = FakeZep()

    with caplog.at_level(logging.WARNING):
        assert await configure_ontology(client, "user-1") is False

    assert client.graph.calls == []
    assert "No ontology models could be built" in caplog.text


async def test_a_plan_that_rejects_ontologies_is_not_fatal(caplog) -> None:
    """Ontology is a paid Zep feature. A session must still run without it."""
    client = FakeZep(error=RuntimeError("not available on your plan"))

    with caplog.at_level(logging.WARNING):
        assert await configure_ontology(client, "user-1") is False

    assert "may not be supported on your plan" in caplog.text


async def test_an_import_error_from_the_sdk_is_not_fatal(caplog) -> None:
    client = FakeZep(error=ImportError("ontology module missing"))

    with caplog.at_level(logging.WARNING):
        assert await configure_ontology(client, "user-1") is False

    assert "skipping ontology configuration" in caplog.text


async def test_a_successful_configuration_is_logged_with_the_counts(caplog) -> None:
    client = FakeZep()

    with caplog.at_level(logging.INFO):
        await configure_ontology(client, "user-1")

    assert "Configured ontology for user-1" in caplog.text
    assert f"{len(DEFAULT_ENTITY_TYPES)} entity types" in caplog.text
