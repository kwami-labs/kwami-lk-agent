"""Telephony bootstrap: a SIP caller sends no `config`, so the agent has to
work out which kwami it is from metadata and fetch the rest over HTTP.

Getting this wrong is not subtle -- the caller talks to the default persona for
the whole call.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
import respx

from src.runtime_bootstrap import (
    _api_timeout_seconds,
    _parse_json_dict,
    fetch_runtime_config,
    resolve_kwami_id,
)


def make_ctx(job_metadata: str | None = None, participants: list | None = None):
    return SimpleNamespace(
        job=SimpleNamespace(metadata=job_metadata),
        room=SimpleNamespace(
            remote_participants={str(i): p for i, p in enumerate(participants or [])}
        ),
    )


def participant(metadata: str | None = None, attributes: dict | None = None):
    return SimpleNamespace(metadata=metadata, attributes=attributes)


# =============================================================================
# _parse_json_dict
# =============================================================================


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty"),
        pytest.param("not json", id="garbage"),
        pytest.param("[1, 2]", id="json but a list"),
        pytest.param('"a string"', id="json but a string"),
        pytest.param("null", id="json null"),
    ],
)
def test_unusable_metadata_parses_to_an_empty_dict(value: str | None) -> None:
    """Metadata is attacker-adjacent and arrives from SIP trunks; nothing here
    may raise."""
    assert _parse_json_dict(value) == {}


def test_a_json_object_parses_through() -> None:
    assert _parse_json_dict('{"kwami_id": "abc"}') == {"kwami_id": "abc"}


# =============================================================================
# _api_timeout_seconds
# =============================================================================


def test_the_timeout_comes_from_settings(env_setting) -> None:
    env_setting("KWAMI_API_TIMEOUT", "7.5")
    assert _api_timeout_seconds() == 7.5


@pytest.mark.parametrize("configured", ["0", "-5", "0.1"])
def test_the_timeout_never_drops_below_a_second(env_setting, configured: str) -> None:
    """A sub-second timeout against a cold API guarantees the default persona."""
    env_setting("KWAMI_API_TIMEOUT", configured)
    assert _api_timeout_seconds() == 1.0


# =============================================================================
# resolve_kwami_id
# =============================================================================


def test_job_metadata_is_preferred() -> None:
    ctx = make_ctx(
        job_metadata='{"kwami_id": "from-job"}',
        participants=[participant(metadata='{"kwami_id": "from-participant"}')],
    )

    assert resolve_kwami_id(ctx) == "from-job"


def test_participant_metadata_is_the_second_source() -> None:
    ctx = make_ctx(participants=[participant(metadata='{"kwami_id": "from-participant"}')])

    assert resolve_kwami_id(ctx) == "from-participant"


def test_participant_attributes_are_the_third_source() -> None:
    """SIP trunks that cannot set metadata can still set attributes."""
    ctx = make_ctx(participants=[participant(attributes={"kwami_id": "from-attrs"})])

    assert resolve_kwami_id(ctx) == "from-attrs"


def test_metadata_wins_over_attributes_on_the_same_participant() -> None:
    ctx = make_ctx(
        participants=[
            participant(metadata='{"kwami_id": "meta"}', attributes={"kwami_id": "attrs"})
        ]
    )

    assert resolve_kwami_id(ctx) == "meta"


def test_participants_are_searched_until_one_answers() -> None:
    ctx = make_ctx(
        participants=[
            participant(metadata="garbage"),
            participant(attributes={}),
            participant(metadata='{"kwami_id": "third"}'),
        ]
    )

    assert resolve_kwami_id(ctx) == "third"


def test_a_non_string_id_is_coerced() -> None:
    """The API takes a string; a JSON number must not travel as an int."""
    ctx = make_ctx(job_metadata='{"kwami_id": 12345}')

    assert resolve_kwami_id(ctx) == "12345"


def test_non_dict_attributes_are_ignored() -> None:
    ctx = make_ctx(participants=[participant(attributes=["not", "a", "dict"])])

    assert resolve_kwami_id(ctx) is None


def test_an_empty_room_resolves_to_none() -> None:
    assert resolve_kwami_id(make_ctx()) is None


def test_an_empty_kwami_id_is_not_accepted() -> None:
    """Falsy-safe: "" would otherwise be resolved and then 404 on every fetch."""
    ctx = make_ctx(job_metadata='{"kwami_id": ""}', participants=[])

    assert resolve_kwami_id(ctx) is None


# =============================================================================
# fetch_runtime_config
# =============================================================================


async def test_no_api_key_disables_the_bootstrap(env_setting, caplog) -> None:
    env_setting("KWAMI_API_KEY", None)

    with caplog.at_level(logging.WARNING):
        assert await fetch_runtime_config("kwami-1") is None

    assert "telephony bootstrap is disabled" in caplog.text


@respx.mock
async def test_a_config_is_fetched_and_returned(env_setting) -> None:
    env_setting("KWAMI_API_KEY", "key-123")
    env_setting("KWAMI_API_URL", "https://api.example.test")
    route = respx.get("https://api.example.test/internal/kwamis/kwami-1/runtime").mock(
        return_value=httpx.Response(200, json={"soul": {"name": "Ada"}})
    )

    assert await fetch_runtime_config("kwami-1") == {"soul": {"name": "Ada"}}
    assert route.call_count == 1


@respx.mock
async def test_the_api_key_travels_in_the_kwami_header(env_setting) -> None:
    env_setting("KWAMI_API_KEY", "key-123")
    env_setting("KWAMI_API_URL", "https://api.example.test")
    route = respx.get("https://api.example.test/internal/kwamis/kwami-1/runtime").mock(
        return_value=httpx.Response(200, json={})
    )

    await fetch_runtime_config("kwami-1")

    assert route.calls[0].request.headers["X-Kwami-API-Key"] == "key-123"


@respx.mock
async def test_a_trailing_slash_on_the_base_url_does_not_double(env_setting) -> None:
    env_setting("KWAMI_API_KEY", "key-123")
    env_setting("KWAMI_API_URL", "https://api.example.test/")
    route = respx.get("https://api.example.test/internal/kwamis/kwami-1/runtime").mock(
        return_value=httpx.Response(200, json={})
    )

    await fetch_runtime_config("kwami-1")

    assert route.call_count == 1


@respx.mock
async def test_a_non_dict_payload_is_rejected(env_setting) -> None:
    """A 200 carrying a list would otherwise be applied as a config."""
    env_setting("KWAMI_API_KEY", "key-123")
    env_setting("KWAMI_API_URL", "https://api.example.test")
    respx.get("https://api.example.test/internal/kwamis/k/runtime").mock(
        return_value=httpx.Response(200, json=["not", "a", "config"])
    )

    assert await fetch_runtime_config("k") is None


@respx.mock
async def test_an_http_error_is_logged_with_the_status_and_body(env_setting, caplog) -> None:
    env_setting("KWAMI_API_KEY", "key-123")
    env_setting("KWAMI_API_URL", "https://api.example.test")
    respx.get("https://api.example.test/internal/kwamis/k/runtime").mock(
        return_value=httpx.Response(404, text="no such kwami")
    )

    with caplog.at_level(logging.WARNING):
        assert await fetch_runtime_config("k") is None

    assert "404" in caplog.text
    assert "no such kwami" in caplog.text


@respx.mock
async def test_a_long_error_body_is_truncated(env_setting, caplog) -> None:
    env_setting("KWAMI_API_KEY", "key-123")
    env_setting("KWAMI_API_URL", "https://api.example.test")
    respx.get("https://api.example.test/internal/kwamis/k/runtime").mock(
        return_value=httpx.Response(500, text="x" * 5000)
    )

    with caplog.at_level(logging.WARNING):
        await fetch_runtime_config("k")

    assert "..." in caplog.text
    assert "x" * 5000 not in caplog.text


@respx.mock
async def test_an_empty_error_body_does_not_break_the_log(env_setting, caplog) -> None:
    env_setting("KWAMI_API_KEY", "key-123")
    env_setting("KWAMI_API_URL", "https://api.example.test")
    respx.get("https://api.example.test/internal/kwamis/k/runtime").mock(
        return_value=httpx.Response(503, text="")
    )

    with caplog.at_level(logging.WARNING):
        assert await fetch_runtime_config("k") is None

    assert "503" in caplog.text


@respx.mock
async def test_an_unreachable_api_explains_the_docker_localhost_trap(env_setting, caplog) -> None:
    """The single most common deploy mistake: KWAMI_API_URL=localhost inside a
    container. The message has to say so, or it reads as an outage."""
    env_setting("KWAMI_API_KEY", "key-123")
    env_setting("KWAMI_API_URL", "http://localhost:8080")
    respx.get("http://localhost:8080/internal/kwamis/k/runtime").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    with caplog.at_level(logging.WARNING):
        assert await fetch_runtime_config("k") is None

    assert "host.docker.internal" in caplog.text
    assert "localhost is the container" in caplog.text
