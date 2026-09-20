"""Tracing export: off by default, and never able to take the worker down.

The repo had no tracing of any kind, so "why did that reply take four seconds?"
could only be answered by reading unstructured logs and inferring the gaps. A
voice agent is a latency product — STT, LLM and TTS run in series while the user
sits in silence — so which of the three was slow is the entire question.

`livekit-agents` already instruments its pipeline; only the export side was
missing. What these tests pin is the part that can hurt: this runs once at
startup, before the worker registers, so any failure here must cost tracing and
nothing else.
"""

from __future__ import annotations

import pytest

from src.settings import Settings
from src.telemetry import (
    DEFAULT_SERVICE_NAME,
    build_tracer_provider,
    configure_tracing,
    parse_headers,
)


# -- Headers ------------------------------------------------------------------


def test_headers_parse_into_a_mapping() -> None:
    assert parse_headers("api-key=abc,x-dataset=prod") == {"api-key": "abc", "x-dataset": "prod"}


def test_whitespace_around_pairs_is_tolerated() -> None:
    assert parse_headers(" api-key = abc , b=c ") == {"api-key": "abc", "b": "c"}


def test_an_empty_header_string_yields_nothing() -> None:
    assert parse_headers("") == {}


def test_malformed_pairs_are_skipped_not_raised() -> None:
    """Most hosted backends authenticate with a header, so this is where the API
    key goes. Losing tracing to a typo is acceptable; losing the worker is not."""
    assert parse_headers("no-equals-sign,good=value,=novalue,key=") == {"good": "value"}


def test_a_value_containing_an_equals_sign_survives() -> None:
    """Base64 credentials end in `=`."""
    assert parse_headers("authorization=Basic YWJjOmRlZg==") == {
        "authorization": "Basic YWJjOmRlZg=="
    }


# -- Provider construction -----------------------------------------------------


def test_no_endpoint_means_no_provider() -> None:
    """Returned as None rather than a no-op provider so the caller can tell
    "nobody configured tracing" from "tracing is on"."""
    assert build_tracer_provider(Settings()) is None


def test_a_blank_endpoint_is_treated_as_unset() -> None:
    assert build_tracer_provider(Settings(otel_endpoint="   ")) is None


def test_an_endpoint_produces_a_provider() -> None:
    provider = build_tracer_provider(Settings(otel_endpoint="http://localhost:4318/v1/traces"))

    assert provider is not None
    assert provider.get_tracer("test") is not None


def test_the_service_name_reaches_the_resource() -> None:
    provider = build_tracer_provider(
        Settings(otel_endpoint="http://localhost:4318/v1/traces", otel_service_name="kwami-staging")
    )

    assert provider.resource.attributes["service.name"] == "kwami-staging"


def test_the_service_name_defaults() -> None:
    provider = build_tracer_provider(Settings(otel_endpoint="http://localhost:4318/v1/traces"))

    assert provider.resource.attributes["service.name"] == DEFAULT_SERVICE_NAME


def test_the_environment_reaches_the_resource() -> None:
    """A deployment running staging and production against one collector has to
    be able to tell them apart."""
    provider = build_tracer_provider(
        Settings(otel_endpoint="http://localhost:4318/v1/traces", environment="staging")
    )

    assert provider.resource.attributes["deployment.environment"] == "staging"


def test_an_unknown_environment_says_so_rather_than_being_absent() -> None:
    provider = build_tracer_provider(Settings(otel_endpoint="http://localhost:4318/v1/traces"))

    assert provider.resource.attributes["deployment.environment"] == "unknown"


def test_spans_are_exported_in_batches() -> None:
    """A span export on the request path would add network latency to a turn the
    user is listening to."""
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = build_tracer_provider(Settings(otel_endpoint="http://localhost:4318/v1/traces"))

    processors = provider._active_span_processor._span_processors
    assert any(isinstance(p, BatchSpanProcessor) for p in processors)


# -- Startup must never fail because of tracing --------------------------------


def test_configuring_without_an_endpoint_is_a_clean_no_op(caplog) -> None:
    import logging

    with caplog.at_level(logging.INFO):
        assert configure_tracing(Settings()) is False

    assert "tracing is off" in caplog.text


def test_a_broken_provider_does_not_raise(monkeypatch, caplog) -> None:
    """This runs before the worker registers. An exception here would take down
    a worker that was otherwise ready to take calls."""
    import logging

    def explode(_settings: Settings) -> None:
        raise RuntimeError("no exporter package")

    monkeypatch.setattr("src.telemetry.build_tracer_provider", explode)

    with caplog.at_level(logging.WARNING):
        assert configure_tracing(Settings(otel_endpoint="http://localhost:4318")) is False

    assert "Could not build a tracer provider" in caplog.text


def test_a_failing_install_does_not_raise(monkeypatch, caplog) -> None:
    import logging

    from livekit.agents import telemetry

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("provider rejected")

    monkeypatch.setattr(telemetry, "set_tracer_provider", explode)

    with caplog.at_level(logging.WARNING):
        result = configure_tracing(Settings(otel_endpoint="http://localhost:4318/v1/traces"))

    assert result is False
    assert "Could not install the tracer provider" in caplog.text


def test_a_working_configuration_installs_the_provider(monkeypatch, caplog) -> None:
    import logging

    from livekit.agents import telemetry

    installed: list[object] = []
    monkeypatch.setattr(telemetry, "set_tracer_provider", lambda p, **kw: installed.append(p))

    with caplog.at_level(logging.INFO):
        result = configure_tracing(
            Settings(
                otel_endpoint="http://localhost:4318/v1/traces", otel_service_name="kwami-test"
            )
        )

    assert result is True
    assert len(installed) == 1
    assert "Tracing enabled" in caplog.text


def test_settings_are_resolved_when_not_passed(monkeypatch) -> None:
    """The entry point calls this with no arguments."""
    monkeypatch.setattr("src.telemetry.get_settings", Settings)

    assert configure_tracing() is False


@pytest.mark.parametrize(
    "name", ["OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS", "OTEL_SERVICE_NAME"]
)
def test_the_tracing_variables_are_in_the_inventory(name: str) -> None:
    """Otherwise the test suite would not scrub them and the Worker would not
    forward them — the two drifts this repo has already had twice."""
    from src.settings import ENV_VAR_NAMES

    assert name in ENV_VAR_NAMES
