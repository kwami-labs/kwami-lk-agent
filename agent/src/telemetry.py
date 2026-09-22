"""Tracing, so a slow turn can be explained rather than guessed at.

The repo had no tracing, no metrics export and no spans of any kind. The only
way to answer "why did that reply take four seconds?" was to read unstructured
logs and infer the gaps between them -- and a voice agent is a latency product:
STT, LLM and TTS run in series while the user sits in silence, and which of the
three was slow is the whole question.

`livekit-agents` already instruments its own pipeline and exposes
`telemetry.set_tracer_provider`, so what is missing is only the export side.
This wires it, and nothing more: no vendor SDK, no hard-coded backend.
Configuration is the standard `OTEL_EXPORTER_OTLP_*` environment, which every
collector and every hosted backend already understands, so choosing one is a
deployment decision rather than a code change.

Off unless an endpoint is configured. A worker that cannot reach a collector
must not spend the user's latency budget retrying span exports, so the absent
case is a clean no-op rather than a degraded one.
"""

from __future__ import annotations

from typing import Any

from .settings import Settings, get_settings
from .utils.logging import get_logger

logger = get_logger("telemetry")

#: Identifies this service in the trace backend. Overridable because a
#: deployment running staging and production against one collector needs to tell
#: them apart.
DEFAULT_SERVICE_NAME = "kwami-lk-agent"


def parse_headers(raw: str) -> dict[str, str]:
    """Parse `OTEL_EXPORTER_OTLP_HEADERS` (`k1=v1,k2=v2`).

    Most hosted backends authenticate with a header, so this is usually where
    the API key goes. Malformed pairs are skipped rather than raising: losing
    tracing is an acceptable outcome of a typo here, losing the worker is not.
    """
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        key, sep, value = pair.partition("=")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if key and value:
            headers[key] = value
    return headers


def build_tracer_provider(settings: Settings) -> Any | None:
    """A tracer provider exporting over OTLP, or None when unconfigured.

    Returns None rather than a no-op provider so the caller can log the
    difference: "tracing is off because nobody configured it" and "tracing is on"
    are both fine, and being unable to tell them apart is not.
    """
    endpoint = settings.otel_endpoint.strip()
    if not endpoint:
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name or DEFAULT_SERVICE_NAME,
            "deployment.environment": settings.environment or "unknown",
        }
    )
    provider = TracerProvider(resource=resource)
    # Batched, never simple: a span export on the request path would add network
    # latency to a turn the user is listening to.
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, headers=parse_headers(settings.otel_headers))
        )
    )
    return provider


def configure_tracing(settings: Settings | None = None) -> bool:
    """Install the tracer provider for livekit-agents. Returns whether it ran.

    Never raises. A misconfigured collector, an unreachable endpoint or a
    missing exporter package must cost tracing, not the session -- this is
    called once at startup and a failure here would take the whole worker down
    before it ever registered.
    """
    settings = settings or get_settings()
    try:
        provider = build_tracer_provider(settings)
    except Exception as e:
        logger.warning("Could not build a tracer provider (%s); tracing is off", e)
        return False

    if provider is None:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT is not set; tracing is off")
        return False

    try:
        from livekit.agents import telemetry

        telemetry.set_tracer_provider(provider)
    except Exception as e:
        logger.warning("Could not install the tracer provider (%s); tracing is off", e)
        return False

    logger.info(
        "Tracing enabled: exporting to %s as %s",
        settings.otel_endpoint,
        settings.otel_service_name or DEFAULT_SERVICE_NAME,
    )
    return True
