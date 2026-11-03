"""The three voice pipelines, and which models each one can be built from.

* ``standard``: STT -> LLM -> TTS, every stage its own provider.
* ``realtime``: one speech-to-speech model hears, thinks and speaks with its
  own voice (Gemini Live, OpenAI Realtime).
* ``hybrid``: the realtime model hears and thinks, but answers in text and a
  TTS speaks for it -- Live's turn-taking with any voice, a Gemini TTS
  direction or an ElevenLabs voice included.

Hybrid only works with a realtime model that can answer in text. Gemini's
native-audio Live models answer only in audio, so a "Gemini Live + Gemini TTS"
session would open a socket the model rejects and leave the user with a silent
agent. `resolve_pipeline` turns such a combination into the nearest one that
works, and says why, instead of building it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..utils.provider import strip_model_prefix

STANDARD_PIPELINE = "standard"
REALTIME_PIPELINE = "realtime"
HYBRID_PIPELINE = "hybrid"

PIPELINES = (STANDARD_PIPELINE, REALTIME_PIPELINE, HYBRID_PIPELINE)

#: Pipelines whose model is a realtime one: an `llm` update from the panel
#: names the realtime model there, and a dead realtime socket is rescued.
REALTIME_MODEL_PIPELINES = frozenset({REALTIME_PIPELINE, HYBRID_PIPELINE})

#: Pipelines that speak through a TTS (the tts_* settings and its direction).
TTS_PIPELINES = frozenset({STANDARD_PIPELINE, HYBRID_PIPELINE})

#: Realtime models that can answer in text, by name prefix (after the provider
#: prefix is stripped). Measured with `scripts/voice_lab/bench.py --kinds
#: realtime`, which asks every Live model for a text answer. On 2026-10-22
#: gemini-3.1-flash-live-preview, gemini-3.8-live and the 2.5 native-audio
#: model all closed with 1007 ("response modalities (TEXT) is not supported"),
#: while gpt-realtime and gpt-realtime-mini answered in text in 0.3-0.7 s.
TEXT_REALTIME_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-realtime", "gpt-4o-realtime", "gpt-4o-mini-realtime"),
    "google": (),
}


def realtime_speaks_text(provider: str | None, model: str | None) -> bool:
    """Whether this realtime model can answer in text, for a TTS to speak."""
    if not provider or not model:
        return False
    key = provider.lower()
    if key == "gemini":
        key = "google"
    name = strip_model_prefix(model, key).lower()
    return any(name.startswith(prefix) for prefix in TEXT_REALTIME_PREFIXES.get(key, ()))


@dataclass(frozen=True)
class PipelineChoice:
    """The pipeline that will be built, and why it differs from the request."""

    pipeline: str
    reason: str = ""

    @property
    def adjusted(self) -> bool:
        return bool(self.reason)


def resolve_pipeline(pipeline: str, realtime_provider: str, realtime_model: str) -> PipelineChoice:
    """The pipeline to build for a request, corrected when it cannot work.

    A hybrid request on a model that only answers in audio becomes the
    realtime pipeline: the Live model the user picked keeps hearing and
    speaking, and only the TTS is left out.
    """
    if pipeline != HYBRID_PIPELINE:
        return PipelineChoice(pipeline)
    if realtime_speaks_text(realtime_provider, realtime_model):
        return PipelineChoice(HYBRID_PIPELINE)
    return PipelineChoice(
        REALTIME_PIPELINE,
        f"{realtime_provider}/{realtime_model} answers only in audio, so it cannot "
        "speak through a TTS; it speaks with its own voice",
    )
