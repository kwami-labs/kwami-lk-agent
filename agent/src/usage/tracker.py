"""Usage tracker for monitoring AI resource consumption during a session.

Tracks:
- LLM token usage (prompt, completion, and cached input tokens)
- STT audio duration (minutes)
- TTS character count
- Realtime model usage (duration in minutes for pricing)
- External paid tool and memory operations (request-based)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ..utils.logging import get_logger

logger = get_logger("usage.tracker")


@dataclass
class ModelUsage:
    """Accumulated usage for a single model."""

    model_type: str  # 'llm', 'stt', 'tts', 'realtime', 'tool', 'memory'
    model_id: str
    total_units: float = 0.0  # tokens for LLM, minutes for STT/realtime, chars for TTS
    event_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_input_tokens: int = 0
    audio_input_minutes: float = 0.0
    audio_output_minutes: float = 0.0
    text_input_tokens: int = 0
    text_output_tokens: int = 0
    request_count: int = 0


def _get_model_id(metrics: Any) -> str:
    """Extract a model identifier from metrics metadata or label."""
    metadata = getattr(metrics, "metadata", None)
    if metadata:
        provider = getattr(metadata, "model_provider", None) or ""
        name = getattr(metadata, "model_name", None) or ""
        if provider and name:
            return f"{provider}/{name}"
        if name:
            return name
    return getattr(metrics, "label", None) or "unknown"


def _get_int_metric(metrics: Any, *names: str) -> int:
    """Read the first present integer-like metric attribute."""
    for name in names:
        value = getattr(metrics, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _get_float_metric(metrics: Any, *names: str) -> float:
    """Read the first present float-like metric attribute."""
    for name in names:
        value = getattr(metrics, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


class UsageTracker:
    """Tracks AI resource consumption during a single agent session.

    Thread-safe accumulator for usage metrics from the voice pipeline.
    Call `get_usage_summary()` at session end to get the totals.
    """

    def __init__(self) -> None:
        self._usage: dict[str, ModelUsage] = {}
        self._lock = Lock()
        self._session_start = time.time()

    def _get_or_create(self, model_type: str, model_id: str) -> ModelUsage:
        """Get or create a usage entry for a model."""
        key = f"{model_type}:{model_id}"
        if key not in self._usage:
            self._usage[key] = ModelUsage(model_type=model_type, model_id=model_id)
        return self._usage[key]

    # -----------------------------------------------------------------
    # LLM tracking
    # -----------------------------------------------------------------

    def on_llm_metrics(self, metrics: Any) -> None:
        """Handle a LiveKit LLMMetrics event.

        Attributes used: prompt_tokens, completion_tokens, total_tokens, label, metadata.
        """
        model_id = _get_model_id(metrics)
        total_tokens = _get_int_metric(metrics, "total_tokens")
        prompt_tokens = _get_int_metric(metrics, "prompt_tokens", "input_tokens")
        completion_tokens = _get_int_metric(metrics, "completion_tokens", "output_tokens")
        cached_input_tokens = _get_int_metric(
            metrics,
            "cached_input_tokens",
            "cached_tokens",
            "prompt_cached_tokens",
        )
        tokens = total_tokens or (prompt_tokens + completion_tokens)

        if tokens <= 0:
            return

        with self._lock:
            entry = self._get_or_create("llm", model_id)
            entry.total_units += tokens
            entry.event_count += 1
            entry.prompt_tokens += prompt_tokens
            entry.completion_tokens += completion_tokens
            entry.cached_input_tokens += cached_input_tokens

        logger.info(
            f"LLM usage: {model_id} +{tokens} tokens "
            f"(total: {entry.total_units:.0f})"
        )

    # -----------------------------------------------------------------
    # STT tracking
    # -----------------------------------------------------------------

    def on_stt_metrics(self, metrics: Any) -> None:
        """Handle a LiveKit STTMetrics event.

        Attributes used: audio_duration, label, metadata.
        """
        model_id = _get_model_id(metrics)
        audio_duration = _get_float_metric(metrics, "audio_duration")

        if audio_duration <= 0:
            return

        minutes = audio_duration / 60.0
        with self._lock:
            entry = self._get_or_create("stt", model_id)
            entry.total_units += minutes
            entry.event_count += 1
            entry.audio_input_minutes += minutes

        logger.info(
            f"STT usage: {model_id} +{minutes:.3f} min "
            f"(total: {entry.total_units:.3f} min)"
        )

    # -----------------------------------------------------------------
    # TTS tracking
    # -----------------------------------------------------------------

    def on_tts_metrics(self, metrics: Any) -> None:
        """Handle a LiveKit TTSMetrics event.

        Attributes used: characters_count, label, metadata.
        """
        model_id = _get_model_id(metrics)
        characters = _get_int_metric(metrics, "characters_count")

        if characters <= 0:
            return

        with self._lock:
            entry = self._get_or_create("tts", model_id)
            entry.total_units += characters
            entry.event_count += 1

        logger.info(
            f"TTS usage: {model_id} +{characters} chars "
            f"(total: {entry.total_units:.0f} chars)"
        )

    # -----------------------------------------------------------------
    # Realtime model tracking
    # -----------------------------------------------------------------

    def on_realtime_metrics(self, metrics: Any) -> None:
        """Handle a LiveKit RealtimeModelMetrics event.

        Uses duration (response time in seconds) so units_used is in minutes,
        matching the API's RealtimePricing (per-minute audio).
        """
        model_id = _get_model_id(metrics)
        duration_seconds = _get_float_metric(metrics, "duration")
        audio_input_minutes = _get_float_metric(
            metrics,
            "audio_input_minutes",
            "input_audio_minutes",
        )
        audio_output_minutes = _get_float_metric(
            metrics,
            "audio_output_minutes",
            "output_audio_minutes",
        )
        text_input_tokens = _get_int_metric(
            metrics,
            "text_input_tokens",
            "input_tokens",
        )
        text_output_tokens = _get_int_metric(
            metrics,
            "text_output_tokens",
            "output_tokens",
        )

        if duration_seconds <= 0 and not any(
            (
                audio_input_minutes,
                audio_output_minutes,
                text_input_tokens,
                text_output_tokens,
            )
        ):
            return
        minutes = duration_seconds / 60.0
        normalized_units = minutes or (audio_input_minutes + audio_output_minutes)
        with self._lock:
            entry = self._get_or_create("realtime", model_id)
            entry.total_units += normalized_units
            entry.event_count += 1
            entry.audio_input_minutes += audio_input_minutes
            entry.audio_output_minutes += audio_output_minutes
            entry.text_input_tokens += text_input_tokens
            entry.text_output_tokens += text_output_tokens

        logger.info(
            f"Realtime usage: {model_id} +{normalized_units:.3f} min "
            f"(total: {entry.total_units:.3f} min)"
        )

    # -----------------------------------------------------------------
    # External service tracking
    # -----------------------------------------------------------------

    def record_external_usage(
        self,
        model_type: str,
        model_id: str,
        *,
        units_used: float = 1.0,
        request_count: int = 1,
    ) -> None:
        """Track a paid external operation such as Tavily or Zep."""
        if units_used <= 0 and request_count <= 0:
            return

        with self._lock:
            entry = self._get_or_create(model_type, model_id)
            entry.total_units += units_used
            entry.event_count += 1
            entry.request_count += request_count

        logger.info(
            "External usage: %s %s +%s units",
            model_type,
            model_id,
            units_used,
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def get_usage_summary(self) -> list[dict]:
        """Get the accumulated usage summary.

        Returns a list of dicts, each with:
            model_type, model_id, units_used
        """
        with self._lock:
            items = []
            for entry in self._usage.values():
                if entry.total_units > 0:
                    item = {
                        "model_type": entry.model_type,
                        "model_id": entry.model_id,
                        "units_used": round(entry.total_units, 6),
                        "event_count": entry.event_count,
                    }
                    if entry.prompt_tokens:
                        item["prompt_tokens"] = entry.prompt_tokens
                    if entry.completion_tokens:
                        item["completion_tokens"] = entry.completion_tokens
                    if entry.cached_input_tokens:
                        item["cached_input_tokens"] = entry.cached_input_tokens
                    if entry.audio_input_minutes:
                        item["audio_input_minutes"] = round(entry.audio_input_minutes, 6)
                    if entry.audio_output_minutes:
                        item["audio_output_minutes"] = round(entry.audio_output_minutes, 6)
                    if entry.text_input_tokens:
                        item["text_input_tokens"] = entry.text_input_tokens
                    if entry.text_output_tokens:
                        item["text_output_tokens"] = entry.text_output_tokens
                    if entry.request_count:
                        item["request_count"] = entry.request_count
                    items.append(item)
            return items

    @property
    def session_duration_seconds(self) -> float:
        """Get the elapsed session duration in seconds."""
        return time.time() - self._session_start

    @property
    def has_usage(self) -> bool:
        """Check if any usage has been recorded."""
        with self._lock:
            return any(e.total_units > 0 for e in self._usage.values())
