"""UsageTracker is the billing ledger, so its arithmetic is load-bearing.

Every unit here turns into a credit deduction. It is pure in-memory code with
no I/O, which makes it both the easiest module to cover properly and the one
where an uncovered branch costs real money.
"""

from __future__ import annotations

import pytest

from src.domain.usage import UsageTracker


class Metrics:
    """Duck-types a LiveKit metrics event; only attributes are read."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class Metadata:
    def __init__(self, model_provider: str | None = None, model_name: str | None = None) -> None:
        self.model_provider = model_provider
        self.model_name = model_name


@pytest.fixture
def tracker() -> UsageTracker:
    return UsageTracker()


def only(tracker: UsageTracker) -> dict:
    summary = tracker.get_usage_summary()
    assert len(summary) == 1, f"expected one entry, got {summary}"
    return summary[0]


# -- model identification ----------------------------------------------------


def test_provider_and_name_combine_into_the_model_id(tracker: UsageTracker) -> None:
    tracker.on_llm_metrics(Metrics(total_tokens=10, metadata=Metadata("openai", "gpt-4o-mini")))
    assert only(tracker)["model_id"] == "openai/gpt-4o-mini"


def test_a_name_without_a_provider_is_used_alone(tracker: UsageTracker) -> None:
    tracker.on_llm_metrics(Metrics(total_tokens=10, metadata=Metadata(None, "gpt-4o-mini")))
    assert only(tracker)["model_id"] == "gpt-4o-mini"


def test_the_label_is_the_fallback(tracker: UsageTracker) -> None:
    tracker.on_llm_metrics(Metrics(total_tokens=10, label="some-llm"))
    assert only(tracker)["model_id"] == "some-llm"


def test_an_unidentifiable_model_is_still_billed(tracker: UsageTracker) -> None:
    """Dropping the event instead would silently lose revenue."""
    tracker.on_llm_metrics(Metrics(total_tokens=10))
    assert only(tracker)["model_id"] == "unknown"


# -- LLM ---------------------------------------------------------------------


def test_llm_tokens_accumulate_across_events(tracker: UsageTracker) -> None:
    for _ in range(3):
        tracker.on_llm_metrics(Metrics(total_tokens=100, label="m"))

    entry = only(tracker)
    assert entry["units_used"] == 300
    assert entry["event_count"] == 3


def test_llm_total_is_derived_when_absent(tracker: UsageTracker) -> None:
    tracker.on_llm_metrics(Metrics(prompt_tokens=30, completion_tokens=70, label="m"))

    entry = only(tracker)
    assert entry["units_used"] == 100
    assert entry["prompt_tokens"] == 30
    assert entry["completion_tokens"] == 70


def test_alternative_token_attribute_names_are_accepted(tracker: UsageTracker) -> None:
    """Providers disagree on naming; billing must not depend on which one."""
    tracker.on_llm_metrics(Metrics(input_tokens=10, output_tokens=5, label="m"))
    assert only(tracker)["units_used"] == 15


def test_cached_input_tokens_are_tracked_separately(tracker: UsageTracker) -> None:
    tracker.on_llm_metrics(Metrics(total_tokens=100, cached_tokens=40, label="m"))
    assert only(tracker)["cached_input_tokens"] == 40


def test_a_zero_token_event_is_not_recorded(tracker: UsageTracker) -> None:
    tracker.on_llm_metrics(Metrics(total_tokens=0, label="m"))
    assert tracker.get_usage_summary() == []


def test_unparseable_token_values_do_not_raise(tracker: UsageTracker) -> None:
    """These come off a provider SDK; a bad value must not kill the session."""
    tracker.on_llm_metrics(Metrics(total_tokens="many", label="m"))
    assert tracker.get_usage_summary() == []


# -- STT / TTS ---------------------------------------------------------------


def test_stt_duration_is_converted_to_minutes(tracker: UsageTracker) -> None:
    tracker.on_stt_metrics(Metrics(audio_duration=90.0, label="dg"))

    entry = only(tracker)
    assert entry["units_used"] == pytest.approx(1.5)
    assert entry["audio_input_minutes"] == pytest.approx(1.5)


def test_zero_duration_stt_is_ignored(tracker: UsageTracker) -> None:
    tracker.on_stt_metrics(Metrics(audio_duration=0, label="dg"))
    assert tracker.get_usage_summary() == []


def test_tts_bills_characters(tracker: UsageTracker) -> None:
    tracker.on_tts_metrics(Metrics(characters_count=250, label="tts"))
    assert only(tracker)["units_used"] == 250


def test_zero_character_tts_is_ignored(tracker: UsageTracker) -> None:
    tracker.on_tts_metrics(Metrics(characters_count=0, label="tts"))
    assert tracker.get_usage_summary() == []


# -- external services -------------------------------------------------------


def test_external_usage_is_recorded_with_request_counts(tracker: UsageTracker) -> None:
    tracker.record_external_usage("search", "tavily/search", units_used=1.0)
    tracker.record_external_usage("search", "tavily/search", units_used=1.0)

    entry = only(tracker)
    assert entry["units_used"] == 2.0
    assert entry["request_count"] == 2


def test_a_no_op_external_record_is_skipped(tracker: UsageTracker) -> None:
    tracker.record_external_usage("search", "x", units_used=0, request_count=0)
    assert tracker.get_usage_summary() == []


def test_a_request_only_record_still_reaches_the_summary(tracker: UsageTracker) -> None:
    """Some providers bill per request, not per unit.

    `record_external_usage` accepted these, but `get_usage_summary` filtered on
    `total_units > 0`, so they were stored and then silently dropped before the
    credits API ever saw them.
    """
    tracker.record_external_usage("search", "x", units_used=0, request_count=3)

    entry = only(tracker)
    assert entry["request_count"] == 3
    assert entry["units_used"] == 0
    assert tracker.has_usage is True


# -- summary and aggregation -------------------------------------------------


def test_different_models_are_billed_separately(tracker: UsageTracker) -> None:
    tracker.on_llm_metrics(Metrics(total_tokens=10, label="a"))
    tracker.on_llm_metrics(Metrics(total_tokens=20, label="b"))
    tracker.on_tts_metrics(Metrics(characters_count=5, label="a"))

    assert len(tracker.get_usage_summary()) == 3


def test_optional_fields_are_omitted_when_empty(tracker: UsageTracker) -> None:
    """The payload goes to the credits API; empty keys are noise."""
    tracker.on_tts_metrics(Metrics(characters_count=10, label="tts"))

    entry = only(tracker)
    assert "prompt_tokens" not in entry
    assert "audio_input_minutes" not in entry


def test_units_are_rounded_for_transport(tracker: UsageTracker) -> None:
    tracker.on_stt_metrics(Metrics(audio_duration=1.0 / 3.0, label="dg"))
    assert only(tracker)["units_used"] == round(1.0 / 180.0, 6)


def test_has_usage_reflects_whether_anything_is_billable(tracker: UsageTracker) -> None:
    assert tracker.has_usage is False

    tracker.on_llm_metrics(Metrics(total_tokens=0, label="m"))
    assert tracker.has_usage is False, "a zero-token event is not billable usage"

    tracker.on_llm_metrics(Metrics(total_tokens=1, label="m"))
    assert tracker.has_usage is True


def test_session_duration_advances(tracker: UsageTracker) -> None:
    assert tracker.session_duration_seconds >= 0.0


# -- realtime ----------------------------------------------------------------


def test_realtime_duration_is_billed_in_minutes(tracker: UsageTracker) -> None:
    """The credits API prices realtime per minute, not per second."""
    tracker.on_realtime_metrics(Metrics(duration=120.0, label="rt"))
    assert only(tracker)["units_used"] == pytest.approx(2.0)


def test_realtime_falls_back_to_audio_minutes_without_a_duration(tracker: UsageTracker) -> None:
    tracker.on_realtime_metrics(
        Metrics(duration=0, audio_input_minutes=1.5, audio_output_minutes=0.5, label="rt")
    )

    entry = only(tracker)
    assert entry["units_used"] == pytest.approx(2.0)
    assert entry["audio_input_minutes"] == pytest.approx(1.5)
    assert entry["audio_output_minutes"] == pytest.approx(0.5)


def test_realtime_accepts_alternative_audio_attribute_names(tracker: UsageTracker) -> None:
    tracker.on_realtime_metrics(
        Metrics(duration=0, input_audio_minutes=1.0, output_audio_minutes=2.0, label="rt")
    )
    assert only(tracker)["units_used"] == pytest.approx(3.0)


def test_realtime_text_tokens_are_reported(tracker: UsageTracker) -> None:
    tracker.on_realtime_metrics(
        Metrics(duration=60.0, text_input_tokens=100, text_output_tokens=40, label="rt")
    )

    entry = only(tracker)
    assert entry["text_input_tokens"] == 100
    assert entry["text_output_tokens"] == 40


def test_a_realtime_event_with_text_but_no_audio_is_still_billed(tracker: UsageTracker) -> None:
    """Text-only turns are real; discarding them loses revenue."""
    tracker.on_realtime_metrics(Metrics(duration=0, input_tokens=50, output_tokens=10, label="rt"))
    assert tracker.has_usage is True


def test_an_entirely_empty_realtime_event_is_ignored(tracker: UsageTracker) -> None:
    tracker.on_realtime_metrics(Metrics(duration=0, label="rt"))
    assert tracker.get_usage_summary() == []


def test_unparseable_float_metrics_do_not_raise(tracker: UsageTracker) -> None:
    tracker.on_stt_metrics(Metrics(audio_duration="a while", label="dg"))
    assert tracker.get_usage_summary() == []


def test_realtime_summary_includes_every_populated_field(tracker: UsageTracker) -> None:
    tracker.on_realtime_metrics(
        Metrics(
            duration=60.0,
            audio_input_minutes=0.5,
            audio_output_minutes=0.25,
            text_input_tokens=10,
            text_output_tokens=5,
            label="rt",
        )
    )

    entry = only(tracker)
    for key in (
        "audio_input_minutes",
        "audio_output_minutes",
        "text_input_tokens",
        "text_output_tokens",
    ):
        assert key in entry
