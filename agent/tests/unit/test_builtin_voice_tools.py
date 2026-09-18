"""The agent's own voice, speed and language controls.

These are the tools a user reaches by saying "speak slower" or "use a different
voice", and they were the largest untested surface in the repo. What matters
here is not that they call `update_options` but what they do when they *cannot*:

* On the realtime pipeline there is no TTS at all. Both tools used to answer
  "TTS not available", which reads to the model as a transient fault rather
  than a wrong-tool answer, so a realtime user asking for a different voice was
  simply refused. They now name the tool that works -- and that name is a
  contract with `tools/pipeline_control.py`, which redirects to it in guidance.
* Provider quirks are the rule, not the exception: ElevenLabs takes `voice_id`
  rather than `voice` and has no speed at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.constants import CartesiaVoices
from src.domain import KwamiConfig
from src.tools.builtin import AgentToolsMixin

REALTIME_VOICE_TOOL = "change_realtime_voice"


class FakeTTS:
    """A TTS that records what it was asked to change."""

    def __init__(self, provider: str = "cartesia", model: str = "sonic-2") -> None:
        self.provider = provider
        self._model = model
        self.updates: list[dict[str, Any]] = []
        self.fail: Exception | None = None

    def update_options(self, **kwargs: Any) -> None:
        if self.fail:
            raise self.fail
        self.updates.append(kwargs)


class FakeSTT:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update_options(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeSession:
    def __init__(self, tts: Any = None, stt: Any = None) -> None:
        self.tts = tts
        self.stt = stt


class Tools(AgentToolsMixin):
    """The mixin on its own, with only the attributes it documents as required."""

    def __init__(self, *, pipeline_type: str = "standard", session: Any = None) -> None:
        self.kwami_config = KwamiConfig()
        self.kwami_config.voice.pipeline_type = pipeline_type
        self._current_voice_config = self.kwami_config.voice
        self._memory = None
        self.session = session
        self.room = None
        self.usage_tracker = None


def _standard(tts: Any = None, stt: Any = None) -> Tools:
    return Tools(session=FakeSession(tts=tts or FakeTTS(), stt=stt or FakeSTT()))


def _realtime() -> Tools:
    # A realtime agent genuinely has no TTS or STT on the session.
    return Tools(pipeline_type="realtime", session=FakeSession(tts=None, stt=None))


# -- The realtime redirect ---------------------------------------------------


async def test_changing_voice_on_realtime_names_the_tool_that_works() -> None:
    """`pipeline_control.change_realtime_voice` redirects here by name.

    If this message stops naming it, that redirect becomes a dead end and the
    model has no way to recover inside the turn.
    """
    result = await _realtime().change_voice(None, "Cedar")

    assert REALTIME_VOICE_TOOL in result
    assert "not available" not in result.lower()


async def test_changing_speed_on_realtime_explains_rather_than_erroring() -> None:
    result = await _realtime().change_speaking_speed(None, 1.5)

    assert "realtime" in result.lower()
    assert "switch_pipeline_mode" in result


async def test_the_realtime_branch_is_chosen_from_config_not_from_the_session() -> None:
    """During a reconfiguration the agent is rebuilt before the session swaps.

    Probing `session.tts` would read the *old* session and take the wrong
    branch for the length of that window.
    """
    tools = Tools(pipeline_type="realtime", session=FakeSession(tts=FakeTTS()))

    assert REALTIME_VOICE_TOOL in await tools.change_voice(None, "Cedar")


# -- Voice -------------------------------------------------------------------


async def test_a_known_voice_name_is_resolved_to_its_id() -> None:
    tts = FakeTTS()
    name, expected_id = next(iter(CartesiaVoices.NAME_MAP.items()))

    await _standard(tts).change_voice(None, name)

    assert tts.updates == [{"voice": expected_id}]


async def test_an_unknown_name_is_passed_through_as_an_id() -> None:
    tts = FakeTTS()

    await _standard(tts).change_voice(None, "79a125e8-cd45-4c13-8a67-188112f4dd22")

    assert tts.updates == [{"voice": "79a125e8-cd45-4c13-8a67-188112f4dd22"}]


async def test_elevenlabs_takes_voice_id_rather_than_voice() -> None:
    """The wrong kwarg is accepted silently and changes nothing."""
    tts = FakeTTS(provider="elevenlabs", model="eleven_flash_v2_5")

    await _standard(tts).change_voice(None, "JBFqnCBsd6RMkjVDRZzb")

    assert tts.updates == [{"voice_id": "JBFqnCBsd6RMkjVDRZzb"}]


async def test_changing_voice_without_a_session_is_reported_not_raised() -> None:
    tools = Tools(session=None)
    assert "session not available" in await tools.change_voice(None, "nova")


async def test_changing_voice_without_tts_is_reported_not_raised() -> None:
    tools = Tools(session=FakeSession(tts=None))
    assert "TTS not available" in await tools.change_voice(None, "nova")


async def test_a_provider_that_rejects_the_voice_is_reported() -> None:
    tts = FakeTTS()
    tts.fail = RuntimeError("no such voice")

    result = await _standard(tts).change_voice(None, "nova")

    assert "couldn't change the voice" in result


# -- Speed -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0.1, 0.5), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (9.0, 2.0)],
)
async def test_speed_is_clamped_to_what_providers_accept(requested: float, expected: float) -> None:
    tts = FakeTTS()

    await _standard(tts).change_speaking_speed(None, requested)

    assert tts.updates == [{"speed": expected}]


async def test_elevenlabs_has_no_speed_and_says_so() -> None:
    tts = FakeTTS(provider="elevenlabs", model="eleven_flash_v2_5")

    result = await _standard(tts).change_speaking_speed(None, 1.5)

    assert "not supported" in result
    assert tts.updates == [], "sent an option the provider does not have"


@pytest.mark.parametrize(
    ("speed", "phrase"),
    [(0.6, "more slowly"), (1.6, "faster"), (1.0, "normal pace")],
)
async def test_the_spoken_confirmation_matches_the_change(speed: float, phrase: str) -> None:
    assert phrase in await _standard().change_speaking_speed(None, speed)


async def test_changing_speed_without_a_session_is_reported_not_raised() -> None:
    assert "session not available" in await Tools(session=None).change_speaking_speed(None, 1.2)


# -- Language ----------------------------------------------------------------


async def test_changing_language_updates_both_ends_of_the_pipeline() -> None:
    tts, stt = FakeTTS(), FakeSTT()

    await _standard(tts, stt).change_language(None, "ES")

    assert stt.updates == [{"language": "es"}]
    assert tts.updates == [{"language": "es"}]


async def test_a_tts_without_language_support_does_not_break_the_change() -> None:
    """Not every provider takes `language`; STT still has to be retuned."""
    tts, stt = FakeTTS(), FakeSTT()
    tts.fail = TypeError("unexpected keyword argument 'language'")

    result = await _standard(tts, stt).change_language(None, "fr")

    assert stt.updates == [{"language": "fr"}]
    assert "couldn't" not in result


async def test_a_known_language_is_confirmed_in_that_language() -> None:
    from src.constants import LANGUAGE_GREETINGS

    result = await _standard().change_language(None, "es")

    assert result == LANGUAGE_GREETINGS["es"]


async def test_an_unknown_language_still_reports_the_change() -> None:
    assert "cy" in await _standard().change_language(None, "cy")


async def test_language_is_noted_even_with_no_session_to_apply_it_to() -> None:
    assert "de" in await Tools(session=None).change_language(None, "de")


# -- Reading current settings ------------------------------------------------


async def test_current_voice_settings_report_the_whole_pipeline() -> None:
    settings = await _standard().get_current_voice_settings(None)

    for key in (
        "tts_provider",
        "tts_model",
        "tts_voice",
        "tts_speed",
        "stt_provider",
        "stt_model",
        "stt_language",
        "llm_provider",
        "llm_model",
        "llm_temperature",
    ):
        assert key in settings


async def test_kwami_info_keeps_the_legacy_persona_key() -> None:
    """Older clients read `persona`; dropping it breaks them silently."""
    info = await _standard().get_kwami_info(None)

    assert info["soul"]["name"] == info["persona"]["name"]


async def test_get_current_time_answers_in_words() -> None:
    answer = await _standard().get_current_time(None)
    assert ":" in answer and "on" in answer
