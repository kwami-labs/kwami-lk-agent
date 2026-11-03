"""A user's own keys: the pipeline runs on them, and that usage is reported as theirs.

The billing half is what makes this safe to offer. Usage on a user's key that went
on to be charged in energy would make them pay twice; usage on ours reported as
theirs would make us pay for it.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from src.domain import KwamiVoiceConfig, UsageTracker
from src.domain.usage import provider_of
from src.factories import llm, realtime, stt, tts
from src.runtime.recovery import install_realtime_rescue

KEY = "sk-the-users-own"


def test_no_log_can_print_a_key() -> None:
    assert KEY not in repr(KwamiVoiceConfig(user_keys={"OPENAI_API_KEY": KEY}))


def test_without_a_key_the_deployment_decides() -> None:
    assert KwamiVoiceConfig().key_for("OPENAI_API_KEY") == {}


def test_the_first_spelling_found_wins() -> None:
    config = KwamiVoiceConfig(user_keys={"B": KEY})
    assert config.key_for("A", "B") == {"api_key": KEY}


@pytest.mark.parametrize(
    ("module", "plugin", "cls", "env", "fields", "build"),
    [
        (llm, "anthropic", "LLM", "ANTHROPIC_API_KEY", {"llm_provider": "anthropic"}, "create_llm"),
        (llm, "groq", "LLM", "GROQ_API_KEY", {"llm_provider": "groq"}, "create_llm"),
        (stt, "deepgram", "STT", "DEEPGRAM_API_KEY", {"stt_provider": "deepgram"}, "create_stt"),
        (stt, "cartesia", "STT", "CARTESIA_API_KEY", {"stt_provider": "cartesia"}, "create_stt"),
        (tts, "openai", "TTS", "OPENAI_API_KEY", {"tts_provider": "openai"}, "create_tts"),
        (tts, "cartesia", "TTS", "CARTESIA_API_KEY", {"tts_provider": "cartesia"}, "create_tts"),
    ],
)
def test_each_stage_is_built_on_the_users_key(
    monkeypatch, module, plugin, cls, env, fields, build
) -> None:
    fake = MagicMock()
    monkeypatch.setattr(module, plugin, fake)

    getattr(module, build)(KwamiVoiceConfig(user_keys={env: KEY}, **fields))

    assert getattr(fake, cls).call_args.kwargs["api_key"] == KEY


def test_elevenlabs_prefers_the_users_key_over_ours(monkeypatch) -> None:
    config = KwamiVoiceConfig(user_keys={"ELEVEN_API_KEY": KEY})
    assert tts.elevenlabs_key(config) == KEY


def test_a_realtime_model_is_built_on_the_users_key(monkeypatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr(realtime, "openai", fake)

    realtime.create_realtime_model(
        KwamiVoiceConfig(realtime_provider="openai", user_keys={"OPENAI_API_KEY": KEY})
    )

    assert fake.realtime.RealtimeModel.call_args.kwargs["api_key"] == KEY


@pytest.mark.parametrize(
    ("model_id", "provider"),
    [
        ("api.openai.com/tts-1", "openai"),
        ("api.anthropic.com/claude-haiku-5-5", "anthropic"),
        ("Gemini/gemini-3.5-flash", "google"),
        ("ElevenLabs/eleven_flash_v2_5", "elevenlabs"),
        ("Deepgram/nova-3", "deepgram"),
        ("livekit/elevenlabs/scribe_v2_realtime", "livekit"),
        ("no-prefix", ""),
    ],
)
def test_the_provider_is_read_from_the_id_as_plugins_spell_it(model_id, provider) -> None:
    assert provider_of(model_id) == provider


class TestReportedAsTheirs:
    def _summary(self, providers: frozenset[str]) -> dict[str, dict]:
        tracker = UsageTracker()
        tracker.mark_own_keys(providers)
        tracker.record_external_usage("tts", "ElevenLabs/eleven_flash_v2_5", units_used=100)
        tracker.record_external_usage("llm", "api.openai.com/gpt-4.1", units_used=100)
        tracker.record_external_usage("tts", "livekit/elevenlabs/flash", units_used=100)
        tracker.record_external_usage("memory", "zep/get_context", units_used=1)
        return {item["model_id"]: item for item in tracker.get_usage_summary()}

    def test_only_the_stage_on_their_key_is_marked(self) -> None:
        items = self._summary(frozenset({"elevenlabs"}))
        assert items["ElevenLabs/eleven_flash_v2_5"]["own_key"] is True
        assert "own_key" not in items["api.openai.com/gpt-4.1"]

    def test_livekit_inference_stays_ours_even_for_the_same_maker(self) -> None:
        items = self._summary(frozenset({"elevenlabs"}))
        assert "own_key" not in items["livekit/elevenlabs/flash"]

    def test_memory_is_never_theirs(self) -> None:
        items = self._summary(frozenset({"zep"}))
        assert "own_key" not in items["zep/get_context"]

    def test_nothing_is_marked_without_keys(self) -> None:
        items = self._summary(frozenset())
        assert not any("own_key" in item for item in items.values())


class TestKeysReadyForTheFirstPipeline:
    @pytest.mark.anyio
    async def test_the_config_waits_for_them_before_building(self) -> None:
        from src.brain.bridge import Bridge

        bridge = Bridge.__new__(Bridge)
        bridge._keys_store = object()
        bridge._voice_keys_loaded = asyncio.Event()
        bridge._voice_keys_loaded.set()
        await bridge.keys_ready(timeout=0.01)

    @pytest.mark.anyio
    async def test_a_slow_load_does_not_hold_the_call(self) -> None:
        from src.brain.bridge import Bridge

        bridge = Bridge.__new__(Bridge)
        bridge._keys_store = object()
        bridge._voice_keys_loaded = asyncio.Event()
        await bridge.keys_ready(timeout=0.01)

    def test_providers_follow_the_keys_loaded(self) -> None:
        from src.brain.bridge import Bridge

        bridge = Bridge.__new__(Bridge)
        bridge.voice_keys = {"DEEPGRAM_API_KEY": KEY, "OPENAI_API_KEY": ""}
        assert bridge.voice_key_providers == frozenset({"deepgram"})


class TestNothingLeaks:
    def test_job_metadata_cannot_set_keys(self) -> None:
        from src.domain import KwamiConfig
        from src.runtime.defaults import apply_voice_fields

        config = KwamiConfig()
        applied = apply_voice_fields(config, {"user_keys": {"OPENAI_API_KEY": KEY}}, "test")

        assert applied == []
        assert config.voice.user_keys == {}

    def test_a_full_config_repr_has_no_key(self) -> None:
        from src.domain import KwamiConfig

        config = KwamiConfig()
        config.voice.user_keys = {"OPENAI_API_KEY": KEY}

        assert KEY not in repr(config)
        assert KEY not in str(config)

    def test_the_status_sent_to_the_app_has_no_key(self, tmp_path, monkeypatch) -> None:
        from src.brain.bridge import Bridge
        from src.settings import Settings

        monkeypatch.chdir(tmp_path)
        bridge = Bridge(keys={}, tavily_key="", settings=Settings.from_env())
        bridge.voice_keys = {"DEEPGRAM_API_KEY": KEY}

        status = json.dumps(bridge.status())

        assert KEY not in status
        assert bridge.voice_providers()["stt"]["deepgram"] is True

    async def test_a_realtime_failure_does_not_repeat_the_providers_message(self) -> None:
        published: list[dict] = []
        handlers: dict = {}
        session = MagicMock()
        session.on = lambda name: lambda fn: handlers.setdefault(name, fn)
        state = MagicMock()
        from src.domain import KwamiConfig

        state.current_agent.kwami_config = KwamiConfig()
        state.current_agent.kwami_config.voice.pipeline_type = "realtime"
        spawned: list = []
        state.spawn = lambda coro, name=None: spawned.append(coro)
        state.run_serialized = lambda coro: coro
        publisher = MagicMock()

        async def publish(message: dict) -> None:
            published.append(message)

        publisher.publish = publish
        install_realtime_rescue(session, state, None, MagicMock(), publisher)
        error = MagicMock(
            type="realtime_model_error",
            recoverable=False,
            error=ValueError(f"Incorrect API key provided: {KEY}"),
        )

        handlers["error"](MagicMock(error=error))
        await spawned[0]

        assert published
        assert KEY not in json.dumps(published)


class TestTheFirstPersonOwnsTheKeys:
    async def test_someone_joining_later_neither_brings_keys_nor_spends_the_first(
        self, monkeypatch
    ) -> None:
        from src.brain import bridge as bridge_module
        from src.brain.bridge import Bridge

        async def load(store, user_id, secret, *_):
            return {"OPENAI_API_KEY": f"key-of-{user_id}"}

        monkeypatch.setattr(bridge_module, "load_user_keys", load)
        bridge = Bridge.__new__(Bridge)
        bridge._keys = {}
        bridge._deployment_keys = {"GOOGLE_API_KEY": "ours"}
        bridge._keys_store = object()
        bridge._keys_secret = KEY
        bridge._keys_owner = ""
        bridge.voice_keys = {}
        bridge._voice_keys_loaded = asyncio.Event()

        async def nothing(*_, **__) -> None:
            return None

        monkeypatch.setattr(bridge, "_refresh_account", nothing, raising=False)
        monkeypatch.setattr(bridge, "publish_status", nothing, raising=False)
        monkeypatch.setattr(bridge, "_regate_voice", nothing, raising=False)

        bridge.user_id = "first"
        await bridge._on_user_known()
        assert bridge.voice_keys == {"OPENAI_API_KEY": "key-of-first"}
        assert bridge._keys == {"OPENAI_API_KEY": "key-of-first"}

        bridge.user_id = "second"
        await bridge._on_user_known()

        assert bridge.voice_keys == {}
        assert bridge._keys == {"GOOGLE_API_KEY": "ours"}
