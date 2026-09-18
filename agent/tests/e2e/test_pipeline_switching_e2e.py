"""Mid-conversation model, voice and pipeline switching, against real providers.

The offline suites prove the routing: the right payload reaches the right
handler and the right fields change. They cannot prove the thing that actually
breaks, which is that the resulting config *constructs*. A realtime model built
with a voice name its provider does not accept fails at connect, and on the
realtime pipeline a failed connect is silence, not a fallback -- so it is worth
paying for a real construction here.

Marked `live`: real clients, real credentials, excluded from the default run.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.domain.config import KwamiConfig
from src.runtime.container import AgentDeps
from src.runtime.pipeline import create_agent_from_config
from src.runtime.reconfigure import Reconfigurator
from src.session import SessionState

pytestmark = pytest.mark.live


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


class Ctx:
    def __init__(self, deps: AgentDeps) -> None:
        self.userdata = deps


def _realtime_config() -> KwamiConfig:
    config = KwamiConfig()
    config.voice.pipeline_type = "realtime"
    config.voice.realtime_provider = "openai"
    config.voice.realtime_model = "gpt-realtime"
    config.voice.realtime_voice = "marin"
    return config


def _standard_config() -> KwamiConfig:
    config = KwamiConfig()
    config.voice.llm_provider = "openai"
    config.voice.llm_model = "gpt-4o-mini"
    config.voice.tts_provider = "openai"
    config.voice.tts_model = "tts-1"
    config.voice.tts_voice = "nova"
    return config


def _wire(agent: Any) -> tuple[Ctx, SessionState]:
    state = SessionState(current_agent=agent)
    deps = AgentDeps(
        reconfigure=Reconfigurator(state=state, vad=None, create_agent_fn=create_agent_from_config)
    )
    return Ctx(deps), state


# -- the app's own connect payload, constructed for real --------------------


async def test_the_apps_realtime_payload_builds_a_realtime_model(openai_key: str) -> None:
    """`voice.type: "realtime"` -- the spelling kwami-app actually sends.

    Before the pipeline-key fix this silently produced an STT+LLM+TTS agent, so
    the assertion that matters is which kind of pipeline came out.
    """
    from src.handlers.config_handler import handle_full_config

    session, state = FakeSession(), SessionState()
    message = {
        "voice": {
            "type": "realtime",
            "stt": {"provider": "deepgram", "language": "en"},
            "llm": {"provider": "openai", "model": "gpt-4o-mini"},
            "tts": {"provider": "openai", "model": "tts-1", "voice": "nova"},
            "realtime": {"provider": "openai", "model": "gpt-realtime", "voice": "cedar"},
        },
        "soul": {"name": "Ada", "personality": "a precise assistant"},
    }

    await handle_full_config(
        session, state, message, vad=None, create_agent_fn=create_agent_from_config
    )

    assert session.agent is not None, "the config produced no agent"
    assert session.agent.kwami_config.voice.pipeline_type == "realtime"
    assert session.agent.tts is None, "a realtime session must not also build a TTS"
    assert session.agent.llm is not None
    assert "realtime" in type(session.agent.llm).__module__.lower()


# -- switching, with real construction on both sides ------------------------


async def test_switching_into_realtime_constructs(openai_key: str) -> None:
    agent = create_agent_from_config(_standard_config(), vad=None)
    ctx, _ = _wire(agent)

    new_agent, message = await agent.switch_pipeline_mode(ctx, "realtime")

    assert new_agent.kwami_config.voice.pipeline_type == "realtime"
    assert new_agent.llm is not None
    assert "realtime" in message.lower()


async def test_switching_back_to_standard_constructs(openai_key: str) -> None:
    agent = create_agent_from_config(_realtime_config(), vad=None)
    ctx, _ = _wire(agent)

    new_agent, _ = await agent.switch_pipeline_mode(ctx, "standard")

    assert new_agent.kwami_config.voice.pipeline_type == "standard"
    assert new_agent.tts is not None, "the standard pipeline came back without a TTS"
    assert new_agent.stt is not None


async def test_the_real_realtime_model_still_takes_a_voice_kwarg(openai_key: str) -> None:
    """The SDK call the whole seamless path rests on.

    Scoped to what it can actually prove. `update_options` on an unconnected
    model stores the value without contacting OpenAI -- verified: it accepts
    `"totally-not-a-voice-xyz"` and returns cleanly -- so this cannot tell a
    good voice name from a bad one. What it does catch, and what would silently
    turn every mid-conversation voice change into an audible rebuild, is the
    SDK renaming or dropping the `voice` kwarg. The provider's own opinion of
    the name is checked in the next test.
    """
    import inspect

    from livekit.plugins import openai as openai_plugin

    from src.handlers.realtime import apply_live_realtime_options

    signature = inspect.signature(openai_plugin.realtime.RealtimeModel.update_options)
    assert "voice" in signature.parameters, (
        "RealtimeModel.update_options no longer takes `voice`; "
        "every live voice change now silently falls back to a rebuild"
    )

    agent = create_agent_from_config(_realtime_config(), vad=None)
    agent.kwami_config.voice.realtime_voice = "cedar"

    assert apply_live_realtime_options(agent, {"realtime_voice"})


async def test_every_realtime_voice_constant_is_one_openai_accepts(openai_key: str) -> None:
    """`RealtimeVoices.OPENAI` against the provider that has to honour it.

    This matters because a wrong name here is not a caught error: the realtime
    session fails at connect, and on that pipeline a failed connect is silence.
    The user hears an agent that joined and never spoke.

    Checked against the session endpoint rather than through `update_options`,
    which does no validation offline. A rejected voice comes back as a 400
    naming the supported set, so a drift in either direction is legible.
    """
    import httpx

    from src.constants import RealtimeVoices

    url = "https://api.openai.com/v1/realtime/client_secrets"
    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # One bad name first, to confirm this endpoint really does validate --
        # otherwise a pass below would mean nothing at all.
        probe = await client.post(
            url,
            headers=headers,
            json={
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime",
                    "audio": {"output": {"voice": "not-a-real-voice-xyz"}},
                }
            },
        )
        assert probe.status_code == 400, (
            "the session endpoint stopped validating voices; this test proves nothing"
        )

        for voice in sorted(RealtimeVoices.OPENAI):
            response = await client.post(
                url,
                headers=headers,
                json={
                    "session": {
                        "type": "realtime",
                        "model": "gpt-realtime",
                        "audio": {"output": {"voice": voice}},
                    }
                },
            )
            assert response.status_code == 200, (
                f"OpenAI rejected the realtime voice {voice!r}: {response.text[:300]}"
            )


async def test_switching_model_carries_the_conversation_through_real_clients(
    openai_key: str,
) -> None:
    agent = create_agent_from_config(_standard_config(), vad=None)
    agent._chat_ctx.add_message(role="user", content="my name is Alex and I live in Barcelona")
    agent._chat_ctx.add_message(role="assistant", content="Good to meet you, Alex.")
    ctx, _ = _wire(agent)

    result = await agent.change_ai_model(ctx, "gpt-4.1-mini")

    assert isinstance(result, tuple), f"the switch was refused: {result}"
    new_agent, _ = result
    contents = [item.content for item in new_agent.chat_ctx.items]
    assert ["my name is Alex and I live in Barcelona"] in contents


async def test_an_unavailable_provider_is_reported_not_announced(openai_key: str) -> None:
    """The failure mode that matters: `create_llm` falls back in silence.

    Whether this exercises the fallback depends on whether
    livekit-plugins-anthropic is installed in the environment. Both outcomes are
    correct; what must never happen is a confident switch onto a model the
    pipeline did not build.
    """
    agent = create_agent_from_config(_standard_config(), vad=None)
    ctx, _ = _wire(agent)

    result = await agent.change_ai_model(ctx, "Claude")

    if isinstance(result, tuple):
        new_agent, _ = result
        built = str(getattr(new_agent.llm, "model", "")).lower()
        assert "claude" in built, f"announced a switch to Claude but built {built!r}"
    else:
        assert "couldn't switch" in result.lower()
