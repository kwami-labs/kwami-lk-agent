"""The agent changing its own model, voice and pipeline, from inside a tool call.

The system prompt has always told the model "you can change your voice or the
AI model being used if the user requests it". Until these tools existed that
was false: every path into `create_agent_from_config` started from a data
message the frontend sent, and a tool had no handle on the session state, the
VAD or the factory. The agent could only describe which panel to open.

Three properties are load-bearing and each has a test here:

* a rebuild is handed back to the framework, not installed inside the tool, so
  the sentence explaining the switch survives the switch;
* a realtime voice change goes over the open socket rather than rebuilding;
* a provider that cannot actually be built is reported as a failure instead of
  being announced as a success -- `create_llm` falls back to OpenAI in silence.
"""

from __future__ import annotations

from typing import Any

import pytest
from livekit.agents.voice import Agent

from src.agent import KwamiAgent
from src.domain import KwamiConfig
from src.runtime.container import AgentDeps
from src.runtime.reconfigure import Reconfigurator
from src.session import SessionState


class FakeRealtimeModel:
    """Carries the attributes `realtime_model_of` keys on, and records pushes."""

    def __init__(self, model: str = "gpt-realtime") -> None:
        self.model = model
        self.updates: list[dict[str, Any]] = []

    def session(self) -> Any:  # pragma: no cover - presence is the signal
        raise NotImplementedError

    def update_options(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeChatLLM:
    """A standard-pipeline LLM. `model` is what a fallback would give away."""

    def __init__(self, model: str) -> None:
        self.model = model


class Ctx:
    """The shape a tool sees: a RunContext carrying the deps container."""

    def __init__(self, deps: AgentDeps | None) -> None:
        self.userdata = deps


def _standard_agent(**voice: Any) -> KwamiAgent:
    config = KwamiConfig()
    for key, value in voice.items():
        setattr(config.voice, key, value)
    return KwamiAgent(config=config, llm=FakeChatLLM(config.voice.llm_model))


def _realtime_agent(**voice: Any) -> KwamiAgent:
    config = KwamiConfig()
    config.voice.pipeline_type = "realtime"
    config.voice.realtime_provider = "openai"
    config.voice.realtime_model = "gpt-realtime"
    config.voice.realtime_voice = "marin"
    for key, value in voice.items():
        setattr(config.voice, key, value)
    return KwamiAgent(config=config, llm=FakeRealtimeModel())


def _wire(agent: KwamiAgent, *, built_model: str | None = None) -> tuple[Ctx, list[Any]]:
    """A deps container whose factory records what it was asked to build.

    `built_model` overrides what the replacement agent's LLM reports, which is
    how the silent-fallback case is reproduced without a real provider.
    """
    built: list[Any] = []
    state = SessionState(current_agent=agent)

    def create_agent_fn(config, vad, memory=None, skip_greeting=False):
        realtime = config.voice.pipeline_type == "realtime"
        if realtime:
            llm: Any = FakeRealtimeModel(config.voice.realtime_model)
        else:
            llm = FakeChatLLM(built_model if built_model is not None else config.voice.llm_model)
        new_agent = KwamiAgent(config=config, llm=llm)
        built.append(new_agent)
        return new_agent

    deps = AgentDeps(
        reconfigure=Reconfigurator(state=state, vad=None, create_agent_fn=create_agent_fn)
    )
    return Ctx(deps), built


# -- change_ai_model --------------------------------------------------------


async def test_change_ai_model_hands_back_a_new_agent() -> None:
    """The framework installs it after the tool output, so the reply survives."""
    agent = _standard_agent(llm_provider="openai", llm_model="gpt-4o-mini")
    ctx, built = _wire(agent)

    result = await agent.change_ai_model(ctx, "Claude")

    assert isinstance(result, tuple), "a rebuild must be handed back, not self-installed"
    new_agent, message = result
    assert isinstance(new_agent, Agent)
    assert new_agent is built[0]
    assert new_agent.kwami_config.voice.llm_provider == "anthropic"
    assert new_agent.kwami_config.voice.llm_model.startswith("claude-")
    assert "claude" in message.lower()


async def test_change_ai_model_does_not_install_the_agent_itself() -> None:
    """Calling session.update_agent inside a tool discards the turn that called it."""
    agent = _standard_agent()
    ctx, built = _wire(agent)

    await agent.change_ai_model(ctx, "Gemini")

    # prepare_handoff has run (state bookkeeping), but no session swap happened.
    state = ctx.userdata.reconfigure.state
    assert state.current_agent is built[0]


async def test_change_ai_model_carries_the_conversation() -> None:
    agent = _standard_agent()
    agent._chat_ctx.add_message(role="user", content="my name is Alex")
    agent._chat_ctx.add_message(role="assistant", content="Hi Alex")
    ctx, built = _wire(agent)

    new_agent, _ = await agent.change_ai_model(ctx, "Groq")

    contents = [item.content for item in new_agent.chat_ctx.items]
    assert ["my name is Alex"] in contents, "the conversation was wiped by the model switch"


async def test_change_ai_model_refuses_to_claim_a_switch_that_fell_back() -> None:
    """`create_llm` answers an unbuildable provider with OpenAI and a log line.

    Reporting success off the request rather than the result is how the agent
    ends up insisting it is Claude while running gpt-4o-mini.
    """
    agent = _standard_agent(llm_provider="openai", llm_model="gpt-4o-mini")
    ctx, built = _wire(agent, built_model="gpt-4o-mini")

    result = await agent.change_ai_model(ctx, "Claude")

    assert not isinstance(result, tuple), "handed off to an agent running the wrong model"
    assert "couldn't switch" in result.lower()
    assert "anthropic" in result.lower()


async def test_change_ai_model_without_a_reconfigure_handle() -> None:
    agent = _standard_agent()

    result = await agent.change_ai_model(Ctx(AgentDeps()), "Claude")

    assert isinstance(result, str)
    assert "can't change" in result.lower()


async def test_change_ai_model_on_realtime_retargets_the_realtime_model() -> None:
    agent = _realtime_agent()
    ctx, built = _wire(agent)

    new_agent, _ = await agent.change_ai_model(ctx, "Gemini")

    voice = new_agent.kwami_config.voice
    assert voice.pipeline_type == "realtime"
    assert voice.realtime_provider == "google"
    assert voice.realtime_model == "gemini-2.0-flash-exp"


async def test_switching_realtime_provider_resets_the_voice() -> None:
    """An OpenAI voice name does not exist on Gemini Live.

    A realtime session handed an unknown voice fails at connect, which on this
    pipeline is silence rather than a fallback.
    """
    agent = _realtime_agent(realtime_voice="cedar")
    ctx, _ = _wire(agent)

    new_agent, _ = await agent.change_ai_model(ctx, "Gemini")

    assert new_agent.kwami_config.voice.realtime_voice == "Puck"


# -- switch_pipeline_mode ---------------------------------------------------


@pytest.mark.parametrize("spoken", ["realtime", "real time", "live", "s2s"])
async def test_switch_to_realtime(spoken: str) -> None:
    agent = _standard_agent()
    ctx, built = _wire(agent)

    new_agent, message = await agent.switch_pipeline_mode(ctx, spoken)

    assert new_agent.kwami_config.voice.pipeline_type == "realtime"
    assert "realtime" in message.lower()


async def test_switch_back_to_standard() -> None:
    agent = _realtime_agent()
    ctx, _ = _wire(agent)

    new_agent, _ = await agent.switch_pipeline_mode(ctx, "standard")

    assert new_agent.kwami_config.voice.pipeline_type == "standard"


async def test_switching_to_the_current_mode_is_a_no_op() -> None:
    agent = _realtime_agent()
    ctx, built = _wire(agent)

    result = await agent.switch_pipeline_mode(ctx, "realtime")

    assert isinstance(result, str)
    assert built == [], "rebuilt the pipeline for a mode it was already on"


async def test_an_unknown_mode_is_refused() -> None:
    agent = _standard_agent()
    ctx, built = _wire(agent)

    result = await agent.switch_pipeline_mode(ctx, "turbo")

    assert isinstance(result, str)
    assert built == []


# -- change_realtime_voice --------------------------------------------------


async def test_realtime_voice_change_goes_over_the_open_socket() -> None:
    """A rebuild here drops the connection and re-greets; the user hears it."""
    model = FakeRealtimeModel()
    agent = KwamiAgent(config=_realtime_agent().kwami_config, llm=model)
    ctx, built = _wire(agent)

    result = await agent.change_realtime_voice(ctx, "cedar")

    assert model.updates == [{"voice": "cedar"}]
    assert built == [], "rebuilt for a change the provider accepts live"
    assert isinstance(result, str)
    assert "cedar" in result.lower()


async def test_realtime_voice_name_is_canonicalised() -> None:
    """Gemini Live rejects `puck` for `Puck`; speech does not capitalise."""
    model = FakeRealtimeModel()
    config = _realtime_agent(realtime_provider="google", realtime_voice="Charon").kwami_config
    agent = KwamiAgent(config=config, llm=model)
    ctx, _ = _wire(agent)

    await agent.change_realtime_voice(ctx, "puck")

    assert model.updates == [{"voice": "Puck"}]


async def test_a_voice_from_the_wrong_provider_is_refused() -> None:
    model = FakeRealtimeModel()
    config = _realtime_agent(realtime_provider="google", realtime_voice="Puck").kwami_config
    agent = KwamiAgent(config=config, llm=model)
    ctx, _ = _wire(agent)

    result = await agent.change_realtime_voice(ctx, "cedar")

    assert model.updates == [], "sent an OpenAI voice to Gemini Live"
    assert "isn't a google realtime voice" in result.lower()


async def test_realtime_voice_tool_redirects_on_the_standard_pipeline() -> None:
    agent = _standard_agent()
    ctx, _ = _wire(agent)

    result = await agent.change_realtime_voice(ctx, "cedar")

    assert "change_voice" in result


# -- introspection ----------------------------------------------------------


async def test_get_pipeline_status_reports_what_is_running() -> None:
    agent = _standard_agent(llm_provider="anthropic", llm_model="claude-3-5-sonnet-latest")
    # The requested config says Anthropic; the constructed object says otherwise.
    agent._llm = FakeChatLLM("gpt-4o-mini")
    ctx, _ = _wire(agent)

    status = await agent.get_pipeline_status(ctx)

    assert status["requested"]["llm"] == "anthropic/claude-3-5-sonnet-latest"
    assert status["running"]["model"] == "gpt-4o-mini"
    assert status["can_reconfigure"] is True


async def test_list_available_voices_follows_the_pipeline() -> None:
    realtime = _realtime_agent()
    ctx, _ = _wire(realtime)
    listing = await realtime.list_available_voices(ctx)
    assert listing["pipeline"] == "realtime"
    assert "cedar" in listing["voices"]

    standard = _standard_agent(tts_provider="openai")
    listing = await standard.list_available_voices(Ctx(AgentDeps()))
    assert listing["pipeline"] == "standard"
    assert "cedar" not in listing["voices"], "a realtime-only voice leaked into the TTS list"
