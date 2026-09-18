"""Live reconfiguration of the realtime pipeline, and switching between pipelines.

`update_voice` in `config_handler` only ever read `tts_*` and `stt_*` keys. The
frontend SDK's `updateRealtimeLive()` sends `realtime_provider`,
`realtime_model` and `realtime_voice` under the *same* `updateType: "voice"`
message, so every one of those keys was read by nothing and dropped on the
floor: changing the realtime voice mid-conversation was a no-op, and there was
no path at all from `standard` to `realtime` (or back) without dropping the
session. This module is that missing half.

The split it draws is the one the provider forces:

* **Voice, speed and temperature are live.** `RealtimeModel.update_options`
  pushes a `session.update` over the open socket, so the next utterance comes
  back in the new voice with no reconnect and no dropped audio.
* **Provider, model and pipeline type are not.** They decide which object was
  constructed, so they need a new agent -- which is `create_agent_fn` plus
  `state.update_agent`, exactly as a TTS provider switch already does.

Preferring the live path matters for the product, not just for tidiness: an
agent swap tears down the realtime socket and re-greets, which the user hears.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..domain import clone_config, number, text
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from ..session import SessionState

logger = get_logger("realtime_config")

REALTIME_PIPELINE = "realtime"
STANDARD_PIPELINE = "standard"
PIPELINE_TYPES = (STANDARD_PIPELINE, REALTIME_PIPELINE)

#: Keys that only mean something to the realtime pipeline. Their presence is
#: what tells a mixed `voice` payload apart from a plain TTS one.
REALTIME_KEYS = (
    "realtime_provider",
    "realtimeProvider",
    "realtime_model",
    "realtimeModel",
    "realtime_voice",
    "realtimeVoice",
)


#: Where a pipeline type can appear, and what the clients call it.
#:
#: `type` is the one that matters in production and it was the one nobody read.
#: The frontend SDK's wire type is `VoicePipelineConfig.type`, whose values are
#: `'stt-llm-tts' | 'realtime' | 'hybrid'`, and `kwami-app` sends exactly that
#: on every connect. The handler looked only for `pipelineType` with the values
#: `'standard' | 'realtime'` -- two spellings that no client has ever sent. The
#: result was not a warning: `pipeline_type` simply stayed `"standard"` forever,
#: so selecting the realtime pipeline in the app built an STT+LLM+TTS agent and
#: parsed the realtime settings into fields nothing would read. The realtime
#: pipeline was unreachable from the product.
PIPELINE_KEYS = ("pipelineType", "pipeline_type", "pipeline", "type")

#: Every spelling either side uses, mapped to this codebase's two.
#: `hybrid` is in the SDK's union but has no implementation here; it resolves to
#: the standard pipeline, which is what it degrades to, and says so out loud
#: rather than being silently dropped.
PIPELINE_ALIASES: dict[str, str] = {
    "standard": STANDARD_PIPELINE,
    "stt-llm-tts": STANDARD_PIPELINE,
    "stt_llm_tts": STANDARD_PIPELINE,
    "sttllmtts": STANDARD_PIPELINE,
    "normal": STANDARD_PIPELINE,
    "classic": STANDARD_PIPELINE,
    "hybrid": STANDARD_PIPELINE,
    "realtime": REALTIME_PIPELINE,
    "real-time": REALTIME_PIPELINE,
    "real_time": REALTIME_PIPELINE,
    "speech-to-speech": REALTIME_PIPELINE,
    "s2s": REALTIME_PIPELINE,
}


def normalize_pipeline_type(raw: Any) -> str | None:
    """Map any client's spelling onto `standard` or `realtime`, or None.

    Returning None for an unrecognised value is deliberate: the pipeline branch
    is chosen by equality against `"realtime"`, so passing a typo through would
    silently mean "standard" and hand the user the wrong pipeline with no error.
    """
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower()
    if not normalized:
        return None
    resolved = PIPELINE_ALIASES.get(normalized)
    if resolved is None:
        logger.warning("Ignoring unknown pipeline type %r", raw)
        return None
    if normalized == "hybrid":
        logger.warning("Pipeline type 'hybrid' is not implemented; using the standard pipeline")
    return resolved


def requested_pipeline(config: dict[str, Any]) -> str | None:
    """The pipeline type this payload asks for, or None if it does not ask."""
    return normalize_pipeline_type(text(config, *PIPELINE_KEYS))


def has_realtime_keys(config: Any) -> bool:
    """True when the payload carries at least one realtime-only field.

    Takes `Any`, not `dict`: this is a decoded data-channel payload, and the
    `config` key of a `config_update` message is whatever the client put there.
    Declaring it a dict would make the guard below dead code on paper while it
    is the only thing standing between a malformed packet and an AttributeError
    in the data handler.
    """
    if not isinstance(config, dict):
        return False
    return any(config.get(key) is not None for key in REALTIME_KEYS)


def apply_realtime_fields(voice_config: Any, config: dict[str, Any]) -> set[str]:
    """Copy realtime fields off the wire onto a voice config.

    Returns the set of field names that actually changed, so the caller can
    tell "voice only" (live-updatable) from "provider or model" (needs a new
    agent) without comparing the two configs itself.
    """
    changed: set[str] = set()

    provider_in = text(config, "realtime_provider", "realtimeProvider")
    if provider_in and provider_in != voice_config.realtime_provider:
        voice_config.realtime_provider = provider_in
        changed.add("realtime_provider")

    model_in = text(config, "realtime_model", "realtimeModel")
    if model_in:
        # Strip against the provider that will actually be used, which may be
        # the one this same payload just set.
        stripped = strip_model_prefix(model_in, voice_config.realtime_provider)
        if stripped != voice_config.realtime_model:
            voice_config.realtime_model = stripped
            changed.add("realtime_model")

    voice_in = text(config, "realtime_voice", "realtimeVoice")
    if voice_in and voice_in != voice_config.realtime_voice:
        voice_config.realtime_voice = voice_in
        changed.add("realtime_voice")

    temperature_in = number(config, "temperature", "llm_temperature", "llmTemperature")
    if temperature_in is not None and temperature_in != voice_config.llm_temperature:
        voice_config.llm_temperature = temperature_in
        changed.add("llm_temperature")

    return changed


def realtime_model_of(agent: Any) -> Any:
    """The live `RealtimeModel` driving this agent, or None.

    In the realtime pipeline the model is handed to `Agent(llm=...)`, so it is
    reachable as `agent.llm`. `agent.realtime_llm_session` is deliberately NOT
    used as the entry point: it raises when there is no running activity, which
    is exactly the window a config message can arrive in.
    """
    model = getattr(agent, "llm", None)
    if model is None:
        return None
    if not hasattr(model, "update_options"):
        return None
    # A plain LLM also has update_options; the realtime one is distinguished by
    # carrying a session factory for the open socket.
    if not hasattr(model, "session"):
        return None
    return model


LIVE_UPDATABLE = {"realtime_voice": "voice", "llm_temperature": "temperature"}


def apply_live_realtime_options(agent: Any, changed: set[str]) -> bool:
    """Push voice/temperature onto the open realtime socket.

    Returns False when the update could not be applied live, which is the
    caller's signal to fall back to rebuilding the agent rather than silently
    reporting success on a change the user will never hear.
    """
    model = realtime_model_of(agent)
    if model is None:
        return False

    voice_config = agent.kwami_config.voice
    updates = {
        kwarg: getattr(voice_config, field)
        for field, kwarg in LIVE_UPDATABLE.items()
        if field in changed
    }
    if not updates:
        return True

    try:
        model.update_options(**updates)
    except Exception as e:
        logger.warning("Live realtime update failed (%s); rebuilding the agent instead", e)
        return False

    logger.info("Updated realtime options live: %s", sorted(updates))
    return True


def _rebuild(
    session: AgentSession,
    state: SessionState,
    agent: Any,
    voice_config: Any,
    vad: Any,
    create_agent_fn: Any,
) -> None:
    """Swap in a new agent built from `voice_config`, keeping memory and browser."""
    new_config = clone_config(agent.kwami_config)
    new_config.voice = voice_config
    new_agent = create_agent_fn(new_config, vad, agent._memory, skip_greeting=True)
    state.update_agent(session, new_agent)


async def update_realtime(
    session: AgentSession,
    state: SessionState,
    agent: Any,
    config: dict[str, Any],
    vad: Any,
    create_agent_fn: Any,
) -> None:
    """Apply a realtime-pipeline voice update, live where the provider allows it."""
    voice_config = replace(agent.kwami_config.voice)
    changed = apply_realtime_fields(voice_config, config)

    if not changed:
        logger.debug("Realtime update carried no changes")
        return

    needs_rebuild = bool(changed - set(LIVE_UPDATABLE))

    if not needs_rebuild:
        # Commit to the live config first: apply_live_realtime_options reads the
        # new values off it, and on success this is already the truth.
        previous = agent.kwami_config.voice
        agent.kwami_config.voice = voice_config
        agent._current_voice_config = voice_config
        if apply_live_realtime_options(agent, changed):
            return
        # The live push failed, so the running model still has the old values.
        agent.kwami_config.voice = previous
        agent._current_voice_config = previous

    logger.info(
        "Rebuilding realtime pipeline: %s/%s voice=%s (changed: %s)",
        voice_config.realtime_provider,
        voice_config.realtime_model,
        voice_config.realtime_voice,
        sorted(changed),
    )
    _rebuild(session, state, agent, voice_config, vad, create_agent_fn)


async def switch_pipeline(
    session: AgentSession,
    state: SessionState,
    agent: Any,
    config: dict[str, Any],
    vad: Any,
    create_agent_fn: Any,
    target: str,
) -> None:
    """Move the session between the standard and realtime pipelines.

    Always a rebuild: the two pipelines are different objects (STT+LLM+TTS
    versus a single realtime model), so there is nothing to update in place.
    Any realtime fields riding along in the same payload are applied first, so
    "switch to realtime with the Cedar voice" takes one message rather than two.
    """
    voice_config = replace(agent.kwami_config.voice)
    voice_config.pipeline_type = target
    apply_realtime_fields(voice_config, config)

    logger.info("Switching pipeline: %s -> %s", agent.kwami_config.voice.pipeline_type, target)
    _rebuild(session, state, agent, voice_config, vad, create_agent_fn)
