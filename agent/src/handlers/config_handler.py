"""Configuration message handlers for Kwami agent."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

from livekit.agents.voice.agent import find_function_tools

from ..domain import KwamiConfig, clone_config, integer, number, section, text
from ..memory import create_memory
from ..tools.limits import enforce_tool_limit
from ..utils.logging import get_logger
from ..utils.provider import detect_provider_change, strip_model_prefix
from .realtime import (
    REALTIME_PIPELINE,
    has_realtime_keys,
    requested_pipeline,
    switch_pipeline,
    update_realtime,
)

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from ..session import SessionState

logger = get_logger("config_handler")


def _reuse_existing_memory(state: SessionState, new_memory_config: Any) -> Any:
    """Return the live memory instance when the new config targets the same Zep user.

    `create_memory` builds a fresh `AsyncZep`, re-runs `set_ontology` (a
    destructive project-level replace), re-upserts the context template and
    mints a new `session_{user}_{uuid4}` thread. Doing that on every config
    message churned one client per message -- none of which could be closed --
    and scattered a single conversation across several threads, so recall found
    nothing. Reuse instead, and carry the retrieval knobs over so live memory
    updates still take effect.
    """
    agent = state.current_agent
    memory = getattr(agent, "_memory", None) if agent is not None else None
    if memory is None or not getattr(memory, "is_initialized", False):
        return None
    if getattr(memory.config, "user_id", None) != getattr(new_memory_config, "user_id", None):
        return None

    for knob in ("max_context_messages", "include_facts", "min_fact_relevance"):
        if hasattr(new_memory_config, knob):
            setattr(memory.config, knob, getattr(new_memory_config, knob))

    logger.info("Reusing existing Zep memory for user %s", memory.config.user_id)
    return memory


def _value_from_keys(config: dict[str, Any], *keys: str) -> Any:
    """Return the first present key value (supports falsy values)."""
    for key in keys:
        if key in config:
            return config[key]
    return None


async def handle_full_config(
    session: AgentSession,
    state: SessionState,
    message: dict[str, Any],
    vad: Any,
    create_agent_fn: Any,
) -> None:
    """Handle the 'config' message which sets the entire identity/pipeline.

    Args:
        session: The LiveKit agent session.
        state: Current session state.
        message: The config message from the client.
        vad: Voice Activity Detection instance.
        create_agent_fn: Function to create a new agent from config.
    """
    try:
        logger.info("Processing full configuration...")

        # 1. Parse into KwamiConfig
        new_config = KwamiConfig()

        # Apply frontend voice config. `section` tolerates a null or wrong-typed
        # "voice" key, which used to raise and drop the entire config message.
        voice_data = section(message, "voice")

        # Pipeline selection. Reading only `pipelineType` with the values
        # `standard`/`realtime` meant reading a spelling no client sends: the
        # SDK's wire field is `voice.type`, with `stt-llm-tts`/`realtime`. So
        # choosing the realtime pipeline in the app left `pipeline_type` at its
        # default and built an STT+LLM+TTS agent, while the realtime settings
        # below were parsed into fields nothing would go on to read.
        # `requested_pipeline` accepts every spelling either side uses.
        pipeline_type_in = requested_pipeline(voice_data)
        if pipeline_type_in is not None:
            # normalize_pipeline_type only ever returns one of the two Literal
            # members or None; an unrecognised spelling is rejected upstream
            # rather than passed through.
            new_config.voice.pipeline_type = cast(
                Literal["standard", "realtime"], pipeline_type_in
            )

        realtime_data = section(voice_data, "realtime")
        realtime_provider_in = text(realtime_data, "provider") or text(
            voice_data, "realtimeProvider", "realtime_provider"
        )
        if realtime_provider_in:
            new_config.voice.realtime_provider = realtime_provider_in
        realtime_model_in = text(realtime_data, "model") or text(
            voice_data, "realtimeModel", "realtime_model"
        )
        if realtime_model_in:
            provider = realtime_provider_in or new_config.voice.realtime_provider
            new_config.voice.realtime_model = strip_model_prefix(realtime_model_in, provider)
        realtime_voice_in = text(realtime_data, "voice") or text(
            voice_data, "realtimeVoice", "realtime_voice"
        )
        if realtime_voice_in:
            new_config.voice.realtime_voice = realtime_voice_in

        # TTS. Numeric fields go through `number`, so an explicit 0 is honoured
        # rather than being read as "not provided".
        tts_data = section(voice_data, "tts")
        tts_provider_in = text(tts_data, "provider")
        if tts_provider_in:
            new_config.voice.tts_provider = tts_provider_in
        tts_model_in = text(tts_data, "model")
        if tts_model_in:
            # Strip provider prefix from model (e.g. "openai/tts-1" -> "tts-1")
            tts_provider = tts_provider_in or new_config.voice.tts_provider
            new_config.voice.tts_model = strip_model_prefix(tts_model_in, tts_provider)
        tts_voice_in = text(tts_data, "voice")
        if tts_voice_in:
            new_config.voice.tts_voice = tts_voice_in
        tts_speed_in = number(tts_data, "speed")
        if tts_speed_in is not None:
            new_config.voice.tts_speed = tts_speed_in

        # LLM
        llm_data = section(voice_data, "llm")
        llm_provider_in = text(llm_data, "provider")
        if llm_provider_in:
            new_config.voice.llm_provider = llm_provider_in
        llm_model_in = text(llm_data, "model")
        if llm_model_in:
            llm_provider = llm_provider_in or new_config.voice.llm_provider
            new_config.voice.llm_model = strip_model_prefix(llm_model_in, llm_provider)
        temperature_in = number(llm_data, "temperature")
        if temperature_in is not None:
            new_config.voice.llm_temperature = temperature_in
        max_tokens_in = integer(llm_data, "maxTokens", "max_tokens")
        if max_tokens_in is not None:
            new_config.voice.llm_max_tokens = max_tokens_in

        # STT
        stt_data = section(voice_data, "stt")
        stt_provider_in = text(stt_data, "provider")
        if stt_provider_in:
            new_config.voice.stt_provider = stt_provider_in
        stt_model_in = text(stt_data, "model")
        if stt_model_in:
            stt_provider = stt_provider_in or new_config.voice.stt_provider
            new_config.voice.stt_model = strip_model_prefix(stt_model_in, stt_provider)
        stt_language_in = text(stt_data, "language")
        if stt_language_in:
            new_config.voice.stt_language = stt_language_in

        # Kwami details
        # Use kwamiId from message, or fall back to user_identity (participant name)
        kwami_id = message.get("kwamiId") or state.user_identity
        if kwami_id:
            new_config.kwami_id = kwami_id
            logger.info("Using kwami_id for memory: %s", kwami_id)
            # Update user_identity for usage reporting (may be None at session start)
            if not state.user_identity:
                state.user_identity = kwami_id
                logger.info("Set user_identity from config: %s", kwami_id)
        if message.get("kwamiName"):
            new_config.kwami_name = message["kwamiName"]

        # Where the user actually is. Optional: an older app that does not send
        # these still works, and `get_current_time` then labels its answer UTC
        # instead of pretending the container's clock is the user's.
        timezone_in = text(message, "timezone", "timeZone", "tz")
        if timezone_in:
            new_config.timezone = timezone_in
        locale_in = text(message, "locale", "lang")
        if locale_in:
            new_config.locale = locale_in

        # Tools (client-side executable tools sent from the frontend)
        tools_data = message.get("tools")
        if tools_data and isinstance(tools_data, list):
            new_config.tools = tools_data
            logger.info("Loaded %s client tools from config", len(tools_data))

        # Soul (supports legacy "persona" key during migration)
        soul_data = message.get("soul") or message.get("persona", {})
        if soul_data.get("name"):
            new_config.soul.name = soul_data["name"]
        if soul_data.get("personality"):
            new_config.soul.personality = soul_data["personality"]
        system_prompt = _value_from_keys(soul_data, "systemPrompt", "system_prompt")
        if system_prompt is not None:
            new_config.soul.system_prompt = system_prompt
        if soul_data.get("traits"):
            new_config.soul.traits = soul_data["traits"]
        conversation_style = _value_from_keys(soul_data, "conversationStyle", "conversation_style")
        if conversation_style:
            new_config.soul.conversation_style = conversation_style

        # What the model *writes*, as opposed to `voice.stt.language`, which is
        # what it hears. The field existed and was read by nothing, so a soul
        # configured for Spanish was transcribed and spoken in Spanish while the
        # replies themselves stayed English.
        soul_language = _value_from_keys(soul_data, "language", "lang")
        if soul_language:
            new_config.soul.language = soul_language
        response_length = _value_from_keys(soul_data, "responseLength", "response_length")
        if response_length:
            new_config.soul.response_length = response_length
        emotional_tone = _value_from_keys(soul_data, "emotionalTone", "emotional_tone")
        if emotional_tone:
            new_config.soul.emotional_tone = emotional_tone
        emotional_traits = _value_from_keys(soul_data, "emotionalTraits", "emotional_traits")
        if isinstance(emotional_traits, dict):
            new_config.soul.emotional_traits = emotional_traits

        # 2. Initialize Memory
        memory = None
        if new_config.memory.enabled or message.get("memory", {}).get("enabled"):
            # Update memory config if present in message
            mem_data = message.get("memory", {})
            if mem_data.get("enabled") is not None:
                new_config.memory.enabled = mem_data["enabled"]
            if mem_data.get("maxContextMessages") is not None:
                new_config.memory.max_context_messages = int(mem_data["maxContextMessages"])
            if mem_data.get("includeFacts") is not None:
                new_config.memory.include_facts = bool(mem_data["includeFacts"])
            if mem_data.get("minFactRelevance") is not None:
                new_config.memory.min_fact_relevance = float(mem_data["minFactRelevance"])

            if new_config.memory.enabled:
                if not new_config.memory.user_id and new_config.kwami_id:
                    # Client sends full memory id (e.g. kwami_<auth>_<kwamiId>); use as-is so each kwami has its own memory
                    new_config.memory.user_id = new_config.kwami_id
                memory = _reuse_existing_memory(state, new_config.memory)
                if memory is None:
                    memory = await create_memory(
                        config=new_config.memory,
                        kwami_id=new_config.kwami_id or "default",
                        kwami_name=new_config.kwami_name,
                        usage_tracker=state.usage_tracker,
                    )

        # 3. Create NEW Agent with this config
        # Only skip greeting if one was already delivered in this session.
        # The first "config" message arrives right after the default agent starts,
        # killing its greeting before it completes. The reconfigured agent must
        # greet in that case.
        skip_greeting = state.greeting_delivered
        new_agent = create_agent_fn(new_config, vad, memory, skip_greeting=skip_greeting)

        # 4. Switch to new agent (state handles memory cleanup)
        state.update_agent(session, new_agent)

        # Mark that we've handled the first config (greeting will be delivered by new agent)
        if not skip_greeting:
            state.greeting_delivered = True

        logger.info(
            "Reconfigured agent: %s/%s",
            new_config.voice.llm_provider,
            new_config.voice.tts_provider,
        )

    except Exception:
        logger.exception("Failed to process full config")


async def handle_config_update(
    session: AgentSession,
    state: SessionState,
    message: dict[str, Any],
    vad: Any,
    create_agent_fn: Any,
) -> None:
    """Handle partial updates (voice, llm, soul).

    Args:
        session: The LiveKit agent session.
        state: Current session state.
        message: The config update message from the client.
        vad: Voice Activity Detection instance.
        create_agent_fn: Function to create a new agent from config.
    """
    from ..agent import KwamiAgent

    update_type = message.get("updateType")
    config_payload = message.get("config", {})

    current_agent = state.current_agent
    if not isinstance(current_agent, KwamiAgent):
        return

    try:
        if update_type == "voice":
            await update_voice(session, state, current_agent, config_payload, vad, create_agent_fn)
        elif update_type == "pipeline":
            target = requested_pipeline(config_payload) or requested_pipeline(message)
            if target is None:
                logger.warning("pipeline update carried no usable pipeline type")
            elif target != current_agent.kwami_config.voice.pipeline_type:
                await switch_pipeline(
                    session, state, current_agent, config_payload, vad, create_agent_fn, target
                )
            else:
                await update_voice(
                    session, state, current_agent, config_payload, vad, create_agent_fn
                )
        elif update_type == "llm":
            await update_llm(session, state, current_agent, config_payload, vad, create_agent_fn)
        elif update_type == "memory":
            await update_memory(current_agent, config_payload)
        elif update_type in {"soul", "persona"}:
            await update_soul(session, current_agent, config_payload)
        elif update_type == "tools":
            await update_tools(current_agent, config_payload)

    except Exception:
        logger.exception("Error updating %s", update_type)


async def update_voice(
    session: AgentSession,
    state: SessionState,
    agent: Any,
    config: dict[str, Any],
    vad: Any,
    create_agent_fn: Any,
) -> None:
    """Update voice configuration, switching pipeline or provider if needed.

    One `updateType: "voice"` message carries three different kinds of change,
    because that is what the frontend SDK sends: a pipeline switch, a realtime
    voice/model change, and a TTS/STT change. They are dispatched here in that
    order of specificity. Before this, only the last of the three was read, so
    `updateRealtimeLive()` was a no-op on the wire.

    Args:
        session: The LiveKit agent session.
        state: Current session state.
        agent: The current KwamiAgent instance.
        config: Voice configuration updates.
        vad: Voice Activity Detection instance.
        create_agent_fn: Function to create a new agent from config.
    """
    current_pipeline = agent.kwami_config.voice.pipeline_type
    target_pipeline = requested_pipeline(config)

    if target_pipeline is not None and target_pipeline != current_pipeline:
        await switch_pipeline(session, state, agent, config, vad, create_agent_fn, target_pipeline)
        return

    # Realtime keys are meaningless to the standard pipeline; routing on the
    # live pipeline type (not on key presence alone) keeps a stale realtime_*
    # field in a mixed payload from hijacking a TTS update.
    if current_pipeline == REALTIME_PIPELINE:
        if has_realtime_keys(config):
            await update_realtime(session, state, agent, config, vad, create_agent_fn)
        else:
            logger.debug("Ignoring TTS/STT voice update while on the realtime pipeline")
        return

    current_provider = agent.kwami_config.voice.tts_provider
    new_model = config.get("tts_model")
    new_voice = config.get("tts_voice")

    # Use utility function to detect provider change
    new_provider, provider_changed = detect_provider_change(
        current_provider,
        new_model=new_model,
        new_voice=new_voice,
    )

    # Override with explicit provider if specified
    if config.get("tts_provider"):
        explicit_provider = config["tts_provider"]
        if explicit_provider != new_provider:
            new_provider = explicit_provider
            provider_changed = new_provider != current_provider

    if provider_changed:
        logger.info("Auto-detected provider change: %s -> %s", current_provider, new_provider)

    # Some providers don't support live speed updates via update_options and need agent recreation.
    # Only trigger recreation if speed actually changed from current value.
    recreate_on_speed_change_providers = {"elevenlabs", "rime"}
    requires_recreate_for_speed = current_provider in recreate_on_speed_change_providers
    is_elevenlabs = current_provider == "elevenlabs"
    current_speed = agent.kwami_config.voice.tts_speed or 1.0
    new_speed = config.get("tts_speed")
    speed_actually_changed = new_speed is not None and float(new_speed) != float(current_speed)
    speed_changed = speed_actually_changed and requires_recreate_for_speed

    if provider_changed or speed_changed:
        reason = "provider change" if provider_changed else f"speed change ({current_provider})"
        logger.info("Switching TTS: %s -> %s (%s)", current_provider, new_provider, reason)

        # Full agent switch needed for provider change
        new_voice_config = replace(agent.kwami_config.voice)
        new_voice_config.tts_provider = new_provider
        if new_model:
            new_voice_config.tts_model = strip_model_prefix(new_model, new_provider)
        elif provider_changed:
            # Clear old model when switching providers so the factory uses the
            # new provider's default. Otherwise the old provider's model
            # (e.g. Cartesia "sonic-3") carries over to OpenAI where it's invalid.
            new_voice_config.tts_model = ""
        if new_voice:
            new_voice_config.tts_voice = new_voice
        elif provider_changed:
            # Clear old voice when switching providers so the factory uses the
            # new provider's default. Otherwise the old provider's voice
            # (e.g. Rime "astra") carries over to the new provider (e.g. ElevenLabs)
            # where it doesn't exist.
            new_voice_config.tts_voice = ""
        speed_in = number(config, "tts_speed")
        if speed_in is not None:
            new_voice_config.tts_speed = speed_in

        new_config = clone_config(agent.kwami_config)
        new_config.voice = new_voice_config

        new_agent = create_agent_fn(new_config, vad, agent._memory, skip_greeting=True)
        state.update_agent(session, new_agent)
        logger.info("Switched to %s TTS", new_provider)
    else:
        # Same provider - just update options if supported
        await _update_tts_options(agent, config, new_voice, is_elevenlabs)

        # Handle STT updates
        await _update_stt_if_needed(session, state, agent, config, vad, create_agent_fn)


async def _update_tts_options(
    agent: Any,
    config: dict[str, Any],
    new_voice: str | None,
    is_elevenlabs: bool,
) -> None:
    """Update TTS options without recreating the agent."""
    if not hasattr(agent, "tts") or not agent.tts:
        return

    # Heterogeneous on purpose: this is **-splatted into `update_options`,
    # where voice is a str and speed a float.
    updates: dict[str, Any] = {}

    # Detect TTS provider from module or passed parameter
    tts_provider = getattr(agent.tts, "provider", "").lower()
    tts_module = type(agent.tts).__module__
    tts_model = str(getattr(agent.tts, "_model", getattr(agent.tts, "model", ""))).lower()

    # Check if TTS is using LiveKit Inference (ElevenLabs, Rime, etc.)
    is_inference_tts = "inference" in tts_module
    is_elevenlabs_tts = (
        is_elevenlabs
        or tts_provider == "elevenlabs"
        or "elevenlabs" in tts_module
        or "elevenlabs" in tts_model
    )
    is_openai_tts = "openai" in tts_module and not is_inference_tts

    if new_voice:
        # Validate voice against current TTS provider to avoid sending
        # unsupported voices (e.g. Rime voice 'orion' to OpenAI fallback)
        if is_openai_tts:
            from ..constants import OpenAIVoices

            if new_voice not in OpenAIVoices.STANDARD:
                logger.warning(
                    "Voice '%s' not valid for current OpenAI TTS, skipping voice update. Valid: %s",
                    new_voice,
                    ", ".join(sorted(OpenAIVoices.STANDARD)),
                )
                new_voice = None  # Skip this update

        if new_voice:
            # inference.TTS (used for ElevenLabs, Rime via LiveKit Inference)
            # always uses "voice", not "voice_id". Only the direct
            # elevenlabs.TTS plugin uses "voice_id".
            if is_elevenlabs_tts and not is_inference_tts:
                updates["voice_id"] = new_voice
            else:
                updates["voice"] = new_voice

    # LiveKit Inference TTS (ElevenLabs, Rime) doesn't support speed in update_options
    speed_in = number(config, "tts_speed")
    if speed_in is not None and not is_inference_tts:
        updates["speed"] = speed_in

    if updates and hasattr(agent.tts, "update_options"):
        try:
            agent.tts.update_options(**updates)
            # Update stored config to reflect new values
            if new_voice:
                agent.kwami_config.voice.tts_voice = new_voice
            if speed_in is not None:
                agent.kwami_config.voice.tts_speed = speed_in
            logger.info("Updated TTS options: %s", updates)
        except Exception as e:
            logger.warning("Failed to update TTS options: %s", e)


async def _update_stt_if_needed(
    session: AgentSession,
    state: SessionState,
    agent: Any,
    config: dict[str, Any],
    vad: Any,
    create_agent_fn: Any,
) -> None:
    """Update STT configuration if needed."""
    stt_provider_changed = (
        config.get("stt_provider")
        and config["stt_provider"] != agent.kwami_config.voice.stt_provider
    )
    stt_model_changed = (
        config.get("stt_model") and config["stt_model"] != agent.kwami_config.voice.stt_model
    )

    if stt_provider_changed or stt_model_changed:
        # STT provider/model change requires agent recreation
        current_stt = agent.kwami_config.voice.stt_provider
        new_stt = config.get("stt_provider", current_stt)
        logger.info("Switching STT: %s -> %s", current_stt, new_stt)

        new_voice_config = replace(agent.kwami_config.voice)
        if config.get("stt_provider"):
            new_voice_config.stt_provider = config["stt_provider"]
        if config.get("stt_model"):
            stt_provider = config.get("stt_provider") or new_voice_config.stt_provider
            new_voice_config.stt_model = strip_model_prefix(config["stt_model"], stt_provider)
        if config.get("stt_language"):
            new_voice_config.stt_language = config["stt_language"]

        new_config = clone_config(agent.kwami_config)
        new_config.voice = new_voice_config

        new_agent = create_agent_fn(new_config, vad, agent._memory, skip_greeting=True)
        state.update_agent(session, new_agent)
        logger.info("Switched to %s STT", new_voice_config.stt_provider)
    elif hasattr(agent, "stt") and agent.stt:
        # Just update STT options (language only)
        updates = {}
        language_in = config.get("stt_language")
        if language_in:
            updates["language"] = language_in
        if updates and hasattr(agent.stt, "update_options"):
            agent.stt.update_options(**updates)
            # Write back, exactly as `_update_tts_options` does above. This
            # branch pushed the language to the live STT and stopped there, so
            # the next rebuild reconstructed STT from `kwami_config.voice` and
            # put the session back into the old language. The TTS sibling ten
            # lines up always did this; the two were meant to match.
            #
            # `language_in` is necessarily truthy here -- it is the only thing
            # that puts anything in `updates` -- so no second guard.
            agent.kwami_config.voice.stt_language = language_in
            logger.info("Updated STT options: %s", updates)


async def update_llm(
    session: AgentSession,
    state: SessionState,
    agent: Any,
    config: dict[str, Any],
    vad: Any,
    create_agent_fn: Any,
) -> None:
    """Update LLM configuration. Always requires agent recreation.

    Args:
        session: The LiveKit agent session.
        state: Current session state.
        agent: The current KwamiAgent instance.
        config: LLM configuration updates.
        vad: Voice Activity Detection instance.
        create_agent_fn: Function to create a new agent from config.
    """
    # On the realtime pipeline there is no separate LLM: "the model" IS the
    # realtime model. The models panel sends provider/model under updateType
    # "llm" regardless of pipeline, so translate rather than rebuild a standard
    # pipeline underneath a realtime session.
    if agent.kwami_config.voice.pipeline_type == REALTIME_PIPELINE:
        await update_realtime(
            session,
            state,
            agent,
            {
                "realtime_provider": text(config, "provider"),
                "realtime_model": text(config, "model"),
                "temperature": number(config, "temperature"),
            },
            vad,
            create_agent_fn,
        )
        return

    new_config = clone_config(agent.kwami_config)
    new_voice = replace(new_config.voice)

    provider_in = text(config, "provider")
    if provider_in:
        new_voice.llm_provider = provider_in
    model_in = text(config, "model")
    if model_in:
        llm_provider = provider_in or new_voice.llm_provider
        new_voice.llm_model = strip_model_prefix(model_in, llm_provider)
    temperature_in = number(config, "temperature")
    if temperature_in is not None:
        new_voice.llm_temperature = temperature_in
    max_tokens_in = integer(config, "maxTokens", "max_tokens")
    if max_tokens_in is not None:
        new_voice.llm_max_tokens = max_tokens_in

    new_config.voice = new_voice
    new_agent = create_agent_fn(new_config, vad, agent._memory, skip_greeting=True)
    state.update_agent(session, new_agent)


async def update_soul(
    session: AgentSession,
    agent: Any,
    config: dict[str, Any],
) -> None:
    """Update soul configuration without recreating the agent.

    Args:
        session: The LiveKit agent session (for update_instructions).
        agent: The current KwamiAgent instance.
        config: Soul configuration updates.
    """
    soul = agent.kwami_config.soul
    updated = False

    if "name" in config:
        soul.name = config["name"]
        updated = True
    if "personality" in config:
        soul.personality = config["personality"]
        updated = True
    if "systemPrompt" in config or "system_prompt" in config:
        soul.system_prompt = _value_from_keys(config, "systemPrompt", "system_prompt")
        updated = True
    if "traits" in config:
        soul.traits = config["traits"]
        updated = True
    if "language" in config or "lang" in config:
        soul.language = _value_from_keys(config, "language", "lang")
        updated = True
    if "conversationStyle" in config or "conversation_style" in config:
        soul.conversation_style = _value_from_keys(
            config, "conversationStyle", "conversation_style"
        )
        updated = True
    if "responseLength" in config or "response_length" in config:
        soul.response_length = _value_from_keys(config, "responseLength", "response_length")
        updated = True
    if "emotionalTone" in config or "emotional_tone" in config:
        soul.emotional_tone = _value_from_keys(config, "emotionalTone", "emotional_tone")
        updated = True
    if "emotionalTraits" in config or "emotional_traits" in config:
        emotional_traits = _value_from_keys(config, "emotionalTraits", "emotional_traits")
        if isinstance(emotional_traits, dict):
            soul.emotional_traits = emotional_traits
            updated = True
    if updated:
        agent.kwami_config.soul = soul

        # Preserve memory context during live soul updates.
        memory_text = None
        if getattr(agent, "_last_memory_context", None) is not None:
            try:
                memory_text = agent._last_memory_context.to_system_prompt_addition()
            except Exception:
                memory_text = None

        # Rebuild and update instructions through the session
        new_instructions = agent._build_system_prompt(memory_text)
        await agent.update_instructions(new_instructions)
        logger.info("Updated soul: %s - %s...", soul.name, (soul.personality or "")[:50])


async def update_tools(
    agent: Any,
    tools: Any,
) -> None:
    """Register or refresh client-side tool definitions on the running agent.

    Called when the frontend sends a tools config_update (e.g. after connect
    when syncConfigToBackend('tools', ...) fires). Re-registers all provided
    tool definitions so the LLM can call them.

    Args:
        agent: The current KwamiAgent instance.
        tools: List of tool definition dicts from the frontend.
    """
    if not isinstance(tools, list) or not tools:
        logger.warning("update_tools: received empty or non-list tools payload, skipping")
        return

    try:
        agent.kwami_config.tools = tools
        # Re-register clears old definitions and registers fresh ones
        agent.client_tools.registered_tools = []
        agent.client_tools._tools = []
        agent.client_tools.register_client_tools(tools)

        client_tools = agent.client_tools.create_client_tools()

        # The framework builds Agent._tools as `tools + find_function_tools(self)`,
        # so it holds the built-ins as well as the client tools. Assigning the
        # client tools straight onto agent._tools therefore deleted all 22
        # built-in tools -- web_search, product_search and every navigation
        # tool -- for the rest of the session. Rediscover the built-ins and go
        # through update_tools(), which is also what re-tools the chat context
        # and pushes the new set to a live realtime session.
        # Discover on the class, not the instance: inspect.getmembers on a live
        # agent evaluates `realtime_llm_session`, which raises when there is no
        # running activity.
        builtin_tools = find_function_tools(type(agent))
        # Capped at the provider's ceiling: the frontend decides how many client
        # tools arrive, and one over the limit is a 400 on every turn rather
        # than one missing feature.
        combined = enforce_tool_limit(builtin_tools, client_tools)
        await agent.update_tools(combined)

        logger.info(
            "update_tools: registered %d client tools alongside %d built-in tools (%d total)",
            len(client_tools),
            len(builtin_tools),
            len(combined),
        )
    except Exception:
        logger.exception("update_tools: failed to register client tools")


async def update_memory(
    agent: Any,
    config: dict[str, Any],
) -> None:
    """Update memory retrieval settings on the running agent."""
    memory_cfg = agent.kwami_config.memory
    updated = False

    if "maxContextMessages" in config:
        try:
            memory_cfg.max_context_messages = max(1, min(50, int(config["maxContextMessages"])))
            updated = True
        except (TypeError, ValueError):
            logger.warning("Invalid maxContextMessages value: %s", config.get("maxContextMessages"))

    if "includeFacts" in config:
        memory_cfg.include_facts = bool(config["includeFacts"])
        updated = True

    if "minFactRelevance" in config:
        try:
            memory_cfg.min_fact_relevance = max(0.0, min(1.0, float(config["minFactRelevance"])))
            updated = True
        except (TypeError, ValueError):
            logger.warning("Invalid minFactRelevance value: %s", config.get("minFactRelevance"))

    if not updated:
        return

    agent.kwami_config.memory = memory_cfg
    if getattr(agent, "_memory", None):
        agent._memory.config = memory_cfg
    logger.info(
        "Updated memory config: max_context_messages=%s include_facts=%s min_fact_relevance=%s",
        memory_cfg.max_context_messages,
        memory_cfg.include_facts,
        memory_cfg.min_fact_relevance,
    )


# Backward-compatible alias for any modules importing the old name.
update_persona = update_soul
