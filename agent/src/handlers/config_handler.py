"""Configuration message handlers for Kwami agent."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..config import KwamiConfig
from ..memory import create_memory
from ..utils.logging import get_logger, log_error
from ..utils.provider import detect_provider_change, strip_model_prefix

if TYPE_CHECKING:
    from livekit.agents import AgentSession
    from ..session import SessionState

logger = get_logger("config_handler")


def _value_from_keys(config: Dict[str, Any], *keys: str) -> Any:
    """Return the first present key value (supports falsy values)."""
    for key in keys:
        if key in config:
            return config[key]
    return None


async def handle_full_config(
    session: "AgentSession",
    state: "SessionState",
    message: Dict[str, Any],
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
        
        # Apply frontend voice config
        voice_data = message.get("voice", {})
        
        # TTS
        tts_data = voice_data.get("tts", {})
        if tts_data.get("provider"):
            new_config.voice.tts_provider = tts_data["provider"]
        if tts_data.get("model"):
            # Strip provider prefix from model (e.g. "openai/tts-1" -> "tts-1")
            tts_provider = tts_data.get("provider") or new_config.voice.tts_provider
            new_config.voice.tts_model = strip_model_prefix(tts_data["model"], tts_provider)
        if tts_data.get("voice"):
            new_config.voice.tts_voice = tts_data["voice"]
        if tts_data.get("speed"):
            new_config.voice.tts_speed = tts_data["speed"]
        
        # LLM
        llm_data = voice_data.get("llm", {})
        if llm_data.get("provider"):
            new_config.voice.llm_provider = llm_data["provider"]
        if llm_data.get("model"):
            llm_provider = llm_data.get("provider") or new_config.voice.llm_provider
            new_config.voice.llm_model = strip_model_prefix(llm_data["model"], llm_provider)
        if llm_data.get("temperature"):
            new_config.voice.llm_temperature = llm_data["temperature"]
        if llm_data.get("maxTokens"):
            new_config.voice.llm_max_tokens = llm_data["maxTokens"]
        
        # STT
        stt_data = voice_data.get("stt", {})
        if stt_data.get("provider"):
            new_config.voice.stt_provider = stt_data["provider"]
        if stt_data.get("model"):
            stt_provider = stt_data.get("provider") or new_config.voice.stt_provider
            new_config.voice.stt_model = strip_model_prefix(stt_data["model"], stt_provider)
        if stt_data.get("language"):
            new_config.voice.stt_language = stt_data["language"]
        
        # Kwami details
        # Use kwamiId from message, or fall back to user_identity (participant name)
        kwami_id = message.get("kwamiId") or state.user_identity
        if kwami_id:
            new_config.kwami_id = kwami_id
            logger.info(f"Using kwami_id for memory: {kwami_id}")
            # Update user_identity for usage reporting (may be None at session start)
            if not state.user_identity:
                state.user_identity = kwami_id
                logger.info(f"Set user_identity from config: {kwami_id}")
        if message.get("kwamiName"):
            new_config.kwami_name = message["kwamiName"]
        
        # Tools (client-side executable tools sent from the frontend)
        tools_data = message.get("tools")
        if tools_data and isinstance(tools_data, list):
            new_config.tools = tools_data
            logger.info(f"Loaded {len(tools_data)} client tools from config")

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
        conversation_style = _value_from_keys(
            soul_data, "conversationStyle", "conversation_style"
        )
        if conversation_style:
            new_config.soul.conversation_style = conversation_style
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
            f"Reconfigured agent: {new_config.voice.llm_provider}/{new_config.voice.tts_provider}"
        )

    except Exception as e:
        log_error(logger, "Failed to process full config", e)


async def handle_config_update(
    session: "AgentSession",
    state: "SessionState",
    message: Dict[str, Any],
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
        elif update_type == "llm":
            await update_llm(session, state, current_agent, config_payload, vad, create_agent_fn)
        elif update_type == "memory":
            await update_memory(current_agent, config_payload)
        elif update_type in {"soul", "persona"}:
            await update_soul(session, current_agent, config_payload)
        elif update_type == "tools":
            await update_tools(current_agent, config_payload)

    except Exception as e:
        log_error(logger, f"Error updating {update_type}", e)


async def update_voice(
    session: "AgentSession",
    state: "SessionState",
    agent: Any,
    config: Dict[str, Any],
    vad: Any,
    create_agent_fn: Any,
) -> None:
    """Update voice/TTS configuration, switching providers if needed.
    
    Args:
        session: The LiveKit agent session.
        state: Current session state.
        agent: The current KwamiAgent instance.
        config: Voice configuration updates.
        vad: Voice Activity Detection instance.
        create_agent_fn: Function to create a new agent from config.
    """
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
        logger.info(f"Auto-detected provider change: {current_provider} -> {new_provider}")
    
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
        logger.info(f"Switching TTS: {current_provider} -> {new_provider} ({reason})")
        
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
        if config.get("tts_speed"):
            new_voice_config.tts_speed = config["tts_speed"]
        
        new_config = replace(agent.kwami_config)
        new_config.voice = new_voice_config
        
        new_agent = create_agent_fn(new_config, vad, agent._memory, skip_greeting=True)
        state.update_agent(session, new_agent)
        logger.info(f"Switched to {new_provider} TTS")
    else:
        # Same provider - just update options if supported
        await _update_tts_options(agent, config, new_voice, is_elevenlabs)
        
        # Handle STT updates
        await _update_stt_if_needed(session, state, agent, config, vad, create_agent_fn)


async def _update_tts_options(
    agent: Any,
    config: Dict[str, Any],
    new_voice: Optional[str],
    is_elevenlabs: bool,
) -> None:
    """Update TTS options without recreating the agent."""
    if not hasattr(agent, "tts") or not agent.tts:
        return
    
    updates = {}
    
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
    is_rime_tts = tts_provider == "rime" or "rime" in tts_model
    is_openai_tts = "openai" in tts_module and not is_inference_tts
    
    if new_voice:
        # Validate voice against current TTS provider to avoid sending
        # unsupported voices (e.g. Rime voice 'orion' to OpenAI fallback)
        if is_openai_tts:
            from ..constants import OpenAIVoices
            if new_voice not in OpenAIVoices.STANDARD:
                logger.warning(
                    f"Voice '{new_voice}' not valid for current OpenAI TTS, skipping voice update. "
                    f"Valid: {', '.join(sorted(OpenAIVoices.STANDARD))}"
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
    if config.get("tts_speed") and not is_inference_tts:
        updates["speed"] = float(config["tts_speed"])
    
    if updates and hasattr(agent.tts, "update_options"):
        try:
            agent.tts.update_options(**updates)
            # Update stored config to reflect new values
            if new_voice:
                agent.kwami_config.voice.tts_voice = new_voice
            if config.get("tts_speed"):
                agent.kwami_config.voice.tts_speed = config["tts_speed"]
            logger.info(f"Updated TTS options: {updates}")
        except Exception as e:
            logger.warning(f"Failed to update TTS options: {e}")


async def _update_stt_if_needed(
    session: "AgentSession",
    state: "SessionState",
    agent: Any,
    config: Dict[str, Any],
    vad: Any,
    create_agent_fn: Any,
) -> None:
    """Update STT configuration if needed."""
    stt_provider_changed = (
        config.get("stt_provider") and 
        config["stt_provider"] != agent.kwami_config.voice.stt_provider
    )
    stt_model_changed = (
        config.get("stt_model") and 
        config["stt_model"] != agent.kwami_config.voice.stt_model
    )
    
    if stt_provider_changed or stt_model_changed:
        # STT provider/model change requires agent recreation
        current_stt = agent.kwami_config.voice.stt_provider
        new_stt = config.get("stt_provider", current_stt)
        logger.info(f"Switching STT: {current_stt} -> {new_stt}")
        
        new_voice_config = replace(agent.kwami_config.voice)
        if config.get("stt_provider"):
            new_voice_config.stt_provider = config["stt_provider"]
        if config.get("stt_model"):
            stt_provider = config.get("stt_provider") or new_voice_config.stt_provider
            new_voice_config.stt_model = strip_model_prefix(config["stt_model"], stt_provider)
        if config.get("stt_language"):
            new_voice_config.stt_language = config["stt_language"]
        
        new_config = replace(agent.kwami_config)
        new_config.voice = new_voice_config
        
        new_agent = create_agent_fn(new_config, vad, agent._memory, skip_greeting=True)
        state.update_agent(session, new_agent)
        logger.info(f"Switched to {new_voice_config.stt_provider} STT")
    elif hasattr(agent, "stt") and agent.stt:
        # Just update STT options (language only)
        updates = {}
        if config.get("stt_language"):
            updates["language"] = config["stt_language"]
        if updates and hasattr(agent.stt, "update_options"):
            agent.stt.update_options(**updates)
            logger.info(f"Updated STT options: {updates}")


async def update_llm(
    session: "AgentSession",
    state: "SessionState",
    agent: Any,
    config: Dict[str, Any],
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
    new_config = replace(agent.kwami_config)
    new_voice = replace(new_config.voice)
    
    if config.get("provider"):
        new_voice.llm_provider = config["provider"]
    if config.get("model"):
        llm_provider = config.get("provider") or new_voice.llm_provider
        new_voice.llm_model = strip_model_prefix(config["model"], llm_provider)
    if config.get("temperature"):
        new_voice.llm_temperature = config["temperature"]
    if config.get("maxTokens"):
        new_voice.llm_max_tokens = config["maxTokens"]
    
    new_config.voice = new_voice
    new_agent = create_agent_fn(new_config, vad, agent._memory, skip_greeting=True)
    state.update_agent(session, new_agent)


async def update_soul(
    session: "AgentSession",
    agent: Any,
    config: Dict[str, Any],
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
        logger.info(f"Updated soul: {soul.name} - {(soul.personality or '')[:50]}...")


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

        # Rebuild the full tool list (built-in + freshly registered client tools)
        combined = agent.client_tools.create_client_tools()
        agent._tools = combined
        logger.info(f"update_tools: registered {len(tools)} client tools on running agent")
    except Exception as e:
        log_error(logger, "update_tools: failed to register client tools", e)


async def update_memory(
    agent: Any,
    config: Dict[str, Any],
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
