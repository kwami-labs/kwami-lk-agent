"""Tools that let the agent change its own model, voice and pipeline, mid-conversation.

Everything needed to rebuild a pipeline already existed, but it was reachable
only from a data message the *frontend* sent. Asked "switch to Claude" or "use
your Cedar voice", the agent could do nothing but describe which settings panel
to open -- while the prompt has claimed, for as long as it has existed, that
"you can change your voice or the AI model being used if the user requests it".

These tools close that gap, with three rules that are the whole point of the
module:

* **Prefer the live path.** A realtime voice or speed change is an
  `update_options` push over the open socket. Rebuilding for it would drop the
  connection and re-greet, which the user hears.
* **Hand off, never self-swap.** When a rebuild is genuinely required, the new
  agent is *returned*. The framework installs it after this tool's output has
  been collected, so the sentence explaining the switch survives the switch.
  Calling `session.update_agent` from inside a tool discards the turn that
  called it.
* **Verify before confirming.** `create_llm` falls back to OpenAI for any
  provider it cannot build -- a missing plugin, a missing key -- and only logs a
  warning. Reporting success off the *request* rather than the *result* means
  telling the user they are on Claude while they are on gpt-4o-mini. Every
  switch here is checked against the object that was actually constructed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from livekit.agents import RunContext, function_tool

from ..constants import RealtimeVoices, resolve_realtime_voice
from ..domain.models import (
    PROVIDER_ALIASES,
    PROVIDER_DEFAULT_MODEL,
    REALTIME_PROVIDER_DEFAULT_MODEL,
    resolve_model,
)
from ..runtime.reconfigure import (
    REALTIME_PIPELINE,
    STANDARD_PIPELINE,
    reconfigurator_from_context,
)
from ..utils.logging import get_logger

logger = get_logger("pipeline_control")

NO_RECONFIGURE = (
    "I can't change my model or pipeline in this session -- "
    "the runtime didn't give me a way to rebuild myself."
)


def _built_model_name(agent: Any) -> str:
    """The model id of whatever was actually constructed for this agent."""
    llm = getattr(agent, "llm", None)
    return str(getattr(llm, "model", "") or "")


def _switch_took_effect(agent: Any, wanted_model: str) -> bool:
    """Whether the rebuilt agent is running the model that was asked for.

    Compared loosely on purpose: a provider may canonicalise or date-stamp what
    it was given (`claude-3-5-sonnet-latest`), so equality would report a false
    failure. The case this must catch is the silent fallback, where the built
    model is a completely different family from the requested one.
    """
    if not wanted_model:
        return True
    built = _built_model_name(agent).lower()
    if not built:
        # Realtime models and some plugins do not expose `.model`; there is
        # nothing to check, and claiming failure would be its own lie.
        return True
    wanted = wanted_model.lower()
    return wanted in built or built in wanted


if TYPE_CHECKING:
    from ..domain import KwamiConfig


class PipelineControlMixin:
    """Function tools for changing the running voice pipeline.

    Mixed into `KwamiAgent` alongside `AgentToolsMixin`. Kept separate because
    everything here reaches through `RunContext` into the session runtime,
    whereas the builtin tools only touch the agent itself.

    The annotations below are the contract with `KwamiAgent`, which is what
    supplies them. Declared rather than described in prose: `AgentToolsMixin`
    documents the same dependency in a docstring and sits on mypy's ignore list
    as a result, so nothing checks that the attributes it reaches for exist.
    A bare annotation creates no class attribute, so it shadows nothing.
    """

    kwami_config: KwamiConfig

    # -- model ---------------------------------------------------------------

    @function_tool()
    async def change_ai_model(self, context: RunContext, model: str) -> Any:
        """Switch to a different AI model or provider, mid-conversation.

        The conversation so far is carried across, so nothing is forgotten.

        Args:
            model: What the user asked for. A provider name ("Claude", "Gemini",
                   "GPT", "Groq", "Mistral"), a model id ("gpt-4.1-mini",
                   "claude-3-5-haiku-latest"), or both ("Claude Sonnet").
        """
        reconfigurator = reconfigurator_from_context(context)
        if reconfigurator is None:
            return NO_RECONFIGURE

        voice = self.kwami_config.voice
        realtime = voice.pipeline_type == REALTIME_PIPELINE
        current_provider = voice.realtime_provider if realtime else voice.llm_provider

        choice = resolve_model(model, current_provider, realtime=realtime)
        if not choice.model and not choice.recognised:
            return f"I couldn't work out which model '{model}' is. Which provider is it?"

        if realtime:
            changes = {
                "realtime_provider": choice.provider,
                "realtime_model": choice.model,
            }
            # A voice from the old provider does not exist on the new one, and a
            # realtime session handed an unknown voice fails at connect -- which
            # on this pipeline is silence, not a fallback.
            if choice.provider != voice.realtime_provider:
                changes["realtime_voice"] = RealtimeVoices.DEFAULTS.get(choice.provider, "")
        else:
            changes = {"llm_provider": choice.provider, "llm_model": choice.model}

        new_voice = reconfigurator.mutated_voice(self, **changes)
        new_agent = reconfigurator.build(new_voice, agent=self)

        if not realtime and not _switch_took_effect(new_agent, choice.model):
            built = _built_model_name(new_agent)
            logger.warning(
                "Requested %s/%s but the pipeline built %s", choice.provider, choice.model, built
            )
            # Do NOT hand off: staying put is better than moving the user onto a
            # model they did not ask for while telling them otherwise.
            return (
                f"I couldn't switch to {choice.provider} -- that provider isn't available "
                f"in this deployment, so I've stayed on {_built_model_name(self) or 'my current model'}."
            )

        label = choice.model or choice.provider
        logger.info("Agent-initiated model switch: %s/%s", choice.provider, choice.model)
        return new_agent, f"Switched to {label}. I've kept everything we've talked about."

    @function_tool()
    async def list_available_models(self, context: RunContext) -> dict[str, Any]:
        """List the AI model providers this agent can switch between, and the current one."""
        voice = self.kwami_config.voice
        realtime = voice.pipeline_type == REALTIME_PIPELINE
        defaults = REALTIME_PROVIDER_DEFAULT_MODEL if realtime else PROVIDER_DEFAULT_MODEL
        return {
            "pipeline": voice.pipeline_type,
            "current_provider": voice.realtime_provider if realtime else voice.llm_provider,
            "current_model": voice.realtime_model if realtime else voice.llm_model,
            "providers": sorted(defaults),
            "note": (
                "Any model id can be requested directly; these are the providers "
                "with a known default. On the realtime pipeline only speech-to-speech "
                "models are usable."
            ),
        }

    # -- pipeline ------------------------------------------------------------

    @function_tool()
    async def switch_pipeline_mode(self, context: RunContext, mode: str) -> Any:
        """Switch between the standard and realtime voice pipelines mid-conversation.

        Args:
            mode: 'realtime' for the low-latency speech-to-speech model (one model
                  hears and speaks), or 'standard' for the separate
                  speech-to-text, LLM and text-to-speech chain.
        """
        reconfigurator = reconfigurator_from_context(context)
        if reconfigurator is None:
            return NO_RECONFIGURE

        # Spoken forms first, then the wire spellings, so the tool and the data
        # channel cannot drift apart on what "realtime" means.
        from ..handlers.realtime import normalize_pipeline_type

        spoken = (mode or "").strip().lower()
        target = {
            "real time": REALTIME_PIPELINE,
            "live": REALTIME_PIPELINE,
            "speech to speech": REALTIME_PIPELINE,
            "fast": REALTIME_PIPELINE,
            "pipeline": STANDARD_PIPELINE,
            "stt": STANDARD_PIPELINE,
        }.get(spoken) or normalize_pipeline_type(spoken)
        if target is None:
            return f"I don't know a '{mode}' mode. I can use 'standard' or 'realtime'."

        if target == self.kwami_config.voice.pipeline_type:
            return f"I'm already on the {target} pipeline."

        new_voice = reconfigurator.mutated_voice(self, pipeline_type=target)
        new_agent = reconfigurator.build(new_voice, agent=self)
        logger.info("Agent-initiated pipeline switch to %s", target)
        if target == REALTIME_PIPELINE:
            return new_agent, "Switched to the realtime pipeline -- I should feel snappier now."
        return new_agent, "Switched back to the standard pipeline."

    # -- realtime voice ------------------------------------------------------

    @function_tool()
    async def change_realtime_voice(self, context: RunContext, voice_name: str) -> Any:
        """Change the voice of the realtime (speech-to-speech) model, mid-sentence.

        Only meaningful on the realtime pipeline; use change_voice on the
        standard one.

        Args:
            voice_name: OpenAI realtime: alloy, ash, ballad, cedar, coral, echo,
                        marin, sage, shimmer, verse. Gemini Live: Aoede, Charon,
                        Fenrir, Kore, Leda, Orus, Puck, Zephyr.
        """
        voice = self.kwami_config.voice
        if voice.pipeline_type != REALTIME_PIPELINE:
            return (
                "I'm on the standard pipeline right now, so that's a text-to-speech "
                "voice -- use change_voice instead, or ask me to switch to realtime first."
            )

        provider = voice.realtime_provider or "openai"
        resolved = resolve_realtime_voice(provider, voice_name)
        if resolved is None:
            available = sorted(RealtimeVoices.BY_PROVIDER.get(provider, ()))
            return (
                f"'{voice_name}' isn't a {provider} realtime voice. "
                f"I can use: {', '.join(available)}."
            )

        if resolved == voice.realtime_voice:
            return f"I'm already using {resolved}."

        # Live first: this is the whole reason the realtime pipeline is worth
        # having, and a rebuild here would be audible.
        from ..handlers.realtime import apply_live_realtime_options

        previous = voice.realtime_voice
        voice.realtime_voice = resolved
        if apply_live_realtime_options(self, {"realtime_voice"}):
            logger.info("Agent-initiated realtime voice change to %s", resolved)
            return f"Switched to the {resolved} voice."

        voice.realtime_voice = previous
        reconfigurator = reconfigurator_from_context(context)
        if reconfigurator is None:
            return NO_RECONFIGURE

        new_voice = reconfigurator.mutated_voice(self, realtime_voice=resolved)
        new_agent = reconfigurator.build(new_voice, agent=self)
        return new_agent, f"Switched to the {resolved} voice."

    @function_tool()
    async def list_available_voices(self, context: RunContext) -> dict[str, Any]:
        """List the voices available on the pipeline currently in use."""
        voice = self.kwami_config.voice
        if voice.pipeline_type == REALTIME_PIPELINE:
            provider = voice.realtime_provider or "openai"
            return {
                "pipeline": REALTIME_PIPELINE,
                "provider": provider,
                "current_voice": voice.realtime_voice,
                "voices": sorted(RealtimeVoices.BY_PROVIDER.get(provider, ())),
            }
        from ..constants import CartesiaVoices, DeepgramVoices, OpenAIVoices

        by_provider = {
            "openai": sorted(OpenAIVoices.STANDARD),
            "deepgram": sorted(DeepgramVoices.ALL),
            "cartesia": sorted(CartesiaVoices.NAME_MAP),
        }
        return {
            "pipeline": STANDARD_PIPELINE,
            "provider": voice.tts_provider,
            "current_voice": voice.tts_voice,
            "voices": by_provider.get(voice.tts_provider, []),
            "note": "ElevenLabs and Rime voices are chosen by id from the voice panel.",
        }

    # -- introspection -------------------------------------------------------

    @function_tool()
    async def get_pipeline_status(self, context: RunContext) -> dict[str, Any]:
        """Report exactly which models and voice are running right now.

        Reads the constructed objects, not the requested config, so a provider
        that silently fell back is visible rather than hidden.
        """
        voice = self.kwami_config.voice
        status: dict[str, Any] = {
            "pipeline": voice.pipeline_type,
            "requested": {
                "llm": f"{voice.llm_provider}/{voice.llm_model}",
                "stt": f"{voice.stt_provider}/{voice.stt_model}",
                "tts": f"{voice.tts_provider}/{voice.tts_model}",
                "realtime": f"{voice.realtime_provider}/{voice.realtime_model}",
                "voice": (
                    voice.realtime_voice
                    if voice.pipeline_type == REALTIME_PIPELINE
                    else voice.tts_voice
                ),
            },
            "running": {
                name: type(obj).__module__.replace("livekit.plugins.", "")
                for name in ("llm", "stt", "tts")
                if (obj := getattr(self, name, None)) is not None
            },
            "can_reconfigure": reconfigurator_from_context(context) is not None,
        }
        built = _built_model_name(self)
        if built:
            status["running"]["model"] = built
        return status

    @function_tool()
    async def list_model_providers(self, context: RunContext) -> list[str]:
        """List every provider name this agent recognises, including spoken aliases."""
        return sorted(set(PROVIDER_ALIASES))
