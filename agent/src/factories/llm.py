"""LLM factory.

Every provider advertised in ``LLMProviders.ALL`` must either construct or fall
back with an explicit log line. Raising here is not an option: the only caller
is ``create_agent_from_config``, whose exception handler merely logs, leaving
the session on the placeholder agent created at startup. The user experiences
that as an agent that joins the room and never speaks, with nothing in the logs
connecting the two.
"""

import re

from livekit.agents import inference
from livekit.plugins import openai

from ..domain import KwamiVoiceConfig
from ..settings import get_settings
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix
from .optional import optional_plugin

logger = get_logger("llm")

# Each of these is a separate `livekit-plugins-*` distribution, declared as an
# optional extra. When one is absent the provider falls back to OpenAI with a
# warning rather than raising -- see the module docstring.
google = optional_plugin("google")
anthropic = optional_plugin("anthropic")
groq = optional_plugin("groq")

# OpenAI models that only accept the default temperature; others support 0..2.
_OPENAI_TEMPERATURE_FIXED_MODELS = ("gpt-5.1", "o1-", "o3-")

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
#: The provider name for models served by LiveKit Inference ("google/...").
INFERENCE_PROVIDER = "inference"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

# Claude models that still take `temperature`. From Opus 4.7 on the API answers
# 400 "`temperature` is deprecated for this model" (measured 2026-10-18 across
# every model the account lists), so this is an allowlist of the old families:
# a model newer than this file must not inherit a parameter it rejects.
_ANTHROPIC_TEMPERATURE_MODELS = re.compile(r"^claude-(3|(opus|sonnet|haiku)-4(-[0-6])?(-\d{8})?$)")


def _anthropic_temperature(config: KwamiVoiceConfig, model: str) -> dict[str, float]:
    if _ANTHROPIC_TEMPERATURE_MODELS.match(model):
        return {"temperature": config.llm_temperature}
    return {}


def _openai_temperature(config: KwamiVoiceConfig, model: str) -> float:
    """Use temperature=1 for models that only support the default (avoids API 400).

    Matches on prefix, not substring: `in` matching forced temperature=1.0 on
    any model whose name merely contained "o1-", such as "llama-o1-8b".
    """
    if not model:
        return config.llm_temperature
    lower = model.lower()
    if any(lower.startswith(prefix) for prefix in _OPENAI_TEMPERATURE_FIXED_MODELS):
        return 1.0
    return config.llm_temperature


def _openai_llm(config: KwamiVoiceConfig, model: str, **kwargs):
    """Build an OpenAI-backed client.

    `max_completion_tokens` is the kwarg the plugin actually takes. The old code
    passed `max_tokens`, so the TypeError fallback fired on every single call
    and the configured token limit silently did nothing.
    """
    return openai.LLM(
        model=model,
        temperature=_openai_temperature(config, model),
        max_completion_tokens=config.llm_max_tokens,
        **kwargs,
    )


def _fallback_to_openai(config: KwamiVoiceConfig, provider: str, reason: str):
    logger.warning(
        "LLM provider '%s' unavailable (%s); falling back to OpenAI %s. "
        "The configured provider is NOT in use.",
        provider,
        reason,
        DEFAULT_OPENAI_MODEL,
    )
    return _openai_llm(config, DEFAULT_OPENAI_MODEL)


def _build(config: KwamiVoiceConfig, provider: str, model: str):
    """Construct the requested provider, or return None if it is unavailable."""
    if provider == INFERENCE_PROVIDER:
        # LiveKit Inference: "google/gemini-2.5-flash", billed to the LiveKit
        # project instead of a provider key -- so a free-tier Google quota
        # does not apply (measured 0.69 s to the first token, 2026-10-20).
        if not model or "/" not in model:
            return None
        return inference.LLM(model=model, extra_kwargs={"temperature": config.llm_temperature})

    if provider == "openai":
        return _openai_llm(config, model or DEFAULT_OPENAI_MODEL)

    if provider == "google":
        if google is None:
            return None
        model = model or "gemini-2.0-flash"
        # The voice LLM talks and delegates (brain/); deep reasoning is the
        # brain's job. Gemini 3 thinks by default, which measured 7.3 s for a
        # one-sentence spoken reply (2026-10-09), so it is turned down to the
        # minimum here. Older models take a token budget instead of a level.
        thinking: dict[str, str] = (
            {"thinking_level": "minimal"} if model.startswith("gemini-3") else {}
        )
        return google.LLM(
            model=model,
            temperature=config.llm_temperature,
            **({"thinking_config": thinking} if thinking else {}),
        )

    if provider == "anthropic":
        # Anthropic is NOT reachable through openai.LLM -- there is no
        # `with_anthropic` helper. It needs the livekit-plugins-anthropic
        # package, installed via the `anthropic` extra.
        if anthropic is None:
            return None
        model = model or DEFAULT_ANTHROPIC_MODEL
        # The plugin marks every tool strict by default, and Anthropic accepts
        # at most 20 strict tools: with the agent's 44 every request was a 400
        # ("Too many strict tools") on LiveKit Cloud, 2026-10-18.
        return anthropic.LLM(
            model=model,
            _strict_tool_schema=False,
            **_anthropic_temperature(config, model),
        )

    if provider == "groq":
        # Likewise: livekit-plugins-groq, via the `groq` extra. There is no
        # `openai.LLM.with_groq`.
        if groq is None:
            return None
        # Groq retired every Llama 3 model; gpt-oss is what it serves now
        # (listed 2026-10-18).
        return groq.LLM(
            model=model or "openai/gpt-oss-120b",
            temperature=config.llm_temperature,
        )

    if provider == "deepseek":
        return openai.LLM.with_deepseek(
            model=model or "deepseek-chat",
            temperature=config.llm_temperature,
        )

    if provider == "mistral":
        # Mistral speaks the OpenAI protocol but has no dedicated helper.
        # This used to go through `with_x_ai(base_url=...)`, which routed to
        # the right endpoint but read XAI_API_KEY -- so a deployment that
        # correctly set MISTRAL_API_KEY failed with "XAI API key is required".
        api_key = get_settings().mistral_api_key
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is required for the Mistral provider")
        return _openai_llm(
            config,
            model or "mistral-large-latest",
            base_url=MISTRAL_BASE_URL,
            api_key=api_key,
        )

    if provider == "cerebras":
        return openai.LLM.with_cerebras(
            model=model or "llama3.1-70b",
            temperature=config.llm_temperature,
        )

    if provider == "ollama":
        return openai.LLM.with_ollama(
            model=model or "llama3.2",
            temperature=config.llm_temperature,
        )

    return None


def create_llm(config: KwamiVoiceConfig):
    """Create an LLM instance from configuration, never raising."""
    provider = (config.llm_provider or "").lower()

    # Strip provider prefix if present (e.g. "openai/gpt-4.1-mini" -> "gpt-4.1-mini")
    model = strip_model_prefix(config.llm_model or "", provider)

    try:
        instance = _build(config, provider, model)
    except Exception as e:
        return _fallback_to_openai(config, provider, f"{type(e).__name__}: {e}")

    if instance is None:
        reason = "plugin not installed" if provider else "no provider configured"
        return _fallback_to_openai(config, provider or "<unset>", reason)

    return instance
