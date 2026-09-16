"""LLM factory.

Every provider advertised in ``LLMProviders.ALL`` must either construct or fall
back with an explicit log line. Raising here is not an option: the only caller
is ``create_agent_from_config``, whose exception handler merely logs, leaving
the session on the placeholder agent created at startup. The user experiences
that as an agent that joins the room and never speaks, with nothing in the logs
connecting the two.
"""

import os

from livekit.plugins import openai

from ..config import KwamiVoiceConfig
from ..utils.logging import get_logger
from ..utils.provider import strip_model_prefix

logger = get_logger("llm")

# Each of these is a separate `livekit-plugins-*` distribution, declared as an
# optional extra. When one is absent the provider falls back to OpenAI with a
# warning rather than raising -- see the module docstring.
try:
    from livekit.plugins import google
except ImportError:
    google = None  # type: ignore[assignment]

try:
    from livekit.plugins import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

try:
    from livekit.plugins import groq
except ImportError:
    groq = None  # type: ignore[assignment]

# OpenAI models that only accept the default temperature; others support 0..2.
_OPENAI_TEMPERATURE_FIXED_MODELS = ("gpt-5.1", "o1-", "o3-")

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


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
    if provider == "openai":
        return _openai_llm(config, model or DEFAULT_OPENAI_MODEL)

    if provider == "google":
        if google is None:
            return None
        return google.LLM(
            model=model or "gemini-2.0-flash",
            temperature=config.llm_temperature,
        )

    if provider == "anthropic":
        # Anthropic is NOT reachable through openai.LLM -- there is no
        # `with_anthropic` helper. It needs the livekit-plugins-anthropic
        # package, installed via the `anthropic` extra.
        if anthropic is None:
            return None
        return anthropic.LLM(
            model=model or "claude-3-5-sonnet-latest",
            temperature=config.llm_temperature,
        )

    if provider == "groq":
        # Likewise: livekit-plugins-groq, via the `groq` extra. There is no
        # `openai.LLM.with_groq`.
        if groq is None:
            return None
        return groq.LLM(
            model=model or "llama-3.1-70b-versatile",
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
        api_key = os.environ.get("MISTRAL_API_KEY", "")
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
