"""Resolving a spoken model name to a provider and a model id.

The settings UI hands the backend exact ids (`openai/gpt-4.1-mini`). Speech
does not: a user says "switch to Claude", "use Gemini Flash", "put me on the
fastest one". Both have to land on the same `(provider, model)` pair, or the
agent will cheerfully announce a switch that `create_llm` then silently
reverses -- its fallback path returns OpenAI for any provider it cannot build,
logging a warning the user never sees.

So this module is deliberately small and total:

* it recognises *families*, not a frozen catalogue, because a model list goes
  stale in weeks and a wrong "I can't do that" is worse than passing a name
  through;
* an unrecognised name is returned unchanged with the current provider, so a
  new model works the day it ships;
* and it never decides whether a provider is *usable* -- that needs credentials
  and installed plugins, so it lives in `factories`, next to the thing that
  would otherwise fall back in silence.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Provider defaults, used when the user names a provider but not a model
#: ("switch to Claude"). Kept to the current flagship-for-voice of each family:
#: these sit in the conversational latency band, which a reasoning model does not.
PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "google": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "deepseek": "deepseek-chat",
    "mistral": "mistral-large-latest",
    "cerebras": "llama3.1-70b",
    "ollama": "llama3.2",
}

#: What people call each provider out loud. Matched as whole words against the
#: lowercased request, longest alias first.
PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "open ai": "openai",
    "gpt": "openai",
    "chatgpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "sonnet": "anthropic",
    "haiku": "anthropic",
    "opus": "anthropic",
    "google": "google",
    "gemini": "google",
    "groq": "groq",
    "llama": "groq",
    "deepseek": "deepseek",
    "deep seek": "deepseek",
    "mistral": "mistral",
    "mixtral": "mistral",
    "cerebras": "cerebras",
    "ollama": "ollama",
    "local": "ollama",
}

#: Realtime (speech-to-speech) providers and their defaults. A realtime session
#: cannot run an ordinary chat model, so "switch to Claude" while on the
#: realtime pipeline has to be answered, not silently mistranslated.
REALTIME_PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "openai": "gpt-realtime",
    "google": "gemini-2.0-flash-exp",
}

REALTIME_PROVIDER_ALIASES: dict[str, str] = {
    "openai": "openai",
    "open ai": "openai",
    "gpt": "openai",
    "gpt realtime": "openai",
    "chatgpt": "openai",
    "google": "google",
    "gemini": "google",
    "gemini live": "google",
}


@dataclass(frozen=True)
class ModelChoice:
    """A resolved model request.

    `recognised` refers to the *provider family*, not the exact model id: a
    brand-new model on a known provider is recognised and passed through on
    trust, because refusing it would cap the agent at the models that existed
    when this table was written. False means no family was identified at all
    and the current provider was kept as a guess -- which the caller should
    surface rather than reporting a confident switch.
    """

    provider: str
    model: str
    recognised: bool


def _match_provider(request: str, aliases: dict[str, str]) -> str | None:
    """Longest alias that appears in `request`, or None.

    Longest-first matters: "open ai" must not be decided by the "ai" of another
    alias, and "gemini live" must beat "gemini".
    """
    for alias in sorted(aliases, key=len, reverse=True):
        if alias in request:
            return aliases[alias]
    return None


def _looks_like_a_model_id(request: str) -> bool:
    """True for things that are already ids rather than spoken names.

    Model ids carry a digit and a separator (`gpt-4.1-mini`, `llama3.1-70b`,
    `claude-3-5-sonnet-latest`). A bare "claude" does not, and must fall through
    to the provider default instead of being sent as a model name.
    """
    return any(ch.isdigit() for ch in request) and ("-" in request or "." in request)


def resolve_model(
    request: str,
    current_provider: str,
    *,
    realtime: bool = False,
) -> ModelChoice:
    """Turn what the user said into a `(provider, model)` pair.

    Args:
        request: A provider name, a model id, or both ("Claude Sonnet",
            "anthropic/claude-3-5-haiku-latest", "gpt-4.1-mini").
        current_provider: Used when the request names a model but no provider.
        realtime: Resolve against the speech-to-speech families instead.
    """
    aliases = REALTIME_PROVIDER_ALIASES if realtime else PROVIDER_ALIASES
    defaults = REALTIME_PROVIDER_DEFAULT_MODEL if realtime else PROVIDER_DEFAULT_MODEL

    raw = (request or "").strip()
    normalized = raw.lower()
    if not normalized:
        return ModelChoice(current_provider, "", recognised=False)

    # An explicit "provider/model" is unambiguous; take it at its word.
    if "/" in raw:
        prefix, _, suffix = raw.partition("/")
        provider = aliases.get(prefix.strip().lower(), prefix.strip().lower())
        return ModelChoice(provider, suffix.strip(), recognised=provider in defaults)

    matched = _match_provider(normalized, aliases)

    if matched is None:
        # No family recognised: keep the current provider and pass the name
        # through, so a model released after this table was written still works.
        return ModelChoice(current_provider, raw, recognised=False)

    if _looks_like_a_model_id(normalized):
        return ModelChoice(matched, raw, recognised=True)

    return ModelChoice(matched, defaults.get(matched, ""), recognised=True)
