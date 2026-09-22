"""Turning what a user *says* into a provider and a model id.

The settings panel sends exact ids; speech does not. Both have to land on the
same `(provider, model)` pair, because `create_llm` answers an unbuildable
provider with a silent fallback to OpenAI -- so a mis-resolution is not an
error the user sees, it is an agent that says "you're on Claude now" and isn't.
"""

from __future__ import annotations

import pytest

from src.domain.models import PROVIDER_DEFAULT_MODEL, resolve_model


@pytest.mark.parametrize(
    ("spoken", "expected_provider"),
    [
        ("Claude", "anthropic"),
        ("claude", "anthropic"),
        ("switch to Claude please", "anthropic"),
        ("Sonnet", "anthropic"),
        ("Gemini", "google"),
        ("gemini flash", "google"),
        ("GPT", "openai"),
        ("ChatGPT", "openai"),
        ("open ai", "openai"),
        ("Groq", "groq"),
        ("Mistral", "mistral"),
        ("DeepSeek", "deepseek"),
        ("deep seek", "deepseek"),
        ("Cerebras", "cerebras"),
        ("Ollama", "ollama"),
    ],
)
def test_spoken_provider_names_resolve(spoken: str, expected_provider: str) -> None:
    choice = resolve_model(spoken, current_provider="openai")
    assert choice.provider == expected_provider
    assert choice.recognised is True
    assert choice.model == PROVIDER_DEFAULT_MODEL[expected_provider]


def test_a_bare_provider_name_does_not_become_a_model_name() -> None:
    """ "Claude" is a family, not a model. Sending it as one is a 404 at the API."""
    choice = resolve_model("Claude", current_provider="openai")
    assert choice.model != "Claude"
    assert choice.model.startswith("claude-")


def test_an_explicit_model_id_is_kept() -> None:
    choice = resolve_model("claude-3-5-haiku-latest", current_provider="openai")
    assert choice.provider == "anthropic"
    assert choice.model == "claude-3-5-haiku-latest"


def test_a_prefixed_id_is_split() -> None:
    choice = resolve_model("openai/gpt-4.1-mini", current_provider="anthropic")
    assert choice.provider == "openai"
    assert choice.model == "gpt-4.1-mini"


def test_a_future_model_on_a_known_provider_passes_through() -> None:
    """A model released after this table was written must still be usable.

    Refusing here would mean the agent can only ever run models that existed
    when it was built, which is a shorter half-life than the table has. The
    *family* is what is recognised; the exact id is taken on trust.
    """
    choice = resolve_model("gpt-6-turbo-ultra", current_provider="anthropic")
    assert choice.provider == "openai"
    assert choice.model == "gpt-6-turbo-ultra"
    assert choice.recognised is True


def test_an_unknown_family_stays_on_the_current_provider() -> None:
    """Nothing recognisable: keep the provider and flag it as a guess.

    `recognised=False` is what stops the caller announcing a confident switch
    to a provider it never actually identified.
    """
    choice = resolve_model("qwen-3-max", current_provider="openai")
    assert choice.provider == "openai"
    assert choice.model == "qwen-3-max"
    assert choice.recognised is False


def test_empty_request_is_not_a_switch() -> None:
    choice = resolve_model("", current_provider="anthropic")
    assert choice.provider == "anthropic"
    assert choice.model == ""
    assert choice.recognised is False


def test_longest_alias_wins() -> None:
    """ "gemini live" must not be decided by the "gemini" substring."""
    choice = resolve_model("gemini live", current_provider="openai", realtime=True)
    assert choice.provider == "google"


# -- realtime ---------------------------------------------------------------


def test_realtime_resolves_against_the_speech_to_speech_families() -> None:
    choice = resolve_model("gemini", current_provider="openai", realtime=True)
    assert choice.provider == "google"
    assert choice.model == "gemini-2.0-flash-exp"


def test_realtime_does_not_offer_a_chat_only_provider() -> None:
    """Anthropic has no realtime model; "Claude" must not resolve to one.

    Resolved as a realtime provider, the session would be built with a model
    name the provider rejects at connect -- which on the realtime pipeline is
    silence, not a fallback.
    """
    choice = resolve_model("Claude", current_provider="openai", realtime=True)
    assert choice.provider == "openai", "a chat-only family leaked into the realtime path"
    assert choice.recognised is False


def test_realtime_openai_default() -> None:
    choice = resolve_model("gpt", current_provider="google", realtime=True)
    assert choice.provider == "openai"
    assert choice.model == "gpt-realtime"
