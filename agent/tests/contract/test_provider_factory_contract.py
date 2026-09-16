"""Contract tests: every provider this codebase advertises must actually work.

`LLMProviders.ALL` is the menu the frontend is allowed to choose from. A menu
entry that raises -- or that silently serves something else -- is worse than no
entry at all, because `create_agent_from_config` catches the failure and leaves
the caller talking to a mute placeholder agent.
"""

from __future__ import annotations

import inspect

import pytest
from livekit.plugins import openai

from src.config import KwamiVoiceConfig
from src.constants import LLMProviders
from src.factories.llm import create_llm


@pytest.mark.parametrize("provider", sorted(LLMProviders.ALL))
def test_every_advertised_llm_provider_constructs(
    provider: str, fake_key, caplog: pytest.LogCaptureFixture
) -> None:
    """No advertised provider may raise out of the factory.

    Falling back is acceptable; crashing is not, because the caller turns the
    crash into a silent agent. `openai.LLM` has no `with_anthropic` or
    `with_groq` helper -- those live in separate `livekit-plugins-*` packages
    that are declared as optional extras and are not installed.
    """
    fake_key(
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "CEREBRAS_API_KEY",
        "XAI_API_KEY",
    )
    config = KwamiVoiceConfig(llm_provider=provider)

    try:
        instance = create_llm(config)
    except Exception as exc:  # noqa: BLE001 - the point of the test
        pytest.fail(f"create_llm({provider!r}) raised {type(exc).__name__}: {exc}")

    assert instance is not None


@pytest.mark.parametrize("provider", sorted(LLMProviders.ALL))
def test_silent_provider_substitution_is_logged(
    provider: str, fake_key, caplog: pytest.LogCaptureFixture
) -> None:
    """If a provider falls back to another, say so.

    `google` and any unknown provider currently return OpenAI `gpt-4o-mini`
    with no log line at all, so a misconfigured Kwami is indistinguishable
    from a working one.
    """
    fake_key(
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "CEREBRAS_API_KEY",
        "XAI_API_KEY",
    )
    config = KwamiVoiceConfig(llm_provider=provider)

    with caplog.at_level("WARNING"):
        try:
            instance = create_llm(config)
        except Exception:  # noqa: BLE001 - covered by the test above
            pytest.skip(f"{provider} raises; see test_every_advertised_llm_provider_constructs")

    served_by_openai = type(instance).__module__.startswith("livekit.plugins.openai")
    is_openai_family = provider in {"openai", "deepseek", "cerebras", "ollama", "mistral"}
    if served_by_openai and not is_openai_family:
        assert caplog.records, (
            f"{provider!r} was silently served by livekit.plugins.openai with no warning logged"
        )


def test_openai_llm_accepts_the_max_tokens_kwarg_the_factory_passes() -> None:
    """Pins the kwarg name the token limit actually rides on.

    The factory passes `max_tokens=`; 1.3.12 takes `max_completion_tokens`, so
    the `except TypeError` fallback fires on every single call and the whole
    `maxTokens` config option silently does nothing.
    """
    params = inspect.signature(openai.LLM.__init__).parameters
    assert "max_completion_tokens" in params

    import src.factories.llm as llm_factory

    source = inspect.getsource(llm_factory)
    assert "max_completion_tokens" in source, (
        "the LLM factory still passes `max_tokens=`, which openai.LLM rejects; "
        "the TypeError fallback silently drops the configured token limit"
    )
    assert "max_tokens=" not in source, "a bare max_tokens= kwarg remains in the LLM factory"


def test_llm_max_tokens_reaches_the_client(fake_key) -> None:
    """End-to-end on the config option, not just the signature."""
    fake_key("OPENAI_API_KEY")
    config = KwamiVoiceConfig(llm_provider="openai", llm_model="gpt-4o-mini", llm_max_tokens=512)

    instance = create_llm(config)

    opts = getattr(instance, "_opts", None)
    assert opts is not None, "openai.LLM no longer exposes _opts; update this contract test"
    assert getattr(opts, "max_completion_tokens", None) == 512, (
        "llm_max_tokens never reached the LLM client"
    )


def test_openai_temperature_override_matches_model_prefixes_not_substrings() -> None:
    """`in` matching forces temperature=1.0 on any model containing 'o1-'."""
    from src.factories.llm import _openai_temperature

    config = KwamiVoiceConfig(llm_temperature=0.2)
    # A fixed-temperature model must be forced to 1.0 ...
    assert _openai_temperature(config, "o1-preview") == 1.0
    # ... but an unrelated model that merely contains the substring must not be.
    assert _openai_temperature(config, "llama-o1-8b") == 0.2


def test_mistral_reads_its_own_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mistral must authenticate with MISTRAL_API_KEY.

    The factory routes Mistral through `openai.LLM.with_x_ai(base_url=...)`.
    The endpoint override is correct -- requests do reach api.mistral.ai -- but
    the credential is not: `with_x_ai` reads `XAI_API_KEY`, so a correctly
    configured deployment fails with "XAI API key is required".
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key-not-real")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    config = KwamiVoiceConfig(llm_provider="mistral", llm_model="mistral-large-latest")

    try:
        instance = create_llm(config)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"MISTRAL_API_KEY alone is not enough to build a Mistral LLM: {exc}")

    client = getattr(instance, "_client", None)
    assert client is not None
    assert "mistral" in str(client.base_url), f"Mistral requests would go to {client.base_url}"
