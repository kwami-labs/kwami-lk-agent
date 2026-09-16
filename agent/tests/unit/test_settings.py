"""Settings is the single place this agent reads the environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.settings import (
    DEFAULT_KWAMI_API_TIMEOUT,
    DEFAULT_KWAMI_API_URL,
    Settings,
    get_settings,
    set_settings,
)


def test_defaults_are_safe_without_any_environment() -> None:
    """A bare process must not crash, and must not claim credentials it lacks."""
    settings = Settings.from_env()

    assert settings.kwami_api_url == DEFAULT_KWAMI_API_URL
    assert settings.kwami_api_key == ""
    assert settings.kwami_api_timeout == DEFAULT_KWAMI_API_TIMEOUT
    assert settings.memory_enabled is False
    assert settings.allow_browser_js is False


def test_memory_is_enabled_exactly_when_a_zep_key_is_present(env_setting) -> None:
    env_setting("ZEP_API_KEY", "zep-test-key")
    assert Settings.from_env().memory_enabled is True

    env_setting("ZEP_API_KEY", None)
    assert Settings.from_env().memory_enabled is False


def test_config_no_longer_depends_on_the_developer_environment(env_setting) -> None:
    """The bug this replaced: KwamiConfig() read ZEP_API_KEY at construction.

    `test_default_config` failed on any machine that happened to have the key
    exported, and its own comment admitted the fragility instead of fixing it.
    """
    from src.config import KwamiConfig

    env_setting("ZEP_API_KEY", None)
    assert KwamiConfig().memory.enabled is False

    env_setting("ZEP_API_KEY", "zep-test-key")
    assert KwamiConfig().memory.enabled is True


def test_trailing_slash_is_stripped_from_the_api_url(env_setting) -> None:
    """Call sites build paths as f"{url}/credits/...", so a trailing slash doubles up."""
    env_setting("KWAMI_API_URL", "https://api.example.com/")
    assert Settings.from_env().kwami_api_url == "https://api.example.com"


@pytest.mark.parametrize(
    "raw,expected",
    [("5", 5.0), ("0.5", 0.5), ("", DEFAULT_KWAMI_API_TIMEOUT), ("abc", DEFAULT_KWAMI_API_TIMEOUT)],
)
def test_timeout_falls_back_on_unparseable_values(env_setting, raw: str, expected: float) -> None:
    """A typo in the environment must not become a crash at session start."""
    env_setting("KWAMI_API_TIMEOUT", raw or None)
    assert Settings.from_env().kwami_api_timeout == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("no", False)],
)
def test_browser_js_flag_parsing(env_setting, raw: str, expected: bool) -> None:
    env_setting("KWAMI_ALLOW_BROWSER_JS", raw)
    assert Settings.from_env().allow_browser_js is expected


def test_provider_keys_record_only_what_is_present(env_setting) -> None:
    env_setting("OPENAI_API_KEY", "sk-test")
    env_setting("GROQ_API_KEY", None)

    settings = Settings.from_env()

    assert settings.has_provider_key("OPENAI_API_KEY") is True
    assert settings.has_provider_key("GROQ_API_KEY") is False
    assert "GROQ_API_KEY" not in settings.provider_keys


def test_describe_never_leaks_a_secret(env_setting) -> None:
    """describe() is written to the log at startup."""
    env_setting("ZEP_API_KEY", "super-secret-value")
    env_setting("OPENAI_API_KEY", "sk-super-secret")

    described = Settings.from_env().describe()

    assert "super-secret-value" not in str(described)
    assert "sk-super-secret" not in str(described)
    assert described["zep_api_key"] == "set"
    assert described["OPENAI_API_KEY"] == "set"


def test_settings_are_memoised_and_replaceable() -> None:
    explicit = Settings(kwami_api_url="https://explicit.example")
    set_settings(explicit)

    assert get_settings() is explicit

    set_settings(None)
    assert get_settings() is not explicit


def test_settings_are_frozen() -> None:
    """A credential must not change under a session that already read it."""
    with pytest.raises(Exception):
        Settings().kwami_api_key = "mutated"  # type: ignore[misc]


def test_every_documented_variable_is_read_by_settings() -> None:
    """.env.sample is the contract; Settings must honour all of it.

    Excludes the provider credentials the livekit plugins read for themselves,
    which Settings mirrors rather than owns.
    """
    sample = (Path(__file__).parent.parent.parent.parent / ".env.sample").read_text()
    documented = {
        line.split("=", 1)[0].strip()
        for line in sample.splitlines()
        if "=" in line and not line.strip().startswith("#")
    }

    source = (Path(__file__).parent.parent.parent / "src" / "settings.py").read_text()
    plugin_owned = {"LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"}

    missing = sorted(name for name in documented - plugin_owned if name not in source)
    assert not missing, f".env.sample documents variables Settings never reads: {missing}"
