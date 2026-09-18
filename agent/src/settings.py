"""Every environment variable this agent reads, in one injectable object.

Credentials used to be read in eleven places: module constants captured at
import time (`runtime_bootstrap`, `usage/reporter`), `default_factory` lambdas
on config dataclasses, and bare `os.environ.get` calls inside tool bodies. That
had three costs:

* **Tests were machine-dependent.** `KwamiConfig().memory.enabled` read
  ZEP_API_KEY at construction, so the suite passed or failed depending on what
  the developer had exported.
* **Import-time capture could not be overridden.** A module that snapshots
  `KWAMI_API_URL` at import cannot be re-pointed without reloading it, so any
  entry point that imported before `load_dotenv` silently reported no billing.
* **There was no seam.** A tool reaching into `os.environ` mid-call cannot be
  handed a fake.

`Settings.from_env()` is called once per process and passed down. Tests build a
`Settings(...)` directly and never touch the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields

DEFAULT_KWAMI_API_URL = "http://localhost:8080"
DEFAULT_KWAMI_API_TIMEOUT = 30.0

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _env_str(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


@dataclass(frozen=True)
class Settings:
    """Process-wide configuration resolved from the environment.

    Frozen so a value cannot drift mid-session: everything downstream should be
    able to assume the credential it was handed at startup is the one still in
    effect.
    """

    # Kwami platform API (runtime config + credit reporting)
    kwami_api_url: str = DEFAULT_KWAMI_API_URL
    kwami_api_key: str = ""
    kwami_api_timeout: float = DEFAULT_KWAMI_API_TIMEOUT

    # Memory
    zep_api_key: str = ""

    # Search and enrichment
    tavily_api_key: str = ""
    serpapi_key: str = ""

    # Cloud browsing
    browser_use_api_key: str = ""
    allow_browser_js: bool = False

    # LLM providers reached through an OpenAI-compatible endpoint, which need
    # their key passed explicitly rather than picked up by a plugin helper.
    mistral_api_key: str = ""

    # Provider credentials the livekit plugins read themselves. Mirrored here so
    # that "is this provider usable?" is answerable without touching os.environ.
    provider_keys: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from the current environment.

        Call once, near process start and after `load_dotenv`.
        """
        return cls(
            kwami_api_url=_env_str("KWAMI_API_URL", DEFAULT_KWAMI_API_URL).rstrip("/"),
            kwami_api_key=_env_str("KWAMI_API_KEY"),
            kwami_api_timeout=_env_float("KWAMI_API_TIMEOUT", DEFAULT_KWAMI_API_TIMEOUT),
            zep_api_key=_env_str("ZEP_API_KEY"),
            tavily_api_key=_env_str("TAVILY_API_KEY"),
            serpapi_key=_env_str("SERPAPI_KEY"),
            browser_use_api_key=_env_str("BROWSER_USE_API_KEY"),
            allow_browser_js=_env_bool("KWAMI_ALLOW_BROWSER_JS"),
            mistral_api_key=_env_str("MISTRAL_API_KEY"),
            provider_keys={
                name: _env_str(name)
                for name in (
                    "OPENAI_API_KEY",
                    "DEEPGRAM_API_KEY",
                    "CARTESIA_API_KEY",
                    "ELEVEN_API_KEY",
                    "ELEVENLABS_API_KEY",
                    "ASSEMBLYAI_API_KEY",
                    "GOOGLE_API_KEY",
                    "GOOGLE_APPLICATION_CREDENTIALS",
                    "ANTHROPIC_API_KEY",
                    "GROQ_API_KEY",
                    "DEEPSEEK_API_KEY",
                    "CEREBRAS_API_KEY",
                    "XAI_API_KEY",
                )
                if os.environ.get(name)
            },
        )

    @property
    def memory_enabled(self) -> bool:
        """Memory is on exactly when a Zep credential is available."""
        return bool(self.zep_api_key)

    def has_provider_key(self, name: str) -> bool:
        return bool(self.provider_keys.get(name))

    def describe(self) -> dict[str, str]:
        """Redacted view for start-up logging: which credentials are present."""
        out: dict[str, str] = {}
        for f in fields(self):
            if f.name == "provider_keys":
                continue
            value = getattr(self, f.name)
            if f.name.endswith(("_key", "_api_key")):
                out[f.name] = "set" if value else "MISSING"
            else:
                out[f.name] = str(value)
        for name, value in sorted(self.provider_keys.items()):
            out[name] = "set" if value else "MISSING"
        return out


_settings: Settings | None = None


def get_settings() -> Settings:
    """The process-wide settings, resolved on first use.

    A transitional accessor for call sites that cannot yet be handed a Settings
    instance. New code should take `settings: Settings` as a parameter; this
    exists so the migration can proceed module by module without a flag day.
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def set_settings(settings: Settings | None) -> None:
    """Install settings explicitly. Pass None to force a re-read on next use."""
    global _settings
    _settings = settings
