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

#: Which cloud-browser vendor backs the navigation panel. Browserbase is the
#: default because its Contexts persist the user's logins across sessions,
#: which is what makes "carry on where I left off" work; "browser_use" keeps
#: the original Browser Use Cloud path available unchanged.
DEFAULT_BROWSER_PROVIDER = "browserbase"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Provider credentials the livekit plugins read themselves, mirrored onto
#: `Settings.provider_keys` so that "is this provider usable?" is answerable
#: without touching os.environ.
PROVIDER_KEY_NAMES: tuple[str, ...] = (
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
    "MISTRAL_API_KEY",
    # LiveKit Inference (Rime, the cloud turn detector) signs with these.
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
)

#: Every environment variable `from_env` reads, in one enumerable place.
#:
#: Exported because two things outside this module have to mirror it, and both
#: silently diverged while the list was only implicit in `from_env`'s body:
#:
#: * The test suite clears these so an exported credential cannot change a test
#:   outcome. It missed five, which made `test_js_execution_is_refused_by_default`
#:   -- the browser JS security default -- pass or fail depending on the
#:   developer's shell.
#: * The Cloudflare Worker forwards them into the container
#:   (`infra/src/env.ts`). It missed the Browserbase credentials, so the default
#:   browser vendor was unreachable on that deploy target.
#:
#: Both now derive from this tuple rather than restating it, and
#: `tests/unit/test_settings_env_inventory.py` fails if it drifts from the code.
ENV_VAR_NAMES: tuple[str, ...] = (
    "KWAMI_API_URL",
    "KWAMI_API_KEY",
    "KWAMI_API_TIMEOUT",
    "KWAMI_LOG_FORMAT",
    "ENVIRONMENT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_SERVICE_NAME",
    "ZEP_API_KEY",
    "TAVILY_API_KEY",
    "SERPAPI_KEY",
    "KWAMI_BROWSER_PROVIDER",
    "BROWSER_USE_API_KEY",
    "BROWSERBASE_API_KEY",
    "BROWSERBASE_PROJECT_ID",
    "KWAMI_ALLOW_BROWSER_JS",
    "SUPABASE_URL",
    "SUPABASE_SECRET_KEY",
    "KWAMI_STORAGE",
    "KWAMI_EXECUTOR_MODE",
    *PROVIDER_KEY_NAMES,
)


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
    #: Whether KWAMI_API_URL was given at all. `kwami_api_url` always holds a
    #: value (the localhost default), so this is what tells "talk to the API on
    #: localhost" apart from "there is no API": on LiveKit Cloud there is none,
    #: and the agent writes to Supabase instead. True for a directly built
    #: Settings, which carries the URL it was given.
    kwami_api_url_set: bool = True

    # The user's own Supabase, written directly when there is no Kwami API.
    # The secret (service-role) key bypasses RLS, so it lives only in the
    # agent's secrets, never in the app.
    supabase_url: str = ""
    supabase_secret_key: str = ""
    #: Where the agent keeps what outlives a session. "auto" (default): the
    #: Kwami API when it is configured, Supabase directly otherwise. "supabase":
    #: Supabase directly even with KWAMI_API_URL set -- what LiveKit Cloud
    #: needs, since its secrets can be added but not removed one by one and an
    #: old KWAMI_API_URL is there. "api": never Supabase directly.
    storage: str = "auto"

    #: "json" turns on one-JSON-object-per-line output with the session
    #: correlation fields at the top level. Anything else is human-readable.
    log_format: str = ""

    #: Which deployment this is, for the trace resource and the logs.
    environment: str = ""

    # Tracing. The standard OTEL_* names on purpose: every collector and
    # hosted backend already understands them, so picking one is a deployment
    # decision rather than a code change. Empty endpoint means tracing is off.
    otel_endpoint: str = ""
    otel_headers: str = ""
    otel_service_name: str = ""

    # Memory
    zep_api_key: str = ""

    # Search and enrichment
    tavily_api_key: str = ""
    serpapi_key: str = ""

    # Cloud browsing
    browser_provider: str = DEFAULT_BROWSER_PROVIDER
    browser_use_api_key: str = ""
    browserbase_api_key: str = ""
    browserbase_project_id: str = ""
    allow_browser_js: bool = False

    # LLM providers reached through an OpenAI-compatible endpoint, which need
    # their key passed explicitly rather than picked up by a plugin helper.
    mistral_api_key: str = ""

    #: Where the brain's code executor works. "local": the user's
    #: Documents/Jarvis. "cloud": a temporary folder whose results are uploaded
    #: to Storage. "auto" (default): cloud when there is no Documents folder,
    #: which is the case in every container and never on the user's PC.
    executor_mode: str = "auto"

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
            kwami_api_url_set=bool(_env_str("KWAMI_API_URL")),
            supabase_url=_env_str("SUPABASE_URL").rstrip("/"),
            supabase_secret_key=_env_str("SUPABASE_SECRET_KEY"),
            storage=_env_str("KWAMI_STORAGE", "auto").strip().lower(),
            log_format=_env_str("KWAMI_LOG_FORMAT").strip().lower(),
            environment=_env_str("ENVIRONMENT"),
            otel_endpoint=_env_str("OTEL_EXPORTER_OTLP_ENDPOINT"),
            otel_headers=_env_str("OTEL_EXPORTER_OTLP_HEADERS"),
            otel_service_name=_env_str("OTEL_SERVICE_NAME"),
            zep_api_key=_env_str("ZEP_API_KEY"),
            tavily_api_key=_env_str("TAVILY_API_KEY"),
            serpapi_key=_env_str("SERPAPI_KEY"),
            browser_provider=_env_str("KWAMI_BROWSER_PROVIDER", DEFAULT_BROWSER_PROVIDER)
            .strip()
            .lower(),
            browser_use_api_key=_env_str("BROWSER_USE_API_KEY"),
            browserbase_api_key=_env_str("BROWSERBASE_API_KEY"),
            browserbase_project_id=_env_str("BROWSERBASE_PROJECT_ID"),
            allow_browser_js=_env_bool("KWAMI_ALLOW_BROWSER_JS"),
            mistral_api_key=_env_str("MISTRAL_API_KEY"),
            executor_mode=_env_str("KWAMI_EXECUTOR_MODE", "auto").strip().lower(),
            provider_keys={
                name: _env_str(name) for name in PROVIDER_KEY_NAMES if os.environ.get(name)
            },
        )

    @property
    def memory_enabled(self) -> bool:
        """Memory is on exactly when a Zep credential is available."""
        return bool(self.zep_api_key)

    @property
    def kwami_api_enabled(self) -> bool:
        """The Kwami API is configured: an explicit URL and the internal key."""
        return bool(self.kwami_api_key and self.kwami_api_url and self.kwami_api_url_set)

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.supabase_url and self.supabase_secret_key)

    @property
    def supabase_direct(self) -> bool:
        """Write straight to Supabase: it is configured and there is no API.

        By default the API wins when both are present, because it is what
        bills credits and what the app's local setup already goes through.
        `storage` overrides that either way.
        """
        if self.storage == "api" or not self.supabase_enabled:
            return False
        return self.storage == "supabase" or not self.kwami_api_enabled

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
