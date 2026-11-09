"""Which model gets a task.

The voice model describes the task (kind, difficulty, importance); this module
turns that into a model from the roster. It never invents one: a model is
usable only when it is enabled *and* its provider's key is present, so a GPT
entry added before OPENAI_API_KEY exists is skipped rather than failing.

Routing table (auto mode):

==========  ==========  ==========  ==========
kind        simple      normal      hard
==========  ==========  ==========  ==========
research    search      standard    expert
code        code        code        code
x_search    x_search    x_search    x_search
other       fast        standard    expert
==========  ==========  ==========  ==========

Each role falls back to its neighbours when it has no usable model, nearest
first, so a roster with only Sonnet still answers everything. Searching X has no
neighbours: nothing else can read it, and a model that cannot would make it up.

A model can hold several roles. Claude models are reached through the Claude
account on the user's PC or the API key, as each one's `access` says; the bridge
puts `ACCOUNT_MARK` among the keys while the PC's work thread is there.
"""

from __future__ import annotations

from collections.abc import Mapping

from .config import BrainConfig, BrainModel

PROVIDER_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    # The executor bills to the Anthropic key through the Agent SDK.
    "local": "ANTHROPIC_API_KEY",
}

#: Not an environment variable: set by the bridge while the PC's work thread is
#: there with a Claude login.
ACCOUNT_MARK = "@claude-account"
ACCOUNT_PROVIDERS = frozenset({"anthropic", "local"})
#: Who can look at the web when asked to.
WEB_PROVIDERS = frozenset({"anthropic", "google", "openai", "xai"})

ROLE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "fast": ("fast", "standard", "search", "expert"),
    "standard": ("standard", "expert", "fast", "search"),
    "expert": ("expert", "standard", "fast", "search"),
    "search": ("search", "standard", "fast", "expert"),
    "x_search": ("x_search",),
    "code": ("code",),
}

KINDS = ("research", "x_search", "code", "analysis", "writing", "question")
DIFFICULTIES = ("simple", "normal", "hard")


def provider_available(provider: str, keys: Mapping[str, str]) -> bool:
    name = PROVIDER_KEYS.get(provider)
    if provider in ACCOUNT_PROVIDERS and keys.get(ACCOUNT_MARK):
        return True
    return bool(name and keys.get(name))


def model_available(model: BrainModel, keys: Mapping[str, str]) -> bool:
    """Whether this model can run now, by the way the user lets it be reached."""
    if model.provider not in ACCOUNT_PROVIDERS:
        return provider_available(model.provider, keys)
    has_key = bool(keys.get(PROVIDER_KEYS[model.provider]))
    has_account = bool(keys.get(ACCOUNT_MARK))
    if model.access == "api":
        return has_key
    if model.access == "account":
        return has_account
    return has_key or has_account


def usable(config: BrainConfig, keys: Mapping[str, str]) -> list[BrainModel]:
    return [m for m in config.models if m.enabled and model_available(m, keys)]


def can_delegate(config: BrainConfig, keys: Mapping[str, str]) -> bool:
    """Whether the brain can take a search: switched on, with a remote model.

    The local executor does not count: it writes code, it does not search.
    """
    return config.enabled and any(m.provider != "local" for m in usable(config, keys))


def role_for(kind: str, difficulty: str) -> str:
    if kind in ("code", "x_search"):
        return kind
    if kind == "research":
        return {"simple": "search", "hard": "expert"}.get(difficulty, "standard")
    return {"simple": "fast", "hard": "expert"}.get(difficulty, "standard")


def find_model(models: list[BrainModel], wanted: str) -> BrainModel | None:
    """Match a spoken name ('Opus', 'gemini', 'gpt') against the roster."""
    needle = wanted.strip().lower()
    if not needle:
        return None
    for model in models:
        if needle in (model.id.lower(), model.model.lower(), model.label.lower()):
            return model
    for model in models:
        haystack = f"{model.id} {model.model} {model.label}".lower()
        if needle in haystack:
            return model
    return None


def pick(
    config: BrainConfig,
    keys: Mapping[str, str],
    kind: str,
    difficulty: str,
    prefer: str = "",
    exclude: frozenset[str] = frozenset(),
) -> BrainModel | None:
    candidates = [m for m in usable(config, keys) if m.id not in exclude]
    wants_code = kind == "code"
    if kind == "research":
        # A model with no web search would answer from memory, sounding just as sure.
        candidates = [m for m in candidates if m.provider in WEB_PROVIDERS] or candidates
    if prefer:
        chosen = find_model(candidates, prefer)
        if (
            chosen is not None
            and (chosen.provider == "local") == wants_code
            and (kind != "x_search" or chosen.has_role("x_search"))
        ):
            return chosen
    if config.routing == "manual" and config.default_model and kind not in ("code", "x_search"):
        chosen = find_model(candidates, config.default_model)
        if chosen is not None and chosen.provider != "local":
            return chosen
    for role in ROLE_FALLBACKS[role_for(kind, difficulty)]:
        for model in candidates:
            if model.has_role(role) and (model.provider == "local") == wants_code:
                return model
    return None


def director_of(config: BrainConfig, keys: Mapping[str, str]) -> BrainModel | None:
    """The model that plans the work: the user's choice, else the router's."""
    if config.director:
        for model in usable(config, keys):
            if model.id == config.director and model.provider != "local":
                return model
    return pick(config, keys, "analysis", "normal")


#: Who writes the answer when Gemini's own search is unavailable and Tavily
#: does the searching instead: models that can call a client-side tool.
TAVILY_WRITERS = ("anthropic", "openai")


def tavily_writer(
    config: BrainConfig,
    keys: Mapping[str, str],
    kind: str,
    difficulty: str,
    exclude: frozenset[str] = frozenset(),
) -> BrainModel | None:
    """The model that answers a search through Tavily when Gemini could not.

    Gemini searches with Google and nothing else, so when its quota is spent
    (429, seen on 2026-10-18) the search moves to Tavily, run on our side, and
    a Claude or GPT model of the role the task would have had writes it up.
    """
    candidates = [
        m for m in usable(config, keys) if m.id not in exclude and m.provider in TAVILY_WRITERS
    ]
    for role in ROLE_FALLBACKS[role_for(kind, difficulty)]:
        for model in candidates:
            if model.has_role(role):
                return model
    return candidates[0] if candidates else None


def council_members(config: BrainConfig, keys: Mapping[str, str]) -> list[BrainModel]:
    """The council, or the best stand-in for one.

    Members are the models flagged for it. With fewer than two, the most
    capable usable models from different providers are added, because a
    council of one is just a slower single answer.
    """
    remote = [m for m in usable(config, keys) if m.provider != "local"]
    members = [m for m in remote if m.council]
    if len(members) < 2:
        order = {"expert": 0, "standard": 1, "search": 2, "fast": 3}
        for model in sorted(remote, key=lambda m: order.get(m.role, 9)):
            if model not in members and all(model.provider != m.provider for m in members):
                members.append(model)
            if len(members) >= 3:
                break
    return members[:5]


def should_convene(config: BrainConfig, kind: str, important: bool, requested: bool) -> bool:
    # A council on X would be several models that cannot read it.
    if kind in ("code", "x_search"):
        return False
    if requested:
        return True
    if config.council == "important":
        return important
    if config.council == "research":
        return kind == "research" or important
    return False
