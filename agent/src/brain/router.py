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
other       fast        standard    expert
==========  ==========  ==========  ==========

Each role falls back to its neighbours when it has no usable model, nearest
first, so a roster with only Sonnet still answers everything.
"""

from __future__ import annotations

from collections.abc import Mapping

from .config import BrainConfig, BrainModel

PROVIDER_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    # The executor bills to the Anthropic key through the Agent SDK.
    "local": "ANTHROPIC_API_KEY",
}

ROLE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "fast": ("fast", "standard", "search", "expert"),
    "standard": ("standard", "expert", "fast", "search"),
    "expert": ("expert", "standard", "fast", "search"),
    "search": ("search", "standard", "fast", "expert"),
    "code": ("code",),
}

KINDS = ("research", "code", "analysis", "writing", "question")
DIFFICULTIES = ("simple", "normal", "hard")


def provider_available(provider: str, keys: Mapping[str, str]) -> bool:
    name = PROVIDER_KEYS.get(provider)
    return bool(name and keys.get(name))


def usable(config: BrainConfig, keys: Mapping[str, str]) -> list[BrainModel]:
    return [m for m in config.models if m.enabled and provider_available(m.provider, keys)]


def can_delegate(config: BrainConfig, keys: Mapping[str, str]) -> bool:
    """Whether the brain can take a search: switched on, with a remote model.

    The local executor does not count: it writes code, it does not search.
    """
    return config.enabled and any(m.provider != "local" for m in usable(config, keys))


def role_for(kind: str, difficulty: str) -> str:
    if kind == "code":
        return "code"
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
    if prefer:
        chosen = find_model(candidates, prefer)
        if chosen is not None and (chosen.provider == "local") == wants_code:
            return chosen
    if config.routing == "manual" and config.default_model and not wants_code:
        chosen = find_model(candidates, config.default_model)
        if chosen is not None and chosen.provider != "local":
            return chosen
    for role in ROLE_FALLBACKS[role_for(kind, difficulty)]:
        for model in candidates:
            if model.role == role and (model.provider == "local") == wants_code:
                return model
    return None


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
            if model.role == role:
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
    if kind == "code":
        return False
    if requested:
        return True
    if config.council == "important":
        return important
    if config.council == "research":
        return kind == "research" or important
    return False
