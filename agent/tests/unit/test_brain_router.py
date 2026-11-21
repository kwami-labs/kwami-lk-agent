"""Routing edge cases: spoken names, manual mode, Tavily writers and the council."""

from __future__ import annotations

from src.brain.config import BrainConfig, BrainModel, default_brain
from src.brain.router import (
    ACCOUNT_MARK,
    council_members,
    find_model,
    pick,
    should_convene,
    tavily_writer,
)

ANTHROPIC = {"ANTHROPIC_API_KEY": "a"}
ALL_KEYS = {"ANTHROPIC_API_KEY": "a", "GOOGLE_API_KEY": "g", "OPENAI_API_KEY": "o"}


def test_find_model_by_exact_name_substring_or_not_at_all() -> None:
    models = default_brain().models
    assert find_model(models, "  ") is None
    assert find_model(models, "claude-opus-5-5").id == "claude-opus"
    assert find_model(models, "gemini").id == "gemini-flash"
    assert find_model(models, "llama") is None


def test_manual_routing_sends_everything_but_code_to_the_default() -> None:
    config = default_brain()
    config.routing = "manual"
    config.default_model = "opus"
    assert pick(config, ALL_KEYS, "question", "simple").id == "claude-opus"
    assert pick(config, ALL_KEYS, "code", "simple").id == "claude-code"


def test_pick_returns_none_when_no_model_is_usable() -> None:
    assert pick(default_brain(), {}, "question", "normal") is None


def test_tavily_writer_prefers_the_role_then_any_writer() -> None:
    config = default_brain()
    assert tavily_writer(config, ANTHROPIC, "research", "hard").id == "claude-opus"
    assert tavily_writer(config, {"GOOGLE_API_KEY": "g"}, "research", "hard") is None

    only_code_role = BrainConfig(
        models=[BrainModel(id="odd", provider="anthropic", model="m", role="code")]
    )
    assert tavily_writer(only_code_role, ANTHROPIC, "question", "normal").id == "odd"


def test_a_council_of_one_is_filled_from_other_providers() -> None:
    config = BrainConfig(
        models=[
            BrainModel(id="sonnet", provider="anthropic", model="s", role="standard", council=True),
            BrainModel(id="opus", provider="anthropic", model="o", role="expert"),
            BrainModel(id="flash", provider="google", model="f", role="search"),
            BrainModel(id="gpt", provider="openai", model="g", role="fast"),
            BrainModel(id="gpt-big", provider="openai", model="gb", role="expert"),
        ]
    )
    members = [m.id for m in council_members(config, ALL_KEYS)]
    assert members[0] == "sonnet"
    assert len(members) == 3
    assert len({m.provider for m in council_members(config, ALL_KEYS)}) == 3


def test_council_policies() -> None:
    config = default_brain()
    assert should_convene(config, "question", important=False, requested=True)
    config.council = "research"
    assert should_convene(config, "research", important=False, requested=False)
    assert not should_convene(config, "question", important=False, requested=False)
    config.council = "on_request"
    assert not should_convene(config, "research", important=True, requested=False)


# -- cheapest first --------------------------------------------------------------------


def _two_standards(access: str = "auto") -> BrainConfig:
    from src.brain.config import BrainModel

    config = default_brain()
    config.models = [
        BrainModel("sonnet", "anthropic", "claude-sonnet-5-5", "standard", "Sonnet", access=access),
        BrainModel("deepseek", "deepseek", "deepseek-chat", "standard", "DeepSeek"),
        BrainModel("mistral", "mistral", "mistral-large", "standard", "Mistral"),
    ]
    return config


def test_of_two_models_with_the_same_role_the_cheaper_one_works() -> None:
    keys = {"ANTHROPIC_API_KEY": "a", "DEEPSEEK_API_KEY": "d", "MISTRAL_API_KEY": "m"}
    # DeepSeek 0.5 + 2.0 beats Mistral 1 + 5 and Sonnet 2 + 10, whatever the order.
    assert pick(_two_standards(), keys, "analysis", "normal").id == "deepseek"  # type: ignore[union-attr]


def test_a_claude_model_on_the_pcs_account_costs_nothing_per_token() -> None:
    keys = {"ANTHROPIC_API_KEY": "a", "DEEPSEEK_API_KEY": "d", ACCOUNT_MARK: "1"}
    assert pick(_two_standards(), keys, "analysis", "normal").id == "sonnet"  # type: ignore[union-attr]
    # Unless the user keeps it on the API key.
    api_only = _two_standards(access="api")
    assert pick(api_only, keys, "analysis", "normal").id == "deepseek"  # type: ignore[union-attr]


def test_a_tie_keeps_the_users_order() -> None:
    from src.brain.config import BrainModel
    from src.brain.router import by_cost

    a = BrainModel("a", "groq", "model-a", "fast")
    b = BrainModel("b", "groq", "model-b", "fast")
    assert by_cost([a, b], {}) == [a, b]
    assert by_cost([b, a], {}) == [b, a]


def test_each_kind_of_work_goes_to_the_model_best_at_it() -> None:
    """The user's rule (2026-11-13): Opus most, Sonnet writes and reads, Haiku chores."""
    from src.brain.router import KINDS, role_for

    assert "reading" in KINDS
    table = {
        (kind, difficulty): role_for(kind, difficulty)
        for kind in ("question", "analysis", "writing", "reading", "research")
        for difficulty in ("simple", "normal", "hard")
    }
    assert table == {
        ("question", "simple"): "fast",
        ("question", "normal"): "expert",
        ("question", "hard"): "expert",
        ("analysis", "simple"): "fast",
        ("analysis", "normal"): "expert",
        ("analysis", "hard"): "expert",
        ("writing", "simple"): "standard",
        ("writing", "normal"): "standard",
        ("writing", "hard"): "standard",
        ("reading", "simple"): "standard",
        ("reading", "normal"): "standard",
        ("reading", "hard"): "standard",
        ("research", "simple"): "search",
        ("research", "normal"): "expert",
        ("research", "hard"): "expert",
    }
