"""What the brain is made of: which models, in which roles, under which rules.

The voice model only listens and speaks. Everything that takes work -- a
search, an analysis, a piece of code -- crosses the bridge to one of these
models. The roster is the user's: any number of models, each with one role, and
a flag saying whether it sits on the council that is consulted in parallel for
important decisions.

The app sends this as a `config_update` of type ``brain``. Parsing is tolerant:
an unknown role, provider or option falls back to the default rather than
dropping the whole message, the same rule `handlers/config_handler.py` follows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Provider = Literal["anthropic", "google", "openai", "local"]
Role = Literal["fast", "standard", "expert", "search", "code"]

PROVIDERS: frozenset[str] = frozenset({"anthropic", "google", "openai", "local"})
ROLES: frozenset[str] = frozenset({"fast", "standard", "expert", "search", "code"})

#: The only place the local executor may work. A folder sent by the app must be
#: inside it: the config arrives over the data channel, and "work in C:\\" must
#: not be one message away.
EXECUTOR_ROOT = Path.home() / "Documents" / "Jarvis"


def executor_in_cloud(mode: str = "auto") -> bool:
    """Whether the executor works in a throwaway folder and uploads its results.

    On LiveKit Cloud the worker is a container: there is no Documents\\Jarvis,
    nothing on its disk outlives the job, and the user's PC is out of reach.
    "auto" decides by the one thing every container lacks and every desktop
    has, a Documents folder; KWAMI_EXECUTOR_MODE=local|cloud overrides it.
    """
    if mode == "cloud":
        return True
    if mode == "local":
        return False
    return not EXECUTOR_ROOT.parent.is_dir()


@dataclass
class BrainModel:
    id: str
    provider: Provider
    model: str
    role: Role
    label: str = ""
    enabled: bool = True
    #: Consulted in parallel when the council meets.
    council: bool = False


@dataclass
class ExecutorConfig:
    folder: str = str(EXECUTOR_ROOT)
    #: full_no_push: edits and runs freely, but never pushes, never touches
    #: `.env` and never leaves the folder. edit_only: edits but runs nothing.
    #: read_only: reads and proposes.
    permissions: Literal["full_no_push", "edit_only", "read_only"] = "full_no_push"
    max_turns: int = 40
    #: Hard ceiling per task, enforced by the executor itself.
    max_task_usd: float = 2.0


@dataclass
class BrainConfig:
    enabled: bool = True
    #: auto: the bridge picks by difficulty. manual: everything but code goes to
    #: `default_model`.
    routing: Literal["auto", "manual"] = "auto"
    default_model: str = ""
    #: important: the council meets for money, security and hard decisions.
    #: on_request: only when the user asks. research: for every research task.
    council: Literal["important", "on_request", "research"] = "important"
    #: native: each model's own search. tavily: Tavily only. both: native, plus
    #: Tavily as an extra tool when its key is set.
    search: Literal["native", "tavily", "both"] = "both"
    monthly_budget_usd: float = 50.0
    #: inform: counts and reports, never cuts. downgrade: cheap models past the
    #: budget. stop: no delegation past the budget.
    budget_policy: Literal["inform", "downgrade", "stop"] = "inform"
    #: next_pause: says it when the user stops talking. interrupt: at once.
    #: on_request: only says it is ready.
    delivery: Literal["next_pause", "interrupt", "on_request"] = "next_pause"
    #: finish: a task running at hang-up completes and waits for the next
    #: greeting. cancel: it dies with the call.
    on_hangup: Literal["finish", "cancel"] = "finish"
    #: Split normal and hard tasks into parts, each on the cheapest model that
    #: can do it (brain/planner.py).
    split: bool = True
    models: list[BrainModel] = field(default_factory=list)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)


def default_models() -> list[BrainModel]:
    """The roster the user chose on 2026-10-09."""
    return [
        BrainModel("claude-haiku", "anthropic", "claude-haiku-5-5", "fast", "Claude Haiku 5.5"),
        BrainModel(
            "claude-sonnet",
            "anthropic",
            "claude-sonnet-5-5",
            "standard",
            "Claude Sonnet 5.5",
            council=True,
        ),
        BrainModel(
            "claude-opus", "anthropic", "claude-opus-5-5", "expert", "Claude Opus 5.5", council=True
        ),
        BrainModel(
            "gemini-flash", "google", "gemini-3.5-flash", "search", "Gemini 3.5 Flash", council=True
        ),
        BrainModel("claude-code", "local", "claude-sonnet-5-5", "code", "Claude Code (local)"),
    ]


def default_brain() -> BrainConfig:
    return BrainConfig(models=default_models())


def _choice(value: Any, allowed: tuple[str, ...], default: str) -> Any:
    return value if isinstance(value, str) and value in allowed else default


def _number(value: Any, default: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return min(max(float(value), low), high)


def _parse_model(raw: Any) -> BrainModel | None:
    if not isinstance(raw, dict):
        return None
    provider = raw.get("provider")
    role = raw.get("role")
    model = raw.get("model")
    if provider not in PROVIDERS or role not in ROLES:
        return None
    if not isinstance(model, str) or not model.strip():
        return None
    # Stripped before it doubles as the id: " gpt-5.2 " must not become an id
    # that no later lookup by "gpt-5.2" can find.
    model = model.strip()
    model_id = raw.get("id") if isinstance(raw.get("id"), str) and raw.get("id") else model
    raw_label = raw.get("label")
    label = raw_label if isinstance(raw_label, str) else ""
    return BrainModel(
        id=str(model_id)[:64],
        provider=provider,
        model=model[:96],
        role=role,
        label=label[:64],
        enabled=raw.get("enabled", True) is not False,
        council=raw.get("council") is True,
    )


def resolve_executor_folder(folder: str) -> Path:
    """The folder the executor may use: the requested one if it is inside the
    root, the root otherwise."""
    root = EXECUTOR_ROOT.resolve()
    try:
        candidate = Path(folder).expanduser().resolve() if folder else root
    except (OSError, ValueError):
        return root
    if candidate == root or root in candidate.parents:
        return candidate
    return root


def parse_brain(data: Any) -> BrainConfig:
    """Build a BrainConfig from the app's camelCase payload."""
    config = default_brain()
    if not isinstance(data, dict):
        return config

    config.enabled = data.get("enabled", True) is not False
    config.routing = _choice(data.get("routing"), ("auto", "manual"), config.routing)
    if isinstance(data.get("defaultModel"), str):
        config.default_model = data["defaultModel"][:64]
    config.council = _choice(
        data.get("council"), ("important", "on_request", "research"), config.council
    )
    config.search = _choice(data.get("search"), ("native", "tavily", "both"), config.search)
    config.monthly_budget_usd = _number(
        data.get("monthlyBudgetUsd"), config.monthly_budget_usd, 0.0, 10_000.0
    )
    config.budget_policy = _choice(
        data.get("budgetPolicy"), ("inform", "downgrade", "stop"), config.budget_policy
    )
    config.delivery = _choice(
        data.get("delivery"), ("next_pause", "interrupt", "on_request"), config.delivery
    )
    config.on_hangup = _choice(data.get("onHangup"), ("finish", "cancel"), config.on_hangup)
    config.split = data.get("splitTasks", True) is not False

    raw_models = data.get("models")
    if isinstance(raw_models, list):
        parsed = [m for m in (_parse_model(r) for r in raw_models[:32]) if m is not None]
        seen: set[str] = set()
        config.models = []
        for model in parsed:
            if model.id in seen:
                continue
            seen.add(model.id)
            config.models.append(model)

    executor = data.get("executor")
    if isinstance(executor, dict):
        raw_folder = executor.get("folder")
        folder = raw_folder if isinstance(raw_folder, str) else ""
        config.executor.folder = str(resolve_executor_folder(folder))
        config.executor.permissions = _choice(
            executor.get("permissions"),
            ("full_no_push", "edit_only", "read_only"),
            config.executor.permissions,
        )
        config.executor.max_turns = int(
            _number(executor.get("maxTurns"), config.executor.max_turns, 1, 200)
        )
        config.executor.max_task_usd = _number(
            executor.get("maxTaskUsd"), config.executor.max_task_usd, 0.05, 50.0
        )
    return config
