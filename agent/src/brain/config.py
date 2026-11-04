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

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

Provider = Literal[
    "anthropic", "google", "openai", "xai", "openrouter", "groq", "deepseek", "mistral", "local"
]
Role = Literal["fast", "standard", "expert", "search", "x_search", "code"]
#: How a Claude model is reached. auto: the Claude account on the user's PC when
#: the work thread is there, the API key otherwise. account: never the API key.
#: api: never the account.
Access = Literal["auto", "account", "api"]

PROVIDERS: frozenset[str] = frozenset(
    {"anthropic", "google", "openai", "xai", "openrouter", "groq", "deepseek", "mistral", "local"}
)
ROLES: frozenset[str] = frozenset({"fast", "standard", "expert", "search", "x_search", "code"})
ACCESS_MODES: tuple[str, ...] = ("auto", "account", "api")

_THINKING: tuple[Role, ...] = ("fast", "standard", "expert")
#: What each provider can actually do. A model offered a job its provider has no
#: tool for would answer anyway, from memory: "search" needs a real web search and
#: "x_search" exists only on Grok. Mirrored in the app's lib/brain.ts.
PROVIDER_ROLES: dict[str, tuple[Role, ...]] = {
    "anthropic": (*_THINKING, "search"),
    "google": (*_THINKING, "search"),
    "openai": (*_THINKING, "search"),
    "xai": (*_THINKING, "search", "x_search"),
    "openrouter": _THINKING,
    "groq": _THINKING,
    "deepseek": _THINKING,
    "mistral": _THINKING,
    "local": ("code",),
}
#: Long enough for a real brief, short enough not to swamp every call's prompt.
MAX_INSTRUCTIONS_CHARS = 2000

#: The only place the local executor may work. A folder sent by the app must be
#: inside it: the config arrives over the data channel, and "work in C:\\" must
#: not be one message away.
EXECUTOR_ROOT = Path.home() / "Documents" / "Jarvis"

#: The user's monthly ceiling (2026-11-01: warns, then stops; raised from 100 to 150). In code, not
#: in the panel: the work thread can spend with nobody on the line.
HARD_MONTHLY_CAP_USD = 150.0
#: Where the warning comes before the stop.
BUDGET_WARNING_SHARE = 0.8


def budget_notice(before: float, after: float, cap: float = HARD_MONTHLY_CAP_USD) -> str:
    """What to tell the user when this spend crossed a line. Once per line."""
    if before < cap <= after:
        return (
            f"Aviso: se ha llegado al tope de {cap:.0f} dólares de este mes. No aceptaré más "
            "tareas hasta el mes que viene."
        )
    warning = cap * BUDGET_WARNING_SHARE
    if before < warning <= after:
        return f"Aviso: este mes ya van {after:.0f} de los {cap:.0f} dólares del tope."
    return ""


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
    #: Every job this model does; `role` is the first. Empty means just `role`.
    roles: list[Role] = field(default_factory=list)
    access: Access = "auto"
    #: The user's brief for this model, added to whatever it is asked.
    instructions: str = ""

    def __post_init__(self) -> None:
        if not self.roles:
            self.roles = [self.role]
        self.role = self.roles[0]

    def has_role(self, role: str) -> bool:
        return role in self.roles


@dataclass
class ExecutorConfig:
    folder: str = str(EXECUTOR_ROOT)
    #: Whose Chrome this run may use (`browser_tool`). Empty: none.
    browser_user_id: str = ""
    #: full_no_push: edits and runs freely, but never pushes, never touches
    #: `.env` and never leaves the folder. edit_only: edits but runs nothing.
    #: read_only: reads and proposes.
    #: think, think_web: answers only, with no tools or only the web's (`worker.py`).
    permissions: Literal["full_no_push", "edit_only", "read_only", "think", "think_web"] = (
        "full_no_push"
    )
    max_turns: int = 40
    #: Hard ceiling per task, enforced by the executor itself.
    max_task_usd: float = 2.0
    #: Whose Google accounts the run may use (`google_tool`). Empty: none.
    google_user_id: str = ""


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
    monthly_budget_usd: float = HARD_MONTHLY_CAP_USD
    #: inform: counts and reports. downgrade: cheap models past the budget. stop: no
    #: delegation past the budget. `HARD_MONTHLY_CAP_USD` stops it whatever this says.
    budget_policy: Literal["inform", "downgrade", "stop"] = "stop"
    #: next_pause: says it when the user stops talking. interrupt: at once.
    #: on_request: only says it is ready.
    delivery: Literal["next_pause", "interrupt", "on_request"] = "next_pause"
    #: finish: a task running at hang-up completes and waits for the next
    #: greeting. cancel: it dies with the call.
    on_hangup: Literal["finish", "cancel"] = "finish"
    #: Split normal and hard tasks into parts, each on the cheapest model that
    #: can do it (brain/planner.py).
    split: bool = True
    #: The model that plans and hands out the work. Empty: the router's choice.
    director: str = ""
    #: everything: the voice answers only small talk and passes the rest to the
    #: brain, so another model does the thinking. work_only: the voice also answers
    #: what it knows.
    relay: Literal["everything", "work_only"] = "everything"
    #: The user's brief for the whole brain, added to every worker.
    instructions: str = ""
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
    # Over the data channel anything can arrive, a list included, which `in` would refuse.
    if not isinstance(provider, str) or provider not in PROVIDERS:
        return None
    if not isinstance(model, str) or not model.strip():
        return None
    allowed = PROVIDER_ROLES[provider]
    raw_roles = raw.get("roles")
    wanted = [*(raw_roles if isinstance(raw_roles, list) else []), role]
    roles = [r for r in dict.fromkeys(w for w in wanted if isinstance(w, str)) if r in allowed]
    if not roles:
        # No job this provider can do: a saved role it never had, or none at all.
        if not raw_roles and (not isinstance(role, str) or role not in ROLES):
            return None
        roles = [allowed[0]]
    # Stripped before it doubles as the id: " gpt-5.2 " must not become an id
    # that no later lookup by "gpt-5.2" can find.
    model = model.strip()
    model_id = raw.get("id") if isinstance(raw.get("id"), str) and raw.get("id") else model
    raw_label = raw.get("label")
    label = raw_label if isinstance(raw_label, str) else ""
    raw_instructions = raw.get("instructions")
    return BrainModel(
        id=str(model_id)[:64],
        provider=cast(Provider, provider),
        model=model[:96],
        role=roles[0],
        label=label[:64],
        enabled=raw.get("enabled", True) is not False,
        council=raw.get("council") is True,
        roles=roles,
        access=_choice(raw.get("access"), ACCESS_MODES, "auto"),
        instructions=(raw_instructions if isinstance(raw_instructions, str) else "").strip()[
            :MAX_INSTRUCTIONS_CHARS
        ],
    )


def extra_executor_folders() -> list[Path]:
    """Folders of the PC the user opened to the executor, besides the Jarvis folder.

    KWAMI_EXECUTOR_FOLDERS, separated by `;`. Empty by default, and only a variable on
    the PC itself: nothing a call or a job carries can widen this.
    """
    raw = os.environ.get("KWAMI_EXECUTOR_FOLDERS", "")
    return [Path(item.strip()).expanduser().resolve() for item in raw.split(";") if item.strip()]


def shut_executor_folders() -> list[Path]:
    """Folders that stay shut even inside an opened one: KWAMI_EXECUTOR_SHUT, as above."""
    raw = os.environ.get("KWAMI_EXECUTOR_SHUT", "")
    return [Path(item.strip()).expanduser().resolve() for item in raw.split(";") if item.strip()]


def resolve_executor_folder(folder: str) -> Path:
    """The folder the executor may use: the requested one if it is inside the
    root or a folder the user opened, the root otherwise."""
    root = EXECUTOR_ROOT.resolve()
    try:
        candidate = Path(folder).expanduser().resolve() if folder else root
    except (OSError, ValueError):
        return root
    shut = shut_executor_folders()
    if any(candidate == place or place in candidate.parents for place in shut):
        return root
    for allowed in (root, *extra_executor_folders()):
        if candidate == allowed or allowed in candidate.parents:
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
    if isinstance(data.get("director"), str):
        config.director = data["director"][:64]
    config.relay = _choice(data.get("relay"), ("everything", "work_only"), config.relay)
    if isinstance(data.get("instructions"), str):
        config.instructions = data["instructions"].strip()[:MAX_INSTRUCTIONS_CHARS]

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
