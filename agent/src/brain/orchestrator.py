"""Claude directs, the other models work: the brain's orchestrator.

The planner (`planner.py`) splits a task once, up front, and cannot change its
mind: a step that comes back thin or contradicts another stays as it is. Here
the director -- Claude, Opus by default -- holds the task in one conversation
and hands pieces to the roster's other models through a `delegate` tool, as
many times as the work needs: a cheap model for a lookup or a summary, one with
web search for current facts, a second opinion where two answers disagree. It
reads each result and decides the next step, then writes the answer itself.

The director's own tokens are the expensive ones, so its brief is to think and
check, and to leave reading and drafting to the workers. Every worker call goes
through the bridge's own `ask`, with the same keys, rests and spend ledger as
any other.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger
from .config import BrainModel
from .ledger import PRICES, PROVIDER_DEFAULT_PRICES, anthropic_cost
from .planner import TEAM_BRIEF_CHARS
from .providers import _FALLBACK_MODELS, EFFORT_BY_DIFFICULTY, BrainAnswer

logger = get_logger("brain.orchestrator")

#: Director turns per task. Each turn may hand out several pieces at once.
MAX_ROUNDS = 8
#: Worker calls per task, whatever the director asks for.
MAX_DELEGATIONS = 10
#: How much of one worker's answer the director reads back.
MAX_RESULT_CHARS = 8000
#: How long one worker may take before the director hears it failed.
WORKER_SECONDS = 240.0

ORCHESTRATOR_BRIEF = """
You direct this task. A team of other models works for you through the `delegate`
tool; you hold the whole task and write the final answer yourself.

- Your tokens cost the most. Hand out the reading, searching, extracting, drafting
  and routine reasoning; keep for yourself the judgement, the checking and the final
  answer.
- Give each piece to the cheapest worker that can do it well; each one's line says
  what it is best at. A web lookup or an extraction goes to the cheapest worker that
  can do it (Haiku), with `web` true for the web, asking for facts and sources, not
  an essay. Drafting a text, or reading and summarising a long one, goes to Sonnet.
  Reasoning, deciding and checking are yours.
- If you can answer well yourself in the time it takes to brief a worker, do.
- Pieces that do not depend on each other go out in the same turn: they run at once.
- Each brief must stand on its own: workers have not seen this conversation.
- Read what comes back critically. When results disagree, or a figure decides the
  answer, check it with a second worker before relying on it.
- A simple task needs no team: answer it directly."""

#: What each family is best at: the user's rule (2026-11-13) and what measured well.
#: Matched on the model id, first match wins.
SPECIALTIES: tuple[tuple[str, str], ...] = (
    ("claude-haiku", "web lookups, extracting, classifying, reformatting, translating"),
    (
        "claude-sonnet",
        "writing and drafting, reading and summarising long text, everyday analysis",
    ),
    ("claude-opus", "hard reasoning, decisions, maths, checking others' work"),
    ("gemini", "quick web lookups grounded on Google"),
    ("grok", "what is said on X, and the web"),
    ("gpt", "general reasoning and web lookups"),
    ("deepseek", "cheap drafts, summaries and routine reasoning"),
    ("llama", "very fast short answers and reformatting"),
    ("mistral", "cheap drafts and summaries"),
)


def specialty(model: BrainModel) -> str:
    """What this worker is best at, by its family; its roles when the family is unknown."""
    name = model.model.lower()
    for family, text in SPECIALTIES:
        if family in name:
            return text
    return ", ".join(model.roles)


DelegateFn = Callable[[BrainModel, str, bool], Awaitable[BrainAnswer]]


@dataclass
class Delegation:
    worker: str
    goal: str
    ok: bool
    cost_usd: float


@dataclass
class Orchestrated:
    answer: BrainAnswer
    delegations: list[Delegation] = field(default_factory=list)


def _price(model: BrainModel) -> tuple[float, float]:
    return PRICES.get(model.model) or PROVIDER_DEFAULT_PRICES.get(model.provider, (5.0, 25.0))


def describe_workers(workers: list[BrainModel], searches: Callable[[BrainModel], bool]) -> str:
    """One line per worker, cheapest first: what the director chooses from."""
    lines = []
    for model in sorted(workers, key=lambda m: sum(_price(m))):
        rate_in, rate_out = _price(model)
        web = "searches the web" if searches(model) else "no web search"
        line = (
            f"- {model.id}: {model.label or model.model} ({model.provider}), best for "
            f"{specialty(model)}; {web}; about {rate_in:g} $ in and {rate_out:g} $ out "
            "per million tokens."
        )
        # What the user wrote for this model also decides who gets a piece.
        brief = " ".join(model.instructions.split())[:TEAM_BRIEF_CHARS]
        if brief:
            line += f" The user's instructions for it: {brief}"
        lines.append(line)
    return "\n".join(lines)


def delegate_tool(workers: list[BrainModel], searches: Callable[[BrainModel], bool]) -> dict:
    return {
        "name": "delegate",
        "description": (
            "Hand one piece of the task to a worker model and get its answer back. "
            "Workers, cheapest first:\n" + describe_workers(workers, searches)
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "worker": {"type": "string", "enum": [m.id for m in workers]},
                "instructions": {
                    "type": "string",
                    "description": "A self-contained brief: what to do and what to return.",
                },
                "web": {
                    "type": "boolean",
                    "description": "True when the piece needs current facts from the web.",
                },
            },
            "required": ["worker", "instructions", "web"],
            "additionalProperties": False,
        },
        "strict": True,
    }


async def orchestrate(
    api_key: str,
    director: BrainModel,
    workers: list[BrainModel],
    task: str,
    system: str,
    difficulty: str,
    delegate: DelegateFn,
    searches: Callable[[BrainModel], bool],
    budget_usd: float = 2.0,
) -> Orchestrated | None:
    """Run the task with the director and its team. None when the director could not
    be reached, so the caller can run the task another way."""
    import anthropic

    by_id = {m.id: m for m in workers}
    params: dict[str, Any] = {
        "model": director.model,
        "max_tokens": 16000,
        "system": f"{system}\n{ORCHESTRATOR_BRIEF}",
        "tools": [delegate_tool(workers, searches)],
        "output_config": {"effort": EFFORT_BY_DIFFICULTY.get(difficulty, "medium")},
        # Every turn resends the system, the tool and the conversation so far.
        "cache_control": {"type": "ephemeral"},
    }
    if director.model.startswith(_FALLBACK_MODELS):
        params["betas"] = ["server-side-fallback-2026-07-01"]
        params["fallbacks"] = "default"
    client = anthropic.AsyncAnthropic(api_key=api_key, timeout=600.0)
    create = client.beta.messages.create if "betas" in params else client.messages.create
    messages: list[Any] = [{"role": "user", "content": task}]
    delegations: list[Delegation] = []
    sources: list[str] = []
    cost = 0.0
    response: Any = None

    async def run(block: Any) -> dict[str, Any]:
        args = block.input if isinstance(block.input, dict) else {}
        worker = by_id.get(str(args.get("worker", "")))
        goal = str(args.get("instructions", "")).strip()
        if worker is None or not goal:
            return _result(block.id, "No such worker, or no instructions.", error=True)
        if len(delegations) >= MAX_DELEGATIONS:
            return _result(block.id, "No more pieces can be handed out: finish now.", error=True)
        # The same ceiling per task as the rest of the brain: a director that keeps
        # handing out pieces must not spend past it with nobody watching.
        if cost + sum(d.cost_usd for d in delegations) >= budget_usd:
            return _result(
                block.id, "This task's budget is spent: finish with what you have.", error=True
            )
        entry = Delegation(worker.id, goal[:160], ok=False, cost_usd=0.0)
        delegations.append(entry)
        try:
            answer = await asyncio.wait_for(
                delegate(worker, goal, bool(args.get("web"))), WORKER_SECONDS
            )
        except TimeoutError:
            return _result(block.id, f"{worker.id} gave no answer in time.", error=True)
        entry.ok, entry.cost_usd = answer.ok, answer.cost_usd
        sources.extend(answer.sources)
        text = answer.text[:MAX_RESULT_CHARS]
        return _result(block.id, text, error=not answer.ok)

    try:
        for _ in range(MAX_ROUNDS):
            response = await create(messages=messages, **params)
            cost += anthropic_cost(director.model, response.usage)
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                break
            results = await asyncio.gather(*(run(block) for block in calls))
            messages.append({"role": "assistant", "content": response.content})
            # Every result of the turn in one message, as the API expects.
            messages.append({"role": "user", "content": list(results)})
        else:
            logger.info("The director used every round on %s", director.id)
    except anthropic.APIError as e:
        logger.warning("The director %s could not be reached: %s", director.id, e)
        return None

    spent = cost + sum(d.cost_usd for d in delegations)
    if response is None or response.stop_reason == "refusal":
        answer = BrainAnswer("El modelo declinó esta tarea.", director.id, director.model, spent)
        answer.ok = False
        return Orchestrated(answer, delegations)
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason == "tool_use":
        # Out of rounds mid-work: say so rather than pass on half a thought.
        text = text or "No terminé la tarea en el número de pasos permitido."
    answer = BrainAnswer(
        text or "Sin contenido.",
        director.id,
        getattr(response, "model", director.model) or director.model,
        spent,
        sources=list(dict.fromkeys(sources))[:12],
        ok=bool(text),
    )
    logger.info(
        "Orchestrated on %s with %d pieces (%s): %.4f $",
        director.id,
        len(delegations),
        json.dumps([d.worker for d in delegations]),
        spent,
    )
    return Orchestrated(answer, delegations)


def _result(tool_use_id: str, text: str, *, error: bool) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text or "(empty)",
        "is_error": error,
    }
