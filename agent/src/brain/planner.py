"""Splitting a task so each part runs on the cheapest model that can do it.

The user's rule (2026-10-09): Haiku only for the simplest things, Sonnet and
Opus for everyday research and reasoning, and when a task can be split, split
it -- so Opus is not spent on the easy parts, and the work moves up to Opus
only for the steps that need real reasoning.

A planner call (Sonnet at low effort, structured output) breaks the task into
at most `MAX_STEPS` steps, each with its own difficulty and dependencies. The
bridge runs independent steps in parallel, feeds each step what its
dependencies found, and has one model combine the parts. A task the planner
judges indivisible runs whole, at the difficulty the planner gives it -- which
also corrects a voice model that guessed the difficulty wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..utils.logging import get_logger
from .ledger import token_cost

logger = get_logger("brain.planner")

MAX_STEPS = 5

PLANNER_SYSTEM = """You plan work for a team of models. Split the task only when its parts
differ in difficulty or can run in parallel; otherwise return one step.

Difficulty of each step:
- simple: look up one fact, extract, reformat, translate, list. A cheap model does it.
- normal: everyday research and reasoning: compare options, summarise sources, explain.
- hard: deep reasoning: decisions with trade-offs, money, conflicting evidence, maths,
  plans with many constraints.

Rules: at most 5 steps. Each goal is a self-contained instruction. `web` is true when
the step needs current information from the internet. `after` lists the ids of steps
whose results this step needs. Do not add a final "combine" step: that is done for you."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "split": {"type": "boolean"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "goal": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["simple", "normal", "hard"]},
                    "web": {"type": "boolean"},
                    "after": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "goal", "difficulty", "web", "after"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["split", "steps"],
    "additionalProperties": False,
}


@dataclass
class PlanStep:
    id: str
    goal: str
    difficulty: str
    web: bool
    after: list[str] = field(default_factory=list)


@dataclass
class Plan:
    steps: list[PlanStep]
    cost_usd: float = 0.0

    @property
    def split(self) -> bool:
        return len(self.steps) > 1


def parse_plan(raw: str) -> list[PlanStep]:
    """Validate the planner's JSON: unknown dependencies and cycles are dropped."""
    data = json.loads(raw)
    steps: list[PlanStep] = []
    seen: set[str] = set()
    for item in (data.get("steps") or [])[:MAX_STEPS]:
        step_id = str(item.get("id") or f"s{len(steps) + 1}")[:16]
        goal = str(item.get("goal") or "").strip()
        if not goal or step_id in seen:
            continue
        difficulty = item.get("difficulty")
        steps.append(
            PlanStep(
                id=step_id,
                goal=goal[:2000],
                difficulty=difficulty if difficulty in ("simple", "normal", "hard") else "normal",
                web=bool(item.get("web")),
                # Only earlier steps: that rules out cycles and forward references.
                after=[a for a in (item.get("after") or []) if a in seen],
            )
        )
        seen.add(step_id)
    if not data.get("split") and steps:
        # One coherent piece: keep the hardest difficulty the planner saw.
        order = {"simple": 0, "normal": 1, "hard": 2}
        hardest = max(steps, key=lambda s: order[s.difficulty])
        return [PlanStep("s1", "", hardest.difficulty, any(s.web for s in steps))]
    return steps


def waves(steps: list[PlanStep]) -> list[list[PlanStep]]:
    """Group steps so each wave depends only on earlier waves."""
    done: set[str] = set()
    remaining = list(steps)
    out: list[list[PlanStep]] = []
    while remaining:
        ready = [s for s in remaining if all(a in done for a in s.after)]
        if not ready:  # cannot happen after parse_plan, but never loop forever
            ready = remaining
        out.append(ready)
        done.update(s.id for s in ready)
        remaining = [s for s in remaining if s not in ready]
    return out


async def plan_task(api_key: str, model: str, task: str) -> Plan | None:
    """Ask the planner. None when planning fails: the caller runs the task whole."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key, timeout=120.0)
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4000,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": task}],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": PLAN_SCHEMA},
            },
        )
    except anthropic.APIError as e:
        logger.warning("Planner call failed: %s", e)
        return None
    cost = token_cost(
        "anthropic", model, response.usage.input_tokens or 0, response.usage.output_tokens or 0
    )
    if response.stop_reason != "end_turn":
        logger.info("Planner stopped with %s; running the task whole", response.stop_reason)
        return Plan(steps=[], cost_usd=cost)
    text = "".join(b.text for b in response.content if b.type == "text")
    try:
        steps = parse_plan(text)
    except (ValueError, AttributeError, TypeError):
        logger.warning("Planner returned an unreadable plan")
        return Plan(steps=[], cost_usd=cost)
    return Plan(steps=steps, cost_usd=cost)


def final_step(steps: list[PlanStep]) -> str | None:
    """The last step, when it builds (directly or not) on every other step.

    Such a step can answer the user itself, so no separate combine call is
    needed. None when the plan ends in independent parts that still need
    joining.
    """
    if len(steps) < 2:
        return None
    ancestors: dict[str, set[str]] = {}
    for step in steps:
        found = set(step.after)
        for parent in step.after:
            found |= ancestors.get(parent, set())
        ancestors[step.id] = found
    last = steps[-1]
    others = {s.id for s in steps if s.id != last.id}
    return last.id if ancestors[last.id] >= others else None
