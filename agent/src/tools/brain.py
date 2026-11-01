"""The voice model's side of the bridge.

Four tools, and the first is the one that matters: `delegate_task` is how the
voice gets anything done that is more than talking. It returns at once -- the
voice says it is on it and keeps the conversation going -- and the result is
spoken at the next pause by the bridge itself (`brain/bridge.py`).
"""

from __future__ import annotations

from typing import Any

from livekit.agents import RunContext, function_tool

from ..brain.router import DIFFICULTIES, KINDS
from ..runtime.container import deps_from_context
from ..utils.logging import get_logger

logger = get_logger("tools.brain")


def _bridge(context: Any) -> Any:
    deps = deps_from_context(context)
    return getattr(deps, "bridge", None) if deps is not None else None


class BrainToolsMixin:
    @function_tool()
    async def delegate_task(
        self,
        context: RunContext,
        task: str,
        title: str,
        kind: str = "question",
        difficulty: str = "normal",
        important: bool = False,
        council: bool = False,
        model: str = "",
        context_notes: str = "",
    ) -> dict[str, Any]:
        """Hand work to the brain: other models search, analyse, write or program.

        Use it for anything beyond conversation: looking something up, current
        facts, prices, news, comparisons, analysis, drafting, calculations, and
        any programming. It returns immediately; say you are on it and keep
        talking. The result is spoken at the next pause on its own -- do not
        poll for it.

        Args:
            task: The full, self-contained instruction. The worker has not heard
                the conversation: include names, numbers and what a good answer is.
            title: Four to eight words naming the task, in the user's language.
            kind: research (needs the web), analysis, writing, question, or code
                (programs and runs things on the user's PC, in the Jarvis folder).
            difficulty: simple, normal or hard. Picks the model: simple is the
                fast one, hard the strongest.
            important: True when money, investments, security or a hard decision
                is involved. Convenes the council of several models.
            council: True when the user asks to consult several models.
            model: Only when the user names a model ('Opus', 'Gemini', 'GPT').
            context_notes: Anything from the conversation the worker needs.
        """
        bridge = _bridge(context)
        if bridge is None:
            return {"error": "The brain is not connected in this session."}
        kind = kind if kind in KINDS else "question"
        difficulty = difficulty if difficulty in DIFFICULTIES else "normal"
        started = bridge.start(
            task=task.strip(),
            title=title.strip(),
            kind=kind,
            difficulty=difficulty,
            important=bool(important),
            council=bool(council),
            model=model.strip(),
            context=context_notes.strip(),
        )
        if isinstance(started, str):
            return {"error": started}
        return {
            "taskId": started.id,
            "status": "running",
            "council": started.council,
            "say": (
                "Tell the user briefly that you are on it; the result will come at the "
                "next pause. Do not guess any part of it meanwhile."
            ),
        }

    @function_tool()
    async def check_tasks(self, context: RunContext, task_id: str = "") -> dict[str, Any]:
        """What the brain is working on, or the full result of one task.

        Use when the user asks how a task is going, or wants the detail of a
        result already summarised.

        Args:
            task_id: A task id for its full result. Empty lists recent tasks.
        """
        bridge = _bridge(context)
        if bridge is None:
            return {"error": "The brain is not connected in this session."}
        if task_id:
            found = bridge.tasks.get(task_id.strip())
            return found.public(full=True) if found else {"error": f"No task {task_id}."}
        return {"tasks": [t.public() for t in list(bridge.tasks.values())[-8:]]}

    @function_tool()
    async def cancel_task(self, context: RunContext, task_id: str) -> str:
        """Stop a running brain task when the user no longer wants it.

        Args:
            task_id: The id `delegate_task` returned.
        """
        bridge = _bridge(context)
        if bridge is None:
            return "The brain is not connected in this session."
        return "Cancelled." if bridge.cancel(task_id.strip()) else f"No running task {task_id}."

    @function_tool()
    async def brain_status(self, context: RunContext) -> dict[str, Any]:
        """The brain's models, which providers have keys, and this month's spend.

        Use when the user asks which models you use, what the brain costs, or
        why a model is unavailable.
        """
        bridge = _bridge(context)
        if bridge is None:
            return {"error": "The brain is not connected in this session."}
        status = bridge.status()
        status.pop("type", None)
        status["roster"] = [
            {
                "id": m.id,
                "label": m.label,
                "provider": m.provider,
                "model": m.model,
                "role": m.role,
                "enabled": m.enabled,
                "council": m.council,
            }
            for m in bridge.config.models
        ]
        status["note"] = "Spend is approximate for non-Anthropic models."
        return status
