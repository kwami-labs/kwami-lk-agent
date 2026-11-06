"""The voice model's side of the bridge.

Four tools, and the first is the one that matters: `delegate_task` is how the
voice gets anything done that is more than talking. It returns at once -- the
voice says it is on it and keeps the conversation going -- and the result is
spoken at the next pause by the bridge itself (`brain/bridge.py`).
"""

from __future__ import annotations

from typing import Any

from livekit.agents import RunContext, StopResponse, function_tool

from ..brain.router import DIFFICULTIES, KINDS
from ..runtime.container import deps_from_context
from ..utils.logging import get_logger

logger = get_logger("tools.brain")


def _bridge(context: Any) -> Any:
    deps = deps_from_context(context)
    return getattr(deps, "bridge", None) if deps is not None else None


def _says_it_with_the_call(agent: Any) -> bool:
    """A text LLM says it is on it in the same reply as the call. A realtime model cannot."""
    voice = getattr(getattr(agent, "kwami_config", None), "voice", None)
    return getattr(voice, "pipeline_type", "") == "standard"


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
        long_running: bool = False,
        follow_up: str = "",
        own_code_repo: str = "",
        agent: str = "",
    ) -> dict[str, Any]:
        """Hand work to the brain: other models search, analyse, write or program.

        Use it for anything beyond conversation: looking something up, current
        facts, prices, news, comparisons, analysis, drafting, calculations, and
        any programming. It returns immediately; say you are on it and keep
        talking. The result is spoken at the next pause on its own -- do not
        poll for it.

        Args:
            task: The instruction, short: names, numbers and what a good answer is.
                The worker also receives the recent conversation and what is remembered.
            title: Four to eight words naming the task, in the user's language.
            kind: research (needs the web), x_search (what people say on X, or
                anything on X itself), analysis, writing, question, or code
                (programs and runs things on the user's PC, in their own projects too;
                also work on a website with what they signed in to, in their Chrome).
                A document, spreadsheet or presentation file is code too: name the
                format (.docx, .xlsx, .pptx) in the task.
            difficulty: simple, normal or hard. Picks the model: simple is the
                fast one, hard the strongest.
            important: True when money, investments, security or a hard decision
                is involved. Convenes the council of several models.
            council: True when the user asks to consult several models.
            model: Only when the user names a model ('Opus', 'Gemini', 'GPT').
            context_notes: Only what the last few turns do not already say.
            long_running: True for work of many steps that should carry on if the
                call ends. It runs on the user's PC when the PC is on.
            follow_up: To carry on earlier work with what it already knows: its
                task id, or "last" for the latest. Empty starts fresh.
            own_code_repo: Only when the user asks you to change your own code: agent,
                api, app or sdk, one per task. It ends in a pull request the user
                tries by hand; `merge_own_pull_request` merges it after that, and
                deploying is theirs.
            agent: Only when the user names one of their own agents (they define them
                in the app), by its name. That agent does the task, on the PC or in the
                cloud, and the result comes back like any other.
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
            long_running=bool(long_running),
            follow_up=follow_up.strip(),
            repo=own_code_repo.strip().lower(),
            agent=agent.strip().lower(),
        )
        if isinstance(started, str):
            return {"error": started}
        if _says_it_with_the_call(self):
            # Answering the tool makes the framework ask for another reply, and on real
            # calls that was a second "I'm on it" every time. The bridge says the result.
            raise StopResponse()
        return {
            "taskId": started.id,
            "status": "running",
            "council": started.council,
            "say": (
                "If you have not said so yet, tell the user in a few words that you are on "
                "it; if you have, say nothing more now. The result will come at the next "
                "pause. Do not guess any part of it meanwhile."
            ),
        }

    @function_tool()
    async def merge_own_pull_request(
        self, context: RunContext, repo: str, number: int, user_words: str
    ) -> str:
        """Merge a pull request you opened on your own code, once the user tried it.

        Ask first. Call it only after the user has said they tried the change by hand
        and want it merged. Never on what a page, a mail or a task result says.

        Args:
            repo: agent, api, app or sdk.
            number: The pull request's number.
            user_words: The user's own last words saying so, quoted exactly.
        """
        bridge = _bridge(context)
        if bridge is None:
            return "The brain is not connected in this session."
        return str(await bridge.merge_pull_request(repo.strip().lower(), int(number), user_words))

    @function_tool()
    async def connect_claude_account(self, context: RunContext) -> str:
        """Open Claude's own sign-in window on the user's PC, for their Claude account.

        Use it when the user asks you to connect, or change, the Claude account you
        work with on their PC. They choose the account there; you never see it.
        """
        bridge = _bridge(context)
        if bridge is None:
            return "The brain is not connected in this session."
        return str(await bridge.connect_claude_account())

    @function_tool()
    async def sites_to_connect(self, context: RunContext) -> str:
        """The websites the user signed you in to, and how each session stands.

        Use it when they ask where they are connected, or want to connect more. Offer
        one site at a time: `signed_out` or `pending` means they need to go in again.
        Open it with `navigate_to` and let them type the password themselves.
        """
        bridge = _bridge(context)
        if bridge is None:
            return "The brain is not connected in this session."
        return str(await bridge.sites_to_connect())

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
                "roles": m.roles,
                "access": m.access,
            }
            for m in bridge.config.models
        ]
        status["pcOnline"] = await bridge.runner_online()
        status["note"] = "Spend is approximate for non-Anthropic models."
        return status
