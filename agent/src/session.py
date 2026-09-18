"""Session state management for Kwami agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .domain import UsageTracker
from .usage import UsageReporter
from .utils.logging import get_logger

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from .agent import KwamiAgent

logger = get_logger("session")

# Shutdown callbacks share a ~10s worker budget; stay well inside it.
USAGE_REPORT_TIMEOUT_SECONDS = 5.0

# How long to wait for the framework to let go of a replaced agent before
# closing its providers anyway. Generous: the cost of waiting is a few idle
# sockets, the cost of not waiting is audio cut off mid-drain.
AGENT_RELEASE_TIMEOUT_SECONDS = 10.0
AGENT_RELEASE_POLL_SECONDS = 0.05


@dataclass
class SessionState:
    """Manages the state of a Kwami agent session.

    This class replaces the mutable dict pattern and provides:
    - Type-safe access to session state
    - Automatic memory cleanup when agents are replaced
    - Centralized state management
    - Usage tracking for the credit system
    """

    current_agent: KwamiAgent | None = None
    user_identity: str | None = None
    room_name: str | None = None
    room: Any = None  # LiveKit room; set in entrypoint so tools (e.g. web_search) can publish
    vad: Any = None
    greeting_delivered: bool = False
    # Owned here, not on the agent: agents are swapped on every voice/LLM/soul
    # change, and a browser left on a discarded agent keeps running -- and
    # billing -- until its own idle timer fires, with the frontend still
    # pointing at it.
    browser_session: Any = None
    usage_tracker: UsageTracker = field(default_factory=UsageTracker)
    usage_reporter: UsageReporter = field(default_factory=UsageReporter)
    _cleanup_tasks: list = field(default_factory=list, repr=False)
    # Serializes config handling. Two config messages arriving close together
    # used to be handled concurrently: both built a Zep client and both called
    # update_agent, so state.current_agent and session._agent could end up
    # disagreeing, with one client never closed.
    config_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _background_tasks: set = field(default_factory=set, repr=False)

    def spawn(self, coro: Any, *, name: str) -> asyncio.Task:
        """Run a coroutine in the background, keeping a strong reference to it.

        The event loop only holds a weak reference to a running task, so a bare
        `asyncio.create_task(...)` whose handle is dropped can be garbage
        collected mid-flight -- and any exception it raised is never retrieved,
        so the failure is completely silent.
        """
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Background task '%s' failed: %s", task.get_name(), exc)

    async def run_serialized(self, coro: Any) -> None:
        """Run a config-handling coroutine with no other config work in flight."""
        async with self.config_lock:
            await coro

    def update_agent(
        self,
        session: AgentSession,
        new_agent: KwamiAgent,
    ) -> None:
        """Update the current agent, cleaning up the old one's resources.

        Only closes memory if the new agent does NOT share the same memory
        instance (i.e. a truly new memory was created). When the same memory
        object is passed through to the new agent, closing it would break
        the new agent's memory.

        Args:
            session: The LiveKit agent session.
            new_agent: The new agent to switch to.
        """
        self.prepare_handoff(new_agent)
        session.update_agent(new_agent)
        logger.debug("Agent updated, cleanup tasks pending: %d", len(self._cleanup_tasks))

    def prepare_handoff(self, new_agent: KwamiAgent) -> KwamiAgent:
        """Carry session-owned resources onto an incoming agent, without swapping it in.

        Split out of `update_agent` because there are two ways an agent is
        replaced and only one of them is ours. When a *function tool* returns an
        Agent, the framework itself calls `AgentSession.update_agent` after the
        tool output is collected -- that is the handoff path that keeps the
        tool's result in the conversation instead of interrupting the turn that
        produced it. A tool that called `update_agent` here as well would swap
        twice; one that skipped this bookkeeping entirely would strand the paid
        cloud browser, orphan in-flight client tool futures, and leave
        `state.current_agent` pointing at an agent the session no longer runs.

        Returns the same agent, so a tool can `return state.prepare_handoff(a), "..."`.
        """
        old_agent = self.current_agent
        if old_agent is new_agent:
            return new_agent
        if old_agent:
            # Close the old agent's STT/LLM/TTS so the provider connections are
            # not leaked -- but only once the framework has let go of it. The
            # task waits for the old activity to be released before touching
            # anything; see `_cleanup_agent_voice_pipeline`.
            cleanup_task = asyncio.create_task(self._cleanup_agent_voice_pipeline(old_agent))
            self._cleanup_tasks.append(cleanup_task)
            if old_agent._memory:
                # Only close memory if the new agent has a DIFFERENT memory instance
                new_memory = getattr(new_agent, "_memory", None)
                if new_memory is not old_agent._memory:
                    cleanup_task = asyncio.create_task(self._cleanup_memory(old_agent._memory))
                    self._cleanup_tasks.append(cleanup_task)

            # Hand the live browser to the new agent instead of stranding it.
            old_browser = getattr(old_agent, "_browser_session", None)
            if old_browser is not None:
                self.browser_session = old_browser
                old_agent._browser_session = None

            # In-flight client tool calls resolve against the manager that
            # registered them. Without this transfer their futures are orphaned
            # and the LLM blocks for the full 30s timeout on a call whose result
            # already came back.
            self._transfer_pending_tool_calls(old_agent, new_agent)

            self._carry_conversation(old_agent, new_agent)

        self.current_agent = new_agent
        if self.browser_session is not None:
            new_agent._browser_session = self.browser_session
        new_agent.usage_tracker = self.usage_tracker
        if getattr(new_agent, "_memory", None) and hasattr(new_agent._memory, "set_usage_tracker"):
            new_agent._memory.set_usage_tracker(self.usage_tracker)
        # Ensure the new agent has the room so server-side tools (e.g. web_search) can publish
        if self.room is not None:
            new_agent.room = self.room
        return new_agent

    @staticmethod
    def _carry_conversation(old_agent: Any, new_agent: Any) -> None:
        """Move the conversation so far onto the incoming agent.

        Generation reads `Agent._chat_ctx`, not the session's, and a freshly
        constructed Agent starts with an empty one. Nothing in the framework
        copies it across a swap -- `AgentSession._update_activity` only inserts
        an `AgentHandoff` marker into its own context. So every reconfiguration
        this class performs (a TTS provider switch, an LLM switch, a realtime
        rebuild, a pipeline change) silently erased the conversation: the user
        asks for a different voice mid-sentence and the agent comes back with no
        idea who they are or what was just said.

        `copy(tools=...)` is the framework's own no-activity path, and the
        `tools` argument is load-bearing: it drops function calls whose tool the
        incoming agent does not have, which is exactly the case when the client
        re-registers a different tool set alongside the swap. Left in, those
        orphaned calls are a malformed context that some providers reject
        outright.
        """
        old_ctx = getattr(old_agent, "_chat_ctx", None)
        if old_ctx is None or not getattr(old_ctx, "items", None):
            return
        if getattr(new_agent, "_activity", None) is not None:
            # The framework owns the context of a running activity; writing to
            # it behind its back would race the turn in flight.
            logger.debug("Incoming agent already has an activity; leaving its context alone")
            return
        try:
            new_agent._chat_ctx = old_ctx.copy(tools=getattr(new_agent, "_tools", None) or [])
        except Exception as e:
            logger.warning("Could not carry the conversation onto the new agent: %s", e)
            return
        logger.info("Carried %d conversation item(s) across the agent swap", len(old_ctx.items))

    @staticmethod
    def _transfer_pending_tool_calls(old_agent: Any, new_agent: Any) -> None:
        """Move unresolved client tool futures onto the incoming agent."""
        old_manager = getattr(old_agent, "client_tools", None)
        new_manager = getattr(new_agent, "client_tools", None)
        if old_manager is None or new_manager is None:
            return
        pending = getattr(old_manager, "pending_calls", None)
        target = getattr(new_manager, "pending_calls", None)
        if not pending or target is None:
            return
        for call_id, future in list(pending.items()):
            if not getattr(future, "done", lambda: True)():
                target[call_id] = future
        pending.clear()
        logger.debug("Transferred %d pending client tool call(s) to the new agent", len(target))

    @staticmethod
    async def _await_agent_release(agent: Any) -> bool:
        """Wait until the framework has finished with `agent`, or time out.

        `AgentActivity.aclose` sets `agent._activity = None` as its last act, so
        that is the only observable "the session is done with this agent". There
        is no event to await, hence the poll; it is bounded and normally
        resolves within a tick or two.

        Returns True when the agent was released.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + AGENT_RELEASE_TIMEOUT_SECONDS
        while getattr(agent, "_activity", None) is not None:
            if loop.time() >= deadline:
                logger.warning(
                    "Old agent still active after %.0fs; closing its pipeline anyway",
                    AGENT_RELEASE_TIMEOUT_SECONDS,
                )
                return False
            await asyncio.sleep(AGENT_RELEASE_POLL_SECONDS)
        return True

    async def _cleanup_agent_voice_pipeline(
        self, agent: Any, *, wait_for_release: bool = True
    ) -> None:
        """Close STT/LLM/TTS connections to avoid unclosed inference connections.

        Waits for the framework to release the agent first. Closing a provider
        out from under a running activity is a race in both replacement paths:
        the framework drains and closes the old activity inside its own task, so
        `aclose()`-ing that agent's TTS the moment a replacement is prepared can
        land mid-drain. It is worse on the tool-handoff path, where the session
        does not swap until *after* the calling tool's output has been
        collected, so the old agent is still the live one for a while.

        Args:
            agent: The agent whose pipeline to close.
            wait_for_release: False at session teardown, where the activity is
                being torn down by the same shutdown and the ~10s worker budget
                will not stretch to waiting for it.
        """
        if wait_for_release:
            await self._await_agent_release(agent)

        seen: set = set()
        for name in ("stt", "llm", "tts", "_stt", "_llm", "_tts"):
            obj = getattr(agent, name, None)
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            try:
                if hasattr(obj, "aclose"):
                    await obj.aclose()
                    logger.debug("Closed agent pipeline component: %s", name)
                elif hasattr(obj, "close"):
                    close_fn = getattr(obj, "close")
                    if asyncio.iscoroutinefunction(close_fn):
                        await close_fn()
                    else:
                        close_fn()
                    logger.debug("Closed agent pipeline component: %s", name)
            except Exception as e:
                logger.debug("Could not close %s: %s", name, e)

    async def _cleanup_memory(self, memory: Any) -> None:
        """Clean up memory resources in the background.

        Args:
            memory: The KwamiMemory instance to clean up.
        """
        try:
            if hasattr(memory, "close"):
                await memory.close()
                logger.debug("Old agent memory closed successfully")
        except Exception as e:
            logger.warning(f"Failed to close memory: {e}")

    async def cleanup(self) -> None:
        """Clean up all pending resources.

        Should be called when the session ends.
        Reports accumulated usage to the credits API before closing.
        """
        # Try to resolve user_identity from agent config if not set
        if not self.user_identity and self.current_agent:
            kwami_id = getattr(self.current_agent.kwami_config, "kwami_id", None)
            if kwami_id:
                self.user_identity = kwami_id
                logger.info(f"Resolved user_identity from agent config: {kwami_id}")

        # Wait for all cleanup tasks to complete
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            self._cleanup_tasks.clear()

        # Close the cloud browser (persists profile cookies, stops per-minute billing).
        # Resolved from SessionState rather than the current agent, so a browser
        # opened before a reconfiguration is still found.
        browser_session = self.active_browser_session
        if browser_session is not None:
            try:
                await browser_session.close()
                logger.info("Closed cloud browser session during cleanup")
            except Exception as e:
                logger.warning("Failed to close cloud browser on cleanup: %s", e)
            self.browser_session = None

        # Close current agent's voice pipeline and memory
        if self.current_agent:
            await self._cleanup_agent_voice_pipeline(self.current_agent, wait_for_release=False)
            if self.current_agent._memory:
                await self._cleanup_memory(self.current_agent._memory)

        # Report usage LAST, and bounded. This runs inside a LiveKit shutdown
        # callback whose process budget defaults to 10s; when the report came
        # first, a slow credits API meant the worker was torn down before any
        # of the cleanup above ran.
        await self._report_usage()

        logger.debug("Session cleanup complete")

    async def _report_usage(self) -> None:
        """Send accumulated usage to the credits API, within a hard time bound."""
        if not (self.user_identity and self.room_name and self.usage_tracker.has_usage):
            logger.warning(
                f"Skipping usage report: user_identity={self.user_identity}, "
                f"room_name={self.room_name}, has_usage={self.usage_tracker.has_usage}"
            )
            return

        # user_identity may be a per-kwami memory id; credits are keyed on the
        # Supabase user id embedded in it.
        credits_user_id = self.user_identity
        if self.user_identity.startswith("kwami_") and self.user_identity.count("_") >= 2:
            credits_user_id = self.user_identity.split("_", 2)[1]

        logger.info(
            f"Reporting usage: user={credits_user_id}, "
            f"room={self.room_name}, has_usage={self.usage_tracker.has_usage}"
        )
        try:
            reported = await asyncio.wait_for(
                self.usage_reporter.report(
                    user_id=credits_user_id,
                    session_id=self.room_name,
                    tracker=self.usage_tracker,
                ),
                timeout=USAGE_REPORT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.error(
                "Usage report timed out after %ss; this session's usage was NOT billed "
                "(user=%s, room=%s)",
                USAGE_REPORT_TIMEOUT_SECONDS,
                credits_user_id,
                self.room_name,
            )
            return
        except Exception as e:
            logger.error(f"Failed to report usage on cleanup: {e}")
            return

        # The reporter signals failure by returning falsy; without this the
        # session's revenue was dropped in silence.
        if reported is False:
            logger.error(
                "Credits API rejected or dropped the usage report (user=%s, room=%s)",
                credits_user_id,
                self.room_name,
            )

    @property
    def active_browser_session(self) -> Any:
        """The live cloud browser, wherever it currently lives.

        A browser opened before a reconfiguration is transferred to SessionState
        by `update_agent`; one opened after sits on the current agent. Callers
        must not guess which, or they close nothing and leak a paid browser.
        """
        session = self.browser_session or getattr(self.current_agent, "_browser_session", None)
        if session is not None and getattr(session, "is_active", False):
            return session
        return None

    @property
    def has_agent(self) -> bool:
        """Check if there's a current agent."""
        return self.current_agent is not None

    def get_agent_or_none(self) -> KwamiAgent | None:
        """Get the current agent if it exists."""
        return self.current_agent


def create_session_state(
    initial_agent: KwamiAgent,
    user_identity: str | None = None,
    room_name: str | None = None,
    vad: Any = None,
) -> SessionState:
    """Factory function to create a SessionState.

    Args:
        initial_agent: The initial KwamiAgent instance.
        user_identity: Optional user identity string.
        room_name: Optional LiveKit room name (used for usage reporting).
        vad: Optional VAD instance.

    Returns:
        Configured SessionState instance.
    """
    return SessionState(
        current_agent=initial_agent,
        user_identity=user_identity,
        room_name=room_name,
        vad=vad,
    )
