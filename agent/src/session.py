"""Session state management for Kwami agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .usage import UsageReporter, UsageTracker
from .utils.logging import get_logger

if TYPE_CHECKING:
    from livekit.agents import AgentSession

    from .agent import KwamiAgent

logger = get_logger("session")


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
    usage_tracker: UsageTracker = field(default_factory=UsageTracker)
    usage_reporter: UsageReporter = field(default_factory=UsageReporter)
    _cleanup_tasks: list = field(default_factory=list, repr=False)

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
        old_agent = self.current_agent
        if old_agent:
            # Close old agent's voice pipeline (STT/LLM/TTS) to avoid unclosed inference connections
            cleanup_task = asyncio.create_task(self._cleanup_agent_voice_pipeline(old_agent))
            self._cleanup_tasks.append(cleanup_task)
            if old_agent._memory:
                # Only close memory if the new agent has a DIFFERENT memory instance
                new_memory = getattr(new_agent, "_memory", None)
                if new_memory is not old_agent._memory:
                    cleanup_task = asyncio.create_task(self._cleanup_memory(old_agent._memory))
                    self._cleanup_tasks.append(cleanup_task)

        # Update the session with the new agent
        session.update_agent(new_agent)
        self.current_agent = new_agent
        new_agent.usage_tracker = self.usage_tracker
        if getattr(new_agent, "_memory", None) and hasattr(new_agent._memory, "set_usage_tracker"):
            new_agent._memory.set_usage_tracker(self.usage_tracker)
        # Ensure the new agent has the room so server-side tools (e.g. web_search) can publish
        if self.room is not None:
            new_agent.room = self.room

        logger.debug(f"Agent updated, cleanup tasks pending: {len(self._cleanup_tasks)}")

    async def _cleanup_agent_voice_pipeline(self, agent: Any) -> None:
        """Close STT/LLM/TTS connections to avoid unclosed inference connections.

        Args:
            agent: The agent whose pipeline to close (e.g. previous agent after reconfigure).
        """
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

        # Report usage to credits system (use Supabase user id for credits; user_identity may be per-kwami memory id)
        if self.user_identity and self.room_name and self.usage_tracker.has_usage:
            credits_user_id = self.user_identity
            if self.user_identity.startswith("kwami_") and self.user_identity.count("_") >= 2:
                credits_user_id = self.user_identity.split("_", 2)[1]
            logger.info(
                f"Reporting usage: user={credits_user_id}, "
                f"room={self.room_name}, has_usage={self.usage_tracker.has_usage}"
            )
            try:
                await self.usage_reporter.report(
                    user_id=credits_user_id,
                    session_id=self.room_name,
                    tracker=self.usage_tracker,
                )
            except Exception as e:
                logger.error(f"Failed to report usage on cleanup: {e}")
        else:
            logger.warning(
                f"Skipping usage report: user_identity={self.user_identity}, "
                f"room_name={self.room_name}, has_usage={self.usage_tracker.has_usage}"
            )

        # Wait for all cleanup tasks to complete
        if self._cleanup_tasks:
            await asyncio.gather(*self._cleanup_tasks, return_exceptions=True)
            self._cleanup_tasks.clear()

        # Close current agent's cloud browser session (persists profile cookies, stops billing)
        if self.current_agent:
            browser_session = getattr(self.current_agent, "_browser_session", None)
            if browser_session and browser_session.is_active:
                try:
                    await browser_session.close()
                    logger.info("Closed cloud browser session during cleanup")
                except Exception as e:
                    logger.warning("Failed to close cloud browser on cleanup: %s", e)

        # Close current agent's voice pipeline and memory
        if self.current_agent:
            await self._cleanup_agent_voice_pipeline(self.current_agent)
            if self.current_agent._memory:
                await self._cleanup_memory(self.current_agent._memory)

        logger.debug("Session cleanup complete")

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
