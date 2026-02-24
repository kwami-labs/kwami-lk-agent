"""Session state management for Kwami agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .usage import UsageTracker, UsageReporter
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
    
    current_agent: Optional["KwamiAgent"] = None
    user_identity: Optional[str] = None
    room_name: Optional[str] = None
    vad: Any = None
    greeting_delivered: bool = False
    usage_tracker: UsageTracker = field(default_factory=UsageTracker)
    usage_reporter: UsageReporter = field(default_factory=UsageReporter)
    _cleanup_tasks: list = field(default_factory=list, repr=False)
    
    def update_agent(
        self,
        session: "AgentSession",
        new_agent: "KwamiAgent",
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
        if old_agent and old_agent._memory:
            # Only close memory if the new agent has a DIFFERENT memory instance
            new_memory = getattr(new_agent, "_memory", None)
            if new_memory is not old_agent._memory:
                cleanup_task = asyncio.create_task(
                    self._cleanup_memory(old_agent._memory)
                )
                self._cleanup_tasks.append(cleanup_task)
        
        # Update the session with the new agent
        session.update_agent(new_agent)
        self.current_agent = new_agent
        
        logger.debug(f"Agent updated, cleanup tasks pending: {len(self._cleanup_tasks)}")
    
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
        
        # Clean up current agent's memory
        if self.current_agent and self.current_agent._memory:
            await self._cleanup_memory(self.current_agent._memory)
        
        logger.debug("Session cleanup complete")
    
    @property
    def has_agent(self) -> bool:
        """Check if there's a current agent."""
        return self.current_agent is not None
    
    def get_agent_or_none(self) -> Optional["KwamiAgent"]:
        """Get the current agent if it exists."""
        return self.current_agent


def create_session_state(
    initial_agent: "KwamiAgent",
    user_identity: Optional[str] = None,
    room_name: Optional[str] = None,
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
