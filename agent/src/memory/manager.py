"""Kwami Memory Manager - Main orchestrator for Zep memory operations.

Handles client lifecycle, message persistence (with proper batching),
context retrieval, and delegates to specialized modules for ontology,
search, and context formatting.

Key improvements over monolithic memory.py:
- Messages include the `name` field for better graph construction
- User + assistant messages are batched in a single add_messages call
- Uses `ignore_roles=["assistant"]` to prevent assistant entity pollution
- Context templates for structured retrieval
- Proper edge source/target constraints (via ontology module)
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from ..config import KwamiMemoryConfig
from .context import MemoryContext, get_context, setup_context_template
from .ontology import configure_ontology, get_ontology
from .search import (
    get_entities_by_type,
    get_user_name,
    search_graph,
    search_thread,
)
from .utils import get_zep_imports, logger

if TYPE_CHECKING:
    from zep_cloud.client import AsyncZep


class KwamiMemory:
    """Memory manager for a single Kwami instance.

    Handles all Zep interactions for persistent memory, including:
    - User and session management
    - Message persistence with batching and name attribution
    - Context retrieval via templates
    - Knowledge graph ontology configuration
    - Entity and fact search
    """

    def __init__(
        self,
        config: KwamiMemoryConfig,
        kwami_id: str,
        kwami_name: str = "Kwami",
        usage_tracker=None,
    ):
        self.config = config
        self.kwami_id = kwami_id
        self.kwami_name = kwami_name
        self._usage_tracker = usage_tracker
        self._client: Optional["AsyncZep"] = None
        self._user_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._initialized = False
        self._template_id: Optional[str] = None

        # Message batching: buffer user message to send with assistant response
        self._pending_user_message: Optional[tuple[str, str | None]] = None

        # Cached user name (avoid repeated lookups)
        self._cached_user_name: Optional[str] = None

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def is_enabled(self) -> bool:
        """Check if memory is enabled and configured."""
        return self.config.enabled and bool(self.config.api_key)

    @property
    def is_initialized(self) -> bool:
        """Check if memory has been initialized."""
        return self._initialized

    @property
    def user_id(self) -> Optional[str]:
        """Get the Zep user ID for this Kwami."""
        return self._user_id

    @property
    def session_id(self) -> Optional[str]:
        """Get the current Zep session ID."""
        return self._session_id

    def set_usage_tracker(self, usage_tracker) -> None:
        """Attach the shared session usage tracker."""
        self._usage_tracker = usage_tracker

    def _record_usage(self, model_id: str, units_used: float = 1.0) -> None:
        """Record a billable Zep operation when tracking is enabled."""
        if self._usage_tracker:
            self._usage_tracker.record_external_usage(
                "memory",
                model_id,
                units_used=units_used,
                request_count=1,
            )

    # ========================================================================
    # Initialization
    # ========================================================================

    async def initialize(self) -> bool:
        """Initialize the Zep client and ensure user/session exist.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        if not self.is_enabled:
            logger.warning("Memory is disabled or API key not configured")
            return False

        AsyncZep, _, _ = get_zep_imports()
        if AsyncZep is None:
            logger.error("zep_cloud not available, disabling memory")
            return False

        try:
            self._client = AsyncZep(api_key=self.config.api_key)
            self._user_id = self.config.user_id or f"kwami_{self.kwami_id}"

            await self._ensure_user_exists()

            self._session_id = (
                self.config.session_id
                or f"session_{self._user_id}_{uuid.uuid4().hex[:8]}"
            )
            await self._ensure_session_exists()

            # Configure ontology (entity/edge types) if enabled
            if self.config.configure_ontology:
                await configure_ontology(self._client, self._user_id)

            # Set up context template for structured retrieval
            self._template_id = await setup_context_template(
                self._client, self._user_id
            )

            self._initialized = True
            logger.info(
                f"Memory initialized for Kwami '{self.kwami_name}' "
                f"(user: {self._user_id}, session: {self._session_id})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            self._initialized = False
            return False

    async def _ensure_user_exists(self) -> None:
        """Create user in Zep if it doesn't exist.

        The Zep "user" represents the HUMAN user, not the AI assistant.
        """
        try:
            await self._client.user.get(self._user_id)
            logger.debug(f"User {self._user_id} already exists")
        except Exception:
            try:
                await self._client.user.add(
                    user_id=self._user_id,
                    metadata={
                        "kwami_id": self.kwami_id,
                        "assistant_name": self.kwami_name,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )
                self._record_usage("zep/create_user")
                logger.info(f"Created Zep user: {self._user_id}")
            except Exception as e:
                error_msg = str(e).lower()
                if "400" in error_msg and "already exists" in error_msg:
                    logger.info(
                        f"User {self._user_id} already exists (race condition)"
                    )
                    return
                logger.error(f"Failed to create user {self._user_id}: {e}")
                raise

    async def _ensure_session_exists(self) -> None:
        """Create thread (session) in Zep if it doesn't exist."""
        try:
            await self._client.thread.get(thread_id=self._session_id)
            logger.debug(f"Thread {self._session_id} already exists")
        except Exception:
            try:
                await self._client.thread.create(
                    thread_id=self._session_id,
                    user_id=self._user_id,
                )
                self._record_usage("zep/create_thread")
                logger.info(f"Created Zep thread: {self._session_id}")
            except Exception as e:
                logger.error(
                    f"Failed to create thread {self._session_id}: {e}"
                )
                raise

    # ========================================================================
    # Message Handling
    # ========================================================================

    async def buffer_user_message(
        self, content: str, name: str | None = None
    ) -> None:
        """Buffer a user message to be sent with the next assistant response.

        This implements Zep's recommended pattern of sending both user and
        assistant messages together in a single add_messages call, which
        produces significantly better graph construction.

        If there's already a buffered message (e.g., user spoke twice without
        agent response), the previous message is flushed first.

        Args:
            content: The user's message content.
            name: The user's real name (improves graph construction).
        """
        if not self._initialized or not self._client:
            return

        if not content or not content.strip():
            return

        # Flush any existing buffered message first
        if self._pending_user_message:
            await self._flush_pending_message()

        self._pending_user_message = (content.strip(), name)
        logger.debug(f"Buffered user message: {content[:50]}...")

    async def add_exchange(
        self, assistant_content: str, assistant_name: str | None = None
    ) -> None:
        """Send buffered user message + assistant response as a batch.

        This is the PRIMARY method for adding messages to memory.
        It implements Zep's recommended pattern:
        1. Sends both messages in a single API call
        2. Includes the `name` field for better entity extraction
        3. Uses `ignore_roles=["assistant"]` so assistant messages provide
           context but don't create graph entities

        Args:
            assistant_content: The assistant's response content.
            assistant_name: The assistant's display name.
        """
        if not self._initialized or not self._client:
            return

        _, ZepMessage, _ = get_zep_imports()
        if ZepMessage is None:
            return

        messages = []
        user_name = self._cached_user_name or "User"
        assistant_name = assistant_name or self.kwami_name
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Add buffered user message
        if self._pending_user_message:
            content, name = self._pending_user_message
            messages.append(
                ZepMessage(
                    role="user",
                    content=content,
                    name=name or user_name,
                    created_at=now,
                )
            )
            self._pending_user_message = None

        # Add assistant message
        if assistant_content and assistant_content.strip():
            messages.append(
                ZepMessage(
                    role="assistant",
                    content=assistant_content.strip(),
                    name=assistant_name,
                    created_at=now,
                )
            )

        if not messages:
            return

        try:
            await self._client.thread.add_messages(
                thread_id=self._session_id,
                messages=messages,
                ignore_roles=["assistant"],
            )
            self._record_usage("zep/add_messages")
            logger.debug(
                f"Added {len(messages)} messages to memory "
                f"(user: {user_name}, assistant: {assistant_name})"
            )
        except Exception as e:
            import traceback

            logger.error(
                f"Failed to add messages to memory: {type(e).__name__}: {e}\n"
                f"Messages count: {len(messages)}\n"
                f"Traceback: {traceback.format_exc()}"
            )

    async def add_message(
        self, role: str, content: str, name: str | None = None
    ) -> None:
        """Add a single message to memory.

        For best results, prefer buffer_user_message() + add_exchange()
        for conversation turns. This method is for standalone messages
        (e.g., system messages, facts).

        Args:
            role: Message role - 'user', 'assistant', or 'system'.
            content: Message content.
            name: Speaker's name (important for graph construction).
        """
        if not self._initialized or not self._client:
            return

        if not content or not content.strip():
            return

        try:
            _, ZepMessage, _ = get_zep_imports()
            if ZepMessage is None:
                return

            role = role.lower().strip()
            if role not in ("user", "assistant", "system"):
                logger.warning(f"Unknown role '{role}', defaulting to 'user'")
                role = "user"

            # Determine the name for the message
            if not name:
                if role == "user":
                    name = self._cached_user_name or "User"
                elif role == "assistant":
                    name = self.kwami_name
                else:
                    name = "System"

            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            message = ZepMessage(
                role=role,
                content=content.strip(),
                name=name,
                created_at=now,
            )

            # Use ignore_roles for assistant messages
            ignore_roles = ["assistant"] if role != "system" else None

            await self._client.thread.add_messages(
                thread_id=self._session_id,
                messages=[message],
                ignore_roles=ignore_roles,
            )
            self._record_usage("zep/add_messages")

            logger.debug(f"Added {role} message to memory: {content[:50]}...")

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to add message to memory: {type(e).__name__}: {e}\n"
                f"Role: {role}, Content length: {len(content)}\n"
                f"Traceback: {traceback.format_exc()}"
            )

    async def _flush_pending_message(self) -> None:
        """Flush a buffered user message without an assistant response.

        Called when a new user message arrives before the previous one
        was paired with an assistant response.
        """
        if not self._pending_user_message:
            return

        content, name = self._pending_user_message
        self._pending_user_message = None

        _, ZepMessage, _ = get_zep_imports()
        if ZepMessage is None:
            return

        try:
            user_name = name or self._cached_user_name or "User"
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            message = ZepMessage(
                role="user",
                content=content,
                name=user_name,
                created_at=now,
            )
            await self._client.thread.add_messages(
                thread_id=self._session_id,
                messages=[message],
            )
            self._record_usage("zep/add_messages")
            logger.debug(f"Flushed pending user message: {content[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to flush pending message: {e}")

    async def add_fact(self, fact: str) -> None:
        """Add a fact about the user as a system message.

        In Zep v3, facts are automatically extracted from messages.
        This frames the fact as a system message to trigger extraction.

        Args:
            fact: The fact to add (e.g., "User's name is Alex").
        """
        await self.add_message(
            role="system",
            content=f"Important information learned: {fact}",
            name="System",
        )

    # ========================================================================
    # Context Retrieval
    # ========================================================================

    async def get_context(self) -> MemoryContext:
        """Get memory context for LLM injection.

        Returns:
            MemoryContext with summary, facts, entities, and recent messages.
        """
        if not self._initialized or not self._client:
            return MemoryContext()

        try:
            context = await get_context(
                client=self._client,
                user_id=self._user_id,
                session_id=self._session_id,
                template_id=self._template_id,
                kwami_name=self.kwami_name,
                max_messages=self.config.max_context_messages,
                min_relevance=self.config.min_fact_relevance,
                include_facts=self.config.include_facts,
            )
            self._record_usage("zep/get_context")
            return context
        except Exception as e:
            logger.error(f"Failed to get memory context: {e}")
            return MemoryContext()

    # ========================================================================
    # Search
    # ========================================================================

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search memory for relevant context.

        Args:
            query: Search query.
            limit: Maximum number of results.

        Returns:
            List of search results with content and score.
        """
        if not self._initialized or not self._client:
            return []
        results = await search_thread(self._client, self._session_id, query, limit)
        self._record_usage("zep/thread_search")
        return results

    async def search_by_entity_type(
        self,
        query: str,
        entity_types: list[str],
        limit: int = 10,
    ) -> list[dict]:
        """Search the knowledge graph filtered by entity types.

        Args:
            query: Search query.
            entity_types: Entity type names to filter by.
            limit: Maximum number of results.

        Returns:
            List of matching nodes.
        """
        if not self._initialized or not self._client:
            return []
        results = await search_graph(
            self._client,
            self._user_id,
            query,
            scope="nodes",
            limit=limit,
            node_labels=entity_types,
        )
        self._record_usage("zep/graph_search")
        return results

    async def get_entities_by_type(
        self,
        entity_type: str,
        limit: int = 20,
    ) -> list[dict]:
        """Get all entities of a specific type.

        Args:
            entity_type: The entity type name.
            limit: Maximum number of results.

        Returns:
            List of entities of the specified type.
        """
        if not self._initialized or not self._client:
            return []
        return await get_entities_by_type(
            self._client, self._user_id, entity_type, limit
        )

    async def get_preferences(self, limit: int = 20) -> list[dict]:
        """Get user preferences from the knowledge graph."""
        return await self.get_entities_by_type("Preference", limit)

    # ========================================================================
    # User Identity
    # ========================================================================

    async def get_user_name(self) -> Optional[str]:
        """Get the user's name from the knowledge graph.

        Caches the result after the first successful lookup.

        Returns:
            The user's name if found, None otherwise.
        """
        if self._cached_user_name:
            return self._cached_user_name

        if not self._initialized or not self._client:
            return None

        try:
            name = await get_user_name(
                self._client, self._user_id, self.kwami_name
            )
            self._record_usage("zep/get_user_name")
            if name:
                self._cached_user_name = name
            return name
        except Exception as e:
            logger.debug(f"Could not get user name: {e}")
            return None

    def set_user_name(self, name: str) -> None:
        """Manually set the cached user name.

        Useful when the user introduces themselves in conversation.

        Args:
            name: The user's name.
        """
        self._cached_user_name = name

    # ========================================================================
    # Ontology
    # ========================================================================

    async def get_ontology(self) -> dict | None:
        """Get the current ontology configuration.

        Returns:
            Dict with 'entity_types' and 'edge_types', or None.
        """
        if not self._client or not self._user_id:
            return None
        return await get_ontology(self._client, self._user_id)

    # ========================================================================
    # Session Management
    # ========================================================================

    async def clear_session(self) -> None:
        """Clear the current thread (session) memory."""
        if not self._initialized or not self._client:
            return

        try:
            await self._client.thread.delete(thread_id=self._session_id)
            logger.info(f"Cleared thread memory: {self._session_id}")
        except Exception as e:
            logger.error(f"Failed to clear thread: {e}")

    async def close(self) -> None:
        """Close the Zep client connection.

        Flushes any pending messages before closing.
        """
        # Flush any pending user message
        if self._pending_user_message:
            try:
                await self._flush_pending_message()
            except Exception:
                pass

        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
        self._initialized = False
        logger.debug("Memory client closed")

    def build_memory_enhanced_prompt(self, base_prompt: str) -> str:
        """Build system prompt with memory context injection placeholder.

        Args:
            base_prompt: The original system prompt.

        Returns:
            Enhanced prompt with memory context placeholder.
        """
        if not self.config.auto_inject_context:
            return base_prompt

        return (
            f"{base_prompt}\n\n"
            "## Memory Context\n"
            "You have access to your persistent memory about past conversations. "
            "Use this context to provide personalized, contextual responses.\n"
            "{{MEMORY_CONTEXT}}"
        )


# ============================================================================
# Factory
# ============================================================================


async def create_memory(
    config: KwamiMemoryConfig,
    kwami_id: str,
    kwami_name: str = "Kwami",
    usage_tracker=None,
) -> Optional[KwamiMemory]:
    """Factory function to create and initialize a KwamiMemory instance.

    Args:
        config: Memory configuration.
        kwami_id: Unique identifier for the Kwami.
        kwami_name: Display name for the Kwami.

    Returns:
        Initialized KwamiMemory instance, or None if initialization fails.
    """
    memory = KwamiMemory(config, kwami_id, kwami_name, usage_tracker=usage_tracker)

    if not memory.is_enabled:
        logger.info(f"Memory disabled for Kwami '{kwami_name}'")
        return None

    if await memory.initialize():
        return memory

    return None
