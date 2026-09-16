"""Memory context retrieval and formatting.

Handles assembling context from Zep for LLM system prompt injection.
Uses Zep context templates for consistent, structured retrieval and
includes temporal validity information for facts.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .utils import logger

if TYPE_CHECKING:
    from zep_cloud.client import AsyncZep

# Template ID prefix for Kwami context templates
TEMPLATE_PREFIX = "kwami-context"
MAX_CONTEXT_BLOCK_CHARS = 2200
MAX_SUMMARY_CHARS = 600
MAX_FACTS = 8
MAX_FACT_CHARS = 180
MAX_ENTITIES = 6
MAX_ENTITY_SUMMARY_CHARS = 120

# Default context template definition
# Uses Zep template variables for structured retrieval
DEFAULT_CONTEXT_TEMPLATE = """# USER PROFILE
%{user_summary}

# RELEVANT FACTS
%{edges limit=20}

# KEY ENTITIES
%{entities limit=10}"""


@dataclass
class MemoryContext:
    """Context retrieved from Zep memory for LLM injection."""

    context_block: str | None = None
    """Pre-formatted context block from Zep context template."""

    summary: str | None = None
    facts: list[str] = None
    entities: list[dict] = None
    recent_messages: list[dict] = None

    def __post_init__(self):
        self.facts = self.facts or []
        self.entities = self.entities or []
        self.recent_messages = self.recent_messages or []

    def has_content(self) -> bool:
        """True when Zep actually returned something.

        The retrieval helpers swallow their own exceptions and hand back an
        empty context, which is indistinguishable from a successful call
        against a brand-new user unless it is checked explicitly. Billing keys
        off this so a failed round-trip is not charged.
        """
        return bool(self.context_block or self.summary or self.facts or self.recent_messages)

    def to_system_prompt_addition(self) -> str:
        """Convert memory context to text for system prompt injection.

        Prefers the pre-formatted context block from Zep templates
        when available. Falls back to manual formatting of individual
        components.
        """
        # Prefer the template-generated context block
        if self.context_block:
            return self.context_block[:MAX_CONTEXT_BLOCK_CHARS]

        # Fallback: manually format from components
        parts = []

        if self.summary:
            parts.append(f"## Conversation Summary\n{self.summary[:MAX_SUMMARY_CHARS]}")

        if self.facts:
            facts_text = "\n".join(f"- {fact[:MAX_FACT_CHARS]}" for fact in self.facts[:MAX_FACTS])
            parts.append(
                "## Known Facts About the Human User\n"
                "These facts are about the HUMAN you are talking to "
                "(NOT about you, the assistant).\n"
                "Facts marked as 'present' are currently valid. "
                "Facts with a past end date are no longer valid.\n"
                f"{facts_text}"
            )

        if self.entities:
            entities_text = "\n".join(
                f"- {str(e.get('name', 'Unknown'))[:60]}: "
                f"{str(e.get('summary', e.get('type', 'entity')))[:MAX_ENTITY_SUMMARY_CHARS]}"
                for e in self.entities[:MAX_ENTITIES]
            )
            parts.append(f"## Relevant Entities\n{entities_text}")

        if not parts:
            return ""

        return "\n\n".join(parts)


async def setup_context_template(
    client: "AsyncZep",
    user_id: str,
    template: str | None = None,
) -> str | None:
    """Create or update a context template for this user.

    Context templates provide consistent, structured context retrieval
    with automatic relevance detection by Zep.

    Args:
        client: The async Zep client.
        user_id: The Zep user ID (used to generate template ID).
        template: Custom template string. Uses default if None.

    Returns:
        The template ID if created/updated successfully, None otherwise.
    """
    template_id = f"{TEMPLATE_PREFIX}-{user_id}"
    template_content = template or DEFAULT_CONTEXT_TEMPLATE

    try:
        # Try to update existing template first
        try:
            await client.context.update_context_template(
                template_id=template_id,
                template=template_content,
            )
            logger.debug(f"Updated context template: {template_id}")
            return template_id
        except Exception:
            pass

        # Create new template
        await client.context.create_context_template(
            template_id=template_id,
            template=template_content,
        )
        logger.info(f"Created context template: {template_id}")
        return template_id

    except Exception as e:
        logger.debug(f"Could not set up context template: {e}")
        return None


async def get_context(
    client: "AsyncZep",
    user_id: str,
    session_id: str,
    template_id: str | None = None,
    kwami_name: str = "Kwami",
    max_messages: int = 10,
    min_relevance: float = 0.5,
    include_facts: bool = True,
) -> MemoryContext:
    """Retrieve memory context from Zep for LLM injection.

    Tries context template first (recommended approach), then falls back
    to manual graph search if templates are unavailable.

    Args:
        client: The async Zep client.
        user_id: The Zep user ID.
        session_id: The current thread/session ID.
        template_id: Context template ID to use.
        kwami_name: Assistant name for filtering assistant-related facts.
        max_messages: Maximum recent messages to include.
        min_relevance: Minimum relevance score for facts.
        include_facts: Whether to include facts in context.

    Returns:
        MemoryContext with all available context.
    """
    context = MemoryContext()

    # Strategy 1: Use context template (preferred)
    if template_id:
        try:
            user_context = await client.thread.get_user_context(
                thread_id=session_id,
                template_id=template_id,
            )
            if user_context and user_context.context:
                context.context_block = user_context.context
                logger.debug("Retrieved context via template")
        except Exception as e:
            logger.debug(f"Template-based context failed, falling back: {e}")

    # Strategy 2: Fallback to thread context + graph search
    if not context.context_block:
        # Get thread context (summary).
        # `thread.get_context` does not exist in zep-cloud; the real call is
        # `get_user_context`, which is the same endpoint the template path uses
        # with template_id omitted. The old name raised AttributeError on every
        # call, so `summary` was always None and this whole fallback was dead.
        try:
            thread_context = await client.thread.get_user_context(
                thread_id=session_id,
                min_rating=min_relevance,
                mode="summary",
            )
            if thread_context and thread_context.context:
                context.summary = thread_context.context
        except Exception as e:
            logger.warning(f"Could not retrieve thread context: {e}")

        # Get facts via graph search
        if include_facts:
            try:
                facts_response = await client.graph.search(
                    user_id=user_id,
                    query="user information preferences interests goals",
                    scope="edges",
                    reranker="cross_encoder",
                    limit=20,
                )
                if facts_response and facts_response.edges:
                    assistant_lower = kwami_name.lower()
                    for edge in facts_response.edges:
                        fact = getattr(edge, "fact", None)
                        if not fact:
                            continue
                        # Skip facts about the assistant
                        if _is_assistant_fact(fact, assistant_lower):
                            continue
                        # Include temporal validity
                        invalid_at = getattr(edge, "invalid_at", None)
                        if invalid_at and str(invalid_at) != "present":
                            fact = f"{fact} (no longer valid since {invalid_at})"
                        context.facts.append(fact)
            except Exception as e:
                logger.debug(f"Could not retrieve facts via graph: {e}")

    # Always get recent messages (not part of context template)
    try:
        # `thread.get_messages` does not exist; the accessor is `thread.get`,
        # with `lastn` selecting the most recent turns. Until this was fixed,
        # `recent_messages` was unconditionally empty, which in turn made
        # `is_returning_user` and the recent-topics greeting unreachable.
        messages_response = await client.thread.get(
            thread_id=session_id,
            lastn=max_messages,
        )
        if messages_response and messages_response.messages:
            context.recent_messages = [
                {
                    "role": msg.role or msg.role_type,
                    "content": msg.content,
                }
                for msg in messages_response.messages
            ]
    except Exception as e:
        logger.debug(f"Could not retrieve thread messages: {e}")

    logger.debug(
        f"Retrieved context: template={'yes' if context.context_block else 'no'}, "
        f"{len(context.facts)} facts, {len(context.recent_messages)} messages"
    )
    return context


def _is_assistant_fact(fact: str, assistant_name_lower: str) -> bool:
    """Check if a fact is about the assistant rather than the user.

    Zep extracts facts from both user and assistant messages. Facts like
    "Kwami is an AI assistant" should not be injected as user facts.

    Args:
        fact: The fact string.
        assistant_name_lower: Lowercase assistant/kwami name.

    Returns:
        True if the fact is about the assistant.
    """
    fact_lower = fact.lower()

    # Skip facts that start with the assistant's name
    if fact_lower.startswith(assistant_name_lower + " "):
        return True

    # Skip facts describing the assistant's identity
    identity_phrases = [
        f"{assistant_name_lower} is",
        f"{assistant_name_lower} was",
        f"{assistant_name_lower} can",
        f"name is {assistant_name_lower}",
        f"called {assistant_name_lower}",
        f"named {assistant_name_lower}",
        f"i'm {assistant_name_lower}",
        f"i am {assistant_name_lower}",
    ]
    return any(phrase in fact_lower for phrase in identity_phrases)
