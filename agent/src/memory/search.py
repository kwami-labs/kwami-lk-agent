"""Graph search and entity retrieval for Zep knowledge graphs.

Provides typed entity search, user identity extraction, and
general graph querying capabilities.
"""

import re
from collections import Counter
from typing import TYPE_CHECKING

from .utils import logger

if TYPE_CHECKING:
    from zep_cloud.client import AsyncZep


async def search_thread(
    client: "AsyncZep",
    session_id: str,
    query: str,
    limit: int = 5,
    user_id: str = "",
) -> list[dict]:
    """Search a user's memories for relevant context.

    Args:
        client: The async Zep client.
        session_id: The thread/session ID, reported back on each result.
        query: Search query.
        limit: Maximum number of results.
        user_id: The Zep user whose graph is searched. Required.

    Returns:
        List of search results with content and score.
    """
    if not user_id:
        logger.warning("search_thread called without a user_id; cannot search the graph")
        return []

    try:
        # zep-cloud has no `thread.search`. Recall is served by the user's
        # knowledge graph, which is where facts actually live -- the old call
        # raised AttributeError every time, so `recall_memories` always
        # answered "I don't have any memories about that yet".
        results = await client.graph.search(
            user_id=user_id,
            query=query,
            scope="edges",
            limit=limit,
        )

        edges = getattr(results, "edges", None) or []
        return [
            {
                "content": fact,
                "score": getattr(edge, "score", 0) or 0,
                "thread_id": session_id,
            }
            for edge in edges
            if (fact := getattr(edge, "fact", None))
        ]

    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        return []


async def get_entities_by_type(
    client: "AsyncZep",
    user_id: str,
    entity_type: str,
    limit: int = 20,
) -> list[dict]:
    """Get all entities of a specific type from the graph.

    Args:
        client: The async Zep client.
        user_id: The Zep user ID.
        entity_type: The entity type name (e.g., "Person", "Project").
        limit: Maximum number of results.

    Returns:
        List of entities of the specified type.
    """
    try:
        nodes_response = await client.graph.node.get_by_user_id(
            user_id=user_id,
            limit=limit * 2,  # Fetch more to filter
        )

        entities = []
        if nodes_response:
            for node in nodes_response:
                node_labels = list(node.labels) if hasattr(node, "labels") and node.labels else []
                if any(label.lower() == entity_type.lower() for label in node_labels):
                    entities.append(
                        {
                            "name": getattr(node, "name", ""),
                            "type": node_labels[0] if node_labels else "entity",
                            "labels": node_labels,
                            "summary": getattr(node, "summary", ""),
                            "uuid": (getattr(node, "uuid_", None) or getattr(node, "uuid", None)),
                            "created_at": (
                                str(node.created_at)
                                if hasattr(node, "created_at") and node.created_at
                                else None
                            ),
                        }
                    )
                    if len(entities) >= limit:
                        break

        return entities

    except Exception as e:
        logger.debug(f"Failed to get entities by type '{entity_type}': {e}")
        return []


# ============================================================================
# User Identity Extraction
# ============================================================================

# Words that are definitely NOT user names
_EXCLUDED_NAMES = {
    "the",
    "a",
    "an",
    "user",
    "assistant",
    "system",
    "ai",
    "today",
    "tomorrow",
    "yesterday",
    "now",
    "then",
    "this",
    "that",
    "they",
    "their",
    "he",
    "she",
    "it",
    "we",
    "you",
    "i",
}

# Regex patterns for extracting names from facts (ordered by confidence)
_NAME_PATTERNS = [
    r"(?:user'?s?\s+)?name\s+is\s+([A-Z][a-z]+)",
    r"(?:i'?m|i am|my name is)\s+([A-Z][a-z]+)",
    r"called\s+([A-Z][a-z]+)",
    r"goes by\s+([A-Z][a-z]+)",
    r"identified (?:as|themselves as)\s+([A-Z][a-z]+)",
    r"introduced (?:as|themselves as)\s+([A-Z][a-z]+)",
    r"([A-Z][a-z]+)\s+(?:is the user|is the human)",
    r"the user(?:'s name)? is\s+([A-Z][a-z]+)",
]

# Pattern for facts starting with a capitalized name followed by a verb
_NAME_VERB_PATTERN = re.compile(
    r"^([A-Z][a-z]+)\s+(?:has|wants|likes|said|asked|mentioned|is|"
    r"intends|lives|works|prefers|enjoys)"
)


def _is_valid_name(name: str, extra_excluded: set[str] | None = None) -> bool:
    """Check if a string is likely a valid user name."""
    if not name or len(name) < 2:
        return False
    excluded = _EXCLUDED_NAMES | (extra_excluded or set())
    if name.lower() in excluded:
        return False
    if not name[0].isupper() or not name.isalpha():
        return False
    return True


def _extract_name_from_fact(fact: str, extra_excluded: set[str] | None = None) -> str | None:
    """Extract a user name from a fact string using regex patterns."""
    if not fact:
        return None

    for pattern in _NAME_PATTERNS:
        match = re.search(pattern, fact, re.IGNORECASE)
        if match:
            name = match.group(1).capitalize()
            if _is_valid_name(name, extra_excluded):
                return name

    return None


async def get_user_name(
    client: "AsyncZep",
    user_id: str,
    kwami_name: str = "Kwami",
) -> str | None:
    """Try to extract the user's name from the knowledge graph.

    Uses multiple strategies in order of confidence:
    1. Search for explicit name statements in facts
    2. Look for patterns in fact subjects
    3. Check graph nodes for user identity

    Args:
        client: The async Zep client.
        user_id: The Zep user ID.
        kwami_name: The assistant name (to exclude from results).

    Returns:
        The user's name if found, None otherwise.
    """
    extra_excluded = {"kwami", kwami_name.lower()}

    # Strategy 1: Search for name-specific facts
    name_queries = [
        "user name called identified",
        "my name is",
        "person identity who",
    ]

    for query in name_queries:
        try:
            search_result = await client.graph.search(
                user_id=user_id,
                query=query,
                scope="edges",
                limit=10,
            )

            if search_result and search_result.edges:
                for edge in search_result.edges:
                    fact = getattr(edge, "fact", "") or ""
                    name = _extract_name_from_fact(fact, extra_excluded)
                    if name:
                        logger.info(f"Found user name from fact: {name}")
                        return name
        except Exception as e:
            logger.debug(f"Graph search failed for query '{query}': {e}")

    # Strategy 2: Look for name patterns in fact subjects
    try:
        search_result = await client.graph.search(
            user_id=user_id,
            query="person preferences likes wants",
            scope="edges",
            limit=20,
        )

        potential_names = []
        if search_result and search_result.edges:
            for edge in search_result.edges:
                fact = getattr(edge, "fact", "") or ""
                match = _NAME_VERB_PATTERN.match(fact)
                if match and _is_valid_name(match.group(1), extra_excluded):
                    potential_names.append(match.group(1))

        if potential_names:
            # Every candidate was validated on the way into the list, so
            # re-checking it here could only ever be true; the most common
            # one is the answer.
            name, count = Counter(potential_names).most_common(1)[0]
            logger.info(f"Found user name from patterns: {name} (appeared {count} times)")
            return name
    except Exception as e:
        logger.debug(f"Pattern-based name search failed: {e}")

    # Strategy 3: Check graph nodes for user identity
    try:
        nodes_result = await client.graph.node.get_by_user_id(user_id=user_id)
        if nodes_result:
            for node in nodes_result:
                label = getattr(node, "label", "") or getattr(node, "name", "") or ""
                summary = getattr(node, "summary", "") or ""
                node_type = getattr(node, "type", "") or ""

                # Check if this is a person/user node
                if node_type.lower() in ("person", "user", "human"):
                    if _is_valid_name(label, extra_excluded):
                        logger.info(f"Found user name from graph node (type={node_type}): {label}")
                        return label

                # Check summary for user identity indicators
                summary_lower = summary.lower()
                if (
                    ("user" in summary_lower or "person" in summary_lower)
                    and ("name" in summary_lower or "called" in summary_lower)
                    and _is_valid_name(label, extra_excluded)
                ):
                    logger.info(f"Found user name from node summary: {label}")
                    return label
    except Exception as e:
        logger.debug(f"Could not get graph nodes for name search: {e}")

    logger.debug("No user name found in memory")
    return None
