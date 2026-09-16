"""Knowledge graph ontology configuration for Zep.

Defines entity types, edge types, and their relationships.
Properly constrains edge source/target to prevent orphan nodes
and ensure entities connect to the right nodes (not just User).

Key fixes over the previous implementation:
- Edge types have explicit source AND target constraints
- Entity models use meaningful domain-specific attributes
- Only defines types NOT covered by Zep defaults (User, Assistant,
  Preference, Location, Event, Object, Topic, Organization, Document)
"""

from typing import TYPE_CHECKING

from .utils import logger

if TYPE_CHECKING:
    from zep_cloud.client import AsyncZep


# ============================================================================
# Default Entity Type Definitions
# ============================================================================
# These are CUSTOM entity types that extend Zep's built-in defaults.
# Zep defaults already include: User, Assistant, Preference, Location,
# Event, Object, Topic, Organization, Document.
# We only define types NOT in the defaults, or override defaults that
# need domain-specific attributes.

DEFAULT_ENTITY_TYPES: list[dict] = [
    {
        "name": "Person",
        "description": (
            "A human being mentioned in conversation. Can be family, "
            "friends, colleagues, or public figures. Do NOT classify the "
            "user themselves as a Person entity."
        ),
        "fields": {
            "relationship": "Relationship to the user (e.g., friend, colleague, family, partner)",
        },
    },
    {
        "name": "Project",
        "description": (
            "A work project, personal initiative, creative endeavor, "
            "or ongoing effort the user is involved in."
        ),
        "fields": {
            "status": "Current status of the project (e.g., active, planned, completed, paused)",
        },
    },
    {
        "name": "Product",
        "description": (
            "Products, services, software, tools, or items the user "
            "uses or is interested in. Prioritize this over Object for "
            "named products and services."
        ),
        "fields": {
            "category": "Category of the product (e.g., software, hardware, service, app)",
        },
    },
    {
        "name": "Goal",
        "description": (
            "User goals, objectives, aspirations, or things they want to achieve or learn."
        ),
        "fields": {
            "timeframe": "Timeframe for the goal (e.g., short-term, long-term, ongoing)",
        },
    },
    {
        "name": "Procedure",
        "description": (
            "Multi-step instructions, workflows, or processes the "
            "agent should follow when performing tasks."
        ),
        "fields": {
            "detail": "Description of the procedure or workflow steps",
        },
    },
]


# ============================================================================
# Default Edge Type Definitions
# ============================================================================
# Each edge type specifies source AND target entity types to ensure
# relationships connect to the correct entities.
# This prevents orphan nodes and incorrect User-centric relationships.

DEFAULT_EDGE_TYPES: list[dict] = [
    {
        "name": "KNOWS",
        "description": "The user knows or is acquainted with a person.",
        "fields": {
            "context": "How the user knows this person (e.g., work, school, family)",
        },
        "source": "User",
        "target": "Person",
    },
    {
        "name": "WORKS_AT",
        "description": "Employment or work relationship with an organization.",
        "fields": {
            "role": "The user's role or position at the organization",
        },
        "source": "User",
        "target": "Organization",
    },
    {
        "name": "LIVES_IN",
        "description": "The user's current residence or location.",
        "fields": {
            "detail": "Details about living arrangement or duration",
        },
        "source": "User",
        "target": "Location",
    },
    {
        "name": "INTERESTED_IN",
        "description": "Topics, activities, or things the user is interested in.",
        "fields": {
            "detail": "Details about the nature of the interest",
        },
        "source": "User",
        "target": "Topic",
    },
    {
        "name": "WORKING_ON",
        "description": "Projects or tasks the user is actively working on.",
        "fields": {
            "detail": "Details about the user's involvement",
        },
        "source": "User",
        "target": "Project",
    },
    {
        "name": "HAS_GOAL",
        "description": "Goals, desires, or aspirations the user has expressed.",
        "fields": {
            "detail": "Details about the goal or aspiration",
        },
        "source": "User",
        "target": "Goal",
    },
    {
        "name": "USES",
        "description": "Products, tools, or services the user uses.",
        "fields": {
            "detail": "How or why the user uses this product",
        },
        "source": "User",
        "target": "Product",
    },
    {
        "name": "PREFERS",
        "description": "Preferences the user has expressed about anything.",
        "fields": {
            "detail": "Details about the preference",
        },
        "source": "User",
        # No target - preferences can be about anything
    },
]


def _build_entity_models(
    entity_types: list[dict],
) -> dict:
    """Build Zep EntityModel subclasses from dict definitions.

    Each entity type becomes a Pydantic model class inheriting from EntityModel.
    The class docstring is used by Zep as the entity type description.

    Args:
        entity_types: List of entity type dicts with 'name', 'description', and 'fields'.

    Returns:
        Dict mapping entity type names to their model classes.
    """
    try:
        from pydantic import Field
        from zep_cloud.external_clients.ontology import EntityModel, EntityText
    except ImportError:
        logger.warning("Zep ontology SDK classes not available")
        return {}

    entities = {}
    for et in entity_types:
        name = et["name"]
        desc = et.get("description", name)
        fields = et.get("fields", {})

        # Build annotations and field defaults for the model
        annotations = {}
        field_defaults = {}
        for field_name, field_desc in fields.items():
            annotations[field_name] = EntityText
            field_defaults[field_name] = Field(description=field_desc, default=None)

        # If no custom fields defined, add a generic 'detail' field
        # (Zep requires at least one custom property per entity/edge type)
        if not fields:
            annotations["detail"] = EntityText
            field_defaults["detail"] = Field(description=desc, default=None)

        model_cls = type(
            name,
            (EntityModel,),
            {
                "__doc__": desc,
                "__annotations__": annotations,
                **field_defaults,
            },
        )
        entities[name] = model_cls

    return entities


def _build_edge_models(
    edge_types: list[dict],
) -> dict:
    """Build Zep EdgeModel subclasses from dict definitions.

    Each edge type becomes a Pydantic model class inheriting from EdgeModel,
    with proper source/target constraints via EntityEdgeSourceTarget.

    Args:
        edge_types: List of edge type dicts with 'name', 'description',
                    'fields', 'source', and optional 'target'.

    Returns:
        Dict mapping edge type names to (model_class, [constraints]) tuples.
    """
    try:
        from pydantic import Field
        from zep_cloud import EntityEdgeSourceTarget
        from zep_cloud.external_clients.ontology import EdgeModel, EntityText
    except ImportError:
        logger.warning("Zep ontology SDK classes not available")
        return {}

    edges = {}
    for edge in edge_types:
        name = edge["name"]
        desc = edge.get("description", name)
        fields = edge.get("fields", {})
        source = edge.get("source", "User")
        target = edge.get("target")

        # Build annotations and field defaults
        annotations = {}
        field_defaults = {}
        for field_name, field_desc in fields.items():
            annotations[field_name] = EntityText
            field_defaults[field_name] = Field(description=field_desc, default=None)

        # Ensure at least one field
        if not fields:
            annotations["detail"] = EntityText
            field_defaults["detail"] = Field(description=desc, default=None)

        model_cls = type(
            name,
            (EdgeModel,),
            {
                "__doc__": desc,
                "__annotations__": annotations,
                **field_defaults,
            },
        )

        # Build source/target constraint
        constraint_kwargs = {"source": source}
        if target:
            constraint_kwargs["target"] = target

        edges[name] = (model_cls, [EntityEdgeSourceTarget(**constraint_kwargs)])

    return edges


async def configure_ontology(
    client: "AsyncZep",
    user_id: str,
    entity_types: list[dict] | None = None,
    edge_types: list[dict] | None = None,
) -> bool:
    """Configure the knowledge graph ontology (entity and edge types).

    Uses the Zep v3 SDK format with EntityModel/EdgeModel classes and
    proper EntityEdgeSourceTarget constraints for each edge type.

    Args:
        client: The async Zep client.
        user_id: The Zep user ID to apply the ontology to.
        entity_types: Custom entity type definitions (uses defaults if None).
        edge_types: Custom edge type definitions (uses defaults if None).

    Returns:
        True if ontology was configured successfully.
    """
    entity_types = entity_types or DEFAULT_ENTITY_TYPES
    edge_types = edge_types or DEFAULT_EDGE_TYPES

    try:
        entities = _build_entity_models(entity_types)
        edges = _build_edge_models(edge_types)

        if not entities and not edges:
            logger.warning("No ontology models could be built, skipping configuration")
            return False

        await client.graph.set_ontology(
            entities=entities,
            edges=edges,
            user_ids=[user_id],
        )

        entity_names = [e["name"] for e in entity_types]
        edge_names = [e["name"] for e in edge_types]
        logger.info(
            f"Configured ontology for {user_id}: "
            f"{len(entity_types)} entity types ({', '.join(entity_names[:5])}...), "
            f"{len(edge_types)} edge types ({', '.join(edge_names[:5])}...)"
        )
        return True

    except ImportError:
        logger.warning("Zep ontology SDK classes not available, skipping ontology configuration")
        return False
    except Exception as e:
        logger.warning(f"Could not configure ontology (may not be supported on your plan): {e}")
        return False
