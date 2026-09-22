"""Message handlers for Kwami agent."""

from .config_handler import (
    handle_config_update,
    handle_full_config,
    update_llm,
    update_persona,
    update_soul,
    update_tools,
    update_voice,
)
from .realtime import (
    REALTIME_PIPELINE,
    STANDARD_PIPELINE,
    apply_realtime_fields,
    has_realtime_keys,
    normalize_pipeline_type,
    requested_pipeline,
    switch_pipeline,
    update_realtime,
)
from .tool_handler import handle_tool_result

__all__ = [
    "handle_full_config",
    "handle_config_update",
    "update_voice",
    "update_llm",
    "update_soul",
    "update_tools",
    "update_persona",
    "handle_tool_result",
    # realtime / pipeline switching
    "update_realtime",
    "switch_pipeline",
    "requested_pipeline",
    "has_realtime_keys",
    "normalize_pipeline_type",
    "apply_realtime_fields",
    "REALTIME_PIPELINE",
    "STANDARD_PIPELINE",
]
