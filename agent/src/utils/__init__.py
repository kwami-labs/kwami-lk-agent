"""Utility functions for Kwami agent."""

from .logging import get_logger, log_error
from .provider import (
    detect_provider_change,
    detect_tts_provider_from_model,
    detect_tts_provider_from_voice,
    strip_model_prefix,
)
from .room import (
    get_other_agents,
    is_agent_participant,
    resolve_user_identity,
    should_disconnect_as_duplicate,
)
from .validation import normalize_config_keys, validate_tool_definition

__all__ = [
    "get_logger",
    "log_error",
    "strip_model_prefix",
    "detect_tts_provider_from_model",
    "detect_tts_provider_from_voice",
    "detect_provider_change",
    "get_other_agents",
    "is_agent_participant",
    "resolve_user_identity",
    "should_disconnect_as_duplicate",
    "validate_tool_definition",
    "normalize_config_keys",
]
