"""Input validation utilities for Kwami agent."""

import re
from typing import Any

from .logging import get_logger

logger = get_logger("validation")

#: A function name every provider the pipeline can switch to accepts: OpenAI
#: allows ``[a-zA-Z0-9_-]{1,64}``, Gemini wants a letter or underscore first.
#: A name outside both is not one tool lost -- the provider rejects the whole
#: request, every turn, so one user-added tool with a space in its name ended
#: the session.
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


def validate_tool_definition(tool_def: dict[str, Any]) -> bool:
    """Validate a tool definition has required fields.

    Args:
        tool_def: Tool definition dictionary.

    Returns:
        True if valid, False otherwise.
    """
    # Handle nested "function" format
    func_def = tool_def.get("function", tool_def)

    name = func_def.get("name")
    if not name:
        logger.warning("Tool definition missing 'name' field")
        return False

    if not isinstance(name, str):
        logger.warning("Tool 'name' must be a string, got %s", type(name))
        return False

    if not TOOL_NAME_PATTERN.match(name):
        logger.warning("Tool name %r is not one every model provider accepts; skipped", name[:80])
        return False

    # Description is optional but recommended
    description = func_def.get("description")
    if description and not isinstance(description, str):
        logger.warning("Tool 'description' must be a string, got %s", type(description))
        return False

    # Parameters should be a dict if present
    parameters = func_def.get("parameters")
    if parameters and not isinstance(parameters, dict):
        logger.warning("Tool 'parameters' must be a dict, got %s", type(parameters))
        return False

    if parameters and not _is_object_schema(parameters):
        logger.warning("Tool %r has parameters that are not an object schema; skipped", name)
        return False

    return True


def _is_object_schema(parameters: dict[str, Any]) -> bool:
    """Whether a parameters block is, or normalises to, an object schema.

    Providers require ``{"type": "object", "properties": {...}}`` with a schema
    -- a dict -- per property. The SDK's shorthand, a bare map of property name
    to schema, is accepted here and wrapped by `normalize_parameters`.
    ``{"city": "string"}``, the shape people type by hand, is neither: its
    property is a string, and every provider rejects the request it is in.
    """
    if isinstance(parameters.get("type"), str):
        if parameters["type"] != "object":
            return False
        properties = parameters.get("properties", {})
    else:
        properties = parameters
    if not isinstance(properties, dict):
        return False
    return all(isinstance(schema, dict) for schema in properties.values())


def normalize_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    """The object schema for a validated parameters block, shorthand included.

    Mirrors the SDK's `normalizeToolParameters`, so a client that sends the
    shorthand straight -- the SDK itself wraps it, others may not -- gives the
    model the same schema instead of one without a top-level type.
    """
    if not parameters:
        return {"type": "object", "properties": {}, "required": []}
    if isinstance(parameters.get("type"), str):
        return parameters
    return {"type": "object", "properties": parameters, "required": []}


# Mapping of camelCase keys to snake_case
CAMEL_TO_SNAKE_MAP = {
    # TTS
    "ttsProvider": "tts_provider",
    "ttsModel": "tts_model",
    "ttsVoice": "tts_voice",
    "ttsSpeed": "tts_speed",
    # LLM
    "llmProvider": "llm_provider",
    "llmModel": "llm_model",
    "llmTemperature": "llm_temperature",
    "maxTokens": "llm_max_tokens",
    # STT
    "sttProvider": "stt_provider",
    "sttModel": "stt_model",
    "sttLanguage": "stt_language",
    # Realtime
    "realtimeProvider": "realtime_provider",
    "realtimeModel": "realtime_model",
    "realtimeVoice": "realtime_voice",
    "realtimeModalities": "realtime_modalities",
    # Persona
    "systemPrompt": "system_prompt",
    "conversationStyle": "conversation_style",
    "responseLength": "response_length",
    "emotionalTone": "emotional_tone",
    "emotionalTraits": "emotional_traits",
    # Other
    "kwamiId": "kwami_id",
    "kwamiName": "kwami_name",
    "pipelineType": "pipeline_type",
}


def normalize_config_keys(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize config keys from camelCase to snake_case.

    Args:
        config: Configuration dictionary with potentially mixed key formats.

    Returns:
        New dictionary with snake_case keys.
    """
    normalized = {}

    for key, value in config.items():
        # Convert known camelCase keys
        snake_key = CAMEL_TO_SNAKE_MAP.get(key, key)

        # Recursively normalize nested dicts
        if isinstance(value, dict):
            value = normalize_config_keys(value)

        normalized[snake_key] = value

    return normalized
