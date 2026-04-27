"""Input validation utilities for Kwami agent."""

from typing import Any, Dict, Optional

from .logging import get_logger

logger = get_logger("validation")


def validate_tool_definition(tool_def: Dict[str, Any]) -> bool:
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
        logger.warning(f"Tool 'name' must be a string, got {type(name)}")
        return False
    
    # Description is optional but recommended
    description = func_def.get("description")
    if description and not isinstance(description, str):
        logger.warning(f"Tool 'description' must be a string, got {type(description)}")
        return False
    
    # Parameters should be a dict if present
    parameters = func_def.get("parameters")
    if parameters and not isinstance(parameters, dict):
        logger.warning(f"Tool 'parameters' must be a dict, got {type(parameters)}")
        return False
    
    return True


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


def normalize_config_keys(config: Dict[str, Any]) -> Dict[str, Any]:
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


def safe_get(
    config: Dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """Safely get a value from nested config, trying multiple key formats.
    
    Args:
        config: Configuration dictionary.
        *keys: Keys to try (e.g., "tts_provider", "ttsProvider").
        default: Default value if none found.
        
    Returns:
        Found value or default.
    """
    for key in keys:
        if key in config:
            return config[key]
    return default
