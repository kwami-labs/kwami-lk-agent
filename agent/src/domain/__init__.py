"""Pure domain layer: configuration, prompt assembly, usage accounting, errors.

Nothing here performs I/O, imports a provider SDK, or reads the environment.
That is the point: this is the code most likely to hold a bug, and it should be
testable without a mock, a network, or an event loop.
"""

from .capabilities import available_capability_names, build_capability_guidance
from .clock import current_time_phrase, resolve_zone
from .cloning import clone_config
from .config import (
    KwamiConfig,
    KwamiMemoryConfig,
    KwamiPersonaConfig,
    KwamiSoulConfig,
    KwamiVoiceConfig,
    get_preset_config,
)
from .errors import (
    ConfigurationError,
    KwamiError,
    ResourceNotFoundError,
    VoiceProviderError,
)
from .models import ModelChoice, resolve_model
from .parsing import boolean, integer, number, section, text, value_from_keys
from .prompt import MAX_SYSTEM_MEMORY_CONTEXT_CHARS, build_system_prompt
from .tool_result import ToolResult, refusal, was_refused
from .usage import UsageTracker

__all__ = [
    # config
    "KwamiConfig",
    "KwamiVoiceConfig",
    "KwamiSoulConfig",
    "KwamiMemoryConfig",
    "KwamiPersonaConfig",
    "get_preset_config",
    "clone_config",
    # clock
    "current_time_phrase",
    "resolve_zone",
    # tool results
    "ToolResult",
    "refusal",
    "was_refused",
    # wire parsing
    "boolean",
    "integer",
    "number",
    "section",
    "text",
    "value_from_keys",
    # prompt
    "build_system_prompt",
    "MAX_SYSTEM_MEMORY_CONTEXT_CHARS",
    "build_capability_guidance",
    "available_capability_names",
    # model resolution
    "ModelChoice",
    "resolve_model",
    # usage
    "UsageTracker",
    # errors
    "KwamiError",
    "ConfigurationError",
    "VoiceProviderError",
    "ResourceNotFoundError",
]
