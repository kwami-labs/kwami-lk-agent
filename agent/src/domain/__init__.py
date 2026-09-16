"""Pure domain logic: no I/O, no SDK imports, no environment reads.

Everything here is a plain function over plain data, which is what makes it
testable without mocks -- and what makes the coverage target reachable.
"""

from .cloning import clone_config
from .parsing import boolean, integer, number, section, text, value_from_keys
from .prompt import MAX_SYSTEM_MEMORY_CONTEXT_CHARS, build_system_prompt

__all__ = [
    "MAX_SYSTEM_MEMORY_CONTEXT_CHARS",
    "boolean",
    "build_system_prompt",
    "clone_config",
    "integer",
    "number",
    "section",
    "text",
    "value_from_keys",
]
