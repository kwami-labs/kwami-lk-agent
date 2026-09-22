"""Importing a `livekit-plugins-*` extra that may not be installed.

Four providers -- Google, Anthropic, Groq, AssemblyAI -- ship as separate
distributions declared as optional extras. Each factory guarded them the same
way::

    try:
        from livekit.plugins import google
    except ImportError:
        google = None  # type: ignore[assignment]

which cannot be made to type-check in both worlds at once. With the extra
absent, mypy flags the import (`livekit.plugins has no attribute "google"`);
with it present, `warn_unused_ignores` flags the ignore that was silencing
that. Twelve of the eighty-one errors in the type-check backlog were this one
pattern repeated, and no arrangement of `type: ignore` codes fixes it, because
which branch is wrong depends on what happens to be installed.

Resolving the module by name sidesteps the question. The result is `Any`, which
is honest: an optional plugin's surface is not knowable at type-check time in an
environment that does not have it. Callers already guard on `is None` before
touching it, which is the check that actually matters at runtime.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..utils.logging import get_logger

logger = get_logger("factories.optional")

#: Where the optional plugins live. Kept as a constant so a namespace change in
#: livekit-agents is one edit rather than four.
PLUGIN_NAMESPACE = "livekit.plugins"


def optional_module(module_path: str) -> Any:
    """Import `module_path`, or return None when it is not installed.

    Only `ImportError` is caught. A plugin that is present but raises on import
    -- a broken build, a missing native library -- is a real fault, and
    swallowing it here would surface much later as "this provider silently
    falls back to OpenAI" with nothing explaining why.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("Optional module %s is not installed", module_path)
        return None


def optional_plugin(name: str) -> Any:
    """Import the `livekit.plugins.<name>` extra, or None when it is absent."""
    return optional_module(f"{PLUGIN_NAMESPACE}.{name}")
