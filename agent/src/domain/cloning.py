"""Copying config trees without sharing mutable state.

`dataclasses.replace` is a *shallow* copy: the new object gets new scalar
fields but the very same `soul`, `memory` and `tools` objects. Agent
reconfiguration used it to build the next config, so the new agent and the
discarded one shared a soul -- and a later mutation on one was visible on the
other, backwards in time.
"""

from __future__ import annotations

import copy
from typing import TypeVar

T = TypeVar("T")

__all__ = ["clone_config"]


def clone_config(config: T) -> T:
    """Return a fully independent copy of a config tree."""
    return copy.deepcopy(config)
