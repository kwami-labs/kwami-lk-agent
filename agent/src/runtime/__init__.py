"""Session runtime: the job lifecycle, decomposed into testable pieces.

`pipeline` is deliberately NOT re-exported here. It imports KwamiAgent, and
agent.py imports `runtime.container`, so re-exporting pipeline from this
__init__ turns a legal one-way dependency into an import cycle. Import it by
module path instead: `from .runtime.pipeline import create_agent_from_config`.
"""

from .container import AgentDeps, deps_from_context, room_from_context
from .dispatch import DataMessageRouter, decode_data_message, route_metrics
from .lifecycle import apply_runtime_config, resolve_identity_on_join
from .reconfigure import Reconfigurator, reconfigurator_from_context

__all__ = [
    "AgentDeps",
    "apply_runtime_config",
    "DataMessageRouter",
    "deps_from_context",
    "decode_data_message",
    "Reconfigurator",
    "reconfigurator_from_context",
    "room_from_context",
    "resolve_identity_on_join",
    "route_metrics",
]
