"""Letting the agent change its own voice, model and pipeline mid-conversation.

Everything needed to rebuild a pipeline already existed -- `create_agent_from_config`,
`SessionState.update_agent`, the live `update_options` paths -- but all of it
was reachable only from a data message sent *by the frontend*. The agent itself
had no handle on any of it, so "switch to Claude" or "use your realtime voice
Cedar" could only ever be answered with a description of which panel to open.

This is that handle. It is attached to `AgentDeps`, which the framework threads
into every tool's `RunContext`, so a tool asks for it the same way it asks for
the room.

Two mechanics matter here and neither is incidental:

* **Live first.** Voice, speed and temperature go over the open socket. A
  rebuild drops the realtime connection and re-greets, which the user hears; an
  `update_options` push does not.
* **Handoff, not self-swap.** When a rebuild *is* required, the new agent is
  returned to the framework rather than installed here. The framework installs
  it after the tool's own output has been collected, so the sentence explaining
  the switch survives the switch. Calling `session.update_agent` from inside the
  tool discards the turn that called it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..domain import KwamiVoiceConfig, clone_config
from ..utils.logging import get_logger

logger = get_logger("reconfigure")

REALTIME_PIPELINE = "realtime"
STANDARD_PIPELINE = "standard"


@dataclass
class Reconfigurator:
    """Rebuilds the running pipeline on request, from inside a tool call.

    Holds the three things `create_agent_from_config` needs that a tool cannot
    otherwise reach: the session state that owns the browser and the usage
    tracker, the prewarmed VAD, and the factory itself.
    """

    state: Any
    vad: Any = None
    create_agent_fn: Any = None

    @property
    def is_available(self) -> bool:
        """False on a deployment wired up without a factory (e.g. a bare test)."""
        return self.create_agent_fn is not None and self.state is not None

    def current_voice(self) -> KwamiVoiceConfig | None:
        agent = getattr(self.state, "current_agent", None)
        if agent is None:
            return None
        return agent.kwami_config.voice

    def build(self, voice_config: KwamiVoiceConfig, *, agent: Any = None) -> Any:
        """Build the replacement agent and carry session resources onto it.

        The returned agent is NOT installed: the caller hands it back to the
        framework, which installs it once the tool output is in the transcript.
        """
        source = agent if agent is not None else self.state.current_agent
        new_config = clone_config(source.kwami_config)
        new_config.voice = voice_config
        new_agent = self.create_agent_fn(
            new_config,
            self.vad,
            getattr(source, "_memory", None),
            skip_greeting=True,
        )
        return self.state.prepare_handoff(new_agent)

    def mutated_voice(self, agent: Any, **changes: Any) -> KwamiVoiceConfig:
        """A copy of the agent's voice config with `changes` applied."""
        voice_config = replace(agent.kwami_config.voice)
        for key, value in changes.items():
            setattr(voice_config, key, value)
        return voice_config


def reconfigurator_from_context(context: Any) -> Reconfigurator | None:
    """Pull the reconfigure handle out of a tool's RunContext, or None."""
    from .container import deps_from_context

    deps = deps_from_context(context)
    if deps is None:
        return None
    reconfigurator = getattr(deps, "reconfigure", None)
    if reconfigurator is None or not reconfigurator.is_available:
        return None
    return reconfigurator
