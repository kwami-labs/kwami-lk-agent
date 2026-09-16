"""Live reconfiguration with real providers.

The frontend reconfigures a running agent mid-call: voice, LLM, soul and tools
all change through data messages. Two production bugs lived on this path --
a `tools` update assigned `agent._tools` directly and so deleted all 22
built-ins for the rest of the session, and every reconfiguration built a fresh
Zep client that could never be closed.

These build real provider clients, so they catch the case where a config the
frontend can actually send produces a pipeline that cannot be constructed.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.domain.config import KwamiConfig, KwamiSoulConfig
from src.handlers.config_handler import handle_full_config, update_tools
from src.runtime.pipeline import create_agent_from_config
from src.session import SessionState

pytestmark = pytest.mark.live

CLIENT_TOOL = {
    "name": "set_theme",
    "description": "Change the interface theme",
    "parameters": {
        "type": "object",
        "properties": {"mode": {"type": "string"}},
        "required": ["mode"],
    },
}


class FakeSession:
    def __init__(self) -> None:
        self.agent: Any = None

    def update_agent(self, agent: Any) -> None:
        self.agent = agent


def _tool_names(tools: Any) -> set[str]:
    return {tool.info.name for tool in tools}


async def test_a_real_pipeline_is_built_from_a_frontend_config(openai_key: str) -> None:
    """The exact shape the frontend sends, through the real factories."""
    session, state = FakeSession(), SessionState()
    message = {
        "voice": {
            "tts": {"provider": "openai", "model": "tts-1", "voice": "nova", "speed": 1.0},
            "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.4},
            "stt": {"provider": "deepgram", "language": "en"},
        },
        "soul": {"name": "Ada", "personality": "a precise assistant"},
    }

    await handle_full_config(
        session, state, message, vad=None, create_agent_fn=create_agent_from_config
    )

    assert session.agent is not None, "the config produced no agent"
    assert session.agent.llm is not None
    assert session.agent.tts is not None


async def test_zero_temperature_survives_into_a_real_client(openai_key: str) -> None:
    """A deterministic setting the truthiness guard used to discard."""
    session, state = FakeSession(), SessionState()
    message = {"voice": {"llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0}}}

    await handle_full_config(
        session, state, message, vad=None, create_agent_fn=create_agent_from_config
    )

    assert session.agent.kwami_config.voice.llm_temperature == 0.0


async def test_a_tools_update_keeps_every_builtin_on_a_real_agent(openai_key: str) -> None:
    """The regression that silently disabled web_search and all navigation tools."""
    agent = create_agent_from_config(
        KwamiConfig(soul=KwamiSoulConfig(name="Ada")), vad=None, skip_greeting=True
    )
    before = _tool_names(agent.tools)
    assert "web_search" in before

    await update_tools(agent, [CLIENT_TOOL])

    after = _tool_names(agent.tools)
    assert before <= after, f"built-ins were dropped: {sorted(before - after)}"
    assert "set_theme" in after


async def test_the_soul_reaches_the_system_prompt_of_a_real_agent(openai_key: str) -> None:
    session, state = FakeSession(), SessionState()
    message = {"soul": {"name": "Ada", "personality": "a laconic archivist"}}

    await handle_full_config(
        session, state, message, vad=None, create_agent_fn=create_agent_from_config
    )

    prompt = session.agent._build_system_prompt()
    assert "Ada" in prompt
    assert "laconic archivist" in prompt
