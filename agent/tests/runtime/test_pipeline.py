"""Building an agent from a config: the branch that chooses the whole pipeline.

This decided realtime-vs-standard for every session and was never exercised
offline. Its realtime branch was dead on arrival for months -- it passed
`openai.realtime.ServerVadOptions`, a name that has never existed in the 1.3.x
plugin -- and nothing caught it because the function lived at module scope in
`main.py` next to the worker singleton, so importing it started a worker.
"""

from __future__ import annotations

import pytest

from src.agent import KwamiAgent
from src.domain.config import KwamiConfig, KwamiSoulConfig
from src.runtime.pipeline import create_agent_from_config


@pytest.fixture
def keys(fake_key) -> None:
    fake_key("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "GOOGLE_API_KEY")


def _config(**voice_kwargs) -> KwamiConfig:
    config = KwamiConfig(soul=KwamiSoulConfig(name="Ada"))
    for key, value in voice_kwargs.items():
        setattr(config.voice, key, value)
    return config


def test_the_standard_pipeline_builds_stt_llm_and_tts(keys: None) -> None:
    agent = create_agent_from_config(_config(pipeline_type="standard"), vad=None)

    assert isinstance(agent, KwamiAgent)
    assert agent.stt is not None
    assert agent.llm is not None
    assert agent.tts is not None


def test_the_realtime_pipeline_builds_a_single_model(keys: None) -> None:
    """The branch that raised AttributeError for every realtime session."""
    agent = create_agent_from_config(_config(pipeline_type="realtime"), vad=None)

    assert isinstance(agent, KwamiAgent)
    assert agent.llm is not None


def test_the_realtime_branch_is_selected_only_by_its_own_flag(keys: None) -> None:
    standard = create_agent_from_config(_config(pipeline_type="standard"), vad=None)
    realtime = create_agent_from_config(_config(pipeline_type="realtime"), vad=None)

    assert standard.stt is not None, "standard pipeline must have STT"
    assert realtime.stt is None, "realtime pipeline should not build a separate STT"


def test_the_config_is_carried_onto_the_agent(keys: None) -> None:
    agent = create_agent_from_config(_config(), vad=None)
    assert agent.kwami_config.soul.name == "Ada"


def test_memory_and_skip_greeting_are_threaded_through(keys: None) -> None:
    sentinel = object()
    agent = create_agent_from_config(_config(), vad=None, memory=sentinel, skip_greeting=True)

    assert agent._memory is sentinel
    assert agent._skip_greeting is True


def test_the_built_in_tools_survive_construction(keys: None, all_tools_available) -> None:
    """A built agent must arrive with its 22 built-ins already discovered."""
    agent = create_agent_from_config(_config(), vad=None)
    names = {tool.info.name for tool in agent.tools}

    for expected in ("web_search", "navigate_to", "get_current_time"):
        assert expected in names
