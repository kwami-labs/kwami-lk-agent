"""The agent, driven by a real LLM, with no room and no audio.

`AgentSession.start(room=...)` is optional in 1.3.12, so behaviour can be
exercised in text. These assert the things unit tests structurally cannot: that
a real model, given this system prompt and these tool schemas, actually picks
the tool we intended.
"""

from __future__ import annotations

from typing import Any

import pytest
from livekit.agents import AgentSession, mock_tools

from src.agent import KwamiAgent
from src.domain.config import KwamiConfig, KwamiSoulConfig

pytestmark = pytest.mark.live


def build_agent(llm: Any, **soul_kwargs: Any) -> KwamiAgent:
    soul = KwamiSoulConfig(name="Kwami", personality="a concise, friendly assistant", **soul_kwargs)
    return KwamiAgent(KwamiConfig(soul=soul), llm=llm, skip_greeting=True)


@pytest.fixture
def agent(real_llm: Any) -> KwamiAgent:
    return build_agent(real_llm)


async def test_the_agent_replies_to_a_plain_question(agent: KwamiAgent, real_llm: Any) -> None:
    """The base case: a real LLM, a real system prompt, an actual answer."""
    async with AgentSession(llm=real_llm) as session:
        await session.start(agent)
        result = await session.run(user_input="Say hello and nothing else.")

    result.expect.next_event(type="message")


async def test_the_agent_asks_the_clock_rather_than_guessing(
    agent: KwamiAgent, real_llm: Any
) -> None:
    """A question about the time must route to the tool, not to the model's prior."""
    async with AgentSession(llm=real_llm) as session:
        await session.start(agent)
        result = await session.run(user_input="What time is it right now?")

    result.expect.contains_function_call(name="get_current_time")


async def test_a_browse_request_reaches_navigate_to(agent: KwamiAgent, real_llm: Any) -> None:
    """Stubbed so no cloud browser is started and nothing is billed."""
    with mock_tools(KwamiAgent, {"navigate_to": lambda *a, **k: "Opened the page."}):
        async with AgentSession(llm=real_llm) as session:
            await session.start(agent)
            result = await session.run(user_input="Please open wikipedia.org for me.")

    result.expect.contains_function_call(name="navigate_to")


async def test_a_shopping_request_reaches_a_search_tool(agent: KwamiAgent, real_llm: Any) -> None:
    """The prompt tells the model to prefer product_search for things to buy."""
    with mock_tools(
        KwamiAgent,
        {
            "product_search": lambda *a, **k: "Found 3 products.",
            "web_search": lambda *a, **k: "Found 3 results.",
        },
    ):
        async with AgentSession(llm=real_llm) as session:
            await session.start(agent)
            result = await session.run(user_input="Find me a black leather bag to buy.")

    events = [e for e in result.events if getattr(e, "type", None) == "function_call"]
    called = {getattr(e.item, "name", None) for e in events}
    assert called & {"product_search", "web_search"}, f"no search tool was called: {called}"


async def test_the_soul_shapes_the_answer(real_llm: Any, judge_llm: Any) -> None:
    """The system prompt is not decoration: a pirate soul must sound like one."""
    agent = build_agent(real_llm, system_prompt="You are a pirate. Always talk like a pirate.")

    async with AgentSession(llm=real_llm) as session:
        await session.start(agent)
        result = await session.run(user_input="Greet me.")

    await result.expect.contains_message(role="assistant").judge(
        judge_llm, intent="A greeting written in exaggerated pirate speech."
    )
