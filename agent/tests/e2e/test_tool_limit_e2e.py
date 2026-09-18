"""The provider's tool ceiling, checked against the provider.

`MAX_TOOLS_PER_REQUEST` is a constant, and a constant about someone else's
service is a guess until something checks it. The agent currently sends 93
tools -- 40 of its own plus 53 the frontend registers -- and the frontend half
grew by sixteen in one afternoon, so the headroom is the thing worth watching.

Marked `live`: it costs two requests and needs a real key.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.tools.limits import MAX_TOOLS_PER_REQUEST

pytestmark = pytest.mark.live


async def test_the_limit_matches_the_provider(openai_key: str) -> None:
    """Pin MAX_TOOLS_PER_REQUEST to what OpenAI actually enforces.

    The constant is a guess until something checks it. If the provider raises
    the ceiling this fails as a *reminder to raise ours*; if it lowers it, this
    fails before production does.
    """
    import httpx

    def payload(count: int) -> dict[str, Any]:
        return {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": f"t{i}",
                        "description": "x",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for i in range(count)
            ],
        }

    headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
    url = "https://api.openai.com/v1/chat/completions"

    async with httpx.AsyncClient(timeout=60.0) as client:
        at_limit = await client.post(url, headers=headers, json=payload(MAX_TOOLS_PER_REQUEST))
        assert at_limit.status_code == 200, (
            f"OpenAI rejected {MAX_TOOLS_PER_REQUEST} tools; our ceiling is too high: "
            f"{at_limit.text[:200]}"
        )

        over = await client.post(url, headers=headers, json=payload(MAX_TOOLS_PER_REQUEST + 1))
        assert over.status_code == 400, (
            f"OpenAI now accepts {MAX_TOOLS_PER_REQUEST + 1} tools; the ceiling has moved "
            "and ours can be raised"
        )
