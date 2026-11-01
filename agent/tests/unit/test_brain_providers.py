"""The brain's provider runners: one call to one model, never a traceback.

The vendor SDKs are replaced at the client boundary -- `AsyncAnthropic`,
`AsyncOpenAI` and `genai.Client` -- by stand-ins that record what they were
asked and return scripted responses shaped like the SDK's. The exceptions are
the SDKs' own classes, so a renamed or reshaped error breaks here first.
Tavily is mocked at the HTTP layer with respx.

What is pinned is our side: which tools each model is offered, how the loop
reacts to each stop reason, what is billed, and what the user hears when a
provider fails.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import openai
import pytest
import respx
from google import genai

from src.brain import providers
from src.brain.ledger import search_cost, token_cost
from src.brain.providers import (
    TAVILY_SEARCH_URL,
    BrainRequest,
    run_anthropic,
    run_gemini,
    run_openai,
)


def _request(**overrides: Any) -> BrainRequest:
    values: dict[str, Any] = {"task": "¿Precio del oro?", "system": "sys"}
    values.update(overrides)
    return BrainRequest(**values)


def _http_error(cls: type[Exception], status: int, url: str) -> Exception:
    request = httpx.Request("POST", url)
    return cls("boom", response=httpx.Response(status, request=request), body=None)  # type: ignore[call-arg]


# -- Tavily -------------------------------------------------------------------


@respx.mock
async def test_tavily_returns_the_answer_and_trimmed_results() -> None:
    route = respx.post(TAVILY_SEARCH_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "Sube",
                "results": [
                    {"title": "A", "url": "https://a", "content": "x" * 2000},
                    "not a result",
                    {"title": "B", "url": "https://b", "content": None},
                ],
            },
        )
    )
    out = await providers._tavily("tv-key", "oro")

    assert route.calls.last.request.headers["Authorization"] == "Bearer tv-key"
    data = providers.json.loads(out)
    assert data["answer"] == "Sube"
    assert [r["url"] for r in data["results"]] == ["https://a", "https://b"]
    assert len(data["results"][0]["content"]) == 800
    assert data["results"][1]["content"] == ""


@respx.mock
async def test_tavily_failure_is_a_sentence_for_the_model() -> None:
    respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(500))
    assert (await providers._tavily("k", "q")).startswith("Tavily search failed")


@respx.mock
async def test_tavily_with_no_results_still_answers() -> None:
    respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json={}))
    assert providers.json.loads(await providers._tavily("k", "q")) == {
        "answer": None,
        "results": [],
    }


# -- Anthropic ----------------------------------------------------------------


def _usage(tin: int = 100, tout: int = 50, searches: int | None = None) -> SimpleNamespace:
    server = SimpleNamespace(web_search_requests=searches) if searches is not None else None
    return SimpleNamespace(input_tokens=tin, output_tokens=tout, server_tool_use=server)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _message(
    stop: str, content: list[Any], usage: SimpleNamespace | None = None, model: str | None = None
) -> SimpleNamespace:
    out = SimpleNamespace(stop_reason=stop, content=content, usage=usage or _usage())
    if model is not None:
        out.model = model
    return out


class FakeAnthropic:
    """Stands in for `anthropic.AsyncAnthropic`: scripted replies, recorded calls."""

    script: list[Any] = []
    calls: list[dict[str, Any]] = []
    built: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        FakeAnthropic.built.append(kwargs)
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._beta_create))

    async def _create(self, **kwargs: Any) -> Any:
        return self._next(kwargs)

    async def _beta_create(self, **kwargs: Any) -> Any:
        return self._next({**kwargs, "_beta": True})

    def _next(self, kwargs: dict[str, Any]) -> Any:
        # Messages are a list the runner keeps appending to; snapshot it.
        FakeAnthropic.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = FakeAnthropic.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> type[FakeAnthropic]:
    FakeAnthropic.script = []
    FakeAnthropic.calls = []
    FakeAnthropic.built = []
    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAnthropic)
    return FakeAnthropic


async def test_claude_searches_the_web_and_cites_what_it_read(fake_anthropic) -> None:
    search_result = SimpleNamespace(
        type="web_search_tool_result",
        content=[SimpleNamespace(url="https://kitco"), SimpleNamespace(url=None)],
    )
    fake_anthropic.script = [
        _message("end_turn", [search_result, _text("El oro "), _text("sube.")], _usage(searches=2))
    ]
    answer = await run_anthropic("k", "claude-opus", "claude-opus-5-5", _request(difficulty="hard"))

    call = fake_anthropic.calls[0]
    # Opus 5 takes the dynamic web tools and the server-side fallback.
    assert [t["type"] for t in call["tools"]] == ["web_search_20260209", "web_fetch_20260209"]
    assert call["_beta"] is True and call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "high"}
    assert answer.ok and answer.text == "El oro sube."
    assert answer.sources == ["https://kitco"]
    assert answer.searches == 2
    assert answer.cost_usd == pytest.approx(
        token_cost("anthropic", "claude-opus-5-5", 100, 50) + search_cost("anthropic", 2)
    )


async def test_older_claude_gets_the_basic_tools_and_no_fallback(fake_anthropic) -> None:
    fake_anthropic.script = [_message("end_turn", [_text("hola")], model="claude-haiku-x")]
    answer = await run_anthropic("k", "haiku", "claude-haiku-5-5", _request(difficulty="weird"))

    call = fake_anthropic.calls[0]
    assert "_beta" not in call
    assert [t["type"] for t in call["tools"]] == ["web_search_20250305", "web_fetch_20250910"]
    assert call["output_config"] == {"effort": "medium"}
    # The model the API says answered is the one reported.
    assert answer.model == "claude-haiku-x"


async def test_claude_without_search_is_sent_no_tools(fake_anthropic) -> None:
    fake_anthropic.script = [_message("end_turn", [])]
    answer = await run_anthropic("k", "h", "claude-haiku-5-5", _request(native_search=False))
    assert "tools" not in fake_anthropic.calls[0]
    assert answer.text == "Sin contenido."


async def test_claude_resumes_a_paused_search(fake_anthropic) -> None:
    fake_anthropic.script = [
        _message("pause_turn", [_text("buscando")]),
        _message("end_turn", [_text("listo")]),
    ]
    answer = await run_anthropic("k", "h", "claude-haiku-5-5", _request())
    assert len(fake_anthropic.calls) == 2
    assert fake_anthropic.calls[1]["messages"][-1]["role"] == "assistant"
    assert answer.text == "listo"


@respx.mock
async def test_claude_runs_tavily_on_our_side(fake_anthropic) -> None:
    respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json={"answer": "2 %"}))
    tool_use = SimpleNamespace(
        type="tool_use", name="tavily_search", id="tu1", input={"query": "oro"}
    )
    odd_input = SimpleNamespace(type="tool_use", name="tavily_search", id="tu2", input="oro")
    fake_anthropic.script = [
        _message("tool_use", [tool_use, odd_input]),
        _message("end_turn", [_text("Sube un 2 %")]),
    ]
    answer = await run_anthropic(
        "k", "h", "claude-haiku-5-5", _request(native_search=False, tavily_key="tv")
    )

    assert [t["name"] for t in fake_anthropic.calls[0]["tools"]] == ["tavily_search"]
    results = fake_anthropic.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu1", "tu2"]
    assert "2 %" in results[0]["content"]
    assert answer.searches == 2
    assert answer.text == "Sube un 2 %"


async def test_a_tool_call_that_is_not_ours_ends_the_loop(fake_anthropic) -> None:
    other = SimpleNamespace(type="tool_use", name="something_else", id="x", input={})
    fake_anthropic.script = [_message("tool_use", [other, _text("parcial")])]
    answer = await run_anthropic("k", "h", "claude-haiku-5-5", _request(tavily_key="tv"))
    assert len(fake_anthropic.calls) == 1
    assert answer.text == "parcial"


async def test_a_refusal_is_reported_and_still_billed(fake_anthropic) -> None:
    fake_anthropic.script = [_message("refusal", [])]
    answer = await run_anthropic("k", "h", "claude-haiku-5-5", _request())
    assert not answer.ok
    assert answer.text == "El modelo declinó esta tarea."
    assert answer.cost_usd > 0


async def test_no_round_at_all_is_no_answer(fake_anthropic, monkeypatch) -> None:
    monkeypatch.setattr(providers, "MAX_TOOL_ROUNDS", 0)
    answer = await run_anthropic("k", "h", "claude-haiku-5-5", _request())
    assert not answer.ok and answer.text == "Sin respuesta."


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (
            _http_error(anthropic.AuthenticationError, 401, "https://api.anthropic.com"),
            "no es válida",
        ),
        (_http_error(anthropic.RateLimitError, 429, "https://api.anthropic.com"), "limitando"),
        (_http_error(anthropic.InternalServerError, 503, "https://api.anthropic.com"), "error 503"),
        (
            anthropic.APIConnectionError(
                request=httpx.Request("POST", "https://api.anthropic.com")
            ),
            "No pude conectar",
        ),
    ],
)
async def test_anthropic_failures_become_sentences(fake_anthropic, error, said) -> None:
    fake_anthropic.script = [error]
    answer = await run_anthropic("k", "h", "claude-haiku-5-5", _request())
    assert not answer.ok
    assert said in answer.text


# -- Gemini -------------------------------------------------------------------


class FakeGenai:
    """Stands in for `genai.Client`: `client.aio.models.generate_content`."""

    reply: Any = None
    calls: list[dict[str, Any]] = []

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, **kwargs: Any) -> Any:
        FakeGenai.calls.append(kwargs)
        if isinstance(FakeGenai.reply, Exception):
            raise FakeGenai.reply
        return FakeGenai.reply


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> type[FakeGenai]:
    FakeGenai.reply = None
    FakeGenai.calls = []
    monkeypatch.setattr(genai, "Client", FakeGenai)
    return FakeGenai


def _gemini_reply(text: str | None, candidates: list[Any] | None) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_token_count=1000,
        tool_use_prompt_token_count=None,
        candidates_token_count=200,
        thoughts_token_count=300,
    )
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=candidates)


async def test_gemini_grounds_on_google_and_counts_its_searches(fake_genai) -> None:
    grounding = SimpleNamespace(
        web_search_queries=["oro hoy", "oro euro"],
        grounding_chunks=[
            SimpleNamespace(web=SimpleNamespace(uri="https://a", title="Kitco")),
            SimpleNamespace(web=SimpleNamespace(uri="https://b", title="")),
            SimpleNamespace(web=SimpleNamespace(uri="", title="sin uri")),
            SimpleNamespace(web=None),
        ],
    )
    fake_genai.reply = _gemini_reply(
        " El oro sube. ",
        [
            SimpleNamespace(grounding_metadata=grounding),
            SimpleNamespace(grounding_metadata=None),
            SimpleNamespace(
                grounding_metadata=SimpleNamespace(web_search_queries=None, grounding_chunks=None)
            ),
        ],
    )
    answer = await run_gemini("g", "gemini-flash", "gemini-3.5-flash", _request())

    config = fake_genai.calls[0]["config"]
    assert config.tools[0].google_search is not None
    assert config.system_instruction == "sys"
    assert answer.ok and answer.text == "El oro sube."
    assert answer.searches == 2
    assert answer.sources == ["Kitco", "https://b"]
    assert answer.cost_usd == pytest.approx(
        token_cost("google", "gemini-3.5-flash", 1000, 500) + search_cost("google", 2)
    )


async def test_gemini_without_search_and_without_text(fake_genai) -> None:
    fake_genai.reply = _gemini_reply(None, None)
    answer = await run_gemini(
        "g", "gemini-flash", "gemini-3.5-flash", _request(native_search=False)
    )
    assert fake_genai.calls[0]["config"].tools is None
    assert not answer.ok and answer.text == "Sin contenido."


class _GenaiError(Exception):
    def __init__(self, code: int | None) -> None:
        super().__init__("failed")
        self.code = code


@pytest.mark.parametrize(
    ("code", "said"),
    [
        (429, "cuota de Gemini está agotada"),
        (500, "error 500"),
        (None, "Gemini devolvió un error."),
    ],
)
async def test_gemini_failures_become_sentences(fake_genai, code, said) -> None:
    fake_genai.reply = _GenaiError(code)
    answer = await run_gemini("g", "gemini-flash", "gemini-3.5-flash", _request())
    assert not answer.ok
    assert said in answer.text
    # Only a spent quota rests the model (bridge.py); an outage is retried.
    assert answer.rate_limited is (code == 429)


# -- OpenAI -------------------------------------------------------------------


class FakeOpenAI:
    """Stands in for `openai.AsyncOpenAI`: `client.responses.create`."""

    script: list[Any] = []
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        FakeOpenAI.calls.append({**kwargs, "input": list(kwargs["input"])})
        item = FakeOpenAI.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[FakeOpenAI]:
    FakeOpenAI.script = []
    FakeOpenAI.calls = []
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAI)
    return FakeOpenAI


def _response(output: list[Any], text: str = "", usage: Any = "default") -> SimpleNamespace:
    if usage == "default":
        usage = SimpleNamespace(input_tokens=1000, output_tokens=100)
    return SimpleNamespace(output=output, output_text=text, usage=usage)


@respx.mock
async def test_gpt_searches_natively_and_through_tavily(fake_openai) -> None:
    respx.post(TAVILY_SEARCH_URL).mock(return_value=httpx.Response(200, json={"answer": "sube"}))
    call = SimpleNamespace(type="function_call", call_id="c1", arguments='{"query": "oro"}')
    broken = SimpleNamespace(type="function_call", call_id="c2", arguments="{not json")
    web = SimpleNamespace(type="web_search_call")
    fake_openai.script = [
        _response([web, call, broken], usage=None),
        _response([SimpleNamespace(type="message")], text=" Sube. "),
    ]
    answer = await run_openai("o", "gpt", "gpt-5.2", _request(tavily_key="tv"))

    first = fake_openai.calls[0]
    assert first["tools"] == [{"type": "web_search"}, providers.TAVILY_FUNCTION]
    outputs = [i for i in fake_openai.calls[1]["input"] if isinstance(i, dict) and i.get("call_id")]
    assert [o["call_id"] for o in outputs] == ["c1", "c2"]
    assert answer.ok and answer.text == "Sube."
    assert answer.searches == 3
    # Only the second round reported usage.
    assert answer.cost_usd == pytest.approx(
        token_cost("openai", "gpt-5.2", 1000, 100) + search_cost("openai", 3)
    )


async def test_gpt_without_tools_and_without_text(fake_openai) -> None:
    fake_openai.script = [_response([], text="")]
    answer = await run_openai("o", "gpt", "gpt-5.2", _request(native_search=False))
    assert fake_openai.calls[0]["tools"] is openai.omit
    assert not answer.ok and answer.text == "Sin contenido."


async def test_gpt_with_no_round_is_no_answer(fake_openai, monkeypatch) -> None:
    monkeypatch.setattr(providers, "MAX_TOOL_ROUNDS", 0)
    answer = await run_openai("o", "gpt", "gpt-5.2", _request())
    assert not answer.ok and answer.text == "Sin contenido."


@pytest.mark.parametrize(
    ("error", "said"),
    [
        (_http_error(openai.AuthenticationError, 401, "https://api.openai.com"), "no es válida"),
        (_http_error(openai.RateLimitError, 429, "https://api.openai.com"), "limitando"),
        (
            openai.APIError("boom", httpx.Request("POST", "https://api.openai.com"), body=None),  # type: ignore[arg-type]
            "OpenAI devolvió un error.",
        ),
    ],
)
async def test_openai_failures_become_sentences(fake_openai, error, said) -> None:
    fake_openai.script = [error]
    answer = await run_openai("o", "gpt", "gpt-5.2", _request())
    assert not answer.ok
    assert said in answer.text
