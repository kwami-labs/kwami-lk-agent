"""One call to one model, with that model's own way of searching the web.

Each runner takes a task and returns a `BrainAnswer`: the text, what it cost,
and where it looked. They never raise for a provider failure -- a model that is
down or out of quota becomes an answer saying so, because the bridge has to
tell the user something either way and a traceback is not it.

Search is per provider, which is the point of having several:

* Claude: Anthropic's server-side ``web_search`` and ``web_fetch``.
* Gemini: grounding with Google Search.
* GPT: the Responses API's ``web_search``.

When the user picked "both" and a Tavily key exists, Claude and GPT also get a
``tavily_search`` tool run here, on our side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..utils.logging import get_logger
from .ledger import search_cost, token_cost

logger = get_logger("brain.providers")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_TOOL_ROUNDS = 8
#: Models that take the dynamic-filtering web tools; the rest get the basic ones.
_DYNAMIC_WEB_TOOLS = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable",
)
#: Models that accept the server-side refusal fallback in its "default" form.
_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")

EFFORT_BY_DIFFICULTY = {"simple": "low", "normal": "medium", "hard": "high"}


@dataclass
class BrainAnswer:
    text: str
    model_id: str
    model: str
    cost_usd: float = 0.0
    searches: int = 0
    sources: list[str] = field(default_factory=list)
    ok: bool = True
    #: The provider said "out of quota" (429): retrying soon only fails again.
    rate_limited: bool = False


@dataclass
class BrainRequest:
    task: str
    system: str
    difficulty: str = "normal"
    native_search: bool = True
    tavily_key: str = ""


def _failure(model_id: str, model: str, message: str) -> BrainAnswer:
    return BrainAnswer(text=message, model_id=model_id, model=model, ok=False)


async def _tavily(api_key: str, query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                TAVILY_SEARCH_URL,
                json={
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": 6,
                    "include_answer": True,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        return f"Tavily search failed: {e}"
    results = [
        {"title": r.get("title"), "url": r.get("url"), "content": (r.get("content") or "")[:800]}
        for r in (data.get("results") or [])[:6]
        if isinstance(r, dict)
    ]
    return json.dumps({"answer": data.get("answer"), "results": results}, ensure_ascii=False)


TAVILY_TOOL = {
    "name": "tavily_search",
    "description": (
        "Search the web through Tavily. Use it alongside web_search when you want a "
        "second, independent set of sources."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

TAVILY_FUNCTION = {
    "type": "function",
    "name": "tavily_search",
    "description": TAVILY_TOOL["description"],
    "parameters": TAVILY_TOOL["input_schema"],
}


async def run_anthropic(
    api_key: str, model_id: str, model: str, request: BrainRequest
) -> BrainAnswer:
    import anthropic

    tools: list[dict[str, Any]] = []
    if request.native_search:
        if model.startswith(_DYNAMIC_WEB_TOOLS):
            tools += [
                {"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
                {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 6},
            ]
        else:
            tools += [
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 8},
                {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 6},
            ]
    if request.tavily_key:
        tools.append(TAVILY_TOOL)

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": 16000,
        "system": request.system,
        "output_config": {"effort": EFFORT_BY_DIFFICULTY.get(request.difficulty, "medium")},
    }
    if tools:
        params["tools"] = tools
    use_fallback = model.startswith(_FALLBACK_MODELS)

    client = anthropic.AsyncAnthropic(api_key=api_key, timeout=600.0)
    # The SDK types messages as a TypedDict union; these are built as plain dicts.
    messages: list[Any] = [{"role": "user", "content": request.task}]
    cost = 0.0
    searches = 0
    sources: list[str] = []
    response: Any = None
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            if use_fallback:
                response = await client.beta.messages.create(
                    messages=messages,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **params,
                )
            else:
                response = await client.messages.create(messages=messages, **params)

            usage = response.usage
            cost += token_cost(
                "anthropic", model, usage.input_tokens or 0, usage.output_tokens or 0
            )
            server = getattr(usage, "server_tool_use", None)
            round_searches = getattr(server, "web_search_requests", 0) or 0 if server else 0
            searches += round_searches
            cost += search_cost("anthropic", round_searches)
            for block in response.content:
                if block.type == "web_search_tool_result" and isinstance(block.content, list):
                    sources += [r.url for r in block.content if getattr(r, "url", None)]

            if response.stop_reason == "pause_turn":
                # The server-side search loop hit its limit; resend and it resumes.
                messages.append({"role": "assistant", "content": response.content})
                continue
            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type == "tool_use" and block.name == "tavily_search":
                        query = (
                            block.input.get("query", "") if isinstance(block.input, dict) else ""
                        )
                        searches += 1
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": await _tavily(request.tavily_key, str(query)),
                            }
                        )
                if not results:
                    break
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": results})
                continue
            break
    except anthropic.AuthenticationError:
        return _failure(model_id, model, "La clave de Anthropic no es válida.")
    except anthropic.RateLimitError:
        return _failure(model_id, model, "Anthropic está limitando las peticiones ahora mismo.")
    except anthropic.APIStatusError as e:
        logger.warning("Anthropic %s failed: %s", model, e)
        return _failure(model_id, model, f"Anthropic devolvió un error {e.status_code}.")
    except anthropic.APIConnectionError:
        return _failure(model_id, model, "No pude conectar con Anthropic.")

    if response is None:
        return _failure(model_id, model, "Sin respuesta.")
    if response.stop_reason == "refusal":
        return BrainAnswer(
            text="El modelo declinó esta tarea.",
            model_id=model_id,
            model=model,
            cost_usd=cost,
            ok=False,
        )
    # With web search the text arrives split at every citation, mid-sentence.
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return BrainAnswer(
        text=text or "Sin contenido.",
        model_id=model_id,
        model=getattr(response, "model", model) or model,
        cost_usd=cost,
        searches=searches,
        sources=list(dict.fromkeys(sources))[:12],
    )


async def run_gemini(api_key: str, model_id: str, model: str, request: BrainRequest) -> BrainAnswer:
    from google import genai
    from google.genai import types

    tools: list[Any] | None = (
        [types.Tool(google_search=types.GoogleSearch())] if request.native_search else None
    )
    try:
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=model,
            contents=request.task,
            config=types.GenerateContentConfig(
                system_instruction=request.system,
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
    except Exception as e:  # google-genai raises one error family for everything
        logger.warning("Gemini %s failed: %s", model, e)
        code = getattr(e, "code", None)
        if code == 429:
            answer = _failure(model_id, model, "La cuota de Gemini está agotada ahora mismo.")
            answer.rate_limited = True
            return answer
        return _failure(model_id, model, f"Gemini devolvió un error{f' {code}' if code else ''}.")

    usage = response.usage_metadata
    tokens_in = (getattr(usage, "prompt_token_count", 0) or 0) + (
        getattr(usage, "tool_use_prompt_token_count", 0) or 0
    )
    tokens_out = (getattr(usage, "candidates_token_count", 0) or 0) + (
        getattr(usage, "thoughts_token_count", 0) or 0
    )
    searches = 0
    sources: list[str] = []
    for candidate in response.candidates or []:
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding is None:
            continue
        searches += len(grounding.web_search_queries or [])
        for chunk in grounding.grounding_chunks or []:
            web = getattr(chunk, "web", None)
            if web is not None and web.uri:
                sources.append(web.title or web.uri)
    cost = token_cost("google", model, tokens_in, tokens_out) + search_cost("google", searches)
    text = (response.text or "").strip()
    return BrainAnswer(
        text=text or "Sin contenido.",
        model_id=model_id,
        model=model,
        cost_usd=cost,
        searches=searches,
        sources=list(dict.fromkeys(sources))[:12],
        ok=bool(text),
    )


async def run_openai(api_key: str, model_id: str, model: str, request: BrainRequest) -> BrainAnswer:
    import openai

    tools: list[Any] = []
    if request.native_search:
        tools.append({"type": "web_search"})
    if request.tavily_key:
        tools.append(TAVILY_FUNCTION)
    client = openai.AsyncOpenAI(api_key=api_key, timeout=600.0)
    cost = 0.0
    searches = 0
    input_items: list[Any] = [{"role": "user", "content": request.task}]
    response: Any = None
    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = await client.responses.create(
                model=model,
                instructions=request.system,
                input=input_items,
                tools=tools or openai.omit,
            )
            usage = response.usage
            if usage is not None:
                cost += token_cost("openai", model, usage.input_tokens, usage.output_tokens)
            calls = [item for item in response.output if item.type == "function_call"]
            searches += sum(1 for item in response.output if item.type == "web_search_call")
            if not calls:
                break
            input_items += list(response.output)
            for call in calls:
                try:
                    query = json.loads(call.arguments).get("query", "")
                except ValueError:
                    query = ""
                searches += 1
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": await _tavily(request.tavily_key, str(query)),
                    }
                )
    except openai.AuthenticationError:
        return _failure(model_id, model, "La clave de OpenAI no es válida.")
    except openai.RateLimitError:
        return _failure(model_id, model, "OpenAI está limitando las peticiones ahora mismo.")
    except openai.APIError as e:
        logger.warning("OpenAI %s failed: %s", model, e)
        return _failure(model_id, model, "OpenAI devolvió un error.")

    cost += search_cost("openai", searches)
    text = (getattr(response, "output_text", "") or "").strip() if response is not None else ""
    return BrainAnswer(
        text=text or "Sin contenido.",
        model_id=model_id,
        model=model,
        cost_usd=cost,
        searches=searches,
        ok=bool(text),
    )
