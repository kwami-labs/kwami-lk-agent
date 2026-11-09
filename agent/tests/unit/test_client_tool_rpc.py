"""Client tools over LiveKit RPC, for an app that says it can take them that way.

An app that registers the `kwami.client_tool` method sets the participant
attribute `kwami.client_tools=rpc`. Only then is a tool call sent as an RPC; an
older app keeps the `tool_call`/`tool_result` data-channel pair, and so does a
call the RPC path refuses for reasons that are not the tool's fault.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from livekit import rtc

from src.tools.client import (
    CLIENT_TOOL_RPC_ATTRIBUTE,
    CLIENT_TOOL_RPC_METHOD,
    CLIENT_TOOL_RPC_VALUE,
    CLIENT_TOOL_TIMEOUT_SECONDS,
    MAX_CLIENT_TOOL_RESULT_CHARS,
    ClientToolManager,
    rpc_destination,
)

Code = rtc.RpcError.ErrorCode


class Participant:
    """The agent's local participant: RPC out, data channel out."""

    def __init__(self, answer: Any = "done") -> None:
        self.answer = answer
        self.rpcs: list[dict[str, Any]] = []
        self.published: list[dict[str, Any]] = []
        self.manager: ClientToolManager | None = None

    async def perform_rpc(self, **kwargs: Any) -> str:
        self.rpcs.append(kwargs)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer

    async def publish_data(self, payload: bytes, *args: Any, **kwargs: Any) -> None:
        message = json.loads(payload.decode())
        self.published.append(message)
        # The app answers the data-channel call right away.
        assert self.manager is not None
        asyncio.get_running_loop().call_soon(
            self.manager.handle_tool_result, message["toolCallId"], "via data channel"
        )


def _app(identity: str = "user-1", rpc: bool = True) -> SimpleNamespace:
    attributes = {CLIENT_TOOL_RPC_ATTRIBUTE: CLIENT_TOOL_RPC_VALUE} if rpc else {}
    return SimpleNamespace(identity=identity, attributes=attributes)


def _setup(answer: Any = "done", *participants: Any) -> tuple[Any, Participant]:
    local = Participant(answer)
    room = SimpleNamespace(
        local_participant=local,
        remote_participants={p.identity: p for p in participants} if participants else {},
    )
    manager = ClientToolManager(kwami_agent=SimpleNamespace(room=room))
    manager.register_client_tools(
        [{"name": "set_theme", "description": "change the theme", "parameters": {}}]
    )
    local.manager = manager
    return manager.create_client_tools()[0]._func, local


class TestWhoTakesRpc:
    def test_the_participant_that_says_so(self) -> None:
        room = SimpleNamespace(
            remote_participants={"a": _app("agent", rpc=False), "u": _app("user-1")}
        )
        assert rpc_destination(room) == "user-1"

    def test_nobody_when_no_participant_says_so(self) -> None:
        assert rpc_destination(SimpleNamespace(remote_participants={"u": _app(rpc=False)})) is None
        assert rpc_destination(SimpleNamespace()) is None

    def test_a_participant_without_an_identity_cannot_be_called(self) -> None:
        nameless = SimpleNamespace(identity="", attributes={CLIENT_TOOL_RPC_ATTRIBUTE: "rpc"})
        assert rpc_destination(SimpleNamespace(remote_participants={"x": nameless})) is None


class TestOverRpc:
    async def test_the_call_goes_to_the_app_and_its_answer_comes_back(self) -> None:
        handler, local = _setup("dark mode on", _app())

        result = await handler({"value": "dark"}, None)

        assert result == "dark mode on"
        assert local.published == []
        [call] = local.rpcs
        assert call["destination_identity"] == "user-1"
        assert call["method"] == CLIENT_TOOL_RPC_METHOD
        assert call["response_timeout"] == CLIENT_TOOL_TIMEOUT_SECONDS
        body = json.loads(call["payload"])
        assert body["name"] == "set_theme"
        assert body["arguments"] == {"value": "dark"}
        assert body["toolCallId"]

    async def test_a_long_answer_is_capped_before_the_model_reads_it(self) -> None:
        handler, _ = _setup("x" * (MAX_CLIENT_TOOL_RESULT_CHARS + 50), _app())

        result = await handler({}, None)

        assert result.endswith("[truncated, 50 more characters]")

    async def test_an_empty_answer_is_an_empty_string(self) -> None:
        handler, _ = _setup(None, _app())

        assert await handler({}, None) == ""

    async def test_the_apps_error_is_what_the_model_reads(self) -> None:
        handler, local = _setup(rtc.RpcError(Code.APPLICATION_ERROR, "no such panel"), _app())

        result = await handler({}, None)

        assert result == "Error from client: no such panel"
        assert local.published == []

    async def test_a_timeout_says_so(self) -> None:
        handler, local = _setup(rtc.RpcError(Code.RESPONSE_TIMEOUT, "timed out"), _app())

        assert await handler({}, None) == "Error: Tool execution timed out"
        assert local.published == []

    async def test_an_unexpected_failure_is_a_string_not_a_raise(self) -> None:
        handler, _ = _setup(ConnectionError("room closed"), _app())

        assert await handler({}, None) == "Error executing tool: room closed"


class TestBackToTheDataChannel:
    async def test_an_app_that_did_not_advertise_rpc_gets_the_data_channel(self) -> None:
        handler, local = _setup("never used", _app(rpc=False))

        result = await handler({}, None)

        assert result == "via data channel"
        assert local.rpcs == []
        assert [m["type"] for m in local.published] == ["tool_call"]

    @pytest.mark.parametrize(
        "code",
        [
            Code.UNSUPPORTED_METHOD,
            Code.UNSUPPORTED_SERVER,
            Code.UNSUPPORTED_VERSION,
            Code.REQUEST_PAYLOAD_TOO_LARGE,
        ],
    )
    async def test_a_call_rpc_cannot_carry_falls_back(self, code: Code) -> None:
        handler, local = _setup(rtc.RpcError(code, "refused"), _app())

        result = await handler({}, None)

        assert result == "via data channel"
        assert len(local.rpcs) == 1
        assert [m["type"] for m in local.published] == ["tool_call"]
