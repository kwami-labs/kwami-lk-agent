"""The work thread's way into Google: the key stays here and refusals reach the model."""

from __future__ import annotations

import httpx
import pytest
import respx
from claude_agent_sdk import create_sdk_mcp_server, tool

from src.brain import google_tool
from src.brain.executor import check_tool_call

URL = "http://api.test/internal/users/u1/google"


@pytest.fixture
def settings(monkeypatch):
    class Settings:
        kwami_api_url = "http://api.test/"
        kwami_api_key = "secret-key"
        kwami_api_enabled = True

    monkeypatch.setattr(google_tool, "get_settings", lambda: Settings)
    return Settings


def test_it_needs_a_user_and_an_api(settings):
    assert google_tool.google_available("u1") is True
    assert google_tool.google_available("") is False
    settings.kwami_api_enabled = False
    assert google_tool.google_available("u1") is False


@respx.mock
async def test_a_call_carries_the_key_and_returns_the_result(settings):
    route = respx.post(URL).respond(json={"result": {"id": "f1", "name": "ñ"}})
    text = await google_tool.call_google("u1", "drive_read", {"file_id": "f1"})
    assert text == '{"id": "f1", "name": "ñ"}'
    assert route.calls.last.request.headers["x-kwami-api-key"] == "secret-key"


@respx.mock
async def test_a_refusal_reaches_the_model_as_text(settings):
    respx.post(URL).respond(400, json={"detail": "Not allowed from the work thread: gmail_send"})
    text = await google_tool.call_google("u1", "gmail_send", {})
    assert text == "Refused (400): Not allowed from the work thread: gmail_send"
    respx.post(URL).respond(500, text="boom")
    assert await google_tool.call_google("u1", "x", {}) == "Refused (500): "


@respx.mock
async def test_a_long_answer_is_cut(settings):
    respx.post(URL).respond(json={"result": "x" * 50_000})
    text = await google_tool.call_google("u1", "drive_read", {})
    assert len(text) == google_tool.MAX_ANSWER_CHARS


def _handler(captured):
    def fake_tool(_name, _description, _schema):
        def wrap(function):
            captured.append(function)
            return function

        return wrap

    return fake_tool


@respx.mock
async def test_the_tool_answers_even_when_the_api_is_down(settings):
    captured = []
    google_tool.google_server("u1", _handler(captured), lambda **_kwargs: None)
    respx.post(URL).mock(side_effect=httpx.ConnectError("down"))
    answer = await captured[0]({"action": "drive_read"})
    assert "could not be reached" in answer["content"][0]["text"]
    respx.post(URL).respond(json={"result": 1})
    answer = await captured[0]({"action": "drive_read", "args": {"file_id": "f"}})
    assert answer["content"][0]["text"] == "1"


def test_it_builds_a_real_sdk_server():
    server = google_tool.google_server("u1", tool, create_sdk_mcp_server)
    assert server["name"] == google_tool.SERVER_NAME


def test_the_guard_lets_the_tool_through(tmp_path):
    assert check_tool_call(tmp_path, google_tool.ALLOWED_TOOL, {"action": "accounts"}) is None
