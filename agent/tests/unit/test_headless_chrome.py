"""The container's own Chromium: throwaway, secret-free, and tried last."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser import browser_session, headless_chrome
from src.browser.browser_session import CloudBrowserSession
from src.browser.headless_chrome import HeadlessChromeProvider, _clean_environment, _stop
from src.browser.launched import HEADLESS, BrowserVendorError, LaunchedBrowser
from src.browser.providers import (
    FailoverBrowserProvider,
    ProviderUnavailableError,
    create_browser_provider,
)
from src.settings import Settings

_POPEN = subprocess.Popen


def _ports(monkeypatch: pytest.MonkeyPatch, answers: list[int | None]) -> None:
    monkeypatch.setattr(headless_chrome, "wait_for_port", AsyncMock(side_effect=answers))


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def popen(args, **kwargs):
        calls.append({"args": args, **kwargs})
        return MagicMock(spec=_POPEN)

    monkeypatch.setattr(headless_chrome.subprocess, "Popen", popen)
    return calls


def _profile(call: dict) -> Path:
    flag = next(a for a in call["args"] if a.startswith("--user-data-dir="))
    return Path(flag.split("=", 1)[1])


@pytest.mark.asyncio
async def test_it_asks_for_the_sandbox_first_and_keeps_secrets_out(spawned, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "secret")
    _ports(monkeypatch, [9333])

    launched = await HeadlessChromeProvider("/usr/bin/chromium").launch("user-1")

    assert launched.vendor == HEADLESS
    assert launched.cdp_http_url == "http://127.0.0.1:9333"
    assert not launched.persistence_id
    assert "--no-sandbox" not in spawned[0]["args"]
    assert "GOOGLE_API_KEY" not in spawned[0]["env"]
    await HeadlessChromeProvider("x").release("nothing-here")


@pytest.mark.asyncio
async def test_a_container_that_forbids_the_sandbox_still_gets_a_browser(spawned, monkeypatch):
    _ports(monkeypatch, [None, 9444])

    launched = await HeadlessChromeProvider("/usr/bin/chromium").launch("user-1")

    assert launched.browser_id == "headless-9444"
    assert "--no-sandbox" in spawned[1]["args"]


@pytest.mark.asyncio
async def test_failing_to_start_leaves_no_profile_behind(spawned, monkeypatch):
    _ports(monkeypatch, [None, None])

    with pytest.raises(BrowserVendorError):
        await HeadlessChromeProvider("/usr/bin/chromium").launch("user-1")

    assert not _profile(spawned[0]).exists()


@pytest.mark.asyncio
async def test_release_deletes_what_the_session_saw(spawned, monkeypatch):
    _ports(monkeypatch, [9333])
    provider = HeadlessChromeProvider("/usr/bin/chromium")

    launched = await provider.launch("user-1")
    profile = _profile(spawned[0])
    assert profile.exists()

    await provider.release(launched.browser_id)
    await provider.release(launched.browser_id)

    assert not profile.exists()


@pytest.mark.asyncio
async def test_a_worker_exiting_stops_its_browsers(spawned, monkeypatch):
    _ports(monkeypatch, [9333])
    provider = HeadlessChromeProvider("/usr/bin/chromium")
    await provider.launch("user-1")

    provider._stop_all()

    assert not _profile(spawned[0]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", ["", "  "])
async def test_it_needs_a_real_user(user_id):
    with pytest.raises(ValueError):
        await HeadlessChromeProvider("/usr/bin/chromium").launch(user_id)


def test_a_browser_that_ignores_terminate_is_killed():
    process = MagicMock(spec=subprocess.Popen)
    process.wait.side_effect = [subprocess.TimeoutExpired("chromium", 5), 0]

    _stop(process)

    process.kill.assert_called_once()


def test_the_environment_is_only_what_it_needs(tmp_path):
    assert set(_clean_environment(tmp_path)) - {"SYSTEMROOT"} == {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
    }


def _settings(**values) -> Settings:
    return Settings(**values)


def test_it_is_the_last_vendor_tried():
    provider = create_browser_provider(
        _settings(browser_use_api_key="k", headless_chromium="/usr/bin/chromium")
    )

    assert isinstance(provider, FailoverBrowserProvider)
    assert [vendor for vendor, _ in provider._candidates] == ["browser_use", HEADLESS]
    assert isinstance(provider._candidates[-1][1](), HeadlessChromeProvider)


def test_it_stands_alone_when_there_is_no_vendor():
    provider = create_browser_provider(_settings(headless_chromium="/usr/bin/chromium"))

    assert [vendor for vendor, _ in provider._candidates] == [HEADLESS]


def test_it_comes_after_the_local_window_too():
    provider = create_browser_provider(
        _settings(browser_provider="local", headless_chromium="/usr/bin/chromium")
    )

    assert [vendor for vendor, _ in provider._candidates] == ["local", HEADLESS]


def test_no_image_and_no_vendor_still_says_so():
    with pytest.raises(ProviderUnavailableError):
        create_browser_provider(_settings())


def _session(vendor: str = HEADLESS) -> tuple[CloudBrowserSession, MagicMock, AsyncMock]:
    writer = AsyncMock()
    room = MagicMock()
    room.local_participant.stream_bytes = AsyncMock(return_value=writer)
    room.local_participant.publish_data = AsyncMock()
    session = CloudBrowserSession(room=room)
    session._vendor = vendor
    session._browser_id = "headless-1"
    session._cdp = MagicMock(is_connected=True)
    return session, room, writer


@pytest.mark.asyncio
async def test_the_panel_gets_a_frame_only_when_the_page_changed(monkeypatch):
    session, room, writer = _session()
    jpeg = base64.b64encode(b"jpeg").decode()
    session._cdp.screenshot = AsyncMock(side_effect=[jpeg, jpeg, RuntimeError("navigating"), ""])

    async def sleep(_seconds):
        if session._cdp.screenshot.await_count == 4:
            session._cdp.is_connected = False

    monkeypatch.setattr(browser_session.asyncio, "sleep", sleep)

    await session._stream_frames()

    room.local_participant.stream_bytes.assert_awaited_once()
    writer.write.assert_awaited_once_with(b"jpeg")
    writer.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_room_means_no_frame():
    session, _, writer = _session()
    session._room = None

    await session._publish_frame(b"jpeg")

    writer.write.assert_not_called()


@pytest.mark.asyncio
async def test_the_panel_hears_about_the_builtin_browser_without_a_live_url():
    session, room, _ = _session()

    await session._publish_session_event("open", url="https://example.com")

    sent = room.local_participant.publish_data.await_args.args[0]
    assert b'"vendor": "headless"' in sent
    assert b'"liveUrl": ""' in sent


@pytest.mark.asyncio
async def test_starting_the_builtin_browser_starts_the_frames(monkeypatch):
    provider = HeadlessChromeProvider("/usr/bin/chromium")
    assert provider.vendor == HEADLESS
    launched = LaunchedBrowser(browser_id="headless-1", vendor=HEADLESS, cdp_http_url="http://x")
    provider.launch = AsyncMock(return_value=launched)
    cdp = MagicMock(is_connected=True, connect=AsyncMock(), send=AsyncMock(), close=AsyncMock())
    cdp.screenshot = AsyncMock(return_value="")
    monkeypatch.setattr(browser_session, "CDPConnection", lambda: cdp)
    session = CloudBrowserSession(room=None, provider=provider)

    await session.start("user-1")

    assert session._frame_task is not None
    await session.close()


def test_the_builtin_browser_is_not_billed():
    session, _, _ = _session()
    session._usage_tracker = MagicMock()
    session._started_at = 1.0

    session._record_browser_minutes()

    session._usage_tracker.record_external_usage.assert_not_called()


@pytest.mark.asyncio
async def test_closing_stops_the_frames():
    session, _, _ = _session()
    task = MagicMock()
    session._frame_task = task
    session._cdp.close = AsyncMock()

    await session.close()

    task.cancel.assert_called_once()
    assert session._frame_task is None
