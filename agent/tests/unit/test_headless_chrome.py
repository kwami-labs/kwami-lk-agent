"""The container's own Chromium: throwaway, secret-free, and tried last."""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser import browser_session, headless_chrome
from src.browser.browser_session import CloudBrowserSession
from src.browser.headless_chrome import HeadlessChromeProvider, _clean_environment, _stop
from src.browser.launched import BUSY, HEADLESS, BrowserVendorError, LaunchedBrowser
from src.browser.providers import (
    FailoverBrowserProvider,
    ProviderUnavailableError,
    create_browser_provider,
)
from src.browser.safety import UnsafeURLError
from src.settings import Settings

_POPEN = subprocess.Popen


def _ports(monkeypatch: pytest.MonkeyPatch, answers: list[int | None]) -> None:
    monkeypatch.setattr(headless_chrome, "wait_for_port", AsyncMock(side_effect=answers))


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def popen(args, **kwargs):
        calls.append({"args": args, **kwargs})
        return MagicMock(spec=_POPEN, pid=4242)

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
async def test_a_container_that_forbids_the_sandbox_gets_no_browser(spawned, monkeypatch):
    """Running without it is never what a failure falls back to."""
    _ports(monkeypatch, [None, 9444])

    with pytest.raises(BrowserVendorError):
        await HeadlessChromeProvider("/usr/bin/chromium").launch("user-1")

    assert len(spawned) == 1
    assert "--no-sandbox" not in spawned[0]["args"]


@pytest.mark.asyncio
async def test_an_operator_can_accept_running_without_it(spawned, monkeypatch):
    _ports(monkeypatch, [None, 9444])
    provider = HeadlessChromeProvider("/usr/bin/chromium", allow_no_sandbox=True)

    launched = await provider.launch("user-1")

    assert launched.browser_id == "headless-9444"
    assert "--no-sandbox" in spawned[1]["args"]
    await provider.release(launched.browser_id)


def test_the_setting_reaches_the_browser():
    built = create_browser_provider(
        Settings(headless_chromium="/usr/bin/chromium", headless_allow_no_sandbox=True)
    )
    assert built._provider(*built._candidates[0])._allow_no_sandbox is True


@pytest.mark.asyncio
async def test_every_connection_goes_through_its_own_proxy(spawned, monkeypatch):
    _ports(monkeypatch, [9333])
    provider = HeadlessChromeProvider("/usr/bin/chromium")

    launched = await provider.launch("user-1")

    args = spawned[0]["args"]
    flag = next(a for a in args if a.startswith("--proxy-server="))
    assert flag.startswith("--proxy-server=http://127.0.0.1:")
    # Without it Chromium goes straight to loopback, where the debugging ports are.
    assert "--proxy-bypass-list=<-loopback>" in args
    _, _, proxy = provider._running[launched.browser_id]
    assert proxy._server is not None
    await provider.release(launched.browser_id)
    assert proxy._server is None


@pytest.mark.asyncio
async def test_failing_to_start_leaves_no_profile_behind(spawned, monkeypatch):
    _ports(monkeypatch, [None])

    with pytest.raises(BrowserVendorError):
        await HeadlessChromeProvider("/usr/bin/chromium").launch("user-1")

    assert not _profile(spawned[0]).exists()


@pytest.fixture
def container(tmp_path, monkeypatch):
    """A /proc and a temp folder of its own, with room to say who is alive."""
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "100").mkdir()
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(headless_chrome, "_PROC", proc)
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _browser(root: Path, name: str, pid: str | None) -> Path:
    profile = root / f"kwami-browser-{name}"
    profile.mkdir()
    if pid is not None:
        (profile / headless_chrome._PID_FILE).write_text(pid)
    return profile


def test_a_profile_whose_browser_died_is_deleted(container):
    alive = _browser(container, "alive", "100")
    dead = _browser(container, "dead", "200")
    garbled = _browser(container, "garbled", "not-a-pid")

    assert headless_chrome._browsers_running() == 1

    assert alive.exists()
    assert not dead.exists()
    assert not garbled.exists()


def test_a_profile_still_starting_counts_and_an_old_one_without_a_pid_does_not(container):
    starting = _browser(container, "starting", None)
    stale = _browser(container, "stale", None)
    long_ago = time.time() - 3600
    os.utime(stale, (long_ago, long_ago))

    assert headless_chrome._browsers_running() == 1

    assert starting.exists()
    assert not stale.exists()


def test_a_profile_that_vanishes_while_counting_is_no_trouble(container, monkeypatch):
    _browser(container, "going", None)
    real_stat = Path.stat

    def gone(path, **kwargs):
        if path.name.startswith("kwami-browser-"):
            raise FileNotFoundError
        return real_stat(path, **kwargs)

    monkeypatch.setattr(Path, "stat", gone)
    assert headless_chrome._browsers_running() == 0


def test_an_unreadable_profile_is_left_alone(container, monkeypatch):
    profile = _browser(container, "locked", "100")

    def refuse(_path, **_kwargs):
        raise PermissionError

    monkeypatch.setattr(Path, "read_text", refuse)
    assert headless_chrome._browsers_running() == 0
    assert profile.exists()


def test_with_no_proc_nothing_is_counted_or_deleted(container, monkeypatch, tmp_path):
    profile = _browser(container, "mine", "200")
    monkeypatch.setattr(headless_chrome, "_PROC", tmp_path / "nothing")

    assert headless_chrome._browsers_running() == 0
    assert profile.exists()


@pytest.mark.asyncio
async def test_a_full_container_says_busy_and_starts_nothing(container, spawned):
    for number in range(headless_chrome.MAX_BROWSERS):
        _browser(container, str(number), "100")

    with pytest.raises(BrowserVendorError) as raised:
        await HeadlessChromeProvider("/usr/bin/chromium").launch("user-1")

    assert raised.value.kind == BUSY
    assert spawned == []


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
    session._cdp.evaluate = AsyncMock(return_value="https://example.com/")
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
@pytest.mark.parametrize(
    "href", ["http://127.0.0.1:9222/json/list", "http://169.254.169.254/", "file:///proc/1/environ"]
)
async def test_a_page_we_would_not_open_is_never_shown(monkeypatch, href):
    session, room, _ = _session()
    session._cdp.evaluate = AsyncMock(return_value=href)
    session._cdp.navigate = AsyncMock()
    session._cdp.screenshot = AsyncMock(return_value=base64.b64encode(b"jpeg").decode())

    async def sleep(_seconds):
        session._cdp.is_connected = False

    monkeypatch.setattr(browser_session.asyncio, "sleep", sleep)

    await session._stream_frames()

    session._cdp.screenshot.assert_not_awaited()
    room.local_participant.stream_bytes.assert_not_awaited()
    session._cdp.navigate.assert_awaited_once_with(browser_session.BLANK_PAGE)


@pytest.mark.asyncio
@pytest.mark.parametrize("href", ["about:blank", "chrome-error://chromewebdata/"])
async def test_the_browsers_own_pages_may_be_shown(href):
    session, _, _ = _session()
    session._cdp.evaluate = AsyncMock(return_value=href)
    assert await session._page_may_be_shown() is True


@pytest.mark.asyncio
async def test_a_page_that_will_not_say_where_it_is_is_not_shown():
    session, _, _ = _session()
    session._cdp.evaluate = AsyncMock(return_value=None)
    assert await session._page_may_be_shown() is False


@pytest.mark.asyncio
async def test_the_first_page_is_checked_before_any_frame(monkeypatch):
    provider = HeadlessChromeProvider("/usr/bin/chromium")
    launched = LaunchedBrowser(browser_id="headless-1", vendor=HEADLESS, cdp_http_url="http://x")
    provider.launch = AsyncMock(return_value=launched)
    provider.release = AsyncMock()
    cdp = MagicMock(
        is_connected=True,
        connect=AsyncMock(),
        send=AsyncMock(),
        close=AsyncMock(),
        navigate=AsyncMock(),
    )
    cdp.page_info = AsyncMock(return_value={"url": "http://127.0.0.1:9222/json/list"})
    cdp.evaluate = AsyncMock(return_value="about:blank")
    cdp.screenshot = AsyncMock(return_value="")
    monkeypatch.setattr(browser_session, "CDPConnection", lambda: cdp)
    monkeypatch.setattr(browser_session, "validate_url", lambda url, **_: _refuse_loopback(url))
    session = CloudBrowserSession(room=None, provider=provider)

    await session.start("user-1", "https://example.com/")

    assert cdp.navigate.await_args_list[-1].args == (browser_session.BLANK_PAGE,)
    await session.close()


def _refuse_loopback(url: str) -> str:
    if "127.0.0.1" in url:
        raise UnsafeURLError(url)
    return url


@pytest.mark.asyncio
async def test_a_dropped_connection_does_not_lose_the_browser(monkeypatch):
    """Starting again used to overwrite the only handle to a browser holding logins."""
    provider = HeadlessChromeProvider("/usr/bin/chromium")
    provider.release = AsyncMock()
    launched = LaunchedBrowser(browser_id="headless-2", vendor=HEADLESS, cdp_http_url="http://x")
    provider.launch = AsyncMock(return_value=launched)
    cdp = MagicMock(is_connected=True, connect=AsyncMock(), send=AsyncMock(), close=AsyncMock())
    cdp.evaluate = AsyncMock(return_value="about:blank")
    cdp.screenshot = AsyncMock(return_value="")
    monkeypatch.setattr(browser_session, "CDPConnection", lambda: cdp)
    session = CloudBrowserSession(room=None, provider=provider)
    session._browser_id = "headless-1"
    session._cdp = MagicMock(is_connected=False, close=AsyncMock())

    await session.start("user-1")

    provider.release.assert_awaited_once_with("headless-1")
    assert session._browser_id == "headless-2"
    await session.close()


@pytest.mark.asyncio
async def test_the_idle_timer_releases_a_browser_whose_connection_dropped(monkeypatch):
    monkeypatch.setattr(browser_session, "IDLE_TIMEOUT_SECONDS", 0)
    provider = HeadlessChromeProvider("/usr/bin/chromium")
    provider.release = AsyncMock()
    session = CloudBrowserSession(room=None, provider=provider)
    session._browser_id = "headless-1"
    session._cdp = MagicMock(is_connected=False, close=AsyncMock())

    await session._idle_timeout()

    provider.release.assert_awaited_once_with("headless-1")
    assert session._browser_id is None


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
