"""The local window: found, started, reused and closed, with no real Chrome."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from src.browser import local_chrome
from src.browser.launched import LOCAL, MISSING, BrowserVendorError
from src.browser.local_chrome import (
    LocalChromeProvider,
    _active_port,
    _version_info,
    find_chrome,
    profile_dir_for,
)
from src.browser.providers import create_browser_provider
from src.settings import Settings

VERSION = "http://127.0.0.1:{}/json/version"
CHROME = {"Browser": "Chrome/140"}


@pytest.fixture
def profiles(tmp_path: Path) -> Any:
    with patch.object(local_chrome, "PROFILES_DIR", tmp_path):
        yield tmp_path


def _write_port(profile: Path, port: int) -> None:
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "DevToolsActivePort").write_text(f"{port}\n/devtools/browser/x\n")


def _no_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(local_chrome.shutil, "which", lambda _: None)
    monkeypatch.setattr(local_chrome, "_MAC_PATHS", ())


class TestFindingChrome:
    def test_windows_install(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_chrome(monkeypatch)
        chrome = tmp_path / "Google" / "Chrome" / "Application" / "chrome.exe"
        chrome.parent.mkdir(parents=True)
        chrome.touch()
        monkeypatch.setattr(local_chrome, "_WINDOWS_PATHS", (str(chrome.relative_to(tmp_path)),))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert find_chrome() == str(chrome)

    def test_mac_install(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_chrome(monkeypatch)
        chrome = tmp_path / "Chrome"
        chrome.touch()
        monkeypatch.setattr(local_chrome, "_MAC_PATHS", (str(chrome),))
        assert find_chrome() == str(chrome)

    def test_on_the_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_chrome(monkeypatch)
        monkeypatch.setattr(local_chrome.shutil, "which", lambda name: f"/bin/{name}")
        assert find_chrome() == "/bin/chrome"

    def test_nowhere(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_chrome(monkeypatch)
        monkeypatch.setattr(local_chrome, "_MAC_PATHS", (str(tmp_path / "absent"),))
        assert find_chrome() == ""


def test_a_user_id_stays_out_of_the_directory_name(profiles: Path) -> None:
    assert "user-1" not in str(profile_dir_for("user-1"))
    assert profile_dir_for("user-1") != profile_dir_for("user-2")


def test_a_missing_or_broken_port_file_is_no_port(tmp_path: Path) -> None:
    assert _active_port(tmp_path) is None
    (tmp_path / "DevToolsActivePort").write_text("not-a-port")
    assert _active_port(tmp_path) is None
    (tmp_path / "DevToolsActivePort").write_text("")
    assert _active_port(tmp_path) is None
    _write_port(tmp_path, 4321)
    assert _active_port(tmp_path) == 4321


class TestVersionInfo:
    @respx.mock
    async def test_chrome_answering(self) -> None:
        respx.get(VERSION.format(1)).mock(return_value=httpx.Response(200, json={"a": 1}))
        assert await _version_info(1) == {"a": 1}

    @respx.mock
    async def test_nothing_listening(self) -> None:
        respx.get(VERSION.format(1)).mock(side_effect=httpx.ConnectError("no"))
        assert await _version_info(1) is None

    @respx.mock
    async def test_something_else_listening(self) -> None:
        respx.get(VERSION.format(1)).mock(return_value=httpx.Response(200, json=[1]))
        assert await _version_info(1) is None


class TestLaunch:
    def test_vendor(self) -> None:
        assert LocalChromeProvider(chrome_path="c").vendor == LOCAL

    async def test_a_real_user_is_required(self, profiles: Path) -> None:
        with pytest.raises(ValueError, match="real user id"):
            await LocalChromeProvider(chrome_path="c").launch(" ")

    async def test_no_chrome_is_named(self, profiles: Path) -> None:
        with pytest.raises(BrowserVendorError, match="Neither Chrome nor Edge") as caught:
            await LocalChromeProvider(chrome_path="").launch("user-1")
        assert caught.value.kind == MISSING

    async def test_chrome_is_looked_for_when_not_given(self, profiles: Path) -> None:
        with (
            patch.object(local_chrome, "find_chrome", return_value=""),
            pytest.raises(BrowserVendorError, match="Neither"),
        ):
            await LocalChromeProvider().launch("user-1")

    @respx.mock
    async def test_a_window_left_open_is_found_again(self, profiles: Path) -> None:
        _write_port(profile_dir_for("user-1"), 5555)
        respx.get(VERSION.format(5555)).mock(return_value=httpx.Response(200, json=CHROME))
        with patch.object(local_chrome.subprocess, "Popen") as popen:
            launched = await LocalChromeProvider(chrome_path="c").launch("user-1")
        popen.assert_not_called()
        assert launched.cdp_http_url == "http://127.0.0.1:5555"
        assert launched.live_url == ""

    @respx.mock
    async def test_a_stale_port_starts_chrome_on_its_own_profile(self, profiles: Path) -> None:
        profile = profile_dir_for("user-1")
        _write_port(profile, 1111)
        respx.get(VERSION.format(1111)).mock(side_effect=httpx.ConnectError("gone"))
        respx.get(VERSION.format(2222)).mock(return_value=httpx.Response(200, json=CHROME))

        def start(args: list[str], **_: Any) -> MagicMock:
            assert f"--user-data-dir={profile}" in args
            assert "--remote-debugging-port=0" in args
            _write_port(profile, 2222)
            return MagicMock()

        with patch.object(local_chrome.subprocess, "Popen", side_effect=start):
            launched = await LocalChromeProvider(chrome_path="c").launch("user-1")
        assert launched.browser_id == "local-2222"

    async def test_chrome_crashing_stops_the_wait(self, profiles: Path) -> None:
        process = MagicMock()
        process.poll.return_value = 1
        with (
            patch.object(local_chrome.subprocess, "Popen", return_value=process),
            pytest.raises(BrowserVendorError, match="did not start"),
        ):
            await LocalChromeProvider(chrome_path="c").launch("user-1")

    async def test_the_wait_has_an_end(self, profiles: Path) -> None:
        process = MagicMock()
        process.poll.return_value = None
        with (
            patch.object(local_chrome.subprocess, "Popen", return_value=process),
            patch.object(local_chrome, "START_TIMEOUT_SECONDS", 0.02),
            patch.object(local_chrome, "_POLL_SECONDS", 0.01),
            pytest.raises(BrowserVendorError, match="did not start"),
        ):
            await LocalChromeProvider(chrome_path="c").launch("user-1")


class TestClosing:
    async def test_release_leaves_the_window(self) -> None:
        provider = LocalChromeProvider(chrome_path="c")
        provider._ports["local-1"] = 1
        await provider.release("local-1")
        assert provider._ports == {"local-1": 1}

    async def test_quit_closes_it(self) -> None:
        provider = LocalChromeProvider(chrome_path="c")
        provider._ports["local-1"] = 1
        connection = AsyncMock()
        connect = MagicMock()
        connect.return_value.__aenter__ = AsyncMock(return_value=connection)
        connect.return_value.__aexit__ = AsyncMock(return_value=False)
        with (
            patch.object(
                local_chrome,
                "_version_info",
                AsyncMock(return_value={"webSocketDebuggerUrl": "ws://x"}),
            ),
            patch("websockets.connect", connect),
        ):
            await provider.quit("local-1")
        connect.assert_called_once()
        assert json.loads(connection.send.await_args.args[0])["method"] == "Browser.close"

    async def test_quit_with_the_window_already_gone_does_nothing(self) -> None:
        provider = LocalChromeProvider(chrome_path="c")
        await provider.quit("local-unknown")
        provider._ports["local-1"] = 1
        with patch.object(local_chrome, "_version_info", AsyncMock(return_value=None)):
            await provider.quit("local-1")


async def test_settings_can_build_the_local_provider(profiles: Path) -> None:
    """Through the factory, as the session does, down to the missing Chrome."""
    provider = create_browser_provider(Settings(browser_provider=LOCAL))
    with (
        patch.object(local_chrome, "find_chrome", return_value=""),
        pytest.raises(RuntimeError, match="Neither Chrome nor Edge"),
    ):
        await provider.launch("user-1")
