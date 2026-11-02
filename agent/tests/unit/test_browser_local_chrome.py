"""The Chrome window on the user's own machine, and the edges of the vendor failover.

The window is found again through `DevToolsActivePort`, belongs to the user, and
its profile holds logins: so one profile per user, and it closes only when asked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.browser import browser_session as session_module
from src.browser import local_chrome
from src.browser.browser_session import CloudBrowserSession
from src.browser.local_chrome import LocalChromeProvider, find_chrome, profile_dir_for
from src.browser.providers import (
    BROWSER_USE,
    LOCAL,
    MISSING,
    BrowserVendorError,
    FailoverBrowserProvider,
    LaunchedBrowser,
    ProviderUnavailableError,
    _browser_use_provider,
    _browserbase_provider,
    _local_provider,
    _resting,
)
from src.settings import Settings

from .test_builtin_navigation_tools import _tools


@pytest.fixture(autouse=True)
def _clean_rest() -> Any:
    _resting.clear()
    yield
    _resting.clear()


@pytest.fixture
def profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(local_chrome, "PROFILES_DIR", tmp_path)
    return tmp_path


def _answering(monkeypatch: pytest.MonkeyPatch, info: dict | None) -> None:
    monkeypatch.setattr(local_chrome, "_version_info", AsyncMock(return_value=info))


class _FakeProcess:
    def __init__(self, code: int | None) -> None:
        self.code = code

    def poll(self) -> int | None:
        return self.code


class TestFindingChrome:
    @pytest.fixture(autouse=True)
    def _nothing_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(local_chrome, "_MAC_PATHS", ())
        monkeypatch.setattr(local_chrome.shutil, "which", lambda _: None)

    def test_a_machine_with_neither_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PROGRAMFILES", str(tmp_path))
        monkeypatch.setattr(local_chrome, "_MAC_PATHS", (str(tmp_path / "absent"),))
        assert find_chrome() == ""

    def test_the_windows_folders_come_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chrome = tmp_path / local_chrome._WINDOWS_PATHS[0]
        chrome.parent.mkdir(parents=True)
        chrome.touch()
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert find_chrome() == str(chrome)

    def test_then_the_mac_applications(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chrome = tmp_path / "Google Chrome"
        chrome.touch()
        monkeypatch.setattr(local_chrome, "_MAC_PATHS", (str(chrome),))
        assert find_chrome() == str(chrome)

    def test_then_the_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_chrome.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert find_chrome() == "/usr/bin/chrome"


class TestTheProfile:
    def test_an_id_does_not_reach_the_file_name(self, profiles: Path) -> None:
        assert "user-1" not in profile_dir_for("user-1").name
        assert profile_dir_for("user-1") != profile_dir_for("user-2")

    @pytest.mark.parametrize("content", ["", "not-a-port\n/devtools"])
    def test_an_unreadable_port_file_is_no_port(self, tmp_path: Path, content: str) -> None:
        (tmp_path / "DevToolsActivePort").write_text(content)
        assert local_chrome._active_port(tmp_path) is None

    def test_no_port_file_is_no_port(self, tmp_path: Path) -> None:
        assert local_chrome._active_port(tmp_path) is None


class TestAskingChrome:
    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        client = MagicMock()
        monkeypatch.setattr(local_chrome, "shared_client", lambda: client)
        return client

    @staticmethod
    def _response(body: Any) -> httpx.Response:
        request = httpx.Request("GET", "http://127.0.0.1:1/json/version")
        return httpx.Response(200, json=body, request=request)

    async def test_what_it_says_about_itself(self, client: MagicMock) -> None:
        client.get = AsyncMock(return_value=self._response({"Browser": "Chrome"}))
        assert await local_chrome._version_info(1) == {"Browser": "Chrome"}

    async def test_nothing_listening(self, client: MagicMock) -> None:
        client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        assert await local_chrome._version_info(1) is None

    async def test_something_else_listening(self, client: MagicMock) -> None:
        client.get = AsyncMock(return_value=self._response(["not", "chrome"]))
        assert await local_chrome._version_info(1) is None


class TestLaunching:
    @pytest.mark.parametrize("user_id", ["", "   "])
    async def test_a_shared_profile_is_refused(self, user_id: str) -> None:
        with pytest.raises(ValueError, match="real user id"):
            await LocalChromeProvider(chrome_path="chrome").launch(user_id)

    async def test_no_chrome_on_the_machine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(local_chrome, "find_chrome", lambda: "")
        with pytest.raises(BrowserVendorError) as caught:
            await LocalChromeProvider().launch("user-1")
        assert caught.value.kind == MISSING

    async def test_a_window_left_open_is_found_again(
        self, profiles: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profile = profile_dir_for("user-1")
        profile.mkdir(parents=True)
        (profile / "DevToolsActivePort").write_text("9333\n/devtools/browser/x")
        _answering(monkeypatch, {"Browser": "Chrome"})
        popen = MagicMock()
        monkeypatch.setattr(local_chrome.subprocess, "Popen", popen)
        provider = LocalChromeProvider(chrome_path="chrome")

        launched = await provider.launch("user-1")

        popen.assert_not_called()
        assert provider.vendor == LOCAL
        assert launched.browser_id == "local-9333"
        assert launched.cdp_http_url == "http://127.0.0.1:9333"
        assert launched.persistence_id == profile.name

    async def test_otherwise_chrome_starts_on_its_own_profile(
        self, profiles: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profile = profile_dir_for("user-1")

        def start(command: list[str], **_: Any) -> _FakeProcess:
            assert f"--user-data-dir={profile}" in command
            assert "--remote-debugging-port=0" in command
            (profile / "DevToolsActivePort").write_text("9444\n/devtools/browser/x")
            return _FakeProcess(None)

        monkeypatch.setattr(local_chrome.subprocess, "Popen", start)
        _answering(monkeypatch, {"Browser": "Chrome"})

        launched = await LocalChromeProvider(chrome_path="chrome").launch("user-1")

        assert launched.browser_id == "local-9444"

    async def test_a_chrome_that_keeps_starting_is_waited_for(
        self, profiles: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        profile = profile_dir_for("user-1")
        monkeypatch.setattr(local_chrome.subprocess, "Popen", lambda *_, **__: _FakeProcess(None))
        _answering(monkeypatch, {"Browser": "Chrome"})

        async def port_appears(_: float) -> None:
            (profile / "DevToolsActivePort").write_text("9555\n/devtools/browser/x")

        monkeypatch.setattr(local_chrome.asyncio, "sleep", port_appears)

        launched = await LocalChromeProvider(chrome_path="chrome").launch("user-1")

        assert launched.browser_id == "local-9555"

    async def test_a_chrome_that_exits_with_an_error_is_named(
        self, profiles: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(local_chrome.subprocess, "Popen", lambda *_, **__: _FakeProcess(1))
        with pytest.raises(BrowserVendorError, match="did not start"):
            await LocalChromeProvider(chrome_path="chrome").launch("user-1")

    async def test_a_chrome_that_never_answers_runs_out_of_time(
        self, profiles: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(local_chrome.subprocess, "Popen", lambda *_, **__: _FakeProcess(None))
        monkeypatch.setattr(local_chrome, "START_TIMEOUT_SECONDS", 0)
        with pytest.raises(BrowserVendorError, match="did not start"):
            await LocalChromeProvider(chrome_path="chrome").launch("user-1")


class TestClosing:
    async def test_hanging_up_leaves_the_window_open(self, monkeypatch: pytest.MonkeyPatch) -> None:
        version = AsyncMock()
        monkeypatch.setattr(local_chrome, "_version_info", version)
        await LocalChromeProvider(chrome_path="chrome").release("local-1")
        version.assert_not_awaited()

    async def test_an_unknown_window_is_left_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        version = AsyncMock()
        monkeypatch.setattr(local_chrome, "_version_info", version)
        await LocalChromeProvider(chrome_path="chrome").quit("local-1")
        version.assert_not_awaited()

    async def test_a_window_already_gone_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answering(monkeypatch, None)
        provider = LocalChromeProvider(chrome_path="chrome")
        provider._ports["local-1"] = 1
        await provider.quit("local-1")
        assert provider._ports == {}

    async def test_asking_closes_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _answering(monkeypatch, {"webSocketDebuggerUrl": "ws://127.0.0.1:1/devtools"})
        connection = AsyncMock()
        connect = MagicMock()
        connect.return_value.__aenter__ = AsyncMock(return_value=connection)
        connect.return_value.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr("websockets.connect", connect)
        provider = LocalChromeProvider(chrome_path="chrome")
        provider._ports["local-1"] = 1

        await provider.quit("local-1")

        sent = json.loads(connection.send.await_args.args[0])
        assert sent["method"] == "Browser.close"


def _launched(vendor: str) -> LaunchedBrowser:
    return LaunchedBrowser(browser_id=f"{vendor}-1", vendor=vendor, cdp_http_url="http://cdp")


class TestTheFailoverEdges:
    async def test_a_vendor_that_names_its_own_failure_is_believed(self) -> None:
        provider = AsyncMock()
        provider.launch.side_effect = BrowserVendorError(LOCAL, MISSING, "No Chrome here.")
        with pytest.raises(ProviderUnavailableError, match="No Chrome here"):
            await FailoverBrowserProvider([(LOCAL, lambda: provider)]).launch("user-1")

    async def test_a_vendor_is_built_once(self) -> None:
        provider = AsyncMock()
        provider.launch.return_value = _launched(BROWSER_USE)
        build = MagicMock(return_value=provider)
        failover = FailoverBrowserProvider([(BROWSER_USE, build)])
        await failover.launch("user-1")
        await failover.launch("user-1")
        build.assert_called_once()

    async def test_releasing_a_browser_nobody_owns(self) -> None:
        await FailoverBrowserProvider([(BROWSER_USE, MagicMock())]).release("unknown")

    async def test_quit_reaches_the_vendor_that_opened_it(self) -> None:
        provider = AsyncMock()
        provider.launch.return_value = _launched(LOCAL)
        failover = FailoverBrowserProvider([(LOCAL, lambda: provider)])
        launched = await failover.launch("user-1")
        await failover.quit(launched.browser_id)
        provider.quit.assert_awaited_once_with(launched.browser_id)

    def test_each_vendor_is_built_from_the_settings(self) -> None:
        settings = Settings(
            browserbase_api_key="bb", browserbase_project_id="p", browser_use_api_key="bu"
        )
        assert _browserbase_provider(settings).vendor == "browserbase"
        assert _browser_use_provider(settings).vendor == BROWSER_USE
        assert _local_provider(settings).vendor == LOCAL


class TestTheSession:
    async def test_the_users_window_comes_to_the_front_at_its_own_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cdp = AsyncMock()
        cdp.send.return_value = {}
        monkeypatch.setattr(session_module, "CDPConnection", lambda: cdp)
        provider = AsyncMock()
        provider.launch.return_value = _launched(LOCAL)
        session = CloudBrowserSession()
        session._provider = provider
        await session.start(user_id="user-1")
        sent = [call.args[0] for call in cdp.send.await_args_list]
        assert "Page.bringToFront" in sent
        assert "Emulation.setDeviceMetricsOverride" not in sent
        await session.close()

    async def test_closing_a_window_with_no_way_to_quit_still_releases_it(self) -> None:
        provider = AsyncMock()
        del provider.quit
        session = CloudBrowserSession()
        session._vendor = LOCAL
        session._provider = provider
        session._browser_id = "local-1"
        session._cdp = AsyncMock()
        await session.close(quit_browser=True)
        provider.release.assert_awaited_once_with("local-1")


class TestTheNavigateTool:
    async def test_it_says_the_page_opened_on_the_users_computer(self) -> None:
        tools, session = _tools(active=False, kwami_id="kwami-42")
        session.runs_locally = True
        assert "Chrome window on the user's computer" in await tools.navigate_to(
            None, "https://example.com"
        )

    async def test_no_browser_service_is_said_plainly(self) -> None:
        tools, session = _tools(active=False, kwami_id="kwami-42")
        session.start = AsyncMock(side_effect=ProviderUnavailableError("Browser Use has no credit"))
        assert "Browser Use has no credit" in await tools.navigate_to(None, "https://example.com")
