"""The user's tabs and Jarvis's sign-in window, read from Windows' accessibility layer.

The user chose not to install a browser extension (2026-11-21). What Windows shows of their
browsers still answers "which tabs do I have open", and of the clean window, when they are in.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any

import pytest

from src.browser import window_reader
from src.browser.window_reader import (
    AddressWatch,
    BrowserWindow,
    as_url,
    browser_tabs,
    describe_tabs,
    parse_windows,
    tab_title,
)

# -- where it can run --------------------------------------------------------------------


def test_only_on_windows_with_powershell_and_not_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(window_reader.SWITCH, raising=False)
    monkeypatch.setattr(window_reader, "powershell", lambda: "powershell.exe")
    assert window_reader.available(windows=True) is True
    assert window_reader.available(windows=False) is False
    monkeypatch.setenv(window_reader.SWITCH, "off")
    assert window_reader.available(windows=True) is False
    monkeypatch.delenv(window_reader.SWITCH)
    monkeypatch.setattr(window_reader, "powershell", lambda: "")
    assert window_reader.available(windows=True) is False
    monkeypatch.setattr(window_reader.os, "name", "posix")
    assert window_reader.available() is False


def test_powershell_is_looked_for_by_both_names(monkeypatch: pytest.MonkeyPatch) -> None:
    found = {"powershell": "C:/ps/powershell"}
    monkeypatch.setattr(window_reader.shutil, "which", found.get)
    assert window_reader.powershell() == "C:/ps/powershell"
    monkeypatch.setattr(window_reader.shutil, "which", lambda _name: None)
    assert window_reader.powershell() == ""


def test_the_script_goes_encoded_and_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(window_reader, "powershell", lambda: "ps.exe")
    command = window_reader._command("Write-Output 'ñ'")
    assert command[0] == "ps.exe"
    assert "-NoProfile" in command and "-NonInteractive" in command
    assert base64.b64decode(command[-1]).decode("utf-16-le") == "Write-Output 'ñ'"


async def test_a_read_starts_powershell_without_a_console(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def create(*args: Any, **kwargs: Any) -> str:
        seen.update(args=args, kwargs=kwargs)
        return "process"

    monkeypatch.setattr(window_reader, "powershell", lambda: "ps.exe")
    monkeypatch.setattr(window_reader.asyncio, "create_subprocess_exec", create)
    assert await window_reader._spawn("1") == "process"
    assert seen["args"][0] == "ps.exe"
    assert seen["kwargs"]["creationflags"] == window_reader.CREATE_NO_WINDOW


# -- what a tab is called ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "title"),
    [
        ("Uso de memoria de Recibidos (3) - Gmail: 330 MB", "Recibidos (3) - Gmail"),
        ("Memory usage of YouTube: 1,2 GB", "YouTube"),
        ("[GitHub] Sudo email verification", "[GitHub] Sudo email verification"),
        ("Precio: 30 MB de datos", "Precio: 30 MB de datos"),
        ("  ", ""),
    ],
)
def test_a_tab_is_called_by_its_page(name: str, title: str) -> None:
    assert tab_title(name) == title


def test_each_browser_by_the_name_the_user_knows() -> None:
    assert BrowserWindow("chrome.exe", "x").name == "Chrome"
    assert BrowserWindow("brave.exe", "x").name == "Brave"
    assert BrowserWindow("msedge.exe", "x").name == "Edge"
    assert BrowserWindow("vivaldi.exe", "x").name == "vivaldi"
    assert BrowserWindow("", "x").name == "browser"


def test_an_address_bar_becomes_a_url() -> None:
    assert as_url("mail.google.com/mail/u/0/") == "https://mail.google.com/mail/u/0/"
    assert as_url("http://intranet.example/") == "http://intranet.example/"


# -- the script's answer --------------------------------------------------------------------


def test_the_windows_are_read_from_the_scripts_json() -> None:
    text = json.dumps(
        [
            {
                "browser": "chrome.exe",
                "title": "Gmail - Google Chrome",
                "address": "mail.google.com/",
                "tabs": ["Uso de memoria de Gmail: 3 MB", "", 7, "Banco"],
            },
            {"browser": "brave.exe", "title": "YouTube - Brave", "tabs": "YouTube"},
            "not a window",
        ]
    )
    windows = parse_windows(text)
    assert [w.name for w in windows] == ["Chrome", "Brave"]
    assert windows[0].tabs == ["Gmail", "Banco"]
    assert windows[0].address == "mail.google.com/"
    assert windows[1].tabs == ["YouTube"], "one tab comes as a bare string"
    assert windows[1].address == ""


def test_one_window_alone_comes_as_an_object() -> None:
    windows = parse_windows(json.dumps({"browser": "chrome.exe", "title": "t", "tabs": None}))
    assert len(windows) == 1 and windows[0].tabs == []


@pytest.mark.parametrize("text", ["", "not json", "42", "null"])
def test_anything_else_is_no_windows(text: str) -> None:
    assert parse_windows(text) == []


class _Done:
    def __init__(self, out: bytes = b"", *, wait: float = 0.0, fail: bool = False) -> None:
        self.out = out
        self.wait = wait
        self.fail = fail
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.fail:
            raise OSError("broken pipe")
        await asyncio.sleep(self.wait)
        return self.out, b""

    def kill(self) -> None:
        self.killed = True


async def test_the_users_tabs_are_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    scripts: list[str] = []
    answer = json.dumps([{"browser": "chrome.exe", "title": "t", "tabs": ["a"]}])

    async def spawn(script: str) -> _Done:
        scripts.append(script)
        return _Done(b"\xef\xbb\xbf" + answer.encode())

    windows = await browser_tabs(spawn)
    assert [w.tabs for w in windows] == [["a"]]
    assert scripts == [window_reader.TABS_SCRIPT]
    assert "browser-profiles" in scripts[0], "Jarvis's own windows are left out"
    assert "Chrome_RenderWidgetHostHWND" in scripts[0], "the pages are not walked"


async def test_a_read_that_fails_or_hangs_is_no_tabs(monkeypatch: pytest.MonkeyPatch) -> None:
    async def broken(_script: str) -> Any:
        raise OSError("no powershell")

    assert await browser_tabs(broken) == []

    slow = _Done(wait=1.0)

    async def hangs(_script: str) -> _Done:
        return slow

    monkeypatch.setattr(window_reader, "READ_TIMEOUT_SECONDS", 0.01)
    assert await browser_tabs(hangs) == []
    assert slow.killed

    class Unkillable(_Done):
        def kill(self) -> None:
            raise ProcessLookupError

    stubborn = Unkillable(fail=True)

    async def gone(_script: str) -> _Done:
        return stubborn

    assert await browser_tabs(gone) == []


async def test_where_windows_cannot_be_read_nothing_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(window_reader, "available", lambda: False)

    async def must_not(_script: str) -> Any:
        raise AssertionError("ran")

    monkeypatch.setattr(window_reader, "_spawn", must_not)
    assert await browser_tabs() == []


async def test_by_default_the_read_uses_powershell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(window_reader, "available", lambda: True)

    async def spawn(_script: str) -> _Done:
        return _Done(b"[]")

    monkeypatch.setattr(window_reader, "_spawn", spawn)
    assert await browser_tabs() == []


# -- what the model is told ------------------------------------------------------------------


WINDOWS = [
    BrowserWindow(
        "chrome.exe",
        "Recibidos (3) - Gmail - Google Chrome",
        "mail.google.com/mail/u/0/",
        ["Factura de luz", "Recibidos (3) - Gmail"],
    ),
    BrowserWindow("brave.exe", "Nueva pestaña - Brave", "", ["Nueva pestaña"]),
    BrowserWindow("chrome.exe", "Mi banco - Google Chrome", "banco.example/", ["Mi banco"]),
]


async def nothing_blocked(_url: str) -> bool:
    return False


async def bank_blocked(url: str) -> bool:
    return "banco" in url


async def test_every_tab_is_listed_with_the_address_of_each_windows_front_tab() -> None:
    said = await describe_tabs(WINDOWS, "all", listing=True, blocked=bank_blocked)
    assert "4 tabs open" in said
    assert "1. Factura de luz" in said and "2. Recibidos (3) - Gmail" in said
    assert "Chrome window, the tab in front is at mail.google.com/mail/u/0/" in said
    assert "Brave window, the tab in front has no address" in said
    assert "keeps you out of" in said and "banco.example" not in said
    assert said.index("3. Nueva pestaña") < said.index("4. Mi banco")


async def test_the_tab_in_front_can_be_opened_by_its_address() -> None:
    said = await describe_tabs(WINDOWS, "gmail", listing=False, blocked=nothing_blocked)
    assert "Tab 2" in said and "https://mail.google.com/mail/u/0/" in said
    assert "navigate_to" in said
    by_number = await describe_tabs(
        WINDOWS, "2", listing=False, blocked=nothing_blocked, open_with="`open`"
    )
    assert "`open`" in by_number


async def test_a_tab_behind_shows_only_its_title() -> None:
    said = await describe_tabs(WINDOWS, "factura luz", listing=False, blocked=nothing_blocked)
    assert "Tab 1" in said and "only its title" in said and "mail.google" not in said
    new_tab = await describe_tabs(WINDOWS, "3", listing=False, blocked=nothing_blocked)
    assert "only its title" in new_tab, "in front, but with no address to open"


async def test_a_tab_the_users_list_forbids_is_not_given() -> None:
    said = await describe_tabs(WINDOWS, "banco", listing=False, blocked=bank_blocked)
    assert "keeps you out of" in said and "banco.example" not in said


async def test_a_tab_nobody_has_is_said() -> None:
    said = await describe_tabs(WINDOWS, "netflix", listing=False, blocked=nothing_blocked)
    assert "No tab of the user's matches 'netflix'" in said
    assert "No tab" in await describe_tabs(WINDOWS, "9", listing=False, blocked=nothing_blocked)


# -- the sign-in window ----------------------------------------------------------------------


class _Watching:
    """A watch script's process: its lines, then the end of its output."""

    def __init__(self, lines: list[bytes]) -> None:
        self.stdout = asyncio.StreamReader()
        for line in lines:
            self.stdout.feed_data(line)
        self.returncode: int | None = None
        self.killed = False

    def finish(self) -> None:
        self.stdout.feed_eof()

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1
        self.finish()

    async def wait(self) -> int:
        return self.returncode or 0


def test_a_watch_needs_a_profile_folder_name() -> None:
    for bad in ("", "x'; Remove-Item C:\\ -Recurse #", "../profile"):
        with pytest.raises(ValueError):
            AddressWatch(bad)


async def test_a_watch_says_each_new_address_and_the_end() -> None:
    process = _Watching(
        [
            b'{"address": "accounts.google.com/v3/signin", "typing": false}\n',
            b"not json\n",
            b"[1, 2]\n",
            b'{"closed": true}\n',
        ]
    )
    scripts: list[str] = []

    async def spawn(script: str) -> _Watching:
        scripts.append(script)
        return process

    watch = AddressWatch("bfa276b9ffd16150", every=0.5, limit=60, spawn=spawn)
    await watch.start()
    first = await watch.next(1)
    assert first == {"address": "accounts.google.com/v3/signin", "typing": False}
    assert await watch.next(1) == {"closed": True}
    assert await watch.next(0.01) is None, "nothing new"
    process.finish()
    await watch.close()
    assert process.killed
    script = scripts[0]
    assert "*bfa276b9ffd16150*" in script and "-lt 60" in script and "500" in script
    assert "__" not in script, "every blank is filled"


async def test_a_watch_whose_script_ended_closes_quietly() -> None:
    process = _Watching([])
    process.returncode = 0
    process.finish()

    async def spawn(_script: str) -> _Watching:
        return process

    watch = AddressWatch("abcdef12", spawn=spawn)
    await watch.start()
    await asyncio.sleep(0)
    await watch.close()
    assert not process.killed, "nothing left to stop"
    await watch.close()


async def test_a_watch_never_started_closes_too() -> None:
    await AddressWatch("abcdef12").close()


@pytest.mark.live
@pytest.mark.skipif(os.name != "nt", reason="Windows' accessibility layer")
async def test_live_the_users_windows_are_read() -> None:
    """On the developer's PC: the real script, against whatever browsers are open."""
    windows = await browser_tabs(window_reader._spawn)
    assert isinstance(windows, list)
    for window in windows:
        assert window.name in ("Chrome", "Brave", "Edge")
        assert all("Uso de memoria" not in tab for tab in window.tabs)
