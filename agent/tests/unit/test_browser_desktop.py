"""Jarvis's own Chrome stays out of the user's way: no focus taken, sent behind KWAMI."""

from __future__ import annotations

import ctypes
import subprocess
from typing import Any

import pytest

from src.browser import desktop
from src.browser.desktop import (
    HWND_BOTTOM,
    SW_SHOWNOACTIVATE,
    SWP_BEHIND,
    Win32,
    keep_behind,
    quiet_startup,
)


class FakeUser32:
    """Top-level windows by handle: (owner pid, visible, has a title)."""

    def __init__(self, windows: dict[int, tuple[int, bool, bool]]) -> None:
        self.windows = windows
        self.moved: list[tuple[Any, ...]] = []

    def EnumWindows(self, callback: Any, param: int) -> bool:  # noqa: N802 - user32's names
        for hwnd in list(self.windows):
            if not callback(hwnd, param):
                break
        return True

    def GetWindowThreadProcessId(self, hwnd: int, owner: Any) -> int:  # noqa: N802
        owner._obj.value = self.windows[hwnd][0]
        return 1

    def IsWindowVisible(self, hwnd: int) -> bool:  # noqa: N802
        return self.windows[hwnd][1]

    def GetWindowTextLengthW(self, hwnd: int) -> int:  # noqa: N802
        return 12 if self.windows[hwnd][2] else 0

    def SetWindowPos(self, *args: Any) -> bool:  # noqa: N802
        self.moved.append(args)
        return True


def same(function: Any) -> Any:
    return function


def test_a_program_starts_shown_but_without_the_focus(monkeypatch: pytest.MonkeyPatch) -> None:
    class Info:
        dwFlags = 0  # noqa: N815 - the Win32 field names
        wShowWindow = 0  # noqa: N815

    monkeypatch.setattr(subprocess, "STARTUPINFO", Info, raising=False)
    info = quiet_startup(windows=True)
    assert info.dwFlags & 1, "the show state is read"
    assert info.wShowWindow == SW_SHOWNOACTIVATE
    assert quiet_startup(windows=False) is None


def test_elsewhere_than_windows_nothing_is_done(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop.os, "name", "posix")
    assert quiet_startup() is None
    assert keep_behind(42) == 0


def test_only_the_processs_visible_titled_windows_are_found() -> None:
    user32 = FakeUser32(
        {
            1: (42, True, True),
            2: (42, False, True),  # hidden
            3: (42, True, False),  # no title: a helper window
            4: (7, True, True),  # someone else's
            5: (42, True, True),
        }
    )
    api = Win32(user32, same)
    assert api.windows_of(42) == [1, 5]
    api.send_back(5)
    assert user32.moved == [(5, HWND_BOTTOM, 0, 0, 0, 0, SWP_BEHIND)]


def test_the_real_calls_are_user32s(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []
    monkeypatch.setattr(ctypes, "WinDLL", lambda name: loaded.append(name) or "lib", raising=False)
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", lambda *types: same, raising=False)
    api = desktop.win32()
    assert loaded == ["user32", "kernel32"]
    assert api._user32 == "lib" and api._callback is same


def test_a_fresh_chrome_goes_behind_bubbles_and_all() -> None:
    user32 = FakeUser32({})
    naps: list[float] = []

    def sleep(seconds: float) -> None:
        naps.append(seconds)
        if len(naps) == 2:
            user32.windows[10] = (42, True, True)  # the window appears
        if seconds == desktop.SETTLE_SECONDS:
            user32.windows[11] = (42, True, True)  # then the translate bubble

    sent = keep_behind(42, windows=True, api=Win32(user32, same), sleep=sleep)
    assert sent == 2
    moved = [args[0] for args in user32.moved]
    assert moved == [10, 10, 11], "the window at once, then again with the bubble"


def test_a_chrome_with_no_window_in_time_is_left_alone() -> None:
    user32 = FakeUser32({})
    assert keep_behind(42, windows=True, api=Win32(user32, same), sleep=lambda _s: None) == 0
    assert user32.moved == []


def test_it_reaches_for_user32_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    user32 = FakeUser32({3: (9, True, True)})
    monkeypatch.setattr(desktop, "win32", lambda: Win32(user32, same))
    assert keep_behind(9, windows=True, sleep=lambda _s: None) == 1


class FocusUser32(FakeUser32):
    """Also who has the keyboard, and what it takes to hand it on."""

    def __init__(self, windows: dict[int, tuple[int, bool, bool]], focused: int) -> None:
        super().__init__(windows)
        self.focused = focused
        self.attached: list[tuple[int, int, bool]] = []
        self.threads = {hwnd: 100 + hwnd for hwnd in windows}

    def GetForegroundWindow(self) -> int:  # noqa: N802
        return self.focused

    def GetWindowThreadProcessId(self, hwnd: int, owner: Any) -> int:  # noqa: N802
        owner._obj.value = self.windows.get(hwnd, (0, False, False))[0]
        return self.threads.get(hwnd, 0)

    def AttachThreadInput(self, mine: int, theirs: int, on: bool) -> bool:  # noqa: N802
        self.attached.append((mine, theirs, on))
        return True

    def SetForegroundWindow(self, hwnd: int) -> bool:  # noqa: N802
        self.focused = hwnd
        return True


class Kernel32:
    def GetCurrentThreadId(self) -> int:  # noqa: N802
        return 7


def test_the_focus_goes_back_where_it_was_if_chrome_took_it() -> None:
    user32 = FocusUser32({1: (500, True, True), 20: (42, True, True)}, focused=20)
    api = Win32(user32, same, Kernel32())
    assert api.foreground() == 20
    assert api.owner(20) == (42, 120)
    sent = keep_behind(42, windows=True, api=api, sleep=lambda _s: None, focus=1)
    assert sent == 1
    assert user32.focused == 1, "KWAMI has the keyboard again"
    assert user32.attached == [(7, 120, True), (7, 120, False)], "joined only for the call"


def test_the_focus_is_left_alone_when_chrome_did_not_take_it() -> None:
    user32 = FocusUser32({1: (500, True, True), 20: (42, True, True)}, focused=1)
    api = Win32(user32, same, Kernel32())
    keep_behind(42, windows=True, api=api, sleep=lambda _s: None, focus=1)
    assert user32.attached == []
    keep_behind(42, windows=True, api=api, sleep=lambda _s: None)  # nothing to give back


def test_handing_on_the_focus_without_a_thread_to_join() -> None:
    user32 = FocusUser32({1: (500, True, True)}, focused=0)
    api = Win32(user32, same, Kernel32())
    assert api.foreground() == 0
    assert api.give_focus(1) is True
    assert user32.attached == [], "no window had the focus: nothing to join"

    user32.focused = 1
    user32.threads[1] = 7  # the focus is already ours
    assert api.give_focus(1) is True
    assert user32.attached == []


def test_the_window_with_the_focus_is_remembered_only_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user32 = FocusUser32({5: (1, True, True)}, focused=5)
    assert desktop.foreground_window(windows=True, api=Win32(user32, same)) == 5
    assert desktop.foreground_window(windows=False) == 0
    monkeypatch.setattr(desktop.os, "name", "posix")
    assert desktop.foreground_window() == 0
    monkeypatch.setattr(desktop.os, "name", "nt")
    monkeypatch.setattr(desktop, "win32", lambda: Win32(user32, same))
    assert desktop.foreground_window() == 5


def test_a_browser_window_that_took_the_focus_gives_it_back() -> None:
    """Claude in Chrome opens its group in a new window of the user's Chrome."""
    user32 = FocusUser32({1: (500, True, True), 30: (42, True, True)}, focused=30)
    api = Win32(user32, same, Kernel32())
    names = {42: "chrome.exe", 500: "kwami.exe"}
    assert desktop.return_focus(1, windows=True, api=api, name_of=names.get) is True
    assert user32.focused == 1, "KWAMI has the keyboard again"
    assert user32.moved and user32.moved[0][0] == 30, "the new window went behind"


def test_the_focus_stays_where_no_browser_took_it(monkeypatch: pytest.MonkeyPatch) -> None:
    user32 = FocusUser32({1: (500, True, True), 30: (42, True, True)}, focused=30)
    api = Win32(user32, same, Kernel32())
    assert not desktop.return_focus(1, windows=True, api=api, name_of=lambda _pid: "notepad.exe")
    user32.focused = 1
    assert not desktop.return_focus(1, windows=True, api=api, name_of=lambda _pid: "chrome.exe")
    user32.focused = 0
    assert not desktop.return_focus(1, windows=True, api=api, name_of=lambda _pid: "chrome.exe")
    assert not desktop.return_focus(0, windows=True, api=api)
    assert not desktop.return_focus(1, windows=False)
    assert user32.moved == [] and user32.attached == []
    monkeypatch.setattr(desktop.os, "name", "posix")
    assert not desktop.return_focus(1)
    monkeypatch.setattr(desktop.os, "name", "nt")
    monkeypatch.setattr(desktop, "win32", lambda: Win32(user32, same, Kernel32()))
    assert not desktop.return_focus(1)  # the default api, and nothing to give back


def test_a_process_is_named_or_left_unnamed(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    import psutil

    assert desktop.process_name(os.getpid()).startswith("python")

    def gone(_pid: int) -> Any:
        raise psutil.NoSuchProcess(_pid)

    monkeypatch.setattr(psutil, "Process", gone)
    assert desktop.process_name(1) == ""
