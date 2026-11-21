"""Jarvis's own Chrome on the user's desktop: out of the way, mirrored on Jarvis's screen.

Measured on 2026-11-17, Windows: a Chrome started from the agent's process took the focus
and covered KWAMI, translation bubble and all, and the session then called
`Page.bringToFront` on top of that. That is the window the user saw beside the HUD,
showing the same page twice. Now:

* **It starts without the focus.** `SW_SHOWNOACTIVATE` in the start-up info, and if Chrome
  takes the focus all the same, it goes back to the window that had it. Measured in the
  real app: with KWAMI untouched for a few minutes, Windows lets a new window take the
  focus, and Chrome did.
* **It goes behind everything** once its windows exist (`HWND_BOTTOM`, without activating
  anything). Measured: a page behind every other window still answers
  `Page.captureScreenshot`, so the HUD keeps its live mirror. Minimised it does not, which
  is why it is sent back rather than minimised.
* **It comes forward only for the user**: a sign-in page (`bring_to_front`), or a click on
  its taskbar button.

Elsewhere than Windows nothing here does anything: the window is the desktop's business.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from typing import Any

#: STARTF_USESHOWWINDOW and SW_SHOWNOACTIVATE: shown, never activated.
STARTF_USESHOWWINDOW = 0x00000001
SW_SHOWNOACTIVATE = 4
#: SetWindowPos: to the bottom, keeping size and place, activating nothing.
HWND_BOTTOM = 1
SWP_BEHIND = 0x0001 | 0x0002 | 0x0010  # NOSIZE | NOMOVE | NOACTIVATE
#: How long a fresh Chrome gets to put up its window, and how often it is looked for.
WINDOW_WAIT_SECONDS = 5.0
POLL_SECONDS = 0.1
#: Bubbles (translate, restore) arrive a moment after the window: one more pass then.
SETTLE_SECONDS = 1.0


def quiet_startup(windows: bool | None = None) -> Any | None:
    """Start-up info that shows a program's first window without taking the focus."""
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return None
    info = subprocess.STARTUPINFO()  # type: ignore[attr-defined,unused-ignore]
    info.dwFlags |= STARTF_USESHOWWINDOW
    info.wShowWindow = SW_SHOWNOACTIVATE
    return info


class Win32:
    """The few calls of user32 this needs, behind a face a test can stand in for."""

    def __init__(self, user32: Any, functype: Callable[..., Any], kernel32: Any = None) -> None:
        self._user32 = user32
        self._callback = functype
        self._kernel32 = kernel32

    def foreground(self) -> int:
        """The window with the keyboard focus now (0 if none)."""
        return int(self._user32.GetForegroundWindow() or 0)

    def owner(self, hwnd: int) -> tuple[int, int]:
        """The process and the thread a window belongs to."""
        import ctypes

        pid = ctypes.c_ulong()
        thread = self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value), int(thread)

    def give_focus(self, hwnd: int) -> bool:
        """Hand the focus back to `hwnd`. A process without it may only do so while its
        input is joined to that of the window that has it, for the length of the call."""
        user32 = self._user32
        _pid, theirs = self.owner(self.foreground())
        mine = int(self._kernel32.GetCurrentThreadId())
        joined = bool(theirs) and theirs != mine and user32.AttachThreadInput(mine, theirs, True)
        try:
            return bool(user32.SetForegroundWindow(hwnd))
        finally:
            if joined:
                user32.AttachThreadInput(mine, theirs, False)

    def windows_of(self, pid: int) -> list[int]:
        """The visible, titled top-level windows of process `pid`."""
        import ctypes

        found: list[int] = []
        user32 = self._user32

        def visit(hwnd: int, _param: int) -> bool:
            owner = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if (
                owner.value == pid
                and user32.IsWindowVisible(hwnd)
                and user32.GetWindowTextLengthW(hwnd)
            ):
                found.append(hwnd)
            return True

        user32.EnumWindows(self._callback(visit), 0)
        return found

    def send_back(self, hwnd: int) -> None:
        self._user32.SetWindowPos(hwnd, HWND_BOTTOM, 0, 0, 0, 0, SWP_BEHIND)


def win32() -> Win32:
    import ctypes

    return Win32(
        ctypes.WinDLL("user32"),  # type: ignore[attr-defined,unused-ignore]
        ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p),  # type: ignore[attr-defined,unused-ignore]
        ctypes.WinDLL("kernel32"),  # type: ignore[attr-defined,unused-ignore]
    )


def foreground_window(windows: bool | None = None, api: Win32 | None = None) -> int:
    """The window with the focus before Chrome starts, to give it back. 0 elsewhere."""
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return 0
    return (api or win32()).foreground()


def keep_behind(
    pid: int,
    *,
    windows: bool | None = None,
    api: Win32 | None = None,
    sleep: Callable[[float], None] = time.sleep,
    wait: float = WINDOW_WAIT_SECONDS,
    focus: int = 0,
) -> int:
    """Send process `pid`'s windows behind every other one. How many it sent; 0 elsewhere.

    `focus` is the window that had the keyboard before: if one of Chrome's has taken it,
    it goes back there. Blocking (it waits for the window to appear): run it in a thread.
    """
    if windows is None:
        windows = os.name == "nt"
    if not windows:
        return 0
    api = api or win32()
    waited = 0.0
    found = api.windows_of(pid)
    while not found and waited < wait:
        sleep(POLL_SECONDS)
        waited += POLL_SECONDS
        found = api.windows_of(pid)
    if not found:
        return 0
    sent: set[int] = set()
    for hwnd in found:
        api.send_back(hwnd)
        sent.add(hwnd)
    # A bubble that opens a moment later (translate, restore) goes back as well.
    sleep(SETTLE_SECONDS)
    for hwnd in api.windows_of(pid):
        api.send_back(hwnd)
        sent.add(hwnd)
    if focus and api.owner(api.foreground())[0] == pid:
        api.give_focus(focus)
    return len(sent)


#: The browsers whose new windows give the focus back (Claude in Chrome opens its group in one).
BROWSER_PROCESSES = frozenset({"chrome.exe", "brave.exe", "msedge.exe", "chrome", "brave"})


def process_name(pid: int) -> str:
    """A process's executable name, lower case ("" if it cannot be told)."""
    try:
        import psutil

        return str(psutil.Process(pid).name()).lower()
    except Exception:
        return ""


def return_focus(
    before: int,
    *,
    windows: bool | None = None,
    api: Win32 | None = None,
    name_of: Callable[[int], str] = process_name,
) -> bool:
    """If a browser window took the focus from `before`, send it behind and give the focus back.

    For the user's own Chrome, where Jarvis cannot start the window without activating it
    (Claude in Chrome opens its tab group in a new window). Whether it gave the focus back.
    """
    if windows is None:
        windows = os.name == "nt"
    if not windows or not before:
        return False
    api = api or win32()
    now = api.foreground()
    if not now or now == before or name_of(api.owner(now)[0]) not in BROWSER_PROCESSES:
        return False
    api.send_back(now)
    return api.give_focus(before)
