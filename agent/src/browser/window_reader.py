"""What Windows' accessibility layer (UI Automation) says about browser windows.

Measured on the user's PC on 2026-11-21 (Windows 10, Chrome 157 and Brave): every window of
the Chrome family exposes its tab strip, one `TabItem` per tab named after the page (some
as "Uso de memoria de <title>: 330 MB"), and its address bar (`OmniboxViewViews`,
`BraveOmniboxViewViews` in Brave) holding the address of the tab in front, without the
scheme. Reading them needs no extension, no debugging port and nothing the user installs.
It reads; it never clicks or types.

The walk skips the pages themselves (`Chrome_RenderWidgetHostHWND`, documents): touching
them would make Chrome build the accessibility tree of every web page, which slows it. A
read costs a PowerShell start and about a second.

Two uses:

* **The user's tabs** (`browser_tabs`), when no extension connects their browser: the titles
  of every tab in Chrome, Brave or Edge, and the address of the one in front of each
  window. Jarvis's own windows (profiles under `.jarvis\\browser-profiles`) are left out.
* **A sign-in window** (`AddressWatch`): the address of the clean window the user signs in
  in, which no program drives, so that Jarvis knows when they are in (`local_chrome`).

`KWAMI_WINDOW_READER=off` turns it off. Elsewhere than Windows there is nothing to read.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..utils.logging import get_logger

logger = get_logger("browser.window_reader")

SWITCH = "KWAMI_WINDOW_READER"
#: A read of every window; the user's tabs are asked for once in a while, never in a loop.
READ_TIMEOUT_SECONDS = 15.0
#: How long a closed watch gets to end before it is killed.
STOP_SECONDS = 2.0
#: Its own console would flash on the user's screen for every read.
CREATE_NO_WINDOW = 0x08000000
_PROFILE = re.compile(r"^[0-9a-f]{8,64}$")

#: What every script starts with: UTF-8 out, quiet errors, and the walk that finds the
#: tab strip and the address bar without going into the pages.
_COMMON = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'SilentlyContinue'
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
$A = [System.Windows.Automation.AutomationElement]
$T = [System.Windows.Automation.ControlType]
$walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
$value = [System.Windows.Automation.ValuePattern]::Pattern
$children = [System.Windows.Automation.TreeScope]::Children
function Read-Window($w) {
  $tabs = New-Object System.Collections.ArrayList
  $address = ''
  $typing = $false
  $stack = New-Object System.Collections.Stack
  $stack.Push($w)
  $visited = 0
  while ($stack.Count -gt 0 -and $visited -lt 3000) {
    $el = $stack.Pop(); $visited++
    $c = $el.Current
    if ($c.ClassName -eq 'Chrome_RenderWidgetHostHWND' -or $c.ControlType -eq $T::Document) { continue }
    if ($c.ControlType -eq $T::TabItem) { [void]$tabs.Add($c.Name); continue }
    if ($c.ClassName -like '*OmniboxViewViews') {
      $address = $el.GetCurrentPattern($value).Current.Value
      $typing = [bool]$c.HasKeyboardFocus
      continue
    }
    $kids = New-Object System.Collections.ArrayList
    $child = $walker.GetFirstChild($el)
    while ($child) { [void]$kids.Add($child); $child = $walker.GetNextSibling($child) }
    for ($i = $kids.Count - 1; $i -ge 0; $i--) { $stack.Push($kids[$i]) }
  }
  return @{ tabs = @($tabs); address = [string]$address; typing = $typing }
}
"""

#: Every window of the user's browsers: which browser, its title, its tabs, its address.
TABS_SCRIPT = (
    _COMMON
    + r"""
$browsers = @{}
Get-CimInstance Win32_Process -Filter "Name='chrome.exe' OR Name='brave.exe' OR Name='msedge.exe'" |
  Where-Object { $_.CommandLine -notlike '*--type=*' -and $_.CommandLine -notlike '*browser-profiles*' } |
  ForEach-Object { $browsers[[int]$_.ProcessId] = $_.Name }
$win = New-Object System.Windows.Automation.PropertyCondition($A::ClassNameProperty, 'Chrome_WidgetWin_1')
$out = @()
foreach ($w in $A::RootElement.FindAll($children, $win)) {
  $id = [int]$w.Current.ProcessId
  if (-not $browsers.ContainsKey($id) -or -not $w.Current.Name) { continue }
  $r = Read-Window $w
  $out += [pscustomobject]@{ browser = $browsers[$id]; title = $w.Current.Name; address = $r.address; tabs = $r.tabs }
}
[Console]::Out.WriteLine((ConvertTo-Json -InputObject @($out) -Depth 4 -Compress))
"""
)

#: One line each time the window of profile __PROFILE__ shows another address, and a last
#: one when it is gone. Ends by itself after __LIMIT__ seconds.
WATCH_SCRIPT = (
    _COMMON
    + r"""
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$last = ''
$id = 0
while ($sw.Elapsed.TotalSeconds -lt __LIMIT__) {
  if (-not $id -or -not (Get-Process -Id $id)) {
    $id = (Get-CimInstance Win32_Process -Filter "Name='chrome.exe' OR Name='msedge.exe'" |
      Where-Object { $_.CommandLine -like '*__PROFILE__*' -and $_.CommandLine -notlike '*--type=*' } |
      Select-Object -First 1).ProcessId
  }
  if (-not $id) { [Console]::Out.WriteLine('{"closed":true}'); [Console]::Out.Flush(); break }
  $cond = New-Object System.Windows.Automation.AndCondition(
    (New-Object System.Windows.Automation.PropertyCondition($A::ProcessIdProperty, [int]$id)),
    (New-Object System.Windows.Automation.PropertyCondition($A::ClassNameProperty, 'Chrome_WidgetWin_1')))
  foreach ($w in $A::RootElement.FindAll($children, $cond)) {
    if (-not $w.Current.Name) { continue }
    $r = Read-Window $w
    $line = ConvertTo-Json -Compress -InputObject ([ordered]@{ address = $r.address; typing = $r.typing; title = $w.Current.Name })
    if ($line -ne $last) { [Console]::Out.WriteLine($line); [Console]::Out.Flush(); $last = $line }
    break
  }
  Start-Sleep -Milliseconds __EVERY_MS__
}
"""
)

_MEMORY_PREFIX = re.compile(r"^(?:Uso de memoria de |Memory usage (?:of |for )?)", re.IGNORECASE)
_MEMORY_SUFFIX = re.compile(r"\s*(?::|-)\s*[\d.,]+\s*(?:KB|MB|GB)\s*$", re.IGNORECASE)


def available(windows: bool | None = None) -> bool:
    """Whether windows can be read here: Windows, with PowerShell, and not turned off."""
    if windows is None:
        windows = os.name == "nt"
    if not windows or os.environ.get(SWITCH, "").strip().lower() in ("off", "0", "false"):
        return False
    return bool(powershell())


def powershell() -> str:
    """Windows PowerShell, which every Windows 10 and 11 has ("" when it is not found)."""
    return shutil.which("powershell.exe") or shutil.which("powershell") or ""


def tab_title(name: str) -> str:
    """A tab's title as the user reads it, without the memory figure Chrome adds to some."""
    name = str(name or "").strip()
    if _MEMORY_PREFIX.match(name):
        name = _MEMORY_SUFFIX.sub("", _MEMORY_PREFIX.sub("", name, count=1))
    return name.strip()


def _command(script: str) -> list[str]:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return [
        powershell(),
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-EncodedCommand",
        encoded,
    ]


async def _spawn(script: str) -> Any:
    return await asyncio.create_subprocess_exec(
        *_command(script),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )


Spawn = Callable[[str], Awaitable[Any]]


@dataclass
class BrowserWindow:
    """One window of the user's browser, as Windows shows it."""

    browser: str
    title: str
    address: str = ""
    tabs: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The browser by the name the user knows."""
        exe = self.browser.lower().removesuffix(".exe")
        return {"chrome": "Chrome", "brave": "Brave", "msedge": "Edge"}.get(exe, exe or "browser")


def parse_windows(text: str) -> list[BrowserWindow]:
    """The windows in the script's answer; nothing when it is not what the script prints."""
    try:
        data = json.loads(text or "[]")
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    windows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        tabs = item.get("tabs") or []
        if isinstance(tabs, str):
            tabs = [tabs]
        windows.append(
            BrowserWindow(
                browser=str(item.get("browser") or ""),
                title=str(item.get("title") or ""),
                address=str(item.get("address") or ""),
                tabs=[t for t in (tab_title(x) for x in tabs if isinstance(x, str)) if t],
            )
        )
    return windows


def as_url(address: str) -> str:
    """An address bar's text as a URL: Chrome leaves out "https://"."""
    address = (address or "").strip()
    if re.match(r"^[a-z][a-z0-9+.-]*://", address, re.IGNORECASE):
        return address
    return f"https://{address}"


def _in_front(window: BrowserWindow, tab: str) -> bool:
    """Whether `tab` is the one in front: the window is called after it ("<tab> - Google Chrome")."""
    return bool(tab) and window.title.startswith(tab)


def _find(numbered: list[tuple[int, BrowserWindow, str]], wanted: str) -> Any:
    wanted = wanted.strip().lower()
    if wanted.isdigit():
        return next((row for row in numbered if row[0] == int(wanted)), None)
    words = wanted.split()
    return next((row for row in numbered if all(w in row[2].lower() for w in words)), None)


async def describe_tabs(
    windows: list[BrowserWindow],
    wanted: str,
    *,
    listing: bool,
    blocked: Callable[[str], Awaitable[bool]],
    open_with: str = "navigate_to",
) -> str:
    """What to tell the model about the user's tabs, read from Windows.

    `listing`: every tab, numbered. Otherwise the one `wanted` names, by number or by words
    of its title, and how to read it: the tab in front of a window has an address, which
    Jarvis can open in its own browser; any other shows only its title. `blocked` says
    whether the user's list of sites keeps Jarvis out of an address.
    """
    numbered = [
        (n, window, tab)
        for n, (window, tab) in enumerate(
            ((w, t) for w in windows for t in w.tabs),
            start=1,
        )
    ]
    if listing:
        lines = [f"The user has {len(numbered)} tabs open in their own browsers:"]
        for window in windows:
            if window.address and await blocked(as_url(window.address)):
                front = "the tab in front is on a site their list keeps you out of"
            elif window.address:
                front = f"the tab in front is at {window.address}"
            else:
                front = "the tab in front has no address (a new tab or one of the browser's own)"
            lines.append(f"{window.name} window, {front}:")
            lines.extend(f"{n}. {tab}" for n, w, tab in numbered if w is window)
        return "\n".join(lines)
    row = _find(numbered, wanted)
    if row is None:
        return f"No tab of the user's matches '{wanted[:60]}'. List them first."
    n, window, tab = row
    if not (_in_front(window, tab) and window.address):
        return (
            f"Tab {n}, '{tab}', is open in their {window.name}, but from here you see only its "
            "title. To read it, ask them to bring it to the front and ask again, or open the "
            f"site yourself with {open_with}."
        )
    url = as_url(window.address)
    if await blocked(url):
        return f"Tab {n} is on a site the user's list keeps you out of."
    return (
        f"Tab {n}, '{tab}', is the one in front of their {window.name} window, at {url}. "
        f"To read it, open that address with {open_with}: it opens in your own browser, "
        "where their sign-in may not be."
    )


async def browser_tabs(spawn: Spawn | None = None) -> list[BrowserWindow]:
    """The windows of the user's own browsers, with their tabs. Empty when there are none,
    when windows cannot be read here, or when the read fails."""
    if spawn is None:
        if not available():
            return []
        spawn = _spawn
    process = None
    try:
        process = await spawn(TABS_SCRIPT)
        out, _ = await asyncio.wait_for(process.communicate(), READ_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("Could not read the browser windows", exc_info=True)
        if process is not None:
            with contextlib.suppress(Exception):
                process.kill()
        return []
    return parse_windows(out.decode("utf-8-sig", errors="replace").strip())


class AddressWatch:
    """The address of one of Jarvis's windows, read every `every` seconds while it is open.

    `next()` gives what changed since the last call: `{"address", "typing", "title"}`, or
    `{"closed": True}` once the window is gone. Nothing (None) if nothing changed in time.
    """

    def __init__(
        self,
        profile: str,
        *,
        every: float = 1.0,
        limit: float = 900.0,
        spawn: Spawn | None = None,
    ) -> None:
        if not _PROFILE.match(profile or ""):
            raise ValueError("a profile folder name is needed")
        self._script = (
            WATCH_SCRIPT.replace("__PROFILE__", profile)
            .replace("__LIMIT__", str(max(1, int(limit))))
            .replace("__EVERY_MS__", str(max(100, int(every * 1000))))
        )
        self._spawn = spawn or _spawn
        self._process: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._seen: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def start(self) -> None:
        self._process = await self._spawn(self._script)
        self._reader = asyncio.create_task(self._read(), name="window_reader_watch")

    async def _read(self) -> None:
        stdout = self._process.stdout
        while True:
            line = await stdout.readline()
            if not line:
                return
            try:
                seen = json.loads(line.decode("utf-8-sig", errors="replace"))
            except ValueError:
                continue
            if isinstance(seen, dict):
                await self._seen.put(seen)

    async def next(self, timeout: float) -> dict[str, Any] | None:
        try:
            return await asyncio.wait_for(self._seen.get(), timeout)
        except TimeoutError:
            return None

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            with contextlib.suppress(Exception):
                process.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), STOP_SECONDS)
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
            self._reader = None
