"""What Jarvis's own Chrome downloads goes to his folder, under its own name, and is told.

Measured on Chrome 157 (2026-11-21): over the session's page connection,
`Browser.setDownloadBehavior` with `allowAndName` stores each file under its guid and says
so through `Browser.downloadWillBegin` and `Browser.downloadProgress`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.browser import browser_session, downloads
from src.browser.browser_session import CloudBrowserSession
from src.browser.downloads import (
    Download,
    DownloadWatch,
    downloads_note,
    safe_name,
    unique_path,
    with_downloads,
)
from src.browser.launched import LOCAL, LaunchedBrowser
from src.browser.local_chrome import LocalChromeProvider

# -- names ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "safe"),
    [
        ("factura 2026.pdf", "factura 2026.pdf"),
        ('a<b>:"c"/d\\e|f?g*.txt', "a_b___c__d_e_f_g_.txt"),
        ("..\\..\\Windows\\evil.exe", "_.._Windows_evil.exe"),
        ("con.txt", "con_.txt"),
        ("NUL", "NUL_"),
        ("", "descarga"),
        ("...", "descarga"),
        (".bashrc", "bashrc"),
        ("x" * 300 + ".pdf", "x" * 116 + ".pdf"),
        ("informe." + "y" * 40, "informe." + "y" * 16),
    ],
)
def test_a_name_windows_accepts(given: str, safe: str) -> None:
    assert safe_name(given) == safe


def test_a_taken_name_gets_a_number(tmp_path: Path) -> None:
    assert unique_path(tmp_path, "a.pdf") == tmp_path / "a.pdf"
    (tmp_path / "a.pdf").write_text("1")
    (tmp_path / "a (2).pdf").write_text("2")
    assert unique_path(tmp_path, "a.pdf") == tmp_path / "a (3).pdf"
    (tmp_path / "notas").write_text("1")
    assert unique_path(tmp_path, "notas") == tmp_path / "notas (2)"


def test_a_folder_full_of_one_name_keeps_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "exists", lambda _self: True)
    assert unique_path(tmp_path, "a.pdf") == tmp_path / "a.pdf"


def test_what_the_model_is_told() -> None:
    done = Download("g", "a.pdf", state="completed", path=Path("C:/J/a.pdf"), size=2048)
    assert done.said() == f"a.pdf saved at {Path('C:/J/a.pdf')}, 2 KB"
    tiny = Download("g", "b.txt", state="completed", path=Path("b.txt"), size=10)
    assert tiny.said().endswith(", 1 KB")
    assert Download("g", "c.zip", state="canceled").said() == "c.zip did not finish (canceled)"
    assert downloads_note([]) == ""
    note = downloads_note([done])
    assert "Documents\\Jarvis\\descargas" in note and "never run one" in note


# -- the watch -----------------------------------------------------------------------------


class _Cdp:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple[str, dict]] = []
        self.handlers: dict[str, Any] = {}

    async def send(self, method: str, **params: Any) -> dict:
        if self.fail:
            raise RuntimeError("'Browser.setDownloadBehavior' wasn't found")
        self.sent.append((method, params))
        return {}

    def on(self, method: str, handler: Any) -> None:
        self.handlers[method] = handler


async def test_the_browser_downloads_into_jarvis_folder(tmp_path: Path) -> None:
    folder = tmp_path / "descargas"
    cdp = _Cdp()
    watch = DownloadWatch(folder)
    assert await watch.attach(cdp) is True
    assert folder.is_dir()
    assert cdp.sent == [
        (
            "Browser.setDownloadBehavior",
            {"behavior": "allowAndName", "downloadPath": str(folder), "eventsEnabled": True},
        )
    ]
    assert set(cdp.handlers) == {"Browser.downloadWillBegin", "Browser.downloadProgress"}


async def test_a_browser_that_will_not_route_keeps_its_default(tmp_path: Path) -> None:
    cdp = _Cdp(fail=True)
    assert await DownloadWatch(tmp_path).attach(cdp) is False
    assert cdp.handlers == {}


def test_by_default_it_is_jarvis_folder(tmp_path: Path) -> None:
    assert DownloadWatch().folder == downloads.DOWNLOADS_DIR == tmp_path


def test_a_finished_download_gets_the_sites_name(tmp_path: Path) -> None:
    told: list[Download] = []
    watch = DownloadWatch(tmp_path, on_done=told.append)
    watch.begin({"guid": "g1", "suggestedFilename": "factura.pdf", "url": "https://a.b/f"})
    (tmp_path / "g1").write_bytes(b"%PDF")
    watch.progress({"guid": "g1", "state": "inProgress", "receivedBytes": 2})
    assert told == []
    watch.progress({"guid": "g1", "state": "completed", "receivedBytes": 4})
    assert (tmp_path / "factura.pdf").read_bytes() == b"%PDF"
    assert not (tmp_path / "g1").exists()
    assert told[0].path == tmp_path / "factura.pdf" and told[0].size == 4
    assert told[0].url == "https://a.b/f"
    assert watch.take_finished() == told
    assert watch.take_finished() == [], "each is told once"


def test_a_second_file_of_the_same_name_does_not_overwrite(tmp_path: Path) -> None:
    watch = DownloadWatch(tmp_path)
    for guid in ("g1", "g2"):
        watch.begin({"guid": guid, "suggestedFilename": "a.pdf"})
        (tmp_path / guid).write_text(guid)
        watch.progress({"guid": guid, "state": "completed"})
    assert (tmp_path / "a.pdf").read_text() == "g1"
    assert (tmp_path / "a (2).pdf").read_text() == "g2"


def test_a_canceled_download_leaves_nothing_behind(tmp_path: Path) -> None:
    watch = DownloadWatch(tmp_path)
    watch.begin({"guid": "g1"})
    (tmp_path / "g1").write_text("half")
    watch.progress({"guid": "g1", "state": "canceled"})
    assert list(tmp_path.iterdir()) == []
    (item,) = watch.take_finished()
    assert item.state == "canceled" and item.name == "descarga"


def test_events_it_did_not_start_are_ignored(tmp_path: Path) -> None:
    watch = DownloadWatch(tmp_path)
    watch.begin({})
    watch.progress({"guid": "nobody", "state": "completed"})
    assert watch.take_finished() == []


def test_a_file_that_cannot_be_renamed_is_kept_under_its_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_a: Any) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(downloads.os, "replace", refuse)
    watch = DownloadWatch(tmp_path)
    watch.begin({"guid": "g1", "suggestedFilename": "a.pdf"})
    (tmp_path / "g1").write_text("x")
    watch.progress({"guid": "g1", "state": "completed"})
    watch.begin({"guid": "g2", "suggestedFilename": "b.pdf"})
    watch.progress({"guid": "g2", "state": "completed"})
    kept, lost = watch.take_finished()
    assert kept.path == tmp_path / "g1" and kept.state == "completed"
    assert lost.path is None and lost.state == "unsaved"


def test_only_the_latest_downloads_are_kept(tmp_path: Path) -> None:
    watch = DownloadWatch(tmp_path)
    for n in range(downloads.MAX_KEPT + 5):
        watch.begin({"guid": f"g{n}"})
        watch.progress({"guid": f"g{n}", "state": "canceled"})
    kept = watch.take_finished()
    assert len(kept) == downloads.MAX_KEPT and kept[-1].guid == f"g{downloads.MAX_KEPT + 4}"


def test_a_listener_that_fails_does_not_lose_the_download(tmp_path: Path) -> None:
    def broken(_item: Download) -> None:
        raise RuntimeError("room gone")

    watch = DownloadWatch(tmp_path, on_done=broken)
    watch.begin({"guid": "g1"})
    watch.progress({"guid": "g1", "state": "canceled"})
    assert len(watch.take_finished()) == 1


def test_the_note_rides_on_whatever_was_said() -> None:
    class Session:
        def __init__(self, finished: list[Download] | Exception) -> None:
            self.finished = finished

        def finished_downloads(self) -> list[Download]:
            if isinstance(self.finished, Exception):
                raise self.finished
            return self.finished

    done = Download("g", "a.pdf", state="completed", path=Path("a.pdf"))
    assert with_downloads(Session([done]), "Clicked.").startswith("Clicked.\n\nThe browser")
    assert with_downloads(Session([]), "Clicked.") == "Clicked."
    assert with_downloads(Session(RuntimeError("x")), "Clicked.") == "Clicked."
    assert with_downloads(object(), "Clicked.") == "Clicked.", "the user's own Chrome"


# -- the session ---------------------------------------------------------------------------


def _cdp() -> MagicMock:
    cdp = MagicMock(is_connected=True, connect=AsyncMock(), send=AsyncMock(), close=AsyncMock())
    cdp.screenshot = AsyncMock(return_value="")
    return cdp


async def test_jarvis_window_on_the_pc_routes_its_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = LocalChromeProvider(chrome_path="chrome.exe")
    launched = LaunchedBrowser(browser_id="local-1", vendor=LOCAL, cdp_http_url="http://x")
    provider.launch = AsyncMock(return_value=launched)  # type: ignore[method-assign]
    cdp = _cdp()
    monkeypatch.setattr(browser_session, "CDPConnection", lambda: cdp)
    room = MagicMock()
    room.local_participant.publish_data = AsyncMock()
    session = CloudBrowserSession(room=room, provider=provider)
    await session.start("user-1")
    methods = [call.args[0] for call in cdp.send.await_args_list]
    assert "Browser.setDownloadBehavior" in methods
    handlers = {call.args[0]: call.args[1] for call in cdp.on.call_args_list}
    handlers["Browser.downloadWillBegin"]({"guid": "g1", "suggestedFilename": "a.pdf"})
    (tmp_path / "g1").write_text("x")
    handlers["Browser.downloadProgress"]({"guid": "g1", "state": "completed", "receivedBytes": 1})
    await asyncio.gather(*session._background)
    payload = room.local_participant.publish_data.await_args_list[-1].args[0]
    assert b'"type": "browser_download"' in payload and b"a.pdf" in payload
    assert [d.path for d in session.finished_downloads()] == [tmp_path / "a.pdf"]
    await session.close()
    assert session.finished_downloads() == []


async def test_a_rented_browser_leaves_downloads_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MagicMock()
    provider.launch = AsyncMock(
        return_value=LaunchedBrowser(browser_id="b-1", vendor="browser_use", cdp_http_url="h")
    )
    provider.release = AsyncMock()
    cdp = _cdp()
    monkeypatch.setattr(browser_session, "CDPConnection", lambda: cdp)
    session = CloudBrowserSession(room=None, provider=provider)
    await session.start("user-1")
    assert "Browser.setDownloadBehavior" not in [c.args[0] for c in cdp.send.await_args_list]
    assert session.finished_downloads() == []
    await session.close()


async def test_a_download_is_told_to_the_app_only_when_there_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = CloudBrowserSession(room=None)
    await session._publish_download(Download("g", "a.pdf", state="canceled"))
    room = MagicMock()
    room.local_participant.publish_data = AsyncMock(side_effect=RuntimeError("gone"))
    session.set_room(room)
    await session._publish_download(Download("g", "a.pdf", state="canceled"))
    room.local_participant.publish_data.assert_awaited_once()


# -- waiting for a navigation that turned into a download -----------------------------------


async def test_a_navigation_waits_for_its_download(tmp_path: Path) -> None:
    watch = DownloadWatch(tmp_path)

    async def chrome() -> None:
        await asyncio.sleep(0.05)
        watch.begin({"guid": "g1", "suggestedFilename": "a.zip"})
        assert [d.name for d in watch.in_progress()] == ["a.zip"]
        await asyncio.sleep(0.05)
        watch.progress({"guid": "g1", "state": "canceled"})

    task = asyncio.create_task(chrome())
    await watch.settle(5, grace=1)
    assert task.done() and watch.in_progress() == []
    await task


async def test_with_nothing_downloading_the_wait_is_short(tmp_path: Path) -> None:
    watch = DownloadWatch(tmp_path)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await watch.settle(5, grace=0.05)
    assert loop.time() - start < 1


async def test_a_slow_download_is_left_going(tmp_path: Path) -> None:
    watch = DownloadWatch(tmp_path)
    watch.begin({"guid": "g1", "suggestedFilename": "big.iso"})
    await watch.settle(0.05, grace=0)
    assert [d.name for d in watch.in_progress()] == ["big.iso"]
    note = downloads_note([], watch.in_progress())
    assert "big.iso is still downloading" in note


def test_the_note_names_what_is_still_going() -> None:
    class Session:
        def finished_downloads(self) -> list[Download]:
            return []

        def pending_downloads(self) -> list[Download]:
            return [Download("g", "big.iso")]

    assert "big.iso is still downloading" in with_downloads(Session(), "Opened.")


async def test_the_session_waits_only_when_chrome_says_it_is_a_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = CloudBrowserSession(room=None)
    await session._settle_download({"isDownload": True})  # no watch: nothing to wait for
    waited: list[float] = []

    class Watch:
        async def settle(self, timeout: float) -> None:
            waited.append(timeout)

        def in_progress(self) -> list[Download]:
            return [Download("g", "a.zip")]

    session._downloads = Watch()  # type: ignore[assignment]
    await session._settle_download({"frameId": "f"})
    await session._settle_download(None)
    assert waited == []
    await session._settle_download({"frameId": "f", "isDownload": True})
    assert waited == [downloads.DOWNLOAD_WAIT_SECONDS]
    assert [d.name for d in session.pending_downloads()] == ["a.zip"]
    session._downloads = None
    assert session.pending_downloads() == []
