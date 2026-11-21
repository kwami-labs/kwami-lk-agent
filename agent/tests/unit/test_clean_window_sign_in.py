"""Signing in in the clean window, measured on the user's PC on 2026-11-21.

Google refused Jarvis's driven window as soon as its sign-in page loaded, and let the user in
through the clean one. Signing in there switched on Chrome's account sync, which brought the
user's extensions into Jarvis's profile. So: both windows start with sync and extensions off,
the profile neither saves nor fills passwords, and the clean window closes by itself once the
user is in.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from src.browser import local_chrome
from src.browser.local_chrome import (
    CLOSED,
    PROFILE_FLAGS,
    SIGNED_IN,
    STILL_OPEN,
    SignInProgress,
    harden_profile,
    is_refused_address,
    is_sign_in_address,
    is_web_address,
)


@pytest.fixture
def profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(local_chrome, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(local_chrome, "_clean_windows", {})
    monkeypatch.setattr(local_chrome, "CLEAN_WINDOW_POLL_SECONDS", 0.001)
    return tmp_path


# -- the profile stays Jarvis's own ----------------------------------------------------------


def test_both_windows_keep_the_users_account_out() -> None:
    assert "--disable-sync" in PROFILE_FLAGS
    assert "--disable-extensions" in PROFILE_FLAGS
    assert "--hide-crash-restore-bubble" in PROFILE_FLAGS
    assert not any(flag.startswith("--remote-debugging") for flag in PROFILE_FLAGS)


def test_a_new_profile_neither_saves_nor_fills_passwords(tmp_path: Path) -> None:
    assert harden_profile(tmp_path) is True
    prefs = json.loads((tmp_path / "Default" / "Preferences").read_text(encoding="utf-8"))
    assert prefs["credentials_enable_service"] is False
    assert prefs["profile"]["password_manager_enabled"] is False
    assert prefs["autofill"] == {"profile_enabled": False, "credit_card_enabled": False}
    assert harden_profile(tmp_path) is False, "nothing left to change"


def test_chromes_own_settings_are_kept(tmp_path: Path) -> None:
    prefs = tmp_path / "Default" / "Preferences"
    prefs.parent.mkdir()
    prefs.write_text(
        json.dumps({"profile": {"name": "Jarvis"}, "autofill": "odd", "intl": {"x": 1}}),
        encoding="utf-8",
    )
    assert harden_profile(tmp_path) is True
    data = json.loads(prefs.read_text(encoding="utf-8"))
    assert data["profile"] == {"name": "Jarvis", "password_manager_enabled": False}
    assert data["intl"] == {"x": 1}
    assert data["autofill"]["credit_card_enabled"] is False


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_preferences_it_cannot_read_are_left_alone(tmp_path: Path, content: str) -> None:
    prefs = tmp_path / "Default" / "Preferences"
    prefs.parent.mkdir()
    prefs.write_text(content, encoding="utf-8")
    assert harden_profile(tmp_path) is False
    assert prefs.read_text(encoding="utf-8") == content


def test_preferences_it_cannot_write_are_said_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_a: Any, **_k: Any) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "write_text", refuse)
    assert harden_profile(tmp_path) is False


def test_a_preferences_path_that_is_a_folder_is_left_alone(tmp_path: Path) -> None:
    (tmp_path / "Default" / "Preferences").mkdir(parents=True)
    assert harden_profile(tmp_path) is False


# -- what an address says --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("address", "web"),
    [
        ("mail.google.com/mail/u/0/#inbox", True),
        ("https://github.com/", True),
        ("http://intranet.example/x", True),
        ("gmail", False),
        ("como entrar en gmail", False),
        ("chrome://newtab/", False),
        ("chrome-extension://abc/onboarding.html", False),
        ("about:blank", False),
        ("", False),
        (".hidden", False),
    ],
)
def test_only_a_pages_address_counts(address: str, web: bool) -> None:
    assert is_web_address(address) is web


@pytest.mark.parametrize(
    ("address", "sign_in"),
    [
        ("accounts.google.com/v3/signin/identifier?continue=x", True),
        ("accounts.google.com/CheckCookie", True),
        ("login.live.com/oauth20_authorize.srf", True),
        ("github.com/login", True),
        ("github.com/sessions/two-factor", True),
        ("https://example.com/auth/sso/callback", True),
        ("accounts.google.com/v3/signin/rejected?x=1", True),
        ("mail.google.com/mail/u/0/#inbox", False),
        ("github.com/", False),
        ("myaccount.google.com/security", False),
    ],
)
def test_a_sign_in_page_is_told_by_its_address(address: str, sign_in: bool) -> None:
    assert is_sign_in_address(address) is sign_in


def test_a_refusal_is_told_apart() -> None:
    assert is_refused_address("accounts.google.com/v3/signin/rejected?continue=x") is True
    assert is_refused_address("accounts.google.com/v3/signin/identifier") is False


def test_the_user_is_in_once_off_every_sign_in_page_for_a_while() -> None:
    progress = SignInProgress(settle=4)
    assert progress.see("accounts.google.com/v3/signin/identifier", typing=False, now=0) is False
    assert progress.seen_sign_in
    assert progress.see("mail.google.com/mail/u/0/", typing=False, now=10) is False, (
        "just got there"
    )
    assert progress.see("mail.google.com/mail/u/0/#inbox", typing=False, now=12) is False
    assert progress.see("mail.google.com/mail/u/0/#inbox", typing=False, now=14) is True


def test_a_second_sign_in_page_starts_the_wait_again() -> None:
    progress = SignInProgress(settle=4)
    progress.see("github.com/login", typing=False, now=0)
    progress.see("github.com/", typing=False, now=1)
    progress.see("github.com/sessions/two-factor", typing=False, now=3)
    assert progress.see("github.com/", typing=False, now=6) is False
    assert progress.see("github.com/", typing=False, now=10) is True


def test_typing_and_browser_pages_say_nothing() -> None:
    progress = SignInProgress(settle=0)
    progress.see("accounts.google.com/signin", typing=False, now=0)
    assert progress.see("mail.google.com/", typing=True, now=5) is False, "the user is typing"
    assert progress.see("chrome://newtab/", typing=False, now=5) is False
    assert progress.see("mail.google.com/", typing=False, now=5) is True


def test_without_a_sign_in_page_it_is_never_taken_as_done() -> None:
    progress = SignInProgress(settle=0)
    assert progress.see("mail.google.com/", typing=False, now=100) is False


def test_a_refused_sign_in_is_never_done() -> None:
    progress = SignInProgress(settle=0)
    assert progress.see("accounts.google.com/v3/signin/rejected", typing=False, now=0) is False
    assert progress.refused


# -- the windows -----------------------------------------------------------------------------


class _Process:
    def __init__(self, code: int | None = None, pid: int = 4242) -> None:
        self.code = code
        self.pid = pid

    def poll(self) -> int | None:
        return self.code


async def test_both_windows_start_with_the_profile_flags(
    profiles: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launched: list[list[str]] = []
    hardened: list[Path] = []

    def popen(args: list[str], **_: Any) -> _Process:
        launched.append(args)
        return _Process()

    async def nothing(_profile: Path) -> None:
        return None

    monkeypatch.setattr(local_chrome.subprocess, "Popen", popen)
    monkeypatch.setattr(local_chrome, "_close_debug_window", nothing)
    monkeypatch.setattr(local_chrome, "harden_profile", hardened.append)
    await local_chrome.open_clean_window("u", "https://accounts.google.com/", chrome="c.exe")
    assert all(flag in launched[0] for flag in PROFILE_FLAGS)
    assert hardened == [local_chrome.profile_dir_for("u")]

    async def port(_process: Any, _profile: Path) -> int:
        return 9

    monkeypatch.setattr(local_chrome, "wait_for_port", port)
    monkeypatch.setattr(local_chrome, "foreground_window", lambda: 0)
    monkeypatch.setattr(local_chrome, "keep_behind", lambda *_a, **_k: 0)
    provider = local_chrome.LocalChromeProvider(chrome_path="c.exe")
    assert await provider._start_chrome("c.exe", profiles / "p") == 9
    assert "--remote-debugging-port=0" in launched[1]
    assert all(flag in launched[1] for flag in PROFILE_FLAGS)
    assert hardened[-1] == profiles / "p"
    await asyncio.gather(*local_chrome._behind)


def test_the_clean_window_is_closed_like_its_close_button(
    profiles: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(local_chrome, "close_windows", lambda pid: closed.append(pid) or 1)
    assert local_chrome.close_clean_window("u") is False, "none open"
    name = local_chrome.profile_dir_for("u").name
    local_chrome._clean_windows[name] = _Process(code=0)
    assert local_chrome.close_clean_window("u") is False, "already gone"
    local_chrome._clean_windows[name] = _Process(pid=77)
    assert local_chrome.close_clean_window("u") is True
    assert closed == [77]
    monkeypatch.setattr(local_chrome, "close_windows", lambda _pid: 0)
    assert local_chrome.close_clean_window("u") is False, "no window of it to close"


# -- waiting for the user ---------------------------------------------------------------------


class _Watch:
    """What the window's address reads, one after another, then nothing new."""

    def __init__(self, seen: list[dict[str, Any]], *, fail: bool = False) -> None:
        self.seen = list(seen)
        self.fail = fail
        self.closed = 0

    async def start(self) -> None:
        if self.fail:
            raise OSError("no powershell")

    async def next(self, _timeout: float) -> dict[str, Any] | None:
        await asyncio.sleep(0)
        return self.seen.pop(0) if self.seen else None

    async def close(self) -> None:
        self.closed += 1


def watching(monkeypatch: pytest.MonkeyPatch, watch: _Watch) -> list[tuple[str, dict]]:
    made: list[tuple[str, dict]] = []

    def make(profile: str, **kwargs: Any) -> _Watch:
        made.append((profile, kwargs))
        return watch

    monkeypatch.setattr(local_chrome.window_reader, "available", lambda: True)
    monkeypatch.setattr(local_chrome.window_reader, "AddressWatch", make)
    return made


async def test_once_the_user_is_in_the_window_closes_for_them(
    profiles: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch = _Watch(
        [
            {"address": "accounts.google.com/v3/signin/identifier", "typing": False},
            {"address": "accounts.google.com/v3/signin/challenge/pwd", "typing": False},
            {"address": "mail.google.com/mail/u/0/#inbox", "typing": False},
            {"address": "mail.google.com/mail/u/0/#inbox", "typing": False},
        ]
    )
    made = watching(monkeypatch, watch)
    monkeypatch.setattr(local_chrome, "SIGNED_IN_SECONDS", 0)
    open_now = {"open": True}
    closed: list[str] = []

    def close(user_id: str) -> bool:
        closed.append(user_id)
        open_now["open"] = False
        return True

    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: open_now["open"])
    monkeypatch.setattr(local_chrome, "close_clean_window", close)
    assert await local_chrome.wait_sign_in("u", 5) == SIGNED_IN
    assert closed == ["u"]
    assert watch.closed == 1
    assert made[0][0] == local_chrome.profile_dir_for("u").name


async def test_a_window_the_user_closes_is_closed(
    profiles: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    looks = [True, True, False]
    watching(monkeypatch, _Watch([{"address": "github.com/login"}, {"closed": True}]))
    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: looks.pop(0))
    assert await local_chrome.wait_sign_in("u", 5) == CLOSED
    looks.extend([True, False])
    assert await local_chrome.wait_clean_window_closed("u", 5) is True


async def test_a_window_that_will_not_close_is_watched_until_the_end(
    profiles: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watching(
        monkeypatch,
        _Watch(
            [
                {"address": "github.com/login"},
                {"address": "github.com/"},
                {"address": "github.com/"},
            ]
        ),
    )
    monkeypatch.setattr(local_chrome, "SIGNED_IN_SECONDS", 0)
    tries: list[str] = []
    monkeypatch.setattr(local_chrome, "close_clean_window", lambda u: tries.append(u) or False)
    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: True)
    assert await local_chrome.wait_sign_in("u", 0.05) == STILL_OPEN
    assert len(tries) == 2, "it keeps asking while the user is in"


async def test_once_closed_for_them_it_is_not_closed_again(
    profiles: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watching(
        monkeypatch,
        _Watch([{"address": "github.com/login"}, {"address": "github.com/"}, {"address": "x.io/"}]),
    )
    monkeypatch.setattr(local_chrome, "SIGNED_IN_SECONDS", 0)
    looks = [True, True, True, True, False]
    tries: list[str] = []
    monkeypatch.setattr(local_chrome, "close_clean_window", lambda u: tries.append(u) or True)
    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: looks.pop(0))
    assert await local_chrome.wait_sign_in("u", 5) == SIGNED_IN
    assert tries == ["u"]


async def test_where_the_address_cannot_be_read_the_user_closes_it(
    profiles: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_chrome.window_reader, "available", lambda: False)
    looks = [True, False]
    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: looks.pop(0))
    assert await local_chrome.wait_sign_in("u", 5) == CLOSED
    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: True)
    assert await local_chrome.wait_clean_window_closed("u", 0.01) is False


async def test_a_watch_that_cannot_start_falls_back_to_waiting(
    profiles: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = _Watch([], fail=True)
    watching(monkeypatch, broken)
    looks = [True, False]
    monkeypatch.setattr(local_chrome, "clean_window_open", lambda _u: looks.pop(0))
    assert await local_chrome.wait_sign_in("u", 5) == CLOSED
    assert broken.closed == 1


async def test_the_watch_lasts_as_long_as_the_wait(
    profiles: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = watching(monkeypatch, _Watch([]))
    assert await local_chrome._start_watch("u", 600) is not None
    assert made[0][1] == {"every": local_chrome.ADDRESS_POLL_SECONDS, "limit": 605}
