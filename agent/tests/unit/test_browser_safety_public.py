"""The check the built-in browser's proxy connects by."""

from __future__ import annotations

import pytest

from src.browser import safety
from src.browser.safety import UnsafeURLError, public_addresses, validate_url


def _resolves(monkeypatch: pytest.MonkeyPatch, *addresses: str) -> None:
    monkeypatch.setattr(safety, "_resolved_addresses", lambda _host: list(addresses))


@pytest.mark.parametrize("host", ["", "localhost", "LOCALHOST.", "a.localhost", "metadata"])
def test_names_that_are_never_public(host: str) -> None:
    with pytest.raises(UnsafeURLError):
        public_addresses(host)


def test_a_public_name_gives_its_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolves(monkeypatch, "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946")
    assert public_addresses("example.com") == [
        "93.184.216.34",
        "2606:2800:220:1:248:1893:25c8:1946",
    ]


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "::1",
        "::ffff:127.0.0.1",
        "not-an-address",
    ],
)
def test_one_address_that_is_not_public_refuses_the_name(
    monkeypatch: pytest.MonkeyPatch, address: str
) -> None:
    _resolves(monkeypatch, "93.184.216.34", address)
    with pytest.raises(UnsafeURLError, match="non-public"):
        public_addresses("rebound.example")


def test_a_name_that_resolves_to_nothing_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _resolves(monkeypatch)
    with pytest.raises(UnsafeURLError, match="does not resolve"):
        public_addresses("nothing.example")


@pytest.mark.parametrize("url", ["http://[::ffff:127.0.0.1]/", "http://100.64.0.1/"])
def test_addresses_in_disguise_are_refused_by_url_too(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        validate_url(url, resolve_dns=False)
