"""URL and output guards for the cloud browser.

The browser profile carries the user's cookies and logins, so a URL the model
was talked into opening is a request made from a trusted position. These are
the pure checks behind `navigate_to`.
"""

from __future__ import annotations

import pytest

from src.browser.safety import (
    MAX_TOOL_OUTPUT_CHARS,
    UnsafeURLError,
    javascript_execution_enabled,
    normalize_url,
    truncate_for_llm,
    validate_url,
)

# Addresses that must never be reachable from a tool the LLM can drive.
BLOCKED = [
    pytest.param("http://169.254.169.254/latest/meta-data/", id="cloud-metadata"),
    pytest.param("http://127.0.0.1:8080/admin", id="loopback-v4"),
    pytest.param("http://[::1]:8080/admin", id="loopback-v6"),
    pytest.param("http://10.0.0.5/internal", id="rfc1918-10"),
    pytest.param("http://192.168.1.1/router", id="rfc1918-192"),
    pytest.param("http://172.16.0.1/internal", id="rfc1918-172"),
    pytest.param("http://0.0.0.0/", id="unspecified"),
]


@pytest.mark.parametrize("url", BLOCKED)
def test_non_public_addresses_are_refused(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        validate_url(url, resolve_dns=False)


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "gopher://example.com/", "javascript:alert(1)", "data:text/html,hi"],
)
def test_only_http_schemes_are_allowed(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        validate_url(url, resolve_dns=False)


def test_public_urls_pass_through() -> None:
    assert validate_url("https://example.com/page", resolve_dns=False) == "https://example.com/page"


def test_a_bare_hostname_is_assumed_https() -> None:
    assert normalize_url("example.com").startswith("https://")
    assert validate_url("example.com", resolve_dns=False) == "https://example.com"


def test_empty_url_is_refused() -> None:
    with pytest.raises(UnsafeURLError):
        validate_url("   ", resolve_dns=False)


def test_output_is_capped_for_the_llm() -> None:
    """CDP can return whatever the page produced, against a 10 MB WS limit."""
    capped = truncate_for_llm("x" * (MAX_TOOL_OUTPUT_CHARS * 3))
    assert len(capped) < MAX_TOOL_OUTPUT_CHARS + 200
    assert "truncated" in capped


def test_short_output_is_untouched() -> None:
    assert truncate_for_llm("hello") == "hello"


def test_javascript_execution_is_off_unless_opted_in(env_setting) -> None:
    env_setting("KWAMI_ALLOW_BROWSER_JS", None)
    assert javascript_execution_enabled() is False

    env_setting("KWAMI_ALLOW_BROWSER_JS", "true")
    assert javascript_execution_enabled() is True

    env_setting("KWAMI_ALLOW_BROWSER_JS", "no")
    assert javascript_execution_enabled() is False


@pytest.mark.parametrize(
    "url,expected",
    [
        ("example.com:8080/path", "https://example.com:8080/path"),
        ("example.com:443", "https://example.com:443"),
    ],
)
def test_bare_host_and_port_is_not_mistaken_for_a_scheme(url: str, expected: str) -> None:
    """`example.com:8080` looks like a scheme to a naive regex.

    Rejecting dangerous schemes must not cost us ordinary host:port URLs,
    which the model will produce whenever a user says "open example.com:8080".
    """
    assert validate_url(url, resolve_dns=False) == expected


def test_blocked_port_like_host_is_still_checked() -> None:
    """The host:port path still goes through the address blocklist."""
    with pytest.raises(UnsafeURLError):
        validate_url("127.0.0.1:8080", resolve_dns=False)
