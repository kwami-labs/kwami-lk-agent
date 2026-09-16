"""Safety checks for agent-driven browsing.

The cloud browser runs with a persistent, per-user profile that deliberately
keeps cookies and logins. That makes two things dangerous that would otherwise
be routine:

* **SSRF.** A URL the model chose can point at loopback, link-local metadata
  (169.254.169.254) or private ranges, reaching services that trust the network.
* **Indirect prompt injection.** Page text read by ``read_navigation_page`` goes
  straight into the LLM context, and a hostile page can instruct the model to
  run JavaScript that exfiltrates ``document.cookie`` from that logged-in
  profile.

So URLs are validated against a blocklist, and JavaScript execution is off
unless an operator explicitly turns it on.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlparse

from ..utils.logging import get_logger

logger = get_logger("browser.safety")

ALLOWED_SCHEMES = ("http", "https")

# Hostnames that must never be resolved, regardless of what they resolve to.
BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
    }
)

# Cap on anything a browser tool feeds back into the LLM context. Without it,
# `run_js_in_navigation` could return whatever CDP produced against a 10 MB
# WebSocket limit -- `document.body.innerHTML` on a large page is enough to
# blow up the next prompt.
MAX_TOOL_OUTPUT_CHARS = 4000

JS_EXECUTION_ENV_VAR = "KWAMI_ALLOW_BROWSER_JS"


class UnsafeURLError(ValueError):
    """Raised when a URL must not be opened by the agent."""


def javascript_execution_enabled() -> bool:
    """Whether arbitrary JS in the user's logged-in browser is permitted.

    Defaults to off. Turn it on with KWAMI_ALLOW_BROWSER_JS=1 only where the
    prompt-injection risk described in this module's docstring is acceptable.
    """
    return os.environ.get(JS_EXECUTION_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolved_addresses(hostname: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, UnicodeError):
        return []
    # sockaddr[0] is the address; typeshed widens it to str | int for AF_*
    # families we never ask for.
    return [str(info[4][0]) for info in infos]


# A scheme is "<letter><letter|digit|+|-|.>*:" per RFC 3986. Testing for "://"
# instead would let `javascript:alert(1)` and `data:text/html,...` through: they
# carry a scheme but no authority, so they were being rewritten into
# `https://javascript:alert(1)` and then accepted as an ordinary hostname.
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


def normalize_url(url: str) -> str:
    """Add a scheme to a bare hostname so validation has something to parse."""
    url = (url or "").strip()
    if not url:
        raise UnsafeURLError("no URL provided")
    match = _SCHEME_RE.match(url)
    if match:
        # "example.com:8080/path" is a host and port, not a scheme -- the
        # regex cannot tell them apart, but a numeric first segment can.
        leading = url[match.end() :].split("/", 1)[0]
        if not leading.isdigit():
            return url
    return f"https://{url}"


def validate_url(url: str, *, resolve_dns: bool = True) -> str:
    """Return a safe, normalized URL, or raise UnsafeURLError.

    Args:
        url: The candidate URL, with or without a scheme.
        resolve_dns: Also reject hostnames that resolve to a blocked address.
            Set False where DNS lookups are undesirable (tests, offline use);
            literal IPs are still rejected.
    """
    candidate = normalize_url(url)
    parsed = urlparse(candidate)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme '{parsed.scheme}' is not allowed")

    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise UnsafeURLError("URL has no hostname")

    if hostname in BLOCKED_HOSTNAMES or hostname.endswith(".localhost"):
        raise UnsafeURLError(f"'{hostname}' is not a public address")

    # A literal IP needs no DNS to be dangerous.
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None and _is_blocked_ip(literal):
        raise UnsafeURLError(f"'{hostname}' is not a public address")

    if resolve_dns and literal is None:
        for address in _resolved_addresses(hostname):
            try:
                resolved = ipaddress.ip_address(address)
            except ValueError:
                continue
            if _is_blocked_ip(resolved):
                raise UnsafeURLError(f"'{hostname}' resolves to the non-public address {address}")

    return candidate


async def validate_url_async(url: str, *, resolve_dns: bool = True) -> str:
    """`validate_url`, with the DNS lookup kept off the event loop.

    `socket.getaddrinfo` is blocking and can take seconds against a slow
    resolver. On a voice agent that is audible dead air, so callers in async
    paths must use this rather than the synchronous function.
    """
    import asyncio

    return await asyncio.to_thread(validate_url, url, resolve_dns=resolve_dns)


def truncate_for_llm(text: str | None, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Clamp tool output before it enters the model's context.

    Accepts None because callers pass it straight from browser helpers whose
    return type is not guaranteed at runtime.
    """
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated, {len(text) - limit} more characters]"
