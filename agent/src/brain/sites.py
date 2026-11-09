"""The sites the user signed Jarvis in to, one at a time, and whether they still are.

Everything here happens in Jarvis's Chrome with no model looking: the user signs in
on their own screen, and a check only asks whether the site sends them back to sign in.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

from ..browser.local_chrome import window_open
from .browser_tool import CONNECTED, REFUSED, SIGNED_OUT, BrowserTools, browser_available

#: How long a page gets to send the browser on to its sign-in page.
REDIRECT_WAIT_SECONDS = 3.0

_SAID = {
    CONNECTED: "Listo: lo que hayas iniciado queda guardado para Jarvis.",
    SIGNED_OUT: "La web sigue pidiendo entrar. Puedes volver a intentarlo cuando quieras.",
    REFUSED: "Esa web no deja iniciar sesión desde este navegador.",
}


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


async def show_site(
    url: str, profile: str, note: Callable[[str], Awaitable[object]] | None = None
) -> str:
    """A site for the user to sign in to, until they close it. What to tell them."""
    tools = BrowserTools(profile)
    try:
        await tools.call("open", {"url": url})
        await tools.show_page()
        state = await tools.wait_until_closed()
    finally:
        await tools.close()
    if note is not None:
        await note(state)
    return _SAID[state]


async def check_site(url: str, profile: str) -> str | None:
    """Whether the session still holds. None when Chrome is in use, so it is left alone.

    The visit is also what keeps a site that renews its session on use signed in.
    """
    if not browser_available(profile) or await window_open(profile):
        return None
    tools = BrowserTools(profile)
    try:
        await tools.call("open", {"url": url})
        await asyncio.sleep(REDIRECT_WAIT_SECONDS)
        return await tools.page_state()
    finally:
        await tools.close(quit_browser=True)
