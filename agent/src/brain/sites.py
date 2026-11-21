"""The sites the user signed Jarvis in to, one at a time, and whether they still are.

Everything here happens in Jarvis's Chrome with no model looking: the user signs in
on their own screen, in a clean window no program controls (Google refuses a driven
one), and a check only asks whether the site sends them back to sign in.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

from ..browser.local_chrome import open_clean_window, wait_clean_window_closed, window_open
from ..browser.safety import validate_url_async
from .browser_tool import (
    CONNECTED,
    REFUSED,
    SIGN_IN_WAIT_SECONDS,
    SIGNED_OUT,
    BrowserTools,
    browser_available,
)

#: How long a page gets to send the browser on to its sign-in page.
REDIRECT_WAIT_SECONDS = 3.0

_SAID = {
    CONNECTED: "Listo: lo que hayas iniciado queda guardado para Jarvis.",
    SIGNED_OUT: "La web sigue pidiendo entrar. Puedes volver a intentarlo cuando quieras.",
    REFUSED: "Esa web no deja iniciar sesión desde este navegador.",
}
STILL_OPEN = (
    "La ventana sigue abierta: inicia sesión en ella. Se cierra sola cuando entras, y Jarvis "
    "lo guarda."
)
UNCHECKED = "Ventana cerrada. Chrome está en uso: lo compruebo en el próximo repaso."


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


async def show_site(
    url: str, profile: str, note: Callable[[str], Awaitable[object]] | None = None
) -> str:
    """A site for the user to sign in to, in a clean window until they are in or close it.

    Then the site is opened again the usual way, to see whether the session took.
    What to tell them.
    """
    url = await validate_url_async(url)
    await open_clean_window(profile, url)
    if not await wait_clean_window_closed(profile, SIGN_IN_WAIT_SECONDS):
        return STILL_OPEN
    state = await check_site(url, profile)
    if state is None:
        return UNCHECKED
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
