"""Playing video and music on request, rather than describing how to.

Asked to put a song on, the agent's only route was `navigate_to` plus a
narrated sequence of `read_navigation_page` / `click_in_navigation` calls --
which works, slowly, out loud, and fails outright on YouTube, whose results are
inside a shadow DOM that `read_navigation_page` cannot see. The advice in the
prompt was "go straight to the search URL", which gets a *results page* on
screen and stops one click short of anything playing.

These tools close that last step. Two design points carry the safety argument:

* **No model-authored JavaScript.** Arbitrary in-page JS is off by default
  (`KWAMI_ALLOW_BROWSER_JS`), because the page is the user's logged-in browser
  and the page's own content is untrusted. The snippets here are *constants in
  this file*, chosen and reviewed once; the model picks which named action to
  run, never what the script says. That is a categorically different risk from
  handing the model an eval, and it is why these do not go through
  `run_js_in_navigation`.
* **Deterministic URLs.** The query is URL-encoded into a known service's
  search path. The model never supplies a URL, so it cannot be talked into
  opening one by a page it just read.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

from livekit.agents import RunContext, function_tool

from ..browser.media_scripts import FIRST_RESULT_JS, MEDIA_ACTION_JS, NOW_PLAYING_JS, VOLUME_JS
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..domain import KwamiConfig

logger = get_logger("media")

#: Services the agent can start playback on, and how to search them.
#:
#: None of them play straight from a search URL, so every one needs its first
#: result clicked -- that is what `_FIRST_RESULT_JS` is for.
SERVICES: dict[str, dict[str, Any]] = {
    "youtube": {
        "search": "https://www.youtube.com/results?search_query={q}",
        "label": "YouTube",
        "kinds": ("video", "music"),
    },
    "youtube_music": {
        "search": "https://music.youtube.com/search?q={q}",
        "label": "YouTube Music",
        "kinds": ("music",),
    },
    "spotify": {
        "search": "https://open.spotify.com/search/{q}",
        "label": "Spotify",
        "kinds": ("music",),
    },
    "soundcloud": {
        "search": "https://soundcloud.com/search/sounds?q={q}",
        "label": "SoundCloud",
        "kinds": ("music",),
    },
}

DEFAULT_SERVICE_FOR_KIND = {"video": "youtube", "music": "youtube_music"}

#: Fixed, audited snippets (`browser.media_scripts`). The model chooses a named action;
#: it never writes one.
_FIRST_RESULT_JS = FIRST_RESULT_JS
_MEDIA_ACTION_JS = MEDIA_ACTION_JS
_VOLUME_JS = VOLUME_JS
_NOW_PLAYING_JS = NOW_PLAYING_JS

#: How long to let a results page settle before clicking into it.
RESULTS_SETTLE_SECONDS = 2.5
#: And how long the chosen item needs before it is actually playing.
PLAYBACK_SETTLE_SECONDS = 2.0

MAX_QUERY_CHARS = 200

#: Jarvis's own screen in the app (2026-11-17): a YouTube result plays in its video
#: window, not in a browser that opens beside it. The app's tool that says it has one.
OWN_SCREEN_TOOL = "show_on_screen"
#: Services whose first result is a YouTube video, which that window plays.
SCREEN_SERVICES = frozenset({"youtube", "youtube_music"})
YOUTUBE_RESULTS = "https://www.youtube.com/results"
YOUTUBE_WATCH = "https://www.youtube.com/watch?v={id}"
#: The first video of a results page, and the title that follows it.
_FIRST_VIDEO = re.compile(r'"videoRenderer":\{"videoId":"([\w-]{11})"')
_VIDEO_TITLE = re.compile(r'"title":\{"runs":\[\{"text":"(.*?)"\}')
#: YouTube's consent page stands in front of results for a request with no cookies in
#: the EU; this is the "reject all" answer, which is enough to see the results.
_YOUTUBE_CONSENT = {"SOCS": "CAI"}
YOUTUBE_TIMEOUT_SECONDS = 8.0


async def first_youtube_video(query: str) -> tuple[str, str] | None:
    """The address and title of YouTube's first video for `query`, or None."""
    from ..adapters.http import shared_client

    try:
        response = await shared_client().get(
            YOUTUBE_RESULTS,
            params={"search_query": query.strip()[:MAX_QUERY_CHARS]},
            headers={"Accept-Language": "es-ES,es;q=0.9,en;q=0.8"},
            cookies=_YOUTUBE_CONSENT,
            follow_redirects=True,
            timeout=YOUTUBE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except Exception as e:
        logger.warning("Could not search YouTube: %s", e)
        return None
    found = _FIRST_VIDEO.search(response.text)
    if found is None:
        return None
    title = ""
    named = _VIDEO_TITLE.search(response.text, found.end(), found.end() + 4000)
    if named is not None:
        try:
            title = json.loads(f'"{named.group(1)}"')
        except ValueError:
            title = named.group(1)
    return YOUTUBE_WATCH.format(id=found.group(1)), title[:120]


def resolve_service(kind: str, service: str | None) -> tuple[str, dict[str, Any]] | None:
    """Pick the service to play on, or None when the request cannot be honoured.

    Returns None rather than silently substituting: asked for a song on
    Spotify, opening YouTube instead is a different answer to the one given.
    """
    normalized_kind = (kind or "").strip().lower()
    if normalized_kind not in ("video", "music"):
        normalized_kind = "music"

    if service:
        key = service.strip().lower().replace(" ", "_").replace("-", "_")
        aliases = {"yt": "youtube", "ytmusic": "youtube_music", "youtubemusic": "youtube_music"}
        key = aliases.get(key, key)
        entry = SERVICES.get(key)
        if entry is None:
            return None
        return key, entry

    key = DEFAULT_SERVICE_FOR_KIND.get(normalized_kind, "youtube")
    return key, SERVICES[key]


def search_url(service_key: str, query: str) -> str:
    """The service's search URL for `query`, URL-encoded.

    Built here rather than taken from the model: a URL the model composed could
    be steered by whatever page it last read.
    """
    template = SERVICES[service_key]["search"]
    return template.format(q=quote_plus(query.strip()[:MAX_QUERY_CHARS]))


class MediaToolsMixin:
    """Playback control for the live browser panel.

    `kwami_config` is supplied by `KwamiAgent`; annotated so a rename there
    surfaces as a type error rather than as playback that quietly stops working.
    """

    kwami_config: KwamiConfig
    #: Whether what plays now is in the app's video window rather than in a browser.
    _media_on_screen: bool = False

    async def _media_session(self) -> Any:
        """The live browser, or None when there is nothing playing to control."""
        getter = getattr(self, "_get_browser_session", None)
        if not callable(getter):
            return None
        session = await getter()
        return session if getattr(session, "is_active", False) else None

    def _has_own_screen(self) -> bool:
        """Whether the app gives Jarvis a screen of its own, with a video window."""
        names = getattr(self, "_registered_client_tool_names", None)
        if not callable(names):
            return False
        try:
            return OWN_SCREEN_TOOL in names()
        except Exception:
            return False

    async def _publish_to_app(self, context: Any, message: dict[str, Any]) -> bool:
        publisher = getattr(self, "_publisher", None)
        if not callable(publisher):
            return False
        try:
            return bool(await publisher(context).publish(message))
        except Exception:
            logger.warning("Could not send %s to the app", message.get("type"), exc_info=True)
            return False

    async def _play_on_screen(self, context: Any, wanted: str, label: str) -> str | None:
        """YouTube's first video for `wanted`, in the app's video window. None if it cannot."""
        found = await first_youtube_video(wanted)
        if found is None:
            return None
        url, title = found
        sent = await self._publish_to_app(
            context,
            {"type": "hud_open", "kind": "video", "url": url, "title": title or wanted},
        )
        if not sent:
            return None
        self._media_on_screen = True
        logger.info("play_media: %s on Jarvis's screen (%s)", wanted[:60], url)
        shown = f" ({title})" if title else ""
        return f"Playing '{wanted}'{shown} from {label} in the video window on Jarvis's screen."

    @staticmethod
    async def _run_named(session: Any, name: str, script: str, level: float = 0.0) -> str:
        """A fixed media action: by name where the browser has its own (the user's Chrome)."""
        runner = getattr(session, "run_named", None)
        if callable(runner):
            return str(await runner(name, level))
        return str(await session.evaluate_js(script))

    @function_tool()
    async def play_media(
        self,
        context: RunContext,
        query: str,
        kind: str = "music",
        service: str = "",
    ) -> str:
        """Play a song, album, artist or video. Starts playback, not just a search.

        With Jarvis's own screen in the app, a YouTube result plays in its video window;
        otherwise in the browser panel. The user hears it through their own speakers.

        Args:
            query: What to play -- a song, artist, album or video title.
            kind: 'music' or 'video'.
            service: Optional. 'youtube', 'youtube_music', 'spotify' or
                     'soundcloud'. Left empty, picks the usual one for `kind`.
        """
        wanted = (query or "").strip()
        if not wanted:
            return "What would you like me to play?"

        resolved = resolve_service(kind, service or None)
        if resolved is None:
            return f"I can't play on '{service}'. I can use: {', '.join(sorted(SERVICES))}."
        service_key, entry = resolved

        if service_key in SCREEN_SERVICES and self._has_own_screen():
            played = await self._play_on_screen(context, wanted, entry["label"])
            if played is not None:
                return played

        opener = getattr(self, "navigate_to", None)
        if opener is None:
            return "I can't open the browser panel in this session."

        url = search_url(service_key, wanted)
        logger.info("play_media: %s on %s", wanted[:60], service_key)
        await opener(context, url)

        session = await self._media_session()
        if session is None:
            return f"I opened {entry['label']} but the browser panel didn't come up."
        self._media_on_screen = False

        # Let the results render, then click the first one. This is the step
        # `navigate_to` alone cannot do: on YouTube the results live in a shadow
        # DOM that read_navigation_page cannot see at all.
        await asyncio.sleep(RESULTS_SETTLE_SECONDS)
        clicked = await self._run_named(session, "first_result", _FIRST_RESULT_JS)
        if "no-result" in clicked:
            return (
                f"I searched {entry['label']} for '{wanted}' but couldn't find a result to "
                "play. It's on screen if you want to pick one."
            )

        await asyncio.sleep(PLAYBACK_SETTLE_SECONDS)
        # Some services need the play button; calling play() is harmless when
        # it is already playing.
        await self._run_named(session, "resume", _MEDIA_ACTION_JS["resume"])
        return f"Playing '{wanted}' on {entry['label']}."

    @function_tool()
    async def control_playback(self, context: RunContext, action: str) -> str:
        """Control whatever is currently playing.

        Args:
            action: 'pause', 'resume', 'stop', 'mute', 'unmute' or 'restart'.
        """
        key = (action or "").strip().lower()
        aliases = {
            "play": "resume",
            "continue": "resume",
            "unpause": "resume",
            "silence": "mute",
            "quiet": "mute",
            "end": "stop",
            "start over": "restart",
            "replay": "restart",
        }
        key = aliases.get(key, key)
        snippet = _MEDIA_ACTION_JS.get(key)
        if snippet is None:
            return (
                f"I don't know how to '{action}'. I can pause, resume, stop, mute, "
                "unmute or restart."
            )

        done = f"Done -- {key}d." if key not in ("stop", "restart") else f"Done -- {key}ped."
        if self._media_on_screen and await self._publish_to_app(
            context, {"type": "hud_media", "action": key}
        ):
            return done

        session = await self._media_session()
        if session is None:
            return "Nothing is playing right now."

        result = await self._run_named(session, key, snippet)
        if "no-media" in result:
            return "I can't find anything playing on this page."
        return done

    @function_tool()
    async def set_playback_volume(self, context: RunContext, level: float) -> str:
        """Set the volume of whatever is playing.

        Args:
            level: 0.0 (silent) to 1.0 (full). Values outside are clamped.
        """
        try:
            clamped = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            return "Give me a volume between 0 and 1."

        if self._media_on_screen:
            sent = await self._publish_to_app(
                context, {"type": "hud_media", "action": "volume", "level": clamped}
            )
            if sent:
                return f"Volume set to {int(clamped * 100)} percent."

        session = await self._media_session()
        if session is None:
            return "Nothing is playing right now."

        result = await self._run_named(
            session, "volume", _VOLUME_JS.format(level=clamped), level=clamped
        )
        if "no-media" in result:
            return "I can't find anything playing on this page."
        return f"Volume set to {int(clamped * 100)} percent."

    @function_tool()
    async def get_now_playing(self, context: RunContext) -> dict[str, Any]:
        """What is playing right now, and how far through it is."""
        if self._media_on_screen:
            return {
                "where": "the video window on Jarvis's screen",
                "reason": "its position and title are shown there, to the user",
            }
        session = await self._media_session()
        if session is None:
            return {"playing": False, "reason": "no browser panel open"}

        raw = await self._run_named(session, "now_playing", _NOW_PLAYING_JS)
        if "no-media" in raw:
            return {"playing": False, "reason": "no media on this page"}

        # `evaluate_js` returns a human-readable wrapper around the value, so
        # the JSON has to be recovered from it rather than parsed directly.
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return {"playing": False, "reason": "could not read the player"}
        try:
            info = json.loads(match.group(0))
        except ValueError:
            return {"playing": False, "reason": "could not read the player"}

        info["playing"] = not info.get("paused", True)
        return info
