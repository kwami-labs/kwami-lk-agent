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
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

from livekit.agents import RunContext, function_tool

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

#: Fixed, audited snippets. The model chooses a named action; it never writes one.
#:
#: Each is written to fail quietly on a page that does not match, so a snippet
#: aimed at YouTube does nothing at all on Spotify rather than throwing.
_FIRST_RESULT_JS = """
(() => {
  const sel = [
    'ytd-video-renderer a#video-title',
    'ytmusic-responsive-list-item-renderer a',
    'a[data-testid="track-row"]',
    'ul.sc-list-nav li a',
    'a[href*="/watch"]', 'a[href*="/track/"]', 'a[href*="/album/"]'
  ];
  for (const s of sel) {
    const el = document.querySelector(s);
    if (el) { el.click(); return 'clicked:' + s; }
  }
  return 'no-result';
})()
"""

_MEDIA_ACTION_JS: dict[str, str] = {
    "pause": "(()=>{const m=document.querySelector('video,audio');"
    "if(!m)return'no-media';m.pause();return'paused';})()",
    "resume": "(()=>{const m=document.querySelector('video,audio');"
    "if(!m)return'no-media';m.play();return'playing';})()",
    "stop": "(()=>{const m=document.querySelector('video,audio');"
    "if(!m)return'no-media';m.pause();m.currentTime=0;return'stopped';})()",
    "mute": "(()=>{const m=document.querySelector('video,audio');"
    "if(!m)return'no-media';m.muted=true;return'muted';})()",
    "unmute": "(()=>{const m=document.querySelector('video,audio');"
    "if(!m)return'no-media';m.muted=false;return'unmuted';})()",
    "restart": "(()=>{const m=document.querySelector('video,audio');"
    "if(!m)return'no-media';m.currentTime=0;m.play();return'restarted';})()",
}

#: Volume is the one action that takes a value, so it is built rather than looked up.
_VOLUME_JS = (
    "(()=>{{const m=document.querySelector('video,audio');"
    "if(!m)return'no-media';m.volume={level};m.muted=false;return'volume:'+m.volume;}})()"
)

_NOW_PLAYING_JS = """
(() => {
  const m = document.querySelector('video,audio');
  if (!m) return 'no-media';
  return JSON.stringify({
    title: document.title,
    paused: m.paused,
    muted: m.muted,
    volume: m.volume,
    position: Math.round(m.currentTime || 0),
    duration: Math.round(m.duration || 0),
  });
})()
"""

#: How long to let a results page settle before clicking into it.
RESULTS_SETTLE_SECONDS = 2.5
#: And how long the chosen item needs before it is actually playing.
PLAYBACK_SETTLE_SECONDS = 2.0

MAX_QUERY_CHARS = 200


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

    async def _media_session(self) -> Any:
        """The live browser, or None when there is nothing playing to control."""
        getter = getattr(self, "_get_browser_session", None)
        if not callable(getter):
            return None
        session = await getter()
        return session if getattr(session, "is_active", False) else None

    @function_tool()
    async def play_media(
        self,
        context: RunContext,
        query: str,
        kind: str = "music",
        service: str = "",
    ) -> str:
        """Play a song, album, artist or video. Starts playback, not just a search.

        The user watches it in the browser panel and hears it through their own
        speakers.

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

        opener = getattr(self, "navigate_to", None)
        if opener is None:  # pragma: no cover - the mixin is always combined
            return "I can't open the browser panel in this session."

        url = search_url(service_key, wanted)
        logger.info("play_media: %s on %s", wanted[:60], service_key)
        await opener(context, url)

        session = await self._media_session()
        if session is None:
            return f"I opened {entry['label']} but the browser panel didn't come up."

        # Let the results render, then click the first one. This is the step
        # `navigate_to` alone cannot do: on YouTube the results live in a shadow
        # DOM that read_navigation_page cannot see at all.
        await asyncio.sleep(RESULTS_SETTLE_SECONDS)
        clicked = await session.evaluate_js(_FIRST_RESULT_JS)
        if "no-result" in str(clicked):
            return (
                f"I searched {entry['label']} for '{wanted}' but couldn't find a result to "
                "play. It's on screen if you want to pick one."
            )

        await asyncio.sleep(PLAYBACK_SETTLE_SECONDS)
        # Some services need the play button; calling play() is harmless when
        # it is already playing.
        await session.evaluate_js(_MEDIA_ACTION_JS["resume"])
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

        session = await self._media_session()
        if session is None:
            return "Nothing is playing right now."

        result = str(await session.evaluate_js(snippet))
        if "no-media" in result:
            return "I can't find anything playing on this page."
        return f"Done -- {key}d." if key not in ("stop", "restart") else f"Done -- {key}ped."

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

        session = await self._media_session()
        if session is None:
            return "Nothing is playing right now."

        result = str(await session.evaluate_js(_VOLUME_JS.format(level=clamped)))
        if "no-media" in result:
            return "I can't find anything playing on this page."
        return f"Volume set to {int(clamped * 100)} percent."

    @function_tool()
    async def get_now_playing(self, context: RunContext) -> dict[str, Any]:
        """What is playing right now, and how far through it is."""
        session = await self._media_session()
        if session is None:
            return {"playing": False, "reason": "no browser panel open"}

        raw = str(await session.evaluate_js(_NOW_PLAYING_JS))
        if "no-media" in raw:
            return {"playing": False, "reason": "no media on this page"}

        import json
        import re

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
