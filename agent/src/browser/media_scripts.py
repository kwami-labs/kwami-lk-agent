"""The media tools' page actions: fixed, audited snippets, picked by name.

The model chooses a named action; it never writes one. Each is written to fail quietly
on a page that does not match, so a snippet aimed at YouTube does nothing at all on
Spotify rather than throwing. Jarvis's own Chrome runs them over its debugging port, and
Claude in Chrome (`claude_chrome`) through its page tool; Kwami's extension carries its
own copies (`chrome-extension/page-functions.js`).
"""

from __future__ import annotations

FIRST_RESULT_JS = """
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

MEDIA_ACTION_JS: dict[str, str] = {
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
VOLUME_JS = (
    "(()=>{{const m=document.querySelector('video,audio');"
    "if(!m)return'no-media';m.volume={level};m.muted=false;return'volume:'+m.volume;}})()"
)

NOW_PLAYING_JS = """
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


def media_script(action: str, level: float = 0.0) -> str | None:
    """The snippet for a named action, or None for a name that is not one."""
    if action == "first_result":
        return FIRST_RESULT_JS
    if action == "now_playing":
        return NOW_PLAYING_JS
    if action == "volume":
        return VOLUME_JS.format(level=max(0.0, min(1.0, float(level))))
    return MEDIA_ACTION_JS.get(action)
