"""Telling the user the time, in their timezone rather than the container's.

`get_current_time` called `datetime.now()` with no tzinfo. On both deploy
targets -- a LiveKit Cloud worker and a Cloudflare Container -- that clock is
UTC, and the answer was phrased as though it were the user's local time. Someone
in Madrid asking "what time is it?" was told 09:00 when it was 11:00, stated
with complete confidence and no way to tell it was wrong.

Three things follow, and they are why this is a module rather than one line:

* A zone can come from two places and neither is guaranteed. The config message
  is the good source; a LiveKit participant attribute is the fallback that makes
  telephony and older app builds work with no protocol change.
* An unknown zone must answer in UTC *and say so*. Silently substituting the
  server's clock is the original bug; silently substituting UTC without a label
  is the same bug with a different number.
* A zone that the platform cannot resolve -- a typo, a Windows-style name, a
  tzdata package missing from a slim image -- must degrade to that same labelled
  UTC rather than raise inside a voice turn.
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: How the time is spoken. `%-I` (no leading zero) is a GNU extension, so the
#: portable `%I` is used and the zero stripped afterwards -- "09:05 PM" reads
#: as a clock display, "9:05 PM" reads as speech.
TIME_FORMAT = "%I:%M %p on %A, %B %d, %Y"


def resolve_zone(*candidates: str | None) -> tuple[tzinfo, str]:
    """The first usable zone among `candidates`, else UTC.

    Returns the zone and the name to say out loud. The name is returned
    separately because "UTC" has to be spoken when it is a fallback, and a real
    zone usually should not be -- nobody wants "it is 11:05 AM in
    Europe/Madrid" when they asked what time it is.
    """
    for candidate in candidates:
        name = (candidate or "").strip()
        if not name:
            continue
        try:
            return ZoneInfo(name), name
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            # A typo, a Windows zone name, or a slim image with no tzdata. Try
            # the next candidate rather than failing the turn.
            continue
    return UTC, "UTC"


def current_time_phrase(*candidates: str | None, now: datetime | None = None) -> str:
    """The current time in the user's zone, phrased for speech.

    Falls back to UTC and says so, because an unlabelled wrong time is worse
    than a labelled right one the user has to convert.

    Args:
        candidates: Zone names in preference order; the first usable one wins.
        now: Injectable clock. Tests pass a fixed instant; nothing else should.
    """
    zone, label = resolve_zone(*candidates)
    moment = (now or datetime.now(UTC)).astimezone(zone)
    phrase = moment.strftime(TIME_FORMAT).lstrip("0")
    if label == "UTC":
        return (
            f"{phrase} UTC. I don't know your timezone, so that's UTC rather than your local time."
        )
    return phrase
