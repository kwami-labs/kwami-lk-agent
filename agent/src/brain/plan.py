"""How the brain pays for Claude, and what is left of the plan on the user's PC.

The user's rule (2026-11-17): everything goes through the PC's Claude plan until it
runs out, and only then to the API key; and a panel in the models section that says
which to use and which first, and that the brain then obeys. Four ways
(`config.BILLING_MODES`):

==============  ==========================================================  ============
billing         Claude                                                      other APIs
==============  ==========================================================  ============
account_first   the plan; the API key when the plan is out, the PC is off   after the
                or not signed in                                            plan
api_first       the API key; the plan without a key, or when the key fails  as usual
account_only    the plan only; when it is out, it says when it comes back   never
api_only        the API key only                                            as usual
==============  ==========================================================  ============

"Only" is the user's word for the whole brain, so a model's own `access` gives way to
it; in the two "first" ways a model set to account or API keeps its own.

What is left of the plan comes from Claude Code itself: its rate-limit events say
how much of a window is used and when it resets, and its error says when it ran out.
The worker keeps the last one (`brain_runners.claude_plan`, migration 032), the agent
reads it before sending a thought there, and the panel shows it.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: How long a plan that ran out without saying when it comes back is left alone.
PLAN_OUT_GUESS = timedelta(minutes=30)

#: What Claude Code says when the plan has run out, in its old and new words. Read only
#: on a run that failed: a task's own text can mention a limit.
PLAN_OUT_WORDS = re.compile(
    r"usage limit|limit reached|hit your (?:usage )?limit|out of (?:extra )?usage"
    r"|rate_limit_error|\brate.?limited\b|plan de claude (?:está|se ha) agotado",
    re.IGNORECASE,
)
#: "Claude AI usage limit reached|1731600000": the old CLI said the reset as a number.
_EPOCH = re.compile(r"limit reached\|(\d{9,11})")
#: "resets 3pm", "resets 3:30pm (Europe/Madrid)", "resets Mon 9am".
_RESETS = re.compile(
    r"resets?\s+(?:at\s+)?(?:(mon|tue|wed|thu|fri|sat|sun)\w*\s+)?"
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s*\(([^)]+)\))?",
    re.IGNORECASE,
)
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

#: The windows Claude Code reports, in the user's words.
WINDOW_WORDS = {
    "five_hour": "de 5 horas",
    "seven_day": "semanal",
    "seven_day_opus": "semanal de Opus",
    "seven_day_sonnet": "semanal de Sonnet",
    "overage": "de uso extra",
}


def _iso(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).isoformat() if moment is not None else None


def _when(value: Any) -> datetime | None:
    """A moment from what was stored: ISO text, or seconds since the epoch."""
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value, UTC)
    if isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value)
        except ValueError:
            return None
        return moment if moment.tzinfo else moment.replace(tzinfo=UTC)
    return None


def from_event(info: Any, now: datetime | None = None) -> dict[str, Any]:
    """What a rate-limit event of the Agent SDK said (`RateLimitInfo`), to store."""
    utilization = getattr(info, "utilization", None)
    return {
        "status": str(getattr(info, "status", "") or ""),
        "type": str(getattr(info, "rate_limit_type", "") or ""),
        "utilization": float(utilization) if isinstance(utilization, (int, float)) else None,
        "resetsAt": _iso(_when(getattr(info, "resets_at", None))),
        "seenAt": _iso(now or datetime.now(UTC)),
    }


def resets_from_text(text: str, now: datetime | None = None) -> datetime | None:
    """When the plan comes back, from Claude Code's words. None when they do not say."""
    now = now or datetime.now(UTC)
    if found := _EPOCH.search(text):
        return datetime.fromtimestamp(int(found.group(1)), UTC)
    found = _RESETS.search(text)
    if found is None:
        return None
    day, hour_text, minute_text, half, zone_name = found.groups()
    hour = int(hour_text)
    if half:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if half.lower() == "pm" else 0)
    if hour > 23:
        return None
    zone: tzinfo = UTC
    if zone_name:
        try:
            zone = ZoneInfo(zone_name.strip())
        except (ZoneInfoNotFoundError, ValueError):
            zone = now.astimezone().tzinfo or UTC
    else:
        zone = now.astimezone().tzinfo or UTC
    local_now = now.astimezone(zone)
    moment = local_now.replace(hour=hour, minute=int(minute_text or 0), second=0, microsecond=0)
    if day:
        ahead = (_WEEKDAYS.index(day.lower()[:3]) - local_now.weekday()) % 7
        moment += timedelta(days=ahead)
    if moment <= local_now:
        moment += timedelta(days=7 if day else 1)
    return moment.astimezone(UTC)


def from_failure(text: str, now: datetime | None = None) -> dict[str, Any] | None:
    """The plan as a failed run on it left it: out, and until when if it said."""
    if not PLAN_OUT_WORDS.search(text):
        return None
    now = now or datetime.now(UTC)
    return {
        "status": "rejected",
        "type": "",
        "utilization": 1.0,
        "resetsAt": _iso(resets_from_text(text, now)),
        "seenAt": _iso(now),
    }


def out_until(plan: dict[str, Any] | None, now: datetime | None = None) -> datetime | None:
    """Until when the plan is out. None while it has room, or once it reset."""
    if not isinstance(plan, dict) or plan.get("status") != "rejected":
        return None
    now = now or datetime.now(UTC)
    until = _when(plan.get("resetsAt"))
    if until is None:
        seen = _when(plan.get("seenAt"))
        until = seen + PLAN_OUT_GUESS if seen is not None else None
    return until if until is not None and until > now else None


def clock(moment: datetime, zone: tzinfo | None = None) -> str:
    """A reset time as the user says it: 15:00 today, or the day and time later on."""
    local = moment.astimezone(zone) if zone is not None else moment.astimezone()
    today = datetime.now(local.tzinfo).date()
    if local.date() == today:
        return f"las {local:%H:%M}"
    if local.date() == today + timedelta(days=1):
        return f"mañana a las {local:%H:%M}"
    days = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
    return f"el {days[local.weekday()]} a las {local:%H:%M}"


def out_words(until: datetime | None, zone: tzinfo | None = None) -> str:
    """The plan is out, in a sentence."""
    if until is None:
        return "Tu plan de Claude se ha agotado."
    return f"Tu plan de Claude está agotado hasta {clock(until, zone)}."
