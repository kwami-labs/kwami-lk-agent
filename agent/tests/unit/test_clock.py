"""Telling the time in the user's zone rather than the container's.

`get_current_time` used `datetime.now()` with no tzinfo. On both deploy targets
that clock is UTC, and the answer was phrased as if it were the user's local
time -- someone in Madrid was told 09:00 when it was 11:00, confidently and
with nothing to signal it was wrong.

The behaviour these pin, in order of how badly each got it wrong before:

1. A known zone is used.
2. An unknown zone answers UTC *and says so*, rather than passing UTC off as
   local time.
3. A broken zone name degrades to that same labelled UTC instead of raising
   inside a voice turn.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from src.domain.clock import current_time_phrase, resolve_zone

#: A fixed instant so the assertions are about zone maths, not about "now".
#: 2026-06-15 21:30 UTC is deliberately in the evening: it crosses midnight in
#: Tokyo, so a broken conversion changes the *date* and not just the hour.
INSTANT = datetime(2026, 6, 15, 21, 30, tzinfo=UTC)


def test_a_known_zone_shifts_the_clock() -> None:
    phrase = current_time_phrase("Europe/Madrid", now=INSTANT)

    assert "11:30 PM" in phrase
    assert "June 15, 2026" in phrase


def test_a_zone_across_the_dateline_shifts_the_date_too() -> None:
    """The failure this guards is subtle: right hour, wrong day."""
    phrase = current_time_phrase("Asia/Tokyo", now=INSTANT)

    assert "6:30 AM" in phrase
    assert "June 16, 2026" in phrase


def test_an_unknown_zone_answers_utc_and_says_so() -> None:
    """Substituting UTC silently is the same bug as substituting the server's
    clock silently -- the user cannot tell the answer needs converting."""
    phrase = current_time_phrase(None, "", now=INSTANT)

    assert "9:30 PM" in phrase
    assert "UTC" in phrase
    assert "don't know your timezone" in phrase


def test_a_bogus_zone_degrades_rather_than_raising() -> None:
    """A typo, a Windows-style name, or a slim image with no tzdata. None of
    those should cost the user the turn."""
    phrase = current_time_phrase("Pacific Standard Time", now=INSTANT)

    assert "UTC" in phrase


def test_the_first_usable_candidate_wins() -> None:
    """Config first, participant attribute second -- the precedence the tool
    relies on."""
    phrase = current_time_phrase("Europe/Madrid", "Asia/Tokyo", now=INSTANT)

    assert "11:30 PM" in phrase


def test_an_unusable_first_candidate_falls_through_to_the_second() -> None:
    """A stale or malformed config value must not mask a good participant one."""
    phrase = current_time_phrase("Not/AZone", "Asia/Tokyo", now=INSTANT)

    assert "6:30 AM" in phrase


def test_blank_candidates_are_skipped_not_treated_as_zones() -> None:
    phrase = current_time_phrase("", "   ", None, "Europe/Madrid", now=INSTANT)

    assert "11:30 PM" in phrase


def test_the_hour_has_no_leading_zero() -> None:
    """ "09:05 PM" reads as a clock display; the agent is speaking."""
    morning = datetime(2026, 6, 15, 8, 5, tzinfo=UTC)

    phrase = current_time_phrase("UTC", now=morning)

    assert phrase.startswith("8:05 AM")


def test_resolve_zone_reports_the_name_it_used() -> None:
    zone, label = resolve_zone("Europe/Madrid")

    assert zone == ZoneInfo("Europe/Madrid")
    assert label == "Europe/Madrid"


def test_resolve_zone_falls_back_to_utc_with_a_utc_label() -> None:
    zone, label = resolve_zone(None)

    assert zone is UTC
    assert label == "UTC"


def test_a_real_zone_is_not_announced() -> None:
    """Nobody asking the time wants "in Europe/Madrid" appended; the label is
    only spoken when it is a fallback the user needs to know about."""
    phrase = current_time_phrase("Europe/Madrid", now=INSTANT)

    assert "Europe/Madrid" not in phrase
    assert "UTC" not in phrase


def test_a_fixed_offset_zone_name_still_resolves() -> None:
    """ZoneInfo accepts "UTC"; a plain offset object is not a candidate here,
    but the UTC spelling has to keep working because it is the fallback label."""
    zone, label = resolve_zone("UTC")

    assert label == "UTC"
    assert datetime(2026, 1, 1, tzinfo=zone).utcoffset() == UTC.utcoffset(None)
