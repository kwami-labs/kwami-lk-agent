"""Who is on a phone call, as the API vouched for it.

The API bridges a Twilio call into LiveKit with SIP headers that the inbound trunk
turns into the caller's attributes: the kwami, its owner's user id, whether the
caller is that owner (the number they confirmed for the kwami's notices), a
timestamp and an HMAC over them. Anybody who knows the project's SIP address could
send the same headers, so only a signed, fresh set is believed: it names the kwami,
and an owner's call is the owner's -- who may then ask for work on the PC or in the
cloud, as from the app. A stranger stays a guest on the line.

The key is derived from the Supabase secret key, which the API holds too
(`src/services/sip_calls.py` in kwami-lk-api builds the same signature).
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Mapping
from typing import Any

from livekit.rtc import ParticipantKind

from ..adapters.supabase import as_user_id
from ..settings import get_settings

SIGNATURE_DOMAIN = "sip-call"
#: Attribute names the inbound trunk maps the API's SIP headers to.
KWAMI_ID = "kwami_id"
USER_ID = "kwami.user_id"
OWNER = "kwami.owner"
TIMESTAMP = "kwami.ts"
SIGNATURE = "kwami.sig"
#: A call is answered within seconds of the API signing it; a minute's ring and
#: some clock drift fit well inside this.
FRESH_SECONDS = 600


def is_sip(participant: Any) -> bool:
    return getattr(participant, "kind", None) == ParticipantKind.PARTICIPANT_KIND_SIP


def _expected(secret: str, kwami_id: str, user_id: str, owner: str, timestamp: str) -> str:
    key = f"{SIGNATURE_DOMAIN}:{secret}".encode()
    message = f"{kwami_id}|{user_id}|{owner}|{timestamp}".encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def signed(attributes: Mapping[str, str], now: float | None = None) -> bool:
    """Whether the call's attributes carry a valid, fresh signature from the API."""
    secret = get_settings().supabase_secret_key
    signature = attributes.get(SIGNATURE) or ""
    timestamp = attributes.get(TIMESTAMP) or ""
    if not secret or not signature or not timestamp.isdigit():
        return False
    moment = time.time() if now is None else now
    if abs(moment - int(timestamp)) > FRESH_SECONDS:
        return False
    expected = _expected(
        secret,
        attributes.get(KWAMI_ID) or "",
        attributes.get(USER_ID) or "",
        attributes.get(OWNER) or "",
        timestamp,
    )
    return hmac.compare_digest(expected, signature)


def owner_on_call(participant: Any, now: float | None = None) -> str | None:
    """The owner's user id when this participant is the owner calling; else None."""
    if not is_sip(participant):
        return None
    attributes = getattr(participant, "attributes", None) or {}
    if not isinstance(attributes, Mapping) or attributes.get(OWNER) != "1":
        return None
    if not signed(attributes, now):
        return None
    return as_user_id(attributes.get(USER_ID))


def owner_in_room(room: Any) -> str | None:
    """The owner's user id, if the owner is one of the callers in the room."""
    for participant in getattr(room, "remote_participants", {}).values():
        if owner := owner_on_call(participant):
            return owner
    return None
