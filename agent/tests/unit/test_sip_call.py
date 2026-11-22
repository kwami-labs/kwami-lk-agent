"""A phone call is believed only as far as the API signed it.

The owner calling is the owner (and may ask for work on the PC or in the cloud);
anybody else, or anybody sending the same headers straight to the SIP address,
stays a guest -- and cannot pick whose kwami answers.
"""

from __future__ import annotations

import hashlib
import hmac
from types import SimpleNamespace

import pytest
from livekit.rtc import ParticipantKind

from src.runtime.sip_call import owner_in_room, owner_on_call, signed
from src.runtime_bootstrap import resolve_kwami_id

SECRET = "test-secret"  # noqa: S105 -- a test key
KWAMI = "kwami-1"
OWNER_ID = "00000000-0000-4000-8000-000000000001"
NOW = 1_800_000_000


def _sig(kwami: str, user: str, owner: str, ts: int, secret: str = SECRET) -> str:
    key = f"sip-call:{secret}".encode()
    return hmac.new(key, f"{kwami}|{user}|{owner}|{ts}".encode(), hashlib.sha256).hexdigest()


def _attrs(owner: str = "1", ts: int = NOW, **over: str) -> dict[str, str]:
    attrs = {
        "kwami_id": KWAMI,
        "kwami.user_id": OWNER_ID,
        "kwami.owner": owner,
        "kwami.ts": str(ts),
        "kwami.sig": _sig(KWAMI, OWNER_ID, owner, ts),
    }
    attrs.update(over)
    return attrs


def _caller(attributes: dict[str, str], kind=ParticipantKind.PARTICIPANT_KIND_SIP):
    return SimpleNamespace(
        identity="sip_+15550000001", kind=kind, attributes=attributes, metadata=None
    )


@pytest.fixture(autouse=True)
def _secret(env_setting):
    env_setting("SUPABASE_SECRET_KEY", SECRET)


@pytest.fixture
def at_now(monkeypatch):
    monkeypatch.setattr("src.runtime.sip_call.time.time", lambda: NOW + 30)


class TestSigned:
    def test_the_api_s_signature_holds(self):
        assert signed(_attrs(), now=NOW + 5)

    def test_a_changed_field_breaks_it(self):
        assert not signed(_attrs(**{"kwami.owner": "0"}), now=NOW)
        assert not signed(_attrs(kwami_id="someone-elses"), now=NOW)

    def test_a_stale_signature_is_refused(self):
        assert not signed(_attrs(), now=NOW + 3600)

    def test_another_key_is_refused(self):
        attrs = _attrs(**{"kwami.sig": _sig(KWAMI, OWNER_ID, "1", NOW, secret="other")})  # noqa: S106
        assert not signed(attrs, now=NOW)

    def test_without_a_key_nothing_is_believed(self, env_setting):
        env_setting("SUPABASE_SECRET_KEY", "")
        assert not signed(_attrs(), now=NOW)

    @pytest.mark.parametrize("missing", ["kwami.sig", "kwami.ts"])
    def test_a_missing_part_is_refused(self, missing):
        attrs = _attrs()
        del attrs[missing]
        assert not signed(attrs, now=NOW)

    def test_a_timestamp_that_is_no_number_is_refused(self):
        assert not signed(_attrs(**{"kwami.ts": "soon"}), now=NOW)


class TestOwnerOnCall:
    def test_the_owner_calling_is_the_owner(self, at_now):
        assert owner_on_call(_caller(_attrs())) == OWNER_ID

    def test_a_stranger_is_not(self, at_now):
        assert owner_on_call(_caller(_attrs(owner="0"))) is None

    def test_forged_headers_are_not(self, at_now):
        assert owner_on_call(_caller(_attrs(**{"kwami.sig": "0" * 64}))) is None

    def test_only_a_phone_caller_can_be_vouched_for(self, at_now):
        standard = _caller(_attrs(), kind=ParticipantKind.PARTICIPANT_KIND_STANDARD)
        assert owner_on_call(standard) is None

    def test_attributes_that_are_no_mapping_are_nobody(self, at_now):
        caller = _caller({})
        caller.attributes = None
        assert owner_on_call(caller) is None

    def test_a_user_id_that_is_no_uuid_is_nobody(self, at_now):
        attrs = _attrs()
        attrs["kwami.user_id"] = "not-a-uuid"
        attrs["kwami.sig"] = _sig(KWAMI, "not-a-uuid", "1", NOW)
        assert owner_on_call(_caller(attrs)) is None

    def test_the_owner_is_found_among_the_room_s_callers(self, at_now):
        room = SimpleNamespace(
            remote_participants={"a": _caller(_attrs(owner="0")), "b": _caller(_attrs())}
        )
        assert owner_in_room(room) == OWNER_ID
        assert owner_in_room(SimpleNamespace(remote_participants={})) is None


class TestWhoseKwamiAnswers:
    def _ctx(self, caller):
        return SimpleNamespace(
            job=SimpleNamespace(metadata=None),
            room=SimpleNamespace(remote_participants={"c": caller}),
        )

    def test_a_signed_call_names_the_kwami(self, at_now):
        assert resolve_kwami_id(self._ctx(_caller(_attrs(owner="0")))) == KWAMI

    def test_a_forged_call_names_none(self, at_now):
        forged = _caller({"kwami_id": "someone-elses"})
        assert resolve_kwami_id(self._ctx(forged)) is None
