"""The purchase line in the browser and the rules around sending."""

from __future__ import annotations

import pytest

from src.domain.capabilities import CAPABILITIES
from src.domain.charter import HARD_LIMITS
from src.tools.builtin import looks_like_card_number


class TestCardNumbers:
    @pytest.mark.parametrize(
        "text",
        [
            "4111111111111111",
            "4111 1111 1111 1111",
            "4111-1111-1111-1111",
            "tarjeta 5500 0000 0000 0004 caduca 12/29",
            "378282246310005",
        ],
    )
    def test_a_card_is_recognised(self, text):
        assert looks_like_card_number(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "Madrid a Lisboa, 3 de noviembre",
            "+34 612 345 678",
            "4111111111111112",
            "1234567890123",
            "ES91 ABCD",
        ],
    )
    def test_what_is_not_a_card_passes(self, text):
        assert not looks_like_card_number(text)


class TestCharter:
    def test_the_exception_is_narrow(self):
        assert "express_order" in HARD_LIMITS
        assert "covers one message" in HARD_LIMITS

    def test_payment_stays_with_the_user(self):
        assert "payment screen" in HARD_LIMITS
        assert "never type card details" in HARD_LIMITS


class TestCapabilities:
    def _by_name(self, name):
        return next(c for c in CAPABILITIES if c.name == name)

    def test_the_two_mailboxes_are_told_apart(self):
        google = self._by_name("google_account")
        assert google.is_available(frozenset({"send_gmail_message"}))
        assert not google.is_available(frozenset({"send_email"}))
        assert "real account" in google.guidance

    def test_reminders_appear_with_their_tools(self):
        tasks = self._by_name("assistant_tasks")
        assert tasks.is_available(frozenset({"create_reminder"}))
        assert not tasks.is_available(frozenset({"send_email"}))
