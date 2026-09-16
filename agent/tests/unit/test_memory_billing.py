"""Zep is billed per call, so a call that produced nothing must not be charged.

The retrieval helpers swallow their own exceptions and return an empty result,
which is indistinguishable from a successful lookup against a new user unless
it is checked explicitly. Combined with the phantom-method bugs, users were
charged for `thread.search` calls that raised on every single attempt.
"""

from __future__ import annotations

from src.memory.context import MemoryContext


def test_an_empty_context_reports_no_content() -> None:
    assert MemoryContext().has_content() is False


def test_a_context_block_counts_as_content() -> None:
    assert MemoryContext(context_block="FACTS: lives in Barcelona").has_content() is True


def test_a_summary_counts_as_content() -> None:
    assert MemoryContext(summary="Talked about travel").has_content() is True


def test_facts_count_as_content() -> None:
    assert MemoryContext(facts=["likes espresso"]).has_content() is True


def test_recent_messages_count_as_content() -> None:
    assert MemoryContext(recent_messages=[{"role": "user", "content": "hi"}]).has_content() is True


def test_entities_alone_are_not_billable_content() -> None:
    """`entities` is never populated on the live path, so it cannot prove a hit."""
    assert MemoryContext(entities=[{"name": "Barcelona"}]).has_content() is False
