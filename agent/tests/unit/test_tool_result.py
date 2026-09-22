"""A refusal is carried as a flag, not inferred from the sentence.

`browser_open_request` arrives when the user clicks a search result. There is no
model turn behind it, so `navigate_to`'s answer has nowhere to be returned to --
a refused URL looked exactly like a successful one, which is the silent failure
that route exists to prevent. The handler detected refusal with::

    result.startswith(("I can't", "Cannot", "Failed"))

so rewording one message would have restored the silent failure with nothing in
CI to say so. These tests pin both halves: the flag survives the trip, and the
text is free to change without breaking the handler.
"""

from __future__ import annotations

from src.domain.tool_result import ToolResult, refusal, was_refused


def test_a_refusal_is_still_a_string() -> None:
    """A function tool's return value goes to the model as text, so the outcome
    has to travel *with* the string rather than replace it."""
    result = refusal("I can't open that address.")

    assert isinstance(result, str)
    assert result == "I can't open that address."


def test_a_refusal_reports_itself() -> None:
    assert was_refused(refusal("no")) is True


def test_a_plain_success_does_not() -> None:
    assert was_refused(ToolResult("Opening the page.")) is False


def test_an_ordinary_string_is_not_a_refusal() -> None:
    """Most tools return a bare str and are not claiming anything; the default
    must be "not a refusal" rather than a guess from the wording."""
    assert was_refused("I can't do that, actually") is False
    assert was_refused("Cannot open browser: boom") is False


def test_non_string_results_are_handled() -> None:
    """Callers hand this whatever a tool returned -- a dict, None, a number."""
    assert was_refused(None) is False
    assert was_refused({"status": "refused"}) is False
    assert was_refused(42) is False


def test_the_wording_is_free_to_change() -> None:
    """The whole point. None of these would have matched the old prefix list."""
    for wording in (
        "I'm not able to open that address.",
        "That address isn't one I can reach.",
        "Sorry — no.",
        "",
    ):
        assert was_refused(refusal(wording)) is True


def test_string_operations_drop_the_flag() -> None:
    """Documented behaviour, and the right one: a transformed message is no
    longer this tool's verdict, so it must not keep claiming to be."""
    result = refusal("I can't open that address.")

    assert was_refused(result.strip()) is False
    assert was_refused(result[:5]) is False
    assert was_refused(result + "!") is False


def test_the_repr_shows_the_outcome() -> None:
    """Debugging a silent-failure bug is much easier when the log line says."""
    assert "refused=True" in repr(refusal("no"))
    assert "refused=False" in repr(ToolResult("yes"))


# -- The tool and the handler agree ------------------------------------------


async def test_navigate_to_refuses_an_unsafe_url_structurally() -> None:
    """The refusal `browser_open_request` has to detect. Asserted through the
    flag, so this test does not re-encode the wording it is meant to free."""
    from src.agent import KwamiAgent

    agent = KwamiAgent()

    result = await agent.navigate_to(None, "http://169.254.169.254/latest/meta-data/")

    assert was_refused(result)


async def test_navigate_to_refuses_without_a_kwami_id() -> None:
    """A browser profile carries the user's cookies, so a session with no tenant
    id must not get one -- and the data-channel path must be able to see that."""
    from src.agent import KwamiAgent

    agent = KwamiAgent()
    assert agent.kwami_config.kwami_id == ""

    result = await agent.navigate_to(None, "https://example.com")

    assert was_refused(result)
