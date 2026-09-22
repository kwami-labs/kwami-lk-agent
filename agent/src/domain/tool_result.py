"""A tool's answer, plus whether it was a refusal.

`browser_open_request` arrives on the data channel when the user clicks a search
result. There is no model turn behind it, so `navigate_to`'s answer has nowhere
to be returned to -- a refused URL would look exactly like a successful one from
outside, which is the silent failure that route exists to prevent. The handler
therefore had to detect refusal itself, and did it like this::

    if result.startswith(("I can't", "Cannot", "Failed")):

Reword one refusal in `navigate_to` -- "I'm not able to open that", say -- and
the browser panel goes back to failing silently, with nothing failing in CI to
say so. The prose of a sentence written for a language model is not a protocol.

`ToolResult` is a `str` subclass, deliberately. A function tool's return value
goes to the model as text and the framework expects a string, so the outcome
cannot replace it -- it travels alongside. Existing call sites that return a
plain `str` keep working, and `was_refused` answers False for them, which is the
right default: a tool that has not opted in is not claiming anything.
"""

from __future__ import annotations

from typing import Any


class ToolResult(str):
    """Text for the model, carrying whether the tool refused.

    Subclasses `str` so it can be returned from a `@function_tool` unchanged.
    Note that string operations on it (slicing, concatenation, `.strip()`)
    return a plain `str` and lose the flag -- which is correct: a transformed
    message is no longer this tool's verdict.
    """

    __slots__ = ("refused",)

    refused: bool

    def __new__(cls, text: str, *, refused: bool = False) -> ToolResult:
        result = super().__new__(cls, text)
        result.refused = refused
        return result

    def __repr__(self) -> str:
        return f"ToolResult({str.__repr__(self)}, refused={self.refused})"


def refusal(text: str) -> ToolResult:
    """A refusal: the tool could not do what was asked, and said why."""
    return ToolResult(text, refused=True)


def was_refused(result: Any) -> bool:
    """Whether `result` is a tool answer that reported a refusal.

    Takes `Any` because callers hand it whatever a tool returned, which may be a
    plain string, a dict, or None. Anything that has not explicitly declared a
    refusal is treated as not one.
    """
    return bool(getattr(result, "refused", False))
