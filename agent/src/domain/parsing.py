"""Wire-format parsing: camelCase JSON from the frontend into config values.

Two bug classes live here, and both were spread across the config handler
rather than being solved once:

* **Truthiness guards made `0` unsettable.** `if data.get("temperature"):`
  rejects `0.0` exactly as it rejects a missing key, so `temperature=0`,
  `speed=0` and `maxTokens=0` were silently ignored -- the value a user is most
  likely to choose deliberately.
* **`null` sections dropped the whole config.** `message.get("voice", {})`
  returns `None` when the client sends `"voice": null`, and the next
  `.get(...)` raised AttributeError, which aborted the entire config message
  and left the placeholder agent live.

Everything here is pure and total: it never raises on malformed input, it
returns `None` for "not provided" and preserves falsy values that were.
"""

from __future__ import annotations

from typing import Any

__all__ = ["boolean", "integer", "number", "section", "text", "value_from_keys"]


def value_from_keys(data: Any, *keys: str) -> Any:
    """First key that is present and not None, preserving falsy values.

    Accepts the camelCase and snake_case spellings of the same field, which is
    why it takes several keys.
    """
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def section(data: Any, *keys: str) -> dict[str, Any]:
    """A nested config section, tolerating missing, null and wrong-typed values.

    Always returns a dict, so callers can chain without a None check.
    """
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def text(data: Any, *keys: str) -> str | None:
    """A non-empty string, or None.

    An empty or whitespace-only string means "not set" for provider, model and
    voice names -- unlike a number, a blank name is never a deliberate choice.
    """
    value = value_from_keys(data, *keys)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def number(data: Any, *keys: str) -> float | None:
    """A float, preserving `0.0`. Returns None for absent or unparseable values."""
    value = value_from_keys(data, *keys)
    if isinstance(value, bool):  # bool is an int; never a temperature
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def integer(data: Any, *keys: str) -> int | None:
    """An int, preserving `0`. Returns None for absent or unparseable values."""
    value = number(data, *keys)
    if value is None:
        return None
    return int(value)


def boolean(data: Any, *keys: str) -> bool | None:
    """A bool, distinguishing `False` from absent.

    Accepts the common JSON-ish spellings so a client sending "false" as a
    string cannot accidentally enable a feature.
    """
    value = value_from_keys(data, *keys)
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return None
