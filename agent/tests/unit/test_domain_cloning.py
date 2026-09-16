"""Reconfiguration must not hand the new agent the old agent's soul."""

from __future__ import annotations

from dataclasses import replace

from src.config import KwamiConfig
from src.domain.cloning import clone_config


def test_a_clone_is_fully_independent() -> None:
    original = KwamiConfig()
    original.soul.traits.append("warm")

    copied = clone_config(original)
    copied.soul.traits.append("only-on-the-copy")
    copied.soul.name = "Ada"

    assert original.soul.traits == ["warm"]
    assert original.soul.name != "Ada"


def test_nested_config_objects_are_distinct() -> None:
    original = KwamiConfig()
    copied = clone_config(original)

    assert copied is not original
    assert copied.soul is not original.soul
    assert copied.voice is not original.voice
    assert copied.memory is not original.memory


def test_values_survive_the_copy() -> None:
    original = KwamiConfig()
    original.voice.tts_speed = 1.4
    original.soul.name = "Ada"

    copied = clone_config(original)

    assert copied.voice.tts_speed == 1.4
    assert copied.soul.name == "Ada"


def test_dataclasses_replace_is_the_shallow_behaviour_we_are_avoiding() -> None:
    """Pins why `clone_config` exists, so nobody reverts to `replace`."""
    original = KwamiConfig()
    shallow = replace(original)

    assert shallow.soul is original.soul, "replace() is shallow -- that was the bug"

    shallow.soul.traits.append("leaked")
    assert "leaked" in original.soul.traits
