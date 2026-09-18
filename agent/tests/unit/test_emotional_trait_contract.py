"""The emotional-trait vocabulary is shared between two repos and defended by neither.

`set_soul_control` in kwami-app lets the user set ten emotional traits by
voice. `describe_emotional_traits` in this repo turns them into the directive
the model actually reads. The join between them is a set of bare strings, and
the failure mode is silent in both directions:

* `describe_emotional_traits` skips any key it does not recognise
  (`if key not in TRAIT_LABELS: continue`), so a trait the app renames stops
  affecting the prompt with no error anywhere -- the user says "be warmer", the
  app reports success, the setting is stored, and nothing about the voice
  changes.
* The scale is signed, -100..100 with 0 neutral. Read as 0..1 it would be
  pinned to one pole; read as 0..100 it could never express the negative half,
  which is where "more reserved" and "more tentative" live.

Neither repo's suite can see the other, so this is the only place the agreement
can be checked. It is skipped when kwami-app is not checked out alongside,
which keeps this suite standalone without giving up the check where it matters.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.domain.prompt import (
    TRAIT_LABELS,
    TRAIT_MIN_MAGNITUDE,
    TRAIT_WEIGHTS,
    describe_emotional_traits,
)

APP_TOOLS_SOURCE = (
    Path(__file__).resolve().parents[4] / "kwami-app/src/composables/useWorkspaceAgentTools.ts"
)

#: The scale the app's sliders use. Mirrored here so the assertions below say
#: what they are checking rather than repeating a magic number.
TRAIT_MIN = -100
TRAIT_MAX = 100


def _app_trait_names() -> list[str]:
    source = APP_TOOLS_SOURCE.read_text(encoding="utf-8")
    match = re.search(r"EMOTIONAL_TRAITS\s*=\s*\[(.*?)\]", source, re.S)
    assert match, "could not find EMOTIONAL_TRAITS in the app; this test has gone stale"
    return re.findall(r"'([a-z]+)'", match.group(1))


# -- the vocabulary ---------------------------------------------------------


def test_labels_and_weights_cover_the_same_traits() -> None:
    """A trait with a label and no weight is silently weighted 1.0, and vice versa."""
    assert set(TRAIT_LABELS) == set(TRAIT_WEIGHTS)


def test_every_trait_has_both_poles() -> None:
    """The scale is signed; a trait with one label cannot express the negative half."""
    for trait, labels in TRAIT_LABELS.items():
        assert len(labels) == 2, f"{trait} has no opposite pole"
        low, high = labels
        assert low and high
        assert low != high


@pytest.mark.skipif(
    not APP_TOOLS_SOURCE.exists(),
    reason="kwami-app is not checked out beside this repo",
)
def test_the_app_and_the_prompt_agree_on_the_trait_names() -> None:
    """The join is bare strings, and a mismatch fails silently on both sides."""
    app_traits = _app_trait_names()

    assert app_traits, "parsed no traits out of the app; this test has gone stale"
    missing = set(app_traits) - set(TRAIT_LABELS)
    assert not missing, (
        f"kwami-app can set {sorted(missing)} by voice, but describe_emotional_traits "
        "skips unknown keys -- those settings would change nothing about how you sound."
    )
    unused = set(TRAIT_LABELS) - set(app_traits)
    assert not unused, (
        f"the prompt describes {sorted(unused)}, which the app cannot set. Harmless, but "
        "it means the guidance names a control the user has no way to reach."
    )


# -- the scale --------------------------------------------------------------


def test_the_poles_produce_opposite_directives() -> None:
    """-100 and +100 must not read the same, or the sign is being dropped."""
    positive = describe_emotional_traits({"empathy": TRAIT_MAX})
    negative = describe_emotional_traits({"empathy": TRAIT_MIN})

    assert positive and negative
    assert positive != negative
    assert "empathic" in positive
    assert "detached" in negative


def test_zero_is_neutral_and_says_nothing() -> None:
    """0 is the slider's rest position, not a weak instruction."""
    assert describe_emotional_traits(dict.fromkeys(TRAIT_LABELS, 0)) is None


def test_a_small_nudge_is_not_worth_a_directive() -> None:
    below = TRAIT_MIN_MAGNITUDE / 2
    assert describe_emotional_traits({"energy": below}) is None


def test_a_full_slider_reads_as_strong() -> None:
    directive = describe_emotional_traits({"confidence": TRAIT_MAX})
    assert directive is not None
    assert "very strongly" in directive


@pytest.mark.parametrize("value", [TRAIT_MIN, TRAIT_MAX])
def test_every_trait_is_expressible_at_both_extremes(value: int) -> None:
    """Each trait, alone at a pole, must produce a directive.

    A weight low enough to leave a full slider under `TRAIT_MIN_MAGNITUDE`
    would make that trait unusable: the user drags it all the way and hears
    nothing.
    """
    for trait in TRAIT_LABELS:
        directive = describe_emotional_traits({trait: value})
        assert directive is not None, f"{trait} at {value} produces no directive at all"


def test_out_of_range_values_do_not_break_the_prompt() -> None:
    """The app clamps, but this data also arrives from stored configs."""
    directive = describe_emotional_traits({"energy": 10_000, "calmness": -10_000})
    assert directive is not None
    assert "10000" not in directive


def test_junk_is_ignored_rather_than_raising() -> None:
    """This comes off the wire; a bad value must not cost the turn."""
    assert describe_emotional_traits({"energy": "very"}) is None
    assert describe_emotional_traits({"not_a_trait": 100}) is None
    assert describe_emotional_traits({"energy": None}) is None
