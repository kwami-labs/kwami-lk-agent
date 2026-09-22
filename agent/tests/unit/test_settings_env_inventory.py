"""`Settings.ENV_VAR_NAMES` must stay equal to what `from_env` actually reads.

Moving the environment inventory into a constant only helps if the constant
cannot drift from the code below it. Two things now derive from it -- the test
suite's credential scrub and the Cloudflare Worker's forwarded env -- so a
variable that `from_env` reads but the tuple omits reintroduces exactly the bug
the tuple was added to kill.

This reads the source rather than the runtime because the failure is a *missing*
read: calling `from_env` with a clean environment cannot tell you about a
variable nobody remembered to enumerate. The AST walk sees every
`_env_str`/`_env_bool`/`_env_float` call site whether or not it fires.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.settings import ENV_VAR_NAMES, PROVIDER_KEY_NAMES, Settings

SETTINGS_SOURCE = Path(__file__).parent.parent.parent / "src" / "settings.py"

#: The helpers that turn an environment variable into a Settings field.
ENV_READERS = frozenset({"_env_str", "_env_bool", "_env_float"})


def _literally_read_env_vars() -> set[str]:
    """Every env var name passed as a literal to an `_env_*` helper."""
    tree = ast.parse(SETTINGS_SOURCE.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in ENV_READERS:
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.add(first.value)
    return found


def test_the_inventory_matches_what_from_env_reads() -> None:
    """No variable is read without being enumerated, and none is enumerated in vain."""
    # `PROVIDER_KEY_NAMES` is consumed by a comprehension, so its members are
    # never literal arguments at a call site -- add them back explicitly.
    actually_read = _literally_read_env_vars() | set(PROVIDER_KEY_NAMES)
    enumerated = set(ENV_VAR_NAMES)

    missing = actually_read - enumerated
    assert not missing, (
        f"from_env reads {sorted(missing)} but ENV_VAR_NAMES omits them. "
        "Anything deriving from that tuple -- the test credential scrub, the "
        "Worker's forwarded env -- silently will not cover these."
    )

    stale = enumerated - actually_read
    assert not stale, (
        f"ENV_VAR_NAMES lists {sorted(stale)}, which from_env no longer reads. "
        "Delete them rather than leaving the inventory describing code that is gone."
    )


def test_the_inventory_finds_a_newly_read_variable() -> None:
    """The AST walk must actually see a call site, not vacuously pass."""
    found = _literally_read_env_vars()
    # A representative of each helper, so a refactor that drops one is caught.
    assert "KWAMI_API_URL" in found, "_env_str call sites are not being seen"
    assert "KWAMI_ALLOW_BROWSER_JS" in found, "_env_bool call sites are not being seen"
    assert "KWAMI_API_TIMEOUT" in found, "_env_float call sites are not being seen"


def test_provider_keys_are_a_subset_of_the_inventory() -> None:
    """`ENV_VAR_NAMES` splices in `PROVIDER_KEY_NAMES`; keep that true."""
    assert set(PROVIDER_KEY_NAMES) <= set(ENV_VAR_NAMES)


def test_the_inventory_has_no_duplicates() -> None:
    """A duplicate would make the tuple disagree with its own set form."""
    assert len(ENV_VAR_NAMES) == len(set(ENV_VAR_NAMES))


def test_every_credential_field_has_an_enumerated_variable(fake_key) -> None:
    """Every provider key Settings can hold comes from a variable we enumerate.

    Guards the other direction from the AST walk: `provider_keys` is populated
    by name, so a key present at runtime but absent from the inventory would be
    a credential the scrub never clears.
    """
    fake_key(*PROVIDER_KEY_NAMES)
    settings = Settings.from_env()
    assert set(settings.provider_keys) <= set(ENV_VAR_NAMES)
    assert set(settings.provider_keys) == set(PROVIDER_KEY_NAMES)
