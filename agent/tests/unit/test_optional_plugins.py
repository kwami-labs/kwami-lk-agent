"""Optional `livekit-plugins-*` extras resolve to a module or to None.

This replaced four copies of a `try: from livekit.plugins import x` guard whose
`type: ignore` could not be correct in both an environment that has the extra
and one that does not. The behaviour it has to preserve is narrow but load
bearing: absent means None, so the factory falls back with a warning, and
present means the real module, so the provider is actually usable.

Both branches are exercised against real imports rather than a patched
`importlib`, because a test that monkeypatches the thing under test only proves
the monkeypatch works.
"""

from __future__ import annotations

import sys

from src.factories.optional import PLUGIN_NAMESPACE, optional_module, optional_plugin


def test_a_module_that_exists_comes_back() -> None:
    """A real, always-present module stands in for an installed extra."""
    assert optional_module("json") is sys.modules["json"]


def test_a_module_that_does_not_exist_is_none() -> None:
    assert optional_module("livekit.plugins.definitely_not_a_real_plugin") is None


def test_a_dotted_path_resolves_to_the_submodule() -> None:
    """The plugins are submodules, so the dotted form has to work."""
    module = optional_module("json.decoder")

    assert module is not None
    assert module.__name__ == "json.decoder"


def test_optional_plugin_builds_the_livekit_namespace_path() -> None:
    """The caller passes a bare name; the namespace is this module's business."""
    assert PLUGIN_NAMESPACE == "livekit.plugins"
    assert optional_plugin("definitely_not_a_real_plugin") is None


def test_an_absent_extra_is_none_rather_than_raising() -> None:
    """The four real ones. None of these extras is installed in the offline env,
    which is exactly the condition the factories must survive."""
    for name in ("google", "anthropic", "groq", "assemblyai"):
        assert optional_plugin(name) is None, (
            f"{name} resolved, so this test no longer covers the absent branch"
        )


def test_an_import_error_from_inside_the_module_is_not_swallowed(monkeypatch) -> None:
    """A plugin present but broken is a fault, not an absent extra.

    `importlib` raises ImportError for both, so the distinction this test pins
    is the one the docstring claims: anything that is *not* ImportError -- a
    missing native library surfacing as OSError, a config error at import --
    propagates instead of being read as "not installed".
    """
    import importlib

    def explode(name: str) -> None:
        raise RuntimeError(f"native library missing for {name}")

    monkeypatch.setattr(importlib, "import_module", explode)

    try:
        optional_module("livekit.plugins.google")
    except RuntimeError as exc:
        assert "native library missing" in str(exc)
    else:  # pragma: no cover - the assertion above is the point of the test
        raise AssertionError("a non-ImportError failure was swallowed")
