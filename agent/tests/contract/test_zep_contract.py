"""Contract tests: every Zep Cloud method this codebase calls must exist.

The memory layer calls five methods that `zep-cloud` does not have. Each call
site swallows the resulting `AttributeError`, so the agent degrades silently:
`recall_memories` always answers "I don't have any memories about that yet"
(while still billing the user), retrieved facts are always empty, and the HTTP
connection pool is never closed.
"""

from __future__ import annotations

import pytest
from zep_cloud.client import AsyncZep

# (dotted path, the source that calls it) -- keep in step with src/memory/.
REQUIRED_ZEP_METHODS = [
    ("user.get", "memory/manager.py"),
    ("user.add", "memory/manager.py"),
    ("thread.get", "memory/context.py"),
    ("thread.create", "memory/manager.py"),
    ("thread.delete", "memory/manager.py"),
    ("thread.add_messages", "memory/manager.py"),
    ("thread.get_user_context", "memory/context.py"),
    ("graph.search", "memory/search.py"),
    ("graph.set_ontology", "memory/ontology.py"),
    ("graph.node.get_by_user_id", "memory/search.py"),
]


@pytest.fixture
def client() -> AsyncZep:
    return AsyncZep(api_key="test-key-not-real")


@pytest.mark.parametrize("path,caller", REQUIRED_ZEP_METHODS)
def test_required_zep_method_exists(client: AsyncZep, path: str, caller: str) -> None:
    target = client
    for part in path.split("."):
        assert hasattr(target, part), f"{caller} calls client.{path}, which zep-cloud does not have"
        target = getattr(target, part)
    assert callable(target)


def test_the_memory_layer_only_calls_methods_that_exist(client: AsyncZep) -> None:
    """Catches the five phantom calls, wherever they are reached from.

    These all exist in the code today and all fail at runtime:
    `thread.search`, `thread.get_context`, `thread.get_messages`,
    `graph.get_ontology`, and `client.close()`.
    """
    import re
    from pathlib import Path

    memory_dir = Path(__file__).parent.parent.parent / "src" / "memory"
    # The receiver must be exactly `client`, `_client` or `self._client` -- the
    # lookbehind keeps `httpx_client.aclose()` from being read as a Zep call.
    pattern = re.compile(r"(?<![\w.])(?:self\.)?_?client\.((?:\w+\.)*\w+)\s*\(")

    def _strip_comments(text: str) -> str:
        return "\n".join(line.split("#", 1)[0] for line in text.splitlines())

    missing: list[str] = []
    for source in sorted(memory_dir.glob("*.py")):
        for call in set(pattern.findall(_strip_comments(source.read_text()))):
            target = client
            for part in call.split("."):
                if not hasattr(target, part):
                    missing.append(f"{source.name}: client.{call}")
                    break
                target = getattr(target, part)

    assert not missing, "memory layer calls Zep methods that do not exist:\n  " + "\n  ".join(
        sorted(missing)
    )


def test_the_zep_client_can_actually_be_closed(client: AsyncZep) -> None:
    """`close()` is swallowed by a bare except, so every reconfiguration leaks a pool."""
    has_direct_close = hasattr(client, "close") or hasattr(client, "aclose")
    reachable = getattr(getattr(client, "_client_wrapper", None), "httpx_client", None)
    assert has_direct_close or reachable is not None, (
        "no way to close the Zep client; connection pools will leak on every agent swap"
    )


def test_get_user_context_supports_the_template_free_call(client: AsyncZep) -> None:
    """The dead `thread.get_context` fallback has a direct replacement.

    `get_user_context` takes an optional `template_id`, so the fallback path can
    be repaired rather than deleted.
    """
    import inspect

    params = inspect.signature(client.thread.get_user_context).parameters
    assert "template_id" in params
    assert params["template_id"].default is None
    assert "min_rating" in params
