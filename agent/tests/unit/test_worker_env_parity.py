"""Every variable `Settings` reads must reach the Cloudflare container.

This is the only finding in the whole audit that was purely "someone added it in
one place and not the other", and it cost the default cloud-browser vendor: the
agent defaults to Browserbase, `infra/src/env.ts` forwarded neither
`BROWSERBASE_API_KEY` nor `BROWSERBASE_PROJECT_ID`, so on that deploy target the
agent fell back to Browser Use -- which `browser/providers.py` itself warns
signs the user out of every site they were logged into -- or, with no Browser
Use key either, had no browser at all.

The same class of bug had already happened twice more without being noticed:
`ELEVENLABS_API_KEY` (documented in `.env.sample` as an accepted alias, read by
`Settings`, never forwarded) and the test suite's own credential scrub, which
had drifted five variables out of date (see `tests/conftest.py`).

So the lists are not compared by eye. `Settings.ENV_VAR_NAMES` is the single
inventory, and this reads the TypeScript to check the Worker honours it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.settings import ENV_VAR_NAMES

WORKER_ENV_TS = Path(__file__).parent.parent.parent.parent / "infra" / "src" / "env.ts"

#: Variables that legitimately do not travel from Worker to container.
#:
#: Each needs a reason, because "add it to the allowlist" is exactly how this
#: check would rot into a rubber stamp.
ALLOWED_ABSENT = {
    # A filesystem path to a service-account JSON file. The file does not exist
    # in the container image, so forwarding the variable would point the Google
    # SDK at nothing. GOOGLE_API_KEY is the supported route on this target.
    "GOOGLE_APPLICATION_CREDENTIALS",
}


def _worker_forwarded_names() -> set[str]:
    """Every env var `containerEnvFromWorker` puts into the container.

    Reads both halves: the `out` object literal (plain vars) and
    `CONTAINER_SECRET_KEYS` (secrets). Checking only the second was how an
    earlier version of this guard reported false positives for `KWAMI_API_URL`.
    """
    source = WORKER_ENV_TS.read_text(encoding="utf-8")

    secrets_block = re.search(r"CONTAINER_SECRET_KEYS\s*=\s*\[(.*?)\]\s*as const", source, re.S)
    assert secrets_block, "CONTAINER_SECRET_KEYS not found -- has env.ts been restructured?"
    names = set(re.findall(r'"([A-Z0-9_]+)"', secrets_block.group(1)))

    out_block = re.search(r"const out: Record<string, string> = \{(.*?)\};", source, re.S)
    assert out_block, "the `out` literal was not found -- has env.ts been restructured?"
    names |= set(re.findall(r"^\s*([A-Z0-9_]+):", out_block.group(1), re.M))

    return names


def test_the_worker_forwards_every_variable_settings_reads() -> None:
    forwarded = _worker_forwarded_names()
    missing = set(ENV_VAR_NAMES) - forwarded - ALLOWED_ABSENT

    assert not missing, (
        f"Settings reads {sorted(missing)} but infra/src/env.ts does not forward them, "
        "so on Cloudflare the agent behaves as if they were unset. Add them to "
        "CONTAINER_SECRET_KEYS (or to the `out` literal for non-secrets), to "
        "infra/secrets.example.json, and to docs/deployment.md."
    )


def test_the_browserbase_credentials_specifically_are_forwarded() -> None:
    """Named explicitly because this is the pair that was actually missing, and
    because Browserbase is the default vendor -- a regression here is silent."""
    forwarded = _worker_forwarded_names()

    assert "BROWSERBASE_API_KEY" in forwarded
    assert "BROWSERBASE_PROJECT_ID" in forwarded
    assert "KWAMI_BROWSER_PROVIDER" in forwarded


def test_the_allowlist_only_holds_variables_settings_actually_reads() -> None:
    """A stale allowlist entry is a reason nobody has to justify any more."""
    stale = ALLOWED_ABSENT - set(ENV_VAR_NAMES)

    assert not stale, f"{sorted(stale)} is allowlisted but Settings no longer reads it"


@pytest.mark.parametrize("name", sorted(ALLOWED_ABSENT))
def test_an_allowlisted_variable_is_genuinely_absent(name: str) -> None:
    """If one starts being forwarded, the allowlist entry has to go with it --
    otherwise the reason recorded above stops matching what the code does."""
    assert name not in _worker_forwarded_names()


def test_the_secrets_template_matches_what_the_worker_forwards() -> None:
    """`infra/secrets.example.json` is what an operator copies and fills in. A
    secret the Worker forwards but the template omits is one nobody sets."""
    import json

    template_path = WORKER_ENV_TS.parent.parent / "secrets.example.json"
    template = set(json.loads(template_path.read_text(encoding="utf-8")))

    source = WORKER_ENV_TS.read_text(encoding="utf-8")
    secrets_block = re.search(r"CONTAINER_SECRET_KEYS\s*=\s*\[(.*?)\]\s*as const", source, re.S)
    assert secrets_block
    secrets = set(re.findall(r'"([A-Z0-9_]+)"', secrets_block.group(1)))

    missing = secrets - template
    assert not missing, (
        f"{sorted(missing)} are forwarded as Worker secrets but are absent from "
        "secrets.example.json, so `pnpm secrets:bulk` will never set them."
    )
