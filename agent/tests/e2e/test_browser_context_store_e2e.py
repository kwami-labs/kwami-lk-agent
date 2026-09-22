"""The browser context store, against the real Kwami API.

`KwamiApiContextStore` maps a user to the opaque Browserbase Context id holding
their cookies and logins. That id is the only way to ever reach that context
again: Browserbase has no lookup-by-name endpoint, and creating the same name
twice is rejected rather than returning the original. So if this store stops
working, every session silently starts a fresh context — the user is signed out
of everything, and the old context is orphaned and keeps being billed for
storage.

Every failure path in the store returns `None` or does nothing, on purpose: not
finding a saved context costs the user their logins, and raising would cost them
the whole browser. That is the right trade, and it is also why a shape change in
`kwami-lk-api` produces no error anywhere — just one `warning` line and a user
who has to log in again.

**Why this is here and not in `contract/`.** Every file in that layer is
in-process introspection of an installed SDK — `hasattr` on a real `AsyncZep`,
`inspect.signature` on `Agent.on_enter` — and none of them makes a network call.
A `respx` test would only pin *our* request handling against a mock we wrote,
which cannot detect the other service changing. Detecting that needs the real
service, which means `live`.
"""

from __future__ import annotations

import os
import uuid

import pytest

from src.browser.context_store import KwamiApiContextStore
from src.settings import Settings

pytestmark = pytest.mark.live

VENDOR = "browserbase"


@pytest.fixture
def api_settings() -> Settings:
    """Settings pointed at the real Kwami API, or a skip."""
    url = os.environ.get("KWAMI_API_URL", "")
    key = os.environ.get("KWAMI_API_KEY", "")
    if not url or not key or "localhost" in url:
        pytest.skip("needs KWAMI_API_URL and KWAMI_API_KEY pointing at a real deployment")
    return Settings(kwami_api_url=url.rstrip("/"), kwami_api_key=key)


@pytest.fixture
def store(api_settings: Settings) -> KwamiApiContextStore:
    return KwamiApiContextStore(settings=api_settings)


@pytest.fixture
def throwaway_user() -> str:
    """A user id no real session will ever use."""
    return f"e2e_context_store_{uuid.uuid4().hex[:12]}"


async def test_a_saved_context_comes_back(store: KwamiApiContextStore, throwaway_user: str) -> None:
    """The round trip the whole feature depends on.

    If this fails, the symptom in production is not an error — it is users
    being signed out of every site, session after session.
    """
    context_id = f"ctx_{uuid.uuid4().hex}"

    await store.put(throwaway_user, VENDOR, context_id)
    found = await store.get(throwaway_user, VENDOR)

    assert found == context_id, (
        "the Kwami API did not return the context id it was just given -- "
        "/internal/browser-contexts has changed shape, and every browser "
        "session is now starting a fresh context"
    )


async def test_an_unknown_user_has_no_context(
    store: KwamiApiContextStore, throwaway_user: str
) -> None:
    """A 404 must read as "no context yet", not as an error.

    This is the first-ever browser open for every user, so getting it wrong
    would break the feature for everybody rather than degrade it.
    """
    assert await store.get(f"{throwaway_user}_never_written", VENDOR) is None


async def test_contexts_are_scoped_per_vendor(
    store: KwamiApiContextStore, throwaway_user: str
) -> None:
    """Browserbase Contexts and Browser Use profiles are different namespaces.
    Handing a Browserbase id to Browser Use would fail at launch."""
    browserbase_id = f"ctx_{uuid.uuid4().hex}"

    await store.put(throwaway_user, VENDOR, browserbase_id)

    assert await store.get(throwaway_user, "browser_use") != browserbase_id


async def test_a_later_write_replaces_the_earlier_one(
    store: KwamiApiContextStore, throwaway_user: str
) -> None:
    """Contexts are re-created when one is lost vendor-side; the store has to
    follow, or every session reaches for a context that no longer exists."""
    first = f"ctx_{uuid.uuid4().hex}"
    second = f"ctx_{uuid.uuid4().hex}"

    await store.put(throwaway_user, VENDOR, first)
    await store.put(throwaway_user, VENDOR, second)

    assert await store.get(throwaway_user, VENDOR) == second


async def test_a_bad_api_key_degrades_rather_than_raising(api_settings: Settings) -> None:
    """A rejected key must cost the user their logins, not their browser."""
    store = KwamiApiContextStore(
        settings=Settings(
            kwami_api_url=api_settings.kwami_api_url,
            kwami_api_key="definitely-not-a-valid-key",
        )
    )

    assert await store.get("anyone", VENDOR) is None
    await store.put("anyone", VENDOR, "ctx_whatever")  # must not raise
