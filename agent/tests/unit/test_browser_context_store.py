"""The context store is what makes "carry on where I left off" true.

Browserbase hands back an opaque context id and offers no way to look one up
again. If that id is lost, the next session comes up signed out of everything
and the old context is orphaned and still billed. So the store is on the
critical path of a feature, but it must never be on the critical path of the
browser opening at all: every failure here degrades to "no saved logins", never
to "no browsing".
"""

from __future__ import annotations

from typing import Any

from src.browser.context_store import (
    InMemoryContextStore,
    KwamiApiContextStore,
    create_context_store,
)
from src.settings import Settings

VENDOR = "browserbase"


class FakeResponse:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    @property
    def text(self) -> str:
        return str(self._payload)

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeHttp:
    def __init__(self, get_result: Any = None, post_result: Any = None) -> None:
        self._get_result = get_result or FakeResponse(404)
        self._post_result = post_result or FakeResponse(200, {})
        self.gets: list[tuple[str, dict, dict]] = []
        self.posts: list[tuple[str, Any, dict]] = []

    async def get(self, url, *, params=None, headers=None, timeout=None):
        self.gets.append((url, params or {}, headers or {}))
        if isinstance(self._get_result, Exception):
            raise self._get_result
        return self._get_result

    async def post(self, url, *, json=None, headers=None, timeout=None):
        self.posts.append((url, json, headers or {}))
        if isinstance(self._post_result, Exception):
            raise self._post_result
        return self._post_result


def _settings(**kwargs: Any) -> Settings:
    return Settings(
        kwami_api_url="https://api.example.test",
        kwami_api_key="secret",
        **kwargs,
    )


def _store(http: FakeHttp) -> KwamiApiContextStore:
    return KwamiApiContextStore(settings=_settings(), http=http)


# -- Reading -----------------------------------------------------------------


async def test_reads_a_saved_context_id() -> None:
    http = FakeHttp(get_result=FakeResponse(200, {"context_id": "ctx-7"}))

    assert await _store(http).get("user-1", VENDOR) == "ctx-7"

    url, params, headers = http.gets[0]
    assert url.endswith("/internal/browser-contexts/user-1")
    assert params == {"vendor": VENDOR}
    assert headers["X-Kwami-API-Key"] == "secret"


async def test_a_user_id_with_a_slash_cannot_escape_its_path() -> None:
    """User ids reach us from room identity; a raw one could address any route."""
    http = FakeHttp()

    await _store(http).get("../../internal/kwamis", VENDOR)

    url, _, _ = http.gets[0]
    assert url.endswith("/internal/browser-contexts/..%2F..%2Finternal%2Fkwamis")


async def test_no_saved_context_reads_as_none() -> None:
    assert await _store(FakeHttp(get_result=FakeResponse(404))).get("user-1", VENDOR) is None


async def test_a_store_outage_does_not_stop_browsing() -> None:
    for failure in (
        FakeHttp(get_result=ConnectionError("down")),
        FakeHttp(get_result=FakeResponse(500, "boom")),
        FakeHttp(get_result=FakeResponse(200, ValueError("not json"))),
        FakeHttp(get_result=FakeResponse(200, {"context_id": ""})),
        FakeHttp(get_result=FakeResponse(200, ["unexpected"])),
    ):
        assert await _store(failure).get("user-1", VENDOR) is None


# -- Writing -----------------------------------------------------------------


async def test_saves_the_context_id_for_next_time() -> None:
    http = FakeHttp()

    await _store(http).put("user-1", VENDOR, "ctx-7")

    url, body, headers = http.posts[0]
    assert url.endswith("/internal/browser-contexts/user-1")
    assert body == {"vendor": VENDOR, "context_id": "ctx-7"}
    assert headers["X-Kwami-API-Key"] == "secret"


async def test_a_failed_save_is_survivable() -> None:
    http = FakeHttp(post_result=ConnectionError("down"))
    await _store(http).put("user-1", VENDOR, "ctx-7")  # must not raise


async def test_nothing_is_written_without_a_user_or_a_context() -> None:
    http = FakeHttp()
    store = _store(http)

    await store.put("", VENDOR, "ctx-7")
    await store.put("user-1", VENDOR, "")

    assert http.posts == []


# -- Configuration -----------------------------------------------------------


async def test_the_store_is_inert_without_an_api_key() -> None:
    http = FakeHttp(get_result=FakeResponse(200, {"context_id": "ctx-7"}))
    store = KwamiApiContextStore(settings=Settings(kwami_api_url="https://x"), http=http)

    assert await store.get("user-1", VENDOR) is None
    await store.put("user-1", VENDOR, "ctx-7")
    assert http.gets == [] and http.posts == []


def test_falls_back_to_memory_when_the_api_is_not_configured() -> None:
    assert isinstance(create_context_store(Settings()), InMemoryContextStore)
    assert isinstance(create_context_store(_settings()), KwamiApiContextStore)


async def test_the_memory_store_keeps_users_and_vendors_apart() -> None:
    store = InMemoryContextStore()

    await store.put("user-1", "browserbase", "ctx-1")
    await store.put("user-2", "browserbase", "ctx-2")
    await store.put("user-1", "browser_use", "profile-1")

    assert await store.get("user-1", "browserbase") == "ctx-1"
    assert await store.get("user-2", "browserbase") == "ctx-2"
    assert await store.get("user-1", "browser_use") == "profile-1"
    assert await store.get("user-3", "browserbase") is None


async def test_a_rejected_save_is_logged_rather_than_raised() -> None:
    """The browser is already open; losing the handle costs the next session."""
    http = FakeHttp(post_result=FakeResponse(500, "server error"))

    await _store(http).put("user-1", VENDOR, "ctx-7")  # must not raise

    assert len(http.posts) == 1


async def test_a_404_on_write_reports_that_persistence_is_not_implemented(caplog) -> None:
    """A write cannot legitimately 404 -- the route exists or it does not.

    This is the only signal that separates "no context saved yet" from "the API
    has no such endpoint". Production has the second: `GET /openapi.json` lists
    two internal routes and this is not one of them. Without this line the
    feature fails completely silently -- `get` returns None forever, every
    session mints a fresh Browserbase Context, users are signed out of every
    site each time, and the orphaned contexts keep being billed.
    """
    import logging

    http = FakeHttp(post_result=FakeResponse(404, {"detail": "Not Found"}))
    store = KwamiApiContextStore(settings=_settings(), http=http)

    with caplog.at_level(logging.ERROR):
        await store.put("user-1", "browserbase", "ctx_1")

    assert "has no" in caplog.text
    assert "browser-contexts" in caplog.text
    assert "NOT working" in caplog.text


async def test_a_404_on_read_is_not_an_error(caplog) -> None:
    """On the read path 404 is ambiguous and usually benign -- it is what every
    user's first browser open looks like. Only the write path can tell."""
    import logging

    http = FakeHttp(get_result=FakeResponse(404, {"detail": "Not Found"}))
    store = KwamiApiContextStore(settings=_settings(), http=http)

    with caplog.at_level(logging.ERROR):
        assert await store.get("user-1", "browserbase") is None

    assert caplog.text == ""
