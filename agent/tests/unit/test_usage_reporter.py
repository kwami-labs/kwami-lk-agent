"""`UsageReporter` is the last thing that runs in a session, and the only thing
that turns a call into revenue.

It is exercised against a real aiohttp server on an ephemeral port rather than
a stubbed ClientSession, because the parts most worth pinning -- that the
4-second timeout is actually applied, that a non-200 is read as a failure
rather than raising -- only exist in the real client.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from aiohttp import web

from src.domain import UsageTracker
from src.usage.reporter import REPORT_TIMEOUT_SECONDS, UsageReporter


class ReportServer:
    """A real HTTP server standing in for kwami-lk-api."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.paths: list[str] = []
        self._runner: web.AppRunner | None = None
        self.url = ""

    async def start(self, *, status: int = 200, body: Any = None, delay: float = 0.0) -> None:
        async def handler(request: web.Request) -> web.Response:
            self.paths.append(request.path)
            self.headers.append(dict(request.headers))
            self.requests.append(await request.json())
            if delay:
                await asyncio.sleep(delay)
            if status == 200:
                return web.json_response(body or {})
            return web.Response(status=status, text=body or "upstream exploded")

        app = web.Application()
        app.router.add_post("/{tail:.*}", handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = self._runner.addresses[0][1]
        self.url = f"http://127.0.0.1:{port}"

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


@pytest.fixture
async def server():
    s = ReportServer()
    yield s
    await s.stop()


def _billable_tracker() -> UsageTracker:
    tracker = UsageTracker()
    tracker.record_external_usage("search", "tavily/search", units_used=1.0)
    assert tracker.has_usage
    return tracker


# =============================================================================
# Construction
# =============================================================================


def test_credentials_come_from_settings_when_not_passed(env_setting) -> None:
    env_setting("KWAMI_API_URL", "https://api.example.test")
    env_setting("KWAMI_API_KEY", "from-settings")

    reporter = UsageReporter()

    assert reporter._api_url == "https://api.example.test"
    assert reporter._api_key == "from-settings"


def test_explicit_arguments_win_over_settings(env_setting) -> None:
    """The DI seam session.py is supposed to use."""
    env_setting("KWAMI_API_URL", "https://api.example.test")
    env_setting("KWAMI_API_KEY", "from-settings")

    reporter = UsageReporter(api_url="https://override.test", api_key="explicit")

    assert reporter._api_url == "https://override.test"
    assert reporter._api_key == "explicit"


# =============================================================================
# Early returns
# =============================================================================


async def test_a_session_with_no_usage_reports_success_without_calling_out(
    caplog,
) -> None:
    """Nothing to bill is not a failure; returning False would make the caller
    log an error on every trivial session."""
    reporter = UsageReporter(api_url="http://127.0.0.1:1", api_key="k")

    with caplog.at_level(logging.INFO):
        assert await reporter.report("user", "room", UsageTracker()) is True

    assert "No usage to report" in caplog.text


async def test_a_missing_api_key_refuses_to_report(caplog) -> None:
    """The warning has to say the money is being lost, not just that a key is
    absent."""
    reporter = UsageReporter(api_url="http://127.0.0.1:1", api_key="")

    with caplog.at_level(logging.WARNING):
        assert await reporter.report("user", "room", _billable_tracker()) is False

    assert "will not be billed" in caplog.text


# =============================================================================
# The real request
# =============================================================================


async def test_a_successful_report_posts_the_usage_and_returns_true(server) -> None:
    await server.start(body={"total_credits_charged": 42, "new_balance": 958})
    reporter = UsageReporter(api_url=server.url, api_key="secret-key")

    assert await reporter.report("user-1", "room-1", _billable_tracker()) is True

    assert server.paths == ["/credits/usage/report"]
    assert server.requests[0]["user_id"] == "user-1"
    assert server.requests[0]["session_id"] == "room-1"
    assert server.requests[0]["usage"][0]["model_id"] == "tavily/search"


async def test_the_api_key_travels_in_the_x_api_key_header(server) -> None:
    await server.start()
    reporter = UsageReporter(api_url=server.url, api_key="secret-key")

    await reporter.report("user-1", "room-1", _billable_tracker())

    assert server.headers[0]["X-API-Key"] == "secret-key"
    assert server.headers[0]["Content-Type"] == "application/json"


async def test_a_successful_report_logs_the_charge_and_balance(server, caplog) -> None:
    await server.start(body={"total_credits_charged": 42, "new_balance": 958})
    reporter = UsageReporter(api_url=server.url, api_key="k")

    with caplog.at_level(logging.INFO):
        await reporter.report("user-1", "room-1", _billable_tracker())

    assert "charged=42" in caplog.text
    assert "new_balance=958" in caplog.text


async def test_a_response_missing_the_billing_fields_still_succeeds(server, caplog) -> None:
    """`result.get(..., 0)` -- a thin 200 is still a 200."""
    await server.start(body={})
    reporter = UsageReporter(api_url=server.url, api_key="k")

    with caplog.at_level(logging.INFO):
        assert await reporter.report("user-1", "room-1", _billable_tracker()) is True

    assert "charged=0" in caplog.text


async def test_a_non_200_is_reported_as_a_failure_with_the_body(server, caplog) -> None:
    await server.start(status=500, body="database is on fire")
    reporter = UsageReporter(api_url=server.url, api_key="k")

    with caplog.at_level(logging.ERROR):
        assert await reporter.report("user-1", "room-1", _billable_tracker()) is False

    assert "HTTP 500" in caplog.text
    assert "database is on fire" in caplog.text


async def test_an_unreachable_api_is_swallowed_and_logged(caplog) -> None:
    """Port 1 refuses instantly. A raise here would abort the shutdown callback
    and take the browser teardown with it."""
    reporter = UsageReporter(api_url="http://127.0.0.1:1", api_key="k")

    with caplog.at_level(logging.ERROR):
        assert await reporter.report("user-1", "room-1", _billable_tracker()) is False

    assert "Failed to send usage report" in caplog.text


async def test_a_slow_api_hits_the_timeout_rather_than_hanging(server, caplog) -> None:
    """The worker gives shutdown ~10s. An unbounded aiohttp wait (300s default)
    meant the process was killed and the session's revenue lost."""
    await server.start(delay=REPORT_TIMEOUT_SECONDS + 1.5)
    reporter = UsageReporter(api_url=server.url, api_key="k")

    loop = asyncio.get_running_loop()
    started = loop.time()
    with caplog.at_level(logging.ERROR):
        assert await reporter.report("user-1", "room-1", _billable_tracker()) is False
    elapsed = loop.time() - started

    assert elapsed < REPORT_TIMEOUT_SECONDS + 1.0
    assert "Failed to send usage report" in caplog.text
