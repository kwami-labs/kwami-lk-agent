"""Usage reporter - sends accumulated usage data to the API.

Called when a session ends to report AI resource consumption
so the credits system can deduct from the user's balance.

Where the agent writes the user's Supabase directly (LiveKit Cloud),
`SupabaseUsageReporter` records the same summary as rows in `agent_usage`:
counted, not billed. `BilledUsageReporter` does both, so writing Supabase directly
does not stop a session being charged. `create_usage_reporter()` picks one.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import aiohttp

from ..settings import Settings, get_settings
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..adapters.supabase import SupabaseClient
    from ..domain import UsageTracker

logger = get_logger("usage.reporter")

# Kept below SessionState.USAGE_REPORT_TIMEOUT_SECONDS so the inner call
# fails first and we log the HTTP cause rather than a bare cancellation.
REPORT_TIMEOUT_SECONDS = 4.0
# What is left of that budget for finding who to bill.
OWNER_LOOKUP_SECONDS = 0.75


class UsageReporter:
    """Reports accumulated usage to the Kwami API credits endpoint.

    Sends a single POST request at the end of a session with all
    usage data so credits can be deducted atomically.
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        # Resolved here rather than at import time: a module-level snapshot
        # cannot be overridden per test, and any entry point importing before
        # load_dotenv silently reported no billing at all.
        settings = get_settings()
        self._api_url = api_url or settings.kwami_api_url
        self._api_key = api_key or settings.kwami_api_key

    async def report(
        self,
        user_id: str,
        session_id: str,
        tracker: UsageTracker,
        *,
        kwami_id: str | None = None,
    ) -> bool:
        """Send usage report to the API.

        Args:
            user_id: The Supabase user ID.
            session_id: The LiveKit room name.
            tracker: UsageTracker with accumulated metrics.

        Returns:
            True if the report was sent successfully.
        """
        if not tracker.has_usage:
            logger.info("No usage to report for session %s", session_id)
            return True

        usage_summary = tracker.get_usage_summary()
        duration = tracker.session_duration_seconds

        logger.info(
            "Reporting usage for session %s: %s models, %.1fs session",
            session_id,
            len(usage_summary),
            duration,
        )

        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "usage": usage_summary,
            "session_minutes": round(duration / 60.0, 3),
        }

        if not self._api_key:
            logger.warning(
                "KWAMI_API_KEY not set, skipping usage report. Usage will not be billed."
            )
            return False

        try:
            url = f"{self._api_url}/credits/usage/report"
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self._api_key,
            }

            # aiohttp defaults to a 300s total timeout. This call runs inside a
            # shutdown callback with a ~10s worker budget, so an unbounded wait
            # meant the process was killed and the session's revenue lost.
            timeout = aiohttp.ClientTimeout(total=REPORT_TIMEOUT_SECONDS)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(url, json=payload, headers=headers) as resp,
            ):
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(
                        "Usage reported successfully: charged=%s micro-credits, new_balance=%s",
                        result.get("total_credits_charged", 0),
                        result.get("new_balance", 0),
                    )
                    return True
                else:
                    body = await resp.text()
                    logger.error("Usage report failed (HTTP %s): %s", resp.status, body)
                    return False

        except Exception:
            logger.exception("Failed to send usage report")
            return False


#: Summary keys that become columns; the rest of each item goes to `details`.
_COLUMN_KEYS = frozenset({"model_type", "model_id", "units_used"})


def _split_model_id(model_id: str) -> tuple[str, str]:
    """`"openai/gpt-5.2"` -> ("openai", "gpt-5.2"). No slash: no provider."""
    provider, sep, model = model_id.partition("/")
    return (provider, model) if sep and model else ("", model_id)


class SupabaseUsageReporter:
    """Records a session's usage in the user's `agent_usage` table.

    One row per model, written once when the session ends. The owner is read
    from the kwami's row when there is one: the participant identity this is
    handed can be the kwami id itself (see `config_handler`), and a uuid that
    is not a user would fail the foreign key and lose every row.
    """

    def __init__(self, store: SupabaseClient) -> None:
        self._store = store

    async def report(
        self,
        user_id: str,
        session_id: str,
        tracker: UsageTracker,
        *,
        kwami_id: str | None = None,
    ) -> bool:
        from ..adapters.supabase import as_user_id, kwami_owner

        if not tracker.has_usage:
            logger.info("No usage to record for session %s", session_id)
            return True

        owner = await kwami_owner(self._store, kwami_id) if kwami_id else None
        rows: list[dict[str, Any]] = []
        for item in tracker.get_usage_summary():
            provider, model = _split_model_id(str(item.get("model_id") or "unknown"))
            rows.append(
                {
                    "user_id": owner or as_user_id(user_id),
                    "kwami_id": kwami_id or None,
                    "session_id": session_id,
                    "provider": provider,
                    "model": model,
                    "kind": item.get("model_type"),
                    "units": float(item.get("units_used") or 0.0),
                    "details": {k: v for k, v in item.items() if k not in _COLUMN_KEYS},
                }
            )
        stored = await self._store.insert("agent_usage", rows, what="the session's usage")
        if stored:
            logger.info("Usage recorded in Supabase: %d row(s) for %s", len(rows), session_id)
        return stored


class BilledUsageReporter:
    """Charges the session through the Kwami API, and keeps the `agent_usage` copy.

    The copy never decides the result: a session is reported when it was charged.
    The bill goes to the kwami's owner, as the copy does: the identity handed over
    can be the kwami id, or a phone caller's.
    """

    def __init__(self, billing: UsageReporter, store: SupabaseClient) -> None:
        self._billing = billing
        self._store = store
        self._copy = SupabaseUsageReporter(store)

    async def _owner(self, kwami_id: str | None) -> str | None:
        from ..adapters.supabase import kwami_owner

        if not kwami_id:
            return None
        try:
            return await asyncio.wait_for(
                kwami_owner(self._store, kwami_id), timeout=OWNER_LOOKUP_SECONDS
            )
        except Exception:
            logger.warning("Could not read the kwami's owner; billing the session's identity")
            return None

    async def report(
        self,
        user_id: str,
        session_id: str,
        tracker: UsageTracker,
        *,
        kwami_id: str | None = None,
    ) -> bool:
        owner = await self._owner(kwami_id)
        billed, copied = await asyncio.gather(
            self._billing.report(owner or user_id, session_id, tracker, kwami_id=kwami_id),
            self._copy.report(user_id, session_id, tracker, kwami_id=kwami_id),
            return_exceptions=True,
        )
        if isinstance(copied, BaseException):
            logger.warning("Could not keep the usage copy: %s", type(copied).__name__)
        if isinstance(billed, BaseException):
            logger.error("Could not bill session %s: %s", session_id, type(billed).__name__)
            return False
        return billed


def create_usage_reporter(settings: Settings | None = None) -> Any:
    """The API when there is one, with a copy in `agent_usage` when writing Supabase too."""
    from ..adapters.supabase import direct_supabase

    settings = settings or get_settings()
    store = direct_supabase(settings)
    if store is None:
        return UsageReporter()
    if not settings.kwami_api_enabled:
        return SupabaseUsageReporter(store)
    billing = UsageReporter(api_url=settings.kwami_api_url, api_key=settings.kwami_api_key)
    return BilledUsageReporter(billing, store)
