"""Usage reporter - sends accumulated usage data to the API.

Called when a session ends to report AI resource consumption
so the credits system can deduct from the user's balance.

Without a Kwami API (LiveKit Cloud writing the user's Supabase directly),
`SupabaseUsageReporter` records the same summary as rows in `agent_usage`:
counted, not billed. `create_usage_reporter()` picks one.
"""

from __future__ import annotations

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


def create_usage_reporter(settings: Settings | None = None) -> Any:
    """`agent_usage` in Supabase when there is no Kwami API, the API otherwise."""
    from ..adapters.supabase import direct_supabase

    store = direct_supabase(settings)
    if store is not None:
        return SupabaseUsageReporter(store)
    return UsageReporter()
