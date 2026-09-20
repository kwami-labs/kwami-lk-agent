"""Usage reporter - sends accumulated usage data to the API.

Called when a session ends to report AI resource consumption
so the credits system can deduct from the user's balance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiohttp

from ..settings import get_settings
from ..utils.logging import get_logger

if TYPE_CHECKING:
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
