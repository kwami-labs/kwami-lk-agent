"""Usage reporter - sends accumulated usage data to the API.

Called when a session ends to report AI resource consumption
so the credits system can deduct from the user's balance.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import aiohttp

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from .tracker import UsageTracker

logger = get_logger("usage.reporter")

# API configuration from environment
API_BASE_URL = os.environ.get("KWAMI_API_URL", "http://localhost:8080")
KWAMI_API_KEY = os.environ.get("KWAMI_API_KEY", "")


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
        self._api_url = api_url or API_BASE_URL
        self._api_key = api_key or KWAMI_API_KEY

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
            logger.info(f"No usage to report for session {session_id}")
            return True

        usage_summary = tracker.get_usage_summary()
        duration = tracker.session_duration_seconds

        logger.info(
            f"Reporting usage for session {session_id}: "
            f"{len(usage_summary)} models, {duration:.1f}s session"
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

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(
                            f"Usage reported successfully: "
                            f"charged={result.get('total_credits_charged', 0)} micro-credits, "
                            f"new_balance={result.get('new_balance', 0)}"
                        )
                        return True
                    else:
                        body = await resp.text()
                        logger.error(f"Usage report failed (HTTP {resp.status}): {body}")
                        return False

        except Exception as e:
            logger.error(f"Failed to send usage report: {e}")
            return False
