"""Publishing structured payloads to the frontend over the LiveKit data channel.

There were three separate trimming policies for this in `tools/builtin.py`, each
slightly different, and one of them re-serialised the whole message up to three
times just to measure its size. The channel has a hard size limit, so trimming
belongs in the adapter that knows about the limit -- not in each caller.
"""

from __future__ import annotations

import json
from typing import Any

from ..ports.publisher import MAX_DATA_PACKET_BYTES
from ..utils.logging import get_logger

logger = get_logger("publisher")

# Trimming runs in stages, cheapest loss first. Images survive as long as
# possible because they are the point of a product card; prose is recoverable
# from the agent's spoken answer, a missing thumbnail is not.
_TRIM_STAGES: tuple[tuple[str, int], ...] = (
    ("content", 120),
    ("content", 60),
)


class NullRoomPublisher:
    """Used when there is no room. Records payloads so tools remain testable."""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def publish(self, payload: dict[str, Any], *, topic: str | None = None) -> bool:
        self.published.append(payload)
        return False


class LiveKitRoomPublisher:
    """Publishes to a LiveKit room, trimming to fit the data-channel limit."""

    def __init__(self, room: Any, *, max_bytes: int = MAX_DATA_PACKET_BYTES) -> None:
        self._room = room
        self._max_bytes = max_bytes

    def set_room(self, room: Any) -> None:
        self._room = room

    async def publish(self, payload: dict[str, Any], *, topic: str | None = None) -> bool:
        """Send a payload, shrinking it if it exceeds the channel limit.

        Returns False rather than raising when there is no room: a tool must
        still be able to answer the user when the UI is absent.
        """
        participant = getattr(getattr(self._room, "local_participant", None), "publish_data", None)
        if participant is None:
            logger.debug("No room to publish to; dropping %s payload", payload.get("type"))
            return False

        encoded = self._encode_within_limit(payload)
        if encoded is None:
            logger.warning(
                "Could not trim %s payload under %d bytes; not published",
                payload.get("type"),
                self._max_bytes,
            )
            return False

        try:
            await participant(encoded, reliable=True)
            return True
        except Exception as e:
            logger.warning("Failed to publish %s payload: %s", payload.get("type"), e)
            return False

    def _encode_within_limit(self, payload: dict[str, Any]) -> bytes | None:
        """Serialise once, then trim progressively only while still too large."""
        encoded = json.dumps(payload).encode("utf-8")
        if len(encoded) <= self._max_bytes:
            return encoded

        trimmed = json.loads(json.dumps(payload))  # don't mutate the caller's dict
        results = trimmed.get("results")
        if not isinstance(results, list):
            return None

        for key, limit in _TRIM_STAGES:
            for item in results:
                if isinstance(item, dict) and isinstance(item.get(key), str):
                    item[key] = item[key][:limit]
                if isinstance(item, dict) and isinstance(item.get("features"), list):
                    item["features"] = item["features"][:3]
            if isinstance(trimmed.get("answer"), str):
                trimmed["answer"] = trimmed["answer"][:150]
            encoded = json.dumps(trimmed).encode("utf-8")
            if len(encoded) <= self._max_bytes:
                return encoded

        # Last resort: images are the largest field and the only one left.
        for item in results:
            if isinstance(item, dict):
                item.pop("image", None)
        encoded = json.dumps(trimmed).encode("utf-8")
        return encoded if len(encoded) <= self._max_bytes else None
