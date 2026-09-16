"""The room-publish boundary.

Tools push structured payloads to the frontend over the LiveKit data channel.
That channel has a hard size limit, so trimming is part of the contract rather
than something each caller reinvents -- there were three slightly different
trimming policies before this existed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# LiveKit rejects oversized data packets; stay under it with headroom for the
# envelope the SDK adds.
MAX_DATA_PACKET_BYTES = 14_000


@runtime_checkable
class RoomPublisherPort(Protocol):
    """Publishes a JSON-serialisable payload to the participants in a room."""

    async def publish(self, payload: dict[str, Any], *, topic: str | None = None) -> bool:
        """Send a payload, trimming it to fit if necessary.

        Returns False rather than raising when there is no room to publish to:
        a tool must still be able to answer the user when the UI is absent.
        """
        ...
