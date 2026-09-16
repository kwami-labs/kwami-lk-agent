"""Concrete implementations of the ports: the only code that touches the world.

Each adapter is a thin translation between a port's vocabulary and one external
system. They take their dependencies (an HTTP client, Settings) by argument
rather than reaching for a module global, which is what lets the layers above
them be tested without a network or a mocking library.

Nothing here is imported by `domain/`. The dependency arrow points inward.
"""

from .http import HttpxClient, HttpxResponse
from .publisher import LiveKitRoomPublisher, NullRoomPublisher

__all__ = [
    "HttpxClient",
    "HttpxResponse",
    "LiveKitRoomPublisher",
    "NullRoomPublisher",
]
