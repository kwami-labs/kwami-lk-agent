"""Protocol definitions for every I/O boundary this agent crosses.

These exist so that logic can be tested against a fake rather than a mock. The
difference matters here: this codebase shipped ten critical bugs behind
`MagicMock`, because a mock agrees with whatever the caller assumes. A
`Protocol` states the contract once, and both the real adapter and the test
fake are checked against it.

Ports are structural (`typing.Protocol`), so adapters do not import or subclass
them -- the dependency arrow points from the runtime into the port, never from
the adapter back out.
"""

from .billing import UsageReporterPort
from .browser import BrowserPort
from .http import HttpClientPort, HttpResponsePort
from .memory import MemoryContextPort, MemoryPort
from .publisher import RoomPublisherPort
from .search import SearchPort, SearchResult

__all__ = [
    "BrowserPort",
    "HttpClientPort",
    "HttpResponsePort",
    "MemoryContextPort",
    "MemoryPort",
    "RoomPublisherPort",
    "SearchPort",
    "SearchResult",
    "UsageReporterPort",
]
