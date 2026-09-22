"""A heartbeat the container's health probe can actually fail on.

`infra/container/health.py` answered `{"status": "ok"}` whenever its own HTTP
thread was alive. It never asked whether the LiveKit worker beside it was
registered, so the only failure it could report was its own. `start.sh` catches
a hard crash -- the container exits and Cloudflare restarts it -- which leaves
the case that actually matters uncovered: a worker process that is *running* but
not registered. A dropped websocket it never recovers from, an auth failure loop,
a wedged event loop. In all three the container reported healthy, the Worker's
`/health` returned 200, and the two-minute keepalive cron happily confirmed that
nothing was wrong while no call could be answered.

So the worker writes a heartbeat and the probe reads it. Two separate facts are
recorded, because they fail differently:

* **registered** -- the worker completed registration with LiveKit at least
  once. Absent for longer than the startup grace means it never came up.
* **freshness** -- the heartbeat is being rewritten on a timer. A stale one
  means the loop stopped: the process is alive and not working.

Deliberately a file rather than a socket or a shared memory segment: the probe
is a separate process started by `start.sh`, stdlib-only by design, and a file
is the one IPC both ends can use without adding a dependency to either.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

from .utils.logging import get_logger

logger = get_logger("health")

#: Overridable so a test never writes to the real path and two workers on one
#: host do not fight over the same file.
HEARTBEAT_PATH_ENV = "KWAMI_HEARTBEAT_FILE"

DEFAULT_HEARTBEAT_PATH = Path(tempfile.gettempdir()) / "kwami-agent" / "heartbeat.json"

#: How often the beat is refreshed. Comfortably inside the probe's staleness
#: window below, so one slow write does not read as a dead worker.
HEARTBEAT_INTERVAL_SECONDS = 10.0


def heartbeat_path() -> Path:
    """Where the heartbeat lives, honouring the environment override."""
    override = os.environ.get(HEARTBEAT_PATH_ENV, "").strip()
    return Path(override) if override else DEFAULT_HEARTBEAT_PATH


def write_heartbeat(state: str, *, now: float | None = None) -> bool:
    """Record one beat. Returns whether it was written.

    Never raises. A read-only filesystem or a full disk must not take down a
    worker that is otherwise serving calls -- the probe will report the staleness
    and that is the correct outcome, but it should not be *caused* here.

    Written via a temporary file and an atomic replace, so a probe reading
    concurrently sees either the old beat or the new one, never a half-written
    file it cannot parse.
    """
    path = heartbeat_path()
    payload = json.dumps({"state": state, "ts": now if now is not None else time.time()})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError as e:
        logger.warning("Could not write the health heartbeat to %s: %s", path, e)
        return False


async def run_heartbeat(
    *,
    state: str = "registered",
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
    iterations: int | None = None,
) -> None:
    """Refresh the heartbeat until cancelled.

    Args:
        state: Recorded with each beat, so the probe can tell "registered" from
            a worker that only ever got as far as starting.
        interval: Seconds between beats.
        iterations: Stop after this many beats. Only for tests -- production
            passes None and the task runs until the process goes away.
    """
    written = 0
    while iterations is None or written < iterations:
        write_heartbeat(state)
        written += 1
        if iterations is not None and written >= iterations:
            return
        await asyncio.sleep(interval)
