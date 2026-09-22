"""Health probe for Cloudflare Containers, reporting the *worker's* health.

This used to answer `{"status": "ok"}` unconditionally, which meant the only
thing it could ever report was whether its own HTTP thread was alive. A LiveKit
worker that was running but not registered -- a dropped websocket it never
recovered from, an auth failure loop, a wedged event loop -- looked exactly like
one happily taking calls: 200 here, 200 from the Worker's `/health`, and a
two-minute keepalive cron confirming every time that nothing was wrong.

It now reads the heartbeat that `agent/src/health.py` writes on
`worker_registered` and refreshes on a timer, and reports three states:

* **starting** (200) -- no heartbeat yet, still inside the grace period.
  Deliberately 200: `startAndWaitForPorts` waits on this port, so failing here
  would stop the container ever coming up.
* **ok** (200) -- heartbeat present and fresh.
* **degraded** (503) -- the worker never registered within the grace period, or
  it registered and then stopped beating.

Stdlib only, on purpose: this runs as a second process next to the agent (see
start.sh) and must not depend on the agent's virtualenv resolving.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "0.0.0.0"  # noqa: S104 - a container port, published to the Cloudflare runtime
PORT = 8080

#: Must match `HEARTBEAT_PATH_ENV` / `DEFAULT_HEARTBEAT_PATH` in
#: agent/src/health.py. Pinned by tests/unit/test_health_heartbeat.py, because
#: two constants that must agree across a process boundary are exactly the kind
#: of pair that silently drifts.
HEARTBEAT_PATH_ENV = "KWAMI_HEARTBEAT_FILE"
DEFAULT_HEARTBEAT_PATH = Path(tempfile.gettempdir()) / "kwami-agent" / "heartbeat.json"

#: The agent beats every 10s. Three missed beats is a real stall rather than one
#: slow write or a GC pause.
STALE_AFTER_SECONDS = 35.0

#: How long the worker is allowed to take to register before absence of a
#: heartbeat counts as failure. Model download and prewarm happen first, and on
#: a cold image that is not quick.
STARTUP_GRACE_SECONDS = 180.0

#: When this process started, which is when the grace period begins.
STARTED_AT = time.monotonic()


def heartbeat_path() -> Path:
    override = os.environ.get(HEARTBEAT_PATH_ENV, "").strip()
    return Path(override) if override else DEFAULT_HEARTBEAT_PATH


def read_heartbeat(path: Path) -> dict | None:
    """The last beat, or None if there is not a readable one.

    A half-written or corrupt file reads as absent rather than raising: the
    agent writes atomically, so this should not happen, and if it does the
    honest answer is "no usable heartbeat".
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def assess(*, now: float | None = None, uptime: float | None = None) -> tuple[int, dict]:
    """Decide the health status. Returns (http_status, body)."""
    wall = time.time() if now is None else now
    since_start = (time.monotonic() - STARTED_AT) if uptime is None else uptime

    beat = read_heartbeat(heartbeat_path())

    if beat is None:
        if since_start < STARTUP_GRACE_SECONDS:
            return 200, {
                "status": "starting",
                "service": "kwami-lk-agent",
                "detail": "waiting for the LiveKit worker to register",
            }
        return 503, {
            "status": "degraded",
            "service": "kwami-lk-agent",
            "detail": (
                f"no worker heartbeat after {STARTUP_GRACE_SECONDS:.0f}s; "
                "the agent process is up but never registered with LiveKit"
            ),
        }

    age = wall - float(beat.get("ts", 0) or 0)
    if age > STALE_AFTER_SECONDS:
        return 503, {
            "status": "degraded",
            "service": "kwami-lk-agent",
            "detail": f"worker heartbeat is {age:.0f}s old (stale after {STALE_AFTER_SECONDS:.0f}s)",
            "worker": beat.get("state", "unknown"),
        }

    return 200, {
        "status": "ok",
        "service": "kwami-lk-agent",
        "worker": beat.get("state", "unknown"),
        "heartbeat_age_seconds": round(max(0.0, age), 1),
    }


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] not in {"/", "/health", "/ready"}:
            self.send_response(404)
            self.end_headers()
            return
        status, payload = assess()
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), HealthHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
