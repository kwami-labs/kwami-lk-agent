#!/bin/sh
set -eu

uv run python /app/health.py &
HEALTH_PID=$!

uv run python -m src.main start &
AGENT_PID=$!

shutdown() {
  kill -TERM "$AGENT_PID" 2>/dev/null || true
  kill -TERM "$HEALTH_PID" 2>/dev/null || true
  wait "$AGENT_PID" 2>/dev/null || true
  wait "$HEALTH_PID" 2>/dev/null || true
  exit 0
}

trap shutdown TERM INT

while kill -0 "$AGENT_PID" 2>/dev/null && kill -0 "$HEALTH_PID" 2>/dev/null; do
  sleep 1
done

if ! kill -0 "$AGENT_PID" 2>/dev/null; then
  status=0
  wait "$AGENT_PID" || status=$?
  kill -TERM "$HEALTH_PID" 2>/dev/null || true
  exit "$status"
fi

kill -TERM "$AGENT_PID" 2>/dev/null || true
exit 1
