# Cloudflare Containers image for the Kwami LiveKit worker.
#
# Same image as agent/Dockerfile, built from the repository root instead of from
# agent/ (see `image_build_context` in infra/wrangler.jsonc) and starting
# start.sh, which runs the health probe alongside the worker.
#
# The two files are checked against each other by
# agent/tests/unit/test_dockerfile_parity.py: base image, Python version,
# non-root uid and the install/prewarm steps have to match, because an image
# that diverges from the one CI builds is an image nobody has tested.
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13

# --- builder -----------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY agent/pyproject.toml agent/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# --- runtime -----------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

COPY --chown=appuser:appuser agent/ ./
COPY --chown=appuser:appuser infra/container/health.py infra/container/start.sh ./
RUN chmod +x /app/start.sh

USER appuser

RUN python -m src.main download-files

EXPOSE 8080
CMD ["./start.sh"]
