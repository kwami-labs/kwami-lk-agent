# Production image for the Kwami LiveKit agent.
#
# Multi-stage. The single-stage version shipped gcc, g++ and python3-dev in the
# runtime image: they are needed to build wheels for packages with native
# extensions and needed by nothing at run time. Carrying them costs image size
# on every pull and, more to the point, keeps a toolchain inside a container
# that runs model-chosen code paths and holds a dozen provider credentials.
#
# Keep in step with infra/container/Dockerfile, which builds the same image from
# the repository root for Cloudflare Containers. The two are checked against
# each other by agent/tests/unit/test_dockerfile_parity.py -- a comment asking
# people to keep two files aligned is not a mechanism.
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13

# --- builder -----------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Build dependencies for packages with native extensions. Present here, absent
# from the runtime stage below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a source-only change does not reinstall them.
COPY pyproject.toml uv.lock ./
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

# The resolved environment, without the toolchain that produced it.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

COPY --chown=appuser:appuser . .

USER appuser

# Pre-download the VAD model so the first session does not wait on it.
RUN python -m src.main download-files

# The worker connects to LiveKit and waits for jobs.
CMD ["python", "-m", "src.main", "start"]
