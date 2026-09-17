# Cloudflare Containers image for the Kwami LiveKit worker.
# Build context is the repository root (see infra/wrangler.jsonc image_build_context).
# Keep the install/prewarm steps aligned with agent/Dockerfile.
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

ENV PYTHONUNBUFFERED=1

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY agent/pyproject.toml agent/uv.lock ./
RUN uv sync --locked

COPY agent/ ./
COPY infra/container/health.py infra/container/start.sh ./
RUN chmod +x /app/start.sh \
  && chown -R appuser:appuser /app

USER appuser

RUN uv run python -m src.main download-files

EXPOSE 8080
CMD ["./start.sh"]
