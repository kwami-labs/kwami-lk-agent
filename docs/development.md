# Development

## Prerequisites

- Python **3.11** locally (`.python-version`). The production image is **3.13**.
  CI runs both.
- [uv](https://docs.astral.sh/uv/) for installs and the lockfile
- A LiveKit Cloud project and the provider keys you actually want to exercise
- Optional: [LiveKit CLI](https://docs.livekit.io/home/cli/) (`lk`) for deploy

This repo uses **uv**, not pnpm/npm. `agent/uv.lock` is the source of truth.

## First run

```bash
cp .env.sample .env
# fill LIVEKIT_* and at least OPENAI_API_KEY, DEEPGRAM_API_KEY

make install    # cd agent && uv sync --extra dev
make dev        # cd agent && uv run python -m src.main dev
```

`make dev` starts the LiveKit agents CLI in dev mode against `LIVEKIT_URL`.
Join a room from the playground or SDK; the worker logs
`Kwami session starting in room: …`.

## Commands

| Command | What it does |
| --- | --- |
| `make install` | `uv sync --extra dev` |
| `make dev` | Local worker (`python -m src.main dev`) |
| `make test` | Offline suite (unit + contract + integration + runtime) |
| `make test-unit` / `test-contract` / `test-integration` | One layer |
| `make test-cov` | Offline suite + terminal and HTML coverage |
| `make test-e2e` | Live tests (`-m live`); needs real keys |
| `make lint` | Ruff check + format check |
| `make format` | Ruff format + fix |
| `make typecheck` | mypy on `src` |
| `make check` | lint + typecheck + test (what CI quality+test approximate) |
| `make create` | `lk agent create .` (first deploy only) |
| `make deploy` | `lk agent deploy` |
| `make clean` | caches, `.venv`, coverage artifacts |

Working directory for uv/pytest is always `agent/`. The Python package name is
`src`, matching the Docker image.

## Layout conventions

- **Domain is pure.** `domain/` must not import provider SDKs or read
  `os.environ`. If you need a credential, take `Settings`.
- **Ports before mocks.** New I/O gets a `typing.Protocol` in `ports/` and a
  fake in tests. Do not stub `livekit` or `zep_cloud`.
- **Settings once.** Add new env vars to `Settings.from_env()`, `.env.sample`,
  and [configuration.md](./configuration.md).
- **Spawn, don't fire-and-forget.** Use `SessionState.spawn` (or retain the
  task on a set). Bare `create_task` can be collected mid-flight.
- **Serialize config.** Config handlers run under `state.run_serialized`.
- **Cap model-bound strings.** Anything that enters the prompt needs a size
  limit (see [security.md](./security.md)).
- **No import cycle through `runtime`.** Import `create_agent_from_config`
  from `runtime.pipeline`, not from `runtime`.

Ruff: line length 100, py311 target, rules `E F I N W UP`. `E501` is ignored.
`src/main.py` is allowed `E402` so `load_dotenv` runs before LiveKit imports.

mypy starts permissive (`ignore_missing_imports`). Modules still on
`ignore_errors` are listed in `pyproject.toml` and that list only shrinks.

## Optional provider extras

```bash
cd agent
uv sync --extra dev --extra anthropic --extra google
```

Without the extra, the corresponding factory falls back to OpenAI.

## Debugging a session

1. `make dev` and watch the worker log for `Settings resolved:` (keys redacted).
2. Confirm the frontend sent `config` (`Received data message: config`).
3. Greeting should happen after memory's 6s budget, not after a Zep timeout.
4. On hang, check whether a client tool is waiting 30s for `tool_result`.
5. On missing billing, check `user_identity` and `KWAMI_API_*`.

## Related docs

- [Architecture](./architecture.md) — how the pieces fit
- [Testing](./testing.md) — how to add a test that cannot lie
- [Contributing](../CONTRIBUTING.md) — PR expectations
