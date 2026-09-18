# Testing

Five layers under `agent/tests/`. The default pytest run is **offline, free,
and fast**. Live tests are opted in.

```mermaid
flowchart TB
    subgraph Offline["make test / CI (every PR)"]
        U["unit/<br/>pure domain, safety, settings"]
        C["contract/<br/>real installed SDKs"]
        I["integration/<br/>real Agent + mocked transports"]
        R["runtime/<br/>pipeline + lifecycle"]
    end

    subgraph Live["make test-e2e / nightly"]
        E["e2e/  mark: live<br/>real providers, real keys"]
    end

    U --> C --> I --> R
    R -.-> E
```

`pyproject.toml` sets `addopts = "-m 'not live'"`. `make test-e2e` passes
`-m live`. An empty live selection exits 5; the Makefile and the e2e workflow
treat that as a warning, not a failure.

## Layers

### `unit/`

No network, no LiveKit worker. Config parsing, prompt assembly, usage maths,
URL safety, Settings, dispatch, cloning, room helpers.

Good unit tests construct `Settings(...)` and `KwamiConfig(...)` directly.
They do not depend on what the developer exported.

### `contract/`

This codebase's assumptions checked against the **real** installed packages:

- Hooks we override exist with the signatures we use (`on_enter` takes no room)
- Every advertised provider constructs
- Every Zep method we call is a real method

**Never stub `livekit` or `zep_cloud`.** An earlier `conftest.py` replaced both
with `MagicMock`, which agreed with every wrong assumption: a bad `on_enter`
signature, a hook the framework never dispatches, and five nonexistent Zep
methods passed for months.

### `integration/`

A real `livekit.agents.Agent` with mocked transports. Covers config wire
parsing, tool publishing, client-tool results, browser-tool safety, memory
reuse across config updates.

### `runtime/`

Pipeline construction (including the realtime branch) and lifecycle helpers
(`apply_runtime_config`, `resolve_identity_on_join`) without standing up a
worker.

### `e2e/` (`live`)

Conversation, memory, and reconfiguration against real providers. Spends
money. Runs on a nightly cron, `workflow_dispatch`, and push to `main` — not
as a pull-request gate.

## Coverage

```bash
make test-cov
```

`fail_under` in `pyproject.toml` is a ratchet (currently 43). It only moves
up. Branch coverage is on. `if TYPE_CHECKING` and `@overload` are excluded.

CI uploads `coverage.xml` per Python version (3.11 and 3.13).

## Writing a test

1. Prefer a **fake** that implements a port over `MagicMock`.
2. If you call a LiveKit or Zep API, import the real package.
3. Put provider-hitting tests under `e2e/` and mark them `@pytest.mark.live`.
4. Reset Settings between tests that change env (`set_settings(None)` / fixture).
5. Do not add sleeps to hide races; serialize through the same locks production uses.

## What CI runs

| Workflow | When | Jobs |
| --- | --- | --- |
| `.github/workflows/test.yml` | PR and push to `dev` / `stg` / `main` | Ruff, mypy, pytest+coverage on 3.11 and 3.13, Docker build |
| `.github/workflows/e2e.yml` | Nightly 03:00 UTC, `main`, manual | Live suite with GitHub `e2e` environment secrets |

Offline CI sets `UV_FROZEN=1` and does not inject provider keys, so a test
that accidentally authenticates fails loudly.
