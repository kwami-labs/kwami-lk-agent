# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project documentation under `docs/` (architecture, protocol, memory, security, billing, configuration, development, testing, deployment) with mermaid diagrams.
- Hexagonal I/O ports (`MemoryPort`, `RoomPublisherPort`, `UsageReporterPort`, `HttpClientPort`, `BrowserPort`, `SearchPort`) and LiveKit / HTTP adapters.
- Process-wide frozen `Settings` object; credentials are no longer read from `os.environ` at import time.
- Layered test suites: unit, contract (real SDKs), integration, runtime, and live e2e.
- CI: Ruff + mypy, pytest with coverage on Python 3.11 and 3.13, Docker image build, nightly live e2e.
- Cloud-browser minute metering on every release path (idle, failed connect, user close, session cleanup).
- Live conversation, memory, and reconfiguration e2e suites.

### Changed

- Room resolution uses `AgentDeps` on `AgentSession.userdata` instead of a process-wide `ContextVar`.
- Config and usage types live in `domain/`; pipeline construction lives in `runtime/pipeline.py`.
- Data-channel routing extracted from the entrypoint into `DataMessageRouter`.
- Search results publish through `LiveKitRoomPublisher` with a single 14 KB trim policy.
- Coverage floor ratcheted to 43%.

### Fixed

- `on_enter` signature matches the framework (no `room` argument), restoring duplicate-agent detection and `self.room`.
- Assistant turns persist to Zep via `conversation_item_added` (the previous hook was never dispatched).
- Memory context injection is hard-capped at 6s so a slow Zep cannot delay the greeting.
- Zep client and thread are reused across `config` messages for the same user.
- Config handling is serialized; background tasks are retained so they cannot be garbage-collected mid-flight.
- Browser session and pending client-tool futures survive agent swaps.
- Client tool results (and errors) are capped at 4 000 characters.
- Realtime pipeline settings (`pipelineType`, `realtime*`) are parsed from the frontend.
- VAD honours configured turn-taking and reuses the prewarmed Silero model.
- UI-control system-prompt guidance is emitted only when those client tools are registered.
- Usage tracker treats request-count-only entries and token-only realtime turns as billable.
- OpenAI TTS fallback is recorded so later voice updates validate against the live provider.
- Builtin `TimeoutError` is caught on memory injection (not only `asyncio.TimeoutError` aliases).

### Security

- Browser URL validation rejects non-HTTP schemes (including `javascript:` / `data:`), loopback, private ranges, link-local metadata, and DNS that resolves to those addresses.
- Arbitrary in-browser JavaScript is disabled unless `KWAMI_ALLOW_BROWSER_JS` is set.
- A cloud browser is never started without a real `kwami_id` (per-user profiles).
- Data-channel decode never raises; malformed packets are discarded.

## [0.1.0]

Initial LiveKit Cloud agent for Kwami AI.

- Runtime-configurable soul, STT / LLM / TTS (or realtime) pipeline, and client tools over the data channel.
- Zep Cloud memory with a custom ontology (Person, Project, Product, Goal, Procedure).
- Built-in tools: voice controls, web/product search, cloud browsing.
- Usage reporting to kwami-lk-api for the credits system.
- Telephony bootstrap from `GET /internal/kwamis/{id}/runtime`.
- Docker image and LiveKit Cloud deploy via `lk agent deploy`.

[Unreleased]: https://github.com/kwami-labs/kwami-lk-agent/commits/dev
[0.1.0]: https://github.com/kwami-labs/kwami-lk-agent/blob/dev/agent/pyproject.toml
