# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Self-service reconfiguration tools, so the agent can do what its system prompt
  has always claimed it could: `change_ai_model` (by family name -- "Claude",
  "Gemini" -- or exact id), `switch_pipeline_mode`, `change_realtime_voice`,
  `list_available_models`, `list_available_voices`, `list_model_providers` and
  `get_pipeline_status`. A rebuild is handed back to the framework rather than
  installed inside the tool, so the sentence explaining the switch survives it.
- `Reconfigurator` on `AgentDeps`: the handle a tool uses to rebuild its own
  pipeline. Previously every path into `create_agent_from_config` started from a
  data message the frontend sent, and a tool had none.
- Capability-derived system-prompt guidance (`domain/capabilities.py`). Guidance
  is assembled per session from the client tools actually registered, one block
  per capability, instead of one fixed block naming four domains.
- `deep_research`: four concurrent search angles (overview, recent, analysis,
  critique), deduplicated by publisher, returned as a bounded briefing and
  published to the existing search-results channel.
- `get_market_quote`: exact, current prices for equities, ETFs, indices, FX and
  crypto. Read-only by design -- there is deliberately no order-placing tool; see
  `TRADING_GUIDANCE` in `tools/knowledge.py`.
- `domain/models.py`: resolves a spoken provider or model name to a
  `(provider, model)` pair, recognising families rather than a frozen catalogue.
- Media playback by voice: `play_media` opens the service, clicks the first
  result and starts it -- the step `navigate_to` cannot do, because YouTube's
  results live in a shadow DOM `read_navigation_page` cannot see. Plus
  `control_playback`, `set_playback_volume` and `get_now_playing`. The in-page
  scripts are constants in `tools/media.py`, never model-authored, so this does
  not depend on `KWAMI_ALLOW_BROWSER_JS`.
- Order placement, gated: `prepare_trade` prices and reads back an order and
  sends nothing; `open_trade_ticket` puts the user's own broker on screen;
  `submit_trade` requires the user to repeat a confirmation code *derived from
  the order itself*, so a code the model could invent, a code spoken for a
  different order, and a code that predates an edit all fail closed.
  `cancel_prepared_trade` and `get_prepared_trade` complete the state machine.
- **Browserbase as a cloud-browser vendor**, and a seam
  (`browser/providers.py`) for both it and Browser Use Cloud to sit behind.
  Selected with `KWAMI_BROWSER_PROVIDER`; if the chosen vendor has no
  credential the other is used, loudly, because the two do not share saved
  logins. Browserbase is the default: its Contexts persist the user's cookies
  and logins, which is what makes "carry on where I left off" true.
- `browser/context_store.py`: remembers which Browserbase Context belongs to
  which user, through the Kwami API. Browserbase returns an opaque Context id
  once at creation and offers no lookup-by-name, so without this every session
  starts a fresh Context -- the user signed out of everything, and the previous
  Context orphaned but still billed. Falls back to process memory when
  `KWAMI_API_KEY` is unset.
- `CDPConnection.connect_ws`: connects to a browser-level CDP WebSocket and
  attaches to a page target in flat mode. Browserbase hands out `connectUrl`,
  the endpoint `connectOverCDP` takes, where `Page.enable` fails outright --
  unlike Browser Use's HTTP base with its `/json/list` discovery.

### Fixed

- **The realtime pipeline was unreachable from the app.** The SDK's wire field is
  `voice.type` with the values `stt-llm-tts` / `realtime`; the handler read
  `pipelineType` with the values `standard` / `realtime`, which no client sends.
  Nothing warned, because the key was simply absent: selecting the realtime
  pipeline built an STT+LLM+TTS agent every time and parsed the user's realtime
  settings into fields that branch never reads. `normalize_pipeline_type` now
  accepts every spelling either side uses.
- **Changing the realtime voice mid-conversation did nothing.**
  `updateRealtimeLive()` sends `realtime_provider` / `realtime_model` /
  `realtime_voice` under `updateType: "voice"`, and `update_voice` read only
  `tts_*` and `stt_*`. Voice and temperature now go over the open socket via
  `RealtimeModel.update_options`; provider and model rebuild.
- **Every agent swap wiped the conversation.** Generation reads
  `Agent._chat_ctx`, a fresh agent starts with an empty one, and nothing copied
  it across -- so any voice, model or pipeline change silently erased the
  session's history. `SessionState._carry_conversation` carries it, filtered by
  the incoming agent's tool set.
- **Replaced agents had their providers closed mid-drain.** The framework closes
  the old activity in its own task, so `aclose()`-ing its TTS the moment a
  replacement was prepared could land while speech was still draining. Cleanup
  now waits for the framework to release the agent, bounded, and skips the wait
  at teardown where the budget will not stretch.
- An `updateType: "llm"` change on the realtime pipeline rebuilt a standard
  pipeline underneath the session, silently downgrading off the realtime model.
  It now retargets the realtime model.
- `change_ai_model` verifies the constructed pipeline before confirming.
  `create_llm` falls back to OpenAI for any provider it cannot build and only
  logs a warning, so reporting success off the request meant telling the user
  they were on Claude while they were on gpt-4o-mini.
- `change_voice` and `change_speaking_speed` dead-ended on "TTS not available"
  for the whole realtime pipeline, which reads to the model as a transient
  fault rather than a wrong-tool answer. They now name the tool that does work.
- Browser minutes were metered as `browser_use/cloud` whatever ran them. The
  usage record now names the vendor, because one line covering both cannot be
  reconciled against either invoice.

### Changed

- `SessionState.update_agent` split into `prepare_handoff` (carry the
  conversation, browser, pending tool calls and session resources) plus the
  session swap, so the framework's own function-tool handoff path keeps the
  bookkeeping.
- The built-in tool contract test derives its expectation from the mixins
  instead of asserting a hard-coded count of 22.
- `tests/unit/test_capabilities.py` fails when `kwami-app` (checked out
  alongside) registers a tool the agent does not describe.
- `tests/integration/test_ui_control_roundtrip.py` drives the app's real tool
  definitions through the data channel and inspects the published envelope, so
  a change to the `tool_call` shape cannot pass both repos' suites and still
  break every UI command.

## [1.0.0] - 2026-09-18

### Added

- Project documentation under `docs/` (architecture, protocol, memory, security, billing, configuration, development, testing, deployment) with mermaid diagrams.
- Hexagonal I/O ports (`MemoryPort`, `RoomPublisherPort`, `UsageReporterPort`, `HttpClientPort`, `BrowserPort`, `SearchPort`) and LiveKit / HTTP adapters.
- Process-wide frozen `Settings` object; credentials are no longer read from `os.environ` at import time.
- Layered test suites: unit, contract (real SDKs), integration, runtime, and live e2e.
- CI: Ruff + mypy, pytest with coverage on Python 3.11 and 3.13, Docker image build, nightly live e2e.
- Cloud-browser minute metering on every release path (idle, failed connect, user close, session cleanup).
- Live conversation, memory, and reconfiguration e2e suites.
- Second deployment target: Cloudflare Workers + Containers (`infra/`), with a
  staging environment, cron keep-alive, and `make deploy-cf` / `make deploy-cf-staging`.
- `GET /health` (and its `/status` alias) on the Worker reports `degraded` rather
  than `error` when the Worker is up but the container is unreachable, and names
  the Workers Paid plan requirement when that is the cause.
- CI type-checks the Worker: `wrangler types` + `tsc --noEmit` on every push and PR.

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
- `ELEVENLABS_API_KEY` now resolves. It was advertised as an alias by
  `EnvVars.ELEVENLABS` but never collected into `Settings.provider_keys`, so
  setting only that spelling warned and silently fell back to OpenAI TTS.
- Google TTS accepts `GOOGLE_API_KEY` as well as `GOOGLE_APPLICATION_CREDENTIALS`.
  `.env.sample` documented the former while the check only looked for the
  latter, so following the sample warned on every Google TTS request.
- Cloudflare install settings are honoured again. pnpm 11 reads only auth and
  registry keys from `.npmrc`, so `only-built-dependencies` and
  `minimum-release-age-exclude` were silently inert: `esbuild` and `workerd`
  postinstall scripts never ran (leaving `wrangler deploy` without a runtime),
  and the pinned Cloudflare toolchain was subject to pnpm 11's new 24h
  `minimumReleaseAge` default. Both now live in `infra/pnpm-workspace.yaml`.

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

[Unreleased]: https://github.com/kwami-labs/kwami-lk-agent/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kwami-labs/kwami-lk-agent/releases/tag/v1.0.0
[0.1.0]: https://github.com/kwami-labs/kwami-lk-agent/blob/dev/agent/pyproject.toml
