# Architecture

Kwami LiveKit Agent is a single LiveKit worker. One process registers one
entrypoint (`kwami-agent`). Different personas are configuration, not separate
agents: the frontend (or the telephony bootstrap) sends a `config` message and
the running agent is rebuilt in place.

```mermaid
flowchart LR
    subgraph Clients
        Web["Kwami frontend / SDK"]
        Phone["SIP / telephony caller"]
    end

    LK["LiveKit Cloud<br/>rooms, audio, data channel"]
    Worker["kwami-lk-agent<br/>AgentServer worker"]

    subgraph Providers
        STT["STT<br/>Deepgram, OpenAI, …"]
        LLM["LLM<br/>OpenAI, Anthropic, …"]
        TTS["TTS<br/>Cartesia, ElevenLabs, …"]
        RT["Realtime<br/>OpenAI, Gemini Live"]
    end

    subgraph Platform
        API["kwami-lk-api<br/>runtime config + credits"]
        Zep["Zep Cloud<br/>memory + graph"]
        Browse["Browser Use Cloud"]
        Search["Tavily / SerpApi"]
    end

    Web <--> LK
    Phone <--> LK
    LK <--> Worker
    Worker --> STT
    Worker --> LLM
    Worker --> TTS
    Worker --> RT
    Worker --> API
    Worker --> Zep
    Worker --> Browse
    Worker --> Search
```

A LiveKit worker can host many concurrent jobs. Each job is one room, one
`AgentSession`, one `SessionState`, and one current `KwamiAgent`.

## Design principles

The recent layout is hexagonal on purpose. Earlier versions hid critical bugs
behind `MagicMock` and import-time `os.environ` reads. The current split exists
so those classes of failure are visible in CI.

| Layer | Responsibility | I/O? |
| --- | --- | --- |
| `domain/` | Config types, wire parsing, prompt assembly, usage maths | No |
| `ports/` | Structural `Protocol`s for every I/O boundary | Contract only |
| `adapters/` | LiveKit publisher, pooled HTTP client | Yes |
| `runtime/` | Job lifecycle, data-channel dispatch, pipeline construction | Orchestration |
| `factories/` | STT / LLM / TTS / VAD / realtime construction | Provider SDKs |
| `handlers/` | `config`, `config_update`, `tool_result` | Yes |
| `memory/`, `browser/`, `usage/`, `tools/` | Feature modules behind ports | Yes |

Dependency arrows point inward. Adapters do not import ports as base classes;
they satisfy them structurally. `runtime.pipeline` is **not** re-exported from
`runtime/__init__.py` because `agent.py` imports `runtime.container` — that
re-export would create an import cycle.

```mermaid
flowchart TB
    subgraph Runtime["runtime / handlers / session"]
        Entry["main.entrypoint"]
        Router["DataMessageRouter"]
        Pipeline["create_agent_from_config"]
        State["SessionState"]
        Deps["AgentDeps"]
    end

    subgraph Domain["domain (pure)"]
        Config["KwamiConfig"]
        Prompt["build_system_prompt"]
        Usage["UsageTracker"]
        Parse["text / number / section"]
    end

    subgraph Ports["ports"]
        MemPort["MemoryPort"]
        PubPort["RoomPublisherPort"]
        BillPort["UsageReporterPort"]
        HttpPort["HttpClientPort"]
        BrowsePort["BrowserPort"]
        SearchPort["SearchPort"]
    end

    subgraph Adapters["adapters + feature modules"]
        Pub["LiveKitRoomPublisher"]
        Http["HttpxClient"]
        Mem["KwamiMemory"]
        Browse["CloudBrowserSession"]
        Report["UsageReporter"]
    end

    Entry --> Router
    Entry --> Pipeline
    Entry --> State
    Entry --> Deps
    Router --> State
    Pipeline --> Config
    Pipeline --> Prompt
    State --> Usage
    Deps --> MemPort
    Deps --> PubPort
    Mem -.-> MemPort
    Pub -.-> PubPort
    Report -.-> BillPort
    Http -.-> HttpPort
    Browse -.-> BrowsePort
```

## Process startup

```mermaid
sequenceDiagram
    participant CLI as livekit-agents CLI
    participant Main as src.main
    participant Settings as Settings
    participant Server as AgentServer
    participant Proc as JobProcess

    CLI->>Main: python -m src.main start|dev
    Main->>Main: load_dotenv(.env)
    Main->>Settings: Settings.from_env()
    Main->>Main: set_settings(...)
    Main->>Server: setup_fnc = prewarm
    Note over Proc: A job is assigned
    Server->>Proc: prewarm() loads Silero VAD
    Server->>Main: entrypoint(JobContext)
```

Credentials are resolved once, after `load_dotenv`, into a frozen `Settings`
object. Downstream code takes settings as a value. Tests construct
`Settings(...)` directly and never depend on the developer's shell.

## Job lifecycle

The entrypoint starts a **placeholder** agent so the room comes up immediately.
The real persona arrives either from the frontend (`config` data message) or,
for telephony, from `GET /internal/kwamis/{id}/runtime` on kwami-lk-api.

The telephony fetch is started **before** `session.start()` so the HTTP round
trip overlaps room setup. A failure never takes the session down: the
placeholder agent is a worse experience than the configured one, but it is
still a working call.

```mermaid
sequenceDiagram
    participant LK as LiveKit
    participant EP as entrypoint
    participant API as kwami-lk-api
    participant Session as AgentSession
    participant State as SessionState
    participant Agent as KwamiAgent

    LK->>EP: rtc_session job
    EP->>Agent: create_agent_from_config(default, skip_greeting=True)
    EP->>Session: AgentSession(userdata=AgentDeps)
    EP->>State: create_session_state(...)
    EP->>EP: resolve_kwami_id(ctx)
    alt telephony kwami_id present
        EP->>API: fetch_runtime_config (background)
    end
    EP->>Session: start(agent, room)
    EP->>State: resolve_user_identity / on participant_connected
    alt runtime config arrived
        EP->>State: apply_runtime_config → handle_full_config
        State->>Session: update_agent(configured)
    end
    Note over Session,Agent: Frontend may still send config / config_update
    LK->>State: shutdown callback
    State->>State: close browser, pipeline, memory
    State->>API: POST /credits/usage/report (≤ 5s)
```

Identity for billing can arrive after the agent: a human may join the room
late. `resolve_identity_on_join` adopts the first non-agent participant so
usage is not silently dropped.

## Voice pipelines

`runtime/pipeline.py` decides the entire inference graph from `KwamiConfig.voice`.

```mermaid
flowchart TD
    C["KwamiConfig.voice"]
    C --> VAD["create_vad<br/>Silero, reuses prewarmed model"]
    C --> Type{"pipeline_type"}

    Type -->|standard| STT["create_stt"]
    Type -->|standard| LLM["create_llm"]
    Type -->|standard| TTS["create_tts"]
    STT --> Agent["KwamiAgent(stt, llm, tts, vad)"]
    LLM --> Agent
    TTS --> Agent

    Type -->|realtime| RT["create_realtime_model"]
    RT --> AgentRT["KwamiAgent(llm=realtime, vad)"]
```

| Pipeline | When | Components |
| --- | --- | --- |
| `standard` (default) | Separate STT, LLM, TTS providers | Deepgram/OpenAI/… → GPT/Claude/… → Cartesia/ElevenLabs/… |
| `realtime` | Ultra-low latency | OpenAI GPT-4o Realtime or Gemini Live, plus VAD |

Missing optional provider packages (`livekit-plugins-anthropic`,
`livekit-plugins-google`, …) log a warning and fall back to OpenAI rather than
failing the session.

VAD settings on the config are honoured. The prewarmed Silero model is reused
unless the session asks for different turn-taking.

## Agent reconfiguration

`SessionState.update_agent` is the only place an agent is swapped. Config
messages are serialized through `state.config_lock` so two close-together
updates cannot race on `session.update_agent`.

On swap the session:

1. Carries the conversation onto the new agent. Generation reads
   `Agent._chat_ctx`, a fresh agent starts with an empty one, and nothing in
   the framework copies it across — so before this, every voice, model or
   pipeline change silently erased the session's history.
2. Closes the old agent's STT / LLM / TTS connections, **after** waiting for the
   framework to release that agent. The old activity is drained and closed in a
   task of the framework's own, so closing its TTS the moment a replacement is
   prepared can land mid-drain.
3. Closes the old Zep client **only** if the new agent does not share it.
4. Hands the live cloud browser to the new agent (otherwise it keeps billing).
5. Transfers unresolved client-tool futures so in-flight UI calls do not time out.
6. Re-attaches `usage_tracker` and `room`.

There are two ways an agent is replaced and only one is ours.
`SessionState.update_agent` does the bookkeeping and then swaps. When a
*function tool* returns an `Agent`, the framework swaps instead, after the
tool's output has been collected — which is what lets the sentence explaining a
model switch survive the switch. `prepare_handoff` is the shared half, so that
path keeps the bookkeeping without swapping twice.

Soul and memory-knob updates happen without a swap (`update_soul`,
`update_memory`), as do realtime voice, speed and temperature, which are pushed
over the open socket. A provider, model or pipeline-type change decides which
object was constructed, so it rebuilds.

## Agent hooks

`KwamiAgent` extends `livekit.agents.Agent` plus `AgentToolsMixin`.

```mermaid
flowchart LR
    Enter["on_enter"] --> Dup["duplicate-agent guard"]
    Dup --> Listen["register conversation_item_added"]
    Listen --> Mem["inject Zep context ≤ 6s"]
    Mem --> Greet["generate_reply greeting"]

    UserTurn["on_user_turn_completed"] --> Buffer["buffer_user_message"]
    Item["conversation_item_added<br/>(assistant)"] --> Batch["add_exchange to Zep"]
```

Important constraints, all learned the hard way:

- `on_enter` takes **no arguments**. A previous `room` parameter was always
  `None`, which disabled duplicate detection and wiped `self.room`.
- Memory injection is hard-capped at `Timeouts.MEMORY_CONTEXT` (6s). A slow Zep
  must never delay the first utterance.
- Assistant turns are persisted from `conversation_item_added`, not from a
  fictional `on_agent_turn_completed` hook the framework never dispatched.
- Background tasks are retained on the agent / session. A bare
  `asyncio.create_task` can be garbage-collected mid-flight.

## Tools

Built-in tools live on `AgentToolsMixin`. Client-side tools are registered from
the frontend and executed over the data channel (see [Protocol](./protocol.md)).

| Mixin | Tools |
| --- | --- |
| `AgentToolsMixin` — identity / voice | `get_kwami_info`, `get_current_time`, `change_voice`, `change_speaking_speed`, `change_language`, `get_current_voice_settings` |
| `AgentToolsMixin` — memory | `remember_fact`, `recall_memories`, `get_memory_status` |
| `AgentToolsMixin` — search | `web_search`, `product_search`, `dismiss_search_result` |
| `AgentToolsMixin` — browser | `navigate_to`, `go_back_in_browser`, `go_forward_in_browser`, `close_navigation`, `click_in_navigation`, `type_in_navigation`, `press_key_in_navigation`, `scroll_navigation`, `run_js_in_navigation`, `read_navigation_page` |
| `PipelineControlMixin` | `change_ai_model`, `switch_pipeline_mode`, `change_realtime_voice`, `list_available_models`, `list_available_voices`, `list_model_providers`, `get_pipeline_status` |
| `KnowledgeToolsMixin` | `deep_research`, `get_market_quote` |
| `MediaToolsMixin` | `play_media`, `control_playback`, `set_playback_volume`, `get_now_playing` |
| `TradingToolsMixin` | `prepare_trade`, `open_trade_ticket`, `submit_trade`, `cancel_prepared_trade`, `get_prepared_trade` |

`PipelineControlMixin` reaches the session runtime through `AgentDeps.reconfigure`,
which is what lets the agent rebuild its own pipeline from inside a tool call.
It returns the new agent rather than installing it; see the swap section above.

`MediaToolsMixin` runs a fixed set of audited scripts in the page, selected by
name. That is deliberately not `run_js_in_navigation`, which executes
model-authored code and stays behind `KWAMI_ALLOW_BROWSER_JS`.

`TradingToolsMixin` is a three-step machine and the steps cannot be collapsed:
`prepare_trade` prices and reads an order back without sending it,
`open_trade_ticket` puts the user's own broker on screen, and `submit_trade`
requires a confirmation code **derived from the order** — so a code the model
invents, a code spoken for a different order, and a code that predates an edit
all fail closed.

Client tools such as `set_ui_control` are **not** defined here; they arrive from
the frontend. System-prompt guidance is assembled per session from the tools
actually registered (`domain/capabilities.py`), one block per capability, so the
model is never told about a tool that is absent — and, just as importantly, is
told about every one that is present.

Tools resolve the room through `AgentDeps` on `AgentSession.userdata`, which
the framework threads into every `RunContext`. There is no process-wide
`ContextVar` for the room.

## Source map

```
agent/src/
├── main.py                 # Worker, prewarm, entrypoint
├── agent.py                # KwamiAgent hooks and greeting
├── session.py              # SessionState: swaps, ownership, billing
├── settings.py             # Frozen env → Settings
├── runtime_bootstrap.py    # Telephony kwami_id + runtime config fetch
├── constants.py            # Provider / voice catalogues, timeouts
├── domain/                 # Pure: config, parsing, prompt, capabilities, models, usage, errors
├── ports/                  # Protocols: memory, search, browser, publisher, billing, HTTP
├── adapters/               # LiveKit publisher, pooled httpx client
├── runtime/                # dispatch, lifecycle, pipeline, reconfigure, AgentDeps
├── factories/              # STT, LLM, TTS, VAD, realtime
├── handlers/               # config, config_update, realtime/pipeline switching, tool_result
├── memory/                 # Zep manager, context, search, ontology
├── tools/                  # Built-in mixins, ClientToolManager, provider tool budget
├── browser/                # Browserbase / Browser Use Cloud, CDP, URL safety
├── usage/                  # Credits API reporter
└── utils/                  # Logging, room, provider, validation
```
