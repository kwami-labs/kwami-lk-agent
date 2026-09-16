# Kwami LiveKit Agent

LiveKit Cloud agent for Kwami AI voice interactions. A Python voice agent built
on [livekit-agents](https://docs.livekit.io/agents/), configured at runtime by
the Kwami frontend over the LiveKit data channel.

## Quick Start

```bash
# Copy and configure environment
cp .env.sample .env
# Edit .env with your credentials

# Install dependencies (including dev tools)
make install

# Run agent locally for development
make dev
```

## Project Structure

```
kwami-lk-agent/
├── agent/
│   ├── src/
│   │   ├── main.py             # Entry point: worker, job lifecycle, data-channel dispatch
│   │   ├── agent.py            # KwamiAgent: prompt assembly, framework hooks, greeting
│   │   ├── session.py          # SessionState: agent swaps, resource ownership, usage reporting
│   │   ├── config.py           # KwamiConfig and nested voice/soul/memory config
│   │   ├── constants.py        # Provider, model and voice catalogues
│   │   ├── runtime_bootstrap.py# Telephony: resolve kwami_id, fetch runtime config
│   │   ├── room_context.py     # ContextVar holding the active room for tools
│   │   ├── factories/          # STT / LLM / TTS / VAD / realtime construction
│   │   ├── handlers/           # config, config_update and tool_result handling
│   │   ├── memory/             # Zep Cloud: manager, context, search, ontology
│   │   ├── tools/              # Built-in agent tools and client-side tool bridge
│   │   ├── browser/            # Browser Use Cloud session, CDP, URL safety
│   │   ├── usage/              # Usage tracking and credit reporting
│   │   └── utils/              # Logging, provider parsing, room and validation helpers
│   ├── tests/                  # unit / contract / integration / e2e
│   ├── livekit.toml            # LiveKit Cloud config
│   ├── pyproject.toml
│   └── Dockerfile
├── .github/workflows/          # CI: lint, types, tests, coverage, Docker, nightly E2E
├── .env.sample
├── Makefile
└── README.md
```

A LiveKit worker registers exactly one agent entrypoint, so this repository
deploys a single agent. Different personas are configuration, not separate
agents: the frontend sends a `config` message and the running agent is rebuilt
in place.

## Commands

| Command                 | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| `make install`          | Install agent dependencies (including dev tools)        |
| `make dev`              | Run agent locally for testing                           |
| `make deploy`           | Deploy agent to LiveKit Cloud                           |
| `make test`             | Run the offline suite (unit + contract + integration)   |
| `make test-cov`         | Offline suite with a coverage report                    |
| `make test-e2e`         | End-to-end tests against real providers (needs keys)    |
| `make lint` / `format`  | Run / apply the linter                                  |
| `make typecheck`        | Run mypy                                                |
| `make check`            | lint + typecheck + test (what CI runs)                  |

## Testing

Four layers under `agent/tests/`:

- **`unit/`** — pure logic: config parsing, prompt building, usage maths, URL safety.
- **`contract/`** — this codebase's assumptions checked against the *real* installed
  SDKs: that the hooks we override exist with the signatures we use, that every
  advertised provider constructs, that every Zep method we call is real.
- **`integration/`** — wiring, against a real `livekit.agents.Agent` and mocked transports.
- **`e2e/`** — marked `live`, excluded by default; runs against real providers.

**Never stub `livekit` or `zep_cloud` in tests.** Both are installed and are
imported for real. An earlier version of `conftest.py` replaced them with
`MagicMock`, which let a wrong `on_enter` signature, a hook the framework never
dispatches, and five nonexistent Zep methods pass for months. The `contract/`
layer exists to make that class of drift a red build.

## Persistent Memory

Each Kwami agent can have independent, persistent memory powered by
[Zep Cloud](https://www.getzep.com/):

- **Conversation history** — user and assistant turns are written to a Zep thread
- **Fact extraction** — facts about the user are extracted into a knowledge graph
- **Temporal knowledge graphs** — facts carry validity, so superseded ones are marked
- **Custom ontology** — Person, Project, Product, Goal and Procedure entity types

### Setup

1. Create a Zep Cloud account at https://www.getzep.com/
2. Get your API key from the dashboard
3. Add `ZEP_API_KEY` to your `.env` file

Memory is enabled automatically when `ZEP_API_KEY` is set.

### How it works

- Each Kwami uses its `kwami_id` as a unique user identifier in Zep
- Context is fetched on join and injected into the system prompt, under a hard
  timeout so a slow Zep can never delay the greeting
- The agent exposes `remember_fact` and `recall_memories` tools

## Browsing

The agent can drive a cloud browser (Browser Use Cloud) that the user watches
live in the app. Two safety rules apply, and both matter because the browser
keeps the user's cookies and logins:

- URLs are validated before navigation. Non-HTTP schemes, loopback, link-local
  (including cloud metadata endpoints) and private ranges are refused.
- Arbitrary JavaScript execution is **disabled by default**. Page text reaches
  the model as untrusted input, so a hostile page could otherwise instruct it to
  exfiltrate session cookies. Set `KWAMI_ALLOW_BROWSER_JS=1` only if you accept
  that risk.

A browser is never started for a session without a real `kwami_id`, because
profiles are per-user and a shared one would leak logins between users.

## Deployment

```bash
cd agent
lk agent create .      # first time only
lk agent deploy        # or: make deploy
```

LiveKit Cloud handles scaling, lifecycle, and hosting. CI builds the image on
every push so deploy breakage is caught before `lk agent deploy`.

## Environment Variables

Copy `.env.sample` to `.env` — it documents every variable the agent reads.
The minimum for a working local run:

```env
# LiveKit (required)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Voice pipeline (required)
OPENAI_API_KEY=your-openai-key
DEEPGRAM_API_KEY=your-deepgram-key
CARTESIA_API_KEY=your-cartesia-key

# Credits / runtime config (required for billing and telephony)
KWAMI_API_URL=http://localhost:8080
KWAMI_API_KEY=your-kwami-api-key

# Optional
ZEP_API_KEY=            # persistent memory
TAVILY_API_KEY=         # web_search
SERPAPI_KEY=            # product_search
BROWSER_USE_API_KEY=    # cloud browsing
```

Some LLM providers (`anthropic`, `groq`, `google`) need their own
`livekit-plugins-*` package, declared here as optional extras. When one is not
installed the agent logs a warning and falls back to OpenAI rather than failing
the session.

## Related Repositories

- **[kwami-lk-api](https://github.com/alexcolls/kwami-lk-api)** - Token endpoint and memory API
- **[kwami-ai](https://github.com/alexcolls/kwami-ai)** - TypeScript SDK
- **[kwami-ai-pg](https://github.com/alexcolls/kwami-ai-pg)** - Playground Vue app

## License

Apache 2.0
