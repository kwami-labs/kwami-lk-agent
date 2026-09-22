# Documentation

This is the documentation for **Kwami LiveKit Agent**, a Python voice agent that
runs on [LiveKit Cloud](https://docs.livekit.io/agents/) and is configured at
runtime by the Kwami frontend over the LiveKit data channel.

If you are new to the repository, start with the root [README](../README.md)
and then read [Architecture](./architecture.md).

## Guides

| Document | What it covers |
| --- | --- |
| [Architecture](./architecture.md) | System context, hexagonal layout, session lifecycle, voice pipelines |
| [Protocol](./protocol.md) | Data-channel messages between the frontend and the agent |
| [Configuration](./configuration.md) | Environment variables, `KwamiConfig`, and provider catalogues |
| [Memory](./memory.md) | Zep Cloud threads, knowledge graph, ontology, and greeting injection |
| [Security](./security.md) | Threat model, browser SSRF, prompt-injection, secrets, billing identity |
| [Billing](./billing.md) | Usage tracking, credit reporting, and identity resolution |
| [Development](./development.md) | Local setup, Makefile, layout, and coding conventions |
| [Testing](./testing.md) | Unit, contract, integration, and live e2e layers |
| [Deployment](./deployment.md) | Docker image, LiveKit Cloud, Cloudflare Workers, CI, telephony, releases |

## Project

| Document | What it covers |
| --- | --- |
| [Changelog](./changelog.md) | User-facing changes, Keep a Changelog format |
| [Contributing](../CONTRIBUTING.md) | How to propose, test, and land a change |
| [Code of Conduct](../CODE_OF_CONDUCT.md) | Contributor Covenant |
| [Security policy](../SECURITY.md) | How to report a vulnerability |

## Related repositories

- [kwami-lk-api](https://github.com/alexcolls/kwami-lk-api) — token endpoint, runtime config, credits API
- [kwami-ai](https://github.com/alexcolls/kwami-ai) — TypeScript SDK that speaks the protocol in [protocol.md](./protocol.md)
- [kwami-ai-pg](https://github.com/alexcolls/kwami-ai-pg) — playground Vue app
