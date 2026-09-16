# Security policy

The agent holds provider API keys, can drive a logged-in cloud browser, writes
long-term user memory, and reports usage that becomes a charge. Please report
vulnerabilities privately.

A longer threat model lives in [docs/security.md](./docs/security.md).

## Supported versions

| Version | Supported |
| --- | --- |
| `main` / `dev` (unreleased 0.1.x) | Yes |
| Older untagged snapshots | No |

There is no LTS branch yet. Fixes land on `dev` and ship from `main`.

## How to report

Use [GitHub private vulnerability reporting](https://github.com/kwami-labs/kwami-lk-agent/security/advisories/new)
on this repository.

Include:

- A description of the issue and its impact (data leak, account takeover,
  unbilled spend, SSRF, prompt injection that reaches a logged-in browser, …)
- Steps to reproduce, or a minimal proof of concept
- Affected commit SHA or branch
- Any mitigation you already see

You should get an acknowledgement within **72 hours**. We will agree a
disclosure date before any public advisory.

Please **do not** open a public issue, discussion, or pull request that
includes exploit details.

## Scope

In scope:

- This repository's agent worker and its Docker image
- Data-channel handling, browser URL / JS gates, memory tenancy, usage reporting
- Secrets handling in `Settings` and logs

Out of scope (report to the owning repo or vendor):

- [kwami-lk-api](https://github.com/alexcolls/kwami-lk-api), the TypeScript SDK, or the playground
- LiveKit Cloud, Zep Cloud, Browser Use Cloud, or model-provider incidents
- Issues that only exist when `KWAMI_ALLOW_BROWSER_JS=1` and are already
  documented as accepted risk in [docs/security.md](./docs/security.md)

## Safe harbor

We will not pursue legal action against researchers who:

- Report in good faith through the private channel above
- Avoid privacy violations, destruction of data, and disruption of production
- Do not access accounts or data that are not their own
- Give us a reasonable window to ship a fix before public disclosure
