# Contributing

Thanks for wanting to improve Kwami LiveKit Agent. This repository is a
production voice worker: a wrong hook signature or a mocked SDK can ship as
silence, leaked logins, or unbilled spend. The bars below exist for that reason.

## Before you start

1. Read [docs/architecture.md](./docs/architecture.md) and
   [docs/testing.md](./docs/testing.md).
2. Open an issue for anything larger than a typo so we can agree on the shape.
3. Follow the [Code of Conduct](./CODE_OF_CONDUCT.md).

## Development setup

```bash
cp .env.sample .env
make install
make check
```

Details: [docs/development.md](./docs/development.md).

You do **not** need provider keys for the offline suite. Do not commit `.env`.

## Branch and PR

- Branch from `dev` unless a maintainer says otherwise.
- Keep the PR focused. One problem per PR.
- Fill in the pull-request template.
- CI must be green: Ruff, mypy, tests on Python 3.11 and 3.13, Docker build.

### Commit messages

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat(memory): reuse Zep client across config updates
fix(browser): reject javascript: URLs
test: raise the coverage floor to 43
docs: add architecture diagrams
```

Types we use: `feat`, `fix`, `test`, `refactor`, `perf`, `docs`, `chore`,
`ci`, `build`. Add a `!` or a `BREAKING CHANGE:` footer when the data-channel
protocol or env vars change.

Do not add `Co-authored-by` trailers for coding agents.

### Changelog

User-facing changes belong under `## [Unreleased]` in [CHANGELOG.md](./CHANGELOG.md):
Added / Changed / Deprecated / Removed / Fixed / Security.

### Releases

A release is a `v*` tag on `main`. The tag must match `version` in
`agent/pyproject.toml`, and that version must have its own changelog section --
`.github/workflows/release.yml` checks both, re-runs the full gate, and publishes
the GitHub Release from the changelog. Steps are in
[docs/deployment.md](./docs/deployment.md#versioning).

## What a good change looks like

- New I/O has a `ports/` protocol and a test fake, not a `MagicMock` of LiveKit or Zep.
- New env vars are on `Settings`, in `.env.sample`, and documented in
  [docs/configuration.md](./docs/configuration.md).
- Anything that enters the LLM is size-capped.
- URLs the model can open go through `validate_url_async`.
- Background tasks are retained (`SessionState.spawn` or an equivalent set).
- Config work stays under `state.run_serialized`.
- The coverage floor in `pyproject.toml` does not go down. Raise it by adding
  tests, never by lowering it to meet the tree. What may be excluded, and the
  two cases where `# pragma: no cover` is allowed, are in
  [docs/testing.md](./docs/testing.md#what-may-be-excluded).

## Tests you should add

| You changed… | Add… |
| --- | --- |
| Domain parsing / prompt / usage | `tests/unit/` |
| A LiveKit or Zep call site | `tests/contract/` against the real SDK |
| Config / tools / browser wiring | `tests/integration/` or `tests/runtime/` |
| A real provider path | `tests/e2e/` marked `@pytest.mark.live` |

Never stub `livekit` or `zep_cloud`. See [docs/testing.md](./docs/testing.md).

## Security issues

Do not file a public issue for a vulnerability. Follow [SECURITY.md](./SECURITY.md).

## License

By contributing you agree that your work is licensed under the
[Apache License 2.0](./LICENSE).
