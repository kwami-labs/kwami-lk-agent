## Summary

<!-- Why this change exists. Link the issue if there is one. -->

## What changed

<!-- User-facing and internal. Add an [Unreleased] entry in CHANGELOG.md when it is user-facing. -->

## How I tested it

- [ ] `make lint`
- [ ] `make typecheck` (or this PR does not touch typed modules)
- [ ] `make test` (offline suite)
- [ ] Added / updated tests in the layer that matches the change ([docs/testing.md](../docs/testing.md))
- [ ] Live e2e (`make test-e2e`) if I touched a provider path

## Checklist

- [ ] No new `os.environ` reads outside `Settings`
- [ ] No `MagicMock` of `livekit` or `zep_cloud`
- [ ] Model-bound strings are size-capped; model-chosen URLs go through `validate_url_async`
- [ ] Background tasks are retained; config work is serialized
- [ ] Docs updated if this changes architecture, protocol, env vars, or security
