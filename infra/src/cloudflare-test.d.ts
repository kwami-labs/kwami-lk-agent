// Types for the `cloudflare:test` module the Worker test runner provides
// (`env`, `createExecutionContext`, `SELF`, ...). Declared here rather than in
// tsconfig's `types` array so `Env` stays the generated one from
// worker-configuration.d.ts.
/// <reference types="@cloudflare/vitest-plugin/types" />

declare module "cloudflare:test" {
  interface ProvidedEnv extends Env {}
}
