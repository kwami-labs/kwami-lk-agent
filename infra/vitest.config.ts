import { cloudflareTest } from "@cloudflare/vitest-plugin";
import { defineConfig } from "vitest/config";

/**
 * Runs the Worker's tests inside workerd, not Node.
 *
 * That distinction is the point. `infra/` had no tests at all while the Python
 * side sat at 100% coverage, and `pnpm check` only ever proved the Worker
 * bundles. A Node-based test would have been little better: the container
 * binding, the Durable Object and `ctx.waitUntil` do not exist there, so the
 * parts most likely to break on deploy would have been the parts stubbed out.
 */
export default defineConfig({
  plugins: [
    cloudflareTest({
      // Reads the real bindings and migrations, so a wrangler.jsonc that would
      // fail to deploy fails here first.
      wrangler: { configPath: "./wrangler.jsonc" },
    }),
  ],
});
