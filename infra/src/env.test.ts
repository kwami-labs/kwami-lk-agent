import { describe, expect, it } from "vitest";

import { CONTAINER_SECRET_KEYS, containerEnvFromWorker } from "./env";

/**
 * What reaches the container decides what the agent can do.
 *
 * Browserbase is the agent's default cloud-browser vendor and neither of its
 * credentials was forwarded here for an entire release: the agent asked for
 * Browserbase, found no key, and fell back to Browser Use — which its own
 * warning describes as signing the user out of every site they were logged
 * into. `ELEVENLABS_API_KEY`, documented as an accepted alias, was missing too.
 *
 * `agent/tests/unit/test_worker_env_parity.py` checks this list against
 * `Settings.ENV_VAR_NAMES` from the Python side. These check the function.
 */

const baseEnv = {
  ENVIRONMENT: "production",
  LIVEKIT_URL: "wss://example.livekit.cloud",
  KWAMI_API_URL: "https://api.example.com",
  KWAMI_API_TIMEOUT: "30.0",
  KWAMI_ALLOW_BROWSER_JS: "false",
  KWAMI_BROWSER_PROVIDER: "browserbase",
} as unknown as Env;

describe("containerEnvFromWorker", () => {
  it("forwards the plain vars", () => {
    const out = containerEnvFromWorker(baseEnv);

    expect(out.ENVIRONMENT).toBe("production");
    expect(out.LIVEKIT_URL).toBe("wss://example.livekit.cloud");
    expect(out.KWAMI_API_URL).toBe("https://api.example.com");
  });

  it("forwards the browser provider, so the agent does not pick its own", () => {
    // Without this the agent falls back to DEFAULT_BROWSER_PROVIDER and can
    // disagree with which credential was actually configured.
    expect(containerEnvFromWorker(baseEnv).KWAMI_BROWSER_PROVIDER).toBe("browserbase");
  });

  it("forwards a secret that is set", () => {
    const env = { ...baseEnv, BROWSERBASE_API_KEY: "bb_test" } as unknown as Env;

    expect(containerEnvFromWorker(env).BROWSERBASE_API_KEY).toBe("bb_test");
  });

  it("omits a secret that is absent rather than passing an empty string", () => {
    // An empty string is truthy to `os.environ.get`, so forwarding one would
    // make `Settings` believe the credential exists.
    const out = containerEnvFromWorker(baseEnv);

    expect("BROWSERBASE_API_KEY" in out).toBe(false);
  });

  it("omits a secret set to an empty string", () => {
    const env = { ...baseEnv, ZEP_API_KEY: "" } as unknown as Env;

    expect("ZEP_API_KEY" in containerEnvFromWorker(env)).toBe(false);
  });

  it("ignores a non-string secret", () => {
    const env = { ...baseEnv, ZEP_API_KEY: 42 } as unknown as Env;

    expect("ZEP_API_KEY" in containerEnvFromWorker(env)).toBe(false);
  });

  it("carries both Browserbase credentials", () => {
    // The regression that motivated all of this. A project id without a key —
    // or the reverse — leaves the vendor unusable.
    expect(CONTAINER_SECRET_KEYS).toContain("BROWSERBASE_API_KEY");
    expect(CONTAINER_SECRET_KEYS).toContain("BROWSERBASE_PROJECT_ID");
  });

  it("carries both spellings of the ElevenLabs key", () => {
    expect(CONTAINER_SECRET_KEYS).toContain("ELEVEN_API_KEY");
    expect(CONTAINER_SECRET_KEYS).toContain("ELEVENLABS_API_KEY");
  });

  it("does not forward GOOGLE_APPLICATION_CREDENTIALS", () => {
    // It is a path to a file that does not exist in the container image, so
    // forwarding it would point the Google SDK at nothing.
    expect(CONTAINER_SECRET_KEYS).not.toContain("GOOGLE_APPLICATION_CREDENTIALS");
  });

  it("has no duplicate keys", () => {
    expect(new Set(CONTAINER_SECRET_KEYS).size).toBe(CONTAINER_SECRET_KEYS.length);
  });
});
