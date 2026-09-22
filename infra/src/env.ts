/**
 * Worker secrets forwarded into the LiveKit agent container.
 *
 * GOOGLE_APPLICATION_CREDENTIALS is deliberately absent: it is a *path* to a
 * service-account JSON file, and no such file exists in the container image.
 * Forwarding the variable would point the Google SDK at a file that is not
 * there. Use GOOGLE_API_KEY on this target (see docs/deployment.md).
 *
 * Values are set with `wrangler secret put` / `wrangler secret bulk`
 * (or `.dev.vars` for local). They are never stored in wrangler.jsonc.
 */
export const CONTAINER_SECRET_KEYS = [
  "LIVEKIT_API_KEY",
  "LIVEKIT_API_SECRET",
  "OPENAI_API_KEY",
  "DEEPGRAM_API_KEY",
  "CARTESIA_API_KEY",
  "ELEVEN_API_KEY",
  "GOOGLE_API_KEY",
  "ASSEMBLYAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "GROQ_API_KEY",
  "DEEPSEEK_API_KEY",
  "MISTRAL_API_KEY",
  "CEREBRAS_API_KEY",
  "XAI_API_KEY",
  "TAVILY_API_KEY",
  "SERPAPI_KEY",
  "ZEP_API_KEY",
  "BROWSER_USE_API_KEY",
  // Browserbase is the *default* cloud-browser vendor (see
  // `DEFAULT_BROWSER_PROVIDER` in agent/src/settings.py), and neither of its
  // credentials was forwarded here. The agent asked for Browserbase, found no
  // key, and fell back to Browser Use -- which its own warning describes as
  // signing the user out of every site they were logged in to. With no Browser
  // Use key either, the browser panel simply did not work on this target.
  "BROWSERBASE_API_KEY",
  "BROWSERBASE_PROJECT_ID",
  // `.env.sample` documents this as an accepted alias for ELEVEN_API_KEY and
  // Settings reads both; only one of the two ever reached the container.
  "ELEVENLABS_API_KEY",
  "KWAMI_API_KEY",
  // Carries the trace backend's API key ("api-key=..."), so it is a secret
  // rather than a var even though the other OTEL_* settings are not.
  "OTEL_EXPORTER_OTLP_HEADERS",
] as const;

/**
 * Worker-only secret gating `POST /start`. Deliberately not in
 * CONTAINER_SECRET_KEYS: the agent has no use for it, and a credential that
 * does not need to cross a boundary should not cross it.
 */
export const ADMIN_TOKEN_KEY = "KWAMI_ADMIN_TOKEN" as const;

export type ContainerSecretKey = (typeof CONTAINER_SECRET_KEYS)[number];

export function containerEnvFromWorker(env: Env): Record<string, string> {
  const out: Record<string, string> = {
    ENVIRONMENT: env.ENVIRONMENT,
    LIVEKIT_URL: env.LIVEKIT_URL,
    KWAMI_API_URL: env.KWAMI_API_URL,
    KWAMI_API_TIMEOUT: env.KWAMI_API_TIMEOUT,
    KWAMI_ALLOW_BROWSER_JS: env.KWAMI_ALLOW_BROWSER_JS,
    KWAMI_BROWSER_PROVIDER: env.KWAMI_BROWSER_PROVIDER,
    KWAMI_LOG_FORMAT: env.KWAMI_LOG_FORMAT,
    OTEL_EXPORTER_OTLP_ENDPOINT: env.OTEL_EXPORTER_OTLP_ENDPOINT,
    OTEL_SERVICE_NAME: env.OTEL_SERVICE_NAME,
  };

  const secrets = env as Env & Partial<Record<ContainerSecretKey, string>>;
  for (const key of CONTAINER_SECRET_KEYS) {
    const value = secrets[key];
    if (typeof value === "string" && value.length > 0) {
      out[key] = value;
    }
  }

  return out;
}
