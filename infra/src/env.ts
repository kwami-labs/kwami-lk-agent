/**
 * Worker secrets forwarded into the LiveKit agent container.
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
  "KWAMI_API_KEY",
] as const;

export type ContainerSecretKey = (typeof CONTAINER_SECRET_KEYS)[number];

export function containerEnvFromWorker(env: Env): Record<string, string> {
  const out: Record<string, string> = {
    ENVIRONMENT: env.ENVIRONMENT,
    LIVEKIT_URL: env.LIVEKIT_URL,
    KWAMI_API_URL: env.KWAMI_API_URL,
    KWAMI_API_TIMEOUT: env.KWAMI_API_TIMEOUT,
    KWAMI_ALLOW_BROWSER_JS: env.KWAMI_ALLOW_BROWSER_JS,
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
