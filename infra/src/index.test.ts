import { createExecutionContext, env, waitOnExecutionContext } from "cloudflare:test";
import { describe, expect, it } from "vitest";

import worker from "./index";

/**
 * The Worker's HTTP surface.
 *
 * `pnpm check` only ever proved this bundles. What it could not prove is what
 * any of these routes actually answer — including `/health`, which the
 * Cloudflare Containers runtime uses as `pingEndpoint` and which a two-minute
 * cron also depends on.
 *
 * Containers do not run under `vitest`, so `ensureRunning()` fails here. That
 * is deliberately useful: it exercises the degraded path, which is the one that
 * fires in production when the account is not on the Workers Paid plan, and
 * which had never been executed by anything.
 */

async function call(path: string, init?: RequestInit): Promise<Response> {
  // `fetch` here declares (request, env) and uses no ExecutionContext, so the
  // handler is called with exactly what it takes -- passing a third argument
  // type-checks as an error even though it runs.
  // The handler takes an incoming request (it carries `cf` properties a
  // hand-built Request does not), so the cast is at the boundary rather than
  // spread through the assertions.
  const request = new Request(`https://example.com${path}`, init) as unknown as Parameters<
    typeof worker.fetch
  >[0];
  return worker.fetch(request, env as never);
}

describe("routing", () => {
  it("answers 404 for an unknown path", async () => {
    const response = await call("/nope");

    expect(response.status).toBe(404);
    expect(await response.json()).toEqual({ error: "not found" });
  });

  it("answers 404 for /start with the wrong method", async () => {
    // /start boots a container; it must not be reachable by GET.
    const response = await call("/start");

    expect(response.status).toBe(404);
  });
});

describe("/start", () => {
  it("refuses an unauthenticated request", async () => {
    // It was an unauthenticated POST that boots a paid compute instance, on a
    // Worker published to *.workers.dev with preview URLs on.
    const response = await call("/start", { method: "POST" });

    expect(response.status).toBe(401);
    expect(await response.json()).toEqual({ error: "unauthorized" });
  });

  it("refuses a wrong token", async () => {
    const response = await call("/start", {
      method: "POST",
      headers: { authorization: "Bearer not-the-token" },
    });

    expect(response.status).toBe(401);
  });

  it("refuses a token with no Bearer prefix", async () => {
    const response = await call("/start", {
      method: "POST",
      headers: { authorization: "not-the-token" },
    });

    expect(response.status).toBe(401);
  });

  it("fails closed when no admin token is configured", async () => {
    // `/health` and the keepalive cron start the container by themselves, so
    // `/start` is a convenience -- and a convenience is not worth an open door.
    expect((env as unknown as Record<string, unknown>).KWAMI_ADMIN_TOKEN).toBeUndefined();

    const response = await call("/start", {
      method: "POST",
      headers: { authorization: "Bearer anything" },
    });

    expect(response.status).toBe(401);
  });
});

describe("/health", () => {
  it("reports degraded rather than throwing when the container is unavailable", async () => {
    // The exact case that reaches production on a free plan. Before this the
    // route's failure path had never run.
    const response = await call("/health");
    const body = (await response.json()) as Record<string, unknown>;

    expect(response.status).toBe(503);
    expect(body.status).toBe("degraded");
    expect(body.worker).toBe("ok");
    expect(body.container).toBe("unavailable");
  });

  it("does not return the raw runtime error to the caller", async () => {
    // Reachable by anyone who finds the workers.dev hostname; runtime error
    // text names internal identifiers and account state.
    const body = (await (await call("/health")).json()) as Record<string, unknown>;

    expect(body.error).toBeUndefined();
    expect(JSON.stringify(body)).not.toContain("durableObject");
  });

  it("never caches its answer", async () => {
    // A cached health check is worse than none: it reports a state that has
    // already changed.
    const response = await call("/health");

    expect(response.headers.get("cache-control")).toBe("no-store");
  });

  it("serves /status identically", async () => {
    const [health, status] = await Promise.all([call("/health"), call("/status")]);

    expect(status.status).toBe(health.status);
  });
});

describe("the scheduled keepalive", () => {
  it("does not throw when the container cannot be started", async () => {
    // A cron that throws is a cron that silently stops being scheduled.
    const ctx = createExecutionContext();

    await worker.scheduled?.(
      { cron: "*/2 * * * *", scheduledTime: Date.now(), noRetry: () => {} } as never,
      env as never,
      ctx,
    );

    await expect(waitOnExecutionContext(ctx)).resolves.not.toThrow();
  });
});
