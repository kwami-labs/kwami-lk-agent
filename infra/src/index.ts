import { Container, getContainer } from "@cloudflare/containers";
import { DurableObject } from "cloudflare:workers";

import { containerEnvFromWorker } from "./env";

const WORKER_INSTANCE = "worker";
const HEALTH_PORT = 8080;

export class KwamiAgentContainer extends Container<Env> {
  defaultPort = HEALTH_PORT;
  requiredPorts = [HEALTH_PORT];
  sleepAfter = "24h";
  enableInternet = true;
  pingEndpoint = "/health";

  constructor(ctx: DurableObject["ctx"], env: Env) {
    super(ctx, env);
    this.envVars = containerEnvFromWorker(env);
  }

  override onStart(): void {
    console.log(
      JSON.stringify({
        msg: "kwami-lk-agent container started",
        environment: this.env.ENVIRONMENT,
      }),
    );
  }

  override onStop(params: { exitCode: number; reason: string }): void {
    console.log(
      JSON.stringify({
        msg: "kwami-lk-agent container stopped",
        exitCode: params.exitCode,
        reason: params.reason,
      }),
    );
  }

  override onError(error: unknown): void {
    console.error(
      JSON.stringify({
        msg: "kwami-lk-agent container error",
        error: error instanceof Error ? error.message : String(error),
      }),
    );
  }

  override async onActivityExpired(): Promise<void> {
    // Default implementation stops the container. A LiveKit worker must stay
    // registered, so renew the inactivity window instead of sleeping.
    this.renewActivityTimeout();
  }

  async ensureRunning(): Promise<{ status: string }> {
    await this.ctx.blockConcurrencyWhile(async () => {
      const state = await this.getState();
      if (state.status === "healthy" || state.status === "running") {
        return;
      }
      await this.startAndWaitForPorts({
        ports: [HEALTH_PORT],
        startOptions: {
          envVars: containerEnvFromWorker(this.env),
          enableInternet: true,
        },
        cancellationOptions: {
          instanceGetTimeoutMS: 20_000,
          portReadyTimeoutMS: 45_000,
        },
      });
    });
    this.renewActivityTimeout();
    const state = await this.getState();
    return { status: state.status };
  }
}

function workerContainer(env: Env) {
  return getContainer(env.KWAMI_AGENT, WORKER_INSTANCE);
}

function json(data: unknown, status = 200): Response {
  return Response.json(data, {
    status,
    headers: { "cache-control": "no-store" },
  });
}

/**
 * Whether a request may boot a container.
 *
 * `/start` was an unauthenticated POST that starts a paid compute instance, on
 * a Worker published to `*.workers.dev` with preview URLs enabled -- reachable
 * by anyone who guessed the hostname.
 *
 * Fails closed. With no `KWAMI_ADMIN_TOKEN` configured the route is refused
 * rather than left open: `/health` and the keepalive cron already start the
 * container on their own, so `/start` is an operator convenience, and a
 * convenience is not worth an open door.
 */
function isAuthorised(request: Request, env: Env): boolean {
  const expected = (env as Env & { KWAMI_ADMIN_TOKEN?: string }).KWAMI_ADMIN_TOKEN;
  if (typeof expected !== "string" || expected.length === 0) {
    return false;
  }
  const offered = request.headers.get("authorization") ?? "";
  const prefix = "Bearer ";
  if (!offered.startsWith(prefix)) {
    return false;
  }
  return timingSafeEqual(offered.slice(prefix.length), expected);
}

/**
 * Compares two strings without leaking their common prefix through timing.
 *
 * `a === b` on a secret returns as soon as it finds a differing byte, which is
 * measurable across enough requests. The lengths are compared first and
 * deliberately non-secretly: the length of the token is not the secret.
 */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url);
    const container = workerContainer(env);

    if (url.pathname === "/health" || url.pathname === "/status") {
      try {
        const state = await container.ensureRunning();
        const probe = await container.fetch(
          new Request(new URL("/health", request.url), { method: "GET" }),
        );
        const body = await probe.text();
        return new Response(body, {
          status: probe.ok ? 200 : 503,
          headers: {
            "content-type": probe.headers.get("content-type") ?? "application/json",
            "cache-control": "no-store",
            "x-container-status": state.status,
          },
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        const needsPaidPlan = /paid plan|unauthorized|do not have access to Cloudflare Containers/i.test(
          message,
        );
        // The full message goes to the log, not to the response. `/health` is
        // reachable by anyone who finds the workers.dev hostname, and runtime
        // error text names internal identifiers and account state.
        console.error(
          JSON.stringify({ msg: "kwami-lk-agent health check failed", error: message }),
        );
        return json(
          {
            status: "degraded",
            service: "kwami-lk-agent",
            worker: "ok",
            container: "unavailable",
            hint: needsPaidPlan
              ? "Cloudflare Containers requires the Workers Paid plan: https://dash.cloudflare.com/?to=/:account/workers/plans"
              : undefined,
          },
          503,
        );
      }
    }

    if (url.pathname === "/start" && request.method === "POST") {
      if (!isAuthorised(request, env)) {
        return json({ error: "unauthorized" }, 401);
      }
      const state = await container.ensureRunning();
      return json({ status: "started", container: state.status });
    }

    if (url.pathname === "/" || url.pathname === "/ready") {
      const state = await container.ensureRunning();
      return json({
        service: "kwami-lk-agent",
        environment: env.ENVIRONMENT,
        container: state.status,
      });
    }

    return json({ error: "not found" }, 404);
  },

  async scheduled(_controller, env, ctx): Promise<void> {
    // The rejection has to be handled here. `ensureRunning()` fails whenever
    // the account is not on the Workers Paid plan, or the image is still
    // building, or the instance limit is reached -- and an unhandled rejection
    // inside `waitUntil` surfaces as a failed cron invocation with no
    // explanation attached, which is the least useful shape that information
    // could take. Log it instead, in the same JSON the other handlers use.
    ctx.waitUntil(
      workerContainer(env)
        .ensureRunning()
        .then((state) => {
          console.log(
            JSON.stringify({
              msg: "kwami-lk-agent keepalive",
              container: state.status,
            }),
          );
        })
        .catch((error: unknown) => {
          console.error(
            JSON.stringify({
              msg: "kwami-lk-agent keepalive failed",
              error: error instanceof Error ? error.message : String(error),
            }),
          );
        }),
    );
  },
} satisfies ExportedHandler<Env>;
