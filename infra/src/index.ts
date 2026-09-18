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
        return json(
          {
            status: "degraded",
            service: "kwami-lk-agent",
            worker: "ok",
            container: "unavailable",
            error: message,
            hint: needsPaidPlan
              ? "Cloudflare Containers requires the Workers Paid plan: https://dash.cloudflare.com/?to=/:account/workers/plans"
              : undefined,
          },
          503,
        );
      }
    }

    if (url.pathname === "/start" && request.method === "POST") {
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
        }),
    );
  },
} satisfies ExportedHandler<Env>;
