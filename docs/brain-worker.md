# Brain work thread

The work thread is the process on the user's PC that does what the brain is
asked to do. It lives in `agent/src/brain/worker.py`. A voice call, a schedule or
the app (`POST /brain/jobs` on the API) queues a job in Supabase (`brain_jobs`);
the work thread takes it, runs it, and leaves the result for the call or the next
greeting.

A second worker, in the cloud (`agent/src/brain/cloud.py`), takes the work the PC
is not there for, for the users who allow it. What it makes comes back to the PC
the next time the PC is on. See [The cloud and the PC](#the-cloud-and-the-pc).

## What it does

- Takes up to **3 jobs at once** (`WORK_SLOTS`), only for the user given in `--user`.
- Each loop (`tick`) first fails stale jobs, then reads the month's spend,
  then claims the next job through the `brain_claim_job` RPC. It sleeps 10 s
  when the queue is empty.
- Sends a heartbeat every 30 s (`BEAT_SECONDS`). A running job reports its
  progress every 5 s and looks for a cancellation request at the same time.
- Runs the job with the Claude executor in `~/Documents/Jarvis`, under the
  executor's guard. A job with a `repo` is *self-improvement*: it works in a
  checkout of one of the known repositories (`REPOS` in `selfwork.py`), runs
  that repository's tests, and opens a pull request.
- Every job tries the PC's Claude Code login first, at no cost. Only if that
  fails with a login, quota or rate-limit error does it retry on the API key, and
  never resuming the earlier session.
- A second line takes the jobs that only answer (`mode` think or research), up to
  3 at once, so the brain's thinking does not wait behind a long job. They run on
  the login with no tools, or only the web's, and never fall back to the key: the
  call that asked decides that, by what the user chose for that model.
- Records the result, and the spend in the ledger, then hands the result over:
  if a live call is watching, it has 15 s to take it; otherwise it is stored as
  a pending result for the next greeting.

## Starting it

From the `agent/` folder:

```bash
uv run python -m src.brain.worker --user <supabase user id>
```

`--user` may be replaced by the `JARVIS_USER_ID` environment variable. The
process loads the `.env` file at the repository root if there is one, and stops on
Ctrl+C. It exits with code 2 and logs the reason when something is missing.

## What it needs

| Need | Why |
| --- | --- |
| Migration 021 | Creates the job queue; only the service role can write jobs, so a job in the queue came from the user's own agent or clock |
| `SUPABASE_URL` | Where the queue lives |
| `SUPABASE_SECRET_KEY` | Service-role key to read and write it |
| Migration 024 | The thinking line, and the worker reporting its Claude login |
| `scripts\\brain-worker.cmd --login` once on the PC | The Claude account it works on, signed in through Anthropic's own flow |
| `ANTHROPIC_API_KEY` | Optional: the fallback when the login cannot do the work |
| `KWAMI_GOOGLE_WORK_TOKEN` | Optional: the user's token for Google, from `POST /google/work-thread/token`. Without it, the thread has no Google. A task uses Google or commands, the web and team members, never both |
| `KWAMI_SELF_REPO_OWNER` | Optional: the GitHub owner whose repositories self-improvement works on. Without it, none |
| Migration 029 | Optional: bringing home what the cloud made (`brain_sync`), and the user's agents (`brain_agents`) |
| A valid `--user` | The Supabase user id whose jobs it takes |

Migration 021 is not in this repository: the code only refers to it. It must be
applied to the Supabase project before the first run.

It also refuses to start in the cloud: `KWAMI_EXECUTOR_MODE=cloud`, or `auto`
on a machine without a Documents folder, exits with code 2.

## Limits

- **Monthly cap:** 150 USD (`HARD_MONTHLY_CAP_USD`), checked against the
  database before each job. The result carries a warning once when a job
  crosses 80 % and again when it reaches the cap; after that, jobs fail
  without running. If the spend cannot be read, the queue is left
  alone.
- **Per task:** 2 USD (`THREAD_MAX_TASK_USD`), or what is left of the month if
  that is less, and at most 100 turns.
- **Login runs:** self-improvement on the PC's login moves no money and is
  recorded at 0 USD, but the SDK is given a 25 USD notional ceiling.
- **Stale jobs:** a job whose worker stops beating for 180 s is marked failed
  and never rerun, because it may have had side effects.
- **Cancellation:** honoured within about 5 s; a cancelled job is not handed
  over to the user.
- **Failures:** a crash in a job is recorded as a failed job; a failing tick is
  logged and the loop carries on.
- **One process per user:** the thread runs up to 3 jobs at once; a fourth waits
  for a free slot.
- **Logs:** from INFO up to stderr (`utils.logging.log_to_console`). The PC's
  launcher writes them to `~\.jarvis\logs\worker.log`.

## The cloud and the PC

Migration 029 (kwami-lk-api) adds where a job may run, a cloud worker, and the
way back to the PC.

### Where a job runs

`brain_jobs.target` says who may take it:

| `target` | Taken by |
| --- | --- |
| `auto` (default) | The user's PC when it is there. The cloud once no PC of the user has been seen for 90 s (`RUNNER_FRESH_SECONDS`) and the job has waited 60 s (`CLOUD_PATIENCE_SECONDS`), so a PC that is switching on gets the first go |
| `pc` | Only the PC, however long that takes |
| `cloud` | Only the cloud, even with the PC on |

The cloud takes only `work` jobs (never thinking or orders), only of users who
allowed it (`brain_cloud_settings.enabled`, off by default, switched from the
app), and never more than 2 of one user at once (`CLOUD_PER_USER`). Both claims
are single SQL statements with `SKIP LOCKED` (`brain_claim_job`,
`brain_claim_cloud_job`), so two workers never take the same job.

In a call with the PC off, the bridge sends long work and code to the cloud's
queue when a cloud worker is there and the user allowed it, so the job outlives
the call; otherwise it runs in the call, as before.

### The cloud work thread

`CloudWorker` (`brain/cloud.py`) is this worker without the PC. It runs inside the
voice agent's own process when `KWAMI_CLOUD_WORKER=true` (see
[deployment](./deployment.md#the-brains-cloud-work-thread)), or anywhere with
`uv run python -m src.brain.cloud`. What differs from the PC:

- **No Claude login:** it spends the user's own Anthropic key, connected from the
  app, or the deployment's `ANTHROPIC_API_KEY` for the user ids in
  `KWAMI_CLOUD_OWNERS`. Same 2 USD per task and monthly cap.
- **No shell:** a host other users share runs Claude Code without Bash, so no
  model-written program can read the agent's environment (`executor.NO_SHELL`).
  The image still carries bubblewrap and socat for Claude Code's own sandbox.
- **No user files:** each thread works in a folder kept in Storage
  (`jarvis-files/<user>/<kwami>/threads/<thread>`), restored for a follow-up;
  only what changed goes up again, never the sandbox's empty placeholders.
- **No GitHub credentials:** a change to Jarvis's own code is made on an anonymous
  clone of `KWAMI_CLOUD_REPO_OWNER`'s public repository (`dev`) and kept as a
  patch, after the same forbidden-file and secret checks as on the PC.
- **No session to resume:** a follow-up is told what the thread found so far.

### Bringing the cloud's work home

What the cloud made is listed in `brain_sync`. The PC's worker brings it home
beside its own jobs (`brain/sync.py`), one item at a time, every minute (every
30 minutes before migration 029):

- **files** go to `Documents\Jarvis\nube\<title> (<thread>)`. A file the user
  changed is never overwritten: the cloud's version lands beside it as
  `name (nube).ext`. `.nube.json` remembers what Jarvis wrote, to tell the two
  apart.
- **patch** is applied with `git am --3way` on the thread's branch of the clone
  the PC already keeps; the repository's tests run and the pull request is opened
  as for work done on the PC. A patch that no longer fits is not forced: it ends
  `conflict` and the user is told.

Each item ends `applied`, `conflict` or `failed`, which the app shows, and the next
greeting says what came home.

### The user's agents

Agents defined in the app (`brain_agents`, `/brain/agents` on the API) join the
executor's team on the PC and in the cloud, under the same rules as the folder's
own `agents/<name>.json` (on the PC a file of the same name wins). A job with
`agent_id` runs as that agent: its brief, its model and the narrowest executor
mode that has its tools. A voice call can hand a task to one by name
(`delegate_task(agent=...)`).
