# Brain work thread

The work thread is the process on the user's PC that does what the brain is
asked to do. It lives in `agent/src/brain/worker.py`. A voice call or a
schedule queues a job in Supabase (`brain_jobs`); the work thread takes it,
runs it, and leaves the result for the call or the next greeting.

## What it does

- Takes **one job at a time**, only for the user given in `--user`.
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
| `KWAMI_SELF_REPO_OWNER` | Optional: the GitHub owner whose repositories self-improvement works on. Without it, none |
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
- **One process per user:** the thread works through jobs serially, so a long
  job delays the ones behind it.
