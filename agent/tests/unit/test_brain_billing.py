"""How the brain pays: the user's choice of plan or API key, and the plan running out.

The user's rule (2026-11-17): everything through the PC's Claude plan until it runs out,
then the API key; and a panel that chooses which first, or only one, and is obeyed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx

from src.adapters.supabase import SupabaseClient
from src.brain import bridge as bridge_module
from src.brain import jobs as jobs_module
from src.brain import plan
from src.brain import worker as worker_module
from src.brain.bridge import BrainTask, Bridge
from src.brain.config import BrainModel, default_brain, parse_brain, reach
from src.brain.executor import ExecutorResult
from src.brain.jobs import BrainJob, JobStore
from src.brain.ledger import Ledger
from src.brain.providers import BrainAnswer, BrainRequest
from src.brain.router import ACCOUNT_MARK, model_available, on_account, pick, running_cost
from src.brain.worker import Worker
from src.settings import Settings

USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
LATER = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
OUT = {"status": "rejected", "type": "five_hour", "resetsAt": LATER, "seenAt": LATER}

OPUS = BrainModel("opus", "anthropic", "claude-opus-5-5", "expert", "Opus")
HAIKU = BrainModel("haiku", "anthropic", "claude-haiku-5-5", "fast", "Haiku")
GEMINI = BrainModel("gemini", "google", "gemini-3.5-flash", "search", "Gemini")


# -- the plan ------------------------------------------------------------------------


def test_a_rate_limit_event_is_kept_as_it_was_said() -> None:
    info = SimpleNamespace(
        status="allowed_warning", rate_limit_type="five_hour", utilization=0.8, resets_at=1.8e9
    )
    kept = plan.from_event(info, NOW)
    assert kept["status"] == "allowed_warning" and kept["type"] == "five_hour"
    assert kept["utilization"] == 0.8 and kept["resetsAt"].startswith("2027-01-15")
    assert kept["seenAt"] == NOW.isoformat()
    bare = plan.from_event(SimpleNamespace(status="allowed"))
    assert bare["utilization"] is None and bare["resetsAt"] is None and bare["type"] == ""


def test_the_reset_is_read_from_claude_codes_words() -> None:
    # The old CLI said it as a number.
    assert plan.resets_from_text("Claude AI usage limit reached|1800000000", NOW) == (
        datetime.fromtimestamp(1800000000, UTC)
    )
    madrid = plan.resets_from_text("You've hit your limit · resets 3pm (Europe/Madrid)", NOW)
    assert madrid is not None
    assert madrid.astimezone(ZoneInfo("Europe/Madrid")).strftime("%d %H:%M") == "17 15:00"
    # A time already gone today is tomorrow's.
    early = plan.resets_from_text("5-hour limit reached ∙ resets 3:30am (UTC)", NOW)
    assert early == datetime(2026, 11, 18, 3, 30, tzinfo=UTC)
    noon = plan.resets_from_text("resets 12am (UTC)", NOW)
    assert noon == datetime(2026, 11, 18, 0, 0, tzinfo=UTC)
    # A weekday is the next one; 17 Nov 2026 is a Tuesday.
    monday = plan.resets_from_text("Weekly limit reached · resets Mon 9am (UTC)", NOW)
    assert monday == datetime(2026, 11, 23, 9, 0, tzinfo=UTC)
    tuesday = plan.resets_from_text("resets Tue 9am (UTC)", NOW)
    assert tuesday == datetime(2026, 11, 24, 9, 0, tzinfo=UTC)


def test_an_unknown_zone_or_time_reads_as_local_or_nothing() -> None:
    odd = plan.resets_from_text("resets 15 (Nowhere/Land)", NOW)
    assert odd is not None and odd > NOW
    local = plan.resets_from_text("resets 23", NOW)
    assert local is not None and local > NOW
    assert plan.resets_from_text("resets 25pm", NOW) is None
    assert plan.resets_from_text("resets 30", NOW) is None
    assert plan.resets_from_text("nothing said", NOW) is None


def test_only_words_of_a_plan_running_out_count() -> None:
    assert plan.from_failure("Error de red", NOW) is None
    out = plan.from_failure("You've hit your limit · resets 3pm (UTC)", NOW)
    assert out is not None and out["status"] == "rejected"
    assert out["resetsAt"] == datetime(2026, 11, 17, 15, 0, tzinfo=UTC).isoformat()
    # The worker's own Spanish, as the bridge reads it back.
    spanish = plan.from_failure("Tu plan de Claude está agotado hasta las 15:00.")
    assert spanish is not None and spanish["resetsAt"] is None


def test_until_when_the_plan_is_out() -> None:
    assert plan.out_until(None) is None
    assert plan.out_until({"status": "allowed", "resetsAt": LATER}) is None
    assert plan.out_until(OUT) == datetime.fromisoformat(LATER)
    gone = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    assert plan.out_until({"status": "rejected", "resetsAt": gone}) is None
    # Out with no reset said: left alone a while after it was seen.
    seen = {"status": "rejected", "resetsAt": None, "seenAt": NOW.isoformat()}
    assert plan.out_until(seen, NOW) == NOW + plan.PLAN_OUT_GUESS
    assert plan.out_until({"status": "rejected", "seenAt": "not a date"}, NOW) is None
    assert plan.out_until({"status": "rejected", "seenAt": "2026-11-17T10:00:00"}, NOW) == (
        NOW + plan.PLAN_OUT_GUESS
    )
    assert plan.out_until({"status": "rejected", "resetsAt": 1.0}, NOW) is None


def test_the_reset_is_said_as_the_user_says_times() -> None:
    zone = ZoneInfo("Europe/Madrid")
    today = datetime.now(zone).replace(hour=23, minute=59, second=0, microsecond=0)
    assert plan.clock(today, zone) == "las 23:59"
    assert plan.clock(today + timedelta(days=1), zone).startswith("mañana a las")
    assert plan.clock(today + timedelta(days=3), zone).startswith("el ")
    assert plan.clock(today).startswith("las ") or plan.clock(today).startswith("mañana")
    assert plan.out_words(None) == "Tu plan de Claude se ha agotado."
    assert plan.out_words(today, zone) == "Tu plan de Claude está agotado hasta las 23:59."


# -- the choice --------------------------------------------------------------------


def test_the_panel_choice_is_parsed_and_defaults_to_the_plan_first() -> None:
    assert default_brain().billing == "account_first"
    assert parse_brain({"billing": "api_only"}).billing == "api_only"
    assert parse_brain({"billing": "lo que sea"}).billing == "account_first"


def test_only_wins_over_a_model_and_first_does_not() -> None:
    by_key = BrainModel("o", "anthropic", "m", "expert", access="api")
    by_plan = BrainModel("o", "anthropic", "m", "expert", access="account")
    assert reach(by_key, "account_only") == "account"
    assert reach(by_plan, "api_only") == "api"
    assert reach(by_key, "account_first") == "api"
    assert reach(by_plan, "api_first") == "account"
    assert reach(OPUS, "api_first") == "api_first"
    assert reach(OPUS, "account_first") == "account_first"


def test_who_answers_on_the_plan() -> None:
    both = {ACCOUNT_MARK: "1", "ANTHROPIC_API_KEY": "k"}
    assert on_account(OPUS, both, "account_first")
    assert not on_account(OPUS, both, "api_first")
    assert on_account(OPUS, {ACCOUNT_MARK: "1"}, "api_first")
    assert not on_account(OPUS, both, "api_only")
    assert not on_account(OPUS, {"ANTHROPIC_API_KEY": "k"}, "account_first")
    assert not on_account(GEMINI, both, "account_first")
    assert running_cost(OPUS, both, "account_first") == 0.0
    assert running_cost(OPUS, both, "api_first") > 0.0


def test_the_plan_only_offers_nothing_paid() -> None:
    keys = {ACCOUNT_MARK: "1", "GOOGLE_API_KEY": "g", "ANTHROPIC_API_KEY": "k"}
    assert model_available(GEMINI, keys, "account_first")
    assert not model_available(GEMINI, keys, "account_only")
    assert not model_available(OPUS, {"ANTHROPIC_API_KEY": "k"}, "account_only")
    assert model_available(OPUS, {"ANTHROPIC_API_KEY": "k"}, "api_first")
    assert not model_available(OPUS, {ACCOUNT_MARK: "1"}, "api_only")


def test_with_the_plan_first_a_search_goes_to_the_plan_before_a_paid_searcher() -> None:
    config = default_brain()
    config.models = [HAIKU, GEMINI, OPUS]
    plan_on = {ACCOUNT_MARK: "1", "GOOGLE_API_KEY": "g"}
    chosen = pick(config, plan_on, "research", "simple")
    assert chosen is not None and chosen.id == "haiku"
    # Without the plan, the searcher it was given.
    chosen = pick(config, {"GOOGLE_API_KEY": "g", "ANTHROPIC_API_KEY": "k"}, "research", "simple")
    assert chosen is not None and chosen.id == "gemini"
    # The key first: the cheapest searcher, as before.
    config.billing = "api_first"
    chosen = pick(config, {**plan_on, "ANTHROPIC_API_KEY": "k"}, "research", "simple")
    assert chosen is not None and chosen.id == "gemini"
    # Code is never handed to a thinking model, plan or not.
    config.billing = "account_first"
    assert pick(config, plan_on, "code", "normal") is None


# -- the queue -----------------------------------------------------------------------


@pytest.fixture
def store() -> JobStore:
    return JobStore(SupabaseClient("https://db.test", "secret"))


@respx.mock
async def test_the_plan_travels_with_the_login(store: JobStore) -> None:
    route = respx.get("https://db.test/rest/v1/brain_runners").mock(
        side_effect=[
            httpx.Response(200, json=[{"runner_id": "r", "claude_plan": OUT}]),
            httpx.Response(200, json=[{"runner_id": "r", "claude_plan": "rubbish"}]),
            httpx.Response(200, json=[]),
            httpx.Response(400),
            httpx.Response(200, json=[{"runner_id": "r"}]),
        ]
    )
    assert await store.account_state(USER) == (True, OUT)
    assert await store.account_state(USER) == (True, None)
    assert await store.account_ready(USER) is False
    # Before migration 032: the login alone.
    assert await store.account_state(USER) == (True, None)
    assert b"claude_plan" not in route.calls[4].request.url.query


@respx.mock
async def test_a_database_without_the_plan_is_not_asked_on_every_look(
    store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = respx.get("https://db.test/rest/v1/brain_runners").mock(
        side_effect=[
            httpx.Response(400),
            httpx.Response(200, json=[{"runner_id": "r"}]),
            httpx.Response(200, json=[{"runner_id": "r"}]),
            httpx.Response(200, json=[{"runner_id": "r", "claude_plan": OUT}]),
        ]
    )
    assert await store.account_state(USER) == (True, None)
    assert await store.account_state(USER) == (True, None)
    assert route.call_count == 3 and b"claude_plan" not in route.calls[2].request.url.query
    # Once applied, it is found again on the next try.
    monkeypatch.setattr(jobs_module.time, "monotonic", lambda: store._plan_retry_at + 1)
    assert await store.account_state(USER) == (True, OUT)
    assert store.plan_column is True


@respx.mock
async def test_the_billing_is_read_from_the_kwamis_saved_brain(store: JobStore) -> None:
    route = respx.get("https://db.test/rest/v1/user_kwamis").mock(
        side_effect=[
            httpx.Response(200, json=[{"billing": "api_first"}]),
            httpx.Response(200, json=[]),
        ]
    )
    assert await store.billing(USER, KWAMI) == "api_first"
    assert "brain-%3E%3Ebilling" in str(route.calls[0].request.url)
    assert await store.billing(USER, KWAMI) == ""
    assert await store.billing(USER, "kwami_not_a_row") == ""
    assert route.call_count == 2


@respx.mock
async def test_the_heartbeat_carries_the_plan_and_retries_without_it(
    store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = respx.post("https://db.test/rest/v1/brain_runners").mock(
        side_effect=[
            httpx.Response(201),
            httpx.Response(400),
            httpx.Response(201),
            httpx.Response(201),
            httpx.Response(201),
        ]
    )
    assert await store.beat(USER, "r", "host", True, OUT) is True
    assert store.plan_column is True and b"rejected" in route.calls[0].request.content
    # Refused once: the next beats go without it...
    assert await store.beat(USER, "r", "host", True, OUT) is True
    assert store.plan_column is False and b"claude_plan" not in route.calls[2].request.content
    assert await store.beat(USER, "r", "host", True, OUT) is True
    assert b"claude_plan" not in route.calls[3].request.content
    # ...until it is time to ask again.
    monkeypatch.setattr(jobs_module.time, "monotonic", lambda: store._plan_retry_at + 1)
    assert await store.beat(USER, "r", "host", True, OUT) is True
    assert b"claude_plan" in route.calls[4].request.content


# -- the PC's worker -----------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_real_login(monkeypatch: pytest.MonkeyPatch) -> None:
    async def signed_in() -> bool:
        return True

    monkeypatch.setattr(worker_module, "account_login", signed_in)


class WorkerJobs:
    def __init__(self, billing: str = "") -> None:
        self.billing_said = billing
        self.reads = 0
        self.finished: list[tuple[str, str, str]] = []

    async def billing(self, user_id: str, kwami_id: str) -> str:
        self.reads += 1
        return self.billing_said

    async def finish(
        self, job_id: str, status: str, result: str, cost_usd: float, session_id: str
    ) -> bool:
        self.finished.append((job_id, status, result))
        return True

    async def update_running(self, job_id: str, progress: str) -> bool:
        return True

    async def get(self, job_id: str) -> BrainJob | None:
        return None


class Runs:
    """Each run answers in turn; who ran it is kept."""

    def __init__(self, *results: ExecutorResult) -> None:
        self.results = list(results)
        self.keys: list[str] = []

    async def __call__(
        self, api_key: str, model: str, task: str, config: Any, **kwargs: Any
    ) -> ExecutorResult:
        self.keys.append(api_key)
        return self.results.pop(0)


DONE = ExecutorResult("Hecho.", 0.4, True)
LIMIT = ExecutorResult(
    "Claude AI usage limit reached|1900000000",
    0.0,
    False,
    plan={"status": "rejected", "resetsAt": LATER, "seenAt": LATER},
)
NOT_SIGNED_IN = ExecutorResult("Not logged in · Please run /login", 0.0, False)
NO_CREDIT = ExecutorResult("Your credit balance is too low", 0.0, False)


def _worker(jobs: WorkerJobs, run: Runs, tmp_path: Path, key: str = "ak") -> Worker:
    worker = Worker(jobs, object(), key, USER, runner_id="r1", run=run, folder=tmp_path)  # type: ignore[arg-type]
    worker.claude_login = True
    return worker


async def _paid(worker: Worker, tmp_path: Path) -> ExecutorResult:
    job = BrainJob(USER, KWAMI, "T", "Haz algo", id="j1")
    return await worker._run_paid(job, "t", tmp_path, "", 2.0, lambda _line: None)


async def test_the_plan_first_runs_on_the_login_for_nothing(tmp_path: Path) -> None:
    run = Runs(DONE)
    result = await _paid(_worker(WorkerJobs(), run, tmp_path), tmp_path)
    assert run.keys == [""] and result.ok and result.cost_usd == 0.0


async def test_a_plan_that_runs_out_moves_to_the_key_and_stays_there(tmp_path: Path) -> None:
    run = Runs(LIMIT, DONE, DONE)
    worker = _worker(WorkerJobs(), run, tmp_path)
    result = await _paid(worker, tmp_path)
    assert run.keys == ["", "ak"] and result.ok and result.cost_usd == 0.4
    assert "agotado hasta" in result.text and "clave de API" in result.text
    assert worker.plan_out() is not None
    # Until it resets, the next job goes straight to the key.
    result = await _paid(worker, tmp_path)
    assert run.keys == ["", "ak", "ak"] and "agotado hasta" in result.text


async def test_a_login_that_cannot_work_says_so_when_the_key_takes_over(tmp_path: Path) -> None:
    run = Runs(NOT_SIGNED_IN, DONE)
    result = await _paid(_worker(WorkerJobs(), run, tmp_path), tmp_path)
    assert run.keys == ["", "ak"] and worker_module.LOGIN_COULD_NOT in result.text


async def test_a_job_that_simply_failed_on_the_plan_is_not_paid_again(tmp_path: Path) -> None:
    run = Runs(ExecutorResult("No encontré el archivo.", 0.0, False))
    result = await _paid(_worker(WorkerJobs(), run, tmp_path), tmp_path)
    assert run.keys == [""] and not result.ok


async def test_the_plan_only_never_pays(tmp_path: Path) -> None:
    run = Runs(LIMIT)
    worker = _worker(WorkerJobs("account_only"), run, tmp_path)
    result = await _paid(worker, tmp_path)
    assert run.keys == [""] and not result.ok
    assert "solo con la suscripción" in result.text and "agotado hasta" in result.text
    # Out already: not even tried.
    result = await _paid(worker, tmp_path)
    assert run.keys == [""] and "solo con la suscripción" in result.text
    # A login that cannot work, with no plan words: what it said, nothing paid.
    run = Runs(NOT_SIGNED_IN)
    result = await _paid(_worker(WorkerJobs("account_only"), run, tmp_path), tmp_path)
    assert run.keys == [""] and result.text == NOT_SIGNED_IN.text


async def test_with_no_key_a_plan_out_is_said_plainly(tmp_path: Path) -> None:
    run = Runs(LIMIT)
    worker = _worker(WorkerJobs(), run, tmp_path, key="")
    result = await _paid(worker, tmp_path)
    assert run.keys == [""] and "No hay clave de API" in result.text
    result = await _paid(worker, tmp_path)
    assert run.keys == [""] and "No hay clave de API" in result.text
    run = Runs(NOT_SIGNED_IN)
    result = await _paid(_worker(WorkerJobs(), run, tmp_path, key=""), tmp_path)
    assert result.text == NOT_SIGNED_IN.text


async def test_the_key_first_falls_to_the_plan_only_when_the_key_cannot(tmp_path: Path) -> None:
    run = Runs(DONE)
    result = await _paid(_worker(WorkerJobs("api_first"), run, tmp_path), tmp_path)
    assert run.keys == ["ak"] and result.cost_usd == 0.4
    run = Runs(NO_CREDIT, DONE)
    result = await _paid(_worker(WorkerJobs("api_first"), run, tmp_path), tmp_path)
    assert run.keys == ["ak", ""] and result.ok and result.cost_usd == 0.0
    # Any other failure is the task's own: not run twice.
    run = Runs(ExecutorResult("No compila.", 0.2, False))
    await _paid(_worker(WorkerJobs("api_first"), run, tmp_path), tmp_path)
    assert run.keys == ["ak"]
    # No key: the plan.
    run = Runs(DONE)
    await _paid(_worker(WorkerJobs("api_first"), run, tmp_path, key=""), tmp_path)
    assert run.keys == [""]


async def test_the_key_only_never_touches_the_plan(tmp_path: Path) -> None:
    run = Runs(NO_CREDIT)
    result = await _paid(_worker(WorkerJobs("api_only"), run, tmp_path), tmp_path)
    assert run.keys == ["ak"] and not result.ok
    run = Runs()
    result = await _paid(_worker(WorkerJobs("api_only"), run, tmp_path, key=""), tmp_path)
    assert run.keys == [] and result.text == worker_module.ONLY_KEY_MISSING


async def test_the_billing_is_read_once_a_minute_per_kwami(tmp_path: Path) -> None:
    jobs = WorkerJobs("nonsense")
    worker = _worker(jobs, Runs(), tmp_path)
    job = BrainJob(USER, KWAMI, "T", "x")
    assert await worker.billing_of(job) == "account_first"
    assert await worker.billing_of(job) == "account_first" and jobs.reads == 1
    # A queue that cannot say (the local one): the plan first.
    bare = Worker(object(), object(), "", USER, run=Runs(), folder=tmp_path)  # type: ignore[arg-type]
    assert await bare.billing_of(job) == "account_first"


async def test_a_thought_on_a_plan_that_is_out_is_refused_at_once(tmp_path: Path) -> None:
    jobs = WorkerJobs()
    run = Runs(LIMIT)
    worker = _worker(jobs, run, tmp_path)
    watched = datetime.now(UTC).isoformat()
    first = BrainJob(USER, KWAMI, "T", "x", id="t1", mode="think", watcher_seen_at=watched)
    await worker.think(first)
    assert jobs.finished[-1][1] == "failed" and "agotado hasta" in jobs.finished[-1][2]
    second = BrainJob(USER, KWAMI, "T", "x", id="t2", mode="think", watcher_seen_at=watched)
    await worker.think(second)
    assert run.keys == [""] and jobs.finished[-1][0] == "t2"
    assert "agotado hasta" in jobs.finished[-1][2]


def test_only_the_logins_runs_tell_the_plan(tmp_path: Path) -> None:
    worker = _worker(WorkerJobs(), Runs(), tmp_path)
    worker._note_plan(DONE)
    assert worker.plan is None
    worker._note_plan(LIMIT)
    worker._note_plan(LIMIT)
    assert worker.plan == LIMIT.plan


# -- the cloud ---------------------------------------------------------------------


async def test_the_cloud_will_not_pay_for_someone_on_the_plan_only(tmp_path: Path) -> None:
    from src.brain.cloud import ONLY_PLAN_HERE, CloudWorker

    run = Runs()
    cloud = CloudWorker(
        WorkerJobs("account_only"),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        "deployment-key",
        runner_id="cloud-test-1",
        run=run,
    )
    result = await _paid(cloud, tmp_path)
    assert result.text == ONLY_PLAN_HERE and run.keys == []


# -- the bridge ----------------------------------------------------------------------


class BridgeJobs:
    """The PC as the call sees it: online, signed in, and what its plan says."""

    def __init__(self, result: str = "Pensado.", status: str = "done") -> None:
        self.online = True
        self.signed_in = True
        self.plan: dict[str, Any] | None = None
        self.result = result
        self.status = status
        self.enqueued: list[BrainJob] = []
        self.cloud_on = True
        self.allowed = True

    async def runner_online(self, user_id: str) -> bool:
        return self.online

    async def account_state(self, user_id: str) -> tuple[bool, dict[str, Any] | None]:
        return self.signed_in, self.plan

    async def cloud_online(self) -> bool:
        return self.cloud_on

    async def cloud_allowed(self, user_id: str) -> bool:
        return self.allowed

    async def enqueue(self, job: BrainJob) -> bool:
        job.id = job.id or "j1"
        self.enqueued.append(job)
        return True

    async def get(self, job_id: str) -> BrainJob | None:
        job = self.enqueued[-1]
        job.status, job.result = self.status, self.result
        return job

    async def mark_delivered(self, job_id: str) -> bool:
        return True

    async def request_cancel(self, job_id: str) -> bool:
        return True


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_module, "THINK_POLL_SECONDS", 0.0)
    monkeypatch.setattr(bridge_module, "THINK_POLL_FAST_SECONDS", 0.0)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    said: list[str] = []

    async def run(key: str, model_id: str, model: str, request: BrainRequest) -> BrainAnswer:
        said.append(key)
        if key == "broke":
            return BrainAnswer("Your credit balance is too low", model_id, model, ok=False)
        return BrainAnswer("Por la API.", model_id, model, cost_usd=0.5)

    monkeypatch.setattr(bridge_module, "run_anthropic", run)
    return said


def _bridge(tmp_path: Path, jobs: BridgeJobs, keys: dict[str, str] | None = None) -> Bridge:
    bridge = Bridge(
        keys=keys if keys is not None else {"ANTHROPIC_API_KEY": "k"},
        ledger=Ledger(tmp_path),
        settings=Settings(executor_mode="local"),
    )
    bridge.jobs = jobs  # type: ignore[assignment]
    bridge.user_id, bridge.kwami_id = USER, KWAMI
    return bridge


async def test_a_plan_the_pc_says_is_out_takes_no_thoughts(tmp_path: Path, api: list[str]) -> None:
    jobs = BridgeJobs()
    jobs.plan = OUT
    bridge = _bridge(tmp_path, jobs)
    await bridge._refresh_account(force=True)
    assert ACCOUNT_MARK not in bridge._keys and bridge.claude_signed_in
    answer = await bridge._ask(OPUS, BrainRequest("T", "S"))
    assert answer.text == "Por la API." and jobs.enqueued == [] and api == ["k"]
    state = bridge.billing_state()
    assert state["now"] == "api" and "agotado hasta" in state["why"] and state["plan"]["out"]
    assert bridge.status()["account"] == {"pcOnline": True, "loggedIn": True, "usable": False}


async def test_a_thought_that_comes_back_out_moves_the_call_to_the_key(
    tmp_path: Path, api: list[str]
) -> None:
    jobs = BridgeJobs(result="Tu plan de Claude está agotado hasta las 15:00.", status="failed")
    bridge = _bridge(tmp_path, jobs)
    await bridge._refresh_account(force=True)
    answer = await bridge._ask(OPUS, BrainRequest("T", "S"))
    assert answer.text == "Por la API." and len(jobs.enqueued) == 1
    assert ACCOUNT_MARK not in bridge._keys and plan.out_until(bridge.claude_plan) is not None
    # The PC's older word does not undo what this call saw.
    jobs.plan = {"status": "allowed", "seenAt": "2000-01-01T00:00:00+00:00"}
    await bridge._refresh_account(force=True)
    assert ACCOUNT_MARK not in bridge._keys
    await bridge._ask(OPUS, BrainRequest("T", "S"))
    assert len(jobs.enqueued) == 1


async def test_a_thought_that_failed_for_another_reason_keeps_the_plan(
    tmp_path: Path, api: list[str]
) -> None:
    jobs = BridgeJobs(result="Error de red", status="failed")
    bridge = _bridge(tmp_path, jobs)
    await bridge._refresh_account(force=True)
    await bridge._ask(OPUS, BrainRequest("T", "S"))
    assert ACCOUNT_MARK in bridge._keys


async def test_the_plan_only_says_why_instead_of_paying(tmp_path: Path, api: list[str]) -> None:
    jobs = BridgeJobs()
    jobs.plan = OUT
    bridge = _bridge(tmp_path, jobs)
    bridge.config.billing = "account_only"
    await bridge._refresh_account(force=True)
    answer = await bridge._ask(OPUS, BrainRequest("T", "S"))
    assert not answer.ok and "solo con la suscripción" in answer.text and api == []
    assert bridge.billing_state()["now"] == "none"
    # Nothing paid per use: no Tavily either.
    bridge._tavily_key = "tv"
    assert bridge._request("x", "normal").tavily_key == ""


async def test_why_the_plan_cannot_answer(tmp_path: Path) -> None:
    jobs = BridgeJobs()
    bridge = _bridge(tmp_path, jobs, keys={})
    jobs.online = False
    await bridge._refresh_account(force=True)
    assert "no está en marcha" in bridge._account_unavailable()
    assert "no está en marcha" in bridge.billing_state()["why"]
    jobs.online, jobs.signed_in = True, False
    await bridge._refresh_account(force=True)
    assert "sesión iniciada" in bridge._account_unavailable()
    assert "sesión iniciada" in bridge.billing_state()["why"]
    assert bridge.billing_state()["now"] == "none"
    jobs.signed_in = True
    await bridge._refresh_account(force=True)
    assert "no está disponible" in bridge._account_unavailable()
    assert "No hay clave" in bridge._account_unavailable()


async def test_the_state_for_each_choice(tmp_path: Path) -> None:
    jobs = BridgeJobs()
    bridge = _bridge(tmp_path, jobs)
    await bridge._refresh_account(force=True)
    assert bridge.billing_state()["now"] == "account"
    bridge.config.billing = "api_first"
    assert bridge.billing_state()["now"] == "api"
    bridge.config.billing = "api_only"
    assert bridge.billing_state()["now"] == "api"
    assert bridge.billing_state()["otherProvidersPaid"] is True
    bridge._keys.pop("ANTHROPIC_API_KEY")
    assert bridge.billing_state()["now"] == "none"
    bridge.config.billing = "account_only"
    state = bridge.billing_state()
    assert state["now"] == "account" and state["otherProvidersPaid"] is False


async def test_the_key_first_falls_to_the_plan_when_the_key_cannot(
    tmp_path: Path, api: list[str]
) -> None:
    jobs = BridgeJobs()
    bridge = _bridge(tmp_path, jobs, keys={"ANTHROPIC_API_KEY": "broke"})
    bridge.config.billing = "api_first"
    await bridge._refresh_account(force=True)
    answer = await bridge._ask(OPUS, BrainRequest("T", "S"))
    assert answer.text == "Pensado." and api == ["broke"] and len(jobs.enqueued) == 1
    # The plan could not either: the key's own failure stands.
    jobs.status, jobs.result = "failed", "Error"
    answer = await bridge._ask(OPUS, BrainRequest("T", "S"))
    assert "credit balance" in answer.text


async def test_the_plan_only_keeps_work_off_the_cloud(tmp_path: Path) -> None:
    jobs = BridgeJobs()
    bridge = _bridge(tmp_path, jobs)
    bridge.config.billing = "account_only"
    assert await bridge.placement("cloud") == ("", bridge_module.ONLY_PLAN_NO_CLOUD)
    assert (await bridge.placement())[0] == "auto"
    jobs.online = False
    target, why = await bridge.placement()
    assert target == "" and "solo con la suscripción" in why
    bridge.config.billing = "account_first"
    assert (await bridge.placement())[0] == "cloud"


async def test_the_voice_changes_how_the_brain_pays(tmp_path: Path) -> None:
    sent: list[dict[str, Any]] = []

    class Publisher:
        async def publish(self, message: dict[str, Any]) -> None:
            if message.get("billing") == "api_only":
                raise RuntimeError("the room is gone")
            sent.append(message)

    jobs = BridgeJobs()
    bridge = _bridge(tmp_path, jobs)
    await bridge._refresh_account(force=True)
    assert "account" in await bridge.set_billing("account_only")
    assert bridge.config.billing == "account_only" and sent == []
    bridge.publisher = Publisher()
    await bridge.set_billing("ACCOUNT_ONLY")
    assert sent == [{"type": "brain_billing", "billing": "account_only"}]
    # A room that is gone does not undo the change.
    await bridge.set_billing("api_only")
    assert bridge.config.billing == "api_only" and len(sent) == 1
    bridge._keys.pop("ANTHROPIC_API_KEY")
    bridge.publisher = None
    assert "(No hay clave" in await bridge.set_billing("api_only")
    assert "Unknown billing" in await bridge.set_billing("gratis")


async def test_the_user_hears_once_that_the_plan_ran_out(tmp_path: Path) -> None:
    jobs = BridgeJobs()
    jobs.plan = OUT
    bridge = _bridge(tmp_path, jobs)
    await bridge._refresh_account(force=True)
    task = BrainTask(id="t1", title="T", task="x", kind="analysis", difficulty="normal")
    task.status, task.result, task.paid_by, task.ran_on = "done", "Hecho.", "api", "here"
    first = bridge._spoken(task)
    assert "Also tell the user, once" in first and "agotado hasta" in first
    assert "Also tell the user, once" not in bridge._spoken(task)


async def test_brain_status_reads_and_changes_the_billing(tmp_path: Path) -> None:
    from src.agent import KwamiAgent
    from src.runtime.container import AgentDeps

    jobs = BridgeJobs()
    bridge = _bridge(tmp_path, jobs)
    context = SimpleNamespace(userdata=AgentDeps(bridge=bridge))
    status = await KwamiAgent().brain_status(context, billing="api_first")  # type: ignore[arg-type]
    assert bridge.config.billing == "api_first" and "Billing is now api_first" in status["changed"]
    assert status["billing"]["mode"] == "api_first"
    assert status["whereWorkRuns"]["billing"]["mode"] == "api_first"
    status = await KwamiAgent().brain_status(  # type: ignore[arg-type]
        context, "brain", "en euros", billing="account_first"
    )
    assert status["changed"].startswith("Instructions for the whole brain saved")
    assert "Billing is now account_first" in status["changed"]
