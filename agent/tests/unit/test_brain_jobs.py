"""`JobStore` on the wire: the table, filters and bodies Supabase will receive."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx

from src.adapters.supabase import SupabaseClient
from src.brain.jobs import WATCH_FRESH_SECONDS, BrainJob, JobStore

URL = "https://proj.supabase.co"
REST = f"{URL}/rest/v1"
USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def jobs() -> JobStore:
    return JobStore(SupabaseClient(URL, "sb_secret_test"))


@respx.mock
async def test_the_pc_counts_as_there_only_when_recently_seen(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_runners").mock(
        side_effect=[httpx.Response(200, json=[{"runner_id": "r"}]), httpx.Response(200, json=[])]
    )
    assert await jobs.runner_online(USER) is True
    assert await jobs.runner_online(USER) is False
    params = route.calls[0].request.url.params
    assert params["user_id"] == f"eq.{USER}"
    assert params["last_seen_at"].startswith("gte.")


@respx.mock
async def test_enqueue_mints_the_ids_and_starts_the_watch(jobs: JobStore) -> None:
    route = respx.post(f"{REST}/brain_jobs").mock(return_value=httpx.Response(201))
    job = BrainJob(USER, KWAMI, "T" * 300, "Haz algo", difficulty="hard", model="claude-opus-5-5")

    assert await jobs.enqueue(job) is True

    body = json.loads(route.calls[0].request.content)
    assert body["id"] == job.id and job.id
    assert body["thread_id"] == job.thread_id and job.thread_id
    assert len(body["title"]) == 200
    assert body["task"] == "Haz algo"
    assert body["difficulty"] == "hard"
    assert body["watcher_seen_at"]


@respx.mock
async def test_a_follow_up_keeps_its_thread(jobs: JobStore) -> None:
    route = respx.post(f"{REST}/brain_jobs").mock(return_value=httpx.Response(201))
    await jobs.enqueue(BrainJob(USER, KWAMI, "T", "x", thread_id="t1"))
    assert json.loads(route.calls[0].request.content)["thread_id"] == "t1"


@respx.mock
async def test_get_reads_the_row_and_tolerates_absence(jobs: JobStore) -> None:
    respx.get(f"{REST}/brain_jobs").mock(
        side_effect=[
            httpx.Response(
                200,
                json=[
                    {
                        "id": "j1",
                        "user_id": USER,
                        "kwami_id": KWAMI,
                        "title": "T",
                        "task": "x",
                        "status": "done",
                        "result": "R",
                        "cost_usd": "0.25",
                        "cancel_requested": False,
                        "progress": None,
                    }
                ],
            ),
            httpx.Response(200, json=[]),
            httpx.Response(500),
        ]
    )
    job = await jobs.get("j1")
    assert job is not None
    assert job.finished and job.cost_usd == 0.25 and job.progress == ""
    assert await jobs.get("j2") is None
    assert await jobs.get("j3") is None


def test_a_watch_lapses() -> None:
    now = datetime.now(UTC)
    job = BrainJob(USER, KWAMI, "T", "x")
    assert not job.watched(now)
    job.watcher_seen_at = now.isoformat().replace("+00:00", "Z")
    assert job.watched(now)
    assert not job.watched(now + timedelta(seconds=WATCH_FRESH_SECONDS + 1))


@respx.mock
async def test_watch_and_delivery_stamps(jobs: JobStore) -> None:
    route = respx.patch(f"{REST}/brain_jobs").mock(return_value=httpx.Response(204))

    await jobs.renew_watch("j1")
    await jobs.end_watch("j1")
    await jobs.mark_delivered("j1")

    assert json.loads(route.calls[0].request.content)["watcher_seen_at"]
    assert json.loads(route.calls[1].request.content) == {"watcher_seen_at": None}
    assert route.calls[2].request.url.params["delivered_at"] == "is.null"


@respx.mock
async def test_cancel_ends_a_queued_job_and_flags_a_running_one(jobs: JobStore) -> None:
    route = respx.patch(f"{REST}/brain_jobs").mock(return_value=httpx.Response(204))

    assert await jobs.request_cancel("j1") is True

    queued, running = route.calls[0].request, route.calls[1].request
    assert queued.url.params["status"] == "eq.queued"
    assert json.loads(queued.content)["status"] == "cancelled"
    assert running.url.params["status"] == "eq.running"
    assert json.loads(running.content) == {"cancel_requested": True}


@respx.mock
async def test_last_thread(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_jobs").mock(
        side_effect=[httpx.Response(200, json=[{"thread_id": "t1"}]), httpx.Response(200, json=[])]
    )
    assert await jobs.last_thread(KWAMI) == "t1"
    assert await jobs.last_thread(KWAMI) == ""
    assert route.calls[0].request.url.params["order"] == "created_at.desc"


@respx.mock
async def test_beat_upserts_the_runner(jobs: JobStore) -> None:
    route = respx.post(f"{REST}/brain_runners").mock(return_value=httpx.Response(201))
    assert await jobs.beat(USER, "r1", "PC") is True
    request = route.calls[0].request
    assert request.url.params["on_conflict"] == "user_id,runner_id"
    assert "merge-duplicates" in request.headers["prefer"]


@respx.mock
async def test_claim_goes_through_the_function(jobs: JobStore) -> None:
    route = respx.post(f"{REST}/rpc/brain_claim_job").mock(
        side_effect=[
            httpx.Response(200, json=[{"id": "j1", "user_id": USER, "kwami_id": KWAMI}]),
            httpx.Response(200, json=[]),
            httpx.Response(500),
        ]
    )
    job = await jobs.claim(USER, "r1")
    assert job is not None and job.id == "j1"
    assert json.loads(route.calls[0].request.content) == {
        "p_user_id": USER,
        "p_runner_id": "r1",
    }
    assert await jobs.claim(USER, "r1") is None
    assert await jobs.claim(USER, "r1") is None


@respx.mock
async def test_progress_only_touches_a_running_job(jobs: JobStore) -> None:
    route = respx.patch(f"{REST}/brain_jobs").mock(return_value=httpx.Response(204))
    await jobs.update_running("j1", "x" * 500)
    request = route.calls[0].request
    assert request.url.params["status"] == "eq.running"
    body = json.loads(request.content)
    assert len(body["progress"]) == 300 and body["heartbeat_at"]


@respx.mock
async def test_finish_records_the_outcome(jobs: JobStore) -> None:
    route = respx.patch(f"{REST}/brain_jobs").mock(return_value=httpx.Response(204))
    await jobs.finish("j1", "done", "Hecho", 0.1234567, "s1")
    body = json.loads(route.calls[0].request.content)
    assert body["status"] == "done"
    assert body["cost_usd"] == 0.123457
    assert body["sdk_session_id"] == "s1"
    assert body["finished_at"]


@respx.mock
async def test_session_of_thread(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_jobs").mock(
        side_effect=[
            httpx.Response(200, json=[{"sdk_session_id": "s1"}]),
            httpx.Response(200, json=[]),
        ]
    )
    assert await jobs.session_of_thread("t1") == "s1"
    assert await jobs.session_of_thread("t1") == ""
    assert route.calls[0].request.url.params["thread_id"] == "eq.t1"


@respx.mock
async def test_stale_jobs(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_jobs").mock(
        side_effect=[httpx.Response(200, json=[{"id": "j1"}]), httpx.Response(500)]
    )
    assert [job.id for job in await jobs.stale_jobs(USER)] == ["j1"]
    assert await jobs.stale_jobs(USER) == []
    assert route.calls[0].request.url.params["heartbeat_at"].startswith("lt.")


@respx.mock
async def test_month_spend_tells_nothing_from_unknown(jobs: JobStore) -> None:
    respx.get(f"{REST}/brain_spend").mock(
        side_effect=[
            httpx.Response(200, json=[{"usd": 1.5}, {"usd": "2.5"}]),
            httpx.Response(200, json=[]),
            httpx.Response(500),
        ]
    )
    assert await jobs.month_spend(USER) == 4.0
    assert await jobs.month_spend(USER) == 0.0
    assert await jobs.month_spend(USER) is None
