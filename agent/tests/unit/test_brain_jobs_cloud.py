"""`JobStore` for the cloud, the user's agents and the PC's sync, on the wire (migration 029)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.adapters.supabase import SupabaseClient
from src.brain.jobs import (
    CLOUD_PATIENCE_SECONDS,
    CLOUD_PER_USER,
    BrainJob,
    JobStore,
)

URL = "https://proj.supabase.co"
REST = f"{URL}/rest/v1"
USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"


@pytest.fixture
def jobs() -> JobStore:
    return JobStore(SupabaseClient(URL, "sb_secret_test"))


@respx.mock
async def test_where_and_who_ask_only_for_what_they_name(jobs: JobStore) -> None:
    route = respx.post(f"{REST}/brain_jobs").mock(return_value=httpx.Response(201))
    await jobs.enqueue(BrainJob(USER, KWAMI, "T", "x"))
    await jobs.enqueue(BrainJob(USER, KWAMI, "T", "x", target="cloud", agent_id="a1"))
    plain, chosen = (json.loads(c.request.content) for c in route.calls)
    # Before migration 029 the defaults must not be named at all.
    assert "target" not in plain and "agent_id" not in plain
    assert (chosen["target"], chosen["agent_id"]) == ("cloud", "a1")


def test_a_row_says_where_it_ran() -> None:
    row = {"id": "j", "target": "cloud", "agent_id": "a1", "runner_id": "cloud-x-1"}
    job = BrainJob.from_row(row)
    assert (job.target, job.agent_id, job.in_cloud) == ("cloud", "a1", True)
    assert BrainJob.from_row({"runner_id": "pc-1"}).in_cloud is False
    assert BrainJob.from_row({}).target == "auto"


@respx.mock
async def test_the_cloud_is_there_only_when_recently_seen(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_cloud_runners").mock(
        side_effect=[httpx.Response(200, json=[{"runner_id": "c"}]), httpx.Response(200, json=[])]
    )
    assert await jobs.cloud_online() is True
    assert await jobs.cloud_online() is False
    assert route.calls[0].request.url.params["last_seen_at"].startswith("gte.")


@respx.mock
async def test_the_users_consent_is_a_yes_or_nothing(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_cloud_settings").mock(
        side_effect=[
            httpx.Response(200, json=[{"enabled": True}]),
            httpx.Response(200, json=[{"enabled": False}]),
            httpx.Response(200, json=[]),
            httpx.Response(404, json={"message": "no table"}),
        ]
    )
    assert [await jobs.cloud_allowed(USER) for _ in range(4)] == [True, False, False, False]
    assert route.calls[0].request.url.params["user_id"] == f"eq.{USER}"


@respx.mock
async def test_an_agent_is_read_only_if_it_is_theirs_and_on(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_agents").mock(
        side_effect=[
            httpx.Response(200, json=[{"id": "a1", "name": "facturas"}]),
            httpx.Response(200, json=[]),
            httpx.Response(200, json=[{"name": "a"}, {"name": "b"}]),
            httpx.Response(404),
        ]
    )
    assert (await jobs.agent(USER, "a1")) == {"id": "a1", "name": "facturas"}
    assert await jobs.agent(USER, "a2") is None
    assert [a["name"] for a in await jobs.agents(USER)] == ["a", "b"]
    assert await jobs.agents(USER) == []
    params = route.calls[0].request.url.params
    assert params["user_id"] == f"eq.{USER}" and params["enabled"] == "is.true"


@respx.mock
async def test_the_cloud_beats_and_claims_through_its_function(jobs: JobStore) -> None:
    beat = respx.post(f"{REST}/brain_cloud_runners").mock(return_value=httpx.Response(201))
    claim = respx.post(f"{REST}/rpc/brain_claim_cloud_job").mock(
        side_effect=[
            httpx.Response(200, json=[{"id": "j1", "runner_id": "cloud-a"}]),
            httpx.Response(200, json=[]),
        ]
    )
    assert await jobs.cloud_beat("cloud-a", "h" * 200) is True
    assert beat.calls[0].request.url.params["on_conflict"] == "runner_id"
    assert len(json.loads(beat.calls[0].request.content)["hostname"]) == 120

    job = await jobs.claim_cloud("cloud-a")
    assert job is not None and job.in_cloud
    assert json.loads(claim.calls[0].request.content) == {
        "p_runner_id": "cloud-a",
        "p_patience_seconds": CLOUD_PATIENCE_SECONDS,
        "p_per_user": CLOUD_PER_USER,
    }
    assert await jobs.claim_cloud("cloud-a") is None


@respx.mock
async def test_stale_cloud_jobs_are_found_by_their_runner(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_jobs").mock(
        return_value=httpx.Response(200, json=[{"id": "j1", "runner_id": "cloud-x"}])
    )
    assert [j.id for j in await jobs.stale_cloud_jobs()] == ["j1"]
    params = route.calls[0].request.url.params
    assert params["runner_id"] == "like.cloud-*" and params["status"] == "eq.running"


@respx.mock
async def test_what_a_thread_found_comes_oldest_first(jobs: JobStore) -> None:
    route = respx.get(f"{REST}/brain_jobs").mock(
        return_value=httpx.Response(
            200, json=[{"result": "segundo"}, {"result": ""}, {"result": "primero"}]
        )
    )
    assert await jobs.thread_results("t1") == ["primero", "segundo"]
    params = route.calls[0].request.url.params
    assert params["thread_id"] == "eq.t1" and params["status"] == "eq.done"


@respx.mock
async def test_what_the_cloud_leaves_for_the_pc(jobs: JobStore) -> None:
    route = respx.post(f"{REST}/brain_sync").mock(return_value=httpx.Response(201))
    job = BrainJob(USER, KWAMI, "T" * 300, "x", id="j1")
    assert await jobs.add_sync(job, "patch", "u/k/sync/j1", files=2, repo="agent", base_ref="abc")
    body = json.loads(route.calls[0].request.content)
    assert body["id"] and len(body["title"]) == 200
    assert {k: body[k] for k in ("user_id", "kwami_id", "job_id", "kind", "repo", "files")} == {
        "user_id": USER,
        "kwami_id": KWAMI,
        "job_id": "j1",
        "kind": "patch",
        "repo": "agent",
        "files": 2,
    }
    assert (body["base_ref"], body["storage_prefix"]) == ("abc", "u/k/sync/j1")


@respx.mock
async def test_the_pc_takes_a_sync_and_says_how_it_ended(jobs: JobStore) -> None:
    claim = respx.post(f"{REST}/rpc/brain_claim_sync").mock(
        side_effect=[
            httpx.Response(200, json=[{"id": "s1"}]),
            httpx.Response(200, json=[]),
            httpx.Response(404, json={"message": "no function"}),
        ]
    )
    end = respx.patch(f"{REST}/brain_sync").mock(return_value=httpx.Response(204))
    assert await jobs.claim_sync(USER, "pc-1") == {"id": "s1"}
    assert await jobs.claim_sync(USER, "pc-1") is None and jobs.sync_ready is True
    # Before migration 029: nothing, and the PC learns to look less often.
    assert await jobs.claim_sync(USER, "pc-1") is None and jobs.sync_ready is False
    assert json.loads(claim.calls[0].request.content) == {"p_user_id": USER, "p_runner_id": "pc-1"}

    assert await jobs.end_sync("s1", "applied", "x" * 3000) is True
    body = json.loads(end.calls[0].request.content)
    assert body["status"] == "applied" and len(body["result"]) == 2000 and body["applied_at"]
    assert end.calls[0].request.url.params["id"] == "eq.s1"
