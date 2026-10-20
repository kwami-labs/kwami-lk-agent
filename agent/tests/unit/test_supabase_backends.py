"""The Supabase backends the agent uses when it has no Kwami API.

On LiveKit Cloud the agent writes the user's Supabase directly: brain spend and
owed results, session usage, browser contexts, the telephony runtime config and
the executor's files. Each backend is exercised against PostgREST/Storage
mocked at the HTTP layer with respx, so what is pinned is the wire format the
real project will receive -- table, filters, headers and body.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from src.adapters.supabase import SupabaseClient, as_user_id, direct_supabase, kwami_owner
from src.brain.config import executor_in_cloud
from src.brain.executor import CloudWorkspace, upload_results
from src.brain.ledger import Ledger, PendingResult, SupabaseLedger, create_ledger
from src.browser.context_store import (
    InMemoryContextStore,
    KwamiApiContextStore,
    SupabaseContextStore,
    create_context_store,
)
from src.domain import UsageTracker
from src.runtime_bootstrap import build_bootstrap_payload, fetch_runtime_config
from src.settings import Settings, set_settings
from src.usage.reporter import SupabaseUsageReporter, UsageReporter, create_usage_reporter

URL = "https://proj.supabase.co"
KEY = "sb_secret_test"
REST = f"{URL}/rest/v1"
USER = "11111111-1111-4111-8111-111111111111"
KWAMI = "22222222-2222-4222-8222-222222222222"


def _direct() -> Settings:
    return Settings(supabase_url=URL, supabase_secret_key=KEY, kwami_api_url_set=False)


@pytest.fixture
def store() -> SupabaseClient:
    return SupabaseClient(URL, KEY)


@pytest.fixture
def direct_settings():
    settings = _direct()
    set_settings(settings)
    yield settings
    set_settings(None)


def _body(call: Any) -> Any:
    return json.loads(call.request.content)


# -- settings: when the agent writes Supabase itself --------------------------


def test_supabase_is_used_directly_only_without_an_api(env_setting) -> None:
    env_setting("SUPABASE_URL", URL + "/")
    env_setting("SUPABASE_SECRET_KEY", KEY)
    env_setting("KWAMI_API_KEY", "internal")
    settings = Settings.from_env()
    assert settings.supabase_url == URL
    # KWAMI_API_URL unset: the localhost default is not an API to talk to.
    assert settings.kwami_api_enabled is False
    assert settings.supabase_direct is True

    env_setting("KWAMI_API_URL", "http://localhost:8080")
    assert Settings.from_env().supabase_direct is False


def test_kwami_storage_overrides_the_api(env_setting) -> None:
    """LiveKit Cloud cannot drop one secret, and it still has an old KWAMI_API_URL."""
    env_setting("SUPABASE_URL", URL)
    env_setting("SUPABASE_SECRET_KEY", KEY)
    env_setting("KWAMI_API_KEY", "internal")
    env_setting("KWAMI_API_URL", "https://api.example.com")

    env_setting("KWAMI_STORAGE", "Supabase")
    settings = Settings.from_env()
    assert settings.supabase_direct is True
    assert isinstance(create_usage_reporter(settings), SupabaseUsageReporter)
    assert isinstance(create_context_store(settings), SupabaseContextStore)
    assert isinstance(create_ledger(settings), SupabaseLedger)

    env_setting("KWAMI_STORAGE", "api")
    env_setting("KWAMI_API_URL", None)
    assert Settings.from_env().supabase_direct is False

    # Forcing Supabase without its credentials cannot invent them.
    env_setting("KWAMI_STORAGE", "supabase")
    env_setting("SUPABASE_SECRET_KEY", None)
    assert Settings.from_env().supabase_direct is False


@respx.mock
async def test_forced_supabase_never_calls_the_api(env_setting) -> None:
    env_setting("SUPABASE_URL", URL)
    env_setting("SUPABASE_SECRET_KEY", KEY)
    env_setting("KWAMI_API_KEY", "internal")
    env_setting("KWAMI_API_URL", "https://api.example.com")
    env_setting("KWAMI_STORAGE", "supabase")
    api = respx.route(host="api.example.com").mock(return_value=httpx.Response(500))
    respx.get(f"{REST}/user_kwamis").mock(return_value=httpx.Response(200, json=[KWAMI_ROW]))

    payload = await fetch_runtime_config(KWAMI)

    assert payload is not None and payload["kwamiId"] == KWAMI
    assert not api.called


def test_the_secret_key_is_redacted_in_the_startup_log() -> None:
    assert _direct().describe()["supabase_secret_key"] == "set"  # noqa: S105


def test_no_direct_client_without_credentials() -> None:
    assert direct_supabase(Settings()) is None
    assert direct_supabase(_direct()) is not None


def test_as_user_id_accepts_only_uuids() -> None:
    assert as_user_id(USER.upper()) == USER
    assert as_user_id("sip_+34600000000") is None
    assert as_user_id(None) is None


# -- the client ----------------------------------------------------------------


@respx.mock
async def test_select_sends_the_key_in_both_headers(store: SupabaseClient) -> None:
    route = respx.get(f"{REST}/user_kwamis").mock(
        return_value=httpx.Response(200, json=[{"user_id": USER}])
    )
    rows = await store.select("user_kwamis", {"select": "user_id", "id": f"eq.{KWAMI}"})

    assert rows == [{"user_id": USER}]
    request = route.calls.last.request
    assert request.headers["apikey"] == KEY
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert request.url.params["id"] == f"eq.{KWAMI}"


@respx.mock
async def test_failures_are_logged_never_raised(store: SupabaseClient) -> None:
    respx.get(f"{REST}/brain_spend").mock(return_value=httpx.Response(500, text="boom"))
    respx.post(f"{REST}/agent_usage").mock(side_effect=httpx.ConnectError("down"))

    assert await store.select("brain_spend", {}) is None
    assert await store.insert("agent_usage", {"model": "x"}) is False


@respx.mock
async def test_upsert_asks_postgrest_to_merge(store: SupabaseClient) -> None:
    route = respx.post(f"{REST}/browser_contexts").mock(return_value=httpx.Response(201))
    assert await store.insert("browser_contexts", {"a": 1}, on_conflict="owner_key,vendor")

    request = route.calls.last.request
    assert request.url.params["on_conflict"] == "owner_key,vendor"
    assert "resolution=merge-duplicates" in request.headers["prefer"]


async def test_an_unfiltered_update_is_refused(store: SupabaseClient) -> None:
    with pytest.raises(ValueError):
        await store.update("brain_pending", {"taken_at": "now"}, {})


@respx.mock
async def test_upload_quotes_each_segment_and_overwrites(store: SupabaseClient) -> None:
    route = respx.post(url__startswith=f"{URL}/storage/v1/object/jarvis-files/").mock(
        return_value=httpx.Response(200, json={"Key": "x"})
    )
    assert await store.upload("jarvis-files", f"{USER}/k/informe final.md", b"hola")

    request = route.calls.last.request
    assert request.url.raw_path.decode().endswith(f"/{USER}/k/informe%20final.md")
    assert request.headers["x-upsert"] == "true"
    assert request.content == b"hola"


@respx.mock
async def test_kwami_owner_reads_user_kwamis(store: SupabaseClient) -> None:
    respx.get(f"{REST}/user_kwamis").mock(
        return_value=httpx.Response(200, json=[{"user_id": USER}])
    )
    assert await kwami_owner(store, KWAMI) == USER
    # Not a user_kwamis id (a SIP identity): no request at all.
    assert await kwami_owner(store, "sip_caller") is None


@respx.mock
async def test_a_body_that_is_not_rows_reads_as_nothing(store: SupabaseClient) -> None:
    respx.get(f"{REST}/brain_spend").mock(return_value=httpx.Response(200, text="<html>"))
    respx.get(f"{REST}/brain_pending").mock(return_value=httpx.Response(200, json={"a": 1}))
    assert await store.select("brain_spend", {}) is None
    assert await store.select("brain_pending", {}) is None


@respx.mock
async def test_rpc_returns_its_result_true_or_none(store: SupabaseClient) -> None:
    route = respx.post(f"{REST}/rpc/brain_add_spend")
    route.mock(return_value=httpx.Response(200, json=3))
    assert await store.rpc("brain_add_spend", {}) == 3
    # A void function answers 204 with no body: it still ran.
    route.mock(return_value=httpx.Response(204))
    assert await store.rpc("brain_add_spend", {}) is True
    route.mock(return_value=httpx.Response(400, text="bad args"))
    assert await store.rpc("brain_add_spend", {}) is None


def test_one_client_per_project() -> None:
    from src.adapters.supabase import get_supabase

    assert get_supabase(Settings()) is None
    client = get_supabase(_direct())
    assert client is get_supabase(_direct())
    assert client is not None and client.url == URL


# -- brain ledger ----------------------------------------------------------------


def test_create_ledger_picks_the_backend() -> None:
    assert type(create_ledger(Settings())) is Ledger
    assert isinstance(create_ledger(_direct()), SupabaseLedger)


def _mock_ledger_reads(spend: list[dict[str, Any]], pending: list[dict[str, Any]]) -> None:
    respx.get(f"{REST}/user_kwamis").mock(
        return_value=httpx.Response(200, json=[{"user_id": USER}])
    )
    respx.get(f"{REST}/brain_spend").mock(return_value=httpx.Response(200, json=spend))
    respx.get(f"{REST}/brain_pending").mock(return_value=httpx.Response(200, json=pending))


@respx.mock
async def test_bind_loads_the_owners_spend_and_owed_results(store: SupabaseClient) -> None:
    _mock_ledger_reads(
        spend=[
            {"model_id": "claude-opus", "usd": 1.5},
            {"model_id": "gemini-flash", "usd": "0.25"},
        ],
        pending=[
            {
                "id": "row-1",
                "task_id": "t1",
                "title": "Oro",
                "text": "Sube un 2 %",
                "models": ["Claude Opus 5.5"],
                "created_at": "2026-10-18T10:00:00Z",
            }
        ],
    )
    ledger = SupabaseLedger(store)
    await ledger.bind(KWAMI)

    assert ledger.user_id == USER
    spend = ledger.month_spend()
    assert spend["usd"] == pytest.approx(1.75)
    assert spend["byModel"]["claude-opus"] == pytest.approx(1.5)

    patch = respx.patch(f"{REST}/brain_pending").mock(return_value=httpx.Response(204))
    owed = ledger.take_pending(KWAMI)
    await ledger.flush()

    assert [(r.title, r.summary, r.models) for r in owed] == [
        ("Oro", "Sube un 2 %", ["Claude Opus 5.5"])
    ]
    assert patch.called
    params = patch.calls.last.request.url.params
    assert params["id"] == "in.(row-1)"
    assert params["taken_at"] == "is.null"
    assert "taken_at" in _body(patch.calls.last)
    assert ledger.take_pending(KWAMI) == []


@respx.mock
async def test_spend_is_added_through_the_atomic_function(store: SupabaseClient) -> None:
    _mock_ledger_reads(spend=[], pending=[])
    rpc = respx.post(f"{REST}/rpc/brain_add_spend").mock(return_value=httpx.Response(200, json=1))
    ledger = SupabaseLedger(store)
    await ledger.bind(KWAMI)

    ledger.add_spend("claude-opus", 0.5)
    ledger.add_spend("claude-opus", 0.25)
    ledger.add_spend("claude-opus", 0.0)  # nothing to record
    await ledger.flush()

    assert ledger.month_spend()["byModel"] == {"claude-opus": 0.75}
    assert rpc.call_count == 2
    body = _body(rpc.calls.last)
    assert body["p_user_id"] == USER
    assert body["p_kwami_id"] == KWAMI
    assert body["p_model_id"] == "claude-opus"
    assert body["p_usd"] == pytest.approx(0.25)


@respx.mock
async def test_spend_before_the_owner_is_known_is_held_then_sent(store: SupabaseClient) -> None:
    rpc = respx.post(f"{REST}/rpc/brain_add_spend").mock(return_value=httpx.Response(200, json=1))
    respx.get(f"{REST}/brain_spend").mock(return_value=httpx.Response(200, json=[]))
    ledger = SupabaseLedger(store)

    ledger.add_spend("claude-haiku", 0.01)
    assert not rpc.called

    await ledger.bind("", user_id=USER)
    await ledger.flush()
    assert rpc.call_count == 1
    assert _body(rpc.calls.last)["p_user_id"] == USER


@respx.mock
async def test_owed_results_are_inserted_with_their_own_id(store: SupabaseClient) -> None:
    _mock_ledger_reads(spend=[], pending=[])
    insert = respx.post(f"{REST}/brain_pending").mock(return_value=httpx.Response(201))
    ledger = SupabaseLedger(store)
    await ledger.bind(KWAMI)

    ledger.add_pending(
        KWAMI, PendingResult("t9", "Vuelos", "El más barato sale el martes", "", ["Gemini"])
    )
    await ledger.flush()

    row = _body(insert.calls.last)
    assert row["kwami_id"] == KWAMI
    assert row["user_id"] == USER
    assert row["text"] == "El más barato sale el martes"
    assert row["models"] == ["Gemini"]
    assert row["id"]
    # Same process, next greeting: it is read out without another round trip.
    respx.patch(f"{REST}/brain_pending").mock(return_value=httpx.Response(204))
    assert [r.task_id for r in ledger.take_pending(KWAMI)] == ["t9"]


@respx.mock
async def test_an_unreachable_supabase_leaves_an_empty_ledger(store: SupabaseClient) -> None:
    respx.route(url__startswith=URL).mock(side_effect=httpx.ConnectError("offline"))
    ledger = SupabaseLedger(store)
    await ledger.bind(KWAMI)

    assert ledger.month_spend()["usd"] == 0.0
    assert ledger.take_pending(KWAMI) == []


async def test_the_disk_ledger_keeps_its_interface(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    await ledger.bind(KWAMI)
    ledger.add_pending(KWAMI, PendingResult("t1", "a", "b", "", ["m"]))
    await ledger.flush()
    assert [r.models for r in ledger.take_pending(KWAMI)] == [["m"]]


# -- session usage -----------------------------------------------------------------


def _tracker_with_usage() -> UsageTracker:
    from livekit.agents.metrics.usage import (
        AgentSessionUsage,
        LLMModelUsage,
        STTModelUsage,
        TTSModelUsage,
    )

    tracker = UsageTracker()
    tracker.on_session_usage(
        AgentSessionUsage(
            model_usage=[
                LLMModelUsage(
                    provider="openai", model="gpt-5.2", input_tokens=100, output_tokens=20
                ),
                STTModelUsage(provider="deepgram", model="nova-3", audio_duration=90.0),
                TTSModelUsage(provider="cartesia", model="sonic", characters_count=400),
            ]
        )
    )
    tracker.record_external_usage("tool", "tavily/search")
    return tracker


def test_create_usage_reporter_picks_the_backend() -> None:
    assert isinstance(create_usage_reporter(Settings()), UsageReporter)
    assert isinstance(create_usage_reporter(_direct()), SupabaseUsageReporter)


@respx.mock
async def test_usage_is_recorded_one_row_per_model(store: SupabaseClient) -> None:
    respx.get(f"{REST}/user_kwamis").mock(
        return_value=httpx.Response(200, json=[{"user_id": USER}])
    )
    insert = respx.post(f"{REST}/agent_usage").mock(return_value=httpx.Response(201))

    reporter = SupabaseUsageReporter(store)
    # The identity handed over is the kwami id itself; the owner comes from its row.
    assert await reporter.report(KWAMI, "room-1", _tracker_with_usage(), kwami_id=KWAMI)

    rows = {r["kind"]: r for r in _body(insert.calls.last)}
    assert set(rows) == {"llm", "stt", "tts", "tool"}
    assert all(r["user_id"] == USER and r["session_id"] == "room-1" for r in rows.values())
    assert (rows["llm"]["provider"], rows["llm"]["model"], rows["llm"]["units"]) == (
        "openai",
        "gpt-5.2",
        120.0,
    )
    assert rows["llm"]["details"]["prompt_tokens"] == 100
    assert rows["stt"]["units"] == pytest.approx(1.5)
    assert rows["tts"]["units"] == 400.0


@respx.mock
async def test_a_failed_usage_write_reports_false(store: SupabaseClient) -> None:
    respx.post(f"{REST}/agent_usage").mock(return_value=httpx.Response(409, text="fk"))
    reporter = SupabaseUsageReporter(store)
    assert await reporter.report(USER, "room-1", _tracker_with_usage()) is False


async def test_no_usage_needs_no_write(store: SupabaseClient) -> None:
    assert await SupabaseUsageReporter(store).report(USER, "room-1", UsageTracker()) is True


# -- telephony runtime config -----------------------------------------------------


KWAMI_ROW = {
    "id": KWAMI,
    "user_id": USER,
    "name": "Jarvis",
    "config": {
        "voice": {
            "stt": {"provider": "deepgram", "model": "nova-3", "language": "es"},
            "tts": {"provider": "google", "voice": "Charon"},
            "soulConfig": {"name": "Jarvis", "language": "es-ES", "charter": {"enabled": True}},
        }
    },
}


def test_the_bootstrap_payload_matches_the_apis() -> None:
    payload = build_bootstrap_payload(KWAMI_ROW)
    assert payload["type"] == "config"
    assert payload["kwamiId"] == KWAMI
    assert payload["voice"]["stt"]["language"] == "es"
    assert payload["voice"]["tts"]["voice"] == "Charon"
    assert payload["voice"]["llm"]["provider"] == "openai"  # the API's default
    assert payload["soul"]["language"] == "es-ES"
    assert payload["soul"]["charter"] == {"enabled": True}


@respx.mock
async def test_runtime_config_comes_from_user_kwamis(direct_settings) -> None:
    route = respx.get(f"{REST}/user_kwamis").mock(
        return_value=httpx.Response(200, json=[KWAMI_ROW])
    )
    payload = await fetch_runtime_config(KWAMI)

    assert payload is not None and payload["kwamiName"] == "Jarvis"
    assert route.calls.last.request.url.params["id"] == f"eq.{KWAMI}"


@respx.mock
async def test_an_unknown_kwami_has_no_runtime_config(direct_settings) -> None:
    respx.get(f"{REST}/user_kwamis").mock(return_value=httpx.Response(200, json=[]))
    assert await fetch_runtime_config(KWAMI) is None
    assert await fetch_runtime_config("sip_caller") is None


@respx.mock
async def test_an_unreachable_supabase_is_not_reported_as_a_missing_kwami(
    direct_settings, caplog
) -> None:
    respx.get(f"{REST}/user_kwamis").mock(return_value=httpx.Response(503))
    with caplog.at_level("WARNING"):
        assert await fetch_runtime_config(KWAMI) is None
    assert "not found in Supabase" not in caplog.text


# -- browser contexts -------------------------------------------------------------


def test_create_context_store_picks_the_backend() -> None:
    assert isinstance(create_context_store(Settings()), InMemoryContextStore)
    assert isinstance(create_context_store(_direct()), SupabaseContextStore)
    api = Settings(kwami_api_key="k", supabase_url=URL, supabase_secret_key=KEY)
    assert isinstance(create_context_store(api), KwamiApiContextStore)


@respx.mock
async def test_browser_context_round_trip(store: SupabaseClient) -> None:
    get = respx.get(f"{REST}/browser_contexts").mock(
        return_value=httpx.Response(200, json=[{"context_id": "ctx_1"}])
    )
    put = respx.post(f"{REST}/browser_contexts").mock(return_value=httpx.Response(201))
    contexts = SupabaseContextStore(store)

    assert await contexts.get(KWAMI, "Browserbase") == "ctx_1"
    assert get.calls.last.request.url.params["vendor"] == "eq.browserbase"

    await contexts.put(KWAMI, "browserbase", "ctx_2")
    row = _body(put.calls.last)
    assert row["owner_key"] == KWAMI
    assert row["context_id"] == "ctx_2"
    assert row["user_id"] == KWAMI  # a uuid owner is recorded for RLS
    assert put.calls.last.request.url.params["on_conflict"] == "owner_key,vendor"


@respx.mock
async def test_an_unknown_vendor_is_not_stored(store: SupabaseClient) -> None:
    contexts = SupabaseContextStore(store)
    assert await contexts.get(KWAMI, "chrome") is None
    await contexts.put(KWAMI, "chrome", "x")
    assert not respx.calls


@respx.mock
async def test_a_phone_caller_context_is_stored_without_a_user_id(
    store: SupabaseClient, caplog
) -> None:
    respx.get(f"{REST}/browser_contexts").mock(return_value=httpx.Response(200, json=[]))
    put = respx.post(f"{REST}/browser_contexts").mock(return_value=httpx.Response(500))
    contexts = SupabaseContextStore(store)

    assert await contexts.get("sip_caller", "browserbase") is None
    with caplog.at_level("WARNING"):
        await contexts.put("sip_caller", "browserbase", "ctx_3")
    assert "user_id" not in _body(put.calls.last)
    assert "will not survive this session" in caplog.text


# -- the cloud executor's files ----------------------------------------------------


def test_the_executor_mode_can_be_forced() -> None:
    assert executor_in_cloud("cloud") is True
    assert executor_in_cloud("local") is False


@respx.mock
async def test_results_are_uploaded_under_user_and_kwami(tmp_path: Path, store) -> None:
    route = respx.post(url__startswith=f"{URL}/storage/v1/object/jarvis-files/").mock(
        return_value=httpx.Response(200, json={})
    )
    (tmp_path / "informe.md").write_text("hola", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "big.js").write_text("x", encoding="utf-8")

    prefix, stored = await upload_results(
        tmp_path, CloudWorkspace(store=store, user_id=USER, kwami_id=KWAMI), "task-1"
    )

    assert (prefix, stored) == (f"{USER}/{KWAMI}/task-1", 2)
    paths = sorted(c.request.url.path for c in route.calls)
    assert paths == [
        f"/storage/v1/object/jarvis-files/{USER}/{KWAMI}/task-1/informe.md",
        f"/storage/v1/object/jarvis-files/{USER}/{KWAMI}/task-1/src/main.py",
    ]


async def test_nothing_is_uploaded_without_an_owner(tmp_path: Path, store) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    assert await upload_results(tmp_path, CloudWorkspace(store=store), "t") == ("", 0)
    assert await upload_results(tmp_path, CloudWorkspace(store=None, user_id=USER), "t") == ("", 0)


# -- session_usage_updated snapshots -------------------------------------------------


def test_snapshots_replace_rather_than_add() -> None:
    from livekit.agents.metrics.usage import AgentSessionUsage, LLMModelUsage

    tracker = UsageTracker()
    for total in (50, 120):
        tracker.on_session_usage(
            AgentSessionUsage(
                model_usage=[
                    LLMModelUsage(
                        provider="openai", model="gpt-5.2", input_tokens=total, output_tokens=0
                    )
                ]
            )
        )
    [item] = tracker.get_usage_summary()
    assert item["units_used"] == 120


def test_realtime_models_are_metered_in_minutes() -> None:
    from livekit.agents.metrics.usage import AgentSessionUsage, LLMModelUsage

    tracker = UsageTracker()
    tracker.on_session_usage(
        AgentSessionUsage(
            model_usage=[
                LLMModelUsage(
                    provider="openai",
                    model="gpt-realtime",
                    input_tokens=1810,
                    output_tokens=1210,
                    input_audio_tokens=1800,
                    output_audio_tokens=1200,
                    input_text_tokens=10,
                    output_text_tokens=10,
                )
            ]
        )
    )
    [item] = tracker.get_usage_summary()
    assert item["model_type"] == "realtime"
    assert item["audio_input_minutes"] == pytest.approx(3.0)
    assert item["audio_output_minutes"] == pytest.approx(1.0)
    assert item["units_used"] == pytest.approx(4.0)
    assert item["text_input_tokens"] == 10


def _jarvis_on_live():
    return {
        "id": "k1",
        "name": "Jarvis",
        "config": {
            "voice": {
                "pipelineMode": "realtime",
                "stt": {
                    "provider": "gemini",
                    "model": "gemini-3.5-transcribe-live",
                    "language": "es",
                },
                "tts": {"provider": "gemini", "voice": "Charon", "style": "Habla como JARVIS"},
                "realtime": {
                    "provider": "google",
                    "model": "gemini-3.1-flash-live-preview",
                    "voice": "Charon",
                    "style": "Habla como JARVIS",
                    "modalities": ["text", "audio"],
                },
            }
        },
    }


def test_a_call_keeps_the_engine_the_voice_and_its_direction():
    """A kwami on Gemini Live answered calls on the standard pipeline, without its direction."""
    voice = build_bootstrap_payload(_jarvis_on_live())["voice"]
    assert voice["type"] == "realtime"
    assert voice["realtime"] == {
        "provider": "google",
        "model": "gemini-3.1-flash-live-preview",
        "voice": "Charon",
        "style": "Habla como JARVIS",
    }
    assert voice["tts"]["style"] == "Habla como JARVIS"


def test_a_call_moves_the_old_default_ear_to_the_fast_one():
    stt = build_bootstrap_payload(_jarvis_on_live())["voice"]["stt"]
    assert stt == {"provider": "deepgram", "model": "nova-3", "language": "es"}


def test_without_an_engine_or_direction_nothing_is_invented():
    voice = build_bootstrap_payload(
        {"id": "k1", "name": "K", "config": {"voice": {"realtime": {}}}}
    )["voice"]
    assert "type" not in voice
    assert "realtime" not in voice
    assert "style" not in voice["tts"]
