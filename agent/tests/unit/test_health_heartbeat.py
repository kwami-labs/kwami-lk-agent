"""The heartbeat that lets the container's health probe report a real failure.

The probe answered `{"status": "ok"}` whenever its own HTTP thread was alive, so
the one failure that matters -- an agent process that is running but not
registered with LiveKit -- was indistinguishable from a healthy one. `start.sh`
catches a hard crash; nothing caught a dropped websocket, an auth loop or a
wedged event loop.

Both halves are tested here, including the probe, which lives in
`infra/container/` and is loaded by path: it is stdlib-only by design (it runs
as a separate process beside the agent) so it cannot be imported normally, but
that is no reason to leave it untested -- it is the thing deciding whether
Cloudflare restarts the container.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

from src.health import (
    DEFAULT_HEARTBEAT_PATH,
    HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_PATH_ENV,
    heartbeat_path,
    run_heartbeat,
    write_heartbeat,
)

PROBE_PATH = Path(__file__).parent.parent.parent.parent / "infra" / "container" / "health.py"


@pytest.fixture
def probe(monkeypatch, tmp_path):
    """The container probe, loaded from its real path."""
    spec = importlib.util.spec_from_file_location("kwami_health_probe", PROBE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv(HEARTBEAT_PATH_ENV, str(tmp_path / "heartbeat.json"))
    return module


@pytest.fixture
def beat_file(monkeypatch, tmp_path) -> Path:
    path = tmp_path / "heartbeat.json"
    monkeypatch.setenv(HEARTBEAT_PATH_ENV, str(path))
    return path


# -- The writer ---------------------------------------------------------------


def test_the_path_honours_the_environment_override(beat_file) -> None:
    assert heartbeat_path() == beat_file


def test_the_path_falls_back_to_the_default(monkeypatch) -> None:
    monkeypatch.delenv(HEARTBEAT_PATH_ENV, raising=False)

    assert heartbeat_path() == DEFAULT_HEARTBEAT_PATH


def test_a_blank_override_falls_back_to_the_default(monkeypatch) -> None:
    monkeypatch.setenv(HEARTBEAT_PATH_ENV, "   ")

    assert heartbeat_path() == DEFAULT_HEARTBEAT_PATH


def test_a_beat_records_the_state_and_a_timestamp(beat_file) -> None:
    assert write_heartbeat("registered", now=1000.0) is True

    payload = json.loads(beat_file.read_text())
    assert payload == {"state": "registered", "ts": 1000.0}


def test_the_parent_directory_is_created(monkeypatch, tmp_path) -> None:
    nested = tmp_path / "a" / "b" / "heartbeat.json"
    monkeypatch.setenv(HEARTBEAT_PATH_ENV, str(nested))

    assert write_heartbeat("registered") is True
    assert nested.exists()


def test_an_unwritable_path_is_reported_not_raised(monkeypatch, tmp_path, caplog) -> None:
    """A read-only filesystem must not take down a worker that is serving calls.
    The probe will report staleness, which is the right outcome -- but it should
    not be *caused* here."""
    import logging

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv(HEARTBEAT_PATH_ENV, str(blocker / "heartbeat.json"))

    with caplog.at_level(logging.WARNING):
        assert write_heartbeat("registered") is False

    assert "Could not write the health heartbeat" in caplog.text


def test_the_beat_is_written_atomically(beat_file) -> None:
    """A probe reading concurrently must see a whole beat or the previous one,
    never a half-written file. Written to .tmp then replaced, so no partial file
    is left behind."""
    write_heartbeat("registered", now=1.0)
    write_heartbeat("registered", now=2.0)

    assert json.loads(beat_file.read_text())["ts"] == 2.0
    assert not beat_file.with_suffix(".tmp").exists()


async def test_the_loop_beats_repeatedly(beat_file) -> None:
    await run_heartbeat(interval=0, iterations=3)

    assert json.loads(beat_file.read_text())["state"] == "registered"


async def test_the_loop_records_the_state_it_was_given(beat_file) -> None:
    await run_heartbeat(state="starting", interval=0, iterations=1)

    assert json.loads(beat_file.read_text())["state"] == "starting"


# -- The probe ----------------------------------------------------------------


def test_the_two_sides_agree_on_where_the_heartbeat_lives(probe) -> None:
    """Two constants across a process boundary is exactly the pair that drifts.
    The agent writes here; a separate stdlib-only process reads it."""
    assert probe.HEARTBEAT_PATH_ENV == HEARTBEAT_PATH_ENV
    assert probe.DEFAULT_HEARTBEAT_PATH == DEFAULT_HEARTBEAT_PATH


def test_the_beat_interval_is_well_inside_the_staleness_window(probe) -> None:
    """Otherwise one slow write reads as a dead worker and the container is
    restarted underneath live calls."""
    assert probe.STALE_AFTER_SECONDS >= 3 * HEARTBEAT_INTERVAL_SECONDS


def test_no_heartbeat_during_startup_is_not_a_failure(probe) -> None:
    """`startAndWaitForPorts` waits on this port; failing here would stop the
    container ever coming up."""
    status, body = probe.assess(uptime=5.0)

    assert status == 200
    assert body["status"] == "starting"


def test_no_heartbeat_after_the_grace_period_is_a_failure(probe) -> None:
    """This is the case the old probe could never report: process up, worker
    never registered."""
    status, body = probe.assess(uptime=probe.STARTUP_GRACE_SECONDS + 1)

    assert status == 503
    assert body["status"] == "degraded"
    assert "never registered" in body["detail"]


def test_a_fresh_heartbeat_is_healthy(probe) -> None:
    write_heartbeat("registered", now=1000.0)

    status, body = probe.assess(now=1005.0, uptime=1000.0)

    assert status == 200
    assert body["status"] == "ok"
    assert body["worker"] == "registered"
    assert body["heartbeat_age_seconds"] == 5.0


def test_a_stale_heartbeat_is_a_failure(probe) -> None:
    """The worker registered and then stopped beating: alive, not working."""
    write_heartbeat("registered", now=1000.0)

    status, body = probe.assess(now=1000.0 + probe.STALE_AFTER_SECONDS + 1, uptime=2000.0)

    assert status == 503
    assert body["status"] == "degraded"
    assert "stale" in body["detail"]


def test_a_corrupt_heartbeat_reads_as_absent(probe, beat_file) -> None:
    beat_file.parent.mkdir(parents=True, exist_ok=True)
    beat_file.write_text("{not json")

    status, body = probe.assess(uptime=probe.STARTUP_GRACE_SECONDS + 1)

    assert status == 503
    assert body["status"] == "degraded"


def test_a_heartbeat_that_is_not_an_object_reads_as_absent(probe, beat_file) -> None:
    beat_file.parent.mkdir(parents=True, exist_ok=True)
    beat_file.write_text(json.dumps([1, 2, 3]))

    assert probe.read_heartbeat(beat_file) is None


def test_a_heartbeat_with_no_timestamp_is_treated_as_ancient(probe, beat_file) -> None:
    """Defensive: a beat written by an older agent build without `ts`."""
    beat_file.parent.mkdir(parents=True, exist_ok=True)
    beat_file.write_text(json.dumps({"state": "registered"}))

    status, _ = probe.assess(now=time.time(), uptime=2000.0)

    assert status == 503


def test_a_clock_skewed_heartbeat_does_not_report_negative_age(probe) -> None:
    """A beat timestamped slightly in the future must not surface as a negative
    age in the payload."""
    write_heartbeat("registered", now=1010.0)

    _, body = probe.assess(now=1000.0, uptime=2000.0)

    assert body["heartbeat_age_seconds"] == 0.0


# -- Wiring: worker_registered starts the beat --------------------------------


def test_zero_iterations_writes_nothing(beat_file) -> None:
    """Covers the loop's own exit condition rather than the early return."""
    import asyncio

    asyncio.run(run_heartbeat(interval=0, iterations=0))

    assert not beat_file.exists()


async def test_worker_registration_writes_a_beat_and_starts_the_loop(beat_file) -> None:
    """The container probe reports `degraded` until this fires, which is the
    whole point: "process is up" and "worker is registered" are different facts."""
    import asyncio

    import src.main as main

    main._heartbeat_task = None
    try:
        main._on_worker_registered()

        assert json.loads(beat_file.read_text())["state"] == "registered"
        assert main._heartbeat_task is not None
        assert not main._heartbeat_task.done()
    finally:
        if main._heartbeat_task is not None:
            main._heartbeat_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await main._heartbeat_task
            main._heartbeat_task = None


async def test_a_reconnect_does_not_stack_a_second_heartbeat(beat_file) -> None:
    """`worker_registered` fires again after every reconnect. One task per
    reconnection would mean N writers racing on one file."""
    import asyncio

    import src.main as main

    main._heartbeat_task = None
    try:
        main._on_worker_registered()
        first = main._heartbeat_task
        main._on_worker_registered()

        assert main._heartbeat_task is first
    finally:
        if main._heartbeat_task is not None:
            main._heartbeat_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await main._heartbeat_task
            main._heartbeat_task = None


async def test_a_finished_heartbeat_task_is_replaced(beat_file) -> None:
    """If the loop ever exits, the next registration must start a new one
    rather than leaving the worker permanently silent."""
    import asyncio

    import src.main as main

    done: asyncio.Task = asyncio.create_task(asyncio.sleep(0))
    await done
    main._heartbeat_task = done
    try:
        main._on_worker_registered()

        assert main._heartbeat_task is not done
    finally:
        if main._heartbeat_task is not None and main._heartbeat_task is not done:
            main._heartbeat_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await main._heartbeat_task
        main._heartbeat_task = None


def test_registration_outside_an_event_loop_is_survivable(beat_file, caplog) -> None:
    """A worker that cannot start its heartbeat should say so and keep serving
    calls, not raise inside an event handler where the traceback goes nowhere."""
    import logging

    import src.main as main

    main._heartbeat_task = None
    with caplog.at_level(logging.WARNING):
        main._on_worker_registered()

    assert main._heartbeat_task is None
    assert "no heartbeat started" in caplog.text
    assert json.loads(beat_file.read_text())["state"] == "registered"
