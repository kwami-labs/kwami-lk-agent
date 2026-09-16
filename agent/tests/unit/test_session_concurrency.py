"""Background work must not be dropped, and config must not race itself."""

from __future__ import annotations

import asyncio

from src.session import SessionState


async def test_spawn_keeps_a_strong_reference_while_running() -> None:
    """The loop only holds a weak reference to a running task.

    A bare `asyncio.create_task(...)` whose handle is dropped can be collected
    mid-flight, which is how a config update could simply vanish.
    """
    state = SessionState()
    started = asyncio.Event()
    release = asyncio.Event()

    async def work() -> None:
        started.set()
        await release.wait()

    task = state.spawn(work(), name="unit-work")
    await started.wait()

    assert task in state._background_tasks

    release.set()
    await task
    assert task not in state._background_tasks, "finished task was not released"


async def test_spawn_logs_a_failure_instead_of_swallowing_it(caplog) -> None:
    """A dropped handle also means the exception is never retrieved."""
    state = SessionState()

    async def boom() -> None:
        raise RuntimeError("kaboom")

    with caplog.at_level("ERROR"):
        task = state.spawn(boom(), name="exploding-task")
        await asyncio.gather(task, return_exceptions=True)

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "exploding-task" in messages
    assert "kaboom" in messages


async def test_a_cancelled_task_is_not_reported_as_an_error(caplog) -> None:
    state = SessionState()

    async def forever() -> None:
        await asyncio.Event().wait()

    with caplog.at_level("ERROR"):
        task = state.spawn(forever(), name="cancelled-task")
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert not [r for r in caplog.records if "cancelled-task" in r.getMessage()]


async def test_config_handling_is_serialized() -> None:
    """Two config messages arriving together used to run concurrently.

    Both built a Zep client and both called `update_agent`, so
    `state.current_agent` and `session._agent` could end up disagreeing and one
    client was never closed.
    """
    state = SessionState()
    events: list[str] = []

    async def handler(tag: str) -> None:
        events.append(f"{tag}:start")
        await asyncio.sleep(0)  # yield; an unserialized handler interleaves here
        await asyncio.sleep(0)
        events.append(f"{tag}:end")

    await asyncio.gather(
        state.run_serialized(handler("a")),
        state.run_serialized(handler("b")),
    )

    # Each handler must run to completion before the next one starts.
    assert events in (
        ["a:start", "a:end", "b:start", "b:end"],
        ["b:start", "b:end", "a:start", "a:end"],
    ), f"config handlers interleaved: {events}"


async def test_serialization_survives_a_failing_handler() -> None:
    """A handler that raises must still release the lock."""
    state = SessionState()

    async def boom() -> None:
        raise RuntimeError("nope")

    await asyncio.gather(state.run_serialized(boom()), return_exceptions=True)

    assert not state.config_lock.locked(), "lock was left held after a failure"
