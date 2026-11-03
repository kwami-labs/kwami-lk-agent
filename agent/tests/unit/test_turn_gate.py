"""`runtime/turn_gate.py`: a turn that stops mid-sentence waits for the rest.

The cases come from real turns of 2026-11-02, which were answered in pieces.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.agents.llm import StopResponse

from src.agent import KwamiAgent
from src.brain.bridge import CONTEXT_MEMORY_CHARS, CONTEXT_TURNS
from src.domain import KwamiConfig
from src.domain.pipelines import REALTIME_PIPELINE
from src.factories.stt import DEFAULT_KEYTERMS, _deepgram_keyterms
from src.handlers.listening import parse_listening
from src.runtime.turn_gate import OPEN_TURN_SECONDS, TurnGate, sounds_unfinished


class TestWhatSoundsUnfinished:
    @pytest.mark.parametrize(
        "said",
        [
            "puedes entrar en el local, digamos, de la cuenta de Cloud,",
            "Quiero que mires el correo y",
            "Dile que",
            "Sácame algo sobre, o sea",
            "Lo que pasa es que.",
            "Vamos a ver si...",
            "I want to look at the",
            "pues…",
        ],
    )
    def test_stops_where_a_sentence_cannot(self, said):
        assert sounds_unfinished(said)

    @pytest.mark.parametrize(
        "said",
        [
            "¿Sabes quién es Alex en mi correo electrónico?",
            "Vale.",
            "Sí.",
            "Quiero que hagas un poquito de investigación.",
            "¿De qué?",
            "No lo sé!",
            "",
            "   ",
            "123",
        ],
    )
    def test_a_finished_thought_is_answered(self, said):
        assert not sounds_unfinished(said)


def _gate(answers: list[str], hold: float = 0.01, now=None) -> TurnGate:
    return TurnGate(answers.append, hold_seconds=hold, now=now)


class TestThePieces:
    async def test_a_finished_turn_goes_straight_through(self):
        gate = _gate([])
        assert gate.take("¿Qué hora es?") == "¿Qué hora es?"
        assert not gate.holding

    async def test_the_pieces_are_answered_as_one(self):
        answers: list[str] = []
        gate = _gate(answers, hold=10)
        assert gate.take("Quiero que mires el correo y") is None
        assert gate.holding
        assert gate.take("me digas quién es Alex.") == (
            "Quiero que mires el correo y me digas quién es Alex."
        )
        assert not gate.holding and answers == []

    async def test_a_second_open_piece_keeps_waiting(self):
        gate = _gate([], hold=10)
        assert gate.take("Quiero que") is None
        assert gate.take("mires el correo y") is None
        assert gate.take("ya.") == "Quiero que mires el correo y ya."

    async def test_silence_answers_what_was_said(self):
        answers: list[str] = []
        gate = _gate(answers)
        assert gate.take("Mira el correo y") is None
        await asyncio.sleep(0.05)
        assert answers == ["Mira el correo y"] and not gate.holding

    async def test_the_wait_does_not_run_while_the_user_speaks(self):
        answers: list[str] = []
        gate = _gate(answers, hold=0.02)
        gate.take("Mira el correo y")
        gate.speaking(True)
        await asyncio.sleep(0.06)
        assert answers == [] and gate.holding
        gate.speaking(False)
        await asyncio.sleep(0.06)
        assert answers == ["Mira el correo y"]

    async def test_stopping_with_nothing_held_starts_nothing(self):
        answers: list[str] = []
        gate = _gate(answers)
        gate.speaking(True)
        gate.speaking(False)
        await asyncio.sleep(0.03)
        assert answers == []

    async def test_a_session_ending_gets_what_was_held_and_answers_nothing(self):
        answers: list[str] = []
        gate = _gate(answers)
        gate.take("Mira el correo y")
        assert gate.drain() == "Mira el correo y"
        await asyncio.sleep(0.03)
        assert answers == [] and gate.drain() == ""


class TestWhenATurnIsOpen:
    async def test_holding_and_speaking_count(self):
        gate = _gate([], hold=10)
        assert not gate.turn_open
        gate.speaking(True)
        assert gate.turn_open
        gate.speaking(False)
        assert not gate.turn_open
        gate.take("Mira y")
        assert gate.turn_open
        gate.drain()

    async def test_a_transcript_opens_one_until_it_commits_or_goes_stale(self):
        clock = [100.0]
        gate = _gate([], now=lambda: clock[0])
        gate.heard()
        assert gate.turn_open
        gate.take("Vale.")
        assert not gate.turn_open
        gate.heard()
        clock[0] += OPEN_TURN_SECONDS + 1
        assert not gate.turn_open

    async def test_uses_the_loop_s_clock_by_default(self):
        gate = TurnGate(lambda _: None)
        gate.heard()
        assert gate.turn_open


def _message(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[text], text_content=text)


def _agent(pipeline: str | None = None, memory: object | None = None) -> KwamiAgent:
    config = KwamiConfig()
    if pipeline:
        config.voice.pipeline_type = pipeline  # type: ignore[assignment]
    return KwamiAgent(config, memory=memory)  # type: ignore[arg-type]


def _memory() -> MagicMock:
    memory = MagicMock(is_initialized=True, _cached_user_name="Ana")
    memory.buffer_user_message = AsyncMock()
    return memory


class TestTheAgent:
    async def test_an_open_turn_gets_no_reply_and_no_memory_yet(self):
        memory = _memory()
        agent = _agent(memory=memory)
        with pytest.raises(StopResponse):
            await agent.on_user_turn_completed(None, _message("Mira el correo y"))
        memory.buffer_user_message.assert_not_called()
        assert agent.turn_open
        agent._turn_gate.drain()

    async def test_the_next_piece_answers_and_remembers_the_whole(self):
        memory = _memory()
        agent = _agent(memory=memory)
        with pytest.raises(StopResponse):
            await agent.on_user_turn_completed(None, _message("Mira el correo y"))
        message = _message("dime quién es Alex.")
        await agent.on_user_turn_completed(None, message)
        whole = "Mira el correo y dime quién es Alex."
        assert message.content == [whole]
        memory.buffer_user_message.assert_awaited_once_with(whole, name="Ana")

    async def test_a_realtime_model_is_left_alone(self):
        memory = _memory()
        agent = _agent(REALTIME_PIPELINE, memory=memory)
        assert agent._turn_gate is None and not agent.turn_open
        await agent.on_user_turn_completed(None, _message("Mira el correo y"))
        memory.buffer_user_message.assert_awaited_once()
        await agent.on_exit()

    async def test_silence_replies_and_remembers(self, monkeypatch):
        memory = _memory()
        agent = _agent(memory=memory)
        session = MagicMock()
        monkeypatch.setattr(KwamiAgent, "session", property(lambda _: session))
        agent._answer_held_turn("Mira el correo y")
        await asyncio.sleep(0)
        session.generate_reply.assert_called_once_with(user_input="Mira el correo y")
        memory.buffer_user_message.assert_awaited_once()

    async def test_a_session_gone_before_the_reply_is_not_an_error(self, monkeypatch):
        agent = _agent()
        session = MagicMock()
        session.generate_reply.side_effect = RuntimeError("closed")
        monkeypatch.setattr(KwamiAgent, "session", property(lambda _: session))
        agent._answer_held_turn("Mira y")

    async def test_leaving_remembers_what_was_held(self):
        memory = _memory()
        agent = _agent(memory=memory)
        with pytest.raises(StopResponse):
            await agent.on_user_turn_completed(None, _message("Mira el correo y"))
        await agent.on_exit()
        memory.buffer_user_message.assert_awaited_once_with("Mira el correo y", name="Ana")

    async def test_a_memory_failure_does_not_stop_the_answer(self):
        memory = _memory()
        memory.buffer_user_message.side_effect = RuntimeError("zep")
        agent = _agent(memory=memory)
        await agent.on_user_turn_completed(None, _message("Vale."))

    async def test_listens_for_the_user(self):
        agent = _agent()
        agent._on_user_state_changed(SimpleNamespace(new_state="speaking"))
        assert agent.turn_open
        agent._on_user_state_changed(SimpleNamespace(new_state="listening"))
        assert not agent.turn_open
        agent._on_user_transcribed(SimpleNamespace(is_final=False))
        assert not agent.turn_open
        agent._on_user_transcribed(SimpleNamespace(is_final=True))
        assert agent.turn_open

    async def test_only_the_standard_pipeline_listens_for_them(self, monkeypatch):
        for pipeline, events in ((None, 3), (REALTIME_PIPELINE, 1)):
            agent = _agent(pipeline)
            session = MagicMock()
            agent._register_session_listeners = KwamiAgent._register_session_listeners.__get__(
                agent
            )
            with monkeypatch.context() as patch:
                patch.setattr(KwamiAgent, "session", session)
                agent._register_session_listeners()
            assert session.on.call_count == events

    async def test_a_realtime_agent_ignores_those_events(self):
        agent = _agent(REALTIME_PIPELINE)
        agent._on_user_state_changed(SimpleNamespace(new_state="speaking"))
        agent._on_user_transcribed(SimpleNamespace(is_final=True))
        assert not agent.turn_open

    @pytest.mark.parametrize(
        ("said", "kept"),
        [("Señor,", False), ("Señor, la investigación ha terminado y", True), ("Hecho.", True)],
    )
    async def test_an_answer_cut_at_its_first_words_is_not_remembered(self, said, kept):
        memory = _memory()
        agent = _agent(memory=memory)
        agent._persist_assistant_turn = AsyncMock()
        item = SimpleNamespace(role="assistant", content=[said], text_content=said)
        agent._on_conversation_item_added(SimpleNamespace(item=item))
        await asyncio.sleep(0)
        assert agent._persist_assistant_turn.called is kept


class TestTheVocabulary:
    def test_only_nova_3_takes_it(self):
        voice = KwamiConfig().voice
        voice.stt_keyterms = ["Nexow", "Claude"]
        words = _deepgram_keyterms("nova-3", voice)
        assert words[:2] == ["Nexow", "Claude"]
        assert words.count("Claude") == 1
        assert set(DEFAULT_KEYTERMS) <= set(words)
        assert not isinstance(_deepgram_keyterms("nova-2", voice), list)

    def test_comes_from_the_panel_clean(self):
        parsed = parse_listening(
            {"enhancements": {"vocabulary": [" Nexow ", "Nexow", "", 3, "x" * 60]}}
        )
        assert parsed["stt_keyterms"] == ["Nexow", "x" * 40]

    def test_absent_changes_nothing(self):
        assert "stt_keyterms" not in parse_listening({"enhancements": {"vocabulary": "Nexow"}})
        assert "stt_keyterms" not in parse_listening({"enhancements": {}})


class TestWhatATaskCarries:
    def _bridge(self, session: object | None):
        from src.brain.bridge import Bridge

        bridge = Bridge.__new__(Bridge)
        bridge.session = session  # type: ignore[assignment]
        return bridge

    def _session(self, turns: int = 3, memory: object = "Vive en Madrid."):
        items = [
            SimpleNamespace(role="user" if i % 2 == 0 else "assistant", text_content=f"turno {i}")
            for i in range(turns)
        ]
        items.insert(0, SimpleNamespace(role="system", text_content="prompt"))
        items.append(SimpleNamespace(role="user", text_content=None))
        agent = SimpleNamespace(_chat_ctx=SimpleNamespace(items=items), _memory_text=memory)
        return SimpleNamespace(current_agent=agent)

    def test_the_conversation_and_the_memory_travel_with_it(self):
        task = self._bridge(self._session())._with_conversation("Busca eso", "de Nexow")
        assert task.startswith("Busca eso\n\nContext from the conversation:\nde Nexow")
        assert "user: turno 0\nassistant: turno 1\nuser: turno 2" in task
        assert "Vive en Madrid." in task and "prompt" not in task

    def test_it_is_bounded(self):
        bridge = self._bridge(self._session(turns=40, memory="x" * 9000))
        recent = bridge._recent_conversation()
        assert len(recent.splitlines()) == CONTEXT_TURNS
        assert "turno 39" in recent
        assert len(bridge._remembered()) == CONTEXT_MEMORY_CHARS

    def test_without_a_call_the_task_stands_alone(self):
        assert self._bridge(None)._with_conversation("Busca eso", "") == "Busca eso"

    def test_a_memory_that_is_not_text_is_ignored(self):
        assert self._bridge(self._session(memory=None))._remembered() == ""


class TestAQuickAnswer:
    def _bridge(self, tts: object = "tts", delivery: str = "next_pause"):
        from src.brain.bridge import Bridge

        bridge = Bridge.__new__(Bridge)
        bridge.config = SimpleNamespace(delivery=delivery, relay="everything")
        voice = SimpleNamespace(tts=tts, turn_open=False, _chat_ctx=None, _memory_text="")
        bridge.session = SimpleNamespace(
            current_agent=voice,
            say=MagicMock(),
            generate_reply=MagicMock(),
            agent_state="listening",
            user_state="listening",
        )
        bridge._closed = False
        return bridge

    def _task(self, **fields):
        from src.brain.bridge import BrainTask

        values = {
            "id": "t1",
            "title": "Semana",
            "task": "x",
            "kind": "question",
            "difficulty": "simple",
        }
        values.update(status="done", result="Siete días.\n\nDetalle:\nDe lunes a domingo.")
        values.update(fields)
        return BrainTask(**values)

    def test_is_said_as_the_worker_wrote_it(self):
        assert self._bridge()._spoken_as_written(self._task(seconds=2.0)) == "Siete días."

    @pytest.mark.parametrize(
        "fields",
        [
            {"status": "failed"},
            {"council": True},
            {"kind": "code"},
            {"seconds": 26.0},
            {"result": "x" * 601},
            {"result": "Mira https://x.y"},
            {"result": "**Siete**"},
        ],
    )
    def test_anything_else_goes_through_the_voice(self, fields):
        assert self._bridge()._spoken_as_written(self._task(**fields)) == ""

    def test_a_realtime_model_says_it_itself(self):
        assert self._bridge(tts=None)._spoken_as_written(self._task()) == ""

    def test_on_request_only_says_it_is_ready(self):
        assert self._bridge(delivery="on_request")._spoken_as_written(self._task()) == ""

    async def test_delivery_skips_the_second_model_call(self, monkeypatch):
        bridge = self._bridge()
        monkeypatch.setattr(bridge, "_wait_for_pause", AsyncMock(return_value=True))
        monkeypatch.setattr(bridge, "_remember", MagicMock())
        task = self._task(seconds=2.0)
        await bridge._deliver(task)
        bridge.session.say.assert_called_once_with("Siete días.", allow_interruptions=True)
        bridge.session.generate_reply.assert_not_called()
        assert task.delivered

    async def test_a_slow_answer_breaks_the_silence_once(self, monkeypatch):
        import src.brain.bridge as bridge_module

        monkeypatch.setattr(bridge_module, "STILL_WORKING_SECONDS", 0.0)
        bridge = self._bridge()
        await bridge._say_still_working(self._task(status="running"))
        bridge.session.generate_reply.assert_called_once()

    @pytest.mark.parametrize(
        "change",
        [
            lambda b, t: setattr(t, "status", "done"),
            lambda b, t: setattr(b, "session", None),
            lambda b, t: setattr(b.session, "agent_state", "speaking"),
            lambda b, t: setattr(b.session, "user_state", "speaking"),
            lambda b, t: setattr(b.session.current_agent, "turn_open", True),
        ],
    )
    async def test_it_stays_quiet_when_there_is_no_silence_to_break(self, monkeypatch, change):
        import src.brain.bridge as bridge_module

        monkeypatch.setattr(bridge_module, "STILL_WORKING_SECONDS", 0.0)
        bridge, task = self._bridge(), self._task(status="running")
        session = bridge.session
        change(bridge, task)
        await bridge._say_still_working(task)
        session.generate_reply.assert_not_called()
