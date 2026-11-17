"""What the memory graph learns from (memory/gate.py).

The turns below are taken from a real graph measured on 2026-11-17, where most
facts were noise the assistant had not understood or errands stored as facts
about the user.
"""

from __future__ import annotations

import pytest

from src.memory.gate import WORK_RESULT_CHARS, stated_name, work_line, worth_extracting


@pytest.mark.parametrize(
    ("user", "assistant"),
    [
        (
            "No, ese coche no vas a probar el coche, ¿sabes? Dime.",
            "No he entendido la frase, señor. Menciona un coche.",
        ),
        ("Como 4º.", "Perdone, señor. No he entendido lo que me ha dicho."),
        ("¿Cuándo", "Señor, no sé a qué se refiere con «cuándo»."),
        ("loot", "Sorry, I didn't catch that. Could you repeat it?"),
        ("blah", "¿Podría repetirlo, señor?"),
    ],
)
def test_noise_the_assistant_did_not_understand_is_not_learned(user: str, assistant: str) -> None:
    assert worth_extracting(user, assistant) is False


@pytest.mark.parametrize(
    ("user", "tools"),
    [
        ("Abre Fox News en la pantalla grande.", ["navigate_to"]),
        ("Lanza la prueba en la nube.", ["delegate_task"]),
        ("Pon la pantalla en cuadrícula.", ["set_ui_control"]),
        ("Search the web for flights to Rome", ["web_search", "recall_memories"]),
    ],
)
def test_an_errand_is_not_learned(user: str, tools: list[str]) -> None:
    assert worth_extracting(user, "Hecho, señor.", tools) is False


@pytest.mark.parametrize(
    "user",
    [
        "A partir de ahora abre siempre las noticias en inglés.",
        "Prefiero la pantalla en cuadrícula, ponla así.",
        "Mi hermana Ana cumple años el martes, mándale un correo.",
        "From now on, open the news in English.",
    ],
)
def test_an_errand_that_states_something_lasting_is_learned(user: str) -> None:
    assert worth_extracting(user, "Hecho, señor.", ["navigate_to"]) is True


def test_conversation_and_memory_lookups_are_learned() -> None:
    assert worth_extracting("Estoy montando un equipo de programadores.", "Suena bien.") is True
    assert worth_extracting("¿Qué sabes de mi proyecto?", "Que...", ["recall_memories"]) is True
    assert worth_extracting("", "") is True


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Me llamo Bruno.", "Bruno"),
        ("Hola, soy Bruno y quiero probar algo", "Bruno"),
        ("mi nombre es Álvaro", "Álvaro"),
        ("My name is Ada", "Ada"),
        ("Llámame mañana", None),
        ("soy programador", None),
        ("Soy Jarvis, el asistente", None),
        ("Yo soy Yo", None),
        ("", None),
    ],
)
def test_the_name_the_user_gives(text: str, name: str | None) -> None:
    assert stated_name(text, agent_name="Jarvis") == name


def test_a_work_line_names_who_did_the_work() -> None:
    line = work_line(
        assistant="Jarvis",
        user="Bruno",
        date="2026-11-16",
        title="Revisión del despliegue del agente.",
        result="Faltaba el secreto KWAMI_SDK_TOKEN.\n\nDetalle: largo informe que no se guarda",
        language="es-ES",
    )
    assert line == (
        "Jarvis hizo un trabajo para Bruno el 2026-11-16: Revisión del despliegue del agente. "
        "Resultado: Faltaba el secreto KWAMI_SDK_TOKEN."
    )


@pytest.mark.parametrize("language", ["es", "Spanish (Spain)", "español"])
def test_a_spanish_assistant_writes_a_spanish_line(language: str) -> None:
    line = work_line(assistant="J", user=None, date="d", title="t", result="", language=language)
    assert line == "J hizo un trabajo para el usuario el d: t."


def test_other_languages_get_an_english_line_and_a_capped_result() -> None:
    line = work_line(
        assistant="Kwami", user=None, date="2026-11-16", title="Report", result="word " * 200
    )
    assert line.startswith("Kwami did a piece of work for the user on 2026-11-16: Report. Result: ")
    assert line.endswith("...")
    assert len(line.split("Result: ", 1)[1]) <= WORK_RESULT_CHARS + 3
    assert work_line(assistant="K", user="Ada", date="d", title="t", result="") == (
        "K did a piece of work for Ada on d: t."
    )
