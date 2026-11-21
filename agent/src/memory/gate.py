"""What the memory graph learns from, and the lines it is written.

Every turn stays in the thread -- the history panel and the next session's
continuity read it -- but Zep extracts entities and facts only from turns that
can hold something lasting. Measured on a real graph (2026-11-17), most of what
it held was one of two kinds of junk:

- noise: a TV or the street, transcribed as if the user spoke, which the
  assistant itself answered that it did not understand ("coche", "loot");
- one-off orders: a turn that made the assistant act ("abre Fox News", "lanza
  prueba-nube") turned into a fact about the user.

An order that also states something lasting ("a partir de ahora", "prefiero",
"mi hermana se llama...") still goes to the graph: that part is not an errand.

Zep's own extraction instructions would be the natural place for this, but
they are not available on every plan, so the agent decides before writing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

#: The assistant's own words when it could not make out the turn.
NOT_UNDERSTOOD = re.compile(
    r"\bno (?:le |te |lo )?(?:he |había )?(?:entendido|entiendo|comprendido|comprendo|captado)\b"
    r"|\bno (?:le |te )?(?:he )?(?:oído|escuchado) bien\b"
    r"|\bno sé a qué (?:se refiere|te refieres|se refería|te referías)\b"
    r"|\b(?:puede|podría|puedes|podrías) repetir"
    r"|\brepet[ií]rmelo\b"
    r"|\b(?:did not|didn't|could not|couldn't) (?:quite )?(?:catch|understand|hear|get) "
    r"(?:that|you|what)\b"
    r"|\b(?:could|can) you (?:please )?(?:repeat|say that again)\b"
    r"|\bnot sure what you mean\b",
    re.IGNORECASE,
)

#: Words that make a turn about the user, not only about the errand.
LASTING = re.compile(
    r"\bsiempre\b|\bnunca\b|\ba partir de ahora\b|\bde ahora en adelante\b|\bdesde ahora\b"
    r"|\bprefiero\b|\b(?:no )?me gusta|\bme encanta|\bodio\b|\bme molesta|\bten en cuenta\b"
    r"|\bme llamo\b|\bmi nombre\b|\bvivo en\b|\btrabajo (?:en|como)\b"
    r"|\bmis? (?:mujer|marido|pareja|novi[ao]s?|hij[ao]s?|herman[ao]s?|madre|padre|padres"
    r"|perros?|gat[ao]s?|empresa|equipo|jef[ae]|socios?|clientes?)\b"
    r"|\balways\b|\bnever\b|\bfrom now on\b|\bi prefer\b|\bi (?:really )?(?:like|love|hate)\b"
    r"|\bkeep in mind\b|\bmy name\b|\bi live\b|\bi work\b"
    r"|\bmy (?:wife|husband|partner|girlfriend|boyfriend|sons?|daughters?|brothers?|sisters?"
    r"|mother|father|parents|dogs?|cats?|company|team|boss|clients?)\b",
    re.IGNORECASE,
)

#: Tools that read or keep memory, or only look something up about the call
#: itself. A turn that used them asked about the user, it did not send the
#: assistant on an errand.
PASSIVE_TOOLS = frozenset(
    {
        "recall_memories",
        "get_memory_status",
        "get_current_time",
        "get_kwami_info",
        "read_user_profile",
        "remember_about_user",
    }
)

#: "Me llamo Bruno", "mi nombre es Bruno", "soy Bruno", "my name is Ada". The name
#: has to come capitalised: "llámame mañana" is not a name, and transcription
#: capitalises names, not the words around them.
_STATED_NAME = re.compile(
    r"(?i:\b(?:me llamo|mi nombre es|ll[aá]mame|soy|my name is|call me))\s+"
    r"([A-ZÁÉÍÓÚÑ][a-záéíóúñü]{1,30})\b"
)
_NEVER_A_NAME = frozenset({"yo", "el", "la", "un", "una", "user", "usuario", "kwami"})

#: How much of a finished job's result the work journal keeps.
WORK_RESULT_CHARS = 300


def worth_extracting(user_text: str, assistant_text: str, tools: Iterable[str] = ()) -> bool:
    """Whether the graph should learn from this user turn.

    Args:
        user_text: What the user said or typed.
        assistant_text: The assistant's answer to it.
        tools: Names of the tools the answer called.
    """
    if NOT_UNDERSTOOD.search(assistant_text or ""):
        return False
    acted = any(name not in PASSIVE_TOOLS for name in tools)
    return not acted or bool(LASTING.search(user_text or ""))


def stated_name(text: str, *, agent_name: str = "") -> str | None:
    """The name the user just gave for themselves, or None."""
    match = _STATED_NAME.search(text or "")
    if not match:
        return None
    name = match.group(1)
    if name.lower() in _NEVER_A_NAME or name.lower() == agent_name.strip().lower():
        return None
    return name


def work_line(
    *,
    assistant: str,
    user: str | None,
    date: str,
    title: str,
    result: str,
    language: str = "",
) -> str:
    """One line of the work journal: who did what, for whom, when, and what came of it.

    `language` is a locale ("es-ES") or a name ("Spanish (Spain)"); Spanish gets
    a Spanish line, anything else English.

    The assistant is the subject on purpose: written as "work finished for Bruno",
    Zep credited the user with the work the assistant had done.
    """
    outcome = " ".join(result.split("Detalle:", 1)[0].split())
    if len(outcome) > WORK_RESULT_CHARS:
        outcome = outcome[:WORK_RESULT_CHARS].rsplit(" ", 1)[0] + "..."
    what = title.strip().rstrip(".")
    lang = language.strip().lower()
    if lang == "es" or lang.startswith(("es-", "es_", "spanish", "español")):
        line = f"{assistant} hizo un trabajo para {user or 'el usuario'} el {date}: {what}."
        return f"{line} Resultado: {outcome}" if outcome else line
    line = f"{assistant} did a piece of work for {user or 'the user'} on {date}: {what}."
    return f"{line} Result: {outcome}" if outcome else line
