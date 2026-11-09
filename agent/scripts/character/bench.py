"""Character bench: does the soul still sound like itself?

Runs every scenario in `scenarios.json` against the realtime model with the
system prompt the agent would really send -- persona, fixed guidance gated on
this deployment's credentials, and the app's capability guidance -- then scores
each reply twice:

* deterministic checks (banned phrases, Latin American vocabulary, tuteo,
  length, markdown), which never drift between runs;
* a judge model with a rubric, for what a regex cannot see (irony at the right
  level, pushback with a reason, initiative).

The soul is read from the SDK preset itself (`kwami/src/soul/<id>.soul.json`),
so the bench and the app can never disagree about what Jarvis is.

Usage, from `agent/`:

    uv run --extra google python scripts/character/bench.py --label baseline
    uv run --extra google python scripts/character/bench.py --voice Iapetus --save-audio

The key comes from GEMINI_API_KEY or GOOGLE_API_KEY, after loading the repo's
`.env` the same way `main.py` does. Reports land in `scripts/character/out/`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parents[1]
REPO_DIR = AGENT_DIR.parent
DEFAULT_SOUL = REPO_DIR.parent / "kwami" / "src" / "soul" / "jarvis.soul.json"
OUT_DIR = HERE / "out"

sys.path.insert(0, str(AGENT_DIR))

from src.envfile import load_env_file  # noqa: E402

load_env_file(REPO_DIR / ".env")

from google import genai  # noqa: E402
from google.genai import errors as genai_errors  # noqa: E402
from google.genai import types  # noqa: E402

from src.domain import KwamiSoulConfig, build_system_prompt  # noqa: E402
from src.domain.capabilities import CAPABILITIES  # noqa: E402
from src.domain.charter import parse_charter  # noqa: E402
from src.domain.tool_gating import unavailable_builtin_tools  # noqa: E402
from src.settings import Settings  # noqa: E402

# -- Deterministic checks ------------------------------------------------------

#: Assistant filler the soul forbids. Matched on the lowercased reply.
BANNED = (
    "¡claro",
    "¡genial",
    "excelente pregunta",
    "buena pregunta",
    "gran pregunta",
    "¡por supuesto!",
    "¿en qué más puedo",
    "¿algo más en lo que",
    "estaré encantado de ayudar",
    "¡qué buena",
)

#: Latin American forms a Castilian speaker would not use.
LATAM = (
    "computadora",
    "celular",
    "ahorita",
    "platicar",
    "chévere",
    "jugo",
    "enojad",
)

#: Second-person informal forms; the soul addresses the user as usted.
TUTEO = re.compile(
    r"\b(tú|tienes|puedes|quieres|necesitas|sabes|estás|eres|hazlo|dime|te|ti|contigo|tus?)\b"
)
MARKDOWN = re.compile(r"[*#_`]|[\U0001F300-\U0001FAFF]")
MAX_WORDS = 60


def check(reply: str) -> dict[str, Any]:
    low = reply.lower()
    return {
        "banned": [p for p in BANNED if p in low],
        "latam": [w for w in LATAM if re.search(rf"\b{w}", low)],
        "tuteo": sorted(set(TUTEO.findall(low))),
        "markdown": bool(MARKDOWN.search(reply)),
        "words": len(reply.split()),
        "too_long": len(reply.split()) > MAX_WORDS,
        "senor": "señor" in low,
    }


def check_passed(c: dict[str, Any]) -> bool:
    return not (c["banned"] or c["latam"] or c["tuteo"] or c["markdown"] or c["too_long"])


# -- Judge ---------------------------------------------------------------------

RUBRIC = """Eres un director de doblaje exigente. Evalúas si una respuesta encaja con este personaje:

Jarvis: mayordomo inglés clásico que habla castellano de España culto. Trata al usuario de usted y le llama «señor». Calma imperturbable, humor seco por atenuación, frases cortas, la cifra antes que el adjetivo. Honesto antes que complaciente: si una idea es mala lo dice una vez, con el dato y una alternativa. Anticipa el siguiente paso. Nunca halaga ni muestra entusiasmo impostado. Ante salud, duelo o peligro: sin ironía.

Situación: {title}
Conversación previa: {history}
El usuario dice: «{say}»
Respuesta de Jarvis: «{reply}»

Expectativas de esta situación: ironía {irony}; {pushback}

Puntúa de 1 a 5 (5 = perfecto) y justifica en una frase:
- personaje: ¿suena a mayordomo inglés en castellano, no a asistente genérico?
- ironia: ¿el humor está al nivel esperado? (si la ironía esperada es "none", cualquier ironía es un 1)
- franqueza: ¿dice la verdad incómoda cuando toca, una sola vez y con alternativa? (si no toca objetar, puntúa si evita el servilismo)
- concision: ¿frases cortas, sin relleno?
- iniciativa: ¿propone el siguiente paso en lugar de preguntar si desea algo más?
- castellano: ¿vocabulario y giros de España, registro culto?
"""

#: Tried in order once the requested judge keeps answering 503.
JUDGE_FALLBACKS = ("gemini-3.7-flash", "gemini-3.6-flash")
#: Free-tier quotas are per minute, so a 429 is worth waiting out once.
JUDGE_RETRY_DELAYS = (5.0, 30.0)

SCORE_KEYS = ("personaje", "ironia", "franqueza", "concision", "iniciativa", "castellano")

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "integer"} for k in SCORE_KEYS},
        "nota": {"type": "string"},
    },
    "required": [*SCORE_KEYS, "nota"],
}


def describe_expectation(expect: dict[str, Any]) -> tuple[str, str]:
    irony = {"none": "prohibida", "light": "ligera, bienvenida"}.get(
        expect.get("irony", ""), "libre"
    )
    if expect.get("pushback"):
        pushback = "debe objetar con criterio antes de obedecer."
    elif expect.get("no_repeat"):
        pushback = "ya objetó antes: NO debe repetir la objeción; debe ayudar."
    else:
        pushback = "no hace falta objetar."
    return irony, pushback


# -- Model calls -----------------------------------------------------------------


@dataclass
class Reply:
    text: str
    audio: bytes
    seconds_to_first: float | None
    error: str | None = None


def _history(scenario: dict[str, Any]) -> list[types.Content]:
    return [
        types.Content(role=t["role"], parts=[types.Part(text=t["text"])])
        for t in scenario.get("turns", [])
    ]


async def ask_live(
    client: genai.Client, model: str, voice: str, prompt: str, scenario: dict[str, Any]
) -> Reply:
    """One fresh Live session per scenario, so no reply colours the next."""
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part(text=prompt)]),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    text: list[str] = []
    audio = bytearray()
    first: float | None = None
    async with client.aio.live.connect(model=model, config=config) as session:
        history = _history(scenario)
        if history:
            await session.send_client_content(turns=history, turn_complete=False)
        started = time.perf_counter()
        await session.send_realtime_input(text=scenario["say"])
        async for message in session.receive():
            content = message.server_content
            if not content:
                continue
            if content.model_turn:
                for part in content.model_turn.parts or []:
                    if part.inline_data and part.inline_data.data:
                        if first is None:
                            first = time.perf_counter() - started
                        audio.extend(part.inline_data.data)
            if content.output_transcription and content.output_transcription.text:
                text.append(content.output_transcription.text)
            if content.turn_complete:
                break
    return Reply("".join(text).strip(), bytes(audio), first)


async def ask_text(
    client: genai.Client, model: str, prompt: str, scenario: dict[str, Any]
) -> Reply:
    contents = [
        *_history(scenario),
        types.Content(role="user", parts=[types.Part(text=scenario["say"])]),
    ]
    started = time.perf_counter()
    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=prompt),
    )
    return Reply((response.text or "").strip(), b"", time.perf_counter() - started)


async def judge(
    client: genai.Client, model: str, scenario: dict[str, Any], reply: str
) -> dict[str, Any]:
    irony, pushback = describe_expectation(scenario.get("expect", {}))
    history = (
        " / ".join(f"{t['role']}: {t['text']}" for t in scenario.get("turns", [])) or "ninguna"
    )
    contents = RUBRIC.format(
        title=scenario["title"],
        history=history,
        say=scenario["say"],
        reply=reply,
        irony=irony,
        pushback=pushback,
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JUDGE_SCHEMA,
        temperature=0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    # Flash models answer 503 under load; wait it out, then try the fallback,
    # so one busy minute does not leave a run with no scores at all.
    failures: list[str] = []
    for candidate in (model, *JUDGE_FALLBACKS):
        for delay in JUDGE_RETRY_DELAYS:
            try:
                response = await client.aio.models.generate_content(
                    model=candidate, contents=contents, config=config
                )
                verdict = json.loads(response.text or "{}")
                verdict["juez"] = candidate
                return verdict
            except (genai_errors.ServerError, genai_errors.ClientError) as exc:
                if isinstance(exc, genai_errors.ClientError) and exc.code not in (429,):
                    failures.append(f"{candidate}: {exc.code}")
                    break
                failures.append(f"{candidate}: {exc.code}")
                await asyncio.sleep(delay)
    raise RuntimeError(f"judge unavailable ({', '.join(failures)})")


# -- Run -------------------------------------------------------------------------


def load_soul(path: Path) -> KwamiSoulConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return KwamiSoulConfig(
        name=data.get("name", "Kwami"),
        personality=data.get("personality", ""),
        system_prompt=data.get("systemPrompt", ""),
        traits=list(data.get("traits", [])),
        language=data.get("language", "en"),
        conversation_style=data.get("conversationStyle", ""),
        response_length=data.get("responseLength", "medium"),
        emotional_tone=data.get("emotionalTone", "neutral"),
        emotional_traits=dict(data.get("emotionalTraits", {})),
        # The preset's own rules, as the app sends them in `soul.charter`.
        charter=parse_charter(data.get("charter")),
    )


def build_prompt(soul: KwamiSoulConfig, bare: bool) -> str:
    app_tools = () if bare else frozenset().union(*(c.requires for c in CAPABILITIES))
    return build_system_prompt(
        soul,
        client_tool_names=app_tools,
        unavailable_tools=unavailable_builtin_tools(Settings.from_env()),
    )


def write_wav(path: Path, pcm: bytes) -> None:
    """Live returns 24 kHz, 16-bit, mono PCM."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(pcm)


async def run_one(
    args: argparse.Namespace,
    client: genai.Client,
    prompt: str,
    scenario: dict[str, Any],
    gate: asyncio.Semaphore,
    audio_dir: Path | None,
) -> dict[str, Any]:
    async with gate:
        try:
            if args.target == "live":
                reply = await asyncio.wait_for(
                    ask_live(client, args.model, args.voice, prompt, scenario), args.timeout
                )
            else:
                reply = await ask_text(client, args.model, prompt, scenario)
        except Exception as exc:  # one broken scenario must not sink the run
            reply = Reply("", b"", None, error=f"{type(exc).__name__}: {exc}")

        result: dict[str, Any] = {
            "id": scenario["id"],
            "title": scenario["title"],
            "say": scenario["say"],
            "reply": reply.text,
            "seconds_to_first_audio": reply.seconds_to_first,
            "error": reply.error,
        }
        if reply.error or not reply.text:
            result["error"] = reply.error or "empty reply (the model chose not to answer)"
            return result

        result["checks"] = check(reply.text)
        result["checks_passed"] = check_passed(result["checks"])
        if audio_dir and reply.audio:
            write_wav(audio_dir / f"{scenario['id']}.wav", reply.audio)
        return result


async def judge_all(
    client: genai.Client,
    model: str,
    scenarios: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    """Score every answered reply that has no verdict yet, one at a time.

    Serial on purpose: the judge shares the key's per-minute quota, and twenty
    parallel requests turned into fifteen 429s on the first run.
    """
    by_id = {s["id"]: s for s in scenarios}
    for result in results:
        if result.get("error") or "judge" in result or result["id"] not in by_id:
            continue
        try:
            result["judge"] = await judge(client, model, by_id[result["id"]], result["reply"])
            result.pop("judge_error", None)
        except Exception as exc:
            result["judge_error"] = f"{type(exc).__name__}: {exc}"


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [r for r in results if not r.get("error")]
    judged = [r["judge"] for r in answered if "judge" in r]
    averages = (
        {k: round(sum(j.get(k, 0) for j in judged) / len(judged), 2) for k in SCORE_KEYS}
        if judged
        else {}
    )
    return {
        "scenarios": len(results),
        "answered": len(answered),
        "checks_passed": sum(1 for r in answered if r.get("checks_passed")),
        "senor_rate": round(sum(1 for r in answered if r["checks"]["senor"]) / len(answered), 2)
        if answered
        else 0,
        "judge_averages": averages,
        "judge_overall": round(sum(averages.values()) / len(averages), 2) if averages else None,
    }


def print_report(results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    for r in results:
        print(f"\n[{r['id']}] {r['title']}\n  > {r['say']}")
        if r.get("error"):
            print(f"  !! {r['error']}")
            continue
        print(f"  < {r['reply']}")
        c = r["checks"]
        flags = [
            f"{name}: {', '.join(c[name])}" for name in ("banned", "latam", "tuteo") if c[name]
        ]
        if c["markdown"]:
            flags.append("markdown")
        if c["too_long"]:
            flags.append(f"{c['words']} palabras")
        print(f"  checks: {'OK' if not flags else ' | '.join(flags)}")
        if "judge" in r:
            j = r["judge"]
            scores = " ".join(f"{k[:4]}={j.get(k)}" for k in SCORE_KEYS)
            print(f"  juez: {scores} -- {j.get('nota', '')}")
        elif r.get("judge_error"):
            print(f"  juez: sin nota ({r['judge_error'][:120]})")
    print("\n== Resumen ==")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--soul", type=Path, default=DEFAULT_SOUL)
    parser.add_argument("--scenarios", type=Path, default=HERE / "scenarios.json")
    parser.add_argument("--target", choices=("live", "text"), default="live")
    parser.add_argument("--model", default="gemini-3.8-live")
    parser.add_argument("--voice", default="Charon")
    parser.add_argument("--judge-model", default="gemini-3.8-flash")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--bare", action="store_true", help="leave out the app's capability guidance"
    )
    parser.add_argument("--only", nargs="*", help="scenario ids to run")
    parser.add_argument("--save-audio", action="store_true", help="write one WAV per reply")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--label", default="run")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument(
        "--no-charter", action="store_true", help="drop the preset's charter: the soul alone"
    )
    parser.add_argument(
        "--rejudge", type=Path, help="score the unjudged replies of a saved report, in place"
    )
    args = parser.parse_args()
    # The Windows console defaults to a legacy code page and mangles every accent.
    sys.stdout.reconfigure(encoding="utf-8")

    soul = load_soul(args.soul)
    if args.no_charter:
        soul.charter = parse_charter(None)
    prompt = build_prompt(soul, args.bare)
    if args.print_prompt:
        print(prompt)
        print(f"\n-- {len(prompt)} caracteres --")
        return

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit(
            "Falta GEMINI_API_KEY o GOOGLE_API_KEY (en el entorno o en el .env del repo)."
        )
    client = genai.Client(api_key=key)

    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))["scenarios"]

    if args.rejudge:
        report = json.loads(args.rejudge.read_text(encoding="utf-8"))
        await judge_all(client, args.judge_model, scenarios, report["results"])
        report["judge_model"] = args.judge_model
        report["summary"] = summarise(report["results"])
        args.rejudge.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print_report(report["results"], report["summary"])
        print(f"\nInforme: {args.rejudge}")
        return
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in set(args.only)]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(exist_ok=True)
    audio_dir = None
    if args.save_audio and args.target == "live":
        audio_dir = OUT_DIR / f"{stamp}-{args.label}-{args.voice}"
        audio_dir.mkdir()

    gate = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(
        *(run_one(args, client, prompt, s, gate, audio_dir) for s in scenarios)
    )
    if not args.no_judge:
        await judge_all(client, args.judge_model, scenarios, results)
    summary = summarise(results)
    report = {
        "label": args.label,
        "target": args.target,
        "model": args.model,
        "voice": args.voice if args.target == "live" else None,
        "judge_model": None if args.no_judge else args.judge_model,
        "soul": str(args.soul),
        "charter": soul.charter.enabled,
        "prompt_chars": len(prompt),
        "summary": summary,
        "results": results,
    }
    out = OUT_DIR / f"{stamp}-{args.label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(results, summary)
    print(f"\nInforme: {out}")


if __name__ == "__main__":
    asyncio.run(main())
