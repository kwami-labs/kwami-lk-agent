"""Voice lab: how fast, and how good, is every voice provider the agent can build?

Every case goes through the agent's own factories (`create_tts`, `create_stt`,
`create_llm`), so what is measured is what a session would really run --
including a factory that silently falls back to another provider, which is
reported as FALLBACK instead of being timed as if it were the one asked for.

Measured, per case and over `--runs` repetitions (median reported):

* TTS: time to the first audio frame (TTFB), total synthesis time, seconds of
  audio and real-time factor. The first run's audio is saved as a WAV so the
  voices can be compared by ear.
* STT: the reference sentence is spoken (by a fixed reference TTS) and fed in
  real time; latency is the time from the end of speech to the final
  transcript that completes it, and accuracy is the word match against the
  sentence.
* LLM: time to the first token (TTFT) and to the whole short answer.

The report also estimates the voice-to-voice delay of the best combinations:
STT final + LLM first token + TTS first audio. Network and turn detection add
to that in a live session; the console's E2E metric is the real figure.

Usage, from `agent/`:

    uv run python -X utf8 scripts/voice_lab/bench.py
    uv run python -X utf8 scripts/voice_lab/bench.py --kinds tts --only eleven --runs 3
    uv run python -X utf8 scripts/voice_lab/bench.py --skip gemini   # spare the Google quota

Keys come from the repo's `.env` (or `--env`), loaded the way `main.py` does.
Reports land in `scripts/voice_lab/out/<timestamp>/` (git-ignored).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import html
import json
import logging
import statistics
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parents[1]
REPO_DIR = AGENT_DIR.parent
OUT_DIR = HERE / "out"

sys.path.insert(0, str(AGENT_DIR))

PHRASE = (
    "Buenas tardes, señor. He revisado la agenda: tiene la reunión con el banco "
    "a las cinco y media, y el informe de mercados ya está listo."
)
LLM_PROMPT = "Responde en una sola frase corta, en castellano: ¿qué día va después del lunes?"
TIMEOUT = 45.0
MADRID = ZoneInfo("Europe/Madrid")


@dataclass
class Case:
    kind: str  # tts | stt | llm
    label: str
    overrides: dict[str, Any]
    expected: str  # module family the instance must come from


@dataclass
class Result:
    kind: str
    label: str
    status: str = "OK"  # OK | FALLBACK | FAIL | EMPTY
    via: str = "-"
    first_s: float | None = None  # TTFB / STT final latency / TTFT
    total_s: float | None = None
    audio_s: float | None = None
    accuracy: float | None = None
    sample: str = ""
    error: str = ""
    runs: list[dict[str, float]] = field(default_factory=list)


# -- Cases ---------------------------------------------------------------------


def tts_cases(extra_voices: list[tuple[str, str]]) -> list[Case]:
    cases = [
        Case(
            "tts",
            "openai tts-1 · onyx",
            dict(tts_provider="openai", tts_model="tts-1", tts_voice="onyx"),
            "openai",
        ),
        Case(
            "tts",
            "openai gpt-4o-mini-tts · ash",
            dict(tts_provider="openai", tts_model="gpt-4o-mini-tts", tts_voice="ash"),
            "openai",
        ),
        Case(
            "tts",
            "gemini tts · Charon",
            dict(tts_provider="gemini", tts_model="", tts_voice="Charon"),
            "google",
        ),
        Case(
            "tts",
            "gemini tts · Aoede",
            dict(tts_provider="gemini", tts_model="", tts_voice="Aoede"),
            "google",
        ),
        Case(
            "tts",
            "cartesia sonic-2 · default",
            dict(tts_provider="cartesia", tts_model="sonic-2", tts_voice=""),
            "cartesia",
        ),
        Case(
            "tts",
            "cartesia sonic-3 · default",
            dict(tts_provider="cartesia", tts_model="sonic-3", tts_voice=""),
            "cartesia",
        ),
        Case(
            "tts",
            "deepgram aura · default",
            dict(tts_provider="deepgram", tts_model="", tts_voice=""),
            "deepgram",
        ),
        Case(
            "tts",
            "deepgram aura-2 · nestor-es",
            dict(tts_provider="deepgram", tts_model="aura-2-nestor-es", tts_voice=""),
            "deepgram",
        ),
        Case(
            "tts",
            "deepgram aura-2 · celeste-es",
            dict(tts_provider="deepgram", tts_model="aura-2-celeste-es", tts_voice=""),
            "deepgram",
        ),
        Case(
            "tts",
            "elevenlabs flash v2.5 · George",
            dict(
                tts_provider="elevenlabs",
                tts_model="eleven_flash_v2_5",
                tts_voice="JBFqnCBsd6RMkjVDRZzb",
            ),
            "elevenlabs",
        ),
        Case(
            "tts",
            "elevenlabs turbo v2.5 · George",
            dict(
                tts_provider="elevenlabs",
                tts_model="eleven_turbo_v2_5",
                tts_voice="JBFqnCBsd6RMkjVDRZzb",
            ),
            "elevenlabs",
        ),
        Case(
            "tts",
            "elevenlabs multilingual v2 · George",
            dict(
                tts_provider="elevenlabs",
                tts_model="eleven_multilingual_v2",
                tts_voice="JBFqnCBsd6RMkjVDRZzb",
            ),
            "elevenlabs",
        ),
        Case(
            "tts",
            "rime arcana · luna (LK Inference)",
            dict(tts_provider="rime", tts_model="arcana", tts_voice="luna"),
            "inference",
        ),
        Case(
            "tts",
            "rime mistv2 · luna (LK Inference)",
            dict(tts_provider="rime", tts_model="mistv2", tts_voice="luna"),
            "inference",
        ),
    ]
    # Custom voices on the user's own ElevenLabs account (cloned or designed).
    for name, voice_id in extra_voices:
        cases.append(
            Case(
                "tts",
                f"elevenlabs flash v2.5 · {name} (propia)",
                dict(tts_provider="elevenlabs", tts_model="eleven_flash_v2_5", tts_voice=voice_id),
                "elevenlabs",
            )
        )
    return cases


STT_CASES = [
    Case(
        "stt",
        "deepgram nova-3",
        dict(stt_provider="deepgram", stt_model="nova-3", stt_language="es"),
        "deepgram",
    ),
    Case(
        "stt",
        "deepgram nova-2",
        dict(stt_provider="deepgram", stt_model="nova-2", stt_language="es"),
        "deepgram",
    ),
    Case(
        "stt",
        "openai whisper-1",
        dict(stt_provider="openai", stt_model="whisper-1", stt_language="es"),
        "openai",
    ),
    Case(
        "stt",
        "openai gpt-4o-mini-transcribe",
        dict(stt_provider="openai", stt_model="gpt-4o-mini-transcribe", stt_language="es"),
        "openai",
    ),
    Case(
        "stt",
        "gemini transcribe-live",
        dict(stt_provider="gemini", stt_model="", stt_language="es"),
        "google",
    ),
    Case(
        "stt",
        "cartesia ink-whisper",
        dict(stt_provider="cartesia", stt_model="ink-whisper", stt_language="es"),
        "cartesia",
    ),
    Case(
        "stt",
        "elevenlabs scribe v2 realtime",
        dict(stt_provider="elevenlabs", stt_model="", stt_language="es"),
        "elevenlabs",
    ),
    Case(
        "stt",
        "assemblyai universal-3.6-pro",
        dict(stt_provider="assemblyai", stt_model="", stt_language="es"),
        "assemblyai",
    ),
    Case(
        "stt",
        "assemblyai streaming multilingual",
        dict(
            stt_provider="assemblyai",
            stt_model="universal-streaming-multilingual",
            stt_language="es",
        ),
        "assemblyai",
    ),
]

LLM_CASES = [
    Case(
        "llm",
        "gemini-3.5-flash",
        dict(llm_provider="google", llm_model="gemini-3.5-flash"),
        "google",
    ),
    Case(
        "llm",
        "claude-haiku-5-5",
        dict(llm_provider="anthropic", llm_model="claude-haiku-5-5"),
        "anthropic",
    ),
    Case(
        "llm",
        "claude-sonnet-5-5",
        dict(llm_provider="anthropic", llm_model="claude-sonnet-5-5"),
        "anthropic",
    ),
    Case(
        "llm", "openai gpt-4o-mini", dict(llm_provider="openai", llm_model="gpt-4o-mini"), "openai"
    ),
    Case(
        "llm",
        "openai gpt-4.1-mini",
        dict(llm_provider="openai", llm_model="gpt-4.1-mini"),
        "openai",
    ),
    Case(
        "llm", "groq gpt-oss-20b", dict(llm_provider="groq", llm_model="openai/gpt-oss-20b"), "groq"
    ),
    Case(
        "llm",
        "groq gpt-oss-120b",
        dict(llm_provider="groq", llm_model="openai/gpt-oss-120b"),
        "groq",
    ),
    Case(
        "llm", "groq qwen3.8-27b", dict(llm_provider="groq", llm_model="qwen/qwen3.8-27b"), "groq"
    ),
    Case(
        "llm", "deepseek-chat", dict(llm_provider="deepseek", llm_model="deepseek-chat"), "openai"
    ),
]

_FAMILIES = (
    "inference",
    "openai",
    "google",
    "anthropic",
    "groq",
    "cartesia",
    "deepgram",
    "assemblyai",
    "elevenlabs",
)


def family(instance: Any) -> str:
    module = type(instance).__module__
    for part in _FAMILIES:
        if f".{part}" in module or module.startswith(part):
            return part
    return module


# -- Measurements ---------------------------------------------------------------


async def synthesize(tts: Any, text: str) -> tuple[float, float, list[Any]]:
    """(ttfb, total, frames) the way a session consumes this TTS."""
    frames: list[Any] = []
    first: float | None = None
    t0 = time.perf_counter()
    if tts.capabilities.streaming:
        stream = tts.stream()
        stream.push_text(text)
        stream.end_input()
        try:
            async for ev in stream:
                if ev.frame is None:
                    continue
                if first is None:
                    first = time.perf_counter()
                frames.append(ev.frame)
        finally:
            await stream.aclose()
    else:
        async with tts.synthesize(text) as stream:
            async for ev in stream:
                if first is None:
                    first = time.perf_counter()
                frames.append(ev.frame)
    end = time.perf_counter()
    return (first or end) - t0, end - t0, frames


def words(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return "".join(c if c.isalnum() else " " for c in plain).split()


def accuracy(heard: str) -> float:
    return difflib.SequenceMatcher(a=words(PHRASE), b=words(heard)).ratio()


def silence(rate: int, channels: int, seconds: float) -> Any:
    from livekit import rtc

    n = int(rate * seconds)
    return rtc.AudioFrame(b"\x00\x00" * n * channels, rate, channels, n)


async def transcribe(stt: Any, frames: list[Any]) -> tuple[float, str]:
    """(latency from end of speech to the completing final, transcript)."""
    if not frames:
        return float("nan"), ""
    from livekit.agents.stt import SpeechEventType

    if not stt.capabilities.streaming:
        t0 = time.perf_counter()
        ev = await stt.recognize(frames)
        text = ev.alternatives[0].text if ev.alternatives else ""
        return time.perf_counter() - t0, text

    stream = stt.stream()
    finals: list[str] = []
    end_of_speech: float | None = None
    completed: float | None = None
    rate, channels = frames[0].sample_rate, frames[0].num_channels

    async def feed() -> None:
        nonlocal end_of_speech
        for f in frames:
            stream.push_frame(f)
            await asyncio.sleep(f.samples_per_channel / f.sample_rate)
        end_of_speech = time.perf_counter()
        # Trailing silence, as a real microphone would send, so server-side
        # VAD and endpointing can close the utterance.
        for _ in range(40):
            stream.push_frame(silence(rate, channels, 0.1))
            await asyncio.sleep(0.1)

    async def read() -> None:
        nonlocal completed
        async for ev in stream:
            if ev.type == SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives:
                finals.append(ev.alternatives[0].text)
                if end_of_speech is not None and accuracy(" ".join(finals)) >= 0.8:
                    completed = time.perf_counter()
                    return

    feeder = asyncio.create_task(feed())
    try:
        await asyncio.wait_for(read(), TIMEOUT)
    except TimeoutError:
        pass
    finally:
        feeder.cancel()
        await stream.aclose()
    if completed is None or end_of_speech is None:
        return float("nan"), " ".join(finals)
    return completed - end_of_speech, " ".join(finals)


async def ask(llm: Any) -> tuple[float, float, str]:
    from livekit.agents.llm import ChatContext

    ctx = ChatContext.empty()
    ctx.add_message(role="user", content=LLM_PROMPT)
    text = ""
    first: float | None = None
    t0 = time.perf_counter()
    async with llm.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                if first is None:
                    first = time.perf_counter()
                text += chunk.delta.content
    end = time.perf_counter()
    return (first or end) - t0, end - t0, text


def median(values: list[float]) -> float | None:
    clean = [v for v in values if v == v]  # drop NaN
    return statistics.median(clean) if clean else None


# -- Runner ----------------------------------------------------------------------


async def run_case(case: Case, runs: int, out: Path, ref_frames: list[Any] | None) -> Result:
    from livekit import rtc

    from src.domain import KwamiVoiceConfig
    from src.factories.llm import create_llm
    from src.factories.stt import create_stt
    from src.factories.tts import create_tts

    result = Result(case.kind, case.label)
    factory = {"tts": create_tts, "stt": create_stt, "llm": create_llm}[case.kind]
    try:
        instance = factory(KwamiVoiceConfig(**case.overrides))
    except Exception as e:
        result.status, result.error = "FAIL", f"{type(e).__name__}: {e}"
        return result
    result.via = family(instance)
    if result.via != case.expected:
        result.status = "FALLBACK"
        result.error = f"pidió {case.expected}, construyó {result.via}"
        return result

    firsts: list[float] = []
    totals: list[float] = []
    for i in range(runs):
        try:
            if case.kind == "tts":
                ttfb, total, frames = await asyncio.wait_for(synthesize(instance, PHRASE), TIMEOUT)
                audio = sum(f.samples_per_channel / f.sample_rate for f in frames)
                firsts.append(ttfb)
                totals.append(total)
                result.audio_s = audio
                result.runs.append({"ttfb": ttfb, "total": total, "audio": audio})
                if i == 0 and frames:
                    wav = out / "tts" / f"{slug(case.label)}.wav"
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    wav.write_bytes(rtc.combine_audio_frames(frames).to_wav_bytes())
                    result.sample = wav.relative_to(out).as_posix()
            elif case.kind == "stt":
                latency, text = await transcribe(instance, ref_frames or [])
                firsts.append(latency)
                result.accuracy = accuracy(text)
                result.sample = text
                result.runs.append({"latency": latency, "accuracy": result.accuracy})
            else:
                ttft, total, text = await asyncio.wait_for(ask(instance), TIMEOUT)
                firsts.append(ttft)
                totals.append(total)
                result.sample = text.strip()
                result.runs.append({"ttft": ttft, "total": total})
        except Exception as e:
            result.status, result.error = "FAIL", f"{type(e).__name__}: {str(e)[:300]}"
            break

    if result.status == "OK":
        result.first_s = median(firsts)
        result.total_s = median(totals)
        if result.first_s is None or (case.kind != "tts" and not result.sample):
            result.status = "EMPTY"
    aclose = getattr(instance, "aclose", None)
    if aclose:
        with contextlib.suppress(Exception):
            await aclose()
    return result


def slug(text: str) -> str:
    return "-".join(words(text))


def fmt(v: float | None, unit: str = "s") -> str:
    return "—" if v is None else f"{v:.2f} {unit}" if unit else f"{v:.2f}"


def line(r: Result) -> str:
    extra = r.error or r.sample
    acc = f" acc={r.accuracy:.0%}" if r.accuracy is not None else ""
    return f"{r.kind.upper():3} {r.label:42} {r.status:8} via={r.via:10} first={fmt(r.first_s):9}{acc}  {extra[:80]}"


# -- Report ----------------------------------------------------------------------


def ranked(results: list[Result], kind: str) -> list[Result]:
    ok = [r for r in results if r.kind == kind and r.status == "OK" and r.first_s is not None]
    rest = [r for r in results if r.kind == kind and r not in ok]
    if kind == "stt":
        # Accurate enough first, then fastest: a fast transcript that is wrong
        # costs more than the latency it saves.
        ok.sort(key=lambda r: ((r.accuracy or 0) < 0.8, r.first_s))
    else:
        ok.sort(key=lambda r: r.first_s)
    return ok + rest


def combos(results: list[Result]) -> list[tuple[float, Result, Result, Result]]:
    top = {
        k: [r for r in ranked(results, k) if r.status == "OK"][:3] for k in ("stt", "llm", "tts")
    }
    stt_ok = [r for r in top["stt"] if (r.accuracy or 0) >= 0.8] or top["stt"]
    rows = [
        (s.first_s + m.first_s + t.first_s, s, m, t)  # type: ignore[operator]
        for s in stt_ok
        for m in top["llm"]
        for t in top["tts"]
    ]
    return sorted(rows, key=lambda row: row[0])[:6]


def write_report(results: list[Result], out: Path, args: argparse.Namespace) -> Path:
    def table(kind: str, head: list[str], cells: Any) -> str:
        rows = "".join(
            f"<tr class='{r.status.lower()}'>"
            + "".join(f"<td>{c}</td>" for c in cells(r))
            + "</tr>"
            for r in ranked(results, kind)
        )
        if not rows:
            return ""
        th = "".join(f"<th>{h}</th>" for h in head)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table>"

    def audio(r: Result) -> str:
        return (
            f"<audio controls preload='none' src='{html.escape(r.sample)}'></audio>"
            if r.sample
            else ""
        )

    def note(r: Result) -> str:
        return html.escape(r.error or "")

    tts = table(
        "tts",
        ["Voz", "Estado", "Primer audio", "Total", "Audio", "Escuchar", "Nota"],
        lambda r: [
            html.escape(r.label),
            r.status,
            fmt(r.first_s),
            fmt(r.total_s),
            fmt(r.audio_s),
            audio(r),
            note(r),
        ],
    )
    stt = table(
        "stt",
        ["Oído", "Estado", "Latencia final", "Precisión", "Transcripción", "Nota"],
        lambda r: [
            html.escape(r.label),
            r.status,
            fmt(r.first_s),
            "—" if r.accuracy is None else f"{r.accuracy:.0%}",
            html.escape(r.sample),
            note(r),
        ],
    )
    llm = table(
        "llm",
        ["Modelo", "Estado", "Primer token", "Respuesta completa", "Respuesta", "Nota"],
        lambda r: [
            html.escape(r.label),
            r.status,
            fmt(r.first_s),
            fmt(r.total_s),
            html.escape(r.sample),
            note(r),
        ],
    )
    best = "".join(
        f"<tr><td>{fmt(total)}</td><td>{html.escape(s.label)}</td><td>{html.escape(m.label)}</td><td>{html.escape(t.label)}</td></tr>"
        for total, s, m, t in combos(results)
    )
    combo = (
        "<table><thead><tr><th>Estimado</th><th>Oído</th><th>Cerebro de voz</th><th>Voz</th></tr></thead>"
        f"<tbody>{best}</tbody></table>"
        if best
        else ""
    )
    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Laboratorio de voz</title>
<style>
:root {{ --bg:#fff; --fg:#1c1c1c; --muted:#6b6b6b; --line:#e4e4e4; --ok:#1d7a3a; --bad:#b3261e; --warn:#9a6700; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141414; --fg:#ececec; --muted:#9a9a9a; --line:#2c2c2c; --ok:#5cc27a; --bad:#ff8a80; --warn:#e3b341; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:24px 16px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:32px 0 8px; }}
p.meta {{ color:var(--muted); margin:0 0 16px; }}
.wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th, td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }}
th {{ color:var(--muted); font-weight:600; }} tr.fail td, tr.empty td {{ color:var(--bad); }} tr.fallback td {{ color:var(--warn); }}
tbody tr:first-child.ok td:first-child {{ color:var(--ok); font-weight:600; }} audio {{ height:32px; max-width:220px; }}
</style></head><body>
<h1>Laboratorio de voz</h1>
<p class="meta">{datetime.now(MADRID):%d-%m-%Y %H:%M} · {args.runs} repetición(es), mediana · frase: «{html.escape(PHRASE)}»</p>
<h2>Combinaciones más rápidas (estimación oído + primer token + primer audio)</h2><div class="wrap">{combo}</div>
<h2>Voz (TTS)</h2><div class="wrap">{tts}</div>
<h2>Oído (STT)</h2><div class="wrap">{stt}</div>
<h2>Cerebro de voz (LLM)</h2><div class="wrap">{llm}</div>
</body></html>"""
    path = out / "report.html"
    path.write_text(page, encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


# -- Main ----------------------------------------------------------------------


def own_elevenlabs_voices() -> list[tuple[str, str]]:
    """Cloned or designed voices on the account, which premade lists never show."""
    import httpx

    from src.factories.tts import elevenlabs_key

    key = elevenlabs_key()
    if not key:
        return []
    try:
        resp = httpx.get(
            "https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key}, timeout=15
        )
        resp.raise_for_status()
    except Exception:
        return []
    return [
        (v["name"], v["voice_id"])
        for v in resp.json().get("voices", [])
        if v.get("category") in ("cloned", "generated", "professional")
    ]


def selected(cases: list[Case], args: argparse.Namespace) -> list[Case]:
    def match(case: Case, needles: list[str]) -> bool:
        return any(n.lower() in case.label.lower() for n in needles)

    if args.only:
        cases = [c for c in cases if match(c, args.only)]
    if args.skip:
        cases = [c for c in cases if not match(c, args.skip)]
    return cases


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kinds", default="tts,stt,llm", help="tts,stt,llm (comma separated)")
    parser.add_argument("--runs", type=int, default=2, help="repetitions per case (median)")
    parser.add_argument(
        "--only", action="append", default=[], help="keep cases whose label contains this"
    )
    parser.add_argument(
        "--skip", action="append", default=[], help="drop cases whose label contains this"
    )
    parser.add_argument(
        "--env", type=Path, default=REPO_DIR / ".env", help=".env with the provider keys"
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(args.env, override=True)
    logging.basicConfig(level=logging.ERROR)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    from livekit.agents.utils import http_context

    from src.domain import KwamiVoiceConfig
    from src.factories.tts import create_tts

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    out = OUT_DIR / datetime.now(MADRID).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []

    async with http_context.open():
        groups: list[Case] = []
        if "tts" in kinds:
            groups += selected(tts_cases(own_elevenlabs_voices()), args)
        if "stt" in kinds:
            groups += selected(STT_CASES, args)
        if "llm" in kinds:
            groups += selected(LLM_CASES, args)

        ref_frames = None
        if any(c.kind == "stt" for c in groups):
            reference = create_tts(
                KwamiVoiceConfig(tts_provider="openai", tts_model="tts-1", tts_voice="onyx")
            )
            _, _, ref_frames = await synthesize(reference, PHRASE)
            print(
                f"(audio de referencia para el oído: OpenAI tts-1 · onyx, {len(ref_frames)} tramas)"
            )

        for case in groups:
            result = await run_case(case, args.runs, out, ref_frames)
            results.append(result)
            print(line(result), flush=True)

    report = write_report(results, out, args)
    print(f"\nInforme: {report}")


if __name__ == "__main__":
    asyncio.run(main())
