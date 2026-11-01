"""Bootstrap telephony sessions from backend-provided kwami config.

The config comes from the Kwami API's internal route, or -- when the agent has
no API and writes Supabase directly -- from the kwami's `user_kwamis` row.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .adapters.http import shared_client
from .adapters.supabase import SupabaseClient, as_user_id, direct_supabase
from .settings import get_settings
from .utils.logging import get_logger

logger = get_logger("runtime_bootstrap")


def _api_timeout_seconds() -> float:
    """Never below a second, however the environment is configured."""
    return max(1.0, get_settings().kwami_api_timeout)


def _parse_json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def resolve_kwami_id(ctx) -> str | None:
    """Resolve the kwami id from job metadata or SIP participant metadata/attributes."""
    metadata = _parse_json_dict(getattr(ctx.job, "metadata", None))
    if metadata.get("kwami_id"):
        return str(metadata["kwami_id"])

    for participant in ctx.room.remote_participants.values():
        raw_metadata = _parse_json_dict(getattr(participant, "metadata", None))
        if raw_metadata.get("kwami_id"):
            return str(raw_metadata["kwami_id"])

        attributes = getattr(participant, "attributes", None) or {}
        if isinstance(attributes, dict) and attributes.get("kwami_id"):
            return str(attributes["kwami_id"])

    return None


#: The ear the app shipped with before Deepgram Nova 3, upgraded the way the
#: app upgrades it on load: it finalised 1.24 s after the end of speech.
LEGACY_DEFAULT_STT = ("gemini", "gemini-3.5-transcribe-live")
FAST_STT = {"provider": "deepgram", "model": "nova-3"}


def _engine_fields(voice_state: dict[str, Any], voice: dict[str, Any]) -> dict[str, Any]:
    """Add what makes a call sound like the app: engine, realtime voice, direction.

    The payload carried only STT, LLM and TTS, so a kwami on Gemini Live
    answered calls on the standard pipeline, and every call lost the
    character's voice direction (`style`) -- Jarvis on the phone was not Jarvis.
    """
    tts_state = voice_state.get("tts") or {}
    style = tts_state.get("style")
    if isinstance(style, str) and style.strip():
        voice["tts"]["style"] = style
    stt = voice["stt"]
    if (stt.get("provider"), stt.get("model")) == LEGACY_DEFAULT_STT:
        stt.update(FAST_STT)
    pipeline = voice_state.get("pipelineMode")
    if isinstance(pipeline, str) and pipeline:
        voice["type"] = pipeline
    realtime = voice_state.get("realtime")
    if isinstance(realtime, dict) and realtime.get("provider"):
        voice["realtime"] = {
            key: realtime[key]
            for key in ("provider", "model", "voice", "style")
            if isinstance(realtime.get(key), str) and realtime.get(key)
        }
    return voice


def build_bootstrap_payload(kwami_row: dict[str, Any]) -> dict[str, Any]:
    """The `config` message for a kwami, from its `user_kwamis` row.

    A port of kwami-lk-api `build_agent_bootstrap_payload`
    (src/services/channels.py), so a call bootstrapped from Supabase sounds the
    same as one bootstrapped through the API. Keep the two in step.
    """
    config = kwami_row.get("config") or {}
    voice_state = config.get("voice") or {}
    soul_state = voice_state.get("soulConfig") or {}
    stt = voice_state.get("stt") or {}
    llm = voice_state.get("llm") or {}
    tts = voice_state.get("tts") or {}

    return {
        "type": "config",
        "kwamiId": kwami_row["id"],
        "kwamiName": kwami_row.get("name") or "Kwami",
        "voice": _engine_fields(
            voice_state,
            {
                "stt": {
                    "provider": stt.get("provider") or "deepgram",
                    "model": stt.get("model") or "nova-2-phonecall",
                    "language": stt.get("language") or "en",
                },
                "llm": {
                    "provider": llm.get("provider") or "openai",
                    "model": llm.get("model") or "gpt-4o-mini",
                    "temperature": llm.get("temperature") or 0.7,
                    "maxTokens": llm.get("maxTokens") or 1024,
                },
                "tts": {
                    "provider": tts.get("provider") or "openai",
                    "model": tts.get("model") or "tts-1",
                    "voice": tts.get("voice") or "nova",
                    "speed": tts.get("speed") or 1.0,
                },
            },
        ),
        "soul": {
            "name": soul_state.get("name") or kwami_row.get("name") or "Kwami",
            "personality": soul_state.get("personality") or "",
            "systemPrompt": soul_state.get("systemPrompt") or "",
            "traits": soul_state.get("traits") or [],
            "conversationStyle": soul_state.get("conversationStyle") or "friendly",
            "responseLength": soul_state.get("responseLength") or "medium",
            "emotionalTone": soul_state.get("emotionalTone") or "warm",
            **{
                key: soul_state[key]
                for key, kind in (
                    ("language", str),
                    ("emotionalTraits", dict),
                    ("charter", dict),
                )
                if isinstance(soul_state.get(key), kind) and soul_state.get(key)
            },
        },
        "memory": {"enabled": True},
    }


async def _fetch_from_supabase(store: SupabaseClient, kwami_id: str) -> dict[str, Any] | None:
    if as_user_id(kwami_id) is None:
        logger.warning("kwami_id %r is not a user_kwamis id; no runtime config", kwami_id)
        return None
    rows = await store.select(
        "user_kwamis",
        {"select": "id,user_id,name,config", "id": f"eq.{kwami_id}", "limit": "1"},
        what="the kwami's runtime config",
    )
    if not rows:
        if rows is not None:
            logger.warning("Kwami %s not found in Supabase", kwami_id)
        return None
    return build_bootstrap_payload(rows[0])


async def fetch_runtime_config(kwami_id: str) -> dict[str, Any] | None:
    # On LiveKit Cloud there is no API: read the kwami's row directly.
    if store := direct_supabase():
        return await _fetch_from_supabase(store, kwami_id)

    if not get_settings().kwami_api_key:
        logger.warning("get_settings().kwami_api_key not set; telephony bootstrap is disabled")
        return None

    url = f"{get_settings().kwami_api_url.rstrip('/')}/internal/kwamis/{kwami_id}/runtime"
    timeout = _api_timeout_seconds()
    try:
        client = shared_client()
        response = await client.get(
            url,
            headers={"X-Kwami-API-Key": get_settings().kwami_api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except httpx.HTTPStatusError as exc:
        body = (exc.response.text[:300] + "...") if exc.response and exc.response.text else ""
        logger.warning(
            "Kwami API returned %s for runtime config: %s",
            exc.response.status_code if exc.response else "?",
            body,
        )
        return None
    except httpx.RequestError as exc:
        logger.warning(
            "Kwami API unreachable for runtime config (%s: %s). "
            "KWAMI_API_URL=%r — if the agent runs in Docker/Kubernetes, localhost is the container, not your laptop; "
            "use the host LAN IP, host.docker.internal, or a public/tunnel URL.",
            type(exc).__name__,
            exc,
            get_settings().kwami_api_url,
        )
        return None
