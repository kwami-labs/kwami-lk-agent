"""Routing for the LiveKit data channel.

This lived as a 49-line closure inside the 130-line `entrypoint`, which meant
no part of it could be exercised without standing up a worker, a room and a
session -- so none of it ever was. Pulling it out makes each branch reachable
from a test with plain fakes.

Decoding is separated from routing on purpose: the payload is attacker-shaped
input from the frontend, and a malformed packet must never take down the data
handler for the rest of the session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..handlers import handle_config_update, handle_full_config, handle_tool_result
from ..utils.logging import get_logger

logger = get_logger("dispatch")

# A "find similar" query is built from a client-supplied title.
MAX_SIMILAR_TITLE_CHARS = 80
SIMILAR_SEARCH_RESULTS = 5


def decode_data_message(raw: Any) -> dict[str, Any] | None:
    """Decode a data-channel payload, or None if it is not a usable message.

    Never raises: bad UTF-8, bad JSON and non-object payloads all return None.
    """
    try:
        payload = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else raw
        message = json.loads(payload)
    except (UnicodeDecodeError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Discarding undecodable data message: %s", e)
        return None

    if not isinstance(message, dict):
        logger.warning("Discarding data message that is not an object: %s", type(message).__name__)
        return None
    return message


def route_metrics(usage_tracker: Any, metrics: Any) -> bool:
    """Send a metrics event to the matching tracker method.

    Returns True when the event was recognised, which is what lets a test tell
    "routed correctly" from "silently dropped".
    """
    handlers = {
        "llm_metrics": "on_llm_metrics",
        "stt_metrics": "on_stt_metrics",
        "tts_metrics": "on_tts_metrics",
        "realtime_model_metrics": "on_realtime_metrics",
    }
    metric_type = getattr(metrics, "type", None)
    if not isinstance(metric_type, str):
        return False
    method_name = handlers.get(metric_type)
    if method_name is None:
        return False
    getattr(usage_tracker, method_name)(metrics)
    return True


@dataclass
class DataMessageRouter:
    """Dispatches decoded data-channel messages to the right handler.

    Config work is spawned through `state.spawn` and serialized through
    `state.run_serialized`: two config messages arriving together used to race
    on `update_agent`, and a bare `create_task` could be collected mid-flight.
    """

    session: Any
    state: Any
    vad: Any = None
    create_agent_fn: Any = None
    room: Any = None

    def handle(self, message: dict[str, Any]) -> str | None:
        """Route one message. Returns the handled type, or None if unrecognised."""
        msg_type = message.get("type")
        logger.info("Received data message: %s", msg_type)
        if not isinstance(msg_type, str):
            logger.warning("Discarding data message with non-string type: %r", msg_type)
            return None

        handler = {
            "config": self._on_config,
            "config_update": self._on_config_update,
            "tool_result": self._on_tool_result,
            "browser_close_request": self._on_browser_close,
            "search_similar": self._on_search_similar,
        }.get(msg_type)

        if handler is None:
            logger.debug("No handler for data message type %r", msg_type)
            return None

        handler(message)
        return msg_type

    # -- individual routes --------------------------------------------------

    def _on_config(self, message: dict[str, Any]) -> None:
        self.state.spawn(
            self.state.run_serialized(
                handle_full_config(
                    self.session, self.state, message, self.vad, self.create_agent_fn
                )
            ),
            name="handle_full_config",
        )

    def _on_config_update(self, message: dict[str, Any]) -> None:
        self.state.spawn(
            self.state.run_serialized(
                handle_config_update(
                    self.session, self.state, message, self.vad, self.create_agent_fn
                )
            ),
            name="handle_config_update",
        )

    def _on_tool_result(self, message: dict[str, Any]) -> None:
        handle_tool_result(
            self.state.current_agent,
            message.get("toolCallId"),
            message.get("result"),
            message.get("error"),
        )

    def _on_browser_close(self, message: dict[str, Any]) -> None:
        browser_session = getattr(self.state, "active_browser_session", None)
        if browser_session is not None:
            self.state.spawn(browser_session.close(), name="browser_close_request")
            logger.info("Closing cloud browser per user request")

    def _on_search_similar(self, message: dict[str, Any]) -> None:
        agent = self.state.current_agent
        if agent is None:
            return
        title = (message.get("title") or "").strip() or "similar products"
        query = f"similar to {title[:MAX_SIMILAR_TITLE_CHARS]} buy"
        logger.info("Running similar search from client: query=%s", query[:60])
        run_context = type("Ctx", (), {"room": self.room})()
        self.state.spawn(
            agent.web_search(
                run_context, query, max_results=SIMILAR_SEARCH_RESULTS, search_for_products=True
            ),
            name="search_similar",
        )
