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
            "browser_open_request": self._on_browser_open,
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

    def _on_browser_open(self, message: dict[str, Any]) -> None:
        """Open a URL the frontend asked for, through the agent's own gate.

        The app publishes this when the user picks a search result. It had no
        route at all: the message was decoded, matched nothing, and was dropped
        at the `debug` line above -- so clicking a result did nothing, silently.

        Deliberately goes through `navigate_to` rather than reaching for the
        browser session directly. That is the only path with the two checks
        this needs, and duplicating them here would mean two gates to keep in
        step:

        * `validate_url_async` rejects loopback, RFC1918, link-local metadata
          and anything that resolves to them. Necessary even though the app
          sent it: the same envelope carries a URL the model was reading on a
          page a moment earlier, and "the frontend published it" is not the
          same as "a human asked for it".
        * The blank-`kwami_id` refusal. A browser profile holds the user's
          cookies and logins, so one started outside that check could hand one
          user's authenticated sessions to the next. This message can arrive
          before any browser exists, which is exactly when that matters.
        """
        agent = self.state.current_agent
        if agent is None:
            logger.warning("Ignoring browser_open_request: no agent yet")
            return

        url = message.get("url")
        if not isinstance(url, str) or not url.strip():
            logger.warning("Ignoring browser_open_request with no usable url")
            return

        logger.info("Opening %s in the browser panel per user request", url[:80])
        run_context = type("Ctx", (), {"room": self.room})()
        self.state.spawn(
            self._open_and_report(agent, run_context, url.strip()),
            name="browser_open_request",
        )

    @staticmethod
    async def _open_and_report(agent: Any, run_context: Any, url: str) -> None:
        """Navigate, and make a refusal visible rather than silent.

        `navigate_to` answers a rejected URL with a sentence for the model. On
        this path there is no model turn to return it to, so a refusal would
        otherwise look identical to success from outside -- which is the failure
        this whole handler exists to stop happening again.
        """
        result = await agent.navigate_to(run_context, url)
        if isinstance(result, str) and result.startswith(("I can't", "Cannot", "Failed")):
            logger.warning("browser_open_request refused for %s: %s", url[:80], result)

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
