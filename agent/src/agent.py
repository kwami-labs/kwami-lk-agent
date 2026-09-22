"""Kwami Agent - Dynamic AI agent configured by the Kwami frontend library."""

import asyncio
from typing import Any

from livekit.agents import Agent
from livekit.agents.voice.agent import find_function_tools

from .constants import Timeouts
from .domain import KwamiConfig, build_system_prompt
from .domain.greeting import (
    MAX_CONTEXT_SUMMARY_CHARS,
    GreetingFacts,
    build_greeting_instructions,
    extract_name_from_facts,
    topics_from_facts,
)
from .domain.tool_gating import describe_gating, unavailable_builtin_tools
from .memory import KwamiMemory
from .runtime.container import room_from_context
from .settings import get_settings
from .tools import (
    AgentToolsMixin,
    ClientToolManager,
    KnowledgeToolsMixin,
    MediaToolsMixin,
    PipelineControlMixin,
    TradingToolsMixin,
)
from .tools.limits import trim_client_tools
from .utils.logging import get_logger, redacted
from .utils.room import should_disconnect_as_duplicate

logger = get_logger("agent")


def _tool_name(tool: Any) -> str:
    """The registered name of a function tool, or "" if it has none."""
    info = getattr(tool, "info", None)
    name = getattr(info, "name", None) if info is not None else None
    return name if isinstance(name, str) else ""


class KwamiAgent(
    Agent,
    AgentToolsMixin,
    PipelineControlMixin,
    KnowledgeToolsMixin,
    MediaToolsMixin,
    TradingToolsMixin,
):
    """Dynamic AI agent configured by the Kwami frontend library.

    This agent supports:
    - Configurable voice pipeline (STT, LLM, TTS)
    - Persistent memory via Zep Cloud
    - Client-side tools executed via data channel
    - Built-in tools for voice/language control
    - Self-service model, voice and pipeline switching (PipelineControlMixin)
    - Multi-angle research and market data (KnowledgeToolsMixin)
    - Music and video playback in the browser panel (MediaToolsMixin)
    - Confirmation-gated order placement (TradingToolsMixin)
    - Dynamic reconfiguration without disconnection
    """

    def __init__(
        self,
        config: KwamiConfig | None = None,
        vad: Any = None,
        memory: KwamiMemory | None = None,
        stt: Any = None,
        llm: Any = None,
        tts: Any = None,
        skip_greeting: bool = False,
    ):
        """Initialize the Kwami agent.

        Args:
            config: Kwami configuration with soul, voice settings, etc.
            vad: Voice Activity Detection instance.
            memory: Optional Zep memory instance for persistent context.
            stt: Speech-to-Text instance.
            llm: Large Language Model instance.
            tts: Text-to-Speech instance.
            skip_greeting: If True, skip the initial greeting (for reconfigurations).
        """
        self.kwami_config = config or KwamiConfig()
        self._vad = vad
        self._memory = memory
        self._skip_greeting = skip_greeting
        # Cached context from _inject_memory_context.
        self._last_memory_context: Any = None

        # Track current voice config for switching
        self._current_voice_config = self.kwami_config.voice
        self.room = None  # Set by main.py/session.py; re-resolved in on_enter
        self.usage_tracker = None
        self._browser_session = None  # Cloud browser session (lazy-created by navigate_to)
        self._session_listeners_registered = False
        # Strong references to fire-and-forget tasks. The event loop only holds
        # a weak reference, so a bare create_task can be collected mid-flight.
        self._background_tasks: set[asyncio.Task] = set()

        # Initialize client tool manager
        self.client_tools = ClientToolManager(self)
        if self.kwami_config.tools:
            self.client_tools.register_client_tools(self.kwami_config.tools)

        # Build system prompt
        instructions = self._build_system_prompt()

        # Only the client tools go to the parent: the framework builds
        # `Agent._tools` as `tools + find_function_tools(self)`, so passing the
        # built-ins here would double them. They are counted, not passed,
        # because the provider's 128-tool ceiling applies to the sum -- and one
        # tool over it is a 400 on every turn, not a degraded feature.
        builtin_count = len(find_function_tools(type(self)))
        self._tools = trim_client_tools(builtin_count, self.client_tools.create_client_tools())

        super().__init__(
            instructions=instructions,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            tools=self._tools,
        )

        # The framework sets `_tools = tools + find_function_tools(self)`, so
        # every built-in is now attached whether or not this deployment can
        # serve it. Withhold the ones whose credential is absent: offering
        # `product_search` with no SERPAPI_KEY spends the user's turn on a
        # refusal the agent advertised, and forty schemas go out on every
        # request of every session whether they work or not.
        self._withhold_unavailable_builtins()

    def _withhold_unavailable_builtins(self) -> None:
        """Drop built-in tools this deployment has no credential for.

        Reassigns `_tools` and refreshes `_chat_ctx`, which is exactly what the
        framework's own `update_tools` does when there is no activity yet --
        and there never is during `__init__`. Doing it here rather than awaiting
        `update_tools` later keeps the very first request correct; a tool
        withheld only after the greeting would already have been advertised.
        """
        unavailable = unavailable_builtin_tools(get_settings())
        if not unavailable:
            return

        # No `if nothing was dropped` guard: every gated name is a real built-in
        # -- pinned by test_the_gated_names_are_real_tools -- so a non-empty
        # `unavailable` always matches at least one tool here. A guard for the
        # other case would be a branch no test could reach.
        kept = [tool for tool in self._tools if _tool_name(tool) not in unavailable]
        self._tools = kept
        self._chat_ctx = self._chat_ctx.copy(tools=kept)
        logger.info(
            "%s",
            describe_gating({_tool_name(t) for t in find_function_tools(type(self))} & unavailable),
        )

    def _build_system_prompt(self, memory_context: str | None = None) -> str:
        """Build the system prompt from soul configuration and memory context.

        Args:
            memory_context: Optional memory context to inject into the prompt.

        Returns:
            Complete system prompt string.
        """
        # Pass the client tools the frontend actually registered, so guidance
        # naming set_ui_control / list_ui_controls is only emitted when those
        # tools exist. Otherwise the model is invited to hallucinate calls.
        return build_system_prompt(
            self.kwami_config.soul,
            memory_context,
            client_tool_names=self._registered_client_tool_names(),
        )

    def _registered_client_tool_names(self) -> set[str]:
        """Names of the client-side tools currently registered on this agent."""
        manager = getattr(self, "client_tools", None)
        registered = getattr(manager, "registered_tools", None) or []
        names: set[str] = set()
        for entry in registered:
            name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
            if isinstance(name, str) and name:
                names.add(name)
        return names

    async def _inject_memory_context(self) -> None:
        """Fetch memory context, cache user name, and update system prompt.

        Also pre-caches the user name so subsequent messages include
        proper attribution in the knowledge graph.
        """
        if not self._memory or not self._memory.is_initialized:
            return

        try:
            # These two are independent Zep round-trips and both sit on the
            # critical path before the first utterance, so run them together
            # rather than back to back.
            user_name, context = await asyncio.gather(
                self._memory.get_user_name(),
                self._memory.get_context(),
            )
            if user_name:
                logger.info("Cached user name from memory: %s", redacted(user_name, keep=1))

            memory_text = context.to_system_prompt_addition()

            if memory_text:
                new_instructions = self._build_system_prompt(memory_text)
                await self.update_instructions(new_instructions)
                logger.info("Injected memory context into system prompt")

            # Store context for greeting use (avoids a second API call)
            self._last_memory_context = context
        except Exception:
            logger.exception("Failed to inject memory context")

    async def on_enter(self) -> None:
        """Called when the agent joins the room.

        The framework dispatches this with no arguments (see
        ``livekit/agents/voice/agent_activity.py``). The previous signature
        accepted a ``room`` parameter, which was therefore always ``None``:
        the duplicate-agent guard below never ran, and ``self.room = room``
        overwrote the reference that ``main.py`` and ``session.py`` had just
        assigned, leaving every tool dependent on the ContextVar fallback.
        """
        # main.py / session.py set self.room; the ContextVar is the fallback
        # for agents constructed before the room was wired up.
        room = self.room or room_from_context(None)
        my_identity = ""
        if room:
            self.room = room
            my_identity = room.local_participant.identity if room.local_participant else ""
            logger.info("Agent %s entering room...", my_identity)

            # Quick check for duplicate agents (non-blocking)
            should_disconnect = await should_disconnect_as_duplicate(room, my_identity)
            if should_disconnect:
                logger.warning("Agent %s disconnecting due to duplicate detection", my_identity)
                await room.disconnect()
                return
        else:
            logger.warning("Agent entered with no room reference available")

        self._register_session_listeners()

        logger.info(
            "Kwami agent '%s' (%s) entered room successfully",
            self.kwami_config.kwami_name,
            self.kwami_config.kwami_id,
        )

        # Inject memory context into system prompt.
        # Hard-bounded: memory is an enhancement, but the greeting is the
        # product. A slow or unreachable Zep used to hold the first utterance
        # for as long as it took, so cap it and greet without context on
        # timeout rather than leaving the caller listening to silence.
        try:
            await asyncio.wait_for(
                self._inject_memory_context(),
                timeout=Timeouts.MEMORY_CONTEXT,
            )
        except TimeoutError:
            logger.warning(
                "Memory context timed out after %.1fs; greeting without it.",
                Timeouts.MEMORY_CONTEXT,
            )
        except Exception:
            logger.exception("Memory context failed; greeting without it.")

        # Greet the user - but only once per session
        if self._skip_greeting:
            logger.debug("Skipping greeting (agent was reconfigured)")
            return

        # Generate a natural, personalized greeting
        try:
            logger.info("Generating greeting for user...")
            greeting_instructions = await self._build_greeting_instructions()
            self.session.generate_reply(
                instructions=greeting_instructions,
                allow_interruptions=True,
            )
        except Exception:
            logger.exception("Failed to generate greeting")
            # Fall back to a simple greeting so the agent still speaks
            try:
                self.session.generate_reply(
                    instructions="Greet the user casually and ask how you can help.",
                    allow_interruptions=True,
                )
            except Exception:
                logger.exception("Failed to generate fallback greeting")

    async def _build_greeting_instructions(self) -> str:
        """Gather what we know about the user, then hand it to the domain layer.

        This method used to be 110 lines that did both: the memory lookups *and*
        the four branches of wording. Only the lookups need an agent, so the
        wording moved to `domain/greeting.py` where every branch is reachable
        from a plain unit test -- see the module docstring there.

        Reuses the context `_inject_memory_context` already fetched, so the
        greeting does not pay a second Zep round trip on the critical path.
        """
        facts = await self._gather_greeting_facts()
        return build_greeting_instructions(facts)

    async def _gather_greeting_facts(self) -> GreetingFacts:
        """The I/O half: everything the greeting needs, fetched."""
        agent_name = self.kwami_config.soul.name or self.kwami_config.kwami_name or "Kwami"
        language = getattr(self.kwami_config.soul, "language", None)

        if not (self._memory and self._memory.is_initialized):
            return GreetingFacts(agent_name=agent_name, language=language)

        try:
            user_name = self._memory._cached_user_name
            context = self._last_memory_context
            if context is None:
                context = await self._memory.get_context()

            is_returning = bool(context.recent_messages or context.facts or context.context_block)
            summary = None
            topics: list[str] = []
            if is_returning:
                logger.debug(
                    "Returning user detected (messages: %s, facts: %s)",
                    len(context.recent_messages),
                    len(context.facts),
                )
                if context.context_block:
                    summary = context.context_block[:MAX_CONTEXT_SUMMARY_CHARS]
                elif context.summary:
                    summary = context.summary
                topics = topics_from_facts(context.facts)

            if not user_name and context.facts:
                found = extract_name_from_facts(context.facts, agent_name=agent_name)
                if found:
                    user_name = found
                    self._memory.set_user_name(found)
                    logger.info("Found user name from facts: %s", redacted(found, keep=1))

            return GreetingFacts(
                agent_name=agent_name,
                user_name=user_name,
                is_returning_user=is_returning,
                recent_context_summary=summary,
                recent_topics=topics,
                language=language,
            )
        except Exception as e:
            # Memory is an enhancement; the greeting is the product. A Zep
            # failure here means greeting as a stranger, not not greeting.
            logger.warning("Could not extract user info from memory: %s", e)
            return GreetingFacts(agent_name=agent_name, language=language)

    async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
        """Called when user finishes speaking.

        Buffers the user message so it can be sent together with the
        assistant's response in a single add_messages call. This produces
        much better knowledge graph construction in Zep.

        Args:
            turn_ctx: Turn context.
            new_message: The user's message.
        """
        if self._memory and self._memory.is_initialized and new_message:
            try:
                content = self._extract_message_content(new_message)
                if content:
                    # Get user name for better graph attribution
                    user_name = self._memory._cached_user_name or None
                    await self._memory.buffer_user_message(content, name=user_name)
            except Exception as e:
                logger.warning("Failed to buffer user message: %s", e)

    def _register_session_listeners(self) -> None:
        """Subscribe to the session events this agent depends on.

        Replaces ``on_agent_turn_completed``, which livekit-agents never
        dispatched -- it was the only caller of ``KwamiMemory.add_exchange``,
        so assistant turns were never written to Zep at all.
        ``conversation_item_added`` is the framework's real notification that
        a turn has been committed to the chat context.
        """
        if self._session_listeners_registered:
            return
        try:
            session = self.session
        except RuntimeError:
            # No activity yet; on_enter will call us again once there is one.
            return
        session.on("conversation_item_added", self._on_conversation_item_added)
        self._session_listeners_registered = True

    def _on_conversation_item_added(self, event: Any) -> None:
        """Persist assistant turns to memory as the session commits them.

        The emitter is synchronous, so the Zep write is handed to a task whose
        reference is retained -- a bare ``create_task`` can be garbage
        collected mid-flight and swallows its own exceptions.
        """
        item = getattr(event, "item", None)
        if item is None or getattr(item, "role", None) != "assistant":
            return
        if not (self._memory and self._memory.is_initialized):
            return
        content = self._extract_message_content(item)
        if not content:
            return

        task = asyncio.create_task(self._persist_assistant_turn(content))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _persist_assistant_turn(self, content: str) -> None:
        """Send the buffered user message plus this reply to Zep as one batch.

        ``add_exchange`` passes ``ignore_roles=["assistant"]`` so only user
        messages create graph entities while the reply still provides context
        for entity extraction.
        """
        try:
            agent_name = self.kwami_config.soul.name or self.kwami_config.kwami_name
            await self._memory.add_exchange(
                assistant_content=content,
                assistant_name=agent_name,
            )
        except Exception as e:
            logger.warning("Failed to add exchange to memory: %s", e)

    def _extract_message_content(self, message: Any) -> str:
        """Extract text content from various message formats.

        Args:
            message: Message object in various possible formats.

        Returns:
            Extracted text content, or empty string if extraction fails.
        """
        if message is None:
            return ""

        # livekit's ChatMessage stores content as list[str | ImageContent];
        # `text_content` joins the text parts. Without this the str-only branch
        # below falls through to str(message) and Zep is fed a pydantic repr
        # ("id='item_...' type='message' role='user' content=[...]") instead of
        # what the speaker actually said.
        text_content = getattr(message, "text_content", None)
        if isinstance(text_content, str) and text_content.strip():
            return text_content.strip()

        # Try common content attributes
        for attr in ("content", "text", "message"):
            if hasattr(message, attr):
                value = getattr(message, attr)
                if value is None:
                    continue
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, list):
                    parts = [part.strip() for part in value if isinstance(part, str)]
                    joined = " ".join(part for part in parts if part)
                    if joined:
                        return joined

        # If message is already a string
        if isinstance(message, str):
            return message.strip()

        # Last resort: stringify but filter out object representations
        text = str(message)
        if text.startswith("<") and text.endswith(">"):
            logger.debug("Could not extract content from message type: %s", type(message))
            return ""

        return text.strip()
