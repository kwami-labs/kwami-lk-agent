"""Kwami Agent - Dynamic AI agent configured by the Kwami frontend library."""

import asyncio
from typing import Any

from livekit.agents import Agent

from .constants import Timeouts
from .domain import KwamiConfig, build_system_prompt
from .memory import KwamiMemory
from .runtime.container import room_from_context
from .tools import AgentToolsMixin, ClientToolManager
from .utils.logging import get_logger
from .utils.room import should_disconnect_as_duplicate

logger = get_logger("agent")


class KwamiAgent(Agent, AgentToolsMixin):
    """Dynamic AI agent configured by the Kwami frontend library.

    This agent supports:
    - Configurable voice pipeline (STT, LLM, TTS)
    - Persistent memory via Zep Cloud
    - Client-side tools executed via data channel
    - Built-in tools for voice/language control
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
        self._last_memory_context = None  # Cached context from _inject_memory_context

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

        # Get client tools to pass to parent Agent
        combined_tools = self.client_tools.create_client_tools()
        self._tools = combined_tools

        super().__init__(
            instructions=instructions,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
            tools=self._tools,
        )

    def _build_system_prompt(self, memory_context: str | None = None) -> str:
        """Build the system prompt from soul configuration and memory context.

        Args:
            memory_context: Optional memory context to inject into the prompt.

        Returns:
            Complete system prompt string.
        """
        return build_system_prompt(self.kwami_config.soul, memory_context)

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
                logger.info(f"Cached user name from memory: {user_name}")

            memory_text = context.to_system_prompt_addition()

            if memory_text:
                new_instructions = self._build_system_prompt(memory_text)
                await self.update_instructions(new_instructions)
                logger.info("Injected memory context into system prompt")

            # Store context for greeting use (avoids a second API call)
            self._last_memory_context = context
        except Exception as e:
            logger.error(f"Failed to inject memory context: {e}")

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
            logger.info(f"Agent {my_identity} entering room...")

            # Quick check for duplicate agents (non-blocking)
            should_disconnect = await should_disconnect_as_duplicate(room, my_identity)
            if should_disconnect:
                logger.warning(f"Agent {my_identity} disconnecting due to duplicate detection")
                await room.disconnect()
                return
        else:
            logger.warning("Agent entered with no room reference available")

        self._register_session_listeners()

        logger.info(
            f"Kwami agent '{self.kwami_config.kwami_name}' "
            f"({self.kwami_config.kwami_id}) entered room successfully"
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
        except Exception as e:
            logger.error("Memory context failed (%s); greeting without it.", e)

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
        except Exception as e:
            logger.error(f"Failed to generate greeting: {e}")
            # Fall back to a simple greeting so the agent still speaks
            try:
                self.session.generate_reply(
                    instructions="Greet the user casually and ask how you can help.",
                    allow_interruptions=True,
                )
            except Exception:
                logger.error("Failed to generate fallback greeting")

    async def _build_greeting_instructions(self) -> str:
        """Build natural greeting instructions based on memory context.

        Reuses the context already fetched by _inject_memory_context()
        to avoid redundant API calls. Only makes a fresh call if no
        cached context is available.

        Returns:
            Greeting instructions for the LLM.
        """
        agent_name = self.kwami_config.soul.name or self.kwami_config.kwami_name or "Kwami"
        user_name = None
        is_returning_user = False
        recent_context_summary = None
        recent_topics = []

        if self._memory and self._memory.is_initialized:
            try:
                # Use cached user name (already looked up in _inject_memory_context)
                user_name = self._memory._cached_user_name

                # Reuse cached context from _inject_memory_context (avoids 2nd API call)
                context = self._last_memory_context
                if context is None:
                    context = await self._memory.get_context()

                if context.recent_messages or context.facts or context.context_block:
                    is_returning_user = True
                    logger.debug(
                        f"Returning user detected "
                        f"(messages: {len(context.recent_messages)}, facts: {len(context.facts)})"
                    )

                    # Extract recent topics from context block or summary
                    if context.context_block:
                        recent_context_summary = context.context_block[:500]
                    elif context.summary:
                        recent_context_summary = context.summary

                    # Get interesting facts to reference in greeting
                    if context.facts:
                        name_skip = ["name is", "called", "i am", "i'm"]
                        for fact in context.facts[:5]:
                            fact_lower = fact.lower()
                            if not any(skip in fact_lower for skip in name_skip):
                                recent_topics.append(fact)

                # If name not cached, try extracting from facts as fallback
                if not user_name and context and context.facts:
                    import re

                    for fact in context.facts:
                        match = re.search(
                            r"(?:name is|called|i'm|i am)\s+([A-Z][a-z]+)", fact, re.IGNORECASE
                        )
                        if match:
                            potential = match.group(1).capitalize()
                            excluded = {
                                "the",
                                "a",
                                "user",
                                "assistant",
                                "kwami",
                                agent_name.lower(),
                            }
                            if potential.lower() not in excluded:
                                user_name = potential
                                self._memory.set_user_name(user_name)
                                logger.info(f"Found user name from facts: {user_name}")
                                break

            except Exception as e:
                logger.warning(f"Could not extract user info from memory: {e}")

        # Build natural greeting instructions based on what we know
        if user_name:
            if recent_topics:
                topics_str = "; ".join(recent_topics[:3])
                return (
                    f"Greet {user_name} warmly by name, like you're happy to see them again. "
                    f"Reference something from your recent conversations naturally. "
                    f"Here's what you remember about recent topics: {topics_str}. "
                    f"Ask a casual follow-up question about one of these topics, or just ask how something is going. "
                    f"Examples: 'Hey {user_name}! How did that [project/thing] turn out?' or "
                    f"'What's up {user_name}? Been thinking about [topic] lately?' "
                    "Keep it short, friendly, and chill. Don't be formal or robotic. "
                    "Pick ONE topic and ask about it naturally - don't list everything you remember."
                )
            elif recent_context_summary:
                return (
                    f"Greet {user_name} warmly by name, like you're happy to see them again. "
                    f"Here's a summary of your past conversations: {recent_context_summary}. "
                    f"Ask a casual follow-up about something relevant, or just check in on how things are going. "
                    f"Example: 'Hey {user_name}! How's everything going?' or reference something specific. "
                    "Keep it short, friendly, and natural."
                )
            else:
                return (
                    f"Greet {user_name} casually by name, like you're happy to see them again. "
                    f"Something like 'Hey {user_name}, great to see you! What's on your mind today?' or "
                    f"'What's up {user_name}? How've you been?' "
                    "Keep it short, friendly, and chill. Don't be formal or robotic. "
                    "Don't repeat the same greeting every time - vary it naturally."
                )
        elif is_returning_user:
            return (
                f"Greet the user casually like you've talked before but can't remember their name. "
                f"Something like 'Hey there! Good to hear from you again. By the way, I'm {agent_name} - "
                "what's your name?' Keep it natural and chill."
            )
        else:
            return (
                f"Introduce yourself casually to this new user. "
                f"Something like 'Hey there! I'm {agent_name}, what's your name?' "
                "Keep it short, friendly, and natural. Don't be overly formal or give a long introduction."
            )

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
                logger.warning(f"Failed to buffer user message: {e}")

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
            logger.warning(f"Failed to add exchange to memory: {e}")

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
            logger.debug(f"Could not extract content from message type: {type(message)}")
            return ""

        return text.strip()
