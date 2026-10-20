"""Kwami Agent - Dynamic AI agent configured by the Kwami frontend library."""

import asyncio
from datetime import datetime
from typing import Any

from livekit.agents import Agent
from livekit.agents.types import NOT_GIVEN
from livekit.agents.voice.agent import find_function_tools

from .brain.config import default_brain
from .brain.router import can_delegate
from .constants import Timeouts
from .domain import KwamiConfig, build_system_prompt
from .domain.charter import pending_items
from .domain.clock import resolve_zone
from .domain.greeting import (
    MAX_CONTEXT_SUMMARY_CHARS,
    GreetingFacts,
    build_greeting_instructions,
    extract_name_from_facts,
    time_of_day,
    topics_from_facts,
)
from .domain.tool_gating import (
    BRAIN_SEARCH_TOOLS,
    VOICE_SWITCH_TOOLS,
    describe_gating,
    unavailable_builtin_tools,
)
from .memory import KwamiMemory
from .runtime.container import room_from_context
from .settings import get_settings
from .tools import (
    AgentToolsMixin,
    BrainToolsMixin,
    ClientToolManager,
    KnowledgeToolsMixin,
    MediaToolsMixin,
    PipelineControlMixin,
    TradingToolsMixin,
)
from .tools.limits import enforce_tool_limit, trim_client_tools
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
    BrainToolsMixin,
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
        turn_handling: Any = NOT_GIVEN,
    ):
        """Initialize the Kwami agent.

        Args:
            config: Kwami configuration with soul, voice settings, etc.
            vad: Voice Activity Detection instance.
            memory: Optional Zep memory instance for persistent context.
            stt: Speech-to-Text instance.
            llm: Large Language Model instance.
            tts: Text-to-Speech instance.
            turn_handling: Per-agent turn-taking overrides (runtime/turn_taking.py).
            skip_greeting: If True, skip the initial greeting (for reconfigurations).
        """
        self.kwami_config = config or KwamiConfig()
        self._vad = vad
        self._memory = memory
        self._skip_greeting = skip_greeting
        # Cached context from _inject_memory_context.
        self._last_memory_context: Any = None
        # The memory text in the prompt, so a re-gated prompt keeps it.
        self._memory_text = ""
        # Whether searches go to the brain; decides tools and prompt alike.
        self._delegating = self._brain_delegates()

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
            turn_handling=turn_handling,
        )

        # The framework sets `_tools = tools + find_function_tools(self)`, so
        # every built-in is now attached whether or not this deployment can
        # serve it. Withhold the ones whose credential is absent: offering
        # `product_search` with no SERPAPI_KEY spends the user's turn on a
        # refusal the agent advertised, and forty schemas go out on every
        # request of every session whether they work or not.
        self._withhold_unavailable_builtins()

    def _brain_delegates(self) -> bool:
        """Whether searches go to the brain instead of the voice's own tools.

        The job's bridge decides once the agent is running. Before that -- in
        `__init__`, where `self.session` raises -- the default roster on this
        deployment's keys stands in for it, and `refresh_brain_gating` corrects
        the guess on entry if the app had switched the brain off.
        """
        bridge = self._bridge()
        if bridge is not None:
            return bool(bridge.delegates())
        return can_delegate(default_brain(), get_settings().provider_keys)

    def _withheld_builtins(self) -> tuple[set[str], frozenset[str]]:
        """Built-in names withheld: (missing a credential, kept back on purpose).

        Kept back: the voice-switching tools while a character voice is set
        (the voice is changed from the Voice panel), and the search tools while
        the brain takes the searches.
        """
        missing = set(unavailable_builtin_tools(get_settings()))
        locked = VOICE_SWITCH_TOOLS if self.kwami_config.voice.voice_locked else frozenset()
        if getattr(self, "_delegating", False):
            locked = locked | BRAIN_SEARCH_TOOLS
        return missing, locked

    def _unavailable_for_prompt(self) -> frozenset[str]:
        """Tools the prompt must not describe: missing ones and delegated searches."""
        missing = unavailable_builtin_tools(get_settings())
        return missing | BRAIN_SEARCH_TOOLS if getattr(self, "_delegating", False) else missing

    async def refresh_brain_gating(self) -> bool:
        """Bring the search tools in line with the brain. True when they changed.

        Runs on entry and whenever the app reconfigures the brain: switched off,
        the voice gets its search tools back in the same call; switched on, it
        loses them before the next turn.
        """
        delegating = self._brain_delegates()
        if delegating == getattr(self, "_delegating", False):
            return False
        self._delegating = delegating
        combined = enforce_tool_limit(
            self.available_builtin_tools(), self.client_tools.create_client_tools()
        )
        await self.update_tools(combined)
        await self.update_instructions(
            self._build_system_prompt(getattr(self, "_memory_text", None) or None)
        )
        logger.info(
            "Searches now %s", "delegated to the brain" if delegating else "done by the voice"
        )
        return True

    def available_builtin_tools(self) -> list[Any]:
        """This agent's built-in tools, bound to it, minus the withheld ones.

        Discovered on the class -- `inspect.getmembers` on a live agent evaluates
        `realtime_llm_session`, which raises without a running activity -- and
        then bound with the descriptor protocol. Unbound, every tool keeps
        `self` in its signature: Gemini Live let it pass, but the Gemini LLM of
        the standard pipeline failed every turn with KeyError 'self' (2026-10-10).
        """
        missing, locked = self._withheld_builtins()
        unavailable = missing | locked
        return [
            tool.__get__(self, type(self))
            for tool in find_function_tools(type(self))
            if _tool_name(tool) not in unavailable
        ]

    def _withhold_unavailable_builtins(self) -> None:
        """Drop built-in tools this deployment has no credential for.

        Reassigns `_tools` and refreshes `_chat_ctx`, which is exactly what the
        framework's own `update_tools` does when there is no activity yet --
        and there never is during `__init__`. Doing it here rather than awaiting
        `update_tools` later keeps the very first request correct; a tool
        withheld only after the greeting would already have been advertised.
        """
        missing, locked = self._withheld_builtins()
        unavailable = missing | locked
        if not unavailable:
            return

        # No `if nothing was dropped` guard: every gated name is a real built-in
        # -- pinned by test_the_gated_names_are_real_tools -- so a non-empty
        # `unavailable` always matches at least one tool here. A guard for the
        # other case would be a branch no test could reach.
        kept = [tool for tool in self._tools if _tool_name(tool) not in unavailable]
        self._tools = kept
        self._chat_ctx = self._chat_ctx.copy(tools=kept)
        builtins = {_tool_name(t) for t in find_function_tools(type(self))}
        logger.info("%s", describe_gating(builtins & missing))
        if locked & VOICE_SWITCH_TOOLS:
            logger.info(
                "Voice locked to the user's character: %s withheld",
                sorted(locked & VOICE_SWITCH_TOOLS),
            )
        if locked & BRAIN_SEARCH_TOOLS:
            logger.info("Searches go to the brain: %s withheld", sorted(BRAIN_SEARCH_TOOLS))

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
            unavailable_tools=self._unavailable_for_prompt(),
            # Only a realtime model speaks with its own voice; a TTS ignores it.
            voice_style=(
                self.kwami_config.voice.realtime_style
                if self.kwami_config.voice.pipeline_type == "realtime"
                else ""
            ),
            voice_locked=self.kwami_config.voice.voice_locked,
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
                self._memory_text = memory_text
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
            # `local_participant` raises (rather than returning None) until the
            # room is connected, and on_enter can run before that.
            try:
                local = room.local_participant
            except Exception:
                local = None
            my_identity = local.identity if local else ""
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
        self._sync_bridge_context()
        # The search tools were gated in __init__ on a guess (no session yet);
        # the bridge knows whether the app switched the brain off.
        try:
            await self.refresh_brain_gating()
        except Exception:
            logger.warning("Could not re-gate the search tools on entry", exc_info=True)

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

        # Greet the user - but only once per session. A greeting cut off by an
        # agent swap (the app's updates rebuild the agent right after connect)
        # is still owed, and the next agent gives it.
        deps = self._deps()
        if self._skip_greeting and not getattr(deps, "greeting_owed", False):
            logger.debug("Skipping greeting (agent was reconfigured)")
            return

        # The app's post-connect updates can replace this agent while it was
        # fetching memory. Greeting from an agent on its way out failed with
        # "speech scheduling is draining" (twice: greeting and fallback), and
        # the greeting is still owed, so the agent that replaces it gives it.
        if self._superseded():
            logger.debug("Agent replaced before greeting; its successor greets")
            return

        # Generate a natural, personalized greeting
        try:
            logger.info("Generating greeting for user...")
            greeting_instructions = await self._build_greeting_instructions()
            if not greeting_instructions:
                # The charter asked for no greeting: wait for the user to speak.
                logger.info("Greeting disabled by the charter")
                return
            handle = self.session.generate_reply(
                instructions=greeting_instructions,
                allow_interruptions=True,
            )
            self._settle_greeting_when_heard(handle)
        except Exception:
            if self._superseded():
                logger.debug("Agent replaced while greeting; its successor greets")
                return
            logger.exception("Failed to generate greeting")
            # Fall back to a simple greeting so the agent still speaks
            try:
                self.session.generate_reply(
                    instructions="Greet the user casually and ask how you can help.",
                    allow_interruptions=True,
                )
            except Exception:
                logger.exception("Failed to generate fallback greeting")

    def _superseded(self) -> bool:
        """True when the session has already moved on to another agent."""
        try:
            current = self.session.current_agent
        except Exception:
            return False
        return current is not None and current is not self

    def _deps(self) -> Any:
        try:
            return self.session.userdata
        except Exception:
            return None

    def _settle_greeting_when_heard(self, handle: Any) -> None:
        """The greeting counts once it played out, not once it was scheduled."""
        deps = self._deps()
        if deps is None or not hasattr(handle, "add_done_callback"):
            return

        def done(h: Any) -> None:
            # Spoken means an assistant message came out of it. A greeting
            # whose model died before saying a word (Gemini Live closing at
            # connect) also ends uninterrupted, and must stay owed for the
            # agent that takes over.
            spoken = any(
                getattr(item, "role", None) == "assistant"
                for item in (getattr(h, "chat_items", None) or [])
            )
            if spoken and not getattr(h, "interrupted", False):
                deps.greeting_owed = False

        handle.add_done_callback(done)

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
        persona = bool(getattr(self.kwami_config.soul, "system_prompt", ""))
        base: dict[str, Any] = {
            "agent_name": agent_name,
            "language": language,
            "persona": persona,
            **self._charter_greeting_fields(),
            "brain_results": await self._take_brain_results(),
        }

        if not (self._memory and self._memory.is_initialized):
            return GreetingFacts(**base)

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
                **base,
                user_name=user_name,
                is_returning_user=is_returning,
                recent_context_summary=summary,
                recent_topics=topics,
            )
        except Exception as e:
            # Memory is an enhancement; the greeting is the product. A Zep
            # failure here means greeting as a stranger, not not greeting.
            logger.warning("Could not extract user info from memory: %s", e)
            return GreetingFacts(**base)

    def _bridge(self) -> Any:
        try:
            # `session` raises until the agent is active, and `userdata` raises
            # on a session started without one.
            userdata = self.session.userdata
        except Exception:
            return None
        return getattr(userdata, "bridge", None)

    def _sync_bridge_context(self) -> None:
        """Tell the brain who it works for. Agents are rebuilt on every config
        change, the bridge is not, so this runs on each entry."""
        bridge = self._bridge()
        if bridge is None:
            return
        soul = self.kwami_config.soul
        charter = getattr(soul, "charter", None)
        bridge.set_context(
            kwami_id=self.kwami_config.kwami_id,
            name=soul.name or self.kwami_config.kwami_name,
            language=self.kwami_config.locale or getattr(soul, "language", "") or "",
            timezone=self.kwami_config.timezone,
            user_profile=charter.user_profile if charter is not None and charter.enabled else "",
        )

    async def _take_brain_results(self) -> list[str]:
        bridge = self._bridge()
        if bridge is None:
            return []
        kwami_id = self.kwami_config.kwami_id
        # On the cloud the owed results live in Supabase; give the load that
        # `_sync_bridge_context` started a moment to land.
        await bridge.ready(kwami_id)
        return [f"{r.title}: {r.summary[:400]}" for r in bridge.take_pending(kwami_id)]

    def _charter_greeting_fields(self) -> dict[str, Any]:
        """What the charter adds to the greeting, or nothing without one."""
        charter = getattr(self.kwami_config.soul, "charter", None)
        if charter is None or not charter.enabled:
            return {}
        zone, _ = resolve_zone(self.kwami_config.timezone)
        return {
            "charter_greeting": charter.greeting,
            "address_as": charter.address_as,
            "pending": pending_items(charter),
            "time_of_day": time_of_day(datetime.now(zone).hour),
        }

    async def on_user_turn_completed(self, turn_ctx: Any, new_message: Any) -> None:
        """Called when user finishes speaking.

        Once the user has spoken the conversation is under way, and a greeting
        cut off earlier is no longer owed.

        Buffers the user message so it can be sent together with the
        assistant's response in a single add_messages call. This produces
        much better knowledge graph construction in Zep.

        Args:
            turn_ctx: Turn context.
            new_message: The user's message.
        """
        deps = self._deps()
        if deps is not None:
            deps.greeting_owed = False
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
