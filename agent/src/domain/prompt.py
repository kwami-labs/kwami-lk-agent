"""System-prompt construction.

Extracted from `KwamiAgent._build_system_prompt`, which rebuilt the whole
prompt -- roughly thirty `list.append` calls and four literal dicts -- on every
invocation, including on every live config update. All of the static prose is
now joined once at import time, and the only per-call work is the parts that
actually depend on the soul.

Pure by construction: it takes a soul config and returns a string.
"""

from __future__ import annotations

from typing import Any

from .capabilities import build_capability_guidance

# Memory context is untrusted in length; cap what reaches the system prompt.
MAX_SYSTEM_MEMORY_CONTEXT_CHARS = 2200

RESPONSE_LENGTH_GUIDE: dict[str, str] = {
    "short": "Keep responses brief and concise (1-2 sentences).",
    "medium": "Provide balanced responses with enough detail (2-4 sentences).",
    "long": "Give comprehensive, detailed responses when appropriate.",
}

EMOTIONAL_TONE_GUIDE: dict[str, str] = {
    "neutral": "Maintain a balanced, objective tone.",
    "warm": "Express warmth and friendliness in your interactions.",
    "enthusiastic": "Show enthusiasm and energy in your responses.",
    "calm": "Maintain a calm, soothing demeanor.",
    "playful": "Use a light, playful voice while staying helpful.",
    "confident": "Speak with confident, decisive language.",
    "serious": "Use a serious, focused, no-fluff voice.",
    "compassionate": "Use compassionate, emotionally supportive language.",
}

# Slider direction labels: (negative pole, positive pole).
TRAIT_LABELS: dict[str, tuple[str, str]] = {
    "happiness": ("sadder", "happier"),
    "energy": ("more low-energy", "more energetic"),
    "confidence": ("more tentative", "more confident"),
    "calmness": ("more tense", "calmer"),
    "optimism": ("more cautious", "more optimistic"),
    "socialness": ("more reserved", "more social"),
    "empathy": ("more detached", "more empathic"),
    "curiosity": ("less exploratory", "more curious"),
    "creativity": ("more literal", "more creative"),
    "patience": ("more brisk", "more patient"),
}

# Some traits read more strongly in a voice than others.
TRAIT_WEIGHTS: dict[str, float] = {
    "happiness": 1.1,
    "energy": 1.0,
    "confidence": 1.2,
    "calmness": 1.25,
    "optimism": 1.05,
    "socialness": 0.9,
    "empathy": 1.35,
    "curiosity": 0.95,
    "creativity": 0.9,
    "patience": 1.15,
}

# Sliders below this weighted magnitude are not worth a directive.
TRAIT_MIN_MAGNITUDE = 10.0
MAX_TRAIT_DIRECTIVES = 5

_STATIC_GUIDANCE_PARTS: tuple[str, ...] = (
    "\n\nYou are interacting via voice. Keep responses concise and conversational.",
    "Do not use emojis, asterisks, markdown, or other special characters.",
    "Speak naturally as if having a real conversation.",
    "\nWhen users share their name, remember it and use it naturally in conversation.",
    "Be genuinely interested in learning about who you're talking to.",
    "\nYou can reconfigure yourself mid-conversation, and the conversation is kept: "
    "change_voice or change_speaking_speed for how you sound, change_realtime_voice on the "
    "realtime pipeline, change_ai_model to move to another model or provider (the user can "
    "say a family name like Claude, Gemini or Groq, or an exact model id), "
    "switch_pipeline_mode to move between the standard and realtime pipelines, and "
    "change_language for the conversation language. Do it when asked rather than "
    "describing which setting to open. Use get_pipeline_status, list_available_models or "
    "list_available_voices when you need to know what you are running or could switch to.",
    "\nWhen the user asks to find products, gifts, or things to buy (e.g. bags, clothes, items), use the product_search tool first so they see actual product cards with product image, name, and price—not store website links. If product_search says it is not configured, use web_search with search_for_products=True instead.",
    "Remember what the user searched for; use your memory of past searches in follow-up answers.",
    "If the user says to discard, remove, or dismiss a result (e.g. 'discard the first one', 'remove that card'), call dismiss_search_result with the 0-based index (first card = 0, second = 1).",
    "If the user wants more like a specific result (e.g. 'find more like this', 'similar to that one'), run web_search with a query like 'similar to [that result title] buy' or 'products like [title]' and set search_for_products=True.",
    "\nYou can browse the web for the user. Use navigate_to to open a website. "
    "The page opens in a live browser panel embedded in the app. The user sees "
    "everything you do in real time. Use read_navigation_page to see page content and interactive elements, then "
    "click_in_navigation to click elements (prefer element_id like 'el-5'), "
    "type_in_navigation to type text, press_key_in_navigation for keys like Enter, "
    "and scroll_navigation to scroll. Describe what you see and what you're doing so the user can follow along.",
    "\nWhen the user asks you to research, investigate, compare or explain something in "
    "depth, use deep_research rather than web_search: it runs several angles at once and "
    "comes back with a briefing. Summarise it conversationally in your own words and offer "
    "to go deeper -- never read it out verbatim.",
    "\nFor markets: get_market_quote gives an exact, current price for a stock, ETF, index, "
    "currency or crypto, which a search snippet does not. Use deep_research for the story "
    "behind a move. You give information, not financial advice -- say so if you are asked to "
    "choose for them.",
    "\nTo play music or video, use play_media with what the user asked for -- it opens the "
    "service, picks the first result and starts it playing, which navigate_to alone cannot "
    "do. Then control_playback (pause, resume, stop, mute, unmute, restart), "
    "set_playback_volume and get_now_playing. Say what you are putting on.",
    "\nTo trade, the order goes through three steps and they cannot be skipped. "
    "prepare_trade validates it, prices it and gives you a confirmation code -- it sends "
    "nothing. Read the order and the estimated value back to the user in full, then ask "
    "them to say the code aloud. open_trade_ticket puts their own broker on screen. "
    "submit_trade takes the code AS THE USER SPOKE IT; never pass a code you produced "
    "yourself, and if you did not clearly hear the ticker, the side or the quantity, ask "
    "again rather than preparing an order. cancel_prepared_trade drops it.",
    "ADVANCED NAVIGATION STRATEGIES:\n"
    "1. DIRECT SEARCHING: If the user asks you to search for something on a major site (YouTube, Google, Amazon, etc.), "
    "DO NOT try to navigate to the homepage and click the search bar. Instead, navigate DIRECTLY to the search URL. "
    "Example: for YouTube, use navigate_to('https://www.youtube.com/results?search_query=song+name').\n"
    "2. COMPLEX DOMs: Modern sites (like YouTube) hide elements inside Shadow DOMs that read_navigation_page cannot see. "
    "If you cannot find the element you need to click, use the run_js_in_navigation tool to execute JavaScript directly "
    "to find and click the element (e.g., `document.querySelector('ytd-video-renderer a#video-title').click()`).\n"
    "Use close_navigation when done.",
)

# Joined once, at import, rather than rebuilt on every call.
STATIC_GUIDANCE = "\n".join(_STATIC_GUIDANCE_PARTS)

#: Client tools the *old* fixed UI block depended on. Kept as a name so the
#: contract tests that assert "guidance is gated on registration" still have
#: something to point at; the guidance itself is now assembled per-client from
#: `domain.capabilities`, because a single block naming four domains described
#: about a sixth of what the app actually registers.
UI_CONTROL_TOOL_NAMES = frozenset({"set_ui_control", "list_ui_controls"})

_MEMORY_HEADER_PARTS: tuple[str, ...] = (
    "\n\n## Your Memory\n",
    "You have persistent memory of past conversations with this user.",
    "Use this context to provide personalized responses:\n",
)
MEMORY_HEADER = "\n".join(_MEMORY_HEADER_PARTS)


def describe_emotional_traits(emotional_traits: Any) -> str | None:
    """Turn -100..100 sliders into a short, ordered voice directive.

    Returns None when nothing clears `TRAIT_MIN_MAGNITUDE`, so the caller can
    leave the section out entirely. Unknown keys and non-numeric values are
    ignored rather than raising -- this data comes off the wire.
    """
    if not emotional_traits:
        return None

    weighted: list[tuple[float, str]] = []
    for key, value in emotional_traits.items():
        if key not in TRAIT_LABELS:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue

        weighted_score = score * TRAIT_WEIGHTS.get(key, 1.0)
        magnitude = min(100.0, abs(weighted_score))
        if magnitude < TRAIT_MIN_MAGNITUDE:
            continue

        low_label, high_label = TRAIT_LABELS[key]
        direction = high_label if weighted_score > 0 else low_label
        if magnitude < 35:
            strength = "slightly"
        elif magnitude < 60:
            strength = "moderately"
        elif magnitude < 85:
            strength = "strongly"
        else:
            strength = "very strongly"
        weighted.append((magnitude, f"{strength} {direction}"))

    if not weighted:
        return None

    weighted.sort(key=lambda item: item[0], reverse=True)
    directives = [directive for _, directive in weighted[:MAX_TRAIT_DIRECTIVES]]
    return (
        "\nVoice emotion profile: "
        + ", ".join(directives)
        + ". Keep this consistent without sounding exaggerated."
    )


def build_system_prompt(
    soul: Any,
    memory_context: str | None = None,
    client_tool_names: Any = (),
) -> str:
    """Build the full system prompt for a soul, optionally with memory context.

    Args:
        soul: The soul configuration driving persona and tone.
        memory_context: Retrieved memory, appended under a header and bounded.
        client_tool_names: Names of client-side tools the frontend registered.
            Capability guidance is assembled from exactly these, so the model is
            never told to call something that does not exist -- and, just as
            importantly, is told about everything that does.
    """
    parts: list[str] = []

    if soul.system_prompt:
        parts.append(soul.system_prompt)
    else:
        parts.append(f"You are {soul.name}, {soul.personality}.")

    if soul.traits:
        parts.append(f"\nKey traits: {', '.join(soul.traits)}")

    if soul.conversation_style:
        parts.append(f"\nConversation style: {soul.conversation_style}")

    if soul.response_length in RESPONSE_LENGTH_GUIDE:
        parts.append(f"\n{RESPONSE_LENGTH_GUIDE[soul.response_length]}")

    if soul.emotional_tone in EMOTIONAL_TONE_GUIDE:
        parts.append(f"\n{EMOTIONAL_TONE_GUIDE[soul.emotional_tone]}")

    emotion_profile = describe_emotional_traits(getattr(soul, "emotional_traits", None))
    if emotion_profile:
        parts.append(emotion_profile)

    parts.append(STATIC_GUIDANCE)

    capability_guidance = build_capability_guidance(client_tool_names)
    if capability_guidance:
        parts.append(capability_guidance)

    if memory_context:
        parts.append(MEMORY_HEADER)
        parts.append(memory_context[:MAX_SYSTEM_MEMORY_CONTEXT_CHARS])

    return "\n".join(parts)
