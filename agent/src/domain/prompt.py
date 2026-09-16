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
    "When the user asks you to control the app workspace or interface, prefer the available client workspace tools instead of telling them what to click.",
    "Use the structured client UI tools for requests like opening panels, changing theme settings, modifying avatar parameters, adjusting scene controls, changing voice settings, tuning enhancements, clearing search results, or checking workspace status.",
    "Prefer set_ui_control as the default tool for free-form interface requests because it gives you one consistent path for domain, control, and value.",
    "For visible UI changes, briefly say what action you are taking. If a request is ambiguous, ask a clarifying question instead of guessing.",
    "Do not change lasting workspace preferences unless the user clearly asks. If a tool requires confirmation, wait for that result before continuing.",
    "If you are unsure which structured UI control to use, call list_ui_controls first to inspect the supported control names and domains.",
    "Examples: if the user says 'make it darker', use set_ui_control with domain='theme', control='mode', value='dark'. "
    "If they say 'move the sidebar right', use domain='theme', control='sidebarPosition', value='right'. "
    "If they say 'open memory', use domain='workspace', control='openPanel', value='memory'. "
    "If they say 'make the blob spikier', use domain='avatar', control='blobSpikes' with a modest increase to x, y, and z values. "
    "If they say 'switch to particles face', use domain='avatar', control='renderer', value='particles-face'. "
    "If they say 'speak a bit faster', use domain='voice', control='ttsSpeed', value set slightly above the current speed.",
    "\nWhen users share their name, remember it and use it naturally in conversation.",
    "Be genuinely interested in learning about who you're talking to.",
    "\nYou can change your voice or the AI model being used if the user requests it.",
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


def build_system_prompt(soul: Any, memory_context: str | None = None) -> str:
    """Build the full system prompt for a soul, optionally with memory context."""
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

    if memory_context:
        parts.append(MEMORY_HEADER)
        parts.append(memory_context[:MAX_SYSTEM_MEMORY_CONTEXT_CHARS])

    return "\n".join(parts)
