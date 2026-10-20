"""System-prompt guidance derived from the client tools actually registered.

The UI-control guidance used to be one fixed block, gated on two tool names
(`set_ui_control`, `list_ui_controls`) and naming four domains: theme, avatar,
scene, voice. The frontend registers twenty-eight tools. Everything else --
opening panels, switching the avatar renderer, loading a background, reading
and answering email, creating calendar events, changing the model, opening the
knowledge graph -- was registered, callable, and never mentioned to the model.
A tool the model is not told about is a tool the user cannot reach by asking,
which is the entire premise of a voice interface.

So guidance is assembled from capability blocks, each gated on the tools it
actually needs. Two failure modes, one on each side, and both matter:

* Telling the model about a tool the client did not register invites a
  hallucinated call and a dead turn.
* *Not* telling it about one that was registered means the feature silently
  does not exist by voice, which is indistinguishable from it being broken.

Blocks are keyed on tool names rather than on a version number or a feature
flag because the registration message is the only thing that is actually true
about a given client at a given moment. An older app, a partial registration
and a future app all describe themselves correctly with no coordination.

What a block says is only what no single tool description can. Every client
tool already reaches the model with its own description -- how it works, what
it confirms, which field to report -- and the blocks used to repeat all of it:
11,661 characters on every request, most of a realtime prompt, burying the
persona at the top. A block now carries a choice between overlapping tools
(soundtrack or play_media, the settings screen or a real voice change, the
app's labels or its speech) or a rule the model could otherwise skip, such as
reading a message back before it sets `confirm=true` itself. A block with
nothing of that kind is empty: its tools are still registered and described,
and gating stays as it was. Rules shared by every tool live once, in
`UI_PREAMBLE`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Capability:
    """One block of guidance and the client tools it depends on.

    `requires` is an ANY-match: a block is emitted when at least one of its
    tools is present. A capability whose tools are partially registered is
    still worth describing -- the alternative is silence about the half that
    does exist.
    """

    name: str
    requires: frozenset[str]
    guidance: str

    def is_available(self, tool_names: frozenset[str]) -> bool:
        return bool(self.requires & tool_names)


# Ordered deliberately: the router first, then what it routes to, then the
# things that are their own tools. The model reads this top to bottom.
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        name="ui_router",
        requires=frozenset({"set_ui_control"}),
        guidance="set_ui_control is the default path for interface requests: 'darker' is theme/mode/dark.",
    ),
    Capability(
        name="ui_discovery",
        requires=frozenset({"list_ui_controls", "show_workspace_status"}),
        guidance="Unsure of a control name? Call list_ui_controls rather than guessing.",
    ),
    Capability(
        name="panels",
        requires=frozenset(
            {
                "open_workspace_panel",
                "close_workspace_panel",
                "set_panel_control",
                "focus_transcription_panel",
            }
        ),
        guidance="Asked where something is, open its panel.",
    ),
    Capability(
        name="panel_layout",
        requires=frozenset({"set_browser_panel", "set_search_panel"}),
        guidance="",
    ),
    Capability(
        name="search_results",
        requires=frozenset({"focus_search_result", "open_search_result"}),
        guidance="",
    ),
    Capability(
        # When to write is the charter's rule (domain/charter.py); this only
        # says which tool does what, so the two never contradict each other.
        name="user_profile",
        requires=frozenset({"remember_about_user", "forget_about_user", "read_user_profile"}),
        guidance=(
            "The user profile in your charter is kept with three tools: remember_about_user "
            "adds a fact, forget_about_user removes lines when asked to forget something, "
            "and read_user_profile reads it when asked what you know about them."
        ),
    ),
    Capability(
        name="soundtrack",
        requires=frozenset({"control_soundtrack"}),
        guidance="control_soundtrack is background music from the app's own crate; a specific song or video is play_media. Report isPlaying, because browsers block audio until a click.",
    ),
    Capability(
        name="media_panel",
        requires=frozenset({"set_browser_panel"}),
        guidance="",
    ),
    Capability(
        name="avatar",
        requires=frozenset({"set_avatar_control", "set_workspace_renderer"}),
        guidance="Your avatar renderers are blob-xyz, black-hole, particles-face and eye-iris. Colors, spikes, amplitude and rotation take x, y and z. Nudge modestly.",
    ),
    Capability(
        name="scene",
        requires=frozenset({"set_scene_control"}),
        guidance="",
    ),
    Capability(
        name="scene_presets",
        requires=frozenset({"list_scene_presets", "apply_scene_preset"}),
        guidance="A described background ('a waterfall behind you') is apply_scene_preset. Never invent an image or HDRI URL: a guessed one loads nothing.",
    ),
    Capability(
        name="app_language",
        requires=frozenset({"set_app_language", "get_app_language"}),
        guidance="Asked for another language, call both set_app_language (the labels) and change_language (your speech).",
    ),
    Capability(
        name="theme",
        requires=frozenset({"set_theme_control", "export_theme", "import_theme"}),
        guidance="",
    ),
    Capability(
        name="voice_ui",
        requires=frozenset({"set_voice_control"}),
        guidance="set_voice_control only edits the settings screen. To really change your voice or model use change_voice, change_realtime_voice, change_ai_model or switch_pipeline_mode; use both to persist it.",
    ),
    Capability(
        name="enhancements",
        requires=frozenset({"set_enhancement_control"}),
        guidance="If you interrupt too much or the room is noisy, tune listening with set_enhancement_control.",
    ),
    Capability(
        name="recall",
        requires=frozenset({"list_memories", "forget_memory", "forget_everything"}),
        guidance="",
    ),
    Capability(
        name="conversations",
        requires=frozenset(
            {
                "list_conversations",
                "open_conversation",
                "return_to_live_conversation",
                "delete_conversation",
                "clear_transcript",
            }
        ),
        guidance="",
    ),
    Capability(
        name="metrics",
        requires=frozenset({"get_performance_metrics", "reset_performance_metrics"}),
        guidance="",
    ),
    Capability(
        name="memory_ui",
        requires=frozenset({"set_memory_ui_control"}),
        guidance="",
    ),
    Capability(
        name="search_ui",
        requires=frozenset({"clear_search_results"}),
        guidance="",
    ),
    Capability(
        name="response_length",
        requires=frozenset({"set_response_length"}),
        guidance="set_response_length is lasting: never use it to shorten a single answer.",
    ),
    Capability(
        name="communications",
        requires=frozenset(
            {
                "send_sms",
                "send_whatsapp_message",
                "place_call",
                "list_phone_channels",
                "search_phone_numbers",
            }
        ),
        guidance="",
    ),
    Capability(
        name="contacts",
        requires=frozenset(
            {"list_contacts", "find_contact", "create_contact", "update_contact", "delete_contact"}
        ),
        guidance="",
    ),
    Capability(
        name="wallet",
        requires=frozenset({"get_wallet_summary", "create_wallet"}),
        guidance="Money never moves by voice: wallet funds are handled by hand in its panel, and trades go through prepare_trade.",
    ),
    Capability(
        name="email",
        requires=frozenset(
            {
                "read_emails",
                "read_email_detail",
                "reply_to_email",
                "send_email",
                "archive_email",
                "check_email_status",
            }
        ),
        guidance="reply_to_email and send_email take confirm=true only after you have read the message back and heard a clear yes.",
    ),
    Capability(
        name="calendar",
        requires=frozenset(
            {
                "list_calendar_events",
                "create_calendar_event",
                "update_calendar_event",
                "delete_calendar_event",
            }
        ),
        guidance="For the calendar, times are ISO: resolve 'tomorrow' against get_current_time. Say what you will create, change or delete and hear a yes before confirm=true.",
    ),
    Capability(
        name="soul",
        requires=frozenset(
            {"set_soul_control", "get_soul_profile", "list_soul_presets", "apply_soul_preset"}
        ),
        guidance="Call get_soul_profile before any partial soul change: emotionalTraits is replaced whole, each from -100 to 100 with 0 neutral.",
    ),
    Capability(
        name="kwami_admin",
        requires=frozenset({"create_kwami", "rename_kwami", "delete_kwami"}),
        guidance="",
    ),
    Capability(
        name="account",
        requires=frozenset({"get_credit_balance", "sign_out"}),
        guidance="",
    ),
    Capability(
        name="randomize",
        requires=frozenset({"randomize_appearance"}),
        guidance="",
    ),
    Capability(
        name="profiles",
        requires=frozenset({"list_kwami_profiles", "switch_kwami_profile"}),
        guidance="",
    ),
    Capability(
        name="reset",
        requires=frozenset({"reset_ui_domain"}),
        guidance="",
    ),
)

#: Guidance that applies whenever the client registered *any* UI tool at all.
UI_PREAMBLE = (
    "\n## Controlling the app\n"
    "When the interface can do what the user asked, do it -- do not explain where to click -- "
    "and say briefly what changed. Each tool's description says how it works; follow it. "
    "Ask rather than guess: an ambiguous name comes back refused with the candidates, so ask "
    "which one. Change lasting preferences only when clearly asked. When a tool asks the user "
    "to confirm in the app, wait for its result and report the field it returns, never your "
    "intent."
)

#: Tools whose presence means the client is a full workspace rather than, say, a
#: phone call with a couple of custom tools bolted on.
_ANY_UI_TOOL = frozenset().union(*(capability.requires for capability in CAPABILITIES))


def build_capability_guidance(tool_names: Iterable[str] | None) -> str:
    """Guidance for exactly the client tools that were registered.

    Returns an empty string when the client registered no UI tools, which is
    the telephony case: there is no app on the other end, and describing one
    would invite calls into a void.
    """
    available = frozenset(name for name in (tool_names or ()) if isinstance(name, str) and name)
    if not (available & _ANY_UI_TOOL):
        return ""

    parts: list[str] = [UI_PREAMBLE]
    parts.extend(
        capability.guidance
        for capability in CAPABILITIES
        if capability.guidance and capability.is_available(available)
    )
    return "\n".join(parts)


def available_capability_names(tool_names: Iterable[str] | None) -> Sequence[str]:
    """Which capability blocks a given tool set turns on. Used by tests and logs."""
    available = frozenset(name for name in (tool_names or ()) if isinstance(name, str) and name)
    return [c.name for c in CAPABILITIES if c.is_available(available)]
