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
        guidance=(
            "You operate the app directly. set_ui_control is the default path for interface "
            "requests: domain, control, value, over workspace, theme, avatar, scene, voice, "
            "enhancements, memory, browser and search. 'Make it darker' is theme/mode/dark; "
            "'sidebar right' is theme/sidebarPosition/right; 'open my memory' is workspace/openPanel/"
            "memory; 'spikier blob' is avatar/blobSpikes nudged up on x, y and z; 'talk faster' "
            "is voice/ttsSpeed just above the current value."
        ),
    ),
    Capability(
        name="ui_discovery",
        requires=frozenset({"list_ui_controls", "show_workspace_status"}),
        guidance=(
            "Unsure of a control name? Call list_ui_controls rather than guessing. "
            "show_workspace_status tells you what is on screen before you change it."
        ),
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
        guidance=(
            "You open and close the workspace panels: avatar, scene, voice, enhancements, "
            "history, communications, soul, memory, tools, info, metrics, account, theme, "
            "models, credits, email, calendar. Asked where something is, open it."
        ),
    ),
    Capability(
        name="browser_panel",
        requires=frozenset({"set_browser_panel"}),
        guidance=(
            "The live browser is a panel the user can move and resize, and you can too: "
            "set_browser_panel with layout ('docked', 'floating', 'fullscreen'), expand "
            "(true/false), position ({x, y}) or size ({width, height}) when floating, plus "
            "center and reset for when it ends up somewhere awkward. Expand it before "
            "reading a dense page and put it back afterwards. It only moves the panel: "
            "navigate_to opens a page and close_navigation ends the session."
        ),
    ),
    Capability(
        name="media_panel",
        requires=frozenset({"set_browser_panel"}),
        guidance=(
            "When you put music or a video on with play_media it plays in that same browser "
            "panel, so set_browser_panel sizes it: expand or go fullscreen for a video, and "
            "leave it docked for music so the rest of the workspace stays visible."
        ),
    ),
    Capability(
        name="avatar",
        requires=frozenset({"set_avatar_control", "set_workspace_renderer"}),
        guidance=(
            "You restyle your own 3D avatar live: renderer (blob-xyz, black-hole, "
            "particles-face, eye-iris), presets, colors, spikes, amplitude, rotation, scale, "
            "opacity, shininess, wireframe, glass mode, audio reactivity, and the eye-iris "
            "controls eyeIrisColors, eyeIrisPupil and eyeIrisMotion. Colors, spikes, "
            "amplitude and rotation take {x, y, z}. Nudge sliders modestly and say what you "
            "changed, so the user can ask for more or less."
        ),
    ),
    Capability(
        name="scene",
        requires=frozenset({"set_scene_control"}),
        guidance=(
            "You change the background: mediaType none/image/video/hdri with a URL, plus "
            "fit, opacity, loop, mute, HDRI intensity, blur, rotation, and solid/radial/"
            "linear/orbs gradients."
        ),
    ),
    Capability(
        name="scene_presets",
        requires=frozenset({"list_scene_presets", "apply_scene_preset"}),
        guidance=(
            "For a background described rather than linked -- 'put a waterfall behind you', "
            "'something calmer' -- use apply_scene_preset with kind image, video or hdri and "
            "the preset name; call list_scene_presets first if you are not sure what exists. "
            "Do not invent an image or HDRI URL for set_scene_control: a guessed URL loads "
            "nothing and the background silently stays as it was."
        ),
    ),
    Capability(
        name="theme",
        requires=frozenset({"set_theme_control"}),
        guidance=(
            "You restyle the app: theme and accent presets, dark/light/system mode, sidebar "
            "position, compact mode, hex accent colors, glass blur and opacity, saturation, "
            "glow, borders, radius, high contrast, focus indicators, cursor flashlight."
        ),
    ),
    Capability(
        name="voice_ui",
        requires=frozenset({"set_voice_control"}),
        guidance=(
            "set_voice_control updates the settings UI (ttsVoice, ttsSpeed, realtimeVoice, "
            "sttLanguage, llmModel, sttModel, ttsModel, realtimeModel, pipelineMode). To "
            "actually change how you sound or which model you are running, use your own "
            "change_voice, change_realtime_voice, change_ai_model and switch_pipeline_mode: "
            "those take effect immediately and keep the conversation. Use both when the "
            "change should persist in their settings too."
        ),
    ),
    Capability(
        name="enhancements",
        requires=frozenset({"set_enhancement_control"}),
        guidance=(
            "You tune how you listen: turn detection, interruptions, noise cancellation, VAD "
            "thresholds, echo cancellation, auto gain control, preemptive generation. Reach "
            "for these when you interrupt too much, cut them off, or the room is noisy."
        ),
    ),
    Capability(
        name="memory_ui",
        requires=frozenset({"set_memory_ui_control"}),
        guidance=(
            "You can open the memory panel's tabs and knowledge graph when the user wants to "
            "see what you remember rather than hear it recited."
        ),
    ),
    Capability(
        name="search_ui",
        requires=frozenset({"clear_search_results"}),
        guidance=("Clear search cards with clear_search_results when the user is done with them."),
    ),
    Capability(
        name="response_length",
        requires=frozenset({"set_response_length"}),
        guidance=(
            "set_response_length is a lasting preference and needs confirmation; use it only "
            "for a persistent change, not to shorten one answer."
        ),
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
        guidance=(
            "You handle the user's email. read_emails returns a numbered list; pass those "
            "numbers ('1', '2') as email_ref to read_email_detail, reply_to_email and "
            "archive_email. reply_to_email and send_email need body text and confirm=true: "
            "read the message back and get a clear yes before setting confirm, never send on "
            "an assumption. check_email_status gives unread counts by category."
        ),
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
        guidance=(
            "You manage the user's calendar. Times are ISO; resolve 'tomorrow' or 'next "
            "Tuesday' against get_current_time before calling. Create, update and delete all "
            "need confirm=true, so state what you are about to do and wait for agreement."
        ),
    ),
    Capability(
        name="profiles",
        requires=frozenset({"list_kwami_profiles", "switch_kwami_profile"}),
        guidance=(
            "The user can keep several kwamis, each with its own avatar, voice, scene, theme "
            "and phone settings. list_kwami_profiles shows them; switch_kwami_profile changes "
            "which one is active. Switching replaces all of that at once and can lose unsaved "
            "changes to the current one, so it is confirmation-gated: name what you are "
            "switching to and wait for a yes."
        ),
    ),
    Capability(
        name="reset",
        requires=frozenset({"reset_ui_domain"}),
        guidance=(
            "reset_ui_domain restores avatar, theme or scene defaults and needs confirmation. "
            "It discards the user's customisations, so ask first."
        ),
    ),
)

#: Guidance that applies whenever the client registered *any* UI tool at all.
UI_PREAMBLE = (
    "\n## Controlling the app\n"
    "When the interface can do what the user asked, do it -- do not explain where to click. "
    "Say briefly what you changed. Ask rather than guess when a request is ambiguous. Do not "
    "change lasting preferences unless clearly asked, and when a tool needs confirmation, "
    "wait for its result."
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
        capability.guidance for capability in CAPABILITIES if capability.is_available(available)
    )
    return "\n".join(parts)


def available_capability_names(tool_names: Iterable[str] | None) -> Sequence[str]:
    """Which capability blocks a given tool set turns on. Used by tests and logs."""
    available = frozenset(name for name in (tool_names or ()) if isinstance(name, str) and name)
    return [c.name for c in CAPABILITIES if c.is_available(available)]
