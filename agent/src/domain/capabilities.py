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
            "set_ui_control is the default path for interface requests: domain, control, "
            "value, over workspace, theme, avatar, scene, voice, enhancements, memory, "
            "browser, soul and search. 'Darker' is theme/mode/dark; 'sidebar right' is "
            "theme/sidebarPosition/right; 'open my memory' is workspace/openPanel/memory; "
            "'spikier blob' is avatar/blobSpikes nudged up on x, y and z; 'talk faster' is "
            "voice/ttsSpeed just above the current value."
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
            "You open and close the workspace panels: avatar, scene, voice, audio, "
            "enhancements, history, communications, soul, memory, tools, info, metrics, "
            "account, theme, models, credits, email, calendar. Asked where something is, "
            "open it."
        ),
    ),
    Capability(
        name="panel_layout",
        requires=frozenset({"set_browser_panel", "set_search_panel"}),
        guidance=(
            "The browser panel and the search panel move the same way: set_browser_panel and "
            "set_search_panel both take layout (docked, floating, fullscreen), expand, "
            "position and size when floating, plus center and reset. For search, docked means "
            "the cards orbiting you. Expand before reading something dense and put it back "
            "after. These only move a panel -- navigate_to opens a page and close_navigation "
            "ends the browsing session."
        ),
    ),
    Capability(
        name="search_results",
        requires=frozenset({"focus_search_result", "open_search_result"}),
        guidance=(
            "Search results are numbered from 1 in the order they appear. focus_search_result "
            "only points at one so the user can see which you mean -- it opens nothing -- so "
            "follow it with open_search_result when they want to read the page. Use focus for "
            "'which one?', open for 'that one'."
        ),
    ),
    Capability(
        name="soundtrack",
        requires=frozenset({"control_soundtrack"}),
        guidance=(
            "control_soundtrack runs the app's own music crate, which your avatar reacts "
            "to: play, pause, toggle, next, stop, status, volume (0-1 or 0-100, and it can "
            "ride along with any action). Use it for background music -- 'put something on', "
            "'skip this', 'quieter'. For a named song, artist or video use play_media, which "
            "opens that specific thing in the browser panel. Report the returned isPlaying "
            "rather than assuming success: a browser can refuse to start audio without a "
            "click, and claiming music is playing to someone who can hear the room is worse "
            "than asking them to tap once."
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
            "opacity, shininess, wireframe, glass mode, audio reactivity, plus eyeIrisColors, "
            "eyeIrisPupil and eyeIrisMotion. Colors, spikes, amplitude and rotation take an "
            "object with x, y and z. Nudge modestly and say what you changed."
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
            "For a background described rather than linked -- 'a waterfall behind you', "
            "'something calmer' -- use apply_scene_preset with kind image, video or hdri and "
            "a preset name, after list_scene_presets if unsure. Never invent an image or "
            "HDRI URL for set_scene_control: a guessed URL loads nothing and the background "
            "silently stays as it was."
        ),
    ),
    Capability(
        name="app_language",
        requires=frozenset({"set_app_language", "get_app_language"}),
        guidance=(
            "set_app_language changes the language of the interface -- the labels on screen. "
            "Your own change_language retunes speech recognition and synthesis and leaves "
            "every label untouched. They are different things and the user almost never "
            "means only one: if they ask for the app in another language, or start speaking "
            "one and ask you to switch over, call both."
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
            "change how you actually sound or which model you run, use your own "
            "change_voice, change_realtime_voice, change_ai_model, switch_pipeline_mode -- "
            "immediate, and they keep the conversation. Use both to persist it too."
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
        guidance=(
            "You can text, message and call for the user. Read list_phone_channels first "
            "when you do not know a channel exists -- sending without one just fails. A name "
            "is resolved against the contacts, never guessed: if more than one matches, the "
            "tool refuses and lists them, so ask which is meant rather than choosing. Say "
            "the name and the last digits together before you send, because a wrong contact "
            "match is the failure that actually happens. A call is heavier than a text -- it "
            "rings someone immediately and cannot be recalled. All three ask the user to "
            "confirm in the app, so wait for the result and never say it is sent while the "
            "dialog is open; then report the tool's own field, not your intent. For messages "
            "that is accepted, meaning the provider took it for delivery, which is not the "
            "same as the recipient having read it; for a call it is dialling, because "
            "whether anyone picks up is not knowable here. search_phone_numbers only "
            "searches -- buying a number spends real money and the user does that themselves "
            "in the phone panel."
        ),
    ),
    Capability(
        name="contacts",
        requires=frozenset(
            {"list_contacts", "find_contact", "create_contact", "update_contact", "delete_contact"}
        ),
        guidance=(
            "list_contacts and find_contact are read-only, so use them freely -- especially "
            "before sending or calling, so the number can be read back. find_contact returns "
            "every match plus a unique flag; check it rather than taking the first. "
            "create_contact and update_contact are ungated, and update changes only the "
            "fields you pass. delete_contact cannot be undone and asks the user first. An "
            "ambiguous name is always refused with the candidates listed, never guessed at."
        ),
    ),
    Capability(
        name="wallet",
        requires=frozenset({"get_wallet_summary"}),
        guidance=(
            "get_wallet_summary reads the wallet and cannot move, send or spend anything. "
            "Asked to transfer or add funds, say that has to be done by hand in the wallet "
            "panel. It is also not how a trade is paid for -- prepare_trade and submit_trade "
            "go through the user's own broker, not this wallet."
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
            "numbers as email_ref to read_email_detail, reply_to_email and archive_email. "
            "reply_to_email and send_email need body text and confirm=true: read the message "
            "back and get a clear yes first, never send on an assumption. "
            "check_email_status gives unread counts by category."
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
        name="soul",
        requires=frozenset(
            {"set_soul_control", "get_soul_profile", "list_soul_presets", "apply_soul_preset"}
        ),
        guidance=(
            "You can change who you are: set_soul_control takes name, personality, "
            "systemPrompt, conversationStyle, language, traits (a list), emotionalTraits, "
            "emotionalTone and responseLength. Call get_soul_profile FIRST for any partial "
            "change -- emotionalTraits is sent as a whole object, so 'be a bit warmer' "
            "written blind overwrites the nine traits nobody asked about. The ten traits are "
            "happiness, energy, confidence, calmness, optimism, socialness, patience, "
            "empathy, curiosity and creativity, each from -100 to 100 where 0 is neutral and "
            "negative is the opposite pole -- not a 0-to-1 scale. Two of these need "
            "confirmation and say why: systemPrompt replaces your entire instruction set, "
            "and apply_soul_preset overwrites name, personality, prompt, traits, style, "
            "length and tone together, including anything the user tuned by hand. "
            "list_soul_presets shows what exists."
        ),
    ),
    Capability(
        name="profiles",
        requires=frozenset({"list_kwami_profiles", "switch_kwami_profile"}),
        guidance=(
            "The user can keep several kwamis, each with its own avatar, voice, scene, theme "
            "and phone settings. list_kwami_profiles shows them, switch_kwami_profile changes "
            "the active one. It replaces all of that at once and can lose unsaved changes, so "
            "it is confirmation-gated: name the target and wait for a yes."
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
