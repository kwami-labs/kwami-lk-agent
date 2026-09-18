"""Guidance must describe exactly the client tools that were registered.

Both directions are failures, and only one of them was ever guarded:

* Describing a tool the client did not register invites a hallucinated call.
  The old code guarded this, with a single block gated on two tool names.
* *Not* describing one that was registered means the feature cannot be reached
  by voice at all. Nothing guarded this, and the app registers twenty-eight
  tools while the guidance named four domains -- so email, calendar, panels,
  the renderer, the memory graph and the scene background were all callable and
  all unmentioned.

The second test class is the parity check the product actually needs: every
tool the app registers has to be covered by some capability block, or asking
for it out loud does nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.domain import KwamiSoulConfig, build_system_prompt
from src.domain.capabilities import CAPABILITIES, available_capability_names
from src.domain.capabilities import build_capability_guidance as guidance

#: Every tool `useWorkspaceAgentTools.ts` registers, as of the kwami-app commit
#: this was written against. Checked in rather than read from the app so the
#: suite stays runnable on its own; `test_manifest_matches_the_app` re-derives
#: it from the app when the app happens to be checked out alongside.
APP_TOOL_NAMES = frozenset(
    {
        "set_ui_control",
        "open_workspace_panel",
        "close_workspace_panel",
        "focus_transcription_panel",
        "set_panel_control",
        "set_theme_control",
        "set_avatar_control",
        "set_scene_control",
        "set_voice_control",
        "set_enhancement_control",
        "set_memory_ui_control",
        "set_workspace_renderer",
        "set_response_length",
        "clear_search_results",
        "reset_ui_domain",
        "list_ui_controls",
        "show_workspace_status",
        "read_emails",
        "read_email_detail",
        "reply_to_email",
        "send_email",
        "archive_email",
        "check_email_status",
        "list_calendar_events",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
        # The draggable/expandable live browser panel, the scene presets that
        # make a *described* background reachable, and multi-kwami switching.
        "set_browser_panel",
        "list_scene_presets",
        "apply_scene_preset",
        "list_kwami_profiles",
        "switch_kwami_profile",
    }
)

APP_TOOLS_SOURCE = (
    Path(__file__).resolve().parents[4] / "kwami-app/src/composables/useWorkspaceAgentTools.ts"
)


# -- gating -----------------------------------------------------------------


def test_no_guidance_without_ui_tools() -> None:
    """The telephony case: no app on the other end, so describe no app."""
    assert guidance([]) == ""
    assert guidance(None) == ""
    assert guidance(["some_custom_domain_tool"]) == ""


def test_a_block_is_emitted_only_for_its_own_tools() -> None:
    only_email = guidance(["read_emails", "send_email"])

    assert "email" in only_email.lower()
    assert "calendar" not in only_email.lower(), "described a calendar the client cannot reach"
    assert "blob-xyz" not in only_email, "described an avatar the client cannot reach"


@pytest.mark.parametrize("capability", CAPABILITIES, ids=lambda c: c.name)
def test_every_capability_can_be_switched_on(capability) -> None:
    """Each block must be reachable, or it is guidance nobody will ever see."""
    names = available_capability_names(sorted(capability.requires))
    assert capability.name in names


@pytest.mark.parametrize("capability", CAPABILITIES, ids=lambda c: c.name)
def test_each_of_a_capabilitys_tools_switches_it_on(capability) -> None:
    """`requires` is an ANY-match; a partial registration still gets described."""
    for tool in capability.requires:
        assert capability.name in available_capability_names([tool]), (
            f"{tool} alone does not enable {capability.name}"
        )


# -- parity with the app ----------------------------------------------------


def test_every_app_tool_is_covered_by_a_capability() -> None:
    """Anything the app registers must be something the model is told about.

    A registered-but-undescribed tool is the exact shape of "the UI can do it
    but asking cannot": the capability exists, the user has no way to reach it.
    """
    covered = frozenset().union(*(c.requires for c in CAPABILITIES))
    uncovered = APP_TOOL_NAMES - covered
    assert not uncovered, f"app tools with no guidance: {sorted(uncovered)}"


def test_capabilities_do_not_invent_tools() -> None:
    """The reverse: guidance must not name a tool the app does not register."""
    declared = frozenset().union(*(c.requires for c in CAPABILITIES))
    invented = declared - APP_TOOL_NAMES
    assert not invented, f"guidance gated on tools the app never registers: {sorted(invented)}"


@pytest.mark.skipif(
    not APP_TOOLS_SOURCE.exists(),
    reason="kwami-app is not checked out beside this repo",
)
def test_manifest_matches_the_app() -> None:
    """Catch drift when the two repos sit side by side.

    Skipped when the app is absent, so the suite stands alone -- but when a
    developer has both, a tool added to the app without guidance fails here
    rather than shipping as a feature that only works by clicking.
    """
    source = APP_TOOLS_SOURCE.read_text(encoding="utf-8")
    registered = set(re.findall(r"registerTool\(\{\s*name:\s*'([a-z_]+)'", source))
    assert registered, "could not parse any registerTool calls; this test has gone stale"

    # Only one direction is a defect. A tool the app registers and the agent
    # does not describe cannot be reached by voice -- that is the bug. The
    # reverse (described here, not yet registered) happens while the two repos
    # land a feature at different moments, and costs nothing: an unregistered
    # tool simply never switches its block on.
    missing = registered - APP_TOOL_NAMES
    assert not missing, (
        "kwami-app registers tools this agent does not describe: "
        f"{json.dumps(sorted(missing))}. Add them to a capability in domain/capabilities.py."
    )


# -- integration with the prompt --------------------------------------------


def test_prompt_includes_guidance_for_registered_tools() -> None:
    soul = KwamiSoulConfig(name="Kwami", personality="helpful")
    prompt = build_system_prompt(soul, client_tool_names=sorted(APP_TOOL_NAMES))

    for expected in ("Controlling the app", "set_ui_control", "email", "calendar", "blob-xyz"):
        assert expected in prompt, f"the full tool set did not produce guidance for {expected}"


def test_prompt_omits_guidance_without_tools() -> None:
    soul = KwamiSoulConfig(name="Kwami", personality="helpful")
    prompt = build_system_prompt(soul, client_tool_names=())

    assert "Controlling the app" not in prompt
    assert "set_ui_control" not in prompt


def test_prompt_growth_is_bounded() -> None:
    """Guidance rides in every request; it must not become the prompt.

    Not a style rule: this text is re-sent on every turn of every session, so
    its size is a recurring cost on every token bill the product generates.
    """
    soul = KwamiSoulConfig(name="Kwami", personality="helpful")
    full = build_system_prompt(soul, client_tool_names=sorted(APP_TOOL_NAMES))
    bare = build_system_prompt(soul, client_tool_names=())

    added = len(full) - len(bare)
    # ~4.6k for 32 tools. The block this replaced was ~1.8k and described 4
    # domains. Raise this only alongside a reason, not to make a build pass.
    assert added < 5200, f"capability guidance added {added} characters to every request"
