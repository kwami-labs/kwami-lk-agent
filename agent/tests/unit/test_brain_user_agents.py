"""The user's agents from the app: what joins the team, with which tools, and how far."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.brain.team import (
    MAX_OWN_MEMBERS,
    OPUS,
    SONNET,
    agent_definitions,
    member_from,
    permissions_for,
    platform_members,
)


def _row(name: str, **extra: Any) -> dict[str, Any]:
    return {"name": name, "description": f"{name} desc", "prompt": f"{name} prompt", **extra}


def test_an_agent_of_the_app_is_a_member_with_no_more_than_the_executor() -> None:
    members = platform_members(
        [_row("facturas", model=OPUS, tools=["Read", "Bash", "Telepathy"], effort="high")]
    )
    member = members["facturas"]
    assert (member.model, member.tools, member.effort) == (OPUS, ["Read", "Bash"], "high")


def test_unknown_models_and_efforts_fall_back_and_missing_tools_read_only() -> None:
    member = platform_members([_row("x1", model="gpt-9", effort="max")])["x1"]
    assert (member.model, member.tools, member.effort) == (
        SONNET,
        ["Read", "Glob", "Grep"],
        "medium",
    )


def test_the_teams_names_and_bad_names_are_kept_out() -> None:
    rows = [
        _row("reviewer"),
        _row("Con Espacios"),
        _row(""),
        {"name": "sin-prompt", "description": "d"},
        "not a row",
        _row("bueno"),
    ]
    assert list(platform_members(rows)) == ["bueno"]


def test_there_is_a_ceiling() -> None:
    rows = [_row(f"agente-{i}") for i in range(MAX_OWN_MEMBERS + 5)]
    assert len(platform_members(rows)) == MAX_OWN_MEMBERS


def test_member_from_refuses_what_is_not_a_member() -> None:
    assert member_from(None, "x") is None
    assert member_from({"description": "", "prompt": "p"}, "x") is None


def test_the_narrowest_mode_that_has_every_tool() -> None:
    assert permissions_for(["Read", "Bash"]) == "full_no_push"
    assert permissions_for(["Read", "Write"]) == "edit_only"
    assert permissions_for(["Edit"]) == "edit_only"
    assert permissions_for(["Read", "WebSearch"]) == "read_only"
    assert permissions_for([]) == "read_only"


def test_on_the_pc_a_file_of_the_same_name_wins(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "facturas.json").write_text(
        '{"description": "del archivo", "prompt": "p"}', encoding="utf-8"
    )
    extra = platform_members([_row("facturas"), _row("informes")])
    team = agent_definitions(lambda **kw: SimpleNamespace(**kw), tmp_path, extra)  # type: ignore[arg-type]
    assert team["facturas"].description == "del archivo"
    assert team["informes"].description == "informes desc"
    assert "reviewer" in team
