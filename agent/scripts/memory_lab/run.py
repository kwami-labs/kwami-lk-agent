"""Memory lab: what a conversation leaves in the memory graph, without voice.

Each fixture in `fixtures/` is a conversation written as `U:`/`A:` lines (with
the tools the answer ran, `[herramientas: ...]`), the questions memory must
answer afterwards, the words that must not end up in it, and the work the brain
finished. Both variants write the same conversations to throwaway Zep users:

* A writes the way the agent did before the memory gate: every user turn is
  extracted, messages are signed "User", finished work leaves nothing.
* B goes through this checkout's own `KwamiMemory`: the gate (noise and errands
  stay in the thread, out of the graph), the name the user gives on their Zep
  record, and the work journal.

Then it waits for Zep to process everything and measures, per variant: junk
facts and nodes, self-loops, what the user's own node is called, and how many
of the questions a graph search answers. Both users are deleted at the end
(`--keep` to look at them in the Zep console first).

Usage, from `agent/`:

    uv run python -X utf8 scripts/memory_lab/run.py --env-file ../.env

It needs `ZEP_API_KEY` (in the environment or the env file) and writes to that
Zep project, only under `e2e_memlab_*` users.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))

from src.domain import KwamiMemoryConfig  # noqa: E402
from src.memory import manager as memory_manager  # noqa: E402
from src.memory.context import BALANCED_BUDGET, LEAN_BUDGET, RICH_BUDGET  # noqa: E402
from src.memory.gate import stated_name  # noqa: E402
from src.memory.manager import KwamiMemory  # noqa: E402
from src.memory.ontology import configure_ontology  # noqa: E402

KWAMI_NAME = "Jarvis"
LANGUAGE = "es-ES"
_TOOLS = re.compile(r"\s*\[herramientas:\s*([^\]]*)\]\s*$")


@dataclass
class Fixture:
    name: str
    turns: list[tuple[str, str, list[str]]] = field(default_factory=list)
    must: list[tuple[str, list[str]]] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    work: list[tuple[str, str, str]] = field(default_factory=list)


def parse(path: Path) -> Fixture:
    fixture, section = Fixture(path.stem), ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            section = line[3:].strip().lower()
        elif section == "conversación" and line[:2] in ("U:", "A:"):
            text, tools = line[2:].strip(), []
            found = _TOOLS.search(text)
            if found:
                tools = [t.strip() for t in found.group(1).split(",") if t.strip()]
                text = text[: found.start()]
            fixture.turns.append(("user" if line[0] == "U" else "assistant", text, tools))
        elif section == "debe recordar" and line.startswith("- "):
            query, _, expect = line[2:].partition("::")
            fixture.must.append((query.strip(), [e.strip() for e in expect.split("|") if e]))
        elif section == "no debe recordar" and line.startswith("- "):
            fixture.must_not.append(line[2:].strip())
        elif section == "trabajo terminado" and line.startswith("- "):
            date, title, result = (part.strip() for part in line[2:].split("|", 2))
            fixture.work.append((date, title, result))
    return fixture


def fold(text: str) -> str:
    bare = unicodedata.normalize("NFKD", text)
    return "".join(c for c in bare if not unicodedata.combining(c)).casefold()


async def write_before(client: Any, uid: str, fixtures: list[Fixture]) -> None:
    """The write path before the memory gate, replayed call for call."""
    from zep_cloud.types import Message

    await client.user.add(user_id=uid, metadata={"assistant_name": KWAMI_NAME, "memlab": True})
    await configure_ontology(client, uid)
    for fx in fixtures:
        thread_id = f"{uid}_{fx.name}"
        await client.thread.create(thread_id=thread_id, user_id=uid)
        pending: str | None = None
        for role, text, _tools in fx.turns:
            if role == "user":
                if pending is not None:  # spoke twice: flushed alone
                    await client.thread.add_messages(
                        thread_id=thread_id,
                        messages=[Message(role="user", content=pending, name="User")],
                    )
                pending = text
                continue
            batch = [Message(role="user", content=pending, name="User")] if pending else []
            batch.append(Message(role="assistant", content=text, name=KWAMI_NAME))
            await client.thread.add_messages(
                thread_id=thread_id, messages=batch, ignore_roles=["assistant"]
            )
            pending = None


async def write_now(api_key: str, uid: str, fixtures: list[Fixture]) -> None:
    """This checkout's `KwamiMemory`, driven the way the agent drives it."""
    for fx in fixtures:
        memory = KwamiMemory(
            KwamiMemoryConfig(
                enabled=True, api_key=api_key, user_id=uid, session_id=f"{uid}_{fx.name}"
            ),
            kwami_id=uid,
            kwami_name=KWAMI_NAME,
        )
        if not await memory.initialize():
            raise SystemExit(f"KwamiMemory did not initialize for {uid}")
        tools: list[str] = []
        for role, text, turn_tools in fx.turns:
            if role == "user":
                # As KwamiAgent._buffer_for_memory: a name given goes to the record.
                name = stated_name(text, agent_name=KWAMI_NAME)
                if name and name != memory._cached_user_name:
                    memory.set_user_name(name)
                await memory.buffer_user_message(text, name=memory._cached_user_name)
                tools = turn_tools
                continue
            await memory.add_exchange(text, KWAMI_NAME, tools=tools)
        for date, title, result in fx.work:
            await memory.add_work_done(title, result, date, language=LANGUAGE)
        # The name goes to the record in the background; a call lasts long enough.
        await asyncio.gather(*memory._background, return_exceptions=True)
        await memory.close()


async def wait_processed(client: Any, uid: str, timeout_s: int = 600) -> float:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        episodes = (await client.graph.episode.get_by_user_id(user_id=uid, lastn=300)).episodes
        if episodes and all(e.processed for e in episodes):
            return time.monotonic() - start
        await asyncio.sleep(10)
    return -1


async def walk(fetch: Any, uid: str) -> list[Any]:
    out, seen, cursor = [], set(), None
    while True:
        kwargs: dict[str, Any] = {"user_id": uid, "limit": 50}
        if cursor:
            kwargs["uuid_cursor"] = cursor
        batch = list(await fetch(**kwargs) or [])
        fresh = [r for r in batch if r.uuid_ not in seen]
        if not fresh:
            return out
        seen.update(r.uuid_ for r in fresh)
        out += fresh
        cursor = batch[-1].uuid_


async def measure(client: Any, uid: str, fixtures: list[Fixture]) -> dict[str, Any]:
    nodes = await walk(client.graph.node.get_by_user_id, uid)
    edges = await walk(client.graph.edge.get_by_user_id, uid)
    names = {n.uuid_: n.name for n in nodes}
    user_node = (await client.user.get_node(uid)).node
    junk = [w for fx in fixtures for w in fx.must_not]
    probes = []
    for fx in fixtures:
        for query, expect in fx.must:
            found = await client.graph.search(user_id=uid, query=query, scope="edges", limit=5)
            facts = [e.fact for e in (found.edges or [])]
            hit = any(fold(x) in fold(f) for f in facts for x in expect)
            probes.append({"fixture": fx.name, "query": query, "hit": hit, "top": facts[:3]})
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "user_node_name": user_node.name if user_node else None,
        "junk_edges": [e.fact for e in edges if any(fold(w) in fold(e.fact or "") for w in junk)],
        "junk_nodes": [n.name for n in nodes if any(fold(w) in fold(n.name or "") for w in junk)],
        "self_loops": [e.fact for e in edges if e.source_node_uuid == e.target_node_uuid],
        "probes": probes,
        "recall": f"{sum(p['hit'] for p in probes)}/{len(probes)}",
        "all_nodes": sorted(f"{(n.labels or ['?'])[-1]}: {n.name}" for n in nodes),
        "all_facts": [
            f"{e.name} [{names.get(e.source_node_uuid)} -> {names.get(e.target_node_uuid)}] "
            f"{e.fact}"
            for e in edges
        ],
        "user_summary": user_node.summary if user_node else None,
    }


def load_key(env_file: str | None) -> str:
    if env_file:
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            if line.startswith("ZEP_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    key = os.environ.get("ZEP_API_KEY", "")
    if not key:
        raise SystemExit("ZEP_API_KEY not found")
    return key


async def main() -> None:
    from zep_cloud.client import AsyncZep

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env-file", help="a .env holding ZEP_API_KEY")
    parser.add_argument("--keep", action="store_true", help="do not delete the users")
    args = parser.parse_args()

    api_key = load_key(args.env_file)
    fixtures = [parse(p) for p in sorted((HERE / "fixtures").glob("*.md"))]
    client = AsyncZep(api_key=api_key, timeout=60)
    run = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    users = {"A": f"e2e_memlab_{run}_A", "B": f"e2e_memlab_{run}_B"}
    report: dict[str, Any] = {"run": run, "fixtures": [f.name for f in fixtures], "variants": {}}
    # The shared context templates are the live agent's; the lab leaves them alone.
    for budget in (LEAN_BUDGET, BALANCED_BUDGET, RICH_BUDGET):
        memory_manager._TEMPLATES_READY.add(budget.template_id)
    try:
        print(f"[A] writing {users['A']}", flush=True)
        await write_before(client, users["A"], fixtures)
        print(f"[B] writing {users['B']}", flush=True)
        await write_now(api_key, users["B"], fixtures)
        for variant, uid in users.items():
            waited = await wait_processed(client, uid)
            print(f"[{variant}] processed after {waited:.0f}s", flush=True)
            await asyncio.sleep(20)  # summaries and dedup settle after the last episode
            report["variants"][variant] = {
                "waited_s": waited,
                **await measure(client, uid, fixtures),
            }
    finally:
        if not args.keep:
            for uid in users.values():
                try:
                    await client.user.delete(uid)
                    print("deleted", uid, flush=True)
                except Exception as e:
                    print("could not delete", uid, e, flush=True)
    out = HERE / "reports" / f"report_{run}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for variant, r in report["variants"].items():
        print(
            f"\n== {variant}: nodes {r['nodes']} facts {r['edges']} "
            f"user-node {r['user_node_name']!r} junk-facts {len(r['junk_edges'])} "
            f"junk-nodes {len(r['junk_nodes'])} self-loops {len(r['self_loops'])} "
            f"recall {r['recall']}"
        )
    print("report:", out)


if __name__ == "__main__":
    asyncio.run(main())
