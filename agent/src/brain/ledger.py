"""What the brain costs, and what it owes the user.

Two backends behind one interface (`add_spend`, `month_spend`, `add_pending`,
`take_pending`):

* `SupabaseLedger` -- the user's Supabase (`brain_spend`, `brain_pending`),
  used whenever the agent writes Supabase directly, which is the case on
  LiveKit Cloud, where local disk does not survive a restart.
* `Ledger` -- two small files on disk, for local work and the tests.

`create_ledger()` picks one. On disk, under ``agent/.data/brain`` (git-ignored):

* ``spend.json`` -- dollars per calendar month, per model. The user set a
  50 $/month figure and asked to be *told*, not cut off, so this only counts.
* ``pending/<kwami_id>.json`` -- results of tasks that finished after the call
  ended. The next greeting reads them out and they are cleared.

Prices are list prices per million tokens. Anthropic's come from the API's own
pricing table; Gemini and OpenAI are estimates, which is why the app labels the
total as approximate. An unknown model is billed at its provider's default
rather than at zero: under-reporting spend is the failure that matters.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..adapters.supabase import SupabaseClient
    from ..settings import Settings

logger = get_logger("brain.ledger")

DATA_DIR = Path(__file__).resolve().parents[2] / ".data" / "brain"

#: (input $/1M, output $/1M)
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
    "gemini-3.5-flash": (0.50, 3.00),
    "gemini-3-pro-preview": (2.00, 12.00),
    "gpt-5.2": (1.75, 14.00),
}
PROVIDER_DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "anthropic": (4.00, 20.00),
    "google": (2.00, 12.00),
    "openai": (2.00, 15.00),
}
#: Per search, on top of tokens.
SEARCH_PRICES: dict[str, float] = {"anthropic": 0.01, "google": 0.014, "openai": 0.01}


def token_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    rate_in, rate_out = PRICES.get(model) or PROVIDER_DEFAULT_PRICES.get(provider, (5.0, 25.0))
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def search_cost(provider: str, searches: int) -> float:
    return SEARCH_PRICES.get(provider, 0.01) * max(searches, 0)


def _month(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m")


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


#: How many owed results one kwami keeps. Older ones are dropped.
MAX_PENDING = 20


@dataclass
class PendingResult:
    task_id: str
    title: str
    summary: str
    finished_at: str
    #: Labels of the models that worked on it.
    models: list[str] = field(default_factory=list)


class Ledger:
    """File-backed spend and pending results. One process writes at a time."""

    backend = "disk"

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self._dir = data_dir
        self._lock = threading.Lock()

    # -- lifecycle (no-ops on disk) ------------------------------------------

    async def bind(self, kwami_id: str, user_id: str | None = None) -> None:
        """Load what this kwami's user has stored. Disk needs no loading."""

    async def flush(self, timeout: float = 5.0) -> None:
        """Wait for writes still in flight. Disk writes are synchronous."""

    # -- spend -------------------------------------------------------------

    def _spend_path(self) -> Path:
        return self._dir / "spend.json"

    def _read(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            logger.warning("Unreadable brain ledger file %s; starting it afresh", path.name)
            return None

    def _write(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def add_spend(self, model_id: str, usd: float) -> None:
        if usd <= 0:
            return
        with self._lock:
            data = self._read(self._spend_path()) or {}
            month = data.setdefault(_month(), {"usd": 0.0, "by_model": {}})
            month["usd"] = round(month.get("usd", 0.0) + usd, 6)
            by_model = month.setdefault("by_model", {})
            by_model[model_id] = round(by_model.get(model_id, 0.0) + usd, 6)
            try:
                self._write(self._spend_path(), data)
            except OSError:
                logger.exception("Could not record brain spend")

    def month_spend(self) -> dict[str, Any]:
        data = self._read(self._spend_path()) or {}
        month = data.get(_month()) or {}
        return {
            "month": _month(),
            "usd": float(month.get("usd", 0.0)),
            "byModel": dict(month.get("by_model", {})),
        }

    # -- pending results ---------------------------------------------------

    def _pending_path(self, kwami_id: str) -> Path:
        return self._dir / "pending" / f"{_SAFE_ID.sub('_', kwami_id)[:80] or 'default'}.json"

    def add_pending(self, kwami_id: str, item: PendingResult) -> None:
        with self._lock:
            path = self._pending_path(kwami_id)
            items = self._read(path) or []
            items.append(asdict(item))
            try:
                self._write(path, items[-MAX_PENDING:])
            except OSError:
                logger.exception("Could not store a pending brain result")

    def take_pending(self, kwami_id: str) -> list[PendingResult]:
        """Read and clear. Called once the greeting has them."""
        with self._lock:
            path = self._pending_path(kwami_id)
            items = self._read(path) or []
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not clear pending brain results for %s", kwami_id)
        out: list[PendingResult] = []
        for raw in items:
            if isinstance(raw, dict):
                try:
                    out.append(PendingResult(**raw))
                except TypeError:
                    continue
        return out


class SupabaseLedger(Ledger):
    """Spend and pending results in the user's Supabase, cached in memory.

    The interface stays synchronous because its callers are: the brain status
    is built on every publish and the greeting is assembled in one pass. So
    reads come from a cache that `bind()` fills when the kwami enters, and
    writes update the cache at once and reach Supabase in the background.
    `flush()` is awaited when the call ends so those writes are not lost with
    the worker.

    Spend is the user's, across kwamis (the budget is per person), and is added
    with the `brain_add_spend` function so two sessions finishing at once cannot
    overwrite each other. Spend made before the owner is known is held and sent
    once `bind()` finds them.
    """

    backend = "supabase"

    def __init__(self, store: SupabaseClient, data_dir: Path = DATA_DIR) -> None:
        super().__init__(data_dir)
        self._store = store
        self._user_id: str | None = None
        self._kwami_id = ""
        self._month = _month()
        self._by_model: dict[str, float] = {}
        #: (kwami_id, model_id, usd) spent before the owner was known.
        self._unsent: list[tuple[str, str, float]] = []
        #: kwami_id -> [(row id, result)] not yet read out.
        self._pending: dict[str, list[tuple[str, PendingResult]]] = {}
        self._loads: dict[str, asyncio.Future[None]] = {}
        self._writes: set[asyncio.Task[Any]] = set()

    @property
    def user_id(self) -> str | None:
        return self._user_id

    # -- lifecycle -------------------------------------------------------------

    async def bind(self, kwami_id: str, user_id: str | None = None) -> None:
        """Load this kwami's owed results and its owner's spend. Once per kwami."""
        from ..adapters.supabase import as_user_id

        if (resolved := as_user_id(user_id)) and resolved != self._user_id:
            self._set_user(resolved)
            if self._loads:
                # A kwami already loaded without knowing its owner: its spend
                # was never read.
                self._spawn(self._load_spend())
        if not kwami_id:
            return
        self._kwami_id = kwami_id
        load = self._loads.get(kwami_id)
        if load is None:
            load = self._loads[kwami_id] = asyncio.ensure_future(self._load(kwami_id))
        # Shielded: a caller that stops waiting must not cancel the load.
        await asyncio.shield(load)

    async def _load(self, kwami_id: str) -> None:
        from ..adapters.supabase import kwami_owner

        try:
            if self._user_id is None and (owner := await kwami_owner(self._store, kwami_id)):
                self._set_user(owner)
            if self._user_id is not None:
                await self._load_spend()
            await self._load_pending(kwami_id)
        except Exception:
            logger.exception("Could not load the brain ledger from Supabase")

    def _set_user(self, user_id: str) -> None:
        if self._user_id == user_id:
            return
        self._user_id = user_id
        unsent, self._unsent = self._unsent, []
        for kwami_id, model_id, usd in unsent:
            self._send_spend(kwami_id, model_id, usd)

    async def _load_spend(self) -> None:
        rows = await self._store.select(
            "brain_spend",
            {
                "select": "model_id,usd",
                "user_id": f"eq.{self._user_id}",
                "month": f"eq.{self._month}",
            },
            what="the brain's monthly spend",
        )
        if rows is None:
            return
        totals: dict[str, float] = {}
        for row in rows:
            model_id = str(row.get("model_id") or "")
            totals[model_id] = totals.get(model_id, 0.0) + float(row.get("usd") or 0.0)
        # What this process spent while the read was in flight went out as an
        # increment the read may not include; keep the larger of the two.
        for model_id, usd in self._by_model.items():
            totals[model_id] = max(totals.get(model_id, 0.0), usd)
        self._by_model = totals

    async def _load_pending(self, kwami_id: str) -> None:
        rows = await self._store.select(
            "brain_pending",
            {
                "select": "id,task_id,title,text,models,created_at",
                "kwami_id": f"eq.{kwami_id}",
                "taken_at": "is.null",
                "order": "created_at.asc",
                "limit": str(MAX_PENDING),
            },
            what="the brain's owed results",
        )
        if not rows:
            return
        known = {row_id for row_id, _ in self._pending.get(kwami_id, [])}
        loaded = [
            (
                str(row["id"]),
                PendingResult(
                    task_id=str(row.get("task_id") or ""),
                    title=str(row.get("title") or ""),
                    summary=str(row.get("text") or ""),
                    finished_at=str(row.get("created_at") or ""),
                    models=[str(m) for m in row.get("models") or []],
                ),
            )
            for row in rows
            if row.get("id") and str(row["id"]) not in known
        ]
        self._pending[kwami_id] = (loaded + self._pending.get(kwami_id, []))[-MAX_PENDING:]

    async def flush(self, timeout: float = 5.0) -> None:
        if not self._writes:
            return
        _done, pending = await asyncio.wait(set(self._writes), timeout=timeout)
        if pending:
            logger.warning("%d brain ledger write(s) did not reach Supabase in time", len(pending))

    def _spawn(self, coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            coro.close()
            logger.warning("No event loop; a brain ledger write was dropped")
            return
        task = loop.create_task(coro)
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    # -- spend -------------------------------------------------------------------

    def _roll_month(self) -> None:
        if (month := _month()) != self._month:
            self._month = month
            self._by_model = {}

    def add_spend(self, model_id: str, usd: float) -> None:
        if usd <= 0:
            return
        self._roll_month()
        self._by_model[model_id] = round(self._by_model.get(model_id, 0.0) + usd, 6)
        if self._user_id is None:
            self._unsent.append((self._kwami_id, model_id, usd))
            return
        self._send_spend(self._kwami_id, model_id, usd)

    def _send_spend(self, kwami_id: str, model_id: str, usd: float) -> None:
        self._spawn(
            self._store.rpc(
                "brain_add_spend",
                {
                    "p_user_id": self._user_id,
                    "p_kwami_id": kwami_id,
                    "p_month": self._month,
                    "p_model_id": model_id,
                    "p_usd": round(usd, 6),
                },
                what="the brain's spend",
            )
        )

    def month_spend(self) -> dict[str, Any]:
        self._roll_month()
        return {
            "month": self._month,
            "usd": round(sum(self._by_model.values()), 6),
            "byModel": dict(self._by_model),
        }

    # -- pending results ---------------------------------------------------------

    def add_pending(self, kwami_id: str, item: PendingResult) -> None:
        # The id is minted here so the row can be stamped as taken later without
        # reading it back.
        row_id = str(uuid.uuid4())
        items = self._pending.setdefault(kwami_id, [])
        items.append((row_id, item))
        del items[:-MAX_PENDING]
        row: dict[str, Any] = {
            "id": row_id,
            "kwami_id": kwami_id,
            "task_id": item.task_id,
            "title": item.title,
            "text": item.summary,
            "models": list(item.models),
        }
        if self._user_id:
            row["user_id"] = self._user_id
        self._spawn(self._store.insert("brain_pending", row, what="an owed brain result"))

    def take_pending(self, kwami_id: str) -> list[PendingResult]:
        """Read and clear. The rows are stamped `taken_at` in the background."""
        items = self._pending.pop(kwami_id, [])
        if items:
            ids = ",".join(row_id for row_id, _ in items)
            self._spawn(
                self._store.update(
                    "brain_pending",
                    {"taken_at": datetime.now(UTC).isoformat()},
                    {"id": f"in.({ids})", "taken_at": "is.null"},
                    what="the owed brain results that were read out",
                )
            )
        return [item for _, item in items]


def create_ledger(settings: Settings | None = None) -> Ledger:
    """Supabase when the agent writes it directly, disk otherwise."""
    from ..adapters.supabase import direct_supabase

    store = direct_supabase(settings)
    if store is None:
        return Ledger()
    logger.info("Brain ledger: Supabase")
    return SupabaseLedger(store)
