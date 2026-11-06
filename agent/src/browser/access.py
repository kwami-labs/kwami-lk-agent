"""Where the user lets Jarvis go on the web, and what it may do on each site.

The user keeps the list in the app (Settings > Access); the API writes it to
``jarvis_site_rules``. Each site has one level, and each level includes the ones
before it:

* ``blocked``: never opened.
* ``read``: opened and read.
* ``browse``: also clicked through, scrolled and searched.
* ``write``: also typed into and forms sent.
* ``account``: also the user's signed-in session, sign-in forms and scripts included.

A pattern covers its subdomains and the longest one that matches decides; the
pattern ``*`` holds the level for every other site. Without any rule every site is
at ``account``, which is how Jarvis worked before the list existed. The API keeps
the same rules (``api/src/services/site_access_service.py``).

Jarvis can ask for a site it has no rule for. The request waits as ``pending``
until the user answers it in the app, and grants nothing before that: a page that
talks the model into asking still needs the user to say yes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..adapters.supabase import SupabaseClient

logger = get_logger("browser.access")

TABLE = "jarvis_site_rules"
LEVELS = ("blocked", "read", "browse", "write", "account")
EVERY_OTHER_SITE = "*"
DEFAULT_LEVEL = "account"
#: How long a list read from the database is trusted. Short, so a site the user
#: approves in the app works on Jarvis's next try.
CACHE_SECONDS = 10.0
MAX_REASON_CHARS = 300
#: Requests one user can have waiting at once. A page cannot flood the app.
MAX_PENDING = 20

#: What each level lets Jarvis do, in the words the model reads.
LEVEL_WORDS = {
    "blocked": "not allowed to open it",
    "read": "allowed to open and read it only",
    "browse": "allowed to open, read, click and scroll, but not to type",
    "write": "allowed to browse and type and send forms, but not to use the user's account",
    "account": "allowed everything, the user's signed-in account included",
}


def host_of(address: str) -> str:
    """``https://www.GitHub.com/x`` -> ``github.com``. Empty when there is no host."""
    raw = (address or "").strip()
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        host = (urlsplit(raw).hostname or "").strip(".").lower()
    except ValueError:
        return ""
    if host.startswith("*."):
        host = host[2:]
    if host.startswith("www."):
        host = host[4:]
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def level_for(host: str, rules: dict[str, str], default: str = DEFAULT_LEVEL) -> str:
    """The level for ``host``: the longest pattern that is it or a parent of it."""
    best = ""
    for pattern in rules:
        if (host == pattern or host.endswith(f".{pattern}")) and len(pattern) > len(best):
            best = pattern
    return rules[best] if best else default


def allows(level: str, needed: str) -> bool:
    """Whether ``level`` covers what ``needed`` asks for. Unknown levels allow nothing."""
    if level not in LEVELS or needed not in LEVELS:
        return False
    return LEVELS.index(level) >= LEVELS.index(needed)


@dataclass
class _Rules:
    sites: dict[str, str] = field(default_factory=dict)
    pending: dict[str, str] = field(default_factory=dict)
    default: str = DEFAULT_LEVEL
    read_at: float = 0.0


@dataclass(frozen=True)
class Decision:
    """What was decided for one address, and why."""

    host: str
    level: str
    pattern: str
    allowed: bool


class SiteAccess:
    """The user's list, read from Supabase and kept for a few seconds.

    Without a database there is no list, and every site is at ``account``. When the
    database fails after a list was read, the last list still applies: a blocked
    site does not open because the network blinked.
    """

    def __init__(self, store: SupabaseClient | None) -> None:
        self._store = store
        self._rules: dict[str, _Rules] = {}

    @property
    def enabled(self) -> bool:
        return self._store is not None

    async def _load(self, user_id: str) -> _Rules:
        cached = self._rules.get(user_id)
        if cached is not None and time.monotonic() - cached.read_at < CACHE_SECONDS:
            return cached
        if self._store is None or not user_id:
            return _Rules()
        rows = await self._store.select(
            TABLE,
            {"select": "pattern,access,status", "user_id": f"eq.{user_id}"},
            what="the sites Jarvis may visit",
        )
        if rows is None:
            if cached is not None:
                logger.warning("Could not read the site list; keeping the last one")
                return cached
            return _Rules()
        rules = _Rules(read_at=time.monotonic())
        for row in rows:
            pattern, access = str(row.get("pattern") or ""), str(row.get("access") or "")
            if access not in LEVELS or not pattern:
                continue
            if row.get("status") == "pending":
                rules.pending[pattern] = access
            elif pattern == EVERY_OTHER_SITE:
                rules.default = access
            else:
                rules.sites[pattern] = access
        self._rules[user_id] = rules
        return rules

    def forget(self, user_id: str) -> None:
        """Read the list again on the next check."""
        self._rules.pop(user_id, None)

    async def decide(self, user_id: str, address: str, needed: str) -> Decision:
        """Whether Jarvis may do ``needed`` at ``address`` for this user."""
        host = host_of(address)
        rules = await self._load(user_id)
        level = level_for(host, rules.sites, rules.default)
        pattern = max(
            (p for p in rules.sites if host == p or host.endswith(f".{p}")),
            key=len,
            default=EVERY_OTHER_SITE,
        )
        return Decision(host=host, level=level, pattern=pattern, allowed=allows(level, needed))

    async def request(self, user_id: str, address: str, access: str, reason: str) -> str:
        """Ask the user for ``access`` at ``address``. What happened, for the model to say.

        Never grants anything: the row waits as ``pending`` for the user. A site
        that already has a rule, or a request, is not asked for again.
        """
        if access not in LEVELS or access == "blocked":
            return "Ask for one of: read, browse, write, account."
        host = host_of(address)
        if not host or "." not in host:
            return "That is not a site address."
        if self._store is None or not user_id:
            return "There is no list of sites here: every site is already open to you."
        self.forget(user_id)
        rules = await self._load(user_id)
        current = level_for(host, rules.sites, rules.default)
        if allows(current, access):
            return f"You already may do that on {host}: you are {LEVEL_WORDS[current]}."
        if host in rules.pending:
            return f"You already asked for {host}. The user answers it in Settings > Access."
        if host in rules.sites:
            return (
                f"The user set {host} themselves: you are {LEVEL_WORDS[current]}. "
                "Only they can change it, in Settings > Access."
            )
        if len(rules.pending) >= MAX_PENDING:
            return "There are already many requests waiting. Ask the user to answer them first."
        stored = await self._store.insert(
            TABLE,
            {
                "user_id": user_id,
                "pattern": host,
                "access": access,
                "status": "pending",
                "reason": (reason or "").strip()[:MAX_REASON_CHARS] or None,
            },
            what="a request for a site",
        )
        self.forget(user_id)
        if not stored:
            return (
                "I could not leave the request. Ask the user to add the site in Settings > Access."
            )
        return (
            f"Request left for {host} ({access}). Tell the user it is waiting in "
            "Settings > Access, and that nothing changes until they approve it."
        )

    async def summary(self, user_id: str) -> str:
        """The list in a few lines, for the model to tell the user."""
        if self._store is None or not user_id:
            return "There is no list of sites here: you may open any public site."
        self.forget(user_id)
        rules = await self._load(user_id)
        lines = [f"Every other site: {LEVEL_WORDS[rules.default]}."]
        lines += [f"- {p}: {LEVEL_WORDS[a]}" for p, a in sorted(rules.sites.items())]
        if rules.pending:
            waiting = ", ".join(f"{p} ({a})" for p, a in sorted(rules.pending.items()))
            lines.append(f"Waiting for the user: {waiting}.")
        return "\n".join(lines)


def refusal_words(decision: Decision, needed: str) -> str:
    """What the model is told when a step is not allowed."""
    if decision.level == "blocked":
        return (
            f"The user has blocked {decision.host}. Do not try another way in; tell them, "
            "and that they can change it in Settings > Access."
        )
    return (
        f"On {decision.host} you are {LEVEL_WORDS[decision.level]}, and this needs "
        f"'{needed}'. Tell the user. If it is needed for what they asked, you can ask for "
        "it with request_site_access, and they approve it in Settings > Access."
    )
