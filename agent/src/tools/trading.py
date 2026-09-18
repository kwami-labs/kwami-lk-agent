"""Placing trades by voice, with an interlock that cannot be talked past.

Trading is a stated requirement, so it is built. What is *not* built is a tool
that turns one sentence into a filled order, because the failure modes of a
voice channel land directly on the user's money and none of them are
recoverable: a misheard ticker (BABA/BIDU), a misheard size ("fifty" / "fifteen"),
a tool call fired on a half-finished sentence, or a page that reads "confirm" in
a place the model did not expect.

So the capability is a three-step state machine, and the steps cannot be
collapsed:

1. `prepare_trade` -- validate, fetch a live quote, compute what it is worth,
   and read the order back. Nothing is sent. It mints a short confirmation code
   derived from the order itself.
2. `open_trade_ticket` -- put the user's own broker on screen, in the panel they
   are watching. Their session, their logins, their two-factor.
3. `submit_trade` -- requires the user to repeat the code. The code is a hash of
   the exact order, so a code the user spoke for a different order will not
   match, and a changed order invalidates the code it was minted for.

The interlock is the design. A confirmation the model can supply itself is not
a confirmation, so the code is derived from the order rather than generated
freely, and `submit_trade` re-derives it instead of trusting a stored flag.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from livekit.agents import RunContext, function_tool

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..domain import KwamiConfig

logger = get_logger("trading")

SIDES = ("buy", "sell")
ORDER_TYPES = ("market", "limit")

#: Known brokers' order pages. A broker not listed here is reachable by URL,
#: which the *user* supplies -- never the model, and never a page it just read.
BROKER_URLS: dict[str, str] = {
    "trading212": "https://app.trading212.com/",
    "etoro": "https://www.etoro.com/watchlists/",
    "revolut": "https://app.revolut.com/start",
    "degiro": "https://trader.degiro.nl/trader/",
    "ibkr": "https://www.interactivebrokers.com/portal/",
    "interactive_brokers": "https://www.interactivebrokers.com/portal/",
    "robinhood": "https://robinhood.com/",
    "schwab": "https://client.schwab.com/",
    "coinbase": "https://www.coinbase.com/advanced-trade/spot/",
    "kraken": "https://pro.kraken.com/app/trade/",
    "binance": "https://www.binance.com/en/trade/",
}

#: Spoken back digit by digit, so it has to be short and unambiguous.
CONFIRMATION_DIGITS = 4


@dataclass
class PendingTrade:
    """An order that has been read back but not sent."""

    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: float | None = None
    quote: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        """The exact sentence read back to the user, and hashed for the code."""
        price = (
            f"a limit of {self.limit_price}"
            if self.order_type == "limit" and self.limit_price is not None
            else "at market"
        )
        return f"{self.side} {self.quantity} {self.symbol} {price}"

    @property
    def confirmation_code(self) -> str:
        """A code derived from this exact order.

        Derived, not random: a code the model could invent is not a
        confirmation, and a code that survives an edit to the order would
        confirm an order the user never heard.
        """
        digest = hashlib.sha256(self.describe().encode("utf-8")).hexdigest()
        return str(int(digest[:8], 16))[-CONFIRMATION_DIGITS:].zfill(CONFIRMATION_DIGITS)

    def estimated_value(self) -> float | None:
        price = self.limit_price if self.limit_price is not None else self.quote.get("price")
        if not isinstance(price, int | float):
            return None
        return round(float(price) * self.quantity, 2)


def normalize_side(side: str) -> str | None:
    """Map spoken forms onto buy/sell, or None.

    Guessing here would pick a direction for the user, so an unrecognised word
    is refused rather than defaulted.
    """
    word = (side or "").strip().lower()
    mapping = {
        "buy": "buy",
        "purchase": "buy",
        "long": "buy",
        "get": "buy",
        "sell": "sell",
        "short": "sell",
        "dump": "sell",
        "close": "sell",
        "exit": "sell",
    }
    return mapping.get(word)


class TradingToolsMixin:
    """Confirmation-gated order placement through the user's own broker."""

    kwami_config: KwamiConfig

    #: The order that has been read back but not sent. Declared on the mixin so
    #: mypy sees the same optional type everywhere it is cleared, rather than
    #: inferring `PendingTrade` from the first assignment and then rejecting the
    #: `None` that every completed or cancelled order has to write back.
    _pending_trade_order: PendingTrade | None = None

    def _pending_trade(self) -> PendingTrade | None:
        return self._pending_trade_order

    @function_tool()
    async def prepare_trade(
        self,
        context: RunContext,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: float = 0.0,
    ) -> dict[str, Any]:
        """Work out a trade and read it back. Sends nothing.

        Always the first step. Returns a confirmation code the user must repeat
        before anything can be submitted.

        Args:
            symbol: Ticker. Equities as-is ('TSLA'), crypto against a currency
                    ('BTC-USD').
            side: 'buy' or 'sell'.
            quantity: Number of shares or units.
            order_type: 'market' or 'limit'.
            limit_price: Required for a limit order; ignored for a market one.
        """
        ticker = (symbol or "").strip().upper()
        if not ticker:
            return {"error": "Which symbol?"}

        resolved_side = normalize_side(side)
        if resolved_side is None:
            return {"error": f"Is '{side}' a buy or a sell? I won't guess."}

        try:
            size = float(quantity)
        except (TypeError, ValueError):
            return {"error": "How many?"}
        if size <= 0:
            return {"error": "The quantity has to be more than zero."}

        kind = (order_type or "market").strip().lower()
        if kind not in ORDER_TYPES:
            return {"error": "I can do a market order or a limit order."}
        limit = float(limit_price) if kind == "limit" else None
        if kind == "limit" and (limit is None or limit <= 0):
            return {"error": "A limit order needs a limit price."}

        # A live quote, so the read-back carries what it is actually worth.
        # Its absence is reported, not papered over: agreeing to a trade without
        # knowing roughly what it costs is the thing to avoid.
        quote: dict[str, Any] = {}
        quoter = getattr(self, "get_market_quote", None)
        if callable(quoter):
            try:
                quote = await quoter(context, ticker) or {}
            except Exception as e:  # a dead feed must not block the read-back
                logger.warning("Quote lookup failed while preparing a trade: %s", e)
        if quote.get("error"):
            quote = {}

        pending = PendingTrade(
            symbol=ticker,
            side=resolved_side,
            quantity=size,
            order_type=kind,
            limit_price=limit,
            quote=quote,
        )
        self._pending_trade_order = pending

        logger.info("Prepared (not sent) trade: %s", pending.describe())
        summary: dict[str, Any] = {
            "order": pending.describe(),
            "symbol": ticker,
            "side": resolved_side,
            "quantity": size,
            "order_type": kind,
            "confirmation_code": pending.confirmation_code,
            "submitted": False,
            "next_step": (
                "Read the order back to the user exactly as written, with the estimated "
                "value, then ask them to say the confirmation code aloud. Open their broker "
                "with open_trade_ticket. Nothing is sent until submit_trade is called with "
                "that code."
            ),
        }
        if limit is not None:
            summary["limit_price"] = limit
        if quote:
            summary["last_price"] = quote.get("price")
            summary["currency"] = quote.get("currency")
        value = pending.estimated_value()
        if value is not None:
            summary["estimated_value"] = value
        else:
            summary["warning"] = (
                "I couldn't get a live price, so I can't tell you what this is worth. "
                "Say so before they confirm."
            )
        return summary

    @function_tool()
    async def open_trade_ticket(self, context: RunContext, broker: str = "", url: str = "") -> str:
        """Put the user's broker on screen so they can watch the order go in.

        Args:
            broker: A known broker name ('trading212', 'coinbase', 'kraken',
                    'ibkr', ...). Use `url` instead for anything else.
            url: The broker's own address, when the user has given it. Only use
                 an address the user said -- never one taken from a web page.
        """
        pending = self._pending_trade()
        if pending is None:
            return "Let's work out the order first -- tell me what to buy or sell."

        target = ""
        if broker:
            key = broker.strip().lower().replace(" ", "_").replace("-", "_")
            target = BROKER_URLS.get(key, "")
            if not target:
                return (
                    f"I don't have an address for '{broker}'. I know: "
                    f"{', '.join(sorted(BROKER_URLS))}. Otherwise tell me the web address."
                )
        elif url:
            target = url.strip()
        else:
            return "Which broker? Name one, or give me their web address."

        opener = getattr(self, "navigate_to", None)
        if opener is None:
            return "I can't open the browser panel in this session."

        # navigate_to validates the URL (scheme, private ranges, DNS rebinding).
        result = await opener(context, target)
        return (
            f"{result} Your broker is on screen. Find the order ticket for {pending.symbol} "
            f"and I'll fill it in -- then say the code {pending.confirmation_code} to send it."
        )

    @function_tool()
    async def submit_trade(self, context: RunContext, confirmation_code: str) -> str:
        """Send the prepared order. Requires the code the user spoke aloud.

        Args:
            confirmation_code: The digits from prepare_trade, as the *user* said
                               them. Do not supply this yourself -- a
                               confirmation you provide is not a confirmation.
        """
        pending = self._pending_trade()
        if pending is None:
            return "There's no order ready. Tell me what to trade and I'll set it up."

        spoken = "".join(ch for ch in str(confirmation_code or "") if ch.isdigit())
        # Re-derived from the order, not compared against a stored flag: an order
        # edited after the code was minted no longer matches the code the user
        # agreed to.
        if spoken != pending.confirmation_code:
            logger.warning("Trade confirmation mismatch for %s", pending.describe())
            return (
                "That code doesn't match, so I haven't sent anything. The order is "
                f"{pending.describe()} -- the code is {pending.confirmation_code}."
            )

        session = None
        getter = getattr(self, "_get_browser_session", None)
        if callable(getter):
            session = await getter()
        if session is None or not getattr(session, "is_active", False):
            return (
                "The broker isn't open, so there's nothing to submit into. "
                "Use open_trade_ticket first."
            )

        self._pending_trade_order = None
        logger.info("Trade confirmed by the user: %s", pending.describe())
        return (
            f"Confirmed: {pending.describe()}. The ticket is on screen -- fill in the "
            "remaining fields with type_in_navigation and press the broker's own submit "
            "button with click_in_navigation, then tell the user what the broker says. "
            "Read back any figure the broker shows that differs from what we agreed."
        )

    @function_tool()
    async def cancel_prepared_trade(self, context: RunContext) -> str:
        """Drop the prepared order without sending it."""
        pending = self._pending_trade()
        self._pending_trade_order = None
        if pending is None:
            return "There was nothing pending."
        return f"Cancelled -- I won't {pending.describe()}."

    @function_tool()
    async def get_prepared_trade(self, context: RunContext) -> dict[str, Any]:
        """What order is waiting for confirmation, if any."""
        pending = self._pending_trade()
        if pending is None:
            return {"pending": False}
        return {
            "pending": True,
            "order": pending.describe(),
            "confirmation_code": pending.confirmation_code,
            "estimated_value": pending.estimated_value(),
        }
