"""The model keys a user connected from the app.

The API stores them encrypted (`provider_keys_service.py` there, table
`kwami_provider_keys`) under a secret both sides hold. They are opened here, only
for that user's job, only into the bridge of that call, and never written anywhere
else or logged. They reach the brain only: the voice pipeline stays on the
deployment's own providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptography.fernet import Fernet, InvalidToken

from ..utils.logging import get_logger
from .router import PROVIDER_KEYS

if TYPE_CHECKING:
    from ..adapters.supabase import SupabaseClient

logger = get_logger("brain.user_keys")

TABLE = "kwami_provider_keys"


async def load_user_keys(store: SupabaseClient, user_id: str, secret: str) -> dict[str, str]:
    """`{ENV_VAR_NAME: key}` for what this user connected. Empty on any trouble."""
    if not secret or not user_id:
        return {}
    try:
        fernet = Fernet(secret.encode())
    except ValueError:
        logger.warning("KWAMI_PROVIDER_KEYS_SECRET is not a valid key; ignoring users' keys")
        return {}
    rows = await store.select(
        TABLE,
        {"select": "provider,encrypted_key", "user_id": f"eq.{user_id}"},
        what="the user's model keys",
    )
    keys: dict[str, str] = {}
    for row in rows or []:
        name = PROVIDER_KEYS.get(str(row.get("provider")))
        # The local executor's entry names the Anthropic key too, under "anthropic".
        if name is None or row.get("provider") == "local":
            continue
        try:
            keys[name] = fernet.decrypt(str(row.get("encrypted_key")).encode()).decode()
        except InvalidToken:
            logger.warning("A stored model key could not be opened; skipping it")
    return keys
