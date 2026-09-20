"""Shared utilities for the memory package.

Provides lazy imports for zep_cloud and common helpers to avoid
startup errors when memory is not being used.
"""

from typing import TYPE_CHECKING

from ..utils.logging import get_logger, redacted

if TYPE_CHECKING:
    pass

logger = get_logger("memory")

__all__ = ["get_zep_imports", "logger", "redacted"]


def get_zep_imports():
    """Lazy import zep_cloud to avoid startup errors when not using memory.

    Returns:
        Tuple of (AsyncZep, ZepMessage, RoleType) or (None, None, None) on failure.
    """
    try:
        from zep_cloud.client import AsyncZep
        from zep_cloud.types import Message as ZepMessage
        from zep_cloud.types import RoleType

        return AsyncZep, ZepMessage, RoleType
    except ImportError:
        logger.exception("Failed to import zep_cloud")
        logger.exception("Install with: pip install zep-cloud")
        return None, None, None
