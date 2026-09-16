"""Consistent logging utilities for Kwami agent."""

import logging
import traceback

# Single logger name for the entire agent
LOGGER_NAME = "kwami-agent"


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Optional sub-logger name. If provided, creates "kwami-agent.{name}".
              If None, returns the main "kwami-agent" logger.

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def log_error(
    logger: logging.Logger,
    message: str,
    error: Exception,
    include_traceback: bool = True,
) -> None:
    """Log an error with consistent formatting.

    Args:
        logger: Logger instance to use.
        message: Error message prefix.
        error: The exception that occurred.
        include_traceback: Whether to include full traceback.
    """
    if include_traceback:
        tb = traceback.format_exc()
        logger.error(f"{message}: {type(error).__name__}: {error}\n{tb}")
    else:
        logger.error(f"{message}: {type(error).__name__}: {error}")
