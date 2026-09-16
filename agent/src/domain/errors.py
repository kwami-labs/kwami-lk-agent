"""Custom exceptions for Kwami Agent."""


class KwamiError(Exception):
    """Base exception for all Kwami errors."""

    pass


class VoiceProviderError(KwamiError):
    """Raised when there is an issue with a voice provider."""

    pass


class ConfigurationError(KwamiError):
    """Raised when there is an invalid configuration."""

    pass


class ResourceNotFoundError(KwamiError):
    """Raised when a requested resource (voice, model, etc.) is not found."""

    pass
