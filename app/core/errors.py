"""Errors raised by the service layer and mapped to HTTP responses in app.main."""


class ConfigurationError(RuntimeError):
    """A required setting (API key, Notion database) is missing."""
