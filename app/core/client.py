"""Shared clients used across the API."""

from functools import lru_cache

from google import genai

from app.core.config import get_settings
from app.core.errors import ConfigurationError


@lru_cache
def get_genai_client() -> genai.Client:
    """Instantiate and cache the Google GenAI client."""

    api_key = get_settings().google_api_key
    if not api_key:
        raise ConfigurationError("GOOGLE_API_KEY is not set")
    return genai.Client(api_key=api_key)
