"""Shared clients used across the API."""
from functools import lru_cache
from google import genai

from app.core.config import get_settings


@lru_cache
def get_genai_client() -> genai.Client:
    """Instantiate and cache the Google GenAI client."""

    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key)
