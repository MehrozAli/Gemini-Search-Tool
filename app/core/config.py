"""Application configuration objects and helpers."""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env early so pydantic can pick up defaults
load_dotenv()


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    google_api_key: str = Field(..., alias="GOOGLE_API_KEY", description="Google API key for Gemini")
    default_store_display_name: str = Field(
        "my-file-search-store", alias="DEFAULT_STORE_DISPLAY_NAME", description="Fallback store name"
    )
    upload_poll_interval_seconds: float = Field(
        5.0,
        alias="UPLOAD_POLL_INTERVAL_SECONDS",
        ge=1.0,
        description="Polling interval (seconds) for long-running operations",
    )
    default_model: str = Field(
        "gemini-2.5-flash",
        alias="DEFAULT_MODEL",
        description="Default Gemini model to use for queries",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
