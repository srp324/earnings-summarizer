from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = ""
    tavily_api_key: str = ""
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/earnings_db"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # LLM Settings
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.1
    
    model_config = SettingsConfigDict(
        # Look for .env in the backend directory (where run.py is)
        env_file=[".env", "../.env"],  # Try backend/.env first, then root/.env
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allow lowercase env var names
        extra="ignore",  # Ignore extra env vars not in model
    )


@lru_cache()
def get_settings() -> Settings:
    """Get settings instance (cached)."""
    settings = Settings()
    
    # Debug logging (only log missing keys, not values for security)
    import logging
    logger = logging.getLogger(__name__)
    
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set or empty. Please check your .env file in the backend/ directory.")
    
    return settings

