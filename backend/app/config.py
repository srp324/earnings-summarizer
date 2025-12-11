from pydantic_settings import BaseSettings
from functools import lru_cache


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
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

