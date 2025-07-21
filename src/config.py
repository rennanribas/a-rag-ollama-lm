"""Configuration management for AI RAG Agent."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # LLM Provider Configuration (free alternatives first)
    llm_provider: str = Field("gemini", env="LLM_PROVIDER")  # gemini, ollama, openai
    
    # Gemini Configuration (FREE)
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", env="GEMINI_MODEL")
    
    # Ollama Configuration (FREE - Local)
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.1:8b", env="OLLAMA_MODEL")
    
    # OpenAI Configuration (PAID - Fallback)
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4-turbo-preview", env="OPENAI_MODEL")
    
    # Embedding Configuration
    embedding_provider: str = Field("huggingface", env="EMBEDDING_PROVIDER")  # huggingface, gemini, ollama, openai
    embedding_model: str = Field("BAAI/bge-small-en-v1.5", env="EMBEDDING_MODEL")
    
    # Vector Store Configuration
    chroma_persist_directory: Path = Field(
        Path("./data/chroma_db"), env="CHROMA_PERSIST_DIRECTORY"
    )
    
    # Crawler Configuration
    crawl_delay: float = Field(1.0, env="CRAWL_DELAY")
    max_concurrent_requests: int = Field(5, env="MAX_CONCURRENT_REQUESTS")
    user_agent: str = Field("AI-RAG-Agent/1.0", env="USER_AGENT")
    verify_ssl: bool = Field(False, env="VERIFY_SSL")  # Set to False to bypass SSL verification
    
    # API Configuration
    api_host: str = Field("localhost", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_reload: bool = Field(True, env="API_RELOAD")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(Path("./logs/agent.log"), env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.chroma_persist_directory.parent.mkdir(parents=True, exist_ok=True)
if settings.log_file:
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)