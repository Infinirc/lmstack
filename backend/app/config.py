"""Application configuration"""

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    app_name: str = "LMStack"
    debug: bool = False

    # Database - use data directory for Docker compatibility
    database_url: str = "sqlite+aiosqlite:///./data/lmstack.db"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS - comma-separated list of allowed origins, or "*" for development
    # Example: "http://localhost:3000,http://localhost:5173,https://your-domain.com"
    cors_origins: str = "*"

    # Authentication
    secret_key: str = secrets.token_hex(32)  # JWT signing key
    access_token_expire_minutes: int = 1440  # 24 hours

    # Worker settings
    worker_heartbeat_interval: int = 30  # seconds
    worker_timeout: int = 90  # seconds to consider worker offline

    # vLLM defaults
    vllm_default_image: str = "vllm/vllm-openai:latest"

    # SGLang defaults
    sglang_default_image: str = "lmsysorg/sglang:latest"

    # Ollama defaults
    ollama_default_image: str = "ollama/ollama:latest"
    ollama_host_port: int = 11434  # Default Ollama API port

    # Model cache directory (for HuggingFace models)
    hf_cache_dir: str = "/root/.cache/huggingface"

    # Data directory
    data_dir: Path = Path("./data")

    def get_cors_origins(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    class Config:
        env_prefix = "LMSTACK_"
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
