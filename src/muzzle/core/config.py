"""
Core configuration settings for Muzzle API.

This module handles all application configuration using Pydantic settings
which automatically loads from environment variables and .env files.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator
from functools import lru_cache
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Pydantic automatically loads values from:
    1. Environment variables
    2. .env file
    3. Default values defined here
    """

    # =============================================================================
    # Application Settings
    # =============================================================================
    app_name: str = "Muzzle"
    app_version: str = "0.1.0"
    debug: bool = True
    log_level: str = "INFO"

    # API Configuration
    api_v1_prefix: str = "/api/v1"
    allowed_hosts: List[str] = ["localhost", "127.0.0.1", "testserver"]

    # =============================================================================
    # Security Settings
    # =============================================================================
    secret_key: str = "dev-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"

    # =============================================================================
    # Database Configuration
    # =============================================================================
    database_url: str = "sqlite:///./muzzle.db"

    # =============================================================================
    # Redis Configuration
    # =============================================================================
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # =============================================================================
    # Machine Learning Settings
    # =============================================================================
    model_artifacts_path: str = "./artifacts"
    huggingface_cache_dir: str = "./artifacts/huggingface_cache"
    text_model_name: str = "distilbert-base-uncased"
    nsfw_model_name: str = "Falconsai/nsfw_image_detection"

    # Model thresholds
    toxicity_threshold: float = 0.7
    nsfw_threshold: float = 0.8

    # =============================================================================
    # API Rate Limiting
    # =============================================================================
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 10

    # =============================================================================
    # External Services
    # =============================================================================
    deepai_api_key: Optional[str] = None
    sentry_dsn: Optional[str] = None

    # =============================================================================
    # Development Settings
    # =============================================================================
    enable_docs: bool = True
    enable_redoc: bool = True
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    @validator("allowed_hosts", pre=True)
    def parse_hosts(cls, v):
        """Parse comma-separated hosts from environment variable."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("model_artifacts_path")
    def create_artifacts_dir(cls, v):
        """Ensure the artifacts directory exists."""
        os.makedirs(v, exist_ok=True)
        return v

    @validator("huggingface_cache_dir")
    def create_cache_dir(cls, v):
        """Ensure the HuggingFace cache directory exists."""
        os.makedirs(v, exist_ok=True)
        return v

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    The @lru_cache decorator ensures we only create one Settings instance
    and reuse it throughout the application lifecycle.

    Returns:
        Settings: Cached settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()