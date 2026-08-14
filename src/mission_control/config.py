from functools import lru_cache

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Mission Control application config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MISSION_CONTROL_",
        extra="ignore",
    )

    inference_base_url: AnyHttpUrl = (
        "http://127.0.0.1:8000/v1"
    )

    model_name: str = (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    )

    request_timeout_seconds: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()