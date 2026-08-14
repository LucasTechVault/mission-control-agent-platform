from functools import lru_cache

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Mission Control application config"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MISSION_CONTROL_",
        extra="ignore",
    )
    
    inference_base_url: AnyHttpUrl = "http://127.0.0.1:8000/v1"
    #     inference_base_url: AnyHttpUrl = "http://127.0.0.1:18001/v1" for remote

    model_name: str = Field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        alias="VLLM_MODEL_NAME"
    )
    
    request_timeout_seconds: float = Field(
        default=60.0,
        alias="VLLM_REQUEST_TIMEOUT"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()