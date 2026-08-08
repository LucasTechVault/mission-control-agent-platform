from mission_control.config import Settings


def test_default_config() -> None:
    settings = Settings()
    
    assert settings.model_name == "Qwen/Qwen3.6-27B"
    assert settings.request_timeout_seconds == 60.0