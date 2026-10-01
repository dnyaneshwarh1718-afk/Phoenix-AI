from app.core.config import Settings


def test_settings_defaults():
    settings = Settings()
    assert settings.app_name == "Phoenix AI"
    assert settings.api_port == 8000
