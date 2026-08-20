"""Конфигурация сервиса. Все значения переопределяются через переменные окружения."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Любой OpenAI-совместимый endpoint: OpenAI, Ollama (/v1), vLLM, локальный прокси.
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_timeout_s: float = 120.0

    # Сколько символов документа уходит в модель за один вызов.
    # Тендерная документация бывает на 200 страниц, в окно модели она не влезает,
    # поэтому страницы отбираются по релевантности (app/pdf.py).
    char_budget: int = 24_000

    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
