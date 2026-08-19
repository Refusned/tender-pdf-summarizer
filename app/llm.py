"""Клиент LLM. Один код на всех: OpenAI, Ollama, vLLM - у них общий /v1 API."""

import json
import logging

import httpx
from pydantic import ValidationError

from app.prompt import build_messages
from app.schemas import TenderSummary

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Модель недоступна или вернула то, что не разбирается в схему."""


class LLMClient:
    def __init__(self, http: httpx.AsyncClient, model: str) -> None:
        self._http = http
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def summarize(self, document: str) -> TenderSummary:
        """Один вызов плюс одна попытка починки.

        Даже модели с JSON-режимом иногда отдают JSON, не совпадающий со схемой.
        Вместо своего парсера-костыля отдаём модели её же ошибку валидации:
        со второго раза схема сходится почти всегда.
        """
        messages = build_messages(document)
        last_error = ""

        for attempt in (1, 2):
            raw = await self._chat(messages)
            try:
                return TenderSummary.model_validate_json(raw)
            except ValidationError as exc:
                last_error = str(exc)
                log.warning("Ответ не прошёл валидацию (попытка %s): %s", attempt, last_error[:300])
                messages = messages + [
                    {"role": "assistant", "content": raw[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "Ответ не прошёл валидацию:\n"
                            f"{last_error[:1500]}\n\n"
                            "Верни исправленный JSON строго по схеме, без комментариев."
                        ),
                    },
                ]

        raise LLMError(f"Модель не вернула валидный JSON: {last_error[:300]}")

    async def _chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        try:
            response = await self._http.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"LLM вернула {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, IndexError) as exc:
            raise LLMError(f"Не удалось получить ответ LLM: {exc}") from exc
