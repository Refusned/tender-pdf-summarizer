"""HTTP-слой: приём PDF, валидация, вызов LLM, ответ."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.llm import LLMClient, LLMError
from app.pdf import (
    NoTextLayerError,
    PdfExtractionError,
    extract_pages,
    render_for_llm,
    select_pages,
)
from app.schemas import Meta, SummarizeResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    http = httpx.AsyncClient(
        base_url=settings.llm_base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        timeout=settings.llm_timeout_s,
    )
    app.state.llm = LLMClient(http, settings.llm_model)
    try:
        yield
    finally:
        await http.aclose()


app = FastAPI(
    title="Суммаризатор тендерной документации",
    version="1.0.0",
    lifespan=lifespan,
)


def get_llm() -> LLMClient:
    """Точка подмены в тестах через app.dependency_overrides."""
    return app.state.llm


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/summarize", response_model=SummarizeResponse)
async def summarize(
    file: UploadFile,
    settings: Annotated[Settings, Depends(get_settings)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> SummarizeResponse:
    started = time.monotonic()
    data = await _read_limited(file, settings.max_upload_mb)

    try:
        pages = extract_pages(data)
    except NoTextLayerError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except PdfExtractionError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Файл не читается как PDF: {exc}"
        ) from exc

    selected = select_pages(pages, settings.char_budget)
    document, pages_sent = render_for_llm(selected, settings.char_budget)

    try:
        summary = await llm.summarize(document)
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return SummarizeResponse(
        summary=summary,
        meta=Meta(
            pages_total=len(pages),
            pages_used=pages_sent,
            chars_sent=len(document),
            model=llm.model,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        ),
    )


async def _read_limited(file: UploadFile, max_mb: int) -> bytes:
    """Читаем кусками: файл на 2 ГБ не должен укладывать процесс."""
    limit = max_mb * 1024 * 1024
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Файл больше {max_mb} МБ",
            )
        chunks.append(chunk)

    if not chunks:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Пустой файл")

    data = b"".join(chunks)
    if not data.startswith(b"%PDF"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Ожидается PDF: файл не начинается с сигнатуры %PDF",
        )
    return data
