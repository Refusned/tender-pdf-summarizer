"""Извлечение текста из PDF и отбор страниц, которые реально стоит показать модели."""

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Ниже этого порога считаем, что текстового слоя нет: скан или картинки.
MIN_TEXT_CHARS = 200

# Маркеры разделов, из которых состоит ответ. Вес грубый и намеренно простой:
# задача не «угадать точно», а не потерять страницу со штрафами в 200-страничном файле.
SECTION_MARKERS: dict[str, tuple[str, ...]] = {
    "sum": (
        "начальная (максимальная) цена",
        "нмцк",
        "цена контракта",
        "цена договора",
        "стоимость работ",
        "источник финансирования",
    ),
    "deadline": (
        "срок выполнения",
        "срок оказания",
        "срок поставки",
        "сроки выполнения",
        "календарных дней",
        "период выполнения",
        "гарантийный срок",
    ),
    "requirements": (
        "требования к участник",
        "требования к исполнител",
        "квалификац",
        "лицензи",
        "свидетельство сро",
        "опыт выполнения",
        "обеспечение исполнения",
    ),
    "penalty": (
        "штраф",
        "пени",
        "неустойк",
        "ответственность сторон",
        "просрочк",
    ),
}


class PdfExtractionError(Exception):
    """Файл не PDF или повреждён."""


class NoTextLayerError(Exception):
    """PDF без текстового слоя: скан. Нужен OCR, которого здесь нет."""


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, как в просмотрщике
    text: str

    @property
    def score(self) -> int:
        low = self.text.lower()
        return sum(low.count(marker) for markers in SECTION_MARKERS.values() for marker in markers)


def extract_pages(data: bytes) -> list[Page]:
    """Достаёт текст постранично. Пустые страницы отбрасываются."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [
            Page(number=i, text=_normalize(page.extract_text() or ""))
            for i, page in enumerate(reader.pages, start=1)
        ]
    except (PdfReadError, OSError, ValueError) as exc:
        raise PdfExtractionError(str(exc)) from exc

    filled = [p for p in pages if p.text]
    if sum(len(p.text) for p in filled) < MIN_TEXT_CHARS:
        raise NoTextLayerError(
            "В PDF нет текстового слоя (вероятно, скан). "
            "Нужен OCR: в этой версии не поддерживается."
        )
    return filled


def select_pages(pages: list[Page], char_budget: int) -> list[Page]:
    """Отбирает страницы под бюджет символов.

    Первые две страницы берём всегда: в извещении и на титуле проекта контракта
    почти всегда стоят предмет закупки и НМЦК. Остальные - по числу доменных
    маркеров. Порядок восстанавливаем по номеру страницы, чтобы модель читала
    документ как документ, а не как мешок кусков.
    """
    if not pages:
        return []

    chosen: dict[int, Page] = {}
    used = 0

    for page in pages[:2]:
        chosen[page.number] = page
        used += len(page.text)

    for page in sorted(pages, key=lambda p: (-p.score, p.number)):
        if page.number in chosen or page.score == 0:
            continue
        if used + len(page.text) > char_budget:
            continue
        chosen[page.number] = page
        used += len(page.text)

    return [chosen[n] for n in sorted(chosen)]


def render_for_llm(pages: list[Page], char_budget: int) -> tuple[str, list[int]]:
    """Склеивает отобранные страницы с номерами: по ним модель проставляет page в цитатах.

    Возвращает текст и номера реально вошедших страниц. select_pages берёт
    первые две страницы не глядя на бюджет, поэтому здесь страница может и не
    поместиться: список нужен, чтобы meta.pages_used не приписывал модели
    страницы, которых она не видела.
    """
    out: list[str] = []
    numbers: list[int] = []
    used = 0
    for page in pages:
        head = f"\n=== Страница {page.number} ===\n"
        room = char_budget - used - len(head)
        if room <= 0:
            break
        body = page.text[:room]
        out.append(head + body)
        numbers.append(page.number)
        used += len(head) + len(body)
    return "".join(out).strip(), numbers


def _normalize(text: str) -> str:
    # Переносы строк внутри абзацев и разрядка пробелами ломают поиск маркеров.
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
