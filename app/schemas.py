"""Схема выжимки. Модель обязана вернуть ровно эту структуру."""

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Дословная цитата из документа, подтверждающая значение.

    Без неё выжимку невозможно проверить, а LLM охотно придумывает
    правдоподобные суммы и сроки. Пустая цитата = повод не доверять полю.
    """

    page: int | None = Field(None, description="Номер страницы исходного PDF")
    quote: str = Field("", description="Цитата из документа, без пересказа")


class Money(BaseModel):
    amount: float | None = None
    currency: str = "RUB"
    vat: str | None = Field(None, description='Например: "включая НДС 20%"')
    evidence: Evidence | None = None


class Deadline(BaseModel):
    stage: str = Field(description='Этап: "выполнение работ", "гарантия", "оплата"')
    value: str = Field(description='Срок как в документе: "90 календарных дней"')
    evidence: Evidence | None = None


class Requirement(BaseModel):
    text: str
    category: str | None = Field(
        None, description="опыт | лицензия | СРО | обеспечение | персонал | прочее"
    )
    evidence: Evidence | None = None


class Penalty(BaseModel):
    reason: str = Field(description="За что начисляется")
    amount: str = Field(description='Часто формула: "0,1% цены контракта за день"')
    evidence: Evidence | None = None


class TenderSummary(BaseModel):
    contract_sum: Money | None = None
    deadlines: list[Deadline] = []
    requirements: list[Requirement] = []
    penalties: list[Penalty] = []
    unresolved: list[str] = Field(
        default=[], description="Что найти не удалось: явный список, а не молчание"
    )


class Meta(BaseModel):
    pages_total: int
    pages_used: list[int]
    chars_sent: int
    model: str
    elapsed_ms: int


class SummarizeResponse(BaseModel):
    summary: TenderSummary
    meta: Meta
