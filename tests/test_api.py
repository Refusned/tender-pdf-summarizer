import pytest
from fastapi.testclient import TestClient

from app.llm import LLMError
from app.main import app, get_llm
from app.schemas import Deadline, Evidence, Money, Penalty, TenderSummary

ANSWER = TenderSummary(
    contract_sum=Money(
        amount=2450000.0,
        currency="RUB",
        vat="включая НДС 20%",
        evidence=Evidence(page=1, quote="НМЦК: 2 450 000,00 рублей"),
    ),
    deadlines=[Deadline(stage="выполнение работ", value="90 календарных дней")],
    penalties=[Penalty(reason="просрочка", amount="1/300 ключевой ставки за день")],
    unresolved=["гарантийные обязательства субподрядчиков"],
)


class FakeLLM:
    """Подмена модели: тесты API не должны зависеть от сети и ключей."""

    model = "fake-model"

    def __init__(self, result=ANSWER, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.seen_document = ""

    async def summarize(self, document: str):
        self.seen_document = document
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def use(llm: FakeLLM) -> FakeLLM:
    app.dependency_overrides[get_llm] = lambda: llm
    return llm


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_возвращает_выжимку_и_метаданные(client, sample_pdf):
    llm = use(FakeLLM())

    response = client.post(
        "/v1/summarize", files={"file": ("t.pdf", sample_pdf, "application/pdf")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["contract_sum"]["amount"] == 2450000.0
    assert body["summary"]["contract_sum"]["evidence"]["page"] == 1
    assert body["meta"]["pages_total"] == 5
    assert body["meta"]["pages_used"] == [1, 2, 3, 4, 5]
    assert "=== Страница 1 ===" in llm.seen_document


def test_не_pdf_отклоняется_до_вызова_модели(client):
    llm = use(FakeLLM())

    response = client.post(
        "/v1/summarize", files={"file": ("x.pdf", b"PK\x03\x04 zip", "application/pdf")}
    )

    assert response.status_code == 400
    assert llm.seen_document == "", "модель не должна вызываться на мусорном файле"


def test_пустой_файл_отклоняется(client):
    use(FakeLLM())

    response = client.post("/v1/summarize", files={"file": ("x.pdf", b"", "application/pdf")})

    assert response.status_code == 400


def test_недоступность_модели_отдаётся_как_502(client, sample_pdf):
    use(FakeLLM(error=LLMError("модель недоступна")))

    response = client.post(
        "/v1/summarize", files={"file": ("t.pdf", sample_pdf, "application/pdf")}
    )

    assert response.status_code == 502
    assert "модель недоступна" in response.json()["detail"]
