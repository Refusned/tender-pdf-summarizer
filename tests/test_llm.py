import httpx
import pytest

from app.llm import LLMClient, LLMError

VALID = (
    '{"contract_sum": {"amount": 2450000.0, "currency": "RUB"}, '
    '"deadlines": [], "requirements": [], "penalties": [], "unresolved": []}'
)


def make_client(handler) -> LLMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://llm/v1")
    return LLMClient(http, "test-model")


def reply(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})


async def test_разбирает_валидный_ответ():
    client = make_client(lambda request: reply(VALID))

    summary = await client.summarize("документ")

    assert summary.contract_sum.amount == 2450000.0


async def test_чинит_невалидный_json_второй_попыткой():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return reply('{"contract_sum": "две тысячи"}' if len(calls) == 1 else VALID)

    summary = await make_client(handler).summarize("документ")

    assert len(calls) == 2, "ожидали ровно одну попытку починки"
    assert summary.contract_sum.amount == 2450000.0


async def test_после_двух_неудач_падает_понятной_ошибкой():
    client = make_client(lambda request: reply("совсем не json"))

    with pytest.raises(LLMError, match="валидный JSON"):
        await client.summarize("документ")


async def test_ошибка_апстрима_не_протекает_наружу():
    client = make_client(lambda request: httpx.Response(500, text="upstream is down"))

    with pytest.raises(LLMError, match="500"):
        await client.summarize("документ")
