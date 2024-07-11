"""Тест чат моделей ChatCompressa."""
from langchain_compressa.chat_models import ChatCompressa
import os

os.environ["COMPRESSA_API_KEY"] = "key" #установите реальное значение перед запуском теста

def test_stream() -> None:
    """Тест стриминга ChatCompressa."""
    llm = ChatCompressa()

    for token in llm.stream("Я Пикл Рик"):
        assert isinstance(token.content, str)


def test_invoke() -> None:
    """Тест вызова ChatCompressa."""
    llm = ChatCompressa()

    result = llm.invoke("Я Пикл Рик")
    assert isinstance(result.content, str)
    
async def test_astream() -> None:
    """Тест асинхронного стриминга ChatCompressa."""
    llm = ChatCompressa()

    async for token in llm.astream("Я Пикл Рик"):
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Тест асинхронного вызова ChatCompressa."""
    llm = ChatCompressa()

    result = await llm.ainvoke("Я Пикл Рик")
    assert isinstance(result.content, str)
