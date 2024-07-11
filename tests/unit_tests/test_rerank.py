"""Тест reranks интеграции."""

from langchain_compressa import CompressaRerank


def test_initialization() -> None:
    """Тест инициализации rerank."""
    CompressaRerank(api_key="test")
