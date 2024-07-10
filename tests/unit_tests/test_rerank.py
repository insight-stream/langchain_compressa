"""Test reranks integration."""

from langchain_compressa import CompressaRerank


def test_initialization() -> None:
    """Test rerank initialization."""
    CompressaRerank(api_key="test")
