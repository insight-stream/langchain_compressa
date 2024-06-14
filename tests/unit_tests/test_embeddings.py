"""Test embedding model integration."""


from langchain_compressa.embeddings import CompressaEmbeddings


def test_initialization() -> None:
    """Test embedding model initialization."""
    CompressaEmbeddings()
