"""Test Compressa embeddings."""
from langchain_compressa.embeddings import CompressaEmbeddings
import os

os.environ["COMPRESSA_API_KEY"] = "foo"


def test_langchain_compressa_embedding_documents() -> None:
    """Test cohere embeddings."""
    documents = ["foo bar"]
    embedding = CompressaEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_compressa_embedding_query() -> None:
    """Test cohere embeddings."""
    document = "foo bar"
    embedding = CompressaEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
