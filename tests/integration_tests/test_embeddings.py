"""Тесты Compressa embeddings."""
from langchain_compressa.embeddings import CompressaEmbeddings
import os

os.environ["COMPRESSA_API_KEY"] = "key" #установите реальное значение перед запуском теста


def test_langchain_compressa_embedding_documents() -> None:
    """Тест compressa embeddings для документов."""
    documents = ["какой-то текст"]
    embedding = CompressaEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 0


def test_langchain_compressa_embedding_query() -> None:
    """Тест compressa embeddings для запроса."""
    document = "какой-то текст"
    embedding = CompressaEmbeddings()
    output = embedding.embed_query(document)
    assert len(output) > 0
