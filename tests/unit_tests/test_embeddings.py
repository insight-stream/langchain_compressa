"""Тест интеграции embedding моделей"""
import os
from langchain_compressa.embeddings import CompressaEmbeddings

os.environ["COMPRESSA_API_KEY"] = "foo"

def test_initialization() -> None:
    """Тест инициализации embedding моделей"""
    embeddings = CompressaEmbeddings()

def test_compressa_default_model_kwargs() -> None:
    """Тест проверки значения по умолчанию для model_kwargs"""
    embed = CompressaEmbeddings()
    assert embed.model_kwargs == {"encoding_format": "float"}
