"""
Тест CompressaRerank.
"""

import pytest
import os
from langchain_core.documents import Document

from langchain_compressa import CompressaRerank

os.environ["COMPRESSA_API_KEY"] = "key" #установите реальное значение перед запуском теста

def test_langchain_cohere_rerank_documents() -> None:
    reranker = CompressaRerank()
    test_documents = [
        Document(page_content="Это текст документа."),
        Document(page_content="Это текст другого документа."),
    ]
    test_query = "Тестовый вопрос"
    results = reranker.compress_documents(test_documents, test_query)
    assert len(results) == 2
