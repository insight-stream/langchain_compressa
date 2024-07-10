"""
Test CompressaRerank.
"""

import pytest
import os
from langchain_core.documents import Document

from langchain_compressa import CompressaRerank

os.environ["COMPRESSA_API_KEY"] = "key" #set real value before run tests

def test_langchain_cohere_rerank_documents() -> None:
    reranker = CompressaRerank()
    test_documents = [
        Document(page_content="This is a test document."),
        Document(page_content="Another test document."),
    ]
    test_query = "Test query"
    results = reranker.rerank(test_documents, test_query)
    assert len(results) == 2
