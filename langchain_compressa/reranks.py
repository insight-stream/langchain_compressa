from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.pydantic_v1 import Field, SecretStr

import os
import typing
import requests

_RerankRequestDocumentsItem = typing.Union[str, typing.Dict]

class _CompressaClient:
    """

    Parameters
    ----------
    base_url : typing.Optional[str]
        The base url to use for requests from the client.

    compressa_api_key : typing.Optional[str]
    """

    def __init__(
        self,
        *,
        base_url: typing.Optional[str] = os.getenv("COMPRESSA_BASE_URL", "https://compressa-api.mil-team.ru/v1"),
        compressa_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    ):
        
        self.compressa_api_key = compressa_api_key.get_secret_value() if compressa_api_key else os.getenv("COMPRESSA_API_KEY")
        if self.compressa_api_key is None:
            raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
        self.base_url = base_url

    def _rerank(
        self,
        *,
        query: str,
        documents: typing.Sequence[_RerankRequestDocumentsItem],
        model: typing.Optional[str] = "mixedbread-ai/mxbai-rerank-large-v1",
        top_n: typing.Optional[int] = 5,
        return_documents: typing.Optional[bool] = False
    ) -> any:  #TODO: обработать ответ RerankResponse 
        
        headers = {
            "Content-Type": "application/json",
            "X-Fern-Language": "Python",
            "Authorization": "Bearer " + self.compressa_api_key
        }
           
        jsonBody = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
            }
	    
        _response = requests.post(f"{self.base_url}/rerank", headers=headers, json=jsonBody)

        if _response.status_code == 200:
            return _response.json() 
        else:
            raise Exception(f"status_code: {_response.status_code}, body: {_response.text}")


class CompressaRerank(BaseDocumentCompressor):
    """Document compressor that uses `Compressa Rerank API`."""

    top_n: Optional[int] = 3
    """Number of documents to return."""
    model: str = "mixedbread-ai/mxbai-rerank-large-v1"
    """Model to use for reranking."""
    compressa_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Compressa API key. Must be specified directly or via environment variable 
        COMPRESSA_API_KEY."""

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        model: Optional[str] = None,
        top_n: Optional[int] = -1,
    ) -> List[Dict[str, Any]]:
        """Returns an ordered list of documents ordered by their relevance to the provided query.

        Args:
            query: The query to use for reranking.
            documents: A sequence of documents to rerank.
            model: The model to use for re-ranking. Default to self.model.
            top_n : The number of results to return. If None returns all results.
                Defaults to self.top_n.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = model or self.model
        
        client = _CompressaClient(compressa_api_key=self.compressa_api_key)
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        results = client._rerank(
            query=query,
            documents=docs,
            model=model,
            top_n=top_n,
        )
        result_dicts = []
        for res in results["results"]:
            result_dicts.append(
                {"index": res["index"], "relevance_score": res["relevance_score"]}
            )
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using Compressa's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed
