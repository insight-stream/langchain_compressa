from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

import os
import typing
import requests

RerankRequestDocumentsItem = typing.Union[str, typing.Dict]

class ApiError(Exception):
    status_code: typing.Optional[int]
    body: typing.Any

    def __init__(self, *, status_code: typing.Optional[int] = None, body: typing.Any = None):
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:
        return f"status_code: {self.status_code}, body: {self.body}"


class CompressaClient:
    """

    Parameters
    ----------
    base_url : typing.Optional[str]
        The base url to use for requests from the client.

    api_key : typing.Optional[str]
    
    Examples
    --------
    import CompressaReranker

    client = CompressaReranker(
        api_key="YOUR_API_KEY",
        base_url="YOUR_BASE_URL"
    )
    """

    def __init__(
        self,
        *,
        base_url: typing.Optional[str] = os.getenv("COMPRESSA_BASE_URL", "https://compressa-api.mil-team.ru/v1"),
        api_key : typing.Optional[str] = os.getenv("COMPRESSA_API_KEY"),
    ):
        if api_key is None:
            raise ApiError(body="The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
        self._api_key = api_key
        self._base_url = base_url

    def rerank(
        self,
        *,
        query: str,
        documents: typing.Sequence[RerankRequestDocumentsItem],
        model: typing.Optional[str] = "mixedbread-ai/mxbai-rerank-large-v1",
        top_n: typing.Optional[int] = 5,
        return_documents: typing.Optional[bool] = False
    ) -> any:  #TODO: обработать ответ RerankResponse 
        """
        This endpoint takes in a query and a list of texts and produces an ordered array with each text assigned a relevance score.

        Parameters
        ----------
        query : str
            The search query

        documents : typing.Sequence[RerankRequestDocumentsItem]
            A list of document objects or strings to rerank.
            If a document is provided the text fields is required and all other fields will be preserved in the response.

            The total max chunks must be less than 10000.

            We recommend a maximum of 1,000 documents for optimal endpoint performance.

        model : typing.Optional[str]
            The identifier of the model to use

        top_n : typing.Optional[int]
            The number of most relevant documents or indices to return, defaults to the length of the documents

        return_documents : typing.Optional[bool]
            - If false, returns results without the doc text - the api will return a list of {index, relevance score} where index is inferred from the list passed into the request.
            - If true, returns results with the doc text passed in - the api will return an ordered list of {index, text, relevance score} where index + text refers to the list passed into the request.


        request_options : typing.Optional[RequestOptions]
            Request-specific configuration.

        Returns
        -------
        RerankResponse
            OK

        Examples
        --------
        from compressaRerenker import CompressaRerenker

        client = CompressaRerenker(
            api_key="YOUR_KEY",
        )
        client.rerank(
            model="mixedbread-ai/mxbai-rerank-large-v1",
            query="What is the capital of the United States?",
            documents=[
                "Carson City is the capital city of the American state of Nevada.",
                "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
                "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
                "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
            ],
        )
        """
        headers = {
            "Content-Type": "application/json",
            "X-Fern-Language": "Python",
            "Authorization": "Bearer " + self._api_key
        }
           
        jsonBody = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
            }
	    
        _response = requests.post(f"{self._base_url}/rerank", headers=headers, json=jsonBody)

        if _response.status_code == 200:
            return _response.json() 
        else:
            raise ApiError(status_code=_response.status_code, body=_response.text)


class CompressaRerank(BaseDocumentCompressor):
    """Document compressor that uses `Compressa Rerank API`."""

    client: Any = None
    """Compressa client to use for compressing documents."""
    top_n: Optional[int] = 3
    """Number of documents to return."""
    model: str = "mixedbread-ai/mxbai-rerank-large-v1"
    """Model to use for reranking."""
    compressa_api_key: Optional[str] = None
    """Compressa API key. Must be specified directly or via environment variable 
        COMPRESSA_API_KEY."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if not values.get("client"):
            cohere_api_key = get_from_dict_or_env(
                values, "compressa_api_key", "COMPRESSA_API_KEY"
            )
            values["client"] = CompressaClient(api_key = cohere_api_key)
        return values

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
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        results = self.client.rerank(
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
