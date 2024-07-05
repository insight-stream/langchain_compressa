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


class CompressaReranker:
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

            The total max chunks (length of documents * max_chunks_per_doc) must be less than 10000.

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
            "documents": documents, #[doc["pageContent"] for doc in documents],
            "top_n": top_n,
            "return_documents": return_documents,
            }
	    
        _response = requests.post(f"{self._base_url}/rerank", headers=headers, json=jsonBody)

        if _response.status_code == 200:
            return _response.json() 
        else:
            raise ApiError(status_code=_response.status_code, body=_response.text)
