from typing import List, Any, Dict, Optional
import os

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, SecretStr, BaseModel
from langchain_openai import OpenAIEmbeddings

COMPRESSA_API_BASE = "https://compressa-api.mil-team.ru/v1"


class CompressaEmbeddings(BaseModel, Embeddings):
    """CompressaEmbeddings embedding model.

    To use, you should have the
    environment variable ``COMPRESSA_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_compressa import CompressaEmbeddings

            model = CompressaEmbeddings()
    """
    model: str = "/app/resources/models/models/Salesforce_SFR-Embedding-Mistral"
    tiktoken_enabled: bool = False
    tiktoken_model_name: Optional[str] = "Salesforce/SFR-Embedding-Mistral"
    model_kwargs: Dict[str, Any] = Field(default={"encoding_format": "float"})
    compressa_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        
        compressa_api_key = self.compressa_api_key if self.compressa_api_key else os.getenv("COMPRESSA_API_KEY")
        
        if compressa_api_key is None:
                raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
            
        embed = OpenAIEmbeddings(
            model=self.model,
            openai_api_base=COMPRESSA_API_BASE,
            openai_api_key=compressa_api_key,
            model_kwargs=self.model_kwargs,
            tiktoken_enabled=self.tiktoken_enabled,
            tiktoken_model_name=self.tiktoken_model_name,
        )
        
        return embed.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        
        compressa_api_key = self.compressa_api_key if self.compressa_api_key else os.getenv("COMPRESSA_API_KEY")
        
        if compressa_api_key is None:
                raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
            
        embed = OpenAIEmbeddings(
            model=self.model,
            openai_api_base=COMPRESSA_API_BASE,
            openai_api_key=compressa_api_key,
            model_kwargs=self.model_kwargs,
            tiktoken_enabled=self.tiktoken_enabled,
            tiktoken_model_name=self.tiktoken_model_name,
        )
        
        return embed.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        
        compressa_api_key = self.compressa_api_key if self.compressa_api_key else os.getenv("COMPRESSA_API_KEY")
        
        if compressa_api_key is None:
                raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
            
        embed = OpenAIEmbeddings(
            model=self.model,
            openai_api_base=COMPRESSA_API_BASE,
            openai_api_key=compressa_api_key,
            model_kwargs=self.model_kwargs,
            tiktoken_enabled=self.tiktoken_enabled,
            tiktoken_model_name=self.tiktoken_model_name,
        )
        
        return embed.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        
        compressa_api_key = self.compressa_api_key if self.compressa_api_key else os.getenv("COMPRESSA_API_KEY")
        
        if compressa_api_key is None:
                raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
            
        embed = OpenAIEmbeddings(
            model=self.model,
            openai_api_base=COMPRESSA_API_BASE,
            openai_api_key=compressa_api_key,
            model_kwargs=self.model_kwargs,
            tiktoken_enabled=self.tiktoken_enabled,
            tiktoken_model_name=self.tiktoken_model_name,
        )
        
        return embed.aembed_query(text)
