"""Compressa chat models."""

from typing import Any, List, Optional, Dict, Iterator, AsyncIterator
import os

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_openai import ChatOpenAI
import subprocess

COMPRESSA_API_BASE = "https://compressa-api.mil-team.ru/v1"

class ChatCompressa(BaseChatModel):
    """Chat chat model integration.

    Setup:
        Install ``langchain_compressa`` and set environment variable ``COMPRESSA_API_KEY``.

        .. code-block:: bash

            pip install pip install git+https://github.com/insight-stream/langchain_compressa.git
            export COMPRESSA_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Compressa model to use.
        temperature: float
            Sampling temperature.

    Key init args — client params:
        api_key: Optional[str]
            Compressa API key. If not passed in will be read from env var COMPRESSA_API_KEY.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_my_test import ChatCompressa

            llm = ChatCompressa(
                model="...",
                temperature=0,
                # api_key="...",
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full
   
    """ 
    
    model_name: str = Field(default="/app/resources/models/models/compressa-ai_Llama-3-8B-Instruct", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    compressa_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `COMPRESSA_API_KEY` if not provided."""
    streaming: bool = False
    """Whether to stream the results or not."""
    client: Any = Field(default=None, exclude=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.compressa_api_key = self.compressa_api_key if self.compressa_api_key else os.getenv("COMPRESSA_API_KEY")
        
        if self.compressa_api_key is None:
            raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
        self.client = self._create_client()
        
    def _create_client(self) -> Any:
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=COMPRESSA_API_BASE,
            api_key=self.compressa_api_key
        )
        return llm

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        return self.client._generate(messages, stop, run_manager, **kwargs)
        

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
    
        return self.client._stream(messages, stop, run_manager, **kwargs)


    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        
        async for chunk in self.client._astream(messages, stop, run_manager, **kwargs):
            yield chunk


    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        return await self.client._agenerate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-compressa"
        
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Compressa API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "temperature": self.temperature,
            **self.model_kwargs,
        }
        return params
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name, **self._default_params}
