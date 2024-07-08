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
import subprocess

COMPRESSA_API_BASE = "https://compressa-api.mil-team.ru/v1"

class ChatCompressa(BaseChatModel):
    """Chat chat model integration.

    Setup:
        Install ``langchain-compressa`` and set environment variable ``COMPRESSA_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-compressa
            export compressa="your-api-key"

    # TODO: Populate with relevant params.
    Key init args — completion params:
        model: str
            Name of Compressa model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    # TODO: Populate with relevant params.
    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            Compressa API key. If not passed in will be read from env var COMPRESSA_API_KEY.

    See full list of supported init args and their descriptions in the params section.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_my_test import ChatCompressa

            llm = ChatCompressa(
                model="...",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if token-level streaming isn't supported.
    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            # TODO: Example output.

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if native async isn't supported.
    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if .bind_tools() isn't supported.
    Tool calling:
        .. code-block:: python

            from langchain_core.pydantic_v1 import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
            ai_msg.tool_calls

        .. code-block:: python

              # TODO: Example output.

        See ``ChatCompressa.bind_tools()`` method for more.

    # TODO: Delete if .with_structured_output() isn't supported.
    Structured output:
        .. code-block:: python

            from typing import Optional

            from langchain_core.pydantic_v1 import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            # TODO: Example output.

        See ``ChatCompressa.with_structured_output()`` for more.

    # TODO: Delete if JSON mode response format isn't supported.
    JSON mode:
        .. code-block:: python

            # TODO: Replace with appropriate bind arg.
            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke("Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]")
            ai_msg.content

        .. code-block:: python

            # TODO: Example output.



    # TODO: Delete if token usage metadata isn't supported.
    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 28, 'output_tokens': 5, 'total_tokens': 33}

    # TODO: Delete if logprobs aren't supported.
    Logprobs:
        .. code-block:: python

            # TODO: Replace with appropriate bind arg.
            logprobs_llm = llm.bind(logprobs=True)
            ai_msg = logprobs_llm.invoke(messages)
            ai_msg.response_metadata["logprobs"]

        .. code-block:: python

              # TODO: Example output.

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

             # TODO: Example output.

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
 

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
    
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            subprocess.check_call(['pip', 'install', 'langchain_openai'])
            from langchain_openai import ChatOpenAI
        
        compressa_api_key = self.compressa_api_key if self.compressa_api_key else os.getenv("COMPRESSA_API_KEY" )

        if compressa_api_key is None:
            raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")

        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=COMPRESSA_API_BASE,
            api_key=compressa_api_key
        )
        
        return llm._generate(messages, stop, run_manager, **kwargs)
        

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
    
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            subprocess.check_call(['pip', 'install', 'langchain_openai'])
            from langchain_openai import ChatOpenAI
    
        compressa_api_key = self.compressa_api_key if self.compressa_api_key else os.getenv("COMPRESSA_API_KEY" )

        if compressa_api_key is None:
            raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
            
            
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=COMPRESSA_API_BASE,
            api_key=compressa_api_key
        )
    
        return llm._stream(messages, stop, run_manager, **kwargs)


    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
    
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            subprocess.check_call(['pip', 'install', 'langchain_openai'])
            from langchain_openai import ChatOpenAI
        
        compressa_api_key = self.compressa_api_key if self.compressa_api_key else os.getenv("COMPRESSA_API_KEY" )

        if compressa_api_key is None:
            raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
            
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=COMPRESSA_API_BASE,
            api_key=compressa_api_key
        )
    
        return llm._astream(messages, stop, run_manager, **kwargs)


    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
    
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            subprocess.check_call(['pip', 'install', 'langchain_openai'])
            from langchain_openai import ChatOpenAI
        
        compressa_api_key = self.compressa_api_key if self.compressa_api_key else os.getenv("COMPRESSA_API_KEY" )

        if compressa_api_key is None:
            raise Exception("status_code: None, body: The client must be instantiated be either passing in api_key or setting COMPRESSA_API_KEY")
            
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            base_url=COMPRESSA_API_BASE,
            api_key=compressa_api_key
        )
        
        return llm._agenerate(messages, stop, run_manager, **kwargs)

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
