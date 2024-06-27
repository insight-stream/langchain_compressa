# langchain-compressa

This package contains the LangChain integration with Compressa

## Installation

```bash
pip install -U langchain-compressa
```

And you should configure credentials by setting the following environment variables:
COMPRESSA_API_KEY

## Embeddings

`CompressaEmbeddings` class exposes embeddings from Compressa.

```python
from langchain_compressa import CompressaEmbeddings

embeddings = CompressaEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`CompressaLLM` class exposes LLMs from Compressa.

```python
from langchain_compressa import CompressaLLM

llm = CompressaLLM()
llm.invoke("The meaning of life is")
```

## Chat model
` ChatCompressa` class exposes Chat model from Compressa.

```python
from langchain_openai import ChatCompressa

llm = ChatCompressa(
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Russian. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

ai_msg = llm.invoke(messages)
print(ai_msg)
print(ai_msg.content)

```
