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
