# langchain-compressa

This package contains the LangChain integration with Compressa

## Installation

```bash
pip install git+https://github.com/insight-stream/langchain_compressa.git
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


## Chat model
` ChatCompressa` class exposes Chat model from Compressa.

```python
from langchain_openai import ChatCompressa

llm = ChatCompressa(
    temperature=0,    
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars    
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

## CompressaRerank

```python
from langchain_core.documents import Document
from langchain_compressa.reranks import CompressaRerank

documents = [
    Document(
        page_content="Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
        metadata={"source": "https://example.com/1"}
    ),
    Document(
        page_content="The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
        metadata={"source": "https://example.com/2"}
    ),
    Document(
        page_content="Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        metadata={"source": "https://example.com/3"}
    ),
    Document(
        page_content="Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district. The President of the USA and many major national government offices are in the territory. This makes it the political center of the United States of America.",
        metadata={"source": "https://example.com/4"}
    ), 
    Document(
        page_content="Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states. The federal government (including the United States military) also uses capital punishment.",
        metadata={"source": "https://example.com/5"}
    )
]

query = "What is the capital of the United States?"

reranker = CompressaRerank()
rerank_res = reranker.rerank(query=query,  documents=documents, top_n=3)
compress_res = reranker.compress_documents(query=query,  documents=documents)
```

## RAG example

```python
import os
from langchain_compressa import CompressaEmbeddings, ChatCompressa
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma

COMPRESSA_API_KEY = os.getenv('COMPRESSA_API_KEY')

compressa_embedding = CompressaEmbeddings(api_key=COMPRESSA_API_KEY)
llm = ChatCompressa(api_key=COMPRESSA_API_KEY)

loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=compressa_embedding)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

system_template = f"""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise."""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", """Context information:

        {context}
        
        Query: {input}		
    """),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

answ = rag_chain.invoke({"input": "how can langsmith help with testing?"})
print(answ["answer"])
```
