from langchain_core.prompts import PromptTemplate
from langchain_compressa.llms import CompressaLLM
import os


os.environ["COMPRESSA_API_KEY"] = "your_key_here"


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

model = CompressaLLM()

chain = prompt | model

print(chain.invoke({"question": "What is LangChain?"}))



from langchain_compressa.embeddings import CompressaEmbeddings

embeddings = CompressaEmbeddings()

import asyncio

async def my_function():
    emb = await embeddings.aembed_documents(["This is a content of the document", "This is another document"])
    print(emb[0][:10])

asyncio.run(my_function())


print("************")
print(embeddings.embed_query("My query to look up")[:10])


print(model.invoke("The meaning of life is"))


print("\n Тест чат модели")
from langchain_compressa.chat_models import ChatCompressa

llm = ChatCompressa(
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

print(llm.model_name)


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to Russian. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)

print("\n")
print(ai_msg.content)
