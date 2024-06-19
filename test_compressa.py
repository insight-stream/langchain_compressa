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
