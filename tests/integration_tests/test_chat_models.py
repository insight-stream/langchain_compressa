"""Test ChatCompressa chat model."""
from langchain_compressa.chat_models import ChatCompressa
import os

os.environ["COMPRESSA_API_KEY"] = "key" #set real value before run tests

def test_stream() -> None:
    """Test streaming tokens from ChatCompressa."""
    llm = ChatCompressa()

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


def test_invoke() -> None:
    """Test invoke tokens from ChatCompressa."""
    llm = ChatCompressa()

    result = llm.invoke("I'm Pickle Rick")
    assert isinstance(result.content, str)
    
async def test_astream() -> None:
    """Test streaming tokens from ChatCompressa."""
    llm = ChatCompressa()

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_ainvoke() -> None:
    """Test invoke tokens from ChatCompressa."""
    llm = ChatCompressa()

    result = await llm.ainvoke("I'm Pickle Rick")
    assert isinstance(result.content, str)
