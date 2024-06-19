"""Test Compressa Chat API wrapper."""
import os
from langchain_compressa import CompressaLLM

os.environ["COMPRESSA_API_KEY"] = "foo"


def test_initialization() -> None:
    """Test integration initialization."""
    CompressaLLM()

def test_compressa_model_param() -> None:
    llm = CompressaLLM(model="foo")
    assert llm.model_name == "foo"
    llm = CompressaLLM(model_name="foo")
    assert llm.model_name == "foo"


def test_compressa_model_kwargs() -> None:
    llm = CompressaLLM(model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}
