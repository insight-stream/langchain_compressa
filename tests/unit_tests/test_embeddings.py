"""Test embedding model integration."""
import os
import pytest
from langchain_compressa.embeddings import CompressaEmbeddings

os.environ["COMPRESSA_API_KEY"] = "foo"

def test_initialization() -> None:
    """Test embedding model initialization."""
    embeddings =CompressaEmbeddings()

def test_compressa_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        CompressaEmbeddings(model_kwargs={"model": "foo"})

def test_compressa_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = CompressaEmbeddings(foo="bar")
    assert llm.model_kwargs == {"encoding_format": "float", "foo": "bar"}
