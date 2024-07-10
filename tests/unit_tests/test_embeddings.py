"""Test embedding model integration."""
import os
import pytest
from langchain_compressa.embeddings import CompressaEmbeddings

os.environ["COMPRESSA_API_KEY"] = "foo"

def test_initialization() -> None:
    """Test embedding model initialization."""
    embeddings = CompressaEmbeddings()

def test_compressa_default_model_kwargs() -> None:
    embed = CompressaEmbeddings()
    assert embed.model_kwargs == {"encoding_format": "float"}
