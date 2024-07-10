from typing import Type, cast

import pytest
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_compressa import (
    ChatCompressa,
    CompressaEmbeddings,
    CompressaRerank,
)


def test_compressa_secrets() -> None:
    o = ChatCompressa(api_key="foo")
    s = str(o)
    assert "foo" not in s


def test_compressa_embeddings_secrets() -> None:
    o = CompressaEmbeddings(api_key="foo")
    s = str(o)
    assert "foo" not in s
    
def test_compressa_reranks_secrets() -> None:
    o = CompressaRerank(api_key="foo")
    s = str(o)
    assert "foo" not in s

