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
    
    
@pytest.mark.parametrize("model_class", [ChatCompressa, CompressaEmbeddings, CompressaRerank])
def test_compressa_api_key_is_secret_string(model_class: Type) -> None:
    """Test that the API key is stored as a SecretStr."""
    model = model_class(api_key="secret-api-key")
    assert isinstance(model.compressa_api_key, SecretStr)


@pytest.mark.parametrize("model_class", [ChatCompressa, CompressaEmbeddings, CompressaRerank])
def test_compressa_api_key_masked_when_passed_via_constructor(
    model_class: Type,
    capsys: CaptureFixture,
) -> None:
    """Test that the API key is masked when passed via the constructor."""
    model = model_class(api_key="secret-api-key")
    print(model.compressa_api_key, end="")
    captured = capsys.readouterr()

    assert captured.out == "**********"


@pytest.mark.parametrize("model_class", [ChatCompressa, CompressaEmbeddings, CompressaRerank])
def test_compressa_uses_actual_secret_value_from_secretstr(model_class: Type) -> None:
    """Test that the actual secret value is correctly retrieved."""
    model = model_class(api_key="secret-api-key")
    assert cast(SecretStr, model.compressa_api_key).get_secret_value() == "secret-api-key"

