"""Test chat model integration."""


from langchain_compressa.chat_models import ChatCompressa


def test_initialization() -> None:
    """Test chat model initialization."""
    ChatCompressa()
