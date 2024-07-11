"""Тест интеграции чат моделей."""


from langchain_compressa.chat_models import ChatCompressa


def test_initialization() -> None:
    """Тест инициализации чат моделей."""
    ChatCompressa()
