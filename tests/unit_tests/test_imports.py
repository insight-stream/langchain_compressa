from langchain_compressa import __all__

EXPECTED_ALL = [
    "ChatCompressa",
    "CompressaEmbeddings",
    "CompressaRerank",
    "__version__",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
