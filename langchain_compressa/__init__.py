from importlib import metadata

from langchain_compressa.chat_models import ChatCompressa
from langchain_compressa.embeddings import CompressaEmbeddings
from langchain_compressa.llms import CompressaLLM

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatCompressa",
    "CompressaLLM",
    "CompressaEmbeddings",
    "__version__",
]
