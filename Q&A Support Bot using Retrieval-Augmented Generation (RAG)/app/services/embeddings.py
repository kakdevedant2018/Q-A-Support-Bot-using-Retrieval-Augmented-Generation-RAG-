"""Local embedding model factory.

The model runs on the CPU inside this process. There is no embedding API call
and no cost. The first call downloads roughly 90 MB of model weights and then
caches them, so the first run is slow and every run after it is fast.

Heavy imports are deliberately kept inside the function. That keeps
`import app.main` cheap, lets the mocked test suite run without torch
installed, and means a broken native dependency cannot stop the API from
importing.
"""

from functools import lru_cache
from typing import Any

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Distance metric for the Chroma collection. Cosine plus normalised vectors
# gives a relevance score in [0, 1] that is stable enough to threshold on.
# With the default L2 metric the numbers are unbounded and a fixed threshold
# is meaningless.
DISTANCE_METRIC = "cosine"
COLLECTION_METADATA = {"hnsw:space": DISTANCE_METRIC}


def _import_embeddings_class() -> Any:
    """Import HuggingFaceEmbeddings from whichever package provides it.

    LangChain moved this class out of `langchain_community` into the dedicated
    `langchain_huggingface` package. The guide this project follows uses the
    old path, and it warns that these imports drift between versions. Trying
    the current location first and falling back keeps the project working
    across both layouts.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings
    except ImportError:
        logger.warning(
            "langchain_huggingface not installed; falling back to the "
            "deprecated langchain_community import path"
        )
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> Any:
    """Return a cached embedding model.

    Cached because loading the weights costs seconds. Ingestion and the query
    path share one instance inside a process.
    """
    embeddings_cls = _import_embeddings_class()
    logger.info("Loading local embedding model: %s", settings.embedding_model)
    return embeddings_cls(
        model_name=settings.embedding_model,
        # Unit-length vectors are what make cosine distance behave as a
        # similarity in [0, 1].
        encode_kwargs={"normalize_embeddings": True},
    )
