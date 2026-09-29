"""Chroma vector store access.

One place builds the store so that ingestion and querying can never disagree
about the directory, the collection name, or the distance metric. A mismatch
in any of those three silently produces an empty result set.
"""

from typing import Any, Optional

from app.config import settings
from app.services.embeddings import COLLECTION_METADATA, get_embeddings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_vector_store(
    chroma_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Any:
    """Open (or create) the persistent Chroma collection.

    `chroma_dir` is an explicit argument rather than a direct read of the
    global setting. That single choice is what lets a test point the service
    at a temporary directory instead of polluting the real knowledge base.
    """
    from langchain_chroma import Chroma

    directory = chroma_dir or settings.chroma_dir
    collection = collection_name or settings.collection_name

    logger.debug("Opening Chroma collection %r in %r", collection, directory)
    return Chroma(
        persist_directory=directory,
        embedding_function=get_embeddings(),
        collection_name=collection,
        collection_metadata=COLLECTION_METADATA,
    )


def collection_is_empty(store: Any) -> bool:
    """True when the collection holds no documents.

    Used to tell "nothing was ingested" apart from "nothing matched", which
    are a 503 and a normal 200 fallback respectively.
    """
    try:
        probe = store.get(limit=1)
    except Exception:
        logger.exception("Could not read the vector store collection")
        return True
    return not probe.get("ids")
