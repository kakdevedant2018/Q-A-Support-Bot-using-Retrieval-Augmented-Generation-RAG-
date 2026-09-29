"""Offline ingestion: documents to a persistent vector index.

Run this once, and again whenever the documents change:

    python -m app.ingestion.ingest
    python -m app.ingestion.ingest --reset     # rebuild from scratch

Ingestion is kept entirely separate from the query path. The API never
re-embeds the knowledge base on startup or per request, so request latency
stays low and a restart is cheap.

Re-running is safe. The reference version in the guide calls
`Chroma.from_documents` every time, which appends and leaves the collection
full of duplicates. Here each chunk gets a stable id derived from its source
file and position, so a second run replaces chunks in place. Chunks that no
longer exist because a document shrank are deleted.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.services.vector_store import get_vector_store
from app.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf", ".csv")


def _load_file(path: Path) -> List[Any]:
    """Load one file into LangChain documents, dispatching on extension."""
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(path), encoding="utf-8").load()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()

    if suffix == ".csv":
        from langchain_community.document_loaders import CSVLoader

        return CSVLoader(str(path), encoding="utf-8").load()

    raise ValueError(f"Unsupported file type: {path.name}")


def load_documents(documents_dir: Path) -> List[Any]:
    """Load every supported document in a directory tree."""
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    documents: List[Any] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            loaded = _load_file(path)
        except Exception:
            logger.exception("Skipping %s: it could not be loaded", path.name)
            continue

        for doc in loaded:
            # Record the origin so retrieved answers can cite a real file and
            # so stale chunks can be found later by source.
            doc.metadata = {**(doc.metadata or {}), "source": path.name}
        documents.extend(loaded)
        logger.info("Loaded %s (%d document(s))", path.name, len(loaded))

    return documents


# A blank line is where these documents change subject: FAQ entries are numbered
# and separated by one, policy sections are titled and separated by one. Any
# boundary measured in characters instead is arbitrary.
BLOCK_SEPARATOR = re.compile(r"\n[ \t]*\n")

# A document or section heading: one short line that does not end a sentence.
#
# The punctuation test is doing real work. Length and line count alone classified
# "Body paragraph one." as a heading and glued it onto the following topic, which
# is the dilution bug arriving by another route - a corpus of short one-line facts
# would have been merged in pairs. A heading does not terminate a sentence; a
# short paragraph does. Trailing ":" stays heading-like on purpose.
MAX_HEADING_CHARS = 60
_SENTENCE_ENDINGS = (".", "!", "?")


def _looks_like_heading(block: str) -> bool:
    return (
        len(block) <= MAX_HEADING_CHARS
        and "\n" not in block
        and not block.endswith(_SENTENCE_ENDINGS)
    )


def split_text_into_topics(text: str) -> List[str]:
    """Split one document on its own topic boundaries.

    One chunk should be about one thing, because a chunk is embedded as a single
    vector and that vector is the *average* of everything in it. Packing text to
    a character budget instead merges unrelated subjects: a 500-character window
    over this FAQ swallows the return policy, order tracking and contact details
    into one embedding, so a question about any one of them matches a blend of
    all three and scores far lower than it should. That was measured, not
    assumed - "How can I contact support?" ranked the chunk containing the answer
    third, below a chunk that only gives opening hours.

    Size-based splitting is kept as a fallback for a topic too long to embed well
    alone. `chunk_overlap` applies only there: overlap exists to rescue a
    sentence cut in half, and a blank-line boundary does not cut sentences.
    """
    blocks = [b.strip() for b in BLOCK_SEPARATOR.split(text) if b.strip()]

    # Attach a bare heading to the block it introduces. Left alone it would be a
    # content-free chunk, which is worse than useless: it can still be retrieved.
    topics: List[str] = []
    heading = ""
    for position, block in enumerate(blocks):
        is_last = position == len(blocks) - 1
        if not is_last and _looks_like_heading(block):
            heading = f"{heading}\n{block}" if heading else block
            continue
        topics.append(f"{heading}\n{block}" if heading else block)
        heading = ""
    if heading:
        topics.append(heading)

    oversized = [t for t in topics if len(t) > settings.chunk_size]
    if not oversized:
        return topics

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    split: List[str] = []
    for topic in topics:
        if len(topic) > settings.chunk_size:
            split.extend(splitter.split_text(topic))
        else:
            split.append(topic)
    return split


def split_documents(documents: List[Any]) -> List[Any]:
    """Split every document into one-topic chunks, numbered per source file."""
    from langchain_core.documents import Document

    chunks: List[Any] = []
    per_source: Dict[str, int] = {}

    for doc in documents:
        metadata = dict(doc.metadata or {})
        source = metadata.get("source", "unknown")

        for topic in split_text_into_topics(doc.page_content):
            index = per_source.get(source, 0)
            per_source[source] = index + 1
            # Numbered within the source file so each chunk gets a stable,
            # human-readable id such as "faq.txt:3".
            chunks.append(
                Document(
                    page_content=topic,
                    metadata={**metadata, "chunk_index": index},
                )
            )

    return chunks


def chunk_id(chunk: Any) -> str:
    return f"{chunk.metadata.get('source', 'unknown')}:{chunk.metadata['chunk_index']}"


def _delete_stale_chunks(store: Any, sources: List[str], fresh_ids: set) -> int:
    """Remove chunks that survive from an earlier, longer version of a file."""
    removed = 0
    for source in sources:
        try:
            existing = store.get(where={"source": source})
        except Exception:
            logger.exception("Could not list existing chunks for %s", source)
            continue

        stale = [i for i in existing.get("ids", []) if i not in fresh_ids]
        if stale:
            store.delete(ids=stale)
            removed += len(stale)
            logger.info("Removed %d stale chunk(s) from %s", len(stale), source)
    return removed


def ingest_documents(
    documents_dir: str | None = None,
    chroma_dir: str | None = None,
    reset: bool = False,
) -> int:
    """Build or refresh the vector index. Returns the number of chunks stored."""
    docs_path = Path(documents_dir or settings.documents_dir)
    target_dir = chroma_dir or settings.chroma_dir

    documents = load_documents(docs_path)
    if not documents:
        raise ValueError(
            f"No documents found in {docs_path}. Supported types: "
            f"{', '.join(SUPPORTED_SUFFIXES)}"
        )

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("Documents loaded but produced no chunks after splitting.")

    store = get_vector_store(chroma_dir=target_dir)

    if reset:
        logger.warning("--reset given: deleting the existing collection")
        try:
            store.delete_collection()
        except Exception:
            logger.exception("Could not delete the collection; continuing")
        store = get_vector_store(chroma_dir=target_dir)

    ids = [chunk_id(c) for c in chunks]
    store.add_documents(documents=chunks, ids=ids)

    if not reset:
        sources = sorted({c.metadata.get("source", "unknown") for c in chunks})
        _delete_stale_chunks(store, sources, set(ids))

    logger.info(
        "Ingestion completed: %d chunks from %d document(s) into %r",
        len(chunks),
        len(documents),
        target_dir,
    )
    return len(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma.")
    parser.add_argument("--documents-dir", default=None, help="Source directory.")
    parser.add_argument("--chroma-dir", default=None, help="Vector store directory.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the collection first for a guaranteed clean rebuild.",
    )
    args = parser.parse_args()

    configure_logging(settings.log_level)

    try:
        count = ingest_documents(
            documents_dir=args.documents_dir,
            chroma_dir=args.chroma_dir,
            reset=args.reset,
        )
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1

    print(f"Ingestion completed: {count} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
