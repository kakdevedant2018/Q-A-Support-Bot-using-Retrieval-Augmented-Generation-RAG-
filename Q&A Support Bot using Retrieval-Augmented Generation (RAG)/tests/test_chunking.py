"""Tests for topic-boundary chunking.

These exist because of a measured failure, not a hypothetical one. Splitting to a
500-character budget packed three unrelated FAQ entries into one chunk, and since
a chunk is embedded as a single vector, that vector became the average of all
three. Two consequences were measured against the real index:

  - "How can I contact support?" ranked the chunk holding the answer third, at
    0.3231, below a chunk that only gives opening hours.
  - "What payment methods are supported?" peaked at 0.3451, four thousandths
    under the relevance threshold, so the bot refused a question it could answer.

Nothing in the suite would have caught either, because no test asserted what a
chunk should *contain*. The assertion that matters is one topic per chunk, so
that is what is asserted here - on the real documents, not only on fixtures, so
that editing the corpus into one unsplittable wall of text fails a test.

This is a pure function: no embedding model, no index, no network. It belongs in
the default run and needs only the light dependency set.
"""

from __future__ import annotations

import pytest

from app.ingestion.ingest import (
    split_documents,
    split_text_into_topics,
)

pytestmark = pytest.mark.rag


FAQ_SAMPLE = """Company FAQ

1. What is the return policy?
Customers can return products within 30 days of delivery
if the product is unused.

2. How can I track my order?
Customers can track orders using the tracking ID provided
in the order confirmation email.

3. How can I contact support?
Customers can contact support through email or the
customer service portal.
"""


class _FakeDoc:
    """Minimal stand-in for a LangChain document."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def test_each_faq_entry_becomes_its_own_chunk():
    """The regression that caused the retrieval failures."""
    topics = split_text_into_topics(FAQ_SAMPLE)

    return_policy = [t for t in topics if "return policy" in t]
    assert len(return_policy) == 1, "the return policy is spread across chunks"

    # The specific defect: unrelated subjects sharing one embedding.
    chunk = return_policy[0]
    assert "track my order" not in chunk
    assert "contact support" not in chunk


def test_every_topic_in_the_sample_is_separated():
    topics = split_text_into_topics(FAQ_SAMPLE)

    # Three numbered entries, and the bare title is not a topic of its own.
    assert len(topics) == 3
    for subject in ("return policy", "track my order", "contact support"):
        matching = [t for t in topics if subject in t]
        assert len(matching) == 1, f"{subject!r} appears in {len(matching)} chunks"


def test_a_bare_heading_is_attached_rather_than_left_alone():
    """A heading with no body is a content-free chunk, which can still be
    retrieved. It is worse than useless: it occupies a slot in the top k."""
    topics = split_text_into_topics(FAQ_SAMPLE)

    assert "Company FAQ" not in topics
    assert any(t.startswith("Company FAQ") for t in topics)


def test_consecutive_headings_are_kept_together():
    topics = split_text_into_topics("Doc Title\n\nSection\n\nBody text here.\n")

    assert topics == ["Doc Title\nSection\nBody text here."]


def test_a_trailing_heading_is_not_silently_dropped():
    """Content must never be lost, even when it looks like a stray heading."""
    topics = split_text_into_topics("Body paragraph one.\n\nStray trailing line")

    assert topics == ["Body paragraph one.", "Stray trailing line"]


def test_blank_and_whitespace_only_blocks_are_discarded():
    topics = split_text_into_topics("First topic body.\n\n   \n\n\nSecond topic body.\n")

    assert topics == ["First topic body.", "Second topic body."]


def test_an_oversized_topic_falls_back_to_size_splitting():
    """One topic too long to embed well is still split, just not preferentially."""
    pytest.importorskip(
        "langchain_text_splitters",
        reason="oversize fallback needs the full requirements.txt",
    )
    from app.config import settings

    long_topic = "Refund terms apply. " * 200
    assert len(long_topic) > settings.chunk_size

    topics = split_text_into_topics(long_topic)

    assert len(topics) > 1
    assert all(len(t) <= settings.chunk_size for t in topics)


def test_chunks_are_numbered_per_source_file():
    """Chunk ids must be stable per file, because ingestion keys on them."""
    documents = [
        _FakeDoc(FAQ_SAMPLE, {"source": "faq.txt"}),
        _FakeDoc("Policy A body.\n\nPolicy B body.\n", {"source": "policy.txt"}),
    ]

    chunks = split_documents(documents)

    by_source: dict = {}
    for chunk in chunks:
        by_source.setdefault(chunk.metadata["source"], []).append(
            chunk.metadata["chunk_index"]
        )

    assert by_source["faq.txt"] == [0, 1, 2]
    assert by_source["policy.txt"] == [0, 1]


def test_splitting_preserves_the_source_metadata():
    chunks = split_documents([_FakeDoc(FAQ_SAMPLE, {"source": "faq.txt", "extra": 1})])

    assert all(c.metadata["source"] == "faq.txt" for c in chunks)
    assert all(c.metadata["extra"] == 1 for c in chunks)


def test_no_content_is_lost_when_splitting():
    """Chunking must be lossless apart from whitespace.

    A splitter that drops a sentence makes the corresponding question
    unanswerable, and the symptom is an unexplained refusal rather than an error.
    """
    topics = split_text_into_topics(FAQ_SAMPLE)

    rejoined = " ".join(" ".join(t.split()) for t in topics)
    for sentence in (
        "Customers can return products within 30 days of delivery",
        "Customers can track orders using the tracking ID provided",
        "Customers can contact support through email or the",
    ):
        assert " ".join(sentence.split()) in rejoined


# --- the real corpus -------------------------------------------------------
# Fixtures prove the algorithm; these prove the documents actually ship in a
# shape it can split. A corpus rewritten as one unbroken block would quietly
# undo the fix, and no fixture-based test would notice.


def _real_documents():
    from pathlib import Path

    from app.config import settings

    directory = Path(settings.documents_dir)
    paths = sorted(directory.glob("*.txt")) if directory.is_dir() else []
    if not paths:
        pytest.skip(f"no documents found in {directory}")
    return paths


def test_the_shipped_documents_split_into_multiple_topics():
    for path in _real_documents():
        topics = split_text_into_topics(path.read_text(encoding="utf-8"))
        assert len(topics) > 1, f"{path.name} produced a single chunk"


def test_the_return_policy_does_not_share_a_chunk_with_other_subjects():
    """The exact dilution that was measured, asserted against the real corpus."""
    for path in _real_documents():
        for topic in split_text_into_topics(path.read_text(encoding="utf-8")):
            if "return policy" not in topic.lower():
                continue
            lowered = topic.lower()
            for unrelated in ("tracking id", "payment methods", "delivery take"):
                assert unrelated not in lowered, (
                    f"{path.name}: the return policy shares a chunk with "
                    f"{unrelated!r}, which dilutes its embedding"
                )
