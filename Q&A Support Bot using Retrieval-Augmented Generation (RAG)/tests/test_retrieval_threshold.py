"""Unit tests for the relevance threshold.

These are the tests that make the out-of-scope guarantee real, and they cost
nothing: the vector store and the model are both stubs, so there is no
download, no Ollama, and no network.

The reference implementation has no threshold at all. It hands whatever the
retriever returned to the model and relies on the prompt to say "I don't know".
A prompt is a request, not a constraint, so that behaviour cannot be asserted.
Enforcing it in code makes it testable.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pytest

from app.services.exceptions import KnowledgeBaseUnavailableError, LLMUnavailableError
from app.services.rag_service import INSUFFICIENT_CONTEXT_ANSWER, RAGService

pytestmark = pytest.mark.rag


@dataclass
class StubDocument:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StubStore:
    """Minimal stand-in for a Chroma collection."""

    def __init__(self, scored: List[Tuple[StubDocument, float]], empty: bool = False):
        self._scored = scored
        self._empty = empty
        self.queries: List[str] = []

    def get(self, limit: int | None = None) -> Dict[str, Any]:
        return {"ids": [] if self._empty else ["faq.txt:0"]}

    def similarity_search_with_relevance_scores(self, query: str, k: int = 3):
        self.queries.append(query)
        return self._scored[:k]


class StubMessage:
    def __init__(self, content: str):
        self.content = content


class RecordingLLM:
    """Captures what it was asked, so the context can be inspected."""

    def __init__(self, reply: str = "Within 30 days of delivery."):
        self.reply = reply
        self.calls: List[Any] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return StubMessage(self.reply)


class ForbiddenLLM:
    """Fails the test if the model is called at all."""

    def invoke(self, messages):
        raise AssertionError(
            "The model was called even though no chunk cleared the threshold"
        )


RELEVANT = StubDocument(
    "Customers can return products within 30 days of delivery.",
    {"source": "faq.txt", "chunk_index": 0},
)
WEAK = StubDocument(
    "The company supports credit cards and debit cards.",
    {"source": "faq.txt", "chunk_index": 3},
)


def _service(scored, llm, threshold=0.35, empty=False) -> RAGService:
    return RAGService(
        store=StubStore(scored, empty=empty),
        llm=llm,
        relevance_threshold=threshold,
        k=3,
    )


def test_out_of_scope_question_returns_the_fallback_without_calling_the_model():
    """R04: every candidate is weak, so the bot declines instead of guessing."""
    service = _service([(WEAK, 0.11), (RELEVANT, 0.08)], ForbiddenLLM())

    result = service.ask("What is the weather on Mars?")

    assert result["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert result["sources"] == []
    assert result["insufficient_context"] is True


def test_fallback_wording_matches_what_the_api_suite_expects():
    """Guards the exact phrase the contract tests assert on."""
    answer = INSUFFICIENT_CONTEXT_ANSWER.lower()
    assert "don't have enough information" in answer


def test_relevant_question_reaches_the_model_with_only_strong_chunks():
    """Weak chunks are dropped from the prompt, not just ranked lower."""
    llm = RecordingLLM()
    service = _service([(RELEVANT, 0.82), (WEAK, 0.09)], llm)

    result = service.ask("What is the return policy?")

    assert result["insufficient_context"] is False
    assert len(result["sources"]) == 1
    assert result["sources"][0]["metadata"]["source"] == "faq.txt"

    prompt_text = str(llm.calls[0])
    assert "30 days" in prompt_text
    assert "credit cards" not in prompt_text


def test_relevance_score_is_reported_on_each_source():
    """Transparency: the caller can see how strong each citation was."""
    service = _service([(RELEVANT, 0.7654321)], RecordingLLM())

    result = service.ask("What is the return policy?")
    assert result["sources"][0]["relevance_score"] == pytest.approx(0.7654)


def test_score_exactly_at_the_threshold_is_kept():
    """The bound is inclusive, so a threshold of 0.4 keeps a 0.4 match."""
    service = _service([(RELEVANT, 0.4)], RecordingLLM(), threshold=0.4)

    result = service.ask("What is the return policy?")
    assert result["insufficient_context"] is False


def test_score_just_below_the_threshold_is_dropped():
    service = _service([(RELEVANT, 0.399)], ForbiddenLLM(), threshold=0.4)

    result = service.ask("What is the return policy?")
    assert result["insufficient_context"] is True


def test_empty_collection_raises_rather_than_answering():
    """An un-ingested index is an operator problem, not an honest "I don't know"."""
    service = _service([], ForbiddenLLM(), empty=True)

    with pytest.raises(KnowledgeBaseUnavailableError):
        service.ask("What is the return policy?")


def test_search_failure_is_wrapped_as_a_knowledge_base_error():
    class BrokenStore(StubStore):
        def similarity_search_with_relevance_scores(self, query, k=3):
            raise RuntimeError("hnsw index corrupt")

    service = RAGService(store=BrokenStore([]), llm=ForbiddenLLM())

    with pytest.raises(KnowledgeBaseUnavailableError):
        service.ask("What is the return policy?")


def test_empty_model_reply_is_treated_as_a_model_failure():
    """A blank answer must not be served as a successful response."""
    service = _service([(RELEVANT, 0.9)], RecordingLLM(reply="   "))

    with pytest.raises(LLMUnavailableError):
        service.ask("What is the return policy?")


def test_model_exception_is_wrapped_as_llm_unavailable():
    class BrokenLLM:
        def invoke(self, messages):
            raise ConnectionError("connection refused")

    service = _service([(RELEVANT, 0.9)], BrokenLLM())

    with pytest.raises(LLMUnavailableError):
        service.ask("What is the return policy?")


def test_k_limits_how_many_chunks_are_considered():
    llm = RecordingLLM()
    service = RAGService(
        store=StubStore([(RELEVANT, 0.9), (WEAK, 0.9), (RELEVANT, 0.9)]),
        llm=llm,
        k=2,
        relevance_threshold=0.1,
    )

    result = service.ask("What is the return policy?")
    assert len(result["sources"]) == 2


def test_chroma_dir_override_is_respected():
    """The knob that keeps tests out of the real knowledge base."""
    service = RAGService(chroma_dir="/tmp/some_test_dir")
    assert service.chroma_dir == "/tmp/some_test_dir"
