"""Tests for the conversational-opener path.

The risk this path introduces is not that a greeting goes unrecognised - that
costs a stiff answer and nothing more. The risk is the opposite: a real question
mistaken for small talk would be answered with a canned pleasantry and never
reach the documents at all. Most of what follows is therefore about what must
*not* be treated as small talk.

Pure functions and a stubbed service, so this belongs in the default run.
"""

from __future__ import annotations

import pytest

from app.services.small_talk import (
    ANSWER_GROUNDED,
    ANSWER_OUT_OF_SCOPE,
    ANSWER_SMALL_TALK,
    FAREWELL_REPLY,
    GREETING_REPLY,
    THANKS_REPLY,
    small_talk_reply,
)

pytestmark = pytest.mark.api


@pytest.mark.parametrize(
    "text",
    [
        "hi",
        "hi.",
        "Hi!",
        "HI",
        "hello",
        "Hello!!",
        "hey",
        "hey there",
        "  hello  ",
        "Good morning",
        "good evening.",
        "greetings",
    ],
)
def test_greetings_are_recognised(text):
    assert small_talk_reply(text) == GREETING_REPLY


@pytest.mark.parametrize("text", ["thanks", "Thank you!", "thanks a lot", "ty", "cheers"])
def test_thanks_is_recognised(text):
    assert small_talk_reply(text) == THANKS_REPLY


@pytest.mark.parametrize("text", ["bye", "Goodbye.", "see you", "good night"])
def test_farewells_are_recognised(text):
    assert small_talk_reply(text) == FAREWELL_REPLY


@pytest.mark.parametrize(
    "question",
    [
        # The important case: a greeting with a real question attached must not
        # be swallowed. Substring matching would break every one of these.
        "hi, what is the return policy?",
        "hello, how long does delivery take?",
        "hey can I return a used product",
        "thanks, but what about refunds?",
        "bye bye policy, what is the refund window?",
        # Ordinary questions that merely contain a greeting word.
        "Is there a high priority for double charges?",
        "What is the return policy?",
        "who do I say hi to for support",
    ],
)
def test_real_questions_are_never_treated_as_small_talk(question):
    assert small_talk_reply(question) is None


@pytest.mark.parametrize("text", ["", "   ", "...", "!!!"])
def test_empty_and_punctuation_only_input_is_not_small_talk(text):
    """Nothing to greet. These are rejected by validation before they get here,
    so the correct behaviour is to decline to classify rather than to guess."""
    assert small_talk_reply(text) is None


def test_the_replies_make_no_claim_about_the_documents():
    """A reply that described the corpus would become false on a different one.

    These strings ship regardless of which documents are indexed, so they must
    not state what the knowledge base covers.
    """
    for reply in (GREETING_REPLY, THANKS_REPLY, FAREWELL_REPLY):
        lowered = reply.lower()
        for claim in ("30 day", "refund", "delivery", "return policy", "support hours"):
            assert claim not in lowered, f"{reply!r} makes a claim about the corpus"


# --- service behaviour -----------------------------------------------------


def test_small_talk_never_touches_the_index():
    """It must work with no index at all, which is the state a new user greets
    it in - and it must not spend a retrieval or a generation."""
    from app.services.rag_service import RAGService

    class ExplodingStore:
        def __getattr__(self, name):
            raise AssertionError(f"the vector store was used ({name})")

    class ExplodingLLM:
        def invoke(self, *_args, **_kwargs):
            raise AssertionError("the model was called for a greeting")

    service = RAGService(store=ExplodingStore(), llm=ExplodingLLM())
    result = service.ask("hello")

    assert result["answer"] == GREETING_REPLY
    assert result["sources"] == []
    assert result["answer_type"] == ANSWER_SMALL_TALK


def test_small_talk_is_not_reported_as_a_knowledge_base_miss():
    """`insufficient_context` must keep its single meaning.

    If a greeting set it, every client would have to render "not in knowledge
    base" over "hello", and the evaluation counts of refusals would include
    greetings.
    """
    from app.services.rag_service import RAGService

    service = RAGService(store=object(), llm=object())
    result = service.ask("hi.")

    assert result["insufficient_context"] is False
    assert result["answer_type"] == ANSWER_SMALL_TALK


def test_an_unsourced_small_talk_payload_passes_output_validation():
    from app.services.response_validator import validate_answer_payload

    payload = {
        "answer": GREETING_REPLY,
        "sources": [],
        "insufficient_context": False,
        "answer_type": ANSWER_SMALL_TALK,
    }

    assert validate_answer_payload(payload) is payload


def test_the_exemption_does_not_extend_to_ordinary_answers():
    """The guard that matters: a normal answer with no sources is still a defect.

    Widening the unsourced exemption beyond small talk would reopen exactly the
    hole output validation exists to close.
    """
    from app.services.exceptions import InvalidAnswerError
    from app.services.response_validator import validate_answer_payload

    for answer_type in (ANSWER_GROUNDED, "anything-else"):
        with pytest.raises(InvalidAnswerError):
            validate_answer_payload(
                {
                    "answer": "Returns are accepted within 30 days.",
                    "sources": [],
                    "insufficient_context": False,
                    "answer_type": answer_type,
                }
            )


def test_out_of_scope_still_reports_a_knowledge_base_miss():
    """The refusal path must be unaffected by any of the above."""
    from app.services.rag_service import RAGService

    class _Doc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {"source": "faq.txt", "chunk_index": 0}

    class WeakMatchStore:
        """Populated, but nothing in it is close to the question."""

        def get(self, limit=None):
            return {"ids": ["faq.txt:0"]}

        def similarity_search_with_relevance_scores(self, query, k=3):
            return [(_Doc("unrelated text"), 0.01)]

    service = RAGService(store=WeakMatchStore(), llm=object(), relevance_threshold=0.35)
    result = service.ask("what is the airspeed of an unladen swallow")

    assert result["insufficient_context"] is True
    assert result["answer_type"] == ANSWER_OUT_OF_SCOPE
    assert result["sources"] == []
