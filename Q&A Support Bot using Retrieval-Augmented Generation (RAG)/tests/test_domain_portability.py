"""Can this system be pointed at a different domain without editing the code?

The question it answers is "what do I change to make this a banking bot", and it
answers it by doing it: a small banking corpus is ingested into a temporary index
and queried, with no change to any module under `app/`.

The claim being tested is narrow and worth stating precisely. Retrieval,
chunking, the relevance threshold, redaction and validation are domain-blind -
they operate on text and have no opinion about what the text describes. Three
strings in the shipped answers were not, and they are now settings. What does
*not* travel is `evaluation/datasets/`: every case there asserts a fact about
returns and shipping, so a new domain needs new datasets and a re-measured
threshold. That is the honest boundary, and the last test in this file is what
keeps it visible.

The wiring checks are mocked and run in the default suite. The end-to-end proof
needs the real embedding model, so it is marked `integration`.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from app.config import settings

pytestmark = [pytest.mark.rag]


# A deliberately different domain: different vocabulary, different entities, no
# overlap with returns and shipping. Written in the same shape as the real corpus
# because `split_text_into_topics` splits on numbered headings, and a fixture that
# quietly used the size-based fallback would be testing a different code path
# than production.
BANKING_FAQ = """Retail Banking FAQ

1. How long does an international transfer take?
International transfers arrive within 3 to 5 business days.
Transfers submitted after 4pm are processed the next business day.

2. What is the daily ATM withdrawal limit?
The standard daily ATM withdrawal limit is 500 units.
The limit can be raised for one day through the mobile app.

3. How do I report a lost or stolen card?
Report a lost or stolen card immediately through the mobile app
or by calling the number printed on your statement.
A replacement card is issued within 7 business days.

4. What happens if a direct debit fails?
A failed direct debit is retried once on the next business day.
A failed payment fee of 10 units applies after the second failure.
"""

BANKING_POLICY = """Dispute Handling Policy

1. Disputed transaction window
A transaction can be disputed within 60 days of the statement date.
Disputes raised after 60 days cannot be investigated.

2. Provisional credit
A provisional credit may be applied while an investigation is open.
Provisional credit is not guaranteed and can be reversed
if the dispute is not upheld.

3. Investigation timeline
A dispute investigation is completed within 45 days.
The customer is notified in writing of the outcome.
"""


@pytest.fixture(scope="module")
def banking_index():
    """A vector store built from the banking corpus alone.

    `data/documents` is not touched and the real corpus is not included, which is
    the point: if any part of retrieval depended on the e-commerce documents
    being present, it fails here.
    """
    from app.ingestion.ingest import ingest_documents

    docs_dir = tempfile.mkdtemp(prefix="rag_banking_docs_")
    index_dir = tempfile.mkdtemp(prefix="rag_banking_chroma_")
    try:
        (Path(docs_dir) / "banking_faq.txt").write_text(BANKING_FAQ, encoding="utf-8")
        (Path(docs_dir) / "dispute_policy.txt").write_text(
            BANKING_POLICY, encoding="utf-8"
        )
        ingest_documents(documents_dir=docs_dir, chroma_dir=index_dir, reset=True)
        yield index_dir
    finally:
        shutil.rmtree(docs_dir, ignore_errors=True)
        shutil.rmtree(index_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# The wiring: no model, no index, runs in the default suite
# --------------------------------------------------------------------------


def test_the_system_prompt_takes_its_role_from_configuration():
    """Re-pointing the bot must not require editing `rag_service.py`.

    Asserted against the setting rather than against the literal text, so the
    test keeps working for whoever changes the default - it checks that the wire
    is connected, not what is currently flowing through it.
    """
    from app.services.rag_service import SYSTEM_PROMPT

    assert f"You are {settings.assistant_role}." in SYSTEM_PROMPT


def test_the_context_placeholder_survives_the_role_interpolation():
    """The prompt is an f-string and a format template at the same time.

    `{context}` has to be escaped as `{{context}}` for that to work. Getting it
    wrong raises at import in one direction, and in the other silently produces a
    prompt with no context in it - an answer with no grounding and no error.
    """
    from app.services.rag_service import SYSTEM_PROMPT

    assert "{context}" in SYSTEM_PROMPT
    filled = SYSTEM_PROMPT.format(context="A transfer takes 3 to 5 business days.")
    assert "A transfer takes 3 to 5 business days." in filled
    assert "{context}" not in filled


def test_the_refusal_keeps_a_stable_first_sentence_and_a_configurable_route():
    """Both halves matter, for opposite reasons.

    The first sentence is what the evaluators and the threshold tests recognise,
    so it is fixed. The escalation route is the part that would be actively
    misleading in a new deployment, so it is configurable.
    """
    from app.services.rag_service import INSUFFICIENT_CONTEXT_ANSWER

    assert INSUFFICIENT_CONTEXT_ANSWER.startswith(
        "I don't have enough information in the knowledge base to answer that."
    )
    assert INSUFFICIENT_CONTEXT_ANSWER.endswith(settings.escalation_hint)


def test_the_greeting_names_the_configured_knowledge_domain():
    from app.services.small_talk import GREETING_REPLY

    assert settings.knowledge_domain in GREETING_REPLY


def test_ingestion_and_retrieval_settings_are_not_domain_specific():
    """A corpus swap must not need a code change to be indexed.

    `documents_dir`, `collection_name` and `chroma_dir` are all settings, so a
    second domain can be a second `.env` pointing at a second index rather than
    a fork of the project.
    """
    for name in ("documents_dir", "collection_name", "chroma_dir", "relevance_threshold"):
        assert hasattr(settings, name), name


# --------------------------------------------------------------------------
# The proof: a different domain, indexed and queried
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "question,expected_source,expected_fragment",
    [
        ("How long does an international transfer take?", "banking_faq.txt", "3 to 5"),
        ("What is the daily ATM withdrawal limit?", "banking_faq.txt", "500"),
        ("How long do I have to dispute a transaction?", "dispute_policy.txt", "60 days"),
        ("How long does a dispute investigation take?", "dispute_policy.txt", "45 days"),
    ],
)
def test_a_banking_corpus_is_retrieved_with_no_code_changes(
    banking_index, question, expected_source, expected_fragment
):
    """Retrieval only, so this runs with Ollama stopped.

    Deliberately asserting on the retrieved chunk rather than on a generated
    answer: this is the claim about the architecture. Whether the model then
    words the answer well is a separate question, measured separately.
    """
    from app.services.rag_service import RAGService

    service = RAGService(chroma_dir=banking_index)
    matches = service.retrieve(question)

    assert matches, f"{question!r} retrieved nothing from the banking index"
    sources = {doc.metadata.get("source") for doc, _ in matches}
    assert expected_source in sources, f"{question!r} -> {sources}"

    text = " ".join(doc.page_content for doc, _ in matches)
    assert expected_fragment in text, f"{question!r} -> {text[:200]!r}"


@pytest.mark.integration
def test_the_old_domain_becomes_out_of_scope_once_the_corpus_changes(banking_index):
    """The threshold travels with the corpus, which is the whole safety story.

    Against a banking index, "what is the return policy" has to be refused. If it
    were answered, the bot would be improvising retail policy out of banking text
    - and that is exactly the failure the threshold exists to prevent, so it has
    to hold for an unfamiliar corpus and not just the one it was tuned on.
    """
    from app.services.rag_service import RAGService

    service = RAGService(chroma_dir=banking_index)
    for question in (
        "What is the return policy?",
        "How long does delivery take?",
        "How do I track my order?",
    ):
        result = service.ask(question)
        assert result["insufficient_context"] is True, (
            f"{question!r} was answered from a banking corpus: {result['answer']!r}"
        )
        assert result["sources"] == []


@pytest.mark.integration
def test_a_banking_answer_is_generated_from_banking_context(
    banking_index, require_ollama
):
    """End to end on the new domain, with the model in the loop.

    One case rather than a parametrised set: the point here is that the whole
    pipeline runs on a foreign corpus, not to measure answer quality on a domain
    with no evaluation dataset behind it.
    """
    from app.services.rag_service import RAGService

    service = RAGService(chroma_dir=banking_index)
    result = service.ask("How long do I have to dispute a transaction?")

    assert result["insufficient_context"] is False
    assert "60" in result["answer"], result["answer"]
    print(f"\nBanking domain answer:\n  {result['answer']}")


def test_the_evaluation_datasets_are_the_part_that_does_not_travel():
    """States the limit of the claim above, so nobody has to discover it.

    A reader who sees a banking corpus working here could reasonably assume the
    quality numbers travel too. They do not: every case in `evaluation/datasets`
    asserts an e-commerce fact, so against a banking corpus the suite would
    measure nothing except that the bot correctly refuses questions about
    returns. A new domain needs new datasets and a re-measured threshold.
    """
    from evaluation.run_eval import ALL_DATASETS, load_dataset

    ecommerce_vocabulary = ("return", "order", "ship", "deliver", "refund", "product")
    domain_specific = 0
    total = 0
    for name in ALL_DATASETS:
        for case in load_dataset(name):
            total += 1
            if any(word in case["question"].lower() for word in ecommerce_vocabulary):
                domain_specific += 1

    assert total > 0
    # Not an exact count: the assertion is that the datasets are overwhelmingly
    # tied to this corpus, which is the fact a new deployment needs to know.
    assert domain_specific > total * 0.5, (
        f"only {domain_specific}/{total} cases are e-commerce specific - if that "
        "is now genuinely low, the docs claiming the datasets must be rewritten "
        "per domain need revisiting"
    )
