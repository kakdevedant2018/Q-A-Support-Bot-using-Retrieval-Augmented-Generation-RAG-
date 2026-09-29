"""Retrieval mechanics and end-to-end behaviour against the real stack.

Everything here is marked `integration` and is excluded from the default run.
Run it deliberately:

    pytest -m integration

Requirements: the embedding model downloads on first use, and the generation
tests need a running Ollama server with the configured model pulled. Retrieval
tests need only the embedding model, so they still run with Ollama stopped.

Why this file is separate from the contract tests: a 200 status code and a
well-formed body say nothing about whether the answer is right. API reliability
and answer quality are different properties and they fail for different
reasons, so they are measured separately.

Scope note. This file covers the plumbing: does the index contain what it should,
are scores normalised, does the threshold separate in-scope from out-of-scope, and
does one question survive the whole round trip. Scored answer quality across a
labelled dataset now lives in `test_ai_quality.py`, and safety behaviour in
`test_security.py`.

Phrase checks here go through `evaluation.normalize.contains_phrase` rather than
`in`. `assert "30 days" in answer` fails on "thirty days", on "a 30-day window",
and on "30 calendar days", all of which are correct answers - so that assertion
measures the model's phrasing, not its accuracy.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from evaluation.normalize import contains_phrase, is_negated
from tests.conftest import TEST_DATA_DIR

pytestmark = [pytest.mark.integration, pytest.mark.rag]

ASK_URL = "/api/v1/ask"


def load_test_data() -> List[Dict[str, Any]]:
    path = TEST_DATA_DIR / "questions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _ids(cases: List[Dict[str, Any]]) -> List[str]:
    return [c["id"] for c in cases]


# --------------------------------------------------------------------------
# Retrieval only. No language model needed.
# --------------------------------------------------------------------------


def test_retrieval_finds_the_return_policy(integration_service):
    """R06: the index actually contains what we expect it to contain."""
    matches = integration_service.retrieve("How long do I have to return a product?")

    assert matches, "no chunk cleared the relevance threshold"
    combined = " ".join(doc.page_content for doc, _ in matches)
    assert contains_phrase(combined, "30 days")


def test_retrieval_scores_are_normalised(integration_service):
    """Scores must sit in [0, 1] or the configured threshold is meaningless.

    This is what the cosine distance metric buys. With the default L2 metric the
    numbers are unbounded and a fixed threshold cannot be reasoned about.
    """
    matches = integration_service.retrieve("What is the return policy?")

    for _, score in matches:
        assert 0.0 <= score <= 1.0, f"score {score} is outside [0, 1]"


def test_relevant_question_scores_higher_than_an_unrelated_one(integration_service):
    """The signal the threshold depends on must actually separate the two."""
    store = integration_service.store

    relevant = store.similarity_search_with_relevance_scores(
        "What is the return policy?", k=1
    )
    unrelated = store.similarity_search_with_relevance_scores(
        "What is the weather on Mars today?", k=1
    )

    print(f"\nrelevant={relevant[0][1]:.3f}  unrelated={unrelated[0][1]:.3f}")
    assert relevant[0][1] > unrelated[0][1]


def test_out_of_scope_question_is_filtered_by_the_threshold(integration_service):
    """R04 against the real index, with no model call required.

    If this fails, the threshold in config.py needs measuring against these
    documents rather than accepting the shipped default.
    """
    matches = integration_service.retrieve("What is the weather on Mars today?")
    assert matches == []


def test_every_chunk_carries_its_source_file(integration_service):
    """R05: a citation must point at a real file."""
    matches = integration_service.retrieve("What is the return policy?")

    for doc, _ in matches:
        assert doc.metadata.get("source") in {"faq.txt", "support_policy.txt"}
        assert isinstance(doc.metadata.get("chunk_index"), int)


def test_reingestion_does_not_duplicate_chunks(seeded_chroma_dir):
    """The duplicate-on-rerun problem, asserted rather than assumed.

    Stable per-chunk ids make a second ingestion replace chunks in place. The
    reference implementation appends, so this count would grow every run.
    """
    from app.ingestion.ingest import ingest_documents
    from app.services.vector_store import get_vector_store

    store = get_vector_store(chroma_dir=seeded_chroma_dir)
    before = len(store.get()["ids"])

    ingest_documents(
        documents_dir=str(Path(TEST_DATA_DIR).parent / "data" / "documents"),
        chroma_dir=seeded_chroma_dir,
        reset=False,
    )

    after = len(get_vector_store(chroma_dir=seeded_chroma_dir).get()["ids"])
    assert after == before, f"chunk count grew from {before} to {after}"


# --------------------------------------------------------------------------
# End to end. Needs Ollama.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", load_test_data(), ids=_ids(load_test_data()))
def test_answers_contain_the_expected_facts(integration_client, require_ollama, case):
    """Keyword grounding across the labelled question set.

    A smoke test, not a quality measure: it catches a bot that has stopped
    retrieving or started inventing figures, and says nothing about whether the
    answer reads well. `tests/test_ai_quality.py` does the scoring.

    Matching is normalised, so "thirty days" satisfies an expected keyword of
    "30 days". Each expected keyword is also checked for negation, because
    "products cannot be returned after 30 days" contains the keyword while
    stating the opposite rule.
    """
    response = integration_client.post(ASK_URL, json={"question": case["question"]})

    assert response.status_code == 200
    answer = response.json()["answer"]

    for keyword in case["expected_keywords"]:
        assert contains_phrase(answer, keyword), (
            f"{case['id']}: expected {keyword!r} in answer: {answer!r}"
        )
        assert not is_negated(answer, keyword), (
            f"{case['id']}: {keyword!r} appears but is negated, so the answer "
            f"states the opposite: {answer!r}"
        )

    for forbidden in case.get("forbidden_keywords", []):
        assert not contains_phrase(answer, forbidden), (
            f"{case['id']}: answer contained forbidden {forbidden!r}: {answer!r}"
        )


def test_return_policy_answer_is_grounded(integration_client, require_ollama):
    """The canonical groundedness check.

    The context says 30 days. An answer saying 90 days is ungrounded even though
    it is fluent, well formed, and returned with a 200.

    Two separate assertions, because they fail for different reasons. The first
    is correctness: the return window has to be communicated, in whatever words
    the model picks. The second is groundedness: no figure may be asserted that
    is absent from the retrieved context. Listing "90 days" and "60 days" by hand
    only catches the two wrong numbers someone thought of; checking every figure
    against the context catches all of them.
    """
    from evaluation.groundedness_evaluator import unsupported_figures

    question = "What is the return policy?"
    response = integration_client.post(ASK_URL, json={"question": question})

    assert response.status_code == 200
    data = response.json()
    answer = data["answer"]
    context = " ".join(source["content"] for source in data["sources"])

    assert contains_phrase(answer, "30 days"), f"return window missing: {answer!r}"
    assert not is_negated(answer, "30 days"), f"return window negated: {answer!r}"
    assert unsupported_figures(answer, context, question) == [], (
        f"answer asserts figures absent from its own sources: {answer!r}"
    )


def test_answer_is_supported_by_its_own_cited_sources(integration_client, require_ollama):
    """A cited figure must appear in the text that was actually retrieved.

    Checks the citation, not the answer. A right answer with sources that do not
    contain the fact means retrieval got lucky or the model used its training
    data, and both are defects in a RAG system.
    """
    response = integration_client.post(
        ASK_URL, json={"question": "What is the return policy?"}
    )
    data = response.json()

    cited = " ".join(source["content"] for source in data["sources"])
    assert contains_phrase(cited, "30 days"), (
        "the answer cites sources that do not contain the fact"
    )


def test_out_of_scope_question_returns_the_fallback_over_http(
    integration_client, require_ollama
):
    """R04 end to end: an honest refusal, still with a 200."""
    response = integration_client.post(
        ASK_URL, json={"question": "What is the weather on Mars today?"}
    )

    assert response.status_code == 200
    data = response.json()

    # This answer is the application's own fixed string, not model output, so it
    # can be asserted exactly.
    from app.services.rag_service import INSUFFICIENT_CONTEXT_ANSWER

    assert data["insufficient_context"] is True
    assert data["sources"] == []
    assert data["answer"] == INSUFFICIENT_CONTEXT_ANSWER


def test_real_latency_is_measured(integration_client, require_ollama):
    """P01: record the real baseline. No assertion until one is measured.

    Local generation speed depends entirely on the machine and the model size,
    so a hardcoded budget here would be a flaky test on someone else's laptop.
    """
    start = time.perf_counter()
    response = integration_client.post(
        ASK_URL, json={"question": "What is the return policy?"}
    )
    duration = time.perf_counter() - start

    assert response.status_code == 200
    print(f"\nEnd-to-end local generation: {duration:.2f}s")


def test_repeated_identical_questions_stay_consistent(integration_client, require_ollama):
    """The same question must not drift in substance between runs.

    Temperature is zero, but that is not a guarantee of byte-identical output, so
    the assertion is on the fact rather than on the string. Comparing the two
    answers for equality would be the wrong test: the bot is allowed to reword,
    and it is not allowed to change the policy.
    """
    answers = []
    for _ in range(2):
        response = integration_client.post(
            ASK_URL, json={"question": "What is the return policy?"}
        )
        assert response.status_code == 200
        answers.append(response.json()["answer"])

    for answer in answers:
        assert contains_phrase(answer, "30 days"), f"drifted: {answer!r}"
        assert not is_negated(answer, "30 days"), f"inverted: {answer!r}"


# --------------------------------------------------------------------------
# Paraphrase consistency
# --------------------------------------------------------------------------
#
# The test above asks one question twice. That measures sampling stability, which
# at temperature 0 is close to free. It does not measure the harder property: the
# same fact requested in different words has to come back the same.
#
# This is where a retrieval system drifts first. Each phrasing is a different
# embedding, so each one retrieves a slightly different set of chunks, and a
# wording that pulls a different chunk to the top can produce a different policy.
# To the user those are the same question, so two different answers is a bug
# regardless of which one is correct.
#
# Written as two tests rather than one, because "the phrasings disagree" and "one
# phrasing was not answered at all" are different faults with different
# severities and different fixes. A contradiction is wrong. A refusal is the
# threshold being conservative - unhelpful, but not incorrect. Folding them into
# one assertion would mean a real contradiction could never be seen while a known
# recall gap was still open.

# (id, phrasings, the fact all of them must carry)
PARAPHRASE_GROUPS = [
    (
        "return-window",
        [
            "What is the return policy?",
            "How long do I have to return something?",
            "how many days can i send an item back",
            "return window?",
        ],
        "30 days",
    ),
    (
        "address-locked",
        [
            "Can I change my delivery address after the order has shipped?",
            "I need to update where my package is going, it already shipped.",
            "adress change after ship?",
        ],
        "cannot be changed",
    ),
    (
        "contact-channel",
        [
            "How do I contact support?",
            "who do i talk to about a problem",
            "I need to reach a human.",
        ],
        "email",
    ),
]

# Measured, not assumed. These groups contain a phrasing that scores below the
# configured relevance threshold of 0.35, so the bot refuses it:
#
#   "return window?"                   best chunk 0.300
#   "who do i talk to about a problem"  best chunk 0.213
#   "I need to reach a human."         best chunk below threshold
#
# Marked xfail rather than deleted, and strict so that it reports when it starts
# passing. The fix is not to lower the threshold: `evaluation.calibrate` puts the
# nearest in-domain question at 0.3518, so dropping to 0.2 would admit
# out-of-scope questions and trade a refusal for a fabrication. A real fix means
# query expansion or a corpus that states these facts in the words users use.
KNOWN_RECALL_GAPS = {"return-window", "contact-channel"}


def _paraphrase_params(*, xfail_known_gaps: bool):
    """Build the parameter list, optionally marking the measured recall gaps."""
    params = []
    for group_id, questions, fact in PARAPHRASE_GROUPS:
        marks = []
        if xfail_known_gaps and group_id in KNOWN_RECALL_GAPS:
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "known recall gap: a phrasing in this group scores below "
                        "the 0.35 relevance threshold and is refused"
                    ),
                )
            )
        params.append(pytest.param(questions, fact, id=group_id, marks=marks))
    return params


def _answers_for(client, questions):
    answers = {}
    for question in questions:
        response = client.post(ASK_URL, json={"question": question})
        assert response.status_code == 200, f"{question!r} -> {response.status_code}"
        answers[question] = response.json()["answer"]
    return answers


def _format(answers):
    return "\n".join(f"  {q!r}\n    -> {a}" for q, a in answers.items())


@pytest.mark.integration
@pytest.mark.parametrize(
    "questions,required_fact", _paraphrase_params(xfail_known_gaps=False)
)
def test_paraphrases_never_contradict_each_other(
    integration_client, require_ollama, questions, required_fact
):
    """No two phrasings may state opposite policies. This one has no excuses.

    Refusals are excluded rather than failed - a refusal states nothing, so it
    cannot contradict anything, and the separate coverage test below is what
    holds the line on those. What is not tolerated is one phrasing saying the
    address can be changed while another says it cannot.

    `is_negated` is the reason this is not a substring check: both answers
    contain the phrase, and only one of them is right.
    """
    answers = _answers_for(integration_client, questions)
    answered = {
        q: a for q, a in answers.items() if not contains_phrase(a, "enough information")
    }
    assert answered, f"every phrasing was refused:\n{_format(answers)}"

    affirmed = {q for q, a in answered.items() if contains_phrase(a, required_fact)}
    denied = {q for q, a in answered.items() if is_negated(a, required_fact)}

    assert not denied, (
        f"a phrasing inverted {required_fact!r} while another affirmed it, so the "
        f"wording alone changed the policy:\n{_format(answered)}"
    )
    assert affirmed, (
        f"no phrasing produced {required_fact!r}, yet none refused either - the "
        f"bot answered with something else entirely:\n{_format(answered)}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "questions,required_fact", _paraphrase_params(xfail_known_gaps=True)
)
def test_every_phrasing_of_an_answerable_question_is_answered(
    integration_client, require_ollama, questions, required_fact
):
    """Coverage, not correctness. The corpus answers this; the bot should too.

    A user who asks "return window?" and is told the knowledge base has nothing
    has been failed, even though refusing is safer than guessing. This test
    measures how much of that gap is still open, which is why the expected
    failures are marked rather than removed.
    """
    answers = _answers_for(integration_client, questions)
    missing = {q: a for q, a in answers.items() if not contains_phrase(a, required_fact)}

    assert not missing, (
        f"{len(missing)} of {len(questions)} phrasings did not produce "
        f"{required_fact!r}:\n{_format(answers)}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "questions,required_fact", _paraphrase_params(xfail_known_gaps=True)
)
def test_paraphrases_retrieve_a_common_document(
    integration_service, questions, required_fact
):
    """Retrieval-side half of the same property, and it needs no Ollama.

    Separated deliberately. When the test above fails, this one says whether the
    cause was retrieval handing the phrasings different evidence or generation
    reading the same evidence differently. Those have different fixes, and a
    pass-rate cannot tell them apart.
    """
    per_question = {}
    for question in questions:
        result = integration_service.ask(question)
        per_question[question] = {
            source.get("metadata", {}).get("source", "?")
            for source in result["sources"]
        }

    retrieved_nothing = [q for q, sources in per_question.items() if not sources]
    assert not retrieved_nothing, (
        "these phrasings fell below the relevance threshold, so they can only "
        f"ever be refused: {retrieved_nothing}"
    )

    shared = set.intersection(*per_question.values())
    assert shared, (
        "no document was retrieved for every phrasing, so the wording alone "
        f"decides what the bot reads: { {q: sorted(s) for q, s in per_question.items()} }"
    )
