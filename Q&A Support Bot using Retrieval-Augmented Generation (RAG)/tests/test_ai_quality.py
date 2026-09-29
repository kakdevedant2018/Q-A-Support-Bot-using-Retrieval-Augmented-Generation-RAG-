"""Dataset-driven answer quality, measured against the real system.

Markers: `integration` and `ai`, so this never runs in the default suite.

    pytest -m "ai"             answer quality
    pytest -m "ai and not judge"   skip the slow model-graded cases

The difference between this file and `test_rag_quality.py`: that one checks the
retrieval mechanism, this one scores answers using the evaluation framework, so
no assertion here depends on the model choosing a particular wording. Every check
routes through `evaluation/`, which is itself unit tested in
`test_evaluation_framework.py`.

One test per dataset case is intentional. A single test looping over twelve
questions reports one failure and hides which question regressed; parametrising
gives twelve named results, and `pytest -k F07` reruns exactly the one that
broke.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from evaluation.report import Scorecard, failure_reasons, render_console
from evaluation.results import CaseResult
from evaluation.retrieval_metrics import RetrievalCase, score_retrieval
from evaluation.run_eval import evaluate_one, load_dataset

pytestmark = [pytest.mark.integration, pytest.mark.ai]

ASK_URL = "/api/v1/ask"

FACTUAL_CASES = load_dataset("factual")
OPEN_ENDED_CASES = load_dataset("open_ended")


def _ids(cases: List[Dict[str, Any]]) -> List[str]:
    return [c["id"] for c in cases]


class ClientAdapter:
    """Adapts a FastAPI TestClient to the interface the evaluators expect."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def ask(self, question: str) -> Dict[str, Any]:
        response = self._client.post(ASK_URL, json={"question": question})
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        return response.json()


@pytest.fixture
def eval_client(integration_client) -> ClientAdapter:
    return ClientAdapter(integration_client)


# --------------------------------------------------------------------------
# Retrieval quality. Needs the embedding model but not Ollama.
# --------------------------------------------------------------------------


def test_retrieval_recall_on_the_labelled_dataset(integration_service):
    """Recall@k against the document each question is labelled with.

    This runs first because answer quality is capped by it. If the chunk holding
    the answer is never retrieved, no prompt and no model can produce a grounded
    answer, and scoring answers alone would report a vague quality problem
    instead of the actual cause.
    """
    cases: List[RetrievalCase] = []
    for case in FACTUAL_CASES:
        matches = integration_service.retrieve(case["question"])
        ranked = [doc.metadata.get("source", "") for doc, _ in matches]
        cases.append(
            RetrievalCase(
                case_id=case["id"],
                expected_sources=set(case["expected_sources"]),
                ranked_sources=ranked,
            )
        )

    metrics = score_retrieval(cases)
    print(
        f"\nRecall@k {metrics.recall_at_k:.1%}  "
        f"Precision@k {metrics.precision_at_k:.1%}  MRR {metrics.mrr:.3f}"
    )
    if metrics.misses:
        print(f"missed: {', '.join(metrics.misses)}")

    # Deliberately not 100%. This is the first measurement on this corpus, and a
    # target should be set from the baseline it produces, not asserted before it
    # has ever been measured.
    assert metrics.recall_at_k >= 0.75, (
        f"retrieval found the labelled document for only "
        f"{metrics.recall_at_k:.0%} of questions; missed {metrics.misses}"
    )


def test_every_retrieved_chunk_names_a_real_document(integration_service):
    known = {"faq.txt", "support_policy.txt"}
    for case in FACTUAL_CASES[:3]:
        for doc, _ in integration_service.retrieve(case["question"]):
            assert doc.metadata.get("source") in known


# --------------------------------------------------------------------------
# Answer quality. Needs Ollama.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", FACTUAL_CASES, ids=_ids(FACTUAL_CASES))
def test_factual_answer_states_the_expected_facts(eval_client, require_ollama, case):
    """Fact-level, wording-independent correctness.

    "thirty days", "a 30-day window", and "30 calendar days" all satisfy the same
    expectation, so this measures whether the bot communicated the fact rather
     than whether it produced a particular sentence.
    """
    result = evaluate_one(case, eval_client, use_semantic=False)

    assert not result.error, result.error
    assert result.facts is not None
    assert result.facts.passed, (
        f"{case['id']} answer did not state the expected facts.\n"
        f"answer: {result.answer!r}\n"
        + "\n".join(f"  - {r}" for r in failure_reasons(result))
    )


@pytest.mark.parametrize("case", FACTUAL_CASES, ids=_ids(FACTUAL_CASES))
def test_factual_answer_is_grounded_in_its_sources(eval_client, require_ollama, case):
    """No figure may appear in the answer that is absent from the context.

    This is the hallucination gate. It does not care how the answer is phrased,
    only that it invents nothing.
    """
    result = evaluate_one(case, eval_client, use_semantic=False)

    assert result.groundedness is not None
    assert result.groundedness.unsupported_numbers == [], (
        f"{case['id']} asserted unsupported figures "
        f"{result.groundedness.unsupported_numbers}: {result.answer!r}"
    )


@pytest.mark.parametrize("case", OPEN_ENDED_CASES, ids=_ids(OPEN_ENDED_CASES))
def test_open_ended_answer_covers_the_required_points(eval_client, require_ollama, case):
    """Open-ended questions have no single right sentence, only required content."""
    result = evaluate_one(case, eval_client, use_semantic=False)

    assert not result.error, result.error
    assert result.facts is not None
    assert result.facts.passed, (
        f"{case['id']} missed required content.\nanswer: {result.answer!r}\n"
        + "\n".join(f"  - {r}" for r in failure_reasons(result))
    )


def test_semantic_similarity_separates_right_from_wrong(eval_client, require_ollama):
    """The real embedding model, not the stub used in the framework's own tests.

    Checks the property the threshold depends on: the bot's answer must be closer
    to its own reference answer than to a different question's reference answer.
    Relative separation is asserted; the absolute threshold is calibration, and
    `python -m evaluation.calibrate --semantic` measures it.
    """
    from evaluation.semantic_evaluator import semantic_similarity

    first, second = FACTUAL_CASES[0], FACTUAL_CASES[9]

    answer = eval_client.ask(first["question"])["answer"]
    own = semantic_similarity(answer, first["reference_answer"])
    other = semantic_similarity(answer, second["reference_answer"])

    print(f"\nmatched {own:.3f}  mismatched {other:.3f}")
    assert own > other


def test_out_of_scope_question_is_refused_without_calling_the_model(integration_service):
    """The refusal path, proved at the service level with no generation at all."""
    assert integration_service.retrieve("What is the weather on Mars today?") == []


def test_full_scorecard_over_the_factual_dataset(eval_client, require_ollama):
    """One run, one scorecard. This is the number to track between changes.

    Prints correctness, groundedness, hallucination count, and latency
    percentiles. The assertion is loose on purpose: this is a baseline-setting
    measurement, and the same numbers are produced for a report by
    `python -m evaluation.run_eval`.
    """
    results: List[CaseResult] = [
        evaluate_one(case, eval_client, use_semantic=False) for case in FACTUAL_CASES
    ]
    scorecard = Scorecard(cases=results, dataset_names=["factual"])
    print(render_console(scorecard))

    assert not scorecard.errors, "some cases could not be evaluated at all"
    assert not [
        c
        for c in results
        if c.groundedness and c.groundedness.unsupported_numbers
    ], "at least one answer contained an invented figure"


@pytest.mark.judge
def test_llm_judge_scores_a_good_answer(eval_client, require_ollama):
    """The judge is advisory, so this tests the mechanism, not the bot.

    It asserts that a judge model can be reached and returns a valid structured
    verdict. It does not assert a minimum quality score: a small local model
    grading its own output is not a defensible gate, and calibration against
    human review has to come first.
    """
    from evaluation.groundedness_evaluator import join_context
    from evaluation.llm_judge import judge_answer, judge_model_name

    case = FACTUAL_CASES[0]
    response = eval_client.ask(case["question"])
    context = join_context([s["content"] for s in response["sources"]])

    verdict = judge_answer(case["question"], context, response["answer"])

    print(f"\njudge model: {judge_model_name()}")
    print(f"scores: {verdict.to_dict()}")

    if not verdict.available:
        pytest.skip(f"judge produced no usable verdict: {verdict.reason}")

    for score in (verdict.groundedness, verdict.relevance, verdict.correctness):
        assert score in (0, 1, 2)
