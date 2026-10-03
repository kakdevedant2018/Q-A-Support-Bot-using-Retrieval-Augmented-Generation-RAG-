"""Metamorphic relations, checked against the real system.

Markers: `integration`, `ai`, `metamorphic`, so this never runs in the default
suite.

    pytest -m metamorphic
    pytest -m metamorphic -k MR06        one relation
    pytest -m metamorphic -s             print the summary block

Why this file exists next to `test_ai_quality.py`
--------------------------------------------------------------------------
`test_ai_quality.py` asks one question and checks the answer against expected
facts. That needs a labelled answer for every question, which is exactly the
cost that keeps those datasets at a dozen cases. A metamorphic relation needs no
new labels at all: it reuses one labelled base question and asserts that *the
same facts come back* under a transformation that should not have changed them.
Adding a phrasing is free, which is why MR01 covers five phrasings where the
factual dataset covers one.

It also catches a different class of defect. A question-at-a-time dataset cannot
see a bot that answers "What is the return period?" correctly and
"WHAT IS THE RETURN PERIOD?" with a refusal, because both answers are scored
against the corpus and never against each other. Inconsistency *is* the defect
here, and it is only visible in the comparison.

The oracle problem, concretely
--------------------------------------------------------------------------
The usual objection to testing a generative system is that there is no expected
output. A metamorphic relation sidesteps it: it does not need to know what the
right answer is, only that two related questions must get answers that agree.
That is why these relations survive a corpus edit that would invalidate a
reference answer.

Flakiness, and why these tests are not flaky
--------------------------------------------------------------------------
Every assertion compares *fact verdicts*, never answer text. The model is at
temperature 0 but is not bit-deterministic, and two correct answers to the same
question routinely share little vocabulary. A string comparison here would fail
on paraphrase and teach everyone to ignore the suite.

Cost: one model call per variant, around 35 calls for the whole file, all local.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from evaluation.metamorphic import (
    build_questions,
    evaluate_relation,
    failed_outcome,
    render_summary,
    score_outcome,
)
from evaluation.results import MetamorphicResult, VariantOutcome
from evaluation.run_eval import load_dataset

pytestmark = [pytest.mark.integration, pytest.mark.ai, pytest.mark.metamorphic]

ASK_URL = "/api/v1/ask"

CASES = load_dataset("metamorphic")
CASE_IDS = [c["id"] for c in CASES]

# Results accumulate on the session stash rather than a module global, so that
# the summary still works under xdist and under `-k`.
_RESULTS_KEY = pytest.StashKey[List[MetamorphicResult]]()


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
def relation_client(integration_client) -> ClientAdapter:
    return ClientAdapter(integration_client)


def _collect(case: Dict[str, Any], client: ClientAdapter) -> List[VariantOutcome]:
    """Ask every variant of one relation and score each answer independently.

    A transport error becomes a failed outcome rather than an exception, so the
    relation still reports which variant broke instead of the whole test
    aborting at the first one.
    """
    expected_facts = case.get("expected_facts", [])
    forbidden = case.get("forbidden_phrases", [])

    outcomes: List[VariantOutcome] = []
    for label, question in build_questions(case):
        try:
            response = client.ask(question)
        except Exception as exc:
            outcomes.append(
                failed_outcome(label, question, f"{type(exc).__name__}: {exc}")
            )
            continue
        outcomes.append(
            score_outcome(label, question, response, expected_facts, forbidden)
        )
    return outcomes


def _report(result: MetamorphicResult) -> str:
    lines = [
        "",
        f"{result.relation_id} [{result.kind}/{result.relation}] "
        f"relation_holds={result.relation_holds} baseline_passed={result.baseline_passed}",
        f"  {result.reason}",
    ]
    # The base is included first. Printing only the variants leaves the reader
    # comparing them against something they cannot see.
    shown = ([result.base] if result.base else []) + list(result.variants)
    for outcome in shown:
        verdict = (
            f"ERROR {outcome.error}"
            if outcome.error
            else (
                f"facts={outcome.states_expected_facts} "
                f"refused={outcome.refused} gate={outcome.insufficient_context}"
            )
        )
        lines.append(f"  [{outcome.label}] {verdict}")
        lines.append(f"      q: {outcome.question[:100]}")
        lines.append(f"      a: {outcome.answer[:160]!r}")
        # Retrieval or generation. A variant that retrieved different chunks
        # from the base is a retriever problem; one that retrieved the same
        # chunks and answered differently is a prompt problem.
        lines.append(f"      retrieved: {', '.join(outcome.sources) or '(none)'}")
    return "\n".join(lines)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_metamorphic_relation_holds(case, relation_client, request):
    """One test per relation, so a failure names the relation that broke.

    The failure message carries every variant's question, answer and verdict.
    Without that, "MR04 failed" sends you to re-run six questions by hand to
    find out which transformation did it.
    """
    outcomes = _collect(case, relation_client)
    result = evaluate_relation(case, outcomes)

    # Stashed so the module-level summary can aggregate across relations.
    request.config.stash.setdefault(_RESULTS_KEY, []).append(result)

    assert result.passed, _report(result)


def test_summary_across_all_relations(request):
    """Prints the aggregate block and asserts nothing new.

    Runs last by file order and exists because the per-relation tests answer
    "did this break" while a release decision needs "how consistent is it
    overall". Skipped rather than failed when no relation ran, since `-k MR06`
    is a legitimate way to use this file.
    """
    results = request.config.stash.get(_RESULTS_KEY, [])
    if not results:
        pytest.skip("no relations were run in this session")

    print("\n" + render_summary(results))

    consistent_but_wrong = [r.relation_id for r in results if r.consistent_but_wrong]
    if consistent_but_wrong:
        # Deliberately not an assertion. These relations held - the bot was
        # consistent - and the base question being wrong is a correctness
        # defect that `test_ai_quality.py` owns and gates. Failing it twice in
        # two files would send whoever is fixing it to the wrong one.
        print(
            "\nNOTE: consistent but wrong (fix in the factual dataset or the "
            f"corpus, not here): {', '.join(consistent_but_wrong)}"
        )
