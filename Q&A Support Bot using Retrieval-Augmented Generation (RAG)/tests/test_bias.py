"""Bias testing: the same question, six persona groups.

Markers: `integration`, `ai`, `bias`, so this never runs in the default suite.

    pytest -m bias
    pytest -m bias -s            print the disparity block
    pytest -m bias -k B02        one template

What is being asserted
--------------------------------------------------------------------------
That the facts in the answer do not change when the only thing that changed is
the asker's given name and pronouns. Not that the wording is identical - that
would fail on paraphrase and prove nothing.

Why the result is attributable here
--------------------------------------------------------------------------
The corpus is policy text. It contains no names, no genders and no locations,
so there is no legitimate reason for a retrieved fact to depend on them. Any
divergence has to have come from the model or the prompt, which is what makes
this a usable test rather than an argument about the documents.

That is a property of *this* corpus, not of the technique. The limitation is
written down in `evaluation/bias.py` and in docs/AI_TESTING_STRATEGY.md §7,
because a reader who carries this file into a system whose policies legitimately
differ between people would be measuring the policy, not the model.

What a failure means and does not mean
--------------------------------------------------------------------------
A failure is evidence that the answer depended on the persona in this run. With
six groups and one sample each it is not a quantified bias measurement, and
`disparity` is a count, not a rate with a confidence interval. Treated as a
smoke alarm: it fires on the thing worth investigating and does not claim to
size the fire.

Cost: six model calls per template, 18 for the file, all local.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from evaluation.bias import (
    PERSONAS,
    build_questions,
    evaluate_bias,
    failed_outcome,
    render_summary,
    score_outcome,
)
from evaluation.results import BiasResult, VariantOutcome
from evaluation.run_eval import load_dataset

pytestmark = [pytest.mark.integration, pytest.mark.ai, pytest.mark.bias]

ASK_URL = "/api/v1/ask"

CASES = load_dataset("bias")
CASE_IDS = [c["id"] for c in CASES]

_RESULTS_KEY = pytest.StashKey[List[BiasResult]]()


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
def bias_client(integration_client) -> ClientAdapter:
    return ClientAdapter(integration_client)


def _collect(case: Dict[str, Any], client: ClientAdapter) -> List[VariantOutcome]:
    expected_facts = case.get("expected_facts", [])
    forbidden = case.get("forbidden_phrases", [])

    outcomes: List[VariantOutcome] = []
    for group, question in build_questions(case["template"]):
        try:
            response = client.ask(question)
        except Exception as exc:
            outcomes.append(
                failed_outcome(group, question, f"{type(exc).__name__}: {exc}")
            )
            continue
        outcomes.append(
            score_outcome(group, question, response, expected_facts, forbidden)
        )
    return outcomes


def _report(result: BiasResult) -> str:
    lines = [
        "",
        f"{result.template_id} disparity {result.disparity:.0%} "
        f"(baseline: {result.baseline_group})",
        f"  {result.reason}",
    ]
    for outcome in result.outcomes:
        verdict = (
            f"ERROR {outcome.error}"
            if outcome.error
            else (
                f"facts={outcome.states_expected_facts} "
                f"refused={outcome.refused} gate={outcome.insufficient_context}"
            )
        )
        marker = " <-- diverged" if outcome.label in result.divergent_groups else ""
        lines.append(f"  [{outcome.label}] {verdict}{marker}")
        lines.append(f"      q: {outcome.question[:110]}")
        lines.append(f"      a: {outcome.answer[:160]!r}")
        lines.append(f"      retrieved: {', '.join(outcome.sources) or '(none)'}")

    lines.append(f"  {_attribution(result.outcomes)}")
    return "\n".join(lines)


def _attribution(outcomes: List[VariantOutcome]) -> str:
    """Retrieval or generation - the one thing the reader needs next.

    A persona token can change the answer two ways: by perturbing the query
    embedding so different chunks clear `relevance_threshold`, or by the model
    treating identical context differently. The fix is in the retriever in the
    first case and in the prompt in the second, so guessing wastes a day.
    """
    signatures = {tuple(o.sources) for o in outcomes if not o.error}
    if len(signatures) <= 1:
        return (
            "attribution: every group retrieved the same chunks, so any "
            "divergence came from generation"
        )
    return (
        f"attribution: {len(signatures)} distinct retrieval signatures across "
        "groups - the persona tokens changed which chunks were retrieved, so "
        "this is a retrieval effect and the threshold is where to look"
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_the_answer_does_not_depend_on_who_is_asking(case, bias_client, request):
    """One test per template, with every group's answer in the failure message.

    A bias failure is the kind nobody believes on a one-line assertion, so the
    report prints all six questions and answers side by side. The first thing
    anyone asks is "what did it actually say to each of them", and the answer
    has to be in the output rather than reproducible only by hand.
    """
    outcomes = _collect(case, bias_client)
    result = evaluate_bias(case["id"], outcomes)

    request.config.stash.setdefault(_RESULTS_KEY, []).append(result)

    assert result.passed, _report(result)


def test_disparity_summary(request):
    """Aggregate block. Asserts nothing the per-template tests did not."""
    results = request.config.stash.get(_RESULTS_KEY, [])
    if not results:
        pytest.skip("no bias templates were run in this session")

    print("\n" + render_summary(results))
    print(f"  groups: {', '.join(p.group for p in PERSONAS)}")
