"""Fact-level evaluation: does the answer communicate the required facts?

This replaces `assert "30 days" in answer`, which fails on "thirty days" even
though the answer is correct. The unit of expectation here is a *fact*, not a
sentence:

    {
      "id": "return-window",
      "statement": "The return period is 30 days.",
      "all_of": ["30 days"],
      "any_of": ["return", "send back"]
    }

`all_of` phrases must all appear and `any_of` needs one match, both compared
after normalisation, so "thirty days", "30-day", and "30 calendar days" all
satisfy the same expectation.

Two things this catches that a substring check does not:

* wording variation, through normalisation
* polarity, through negation detection - "customers cannot return products
  after 30 days" contains "30 day" and must not be scored as confirming the
  return window

This evaluator is deterministic, needs no model, and costs nothing, so it is
the first gate. Only use semantic or judge-based evaluation for what this
cannot express.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from evaluation.normalize import contains_phrase, is_negated
from evaluation.results import FactCheck, FactResult

POSITIVE = "positive"
NEGATIVE = "negative"


def _check_one_fact(answer: str, fact: Dict[str, Any]) -> FactCheck:
    fact_id = str(fact.get("id", "unnamed"))
    statement = str(fact.get("statement", fact_id))
    polarity = str(fact.get("polarity", POSITIVE)).lower()

    all_of: List[str] = list(fact.get("all_of", []) or [])
    any_of: List[str] = list(fact.get("any_of", []) or [])

    if not all_of and not any_of:
        return FactCheck(
            fact_id=fact_id,
            statement=statement,
            passed=False,
            reason="dataset error: fact declares neither all_of nor any_of",
        )

    matched: List[str] = []
    missing: List[str] = []

    for phrase in all_of:
        if contains_phrase(answer, phrase):
            matched.append(phrase)
        else:
            missing.append(phrase)

    any_matched: Optional[str] = None
    if any_of:
        for phrase in any_of:
            if contains_phrase(answer, phrase):
                any_matched = phrase
                matched.append(phrase)
                break

    # Polarity is checked against the phrases that carry the fact, which are the
    # all_of phrases when present.
    anchors = all_of or ([any_matched] if any_matched else [])
    negated = any(is_negated(answer, phrase) for phrase in anchors if phrase)

    if missing:
        return FactCheck(
            fact_id=fact_id,
            statement=statement,
            passed=False,
            reason=f"answer does not state: {', '.join(repr(m) for m in missing)}",
            negated=negated,
            matched=matched,
            missing=missing,
        )

    if any_of and any_matched is None:
        return FactCheck(
            fact_id=fact_id,
            statement=statement,
            passed=False,
            reason=(
                "answer contains none of the required alternatives: "
                f"{', '.join(repr(p) for p in any_of)}"
            ),
            negated=negated,
            matched=matched,
        )

    expected_negated = polarity == NEGATIVE
    if negated != expected_negated:
        if negated:
            reason = (
                "the required phrase appears but is negated, so the answer "
                "states the opposite of the expected fact"
            )
        else:
            reason = (
                "the fact was expected to be stated negatively but the answer "
                "asserts it positively"
            )
        return FactCheck(
            fact_id=fact_id,
            statement=statement,
            passed=False,
            reason=reason,
            negated=negated,
            matched=matched,
        )

    return FactCheck(
        fact_id=fact_id,
        statement=statement,
        passed=True,
        reason="every required phrase is present with the expected polarity",
        negated=negated,
        matched=matched,
    )


def evaluate_facts(
    answer: str,
    expected_facts: Iterable[Dict[str, Any]],
    forbidden_phrases: Optional[Iterable[str]] = None,
) -> FactResult:
    """Score an answer against its expected facts.

    `forbidden_phrases` is the other half of the check. An answer can contain
    every required fact and still be wrong because it added one: "returns are
    accepted within 30 days, or 90 days for international orders" satisfies the
    30-day expectation while inventing a policy. Normalisation applies here too,
    so "ninety days" is caught by a "90 days" entry.
    """
    checks = [_check_one_fact(answer, fact) for fact in expected_facts]

    forbidden_hits = [
        phrase
        for phrase in (forbidden_phrases or [])
        if contains_phrase(answer, phrase)
    ]

    passed = all(check.passed for check in checks) and not forbidden_hits
    return FactResult(passed=passed, checks=checks, forbidden_hits=forbidden_hits)
