"""Groundedness: is the answer supported by the context it was given?

For a RAG system this matters more than similarity to a reference answer. The
knowledge base is the source of truth, so the question to ask is not "does this
match what I wrote down" but "does the retrieved text support every claim made".

Two deterministic signals, no model and no cost:

**Numeric support.** Every figure in the answer must appear in the retrieved
context. This is the highest-value check in the whole framework, because an
invented policy almost always carries an invented number. The context says 30
days; an answer saying 90 days is fluent, well formed, returned with a 200, and
wrong. This catches it without a judge and without exact-match assertions.

**Claim support.** Each sentence is scored on how much of its content vocabulary
appears in the context. A sentence that shares almost nothing with the retrieved
text is a candidate hallucination and is reported for review.

Claim overlap is the weaker of the two: a correct paraphrase using different
words scores low, and a fabrication reusing context vocabulary scores high. It
is reported as a score and a list of suspect sentences, and the pass gate leans
on the numeric check. The LLM judge in `llm_judge.py` covers what lexical
overlap cannot.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from evaluation.normalize import (
    content_words,
    extract_numbers,
    sentences,
)
from evaluation.results import GroundednessResult

# Deliberately lenient. Claim overlap exists to surface candidate hallucinations
# for review, not to decide the build: a correct paraphrase that reuses none of
# the source's vocabulary scores low through no fault of its own. The numeric
# check is the strict gate. Both values are calibration parameters - measure them
# on your own corpus rather than treating them as constants.
DEFAULT_MIN_CLAIM_OVERLAP = 0.3
DEFAULT_MIN_CLAIM_SCORE = 0.5

# Sentences that carry no factual claim of their own. A support answer routinely
# ends with one, and it would score zero on overlap without being a
# hallucination.
#
# The second group is the one that cost a false positive. Asked to invent a
# policy, the bot answered "I'm not allowed to invent policy details. I can only
# provide information based on the context provided." - exactly the behaviour
# wanted - and scored 0.00 on claim overlap, because a sentence about the
# assistant's own limits shares no vocabulary with a document about refunds. The
# class to exempt is a first-person statement about what the assistant may do,
# not the particular wording it happened to use; matching only the exact strings
# leaves the same bug waiting for the next phrasing.
_BOILERPLATE_MARKERS = (
    # Closing pleasantries and attribution.
    "contact support",
    "contact a support agent",
    "let me know",
    "hope this help",
    "happy to help",
    "anything else",
    "based on the context",
    "based on the provided context",
    "according to the context",
    # Statements about the assistant's own capability or permission.
    "i do not have enough information",
    "i don't have enough information",
    "i am not allowed",
    "i'm not allowed",
    "i am not able",
    "i'm not able",
    "i am unable",
    "i'm unable",
    "i cannot provide",
    "i can't provide",
    "i cannot invent",
    "i can't invent",
    "i can only provide",
    "i can only answer",
    "i do not have access",
    "i don't have access",
    # Statements that the source material is silent on something. "The FAQ does
    # not mention a restocking fee" is a correct answer when the corpus indeed
    # does not, but a claim about an *absence* can never overlap the context
    # lexically - the words it needs are the ones that are missing. Overlap is
    # simply the wrong instrument for this sentence type. Whether such a claim is
    # true is checked elsewhere: the hallucination dataset asserts forbidden
    # substrings, and the fact checks assert what must be present.
    "does not mention",
    "do not mention",
    "does not specify",
    "do not specify",
    "does not state",
    "does not say",
    "does not cover",
    "is not mentioned",
    "are not mentioned",
    "no mention of",
    "no information about",
)


def _is_boilerplate(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(marker in lowered for marker in _BOILERPLATE_MARKERS)


def join_context(sources: Sequence[str]) -> str:
    """Flatten retrieved chunk texts into the context the answer was built from."""
    return "\n\n".join(s for s in sources if s)


def evaluate_groundedness(
    answer: str,
    context: str,
    *,
    question: str = "",
    is_refusal: bool = False,
    min_claim_overlap: float = DEFAULT_MIN_CLAIM_OVERLAP,
    min_claim_score: float = DEFAULT_MIN_CLAIM_SCORE,
) -> GroundednessResult:
    """Check an answer against the context that produced it.

    `is_refusal` is for the honest "I don't have enough information" path, where
    there is legitimately no context. That answer is grounded as long as it
    asserts no figures of its own.

    `question` matters more than it looks. Asked "can I return a product 45 days
    after delivery?", a correct answer says no and cites the 30 day window, and
    it may well repeat the 45. That figure came from the user, so counting it as
    an invented fact would fail the bot for quoting the question back. Numbers
    present in the question are therefore permitted.
    """
    answer = answer or ""
    question_numbers = set(extract_numbers(question)) if question else set()

    if is_refusal:
        invented = [n for n in extract_numbers(answer) if n not in question_numbers]
        if invented:
            return GroundednessResult(
                passed=False,
                score=0.0,
                unsupported_numbers=invented,
                reason=(
                    "the answer declined to answer yet still stated figures: "
                    f"{', '.join(invented)}"
                ),
            )
        return GroundednessResult(
            passed=True,
            score=1.0,
            reason="refusal with no asserted facts, nothing to ground",
        )

    if not context.strip():
        return GroundednessResult(
            passed=False,
            score=0.0,
            unsupported_claims=sentences(answer),
            reason="an answer was produced with no retrieved context at all",
        )

    permitted_numbers = set(extract_numbers(context)) | question_numbers
    answer_numbers = extract_numbers(answer)
    unsupported_numbers = sorted(
        {n for n in answer_numbers if n not in permitted_numbers}
    )

    context_vocabulary = content_words(context)
    unsupported_claims: List[str] = []
    measured = 0

    for sentence in sentences(answer):
        if _is_boilerplate(sentence):
            continue
        words = content_words(sentence)
        if len(words) < 3:
            continue

        measured += 1
        overlap = len(words & context_vocabulary) / len(words)
        if overlap < min_claim_overlap:
            unsupported_claims.append(sentence)

    claim_score = 1.0 if measured == 0 else (measured - len(unsupported_claims)) / measured

    numeric_ok = not unsupported_numbers
    claims_ok = claim_score >= min_claim_score
    passed = numeric_ok and claims_ok

    if passed:
        reason = "every figure appears in the context and claims overlap it"
    elif not numeric_ok:
        reason = (
            "the answer states figures that appear nowhere in the retrieved "
            f"context: {', '.join(unsupported_numbers)}"
        )
    else:
        reason = (
            f"claim support {claim_score:.2f} is below {min_claim_score}; "
            f"{len(unsupported_claims)} sentence(s) share little with the context"
        )

    # A numeric violation is the strong signal, so it dominates the score.
    score = 0.0 if not numeric_ok else claim_score

    return GroundednessResult(
        passed=passed,
        score=score,
        unsupported_numbers=unsupported_numbers,
        unsupported_claims=unsupported_claims,
        reason=reason,
    )


def unsupported_figures(answer: str, context: str, question: str = "") -> List[str]:
    """Just the numeric check, for callers that want the one strong signal."""
    permitted = set(extract_numbers(context)) | set(
        extract_numbers(question) if question else []
    )
    return sorted({n for n in extract_numbers(answer) if n not in permitted})


def evaluate_from_sources(
    answer: str,
    sources: Sequence[str],
    *,
    question: str = "",
    is_refusal: bool = False,
    min_claim_overlap: Optional[float] = None,
) -> GroundednessResult:
    """Convenience wrapper that grounds an answer against its own citations."""
    return evaluate_groundedness(
        answer,
        join_context(sources),
        question=question,
        is_refusal=is_refusal,
        min_claim_overlap=(
            DEFAULT_MIN_CLAIM_OVERLAP if min_claim_overlap is None else min_claim_overlap
        ),
    )
