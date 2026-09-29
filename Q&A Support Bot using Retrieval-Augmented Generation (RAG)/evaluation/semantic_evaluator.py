"""Semantic evaluation: compare meaning, not wording.

Two answers can share almost no vocabulary and mean the same thing:

    Customers can return products within 30 days.
    You have thirty days from delivery to send an item back.

Cosine similarity between embeddings scores those as close. It reuses the
embedding model the application already depends on, so it runs locally, needs no
judge model, and costs nothing.

The limitation is important enough to state next to the code, because it is the
classic way a semantic-only suite passes a wrong answer:

    Customers can return products within 30 days.      (correct)
    Customers cannot return products after 30 days.    (different rule)

Those two are lexically near-identical, so their embeddings are very close.
Similarity alone would call the second one a pass. `negation_mismatch` guards
the specific case by anchoring on shared numbers and policy verbs and checking
whether they carry a negation in one text but not the other.

Use this as one signal. The deterministic fact and groundedness checks stay the
primary gates; similarity is for open-ended answers where no exact phrase can be
required.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from evaluation.normalize import extract_numbers, is_negated, normalize
from evaluation.results import SemanticResult

DEFAULT_THRESHOLD = 0.72

EmbedFn = Callable[[str], Sequence[float]]

# Verbs that carry the policy in a support answer. If one text negates one of
# these and the other does not, the two are not saying the same thing however
# close their embeddings are.
_POLICY_VERBS = (
    "return",
    "refund",
    "cancel",
    "replace",
    "ship",
    "charge",
    "close",
    "change",
    "track",
    "report",
    "accept",
    "extend",
)


def _default_embed() -> EmbedFn:
    """Resolve the application's embedding model, imported lazily.

    Deferred on purpose: importing this module must not pull in torch, so the
    fast mocked test suite can use the evaluators with stub embeddings.
    """
    from app.services.embeddings import get_embeddings

    embeddings = get_embeddings()
    return embeddings.embed_query


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity without a numpy dependency.

    The application normalises embeddings to unit length, which would make a dot
    product sufficient, but the full form keeps this correct if that setting
    ever changes.
    """
    if not left or not right or len(left) != len(right):
        raise ValueError("embeddings must be non-empty and the same length")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _shared_anchors(left: str, right: str) -> List[str]:
    """Tokens present in both texts whose polarity decides the meaning."""
    anchors: List[str] = []

    left_numbers = set(extract_numbers(left))
    right_numbers = set(extract_numbers(right))
    anchors.extend(sorted(left_numbers & right_numbers))

    left_words = set(normalize(left).split(" "))
    right_words = set(normalize(right).split(" "))
    for verb in _POLICY_VERBS:
        if verb in left_words and verb in right_words:
            anchors.append(verb)

    return anchors


def detect_negation_mismatch(left: str, right: str) -> bool:
    """Whether a shared anchor is negated in one text but not the other."""
    for anchor in _shared_anchors(left, right):
        if is_negated(left, anchor) != is_negated(right, anchor):
            return True
    return False


def semantic_similarity(
    left: str, right: str, *, embed: Optional[EmbedFn] = None
) -> float:
    """Cosine similarity between two texts, in [-1, 1]."""
    embed_fn = embed or _default_embed()
    return cosine_similarity(embed_fn(left), embed_fn(right))


def evaluate_semantic(
    answer: str,
    reference: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    embed: Optional[EmbedFn] = None,
) -> SemanticResult:
    """Score an answer against a reference answer by meaning.

    The threshold is a calibration parameter, not a constant of nature. Measure
    it on your own dataset with:

        python -m evaluation.calibrate
    """
    if not answer.strip():
        return SemanticResult(
            passed=False,
            similarity=0.0,
            threshold=threshold,
            reason="answer is empty",
        )

    similarity = semantic_similarity(answer, reference, embed=embed)
    mismatch = detect_negation_mismatch(answer, reference)

    if mismatch:
        return SemanticResult(
            passed=False,
            similarity=similarity,
            threshold=threshold,
            negation_mismatch=True,
            reason=(
                f"similarity {similarity:.3f} is high but a shared claim is "
                "negated in one text and not the other, so the two state "
                "different rules"
            ),
        )

    passed = similarity >= threshold
    return SemanticResult(
        passed=passed,
        similarity=similarity,
        threshold=threshold,
        reason=(
            f"similarity {similarity:.3f} "
            f"{'meets' if passed else 'is below'} the threshold {threshold}"
        ),
    )
