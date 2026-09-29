"""Retrieval metrics: is the right document being found at all?

Answer quality is bounded by retrieval quality. If the chunk containing the
return policy is never retrieved, no prompt and no model can produce a grounded
answer, and a suite that only scores answers will report a vague quality problem
instead of the actual cause.

These are the deterministic equivalents of the context metrics an evaluation
framework such as RAGAS reports. They need labelled data - each question tagged
with the document that should answer it - and no model beyond the embeddings the
retriever already uses, so they are free and exactly reproducible.

    Recall@k     did the expected document appear in the top k
    Precision@k  what share of the retrieved chunks were from expected documents
    MRR          how highly the first correct document ranked

Recall is the one to gate on. Precision@k is low by construction when k exceeds
the number of relevant chunks, so read it as a trend, not a target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


@dataclass
class RetrievalCase:
    """One labelled retrieval observation."""

    case_id: str
    expected_sources: Set[str]
    ranked_sources: List[str]

    @property
    def hit(self) -> bool:
        return any(source in self.expected_sources for source in self.ranked_sources)

    @property
    def first_hit_rank(self) -> int:
        """1-indexed rank of the first expected document, or 0 if absent."""
        for index, source in enumerate(self.ranked_sources, start=1):
            if source in self.expected_sources:
                return index
        return 0

    @property
    def precision(self) -> float:
        if not self.ranked_sources:
            return 0.0
        hits = sum(1 for s in self.ranked_sources if s in self.expected_sources)
        return hits / len(self.ranked_sources)


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval quality across a labelled dataset."""

    total: int = 0
    hits: int = 0
    precision_sum: float = 0.0
    reciprocal_rank_sum: float = 0.0
    misses: List[str] = field(default_factory=list)

    @property
    def recall_at_k(self) -> float:
        return 0.0 if self.total == 0 else self.hits / self.total

    @property
    def precision_at_k(self) -> float:
        return 0.0 if self.total == 0 else self.precision_sum / self.total

    @property
    def mrr(self) -> float:
        return 0.0 if self.total == 0 else self.reciprocal_rank_sum / self.total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cases": self.total,
            "recall_at_k": round(self.recall_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "misses": self.misses,
        }


def score_retrieval(cases: Iterable[RetrievalCase]) -> RetrievalMetrics:
    """Aggregate a set of labelled retrieval observations."""
    metrics = RetrievalMetrics()

    for case in cases:
        metrics.total += 1
        metrics.precision_sum += case.precision

        rank = case.first_hit_rank
        if rank:
            metrics.hits += 1
            metrics.reciprocal_rank_sum += 1.0 / rank
        else:
            metrics.misses.append(case.case_id)

    return metrics


def case_from_response(
    case_id: str,
    expected_sources: Sequence[str],
    response: Dict[str, Any],
) -> RetrievalCase:
    """Build a RetrievalCase from an /ask response body.

    Ranked order is the order the API returns sources in, which is the retriever's
    relevance order.
    """
    ranked = [
        source.get("metadata", {}).get("source", "")
        for source in response.get("sources", [])
    ]
    return RetrievalCase(
        case_id=case_id,
        expected_sources=set(expected_sources),
        ranked_sources=[r for r in ranked if r],
    )


def summarise(metrics: RetrievalMetrics) -> List[Tuple[str, str]]:
    """Label and value pairs for report rendering."""
    return [
        ("Retrieval cases", str(metrics.total)),
        ("Recall@k", f"{metrics.recall_at_k * 100:.1f}%"),
        ("Precision@k", f"{metrics.precision_at_k * 100:.1f}%"),
        ("MRR", f"{metrics.mrr:.3f}"),
    ]
