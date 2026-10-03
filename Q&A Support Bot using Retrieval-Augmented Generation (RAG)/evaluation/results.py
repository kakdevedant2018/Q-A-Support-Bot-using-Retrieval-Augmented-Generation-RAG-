"""Result types shared by the evaluators.

Every evaluator returns a structured object rather than a bare bool. A bool
tells you a case failed; these tell you which fact was missing, which figure was
unsupported, and what the score was, which is what a reviewer actually needs to
triage a failing evaluation run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FactCheck:
    """The verdict on one expected fact."""

    fact_id: str
    statement: str
    passed: bool
    reason: str
    negated: bool = False
    matched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "statement": self.statement,
            "passed": self.passed,
            "reason": self.reason,
            "negated": self.negated,
            "matched": self.matched,
            "missing": self.missing,
        }


@dataclass
class FactResult:
    """The verdict across every expected fact for one answer."""

    passed: bool
    checks: List[FactCheck] = field(default_factory=list)
    forbidden_hits: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.checks:
            return 1.0
        return sum(1 for c in self.checks if c.passed) / len(self.checks)

    @property
    def failures(self) -> List[FactCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "forbidden_hits": self.forbidden_hits,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class GroundednessResult:
    """Whether an answer is supported by the context it was given."""

    passed: bool
    score: float
    unsupported_numbers: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "unsupported_numbers": self.unsupported_numbers,
            "unsupported_claims": self.unsupported_claims,
            "reason": self.reason,
        }


@dataclass
class SemanticResult:
    """Cosine similarity between an answer and a reference answer."""

    passed: bool
    similarity: float
    threshold: float
    negation_mismatch: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "similarity": round(self.similarity, 4),
            "threshold": self.threshold,
            "negation_mismatch": self.negation_mismatch,
            "reason": self.reason,
        }


@dataclass
class SafetyResult:
    """Whether a response violated a safety rule."""

    passed: bool
    violations: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "reason": self.reason,
        }


@dataclass
class JudgeResult:
    """Scores from the LLM-as-a-judge evaluator.

    `available` is False when no judge model was reachable. That is reported as
    "not evaluated", never silently as a pass.
    """

    available: bool
    groundedness: Optional[int] = None
    relevance: Optional[int] = None
    correctness: Optional[int] = None
    reason: str = ""
    raw: str = ""

    @property
    def passed(self) -> Optional[bool]:
        if not self.available:
            return None
        scores = [self.groundedness, self.relevance, self.correctness]
        if any(s is None for s in scores):
            return None
        return all(s >= 1 for s in scores)  # type: ignore[operator]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "groundedness": self.groundedness,
            "relevance": self.relevance,
            "correctness": self.correctness,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass
class VariantOutcome:
    """One question in a related group, and what the bot did with it.

    The unit of comparison for metamorphic and bias testing. Deliberately holds
    the fact verdict rather than the answer text, because the whole point of
    both techniques is that two correct answers to related questions will not
    be the same string.
    """

    label: str
    question: str
    answer: str
    insufficient_context: bool = False
    # Whether the bot declined, by either of the two routes it has: the
    # retrieval gate setting `insufficient_context`, or the model itself saying
    # it does not have enough information even though chunks cleared the
    # threshold. These are not the same event and the second one is common -
    # a loosely related chunk scores 0.66 and is retrieved, and the model
    # correctly declines anyway. Anything asserting on refusal has to read this
    # rather than `insufficient_context`, or it sees a refusal as an answer.
    refused: bool = False
    # "source@score" per retrieved chunk. Carried because the first two real
    # divergences this found were both caused by *which chunks were retrieved*
    # changing, not by the model changing its mind about the same context -
    # and without this field, telling those two apart meant re-running the
    # variants by hand through RAGService.retrieve.
    sources: List[str] = field(default_factory=list)
    facts: Optional[FactResult] = None
    error: Optional[str] = None

    @property
    def states_expected_facts(self) -> Optional[bool]:
        """None when nothing could be measured, which is never a pass."""
        if self.error:
            return None
        return self.facts.passed if self.facts else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "question": self.question,
            "answer": self.answer,
            "insufficient_context": self.insufficient_context,
            "refused": self.refused,
            "sources": self.sources,
            "states_expected_facts": self.states_expected_facts,
            "error": self.error,
            "facts": self.facts.to_dict() if self.facts else None,
        }


@dataclass
class MetamorphicResult:
    """Whether a metamorphic relation held across a group of related inputs.

    `passed` requires two separate things, and keeping them apart is the point:

    * `relation_holds` - the bot treated the related inputs consistently
    * `baseline_passed` - the base question was answered correctly at all

    A bot that refuses every question satisfies every invariance relation, so
    `relation_holds` alone is not evidence of anything. When the relation holds
    and the baseline failed, `consistent_but_wrong` is set: that is a retrieval
    or corpus defect surfacing through this suite, not an invariance defect, and
    reporting it as the latter sends the fix in the wrong direction.
    """

    relation_id: str
    kind: str
    relation: str
    relation_holds: bool
    baseline_passed: bool
    base: Optional[VariantOutcome] = None
    variants: List[VariantOutcome] = field(default_factory=list)
    divergences: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.relation_holds and self.baseline_passed

    @property
    def consistent_but_wrong(self) -> bool:
        return self.relation_holds and not self.baseline_passed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "kind": self.kind,
            "relation": self.relation,
            "passed": self.passed,
            "relation_holds": self.relation_holds,
            "baseline_passed": self.baseline_passed,
            "consistent_but_wrong": self.consistent_but_wrong,
            "divergences": self.divergences,
            "reason": self.reason,
            "base": self.base.to_dict() if self.base else None,
            "variants": [v.to_dict() for v in self.variants],
        }


@dataclass
class BiasResult:
    """Whether the stated policy changed when only the persona changed.

    The corpus is policy text that says nothing about who is asking, so every
    group must receive the same facts. Any divergence is attributable to the
    model, not the documents - which is what makes this measurable here and
    hard to measure in a system whose corpus is itself about people.
    """

    template_id: str
    baseline_group: str
    passed: bool
    outcomes: List[VariantOutcome] = field(default_factory=list)
    divergent_groups: List[str] = field(default_factory=list)
    reason: str = ""

    @property
    def disparity(self) -> float:
        """Fraction of non-baseline groups whose fact verdict differed."""
        comparable = [o for o in self.outcomes if o.label != self.baseline_group]
        if not comparable:
            return 0.0
        return len(self.divergent_groups) / len(comparable)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "baseline_group": self.baseline_group,
            "passed": self.passed,
            "disparity": round(self.disparity, 4),
            "divergent_groups": self.divergent_groups,
            "reason": self.reason,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


@dataclass
class CaseResult:
    """Everything known about one evaluated question."""

    case_id: str
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    insufficient_context: bool = False
    facts: Optional[FactResult] = None
    groundedness: Optional[GroundednessResult] = None
    semantic: Optional[SemanticResult] = None
    safety: Optional[SafetyResult] = None
    judge: Optional[JudgeResult] = None
    retrieval_hit: Optional[bool] = None
    latency_seconds: Optional[float] = None
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        """A case passes only if every evaluator that ran passed.

        The judge is excluded from the gate on purpose. It is advisory: a small
        local model is not reliable enough to fail a build on its own. See
        docs/AI_TESTING_STRATEGY.md.
        """
        if self.error:
            return False
        gates = [
            self.facts.passed if self.facts else None,
            self.groundedness.passed if self.groundedness else None,
            self.semantic.passed if self.semantic else None,
            self.safety.passed if self.safety else None,
        ]
        return all(g for g in gates if g is not None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "insufficient_context": self.insufficient_context,
            "passed": self.passed,
            "retrieval_hit": self.retrieval_hit,
            "latency_seconds": (
                round(self.latency_seconds, 3)
                if self.latency_seconds is not None
                else None
            ),
            "error": self.error,
            "facts": self.facts.to_dict() if self.facts else None,
            "groundedness": (
                self.groundedness.to_dict() if self.groundedness else None
            ),
            "semantic": self.semantic.to_dict() if self.semantic else None,
            "safety": self.safety.to_dict() if self.safety else None,
            "judge": self.judge.to_dict() if self.judge else None,
        }
