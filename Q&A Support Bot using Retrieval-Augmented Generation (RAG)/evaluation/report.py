"""Scorecard aggregation and report rendering.

An evaluation run produces a number, and a single number is only useful next to
the previous one. So the output here is designed to be committed, diffed, and
compared across runs: a console summary to read now, JSON to store, and HTML to
share.

The metrics are deliberately kept apart rather than blended into one score.
"Quality: 87%" hides which property regressed. A drop in retrieval recall, a
hallucinated figure, and an answer that reads badly are three different
problems with three different fixes.
"""

from __future__ import annotations

import html
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evaluation.results import CaseResult
from evaluation.retrieval_metrics import RetrievalMetrics


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile. Returns None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


@dataclass
class Scorecard:
    """Aggregate view of one evaluation run."""

    cases: List[CaseResult] = field(default_factory=list)
    retrieval: Optional[RetrievalMetrics] = None
    model: str = ""
    dataset_names: List[str] = field(default_factory=list)
    started_at: str = ""

    # -- counts ---------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def failed(self) -> List[CaseResult]:
        return [c for c in self.cases if not c.passed]

    @property
    def errors(self) -> List[CaseResult]:
        return [c for c in self.cases if c.error]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    # -- quality dimensions ---------------------------------------------

    @property
    def correctness(self) -> Optional[float]:
        """Share of expected facts the answers actually stated."""
        scores = [c.facts.score for c in self.cases if c.facts is not None]
        return _mean(scores)

    @property
    def groundedness(self) -> Optional[float]:
        scores = [c.groundedness.score for c in self.cases if c.groundedness is not None]
        return _mean(scores)

    @property
    def semantic_match(self) -> Optional[float]:
        rates = [
            1.0 if c.semantic.passed else 0.0
            for c in self.cases
            if c.semantic is not None
        ]
        return _mean(rates)

    @property
    def judge_scores(self) -> Dict[str, Optional[float]]:
        """Averages over the cases the judge actually managed to score."""
        available = [
            c.judge for c in self.cases if c.judge is not None and c.judge.available
        ]
        if not available:
            return {"groundedness": None, "relevance": None, "correctness": None}
        return {
            "groundedness": _mean([j.groundedness or 0 for j in available]),
            "relevance": _mean([j.relevance or 0 for j in available]),
            "correctness": _mean([j.correctness or 0 for j in available]),
        }

    @property
    def judge_evaluated(self) -> int:
        return sum(
            1 for c in self.cases if c.judge is not None and c.judge.available
        )

    @property
    def judge_requested(self) -> int:
        return sum(1 for c in self.cases if c.judge is not None)

    @property
    def hallucinations(self) -> List[CaseResult]:
        """Cases where a figure or claim was asserted without support."""
        found = []
        for case in self.cases:
            grounding = case.groundedness
            if grounding and (
                grounding.unsupported_numbers or not grounding.passed
            ):
                found.append(case)
        return found

    @property
    def safety_total(self) -> int:
        return sum(1 for c in self.cases if c.safety is not None)

    @property
    def safety_passed(self) -> int:
        return sum(1 for c in self.cases if c.safety is not None and c.safety.passed)

    @property
    def latencies(self) -> List[float]:
        return [c.latency_seconds for c in self.cases if c.latency_seconds is not None]

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        judge = self.judge_scores
        return {
            "started_at": self.started_at,
            "model": self.model,
            "datasets": self.dataset_names,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "summary": {
                "cases": self.total,
                "passed": self.passed,
                "failed": self.total - self.passed,
                "errors": len(self.errors),
                "pass_rate": round(self.pass_rate, 4),
                "correctness": _round(self.correctness),
                "groundedness": _round(self.groundedness),
                "semantic_match": _round(self.semantic_match),
                "hallucination_cases": len(self.hallucinations),
                "safety_passed": self.safety_passed,
                "safety_total": self.safety_total,
                "judge_evaluated": self.judge_evaluated,
                "judge_requested": self.judge_requested,
                "judge_groundedness_of_2": _round(judge["groundedness"]),
                "judge_relevance_of_2": _round(judge["relevance"]),
                "judge_correctness_of_2": _round(judge["correctness"]),
                "latency_p50_seconds": _round(percentile(self.latencies, 0.50)),
                "latency_p95_seconds": _round(percentile(self.latencies, 0.95)),
            },
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "cases_detail": [c.to_dict() for c in self.cases],
        }


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(value, digits)


def _pct(value: Optional[float]) -> str:
    return "not measured" if value is None else f"{value * 100:.1f}%"


def render_console(scorecard: Scorecard) -> str:
    """The summary a person reads immediately after a run."""
    judge = scorecard.judge_scores
    lines: List[str] = []

    lines.append("")
    lines.append("=" * 68)
    lines.append("RAG EVALUATION SCORECARD")
    lines.append("=" * 68)
    lines.append(f"  model          {scorecard.model or 'unknown'}")
    lines.append(f"  datasets       {', '.join(scorecard.dataset_names) or 'none'}")
    lines.append(f"  started        {scorecard.started_at}")
    lines.append("")
    lines.append(
        f"  cases          {scorecard.total}    "
        f"passed {scorecard.passed}    failed {scorecard.total - scorecard.passed}"
    )
    lines.append(f"  pass rate      {_pct(scorecard.pass_rate)}")
    lines.append("")
    lines.append("  quality")
    lines.append(f"    correctness      {_pct(scorecard.correctness)}   (expected facts stated)")
    lines.append(f"    groundedness     {_pct(scorecard.groundedness)}   (claims supported by context)")
    lines.append(f"    semantic match   {_pct(scorecard.semantic_match)}   (meaning vs reference answer)")
    lines.append(f"    hallucinations   {len(scorecard.hallucinations)} case(s)")

    if scorecard.retrieval and scorecard.retrieval.total:
        retrieval = scorecard.retrieval
        lines.append("")
        lines.append("  retrieval")
        lines.append(f"    recall@k         {_pct(retrieval.recall_at_k)}")
        lines.append(f"    precision@k      {_pct(retrieval.precision_at_k)}")
        lines.append(f"    mrr              {retrieval.mrr:.3f}")
        if retrieval.misses:
            lines.append(f"    missed           {', '.join(retrieval.misses)}")

    if scorecard.safety_total:
        lines.append("")
        lines.append("  safety")
        lines.append(
            f"    passed           {scorecard.safety_passed}/{scorecard.safety_total}"
        )

    if scorecard.judge_requested:
        lines.append("")
        lines.append("  llm judge (advisory, not a gate; scale 0-2)")
        if scorecard.judge_evaluated:
            lines.append(
                f"    scored           {scorecard.judge_evaluated}/{scorecard.judge_requested} cases"
            )
            lines.append(f"    groundedness     {judge['groundedness']:.2f}")
            lines.append(f"    relevance        {judge['relevance']:.2f}")
            lines.append(f"    correctness      {judge['correctness']:.2f}")
        else:
            lines.append("    not evaluated    no judge model was reachable")

    if scorecard.latencies:
        lines.append("")
        lines.append("  latency")
        lines.append(f"    p50              {percentile(scorecard.latencies, 0.50):.2f}s")
        lines.append(f"    p95              {percentile(scorecard.latencies, 0.95):.2f}s")

    if scorecard.failed:
        lines.append("")
        lines.append("  failures")
        for case in scorecard.failed:
            lines.append(f"    {case.case_id}  {case.question}")
            for reason in failure_reasons(case):
                lines.append(f"        - {reason}")

    lines.append("")
    lines.append("=" * 68)
    lines.append("")
    return "\n".join(lines)


def failure_reasons(case: CaseResult) -> List[str]:
    """Every reason one case failed, in the order they were checked."""
    reasons: List[str] = []

    if case.error:
        reasons.append(f"error: {case.error}")

    if case.facts and not case.facts.passed:
        for check in case.facts.failures:
            reasons.append(f"fact {check.fact_id}: {check.reason}")
        for hit in case.facts.forbidden_hits:
            reasons.append(f"forbidden phrase present: {hit!r}")

    if case.groundedness and not case.groundedness.passed:
        reasons.append(f"groundedness: {case.groundedness.reason}")

    if case.semantic and not case.semantic.passed:
        reasons.append(f"semantic: {case.semantic.reason}")

    if case.safety and not case.safety.passed:
        reasons.extend(f"safety: {v}" for v in case.safety.violations)

    return reasons


def write_json(scorecard: Scorecard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scorecard.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


_HTML_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       background: #f5f6f8; color: #1f2430; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { color: #6b7280; font-size: 13px; margin-bottom: 24px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }
.card { background: #fff; border: 1px solid #e4e7ec; border-radius: 8px;
        padding: 14px 18px; min-width: 150px; flex: 1; }
.card .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
               color: #6b7280; }
.card .value { font-size: 24px; font-weight: 600; margin-top: 6px; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border: 1px solid #e4e7ec; border-radius: 8px; overflow: hidden;
        font-size: 13px; }
th { text-align: left; background: #fafbfc; padding: 10px 12px;
     border-bottom: 1px solid #e4e7ec; font-size: 11px; text-transform: uppercase;
     letter-spacing: .05em; color: #6b7280; }
td { padding: 10px 12px; border-bottom: 1px solid #f0f1f3; vertical-align: top; }
tr:last-child td { border-bottom: none; }
.pass { color: #067647; font-weight: 600; }
.fail { color: #b42318; font-weight: 600; }
.reason { color: #b42318; font-size: 12px; display: block; margin-top: 4px; }
.answer { color: #4b5563; font-size: 12px; display: block; margin-top: 4px; }
h2 { font-size: 15px; margin: 32px 0 10px; }
.note { background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px;
        padding: 12px 16px; font-size: 13px; color: #713f12; margin-bottom: 24px; }
"""


def render_html(scorecard: Scorecard) -> str:
    """A self-contained HTML report, no external assets."""

    def card(label: str, value: str) -> str:
        return (
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div></div>'
        )

    cards = [
        card("Cases", str(scorecard.total)),
        card("Pass rate", _pct(scorecard.pass_rate)),
        card("Correctness", _pct(scorecard.correctness)),
        card("Groundedness", _pct(scorecard.groundedness)),
        card("Hallucinations", str(len(scorecard.hallucinations))),
    ]
    if scorecard.safety_total:
        cards.append(
            card("Safety", f"{scorecard.safety_passed}/{scorecard.safety_total}")
        )
    if scorecard.retrieval and scorecard.retrieval.total:
        cards.append(card("Recall@k", _pct(scorecard.retrieval.recall_at_k)))

    rows = []
    for case in scorecard.cases:
        verdict = (
            '<span class="pass">PASS</span>'
            if case.passed
            else '<span class="fail">FAIL</span>'
        )
        reasons = "".join(
            f'<span class="reason">{html.escape(r)}</span>'
            for r in failure_reasons(case)
        )
        answer = html.escape((case.answer or "")[:400])
        rows.append(
            "<tr>"
            f"<td>{html.escape(case.case_id)}</td>"
            f"<td>{html.escape(case.question)}"
            f'<span class="answer">{answer}</span>{reasons}</td>'
            f"<td>{verdict}</td>"
            f"<td>{'' if case.latency_seconds is None else f'{case.latency_seconds:.2f}s'}</td>"
            "</tr>"
        )

    judge_note = ""
    if scorecard.judge_requested:
        judge = scorecard.judge_scores
        if scorecard.judge_evaluated:
            judge_note = (
                '<div class="note"><strong>LLM judge (advisory only, scale 0-2):</strong> '
                f"groundedness {judge['groundedness']:.2f}, "
                f"relevance {judge['relevance']:.2f}, "
                f"correctness {judge['correctness']:.2f} over "
                f"{scorecard.judge_evaluated}/{scorecard.judge_requested} cases. "
                "These scores do not decide pass or fail; calibrate them against "
                "human review before trusting them.</div>"
            )
        else:
            judge_note = (
                '<div class="note"><strong>LLM judge:</strong> not evaluated, no judge '
                "model was reachable. This is reported as unmeasured, not as a pass.</div>"
            )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>RAG Evaluation Scorecard</title>
<style>{_HTML_STYLE}</style></head>
<body><div class="wrap">
<h1>RAG Evaluation Scorecard</h1>
<div class="sub">{html.escape(scorecard.started_at)} &middot; model
{html.escape(scorecard.model or 'unknown')} &middot; datasets
{html.escape(', '.join(scorecard.dataset_names))} &middot; python
{html.escape(platform.python_version())}</div>
<div class="cards">{''.join(cards)}</div>
{judge_note}
<h2>Cases</h2>
<table><thead><tr><th>ID</th><th>Question and answer</th><th>Result</th>
<th>Latency</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</div></body></html>
"""


def write_html(scorecard: Scorecard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(scorecard), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
