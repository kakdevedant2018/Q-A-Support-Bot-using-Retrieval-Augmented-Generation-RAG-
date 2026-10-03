"""Run an evaluation dataset and print a scorecard.

    python -m evaluation.run_eval                      # every dataset, in-process
    python -m evaluation.run_eval --dataset factual
    python -m evaluation.run_eval --base-url http://127.0.0.1:8000
    python -m evaluation.run_eval --judge --html reports/eval.html
    python -m evaluation.run_eval --fail-under 0.9     # non-zero exit below that

This is the tool a person runs; `tests/` is what CI runs. The difference matters.
A test answers "is this broken", and pytest's pass/fail output is the right shape
for that. An evaluation answers "how good is it now compared with last week",
which needs a scorecard rather than a verdict, so it gets its own entry point and
writes an artifact you can keep.

Two ways to reach the system:

* in-process (default) calls RAGService directly, so no server is needed
* --base-url goes over HTTP, which also exercises the API, validation, and
  serialisation layers

Everything runs locally against Ollama, so a full run costs nothing but time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from evaluation import report as report_module
from evaluation.fact_evaluator import evaluate_facts
from evaluation.groundedness_evaluator import evaluate_from_sources, join_context
from evaluation.report import Scorecard
from evaluation.results import CaseResult
from evaluation.retrieval_metrics import RetrievalCase, case_from_response, score_retrieval
from evaluation.safety_evaluator import evaluate_case as evaluate_safety_case

DATASET_DIR = Path(__file__).resolve().parent / "datasets"

# `regression` is last on purpose. It holds failures that were observed in a real
# session and then fixed, so every case in it is a bug that has already happened
# once. A fix with no case here is a fix that can be silently undone: the suite
# would stay green because nothing in it was ever asked to notice.
#
# Unlike the others it mixes content cases with behaviour cases, because real
# failures do not sort themselves by evaluation technique.
ALL_DATASETS = ["factual", "open_ended", "hallucination", "security", "regression"]

# `metamorphic.json` and `bias.json` sit in the same directory but are not in
# the list above, and that is deliberate rather than an omission.
#
# Every dataset in ALL_DATASETS is a list of independent cases: one question,
# one set of expectations, one verdict. `evaluate_one` is built for exactly that
# shape. A metamorphic relation is not a case - it is a *group* of questions
# plus a property that must hold between their answers, and a bias template is
# one question rendered once per persona. Neither has a single answer to score,
# so neither can be fed to `evaluate_one` without the relation collapsing into
# unrelated cases and the property going unchecked.
#
# They therefore get their own runners in `tests/test_metamorphic.py` and
# `tests/test_bias.py`, and are listed here so that the dataset-inventory test
# in `test_evaluation_framework.py` still accounts for every file on disk - an
# unlisted dataset is one nobody maintains.
RELATION_DATASETS = ["metamorphic", "bias"]

ASK_PATH = "/api/v1/ask"


class SupportsAsk(Protocol):
    def ask(self, question: str) -> Dict[str, Any]: ...


class DirectClient:
    """Calls the service in-process. No server, no HTTP, no serialisation."""

    def __init__(self, chroma_dir: Optional[str] = None) -> None:
        from app.services.rag_service import RAGService

        self._service = RAGService(chroma_dir=chroma_dir)

    def ask(self, question: str) -> Dict[str, Any]:
        return self._service.ask(question)


class HttpClient:
    """Calls a running server, so the API layer is evaluated too."""

    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        import httpx

        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def ask(self, question: str) -> Dict[str, Any]:
        response = self._client.post(ASK_PATH, json={"question": question})
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP {response.status_code}: {response.text[:200]}"
            )
        return response.json()


def load_dataset(name: str) -> List[Dict[str, Any]]:
    """Load a dataset by stem name or explicit path."""
    candidate = Path(name)
    path = candidate if candidate.suffix == ".json" else DATASET_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"no dataset at {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of cases")
    return data


def evaluate_one(
    case: Dict[str, Any],
    client: SupportsAsk,
    *,
    use_semantic: bool = True,
    use_judge: bool = False,
    semantic_threshold: Optional[float] = None,
) -> CaseResult:
    """Ask one question and apply every applicable evaluator to the answer."""
    question = case["question"]
    case_id = case.get("id", question[:24])

    started = time.perf_counter()
    try:
        response = client.ask(question)
    except Exception as exc:
        return CaseResult(
            case_id=case_id,
            question=question,
            answer="",
            error=f"{type(exc).__name__}: {exc}",
            latency_seconds=time.perf_counter() - started,
        )
    latency = time.perf_counter() - started

    answer = response.get("answer", "")
    raw_sources = response.get("sources", []) or []
    source_texts = [s.get("content", "") for s in raw_sources]
    is_refusal = bool(response.get("insufficient_context"))

    result = CaseResult(
        case_id=case_id,
        question=question,
        answer=answer,
        sources=[s.get("metadata", {}).get("source", "?") for s in raw_sources],
        insufficient_context=is_refusal,
        latency_seconds=latency,
    )

    result.groundedness = evaluate_from_sources(
        answer, source_texts, question=question, is_refusal=is_refusal
    )

    behaviour = case.get("expected_behavior")
    if behaviour:
        # Hallucination and security cases are judged on behaviour, not content.
        result.safety = evaluate_safety_case(case, response)
    else:
        result.facts = evaluate_facts(
            answer,
            case.get("expected_facts", []),
            case.get("forbidden_phrases", []),
        )

        reference = case.get("reference_answer")
        if use_semantic and reference:
            from evaluation.semantic_evaluator import DEFAULT_THRESHOLD, evaluate_semantic

            result.semantic = evaluate_semantic(
                answer,
                reference,
                threshold=(
                    DEFAULT_THRESHOLD if semantic_threshold is None else semantic_threshold
                ),
            )

        if use_judge:
            from evaluation.llm_judge import judge_answer

            result.judge = judge_answer(
                question, join_context(source_texts), answer
            )

    return result


def run(
    dataset_names: Sequence[str],
    client: SupportsAsk,
    *,
    use_semantic: bool = True,
    use_judge: bool = False,
    semantic_threshold: Optional[float] = None,
    only: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> Scorecard:
    """Evaluate every case in the named datasets and aggregate the results."""
    from app.config import settings

    cases: List[Dict[str, Any]] = []
    for name in dataset_names:
        cases.extend(load_dataset(name))

    if only:
        wanted = set(only)
        cases = [c for c in cases if c.get("id") in wanted]
    if limit:
        cases = cases[:limit]

    results: List[CaseResult] = []
    retrieval_cases: List[RetrievalCase] = []

    for index, case in enumerate(cases, start=1):
        if verbose:
            print(
                f"[{index}/{len(cases)}] {case.get('id', '?')} {case['question'][:60]}",
                flush=True,
            )

        result = evaluate_one(
            case,
            client,
            use_semantic=use_semantic,
            use_judge=use_judge,
            semantic_threshold=semantic_threshold,
        )
        results.append(result)

        expected_sources = case.get("expected_sources")
        if expected_sources and not result.error:
            retrieval_case = case_from_response(
                result.case_id,
                expected_sources,
                {"sources": [{"metadata": {"source": s}} for s in result.sources]},
            )
            retrieval_cases.append(retrieval_case)
            result.retrieval_hit = retrieval_case.hit

    return Scorecard(
        cases=results,
        retrieval=score_retrieval(retrieval_cases) if retrieval_cases else None,
        model=settings.llm_model,
        dataset_names=list(dataset_names),
        started_at=report_module.now_iso(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_eval",
        description="Run the RAG evaluation datasets and print a scorecard.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "dataset name or path, repeatable. Defaults to all of: "
            + ", ".join(ALL_DATASETS)
        ),
    )
    parser.add_argument(
        "--base-url",
        help="evaluate a running server over HTTP instead of calling the service in-process",
    )
    parser.add_argument(
        "--chroma-dir",
        help="vector store directory for in-process runs (defaults to the configured one)",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="also score answers with the LLM judge (slower, advisory only)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="skip embedding-based similarity, which avoids loading the embedding model",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        help="override the similarity threshold; measure it with evaluation.calibrate",
    )
    parser.add_argument("--only", action="append", help="run a single case id, repeatable")
    parser.add_argument("--limit", type=int, help="stop after N cases")
    parser.add_argument("--json", dest="json_path", help="write the full scorecard as JSON")
    parser.add_argument("--html", dest="html_path", help="write an HTML report")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.0,
        help="exit non-zero if the pass rate falls below this fraction, e.g. 0.9",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-case progress")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    datasets = args.dataset or ALL_DATASETS
    client: SupportsAsk
    if args.base_url:
        client = HttpClient(args.base_url)
    else:
        client = DirectClient(chroma_dir=args.chroma_dir)

    scorecard = run(
        datasets,
        client,
        use_semantic=not args.no_semantic,
        use_judge=args.judge,
        semantic_threshold=args.semantic_threshold,
        only=args.only,
        limit=args.limit,
        verbose=not args.quiet,
    )

    print(report_module.render_console(scorecard))

    if args.json_path:
        report_module.write_json(scorecard, Path(args.json_path))
        print(f"JSON scorecard: {args.json_path}")
    if args.html_path:
        report_module.write_html(scorecard, Path(args.html_path))
        print(f"HTML report:    {args.html_path}")

    # Exit code policy. Safety failures and crashes are never acceptable at any
    # pass rate, so they fail the run on their own.
    safety_failures = scorecard.safety_total - scorecard.safety_passed
    if scorecard.errors:
        print(f"FAILED: {len(scorecard.errors)} case(s) errored", file=sys.stderr)
        return 1
    if safety_failures:
        print(f"FAILED: {safety_failures} safety violation(s)", file=sys.stderr)
        return 1
    if args.fail_under and scorecard.pass_rate < args.fail_under:
        print(
            f"FAILED: pass rate {scorecard.pass_rate:.1%} is below the "
            f"required {args.fail_under:.1%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
