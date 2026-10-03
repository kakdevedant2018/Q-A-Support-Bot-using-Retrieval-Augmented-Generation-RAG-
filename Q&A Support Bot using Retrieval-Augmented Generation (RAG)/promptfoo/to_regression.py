"""Fold surviving red-team attacks into the permanent regression dataset.

    python promptfoo/to_regression.py promptfoo/results.json
    python promptfoo/to_regression.py promptfoo/results.json --write

This is the step that makes red-teaming worth doing twice. A promptfoo run is
a search: it generates attacks, finds the ones that get through, and then those
results sit in a JSON file nobody opens again. Fix the prompt, and nothing in
the repository remembers why - the next refactor can reintroduce the hole and
every test stays green.

Converting a finding into a `regression.json` case makes it permanent and free.
The dataset is already run by `run_eval` and by `tests/test_ai_quality.py`, both
deterministic, so a reintroduced hole fails a named test instead of waiting for
someone to schedule another red-team session.

Deliberately prints by default and only writes with `--write`. Generated
attacks need a human read before they become a test: the local grader has a
false-positive rate, and a case that encodes a wrong expectation is worse than
no case at all because it has to be argued with later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

REGRESSION_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "datasets" / "regression.json"

# promptfoo plugin -> the behaviour the safety evaluator already knows how to
# check. The mapping exists so a converted case is scored by the same evaluator
# as every hand-written security case; inventing a new behaviour string here
# would produce a case that silently passes because nothing checks it.
BEHAVIOUR_BY_PLUGIN = {
    "prompt-extraction": "refuse_sensitive_request",
    "pii": "refuse_sensitive_request",
    "pii:direct": "refuse_sensitive_request",
    "hallucination": "do_not_invent_information",
    "overreliance": "do_not_invent_information",
    "excessive-agency": "refuse_sensitive_request",
    "harmful:misinformation-disinformation": "do_not_invent_information",
}
DEFAULT_BEHAVIOUR = "stay_in_knowledge_base"


def _plugin_of(result: Dict[str, Any]) -> str:
    meta = result.get("testCase", {}).get("metadata", {}) or {}
    return str(meta.get("pluginId") or meta.get("plugin") or "")


def _question_of(result: Dict[str, Any]) -> str:
    vars_ = result.get("testCase", {}).get("vars", {}) or {}
    for key in ("prompt", "query", "question"):
        value = vars_.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    prompt = result.get("prompt", {})
    if isinstance(prompt, dict):
        return str(prompt.get("raw", "")).strip()
    return ""


def failures(payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield the results promptfoo graded as attacks that got through.

    Walks both the current and the older results layout, because the schema has
    moved between versions and a converter that silently finds zero failures
    after an upgrade is the worst possible outcome here.
    """
    results = payload.get("results", payload)
    if isinstance(results, dict):
        results = results.get("results", [])
    for result in results or []:
        if not isinstance(result, dict):
            continue
        if result.get("success") is False or result.get("pass") is False:
            yield result


def to_cases(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    seen = set()

    for index, result in enumerate(failures(payload), start=1):
        question = _question_of(result)
        if not question or question in seen:
            continue
        seen.add(question)

        plugin = _plugin_of(result)
        strategy = (
            result.get("testCase", {}).get("metadata", {}) or {}
        ).get("strategyId", "")
        grader_reason = str(
            (result.get("gradingResult") or {}).get("reason", "")
        ).replace("\n", " ")[:300]

        cases.append(
            {
                "id": f"RT{index:02d}",
                "question": question,
                "expected_behavior": BEHAVIOUR_BY_PLUGIN.get(plugin, DEFAULT_BEHAVIOUR),
                "note": (
                    f"Found by promptfoo red-teaming (plugin: {plugin or '?'}"
                    + (f", strategy: {strategy}" if strategy else "")
                    + "). Grader's reason, for the record rather than as the "
                    f"oracle: {grader_reason!r}. REVIEW BEFORE MERGING - renumber "
                    "the id, confirm the behaviour is the right one, and replace "
                    "this note with what the defect actually was."
                ),
            }
        )
    return cases


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="promptfoo results JSON (from `redteam eval -o`)")
    parser.add_argument(
        "--write",
        action="store_true",
        help="append to evaluation/datasets/regression.json instead of printing",
    )
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    cases = to_cases(payload)

    if not cases:
        print("No surviving attacks in that run. Nothing to pin.")
        return 0

    print(f"{len(cases)} attack(s) got through:\n")
    print(json.dumps(cases, indent=2))

    if not args.write:
        print(
            "\nPrinted only. Read each one, fix the defect, then re-run with "
            "--write (or paste the cases in by hand, which is usually better - "
            "the ids and notes need editing anyway).",
            file=sys.stderr,
        )
        return 0

    existing = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))
    taken = {c.get("question") for c in existing}
    added = [c for c in cases if c["question"] not in taken]

    # Renumber against what is already in the file, so two conversion runs do
    # not both produce an RT01.
    offset = sum(1 for c in existing if str(c.get("id", "")).startswith("RT"))
    for position, case in enumerate(added, start=offset + 1):
        case["id"] = f"RT{position:02d}"

    REGRESSION_PATH.write_text(
        json.dumps(existing + added, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nAppended {len(added)} case(s) to {REGRESSION_PATH}", file=sys.stderr)
    print(
        "Now run: pytest -m integration -k regression   (and edit the notes)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
