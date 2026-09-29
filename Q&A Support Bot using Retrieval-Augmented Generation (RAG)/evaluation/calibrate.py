"""Measure the thresholds instead of guessing them.

Two numbers decide a lot of this system's behaviour and both ship as a guess:

* `RELEVANCE_THRESHOLD` decides when the bot refuses to answer
* the semantic similarity threshold decides when an answer counts as matching

A guessed threshold is the most common way an evaluation suite ends up either
rejecting correct answers or waving through wrong ones. This tool measures both
against the real knowledge base and prints the separation it found, so the value
in `.env` can be chosen from data.

    python -m evaluation.calibrate                # both
    python -m evaluation.calibrate --retrieval    # no generation needed
    python -m evaluation.calibrate --semantic

Needs the embedding model and a built index. The retrieval half does not need
Ollama; the semantic half does, because it compares real answers.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from evaluation.run_eval import DirectClient, load_dataset


def _percentiles(values: Sequence[float]) -> str:
    if not values:
        return "no samples"
    ordered = sorted(values)
    return (
        f"min {ordered[0]:.3f}  p25 {ordered[len(ordered) // 4]:.3f}  "
        f"median {statistics.median(ordered):.3f}  "
        f"p75 {ordered[(3 * len(ordered)) // 4]:.3f}  max {ordered[-1]:.3f}"
    )


def top_scores(store: Any, questions: Sequence[str]) -> List[float]:
    """Best relevance score for each question."""
    scores: List[float] = []
    for question in questions:
        matches = store.similarity_search_with_relevance_scores(question, k=1)
        scores.append(matches[0][1] if matches else 0.0)
    return scores


def calibrate_retrieval(chroma_dir: Optional[str] = None) -> Dict[str, Any]:
    """Compare in-domain and out-of-domain retrieval scores.

    In-domain questions come from the factual dataset, which is labelled with the
    document that should answer each one. Out-of-domain questions come from the
    hallucination dataset, where the correct behaviour is to retrieve nothing.
    """
    from app.config import settings
    from app.services.vector_store import get_vector_store

    store = get_vector_store(chroma_dir=chroma_dir)

    in_domain = [c["question"] for c in load_dataset("factual")]
    out_of_domain = [
        c["question"]
        for c in load_dataset("hallucination")
        # Cases whose expected behaviour is a grounded restatement do retrieve
        # something legitimately, so they are not out-of-domain samples.
        if c.get("expected_behavior") == "refuse_sensitive_request"
    ]

    relevant = top_scores(store, in_domain)
    unrelated = top_scores(store, out_of_domain)

    print("\nRETRIEVAL THRESHOLD CALIBRATION")
    print("-" * 68)
    print(f"  in-domain      ({len(relevant)} questions)  {_percentiles(relevant)}")
    print(f"  out-of-domain  ({len(unrelated)} questions)  {_percentiles(unrelated)}")

    lowest_relevant = min(relevant) if relevant else 0.0
    highest_unrelated = max(unrelated) if unrelated else 0.0

    result: Dict[str, Any] = {
        "in_domain_min": lowest_relevant,
        "out_of_domain_max": highest_unrelated,
        "configured": settings.relevance_threshold,
    }

    print()
    if lowest_relevant > highest_unrelated:
        suggestion = (lowest_relevant + highest_unrelated) / 2
        result["suggested"] = suggestion
        print(f"  the two groups separate cleanly: {highest_unrelated:.3f} < {lowest_relevant:.3f}")
        print(f"  suggested RELEVANCE_THRESHOLD = {suggestion:.2f}")
    else:
        result["suggested"] = None
        print(
            f"  the groups OVERLAP: an unrelated question scored {highest_unrelated:.3f} "
            f"while a real one scored only {lowest_relevant:.3f}"
        )
        print("  no single threshold separates them. Options, in order of value:")
        print("    - improve chunking so relevant chunks score higher")
        print("    - add the missing content to the knowledge base")
        print("    - accept the overlap and rely on groundedness checks downstream")

    print(f"  currently configured = {settings.relevance_threshold}")
    print()
    return result


def calibrate_semantic(chroma_dir: Optional[str] = None) -> Dict[str, Any]:
    """Measure similarity for matched and mismatched answer pairs.

    Positive pairs are the bot's own answer against that case's reference answer.
    Negative pairs are the same answer against a *different* case's reference. No
    extra labelling is needed, because the dataset already says which answer
    belongs to which question.
    """
    from evaluation.semantic_evaluator import semantic_similarity

    cases = [c for c in load_dataset("factual") if c.get("reference_answer")]
    client = DirectClient(chroma_dir=chroma_dir)

    answers: List[Tuple[str, str, str]] = []
    for case in cases:
        response = client.ask(case["question"])
        answer = response.get("answer", "")
        if answer:
            answers.append((case["id"], answer, case["reference_answer"]))

    matched = [semantic_similarity(a, r) for _, a, r in answers]
    mismatched: List[float] = []
    for index, (_, answer, _) in enumerate(answers):
        other = answers[(index + 1) % len(answers)] if len(answers) > 1 else None
        if other and other[0] != answers[index][0]:
            mismatched.append(semantic_similarity(answer, other[2]))

    print("\nSEMANTIC THRESHOLD CALIBRATION")
    print("-" * 68)
    print(f"  matched pairs     ({len(matched)})  {_percentiles(matched)}")
    print(f"  mismatched pairs  ({len(mismatched)})  {_percentiles(mismatched)}")

    result: Dict[str, Any] = {
        "matched_min": min(matched) if matched else None,
        "mismatched_max": max(mismatched) if mismatched else None,
    }

    print()
    if matched and mismatched and min(matched) > max(mismatched):
        suggestion = (min(matched) + max(mismatched)) / 2
        result["suggested"] = suggestion
        print(f"  suggested --semantic-threshold = {suggestion:.2f}")
    else:
        result["suggested"] = None
        print("  matched and mismatched pairs overlap.")
        print("  similarity alone cannot separate them on this dataset, which is")
        print("  exactly why the fact and groundedness checks stay the primary gate.")
    print()
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.calibrate",
        description="Measure the retrieval and semantic thresholds against real data.",
    )
    parser.add_argument("--retrieval", action="store_true", help="retrieval threshold only")
    parser.add_argument("--semantic", action="store_true", help="semantic threshold only")
    parser.add_argument("--chroma-dir", help="vector store directory to measure against")
    args = parser.parse_args(argv)

    run_both = not (args.retrieval or args.semantic)

    if args.retrieval or run_both:
        calibrate_retrieval(chroma_dir=args.chroma_dir)
    if args.semantic or run_both:
        calibrate_semantic(chroma_dir=args.chroma_dir)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
