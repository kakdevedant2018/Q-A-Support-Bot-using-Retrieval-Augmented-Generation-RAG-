"""Evaluation framework for the RAG support bot.

Separate from `tests/` on purpose. The contents of `tests/` assert that the
application behaves correctly; this package measures how good its answers are,
which is a different activity with a different output. Tests return pass or
fail. Evaluation returns a scorecard, and a scorecard is only meaningful next to
the previous one.

Layout:

    normalize.py               shared text canonicalisation
    results.py                 structured result types
    fact_evaluator.py          deterministic fact checks, no model
    groundedness_evaluator.py  is every claim supported by the context
    semantic_evaluator.py      meaning comparison via local embeddings
    retrieval_metrics.py       Recall@k, Precision@k, MRR from labelled data
    safety_evaluator.py        injection, leakage, invented policy
    llm_judge.py               model-graded quality, advisory only
    report.py                  scorecard aggregation, console/JSON/HTML output
    run_eval.py                CLI: run a dataset and print a scorecard
    calibrate.py               measure thresholds instead of guessing them
    datasets/                  labelled data, versioned with the code

Everything here runs locally and costs nothing. The only optional dependency is
a local Ollama server for the judge.

Two consumers, one implementation: `run_eval.py` produces a report for a person,
and `tests/test_ai_quality.py` / `tests/test_security.py` import the same
evaluators to turn selected measurements into build gates. An evaluator bug can
therefore never make the tests and the report disagree.

The dependency direction is strict: this package may import `app`, and `app`
never imports this package. A measuring instrument does not ship inside the
product.

Design notes, including the limits of each evaluator, are in
`docs/AI_TESTING_STRATEGY.md`.
"""

__all__ = [
    "fact_evaluator",
    "groundedness_evaluator",
    "llm_judge",
    "normalize",
    "report",
    "results",
    "retrieval_metrics",
    "run_eval",
    "safety_evaluator",
    "semantic_evaluator",
]
