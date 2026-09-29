"""LLM-as-a-judge: a model scores the answer another model produced.

Useful for what deterministic rules cannot express - whether an open-ended
explanation is coherent, complete, and actually responsive to the question.

Everything about this is constrained, and the constraints are the interesting
part of the design:

**The judge is advisory, never the gate.** `CaseResult.passed` ignores it. The
judge here runs on the same local Ollama server, often the same small model that
wrote the answer, which is close to asking a student to mark their own paper. A
7B model as sole quality gate would produce a scorecard that looks authoritative
and means very little. The deterministic fact, groundedness, and safety checks
decide pass and fail; the judge adds a score to look at and a written reason to
read.

**Configure a different model where possible.** `JUDGE_MODEL` defaults to the
answering model but should be set to a different, ideally larger one. Two
different models making the same mistake is less likely than one model being
consistent with itself.

**Structured output, validated.** The judge is asked for JSON with a fixed
schema and fixed scale. Unparseable or out-of-range output is a judge failure
reported as "not evaluated", never coerced into a pass.

**Unavailability is not success.** If no judge is reachable, `available` is
False and the metric is reported as not evaluated. A quality gate that silently
passes when its evaluator is missing is worse than no gate.

Calibration is your job before trusting any number here: score 20 or so cases by
hand, compare, and measure how often the judge agrees. Do not use it as the only
check on a figure that carries money or safety.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from evaluation.results import JudgeResult

JUDGE_PROMPT = """You are evaluating a support chatbot's answer. You are not \
answering the question yourself.

QUESTION:
{question}

RETRIEVED CONTEXT (the only information the chatbot was allowed to use):
{context}

CHATBOT ANSWER:
{answer}

Score the answer on three criteria, each 0, 1, or 2.

groundedness: is every claim in the answer supported by the retrieved context?
  0 = contains claims the context does not support
  1 = mostly supported, some unsupported detail
  2 = fully supported by the context

relevance: does the answer address the question that was asked?
  0 = does not address the question
  1 = partially addresses it
  2 = directly addresses it

correctness: is the answer factually correct with respect to the context?
  0 = incorrect
  1 = partially correct
  2 = correct

Wording does not matter. An answer that says "thirty days" is equivalent to one \
that says "30 days". Judge the meaning.

If the answer declines to answer and the context is empty, that is correct \
behaviour: score groundedness 2, relevance 2, correctness 2.

Reply with JSON only, no other text:
{{"groundedness": 0, "relevance": 0, "correctness": 0, "reason": "one sentence"}}
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_VALID_SCORES = {0, 1, 2}


def judge_model_name() -> str:
    """Which model to judge with.

    Set JUDGE_MODEL to something other than the answering model. Defaults to the
    answering model only so that the framework runs out of the box.
    """
    from app.config import settings

    return os.getenv("JUDGE_MODEL", settings.llm_model)


def _build_judge(model: Optional[str] = None, temperature: float = 0.0) -> Any:
    """Construct the judge client, imported lazily.

    `format="json"` makes Ollama constrain its output to valid JSON, which
    removes most of the parsing failures a free-text judge produces.
    """
    from langchain_ollama import ChatOllama

    from app.config import settings

    return ChatOllama(
        model=model or judge_model_name(),
        base_url=settings.ollama_base_url,
        temperature=temperature,
        format="json",
    )


def parse_judge_output(raw: str) -> JudgeResult:
    """Parse and validate a judge reply.

    Anything malformed becomes `available=False` with the reason recorded, so a
    broken judge shows up as an unevaluated metric instead of a pass.
    """
    if not raw or not raw.strip():
        return JudgeResult(available=False, reason="judge returned an empty response")

    match = _JSON_BLOCK.search(raw)
    if not match:
        return JudgeResult(
            available=False, reason="judge reply contained no JSON object", raw=raw
        )

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return JudgeResult(
            available=False, reason=f"judge reply was not valid JSON: {exc}", raw=raw
        )

    if not isinstance(payload, dict):
        return JudgeResult(
            available=False, reason="judge reply was not a JSON object", raw=raw
        )

    scores: Dict[str, Any] = {}
    for key in ("groundedness", "relevance", "correctness"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return JudgeResult(
                available=False,
                reason=f"judge omitted or mistyped {key!r}",
                raw=raw,
            )
        as_int = int(value)
        if as_int not in _VALID_SCORES:
            return JudgeResult(
                available=False,
                reason=f"judge returned {key}={value}, outside the 0-2 scale",
                raw=raw,
            )
        scores[key] = as_int

    reason = payload.get("reason", "")
    return JudgeResult(
        available=True,
        groundedness=scores["groundedness"],
        relevance=scores["relevance"],
        correctness=scores["correctness"],
        reason=str(reason)[:500],
        raw=raw,
    )


def judge_answer(
    question: str,
    context: str,
    answer: str,
    *,
    model: Optional[str] = None,
    attempts: int = 2,
    invoke: Optional[Any] = None,
) -> JudgeResult:
    """Score one answer with the judge model.

    `invoke` accepts a callable taking a prompt and returning text, which is how
    the framework's own tests exercise this without a running model.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context or "(no context was retrieved)",
        answer=answer,
    )

    if invoke is None:
        try:
            judge = _build_judge(model)
        except Exception as exc:  # pragma: no cover - depends on local install
            return JudgeResult(
                available=False, reason=f"judge model unavailable: {type(exc).__name__}"
            )

        def invoke(text: str) -> str:  # type: ignore[misc]
            return str(judge.invoke(text).content)

    last = JudgeResult(available=False, reason="judge was never invoked")
    for _ in range(max(1, attempts)):
        try:
            raw = invoke(prompt)
        except Exception as exc:  # pragma: no cover - network dependent
            last = JudgeResult(
                available=False, reason=f"judge call failed: {type(exc).__name__}"
            )
            continue

        last = parse_judge_output(raw)
        if last.available:
            return last

    return last
