"""Metamorphic testing: assert relations between inputs, not expected outputs.

Every other evaluator in this package needs to know what a good answer looks
like. A metamorphic relation does not. It states a property that must hold
between the answers to two *related* questions, which means it can test inputs
nobody has labelled:

    "What is the return period?"                 -> states 30 days
    "How long do I have to send an item back?"   -> must also state 30 days

The second question has no reference answer and no entry in `factual.json`. The
relation supplies the oracle instead, and that is the whole value: labelling is
the bottleneck in LLM testing, and this technique sidesteps it.

docs/AI_TESTING_STRATEGY.md §1 already contained the insight informally - it
lists six correct phrasings of the same fact to argue against string matching.
This module turns that observation into a runnable oracle.

Two kinds of relation, and the suite is unsound without both
--------------------------------------------------------------------------
**Invariance.** Meaning-preserving change to the input, same facts out.
Paraphrase, casing, politeness, an irrelevant trailing sentence.

**Change.** Meaning-altering change to the input, *different* behaviour out. An
out-of-scope variant must be refused even though the base question was answered.

The second kind is not optional. A bot that refuses everything, or that replies
with a fixed string, satisfies every invariance relation perfectly. Invariance
alone cannot distinguish a robust system from a broken constant one, so a
metamorphic suite built only from invariance relations is vacuous in exactly the
way §3.3 warns a widened matcher is vacuous.

What this module does not do
--------------------------------------------------------------------------
It does not generate paraphrases. A model could, but then the oracle depends on
the system under test's own family of models, and a bad paraphrase becomes a
false failure nobody can reproduce. Paraphrases are written by hand in
`datasets/metamorphic.json` and reviewed like any other test data. Only the
mechanical transformations below are generated, because those are exactly
reproducible.

Nothing here imports a model or makes a network call. The transformations are
pure string functions and the evaluators are pure functions over outcomes the
caller collected, so the whole module is unit tested in the default suite with
no Ollama and no embeddings - see `tests/test_metamorphic_framework.py`.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from evaluation.fact_evaluator import evaluate_facts
from evaluation.results import MetamorphicResult, VariantOutcome
from evaluation.safety_evaluator import looks_like_refusal

# Relation kinds.
INVARIANT = "invariant"
CHANGE = "change"

# What a `change` relation expects of its variants.
EXPECT_REFUSAL = "refusal"
EXPECT_DIFFERENT_FACTS = "different_facts"

BASE_LABEL = "base"

# Appended by the `distractor` transformation. Chosen to contain no digits and
# no personal pronouns: a digit would interact with the numeric groundedness
# check, and a gendered pronoun would confound the bias dimension, which varies
# exactly that token on purpose. See evaluation/bias.py.
DISTRACTOR_SENTENCE = "By the way, my neighbour has a cat called Pixel."


# --------------------------------------------------------------------------
# Mechanical transformations
#
# Each is meaning-preserving for a support question, so each belongs to an
# invariance relation. They are generated rather than written out because they
# are exactly reproducible - unlike a paraphrase, there is no judgement in
# lowercasing a string.
# --------------------------------------------------------------------------


def _identity(question: str) -> str:
    return question


def _lowercase(question: str) -> str:
    return question.lower()


def _uppercase(question: str) -> str:
    """Shouting is still asking. Also catches case-sensitive handling."""
    return question.upper()


def _no_punctuation(question: str) -> str:
    """Users drop the question mark constantly."""
    return re.sub(r"[?.!]+\s*$", "", question).strip()


def _extra_whitespace(question: str) -> str:
    """Ragged spacing from a copy-paste out of a chat window."""
    return "  " + re.sub(r"\s+", "   ", question.strip()) + "  "


def _polite_prefix(question: str) -> str:
    """A greeting attached to a real question.

    This one is load-bearing beyond politeness. `app/services/small_talk.py`
    matches greetings on the *whole normalised string* specifically so that
    "hi, what is the return policy?" is treated as a question rather than
    swallowed as small talk. Its own docstring says so. That guarantee had no
    end-to-end test against the real model until this relation existed.
    """
    if not question:
        return question
    body = question[0].lower() + question[1:]
    return f"Hi, could you please tell me: {body}"


def _polite_suffix(question: str) -> str:
    return f"{question} Thanks in advance!"


def _distractor(question: str) -> str:
    """An irrelevant trailing sentence.

    Retrieval embeds the whole question, so unrelated text dilutes the query
    vector and can push the answering chunk below `relevance_threshold`. That
    is a real failure mode with a real user behind it, and it is invisible to a
    dataset of clean one-line questions.
    """
    return f"{question} {DISTRACTOR_SENTENCE}"


def _typo(question: str) -> str:
    """Swap two adjacent characters inside the longest word.

    Deterministic by construction - longest word, leftmost on a tie, fixed
    offset - because a randomised typo produces a failure that cannot be
    reproduced from the report.
    """
    words = question.split(" ")
    if not words:
        return question

    index = max(range(len(words)), key=lambda i: (len(words[i]), -i))
    word = words[index]
    if len(word) < 6:
        return question

    middle = len(word) // 2
    words[index] = word[:middle] + word[middle + 1] + word[middle] + word[middle + 2 :]
    return " ".join(words)


TRANSFORMATIONS: Dict[str, Callable[[str], str]] = {
    "identity": _identity,
    "lowercase": _lowercase,
    "uppercase": _uppercase,
    "no_punctuation": _no_punctuation,
    "extra_whitespace": _extra_whitespace,
    "polite_prefix": _polite_prefix,
    "polite_suffix": _polite_suffix,
    "distractor": _distractor,
    "typo": _typo,
}


def apply_transformation(question: str, name: str) -> str:
    """Apply a named transformation, failing loudly on an unknown name.

    A typo in a dataset must not silently become an identity transformation -
    that would add a passing case that tests nothing.
    """
    try:
        transform = TRANSFORMATIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown transformation {name!r}; available: "
            f"{', '.join(sorted(TRANSFORMATIONS))}"
        ) from None
    return transform(question)


# --------------------------------------------------------------------------
# Building the question group for one relation
# --------------------------------------------------------------------------


def build_questions(case: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(label, question) pairs for one relation, with the base first.

    A relation may declare hand-written `variants`, mechanical
    `transformations`, or both.
    """
    base = case["base_question"]
    pairs: List[Tuple[str, str]] = [(BASE_LABEL, base)]

    relation = case.get("relation", "variant")
    for position, variant in enumerate(case.get("variants", []) or [], start=1):
        pairs.append((f"{relation}[{position}]", variant))

    for name in case.get("transformations", []) or []:
        pairs.append((name, apply_transformation(base, name)))

    return pairs


def source_signature(response: Dict[str, Any]) -> List[str]:
    """"source@score" for each retrieved chunk, in rank order.

    This is the diagnostic that turns "the answer changed" into "the *context*
    changed". A divergence where every variant retrieved the same chunks is a
    generation defect; one where the signatures differ is a retrieval defect,
    and the fix lives in a different file in each case.
    """
    signature: List[str] = []
    for source in response.get("sources", []) or []:
        name = (source.get("metadata") or {}).get("source", "?")
        score = source.get("relevance_score")
        signature.append(f"{name}@{score:.3f}" if isinstance(score, float) else str(name))
    return signature


def score_outcome(
    label: str,
    question: str,
    response: Dict[str, Any],
    expected_facts: Iterable[Dict[str, Any]],
    forbidden_phrases: Optional[Iterable[str]] = None,
) -> VariantOutcome:
    """Turn one raw `/ask` response into a comparable outcome.

    The same `evaluate_facts` the rest of the framework uses, so a metamorphic
    divergence and a plain factual failure can never disagree about whether an
    answer stated a fact.
    """
    answer = response.get("answer", "") or ""
    facts = evaluate_facts(answer, expected_facts, forbidden_phrases)
    insufficient = bool(response.get("insufficient_context"))
    return VariantOutcome(
        label=label,
        question=question,
        answer=answer,
        insufficient_context=insufficient,
        # Both refusal routes, via the detector the safety evaluator already
        # uses, so a refusal means the same thing in every dataset.
        refused=looks_like_refusal(answer, insufficient_context=insufficient),
        sources=source_signature(response),
        facts=facts,
    )


def failed_outcome(label: str, question: str, error: str) -> VariantOutcome:
    return VariantOutcome(label=label, question=question, answer="", error=error)


# --------------------------------------------------------------------------
# The relations themselves
# --------------------------------------------------------------------------


def _split_base(
    outcomes: Sequence[VariantOutcome],
) -> Tuple[Optional[VariantOutcome], List[VariantOutcome]]:
    base = next((o for o in outcomes if o.label == BASE_LABEL), None)
    variants = [o for o in outcomes if o.label != BASE_LABEL]
    return base, variants


def evaluate_invariance(
    relation_id: str,
    relation: str,
    outcomes: Sequence[VariantOutcome],
) -> MetamorphicResult:
    """Every variant must reach the same fact verdict as the base.

    Note what is *not* compared: the answer text. Two correct answers to two
    paraphrases will differ in wording by construction, so comparing strings
    here would rebuild the exact brittleness this project removed in §3.1.
    """
    base, variants = _split_base(outcomes)

    if base is None:
        return MetamorphicResult(
            relation_id=relation_id,
            kind=INVARIANT,
            relation=relation,
            relation_holds=False,
            baseline_passed=False,
            variants=list(variants),
            reason="no base outcome was collected",
        )

    if base.error:
        return MetamorphicResult(
            relation_id=relation_id,
            kind=INVARIANT,
            relation=relation,
            relation_holds=False,
            baseline_passed=False,
            base=base,
            variants=list(variants),
            reason=f"the base question could not be evaluated: {base.error}",
        )

    base_verdict = base.states_expected_facts

    divergences: List[str] = []
    for variant in variants:
        if variant.error:
            divergences.append(f"{variant.label}: errored - {variant.error}")
            continue
        if variant.states_expected_facts != base_verdict:
            missing = (
                ", ".join(c.fact_id for c in variant.facts.failures)
                if variant.facts
                else "?"
            )
            divergences.append(
                f"{variant.label}: fact verdict "
                f"{variant.states_expected_facts} != base {base_verdict} "
                f"(diverged on: {missing}) -> {variant.answer[:110]!r}"
            )

    relation_holds = not divergences
    baseline_passed = bool(base_verdict)

    if not relation_holds:
        reason = (
            f"{len(divergences)} of {len(variants)} meaning-preserving "
            f"variant(s) changed the stated facts"
        )
    elif not baseline_passed:
        # Consistency without correctness. Worth its own sentence because the
        # fix is in retrieval or the corpus, not in robustness.
        reason = (
            "the relation held, but the base question itself did not state the "
            "expected facts, so this is a correctness defect reported "
            "consistently rather than an invariance defect"
        )
    else:
        reason = (
            f"all {len(variants)} variant(s) stated the same facts as the base"
        )

    return MetamorphicResult(
        relation_id=relation_id,
        kind=INVARIANT,
        relation=relation,
        relation_holds=relation_holds,
        baseline_passed=baseline_passed,
        base=base,
        variants=list(variants),
        divergences=divergences,
        reason=reason,
    )


def evaluate_change(
    relation_id: str,
    relation: str,
    outcomes: Sequence[VariantOutcome],
    *,
    expect: str,
) -> MetamorphicResult:
    """The variants altered the meaning, so the behaviour must change too.

    `expect=refusal` - the variant asks something the corpus cannot answer, so
    it must be refused even though the base was answered. This is the relation
    that makes the invariance relations mean something: a bot that refuses
    everything passes those and fails this one.

    `expect=different_facts` - the variant must not state the base's facts.
    Weaker, and used where a refusal is not the right outcome.
    """
    base, variants = _split_base(outcomes)

    if base is None or base.error:
        return MetamorphicResult(
            relation_id=relation_id,
            kind=CHANGE,
            relation=relation,
            relation_holds=False,
            baseline_passed=False,
            base=base,
            variants=list(variants),
            reason=(
                f"the base question could not be evaluated: "
                f"{base.error if base else 'no base outcome'}"
            ),
        )

    if expect not in (EXPECT_REFUSAL, EXPECT_DIFFERENT_FACTS):
        raise ValueError(
            f"unknown change expectation {expect!r}; "
            f"use {EXPECT_REFUSAL!r} or {EXPECT_DIFFERENT_FACTS!r}"
        )

    # For a change relation the base must have been answered, otherwise "the
    # variant behaved differently" is trivially satisfiable.
    baseline_passed = bool(base.states_expected_facts) and not base.refused

    divergences: List[str] = []
    for variant in variants:
        if variant.error:
            divergences.append(f"{variant.label}: errored - {variant.error}")
            continue

        if expect == EXPECT_REFUSAL:
            # `refused`, not `insufficient_context`. A scope-change variant
            # often retrieves a loosely related chunk above the threshold, so
            # the gate stays open and the model declines in prose instead.
            # Reading the gate alone reported those correct refusals as
            # failures - the first thing this relation caught was a bug in
            # this check rather than in the bot.
            if not variant.refused:
                divergences.append(
                    f"{variant.label}: answered a question the corpus cannot "
                    f"support instead of refusing -> {variant.answer[:110]!r}"
                )
        elif variant.states_expected_facts:
            divergences.append(
                f"{variant.label}: still stated the base question's facts "
                f"after the meaning changed -> {variant.answer[:110]!r}"
            )

    relation_holds = not divergences

    if not relation_holds:
        reason = (
            f"{len(divergences)} of {len(variants)} meaning-changing "
            f"variant(s) did not change the behaviour"
        )
    elif not baseline_passed:
        reason = (
            "the variants behaved differently as required, but the base "
            "question was not answered correctly, so the contrast proves "
            "nothing on its own"
        )
    else:
        reason = (
            f"the base was answered and all {len(variants)} variant(s) "
            f"changed behaviour as required ({expect})"
        )

    return MetamorphicResult(
        relation_id=relation_id,
        kind=CHANGE,
        relation=relation,
        relation_holds=relation_holds,
        baseline_passed=baseline_passed,
        base=base,
        variants=list(variants),
        divergences=divergences,
        reason=reason,
    )


def evaluate_relation(
    case: Dict[str, Any], outcomes: Sequence[VariantOutcome]
) -> MetamorphicResult:
    """Dispatch on the relation kind declared by the dataset."""
    relation_id = str(case.get("id", "unnamed"))
    relation = str(case.get("relation", "variant"))
    kind = str(case.get("kind", INVARIANT))

    if kind == INVARIANT:
        return evaluate_invariance(relation_id, relation, outcomes)
    if kind == CHANGE:
        return evaluate_change(
            relation_id,
            relation,
            outcomes,
            expect=str(case.get("expect", EXPECT_REFUSAL)),
        )
    raise ValueError(
        f"{relation_id}: unknown relation kind {kind!r}; "
        f"use {INVARIANT!r} or {CHANGE!r}"
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def render_summary(results: Sequence[MetamorphicResult]) -> str:
    """A short block for the console, in the shape of the main scorecard."""
    if not results:
        return "no metamorphic relations evaluated"

    held = sum(1 for r in results if r.relation_holds)
    passed = sum(1 for r in results if r.passed)
    wrong = [r for r in results if r.consistent_but_wrong]
    variants = sum(len(r.variants) for r in results)

    lines = [
        "metamorphic",
        f"  relations        {len(results)}  ({variants} generated variants)",
        f"  relation held    {held}/{len(results)}",
        f"  passed           {passed}/{len(results)}  (relation held AND base correct)",
    ]
    if wrong:
        lines.append(
            f"  consistent but wrong  {len(wrong)}: "
            f"{', '.join(r.relation_id for r in wrong)}"
        )

    for result in results:
        if result.passed:
            continue
        lines.append(f"  {result.relation_id} ({result.relation}) {result.reason}")
        for divergence in result.divergences:
            lines.append(f"      - {divergence}")

    return "\n".join(lines)
