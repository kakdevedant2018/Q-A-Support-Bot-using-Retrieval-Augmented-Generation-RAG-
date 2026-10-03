"""Unit tests for the metamorphic and bias evaluators themselves.

Layer L0, and unmarked, so this runs in the default suite: no Ollama, no
embedding model, no network, no cost. Everything under test here is a pure
function over hand-built outcomes, which is the point - the transformations and
the comparison logic are exactly reproducible, and only the model calls in
`test_metamorphic.py` / `test_bias.py` are not.

The reason this file exists at all is the same reason
`test_evaluation_framework.py` exists: an evaluator that is itself untested
cannot be trusted to report a failure in the application. A broken invariance
check reports a green suite forever, which is worse than having no suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import bias, metamorphic
from evaluation.metamorphic import (
    BASE_LABEL,
    CHANGE,
    EXPECT_DIFFERENT_FACTS,
    EXPECT_REFUSAL,
    INVARIANT,
    apply_transformation,
    build_questions,
    evaluate_change,
    evaluate_invariance,
    evaluate_relation,
    failed_outcome,
    score_outcome,
)

RETURN_FACT = [
    {
        "id": "return-window-30-days",
        "statement": "Products can be returned within 30 days of delivery.",
        "all_of": ["30 days"],
        "any_of": ["return", "send back"],
    }
]

GOOD_ANSWER = "Customers can return products within 30 days of delivery."
WORDED_DIFFERENTLY = "You have thirty days from delivery to send the item back."
WRONG_ANSWER = "Customers can return products within 90 days of delivery."
REFUSAL = "I don't have enough information in the knowledge base to answer that."


def _response(answer: str, *, refused: bool = False) -> dict:
    return {"answer": answer, "sources": [], "insufficient_context": refused}


def _outcome(label: str, answer: str, *, refused: bool = False):
    return score_outcome(
        label, f"q-{label}", _response(answer, refused=refused), RETURN_FACT
    )


# --------------------------------------------------------------------------
# Transformations
# --------------------------------------------------------------------------


def test_every_transformation_is_meaning_preserving_enough_to_stay_a_question():
    """A transformation that empties the question would test nothing."""
    question = "What is the return period?"
    for name in metamorphic.TRANSFORMATIONS:
        transformed = apply_transformation(question, name)
        assert transformed.strip(), f"{name} produced an empty question"


def test_lowercase_and_uppercase_change_only_the_casing():
    question = "What is the return period?"
    assert apply_transformation(question, "lowercase") == question.lower()
    assert apply_transformation(question, "uppercase") == question.upper()


def test_no_punctuation_drops_only_trailing_marks():
    assert (
        apply_transformation("What is the return period?", "no_punctuation")
        == "What is the return period"
    )


def test_no_punctuation_keeps_internal_punctuation():
    """"9:00 AM" must survive; dropping the colon would change the fact."""
    assert "9:00" in apply_transformation("Are you open at 9:00 AM?", "no_punctuation")


def test_extra_whitespace_preserves_the_words_in_order():
    question = "What are the support hours?"
    transformed = apply_transformation(question, "extra_whitespace")
    assert transformed != question
    assert transformed.split() == question.split()


def test_polite_prefix_keeps_the_question_recoverable():
    transformed = apply_transformation("What is the return period?", "polite_prefix")
    assert transformed.startswith("Hi, could you please tell me:")
    assert "return period" in transformed


def test_polite_prefix_is_not_a_bare_greeting():
    """The guarantee this relation exists to prove, asserted on the input side.

    `small_talk_reply` must return None for a greeting with a question attached,
    or the question is silently swallowed. Proved here against the real
    function, with no model involved.
    """
    from app.services.small_talk import small_talk_reply

    transformed = apply_transformation("What is the return period?", "polite_prefix")
    assert small_talk_reply(transformed) is None


def test_distractor_adds_no_digits():
    """A digit in the question would interact with the groundedness check.

    `groundedness_evaluator` treats figures echoed from the question as
    supported, so a distractor carrying a number could mask an invented one.
    """
    transformed = apply_transformation("What is the return period?", "distractor")
    assert not any(character.isdigit() for character in metamorphic.DISTRACTOR_SENTENCE)
    assert metamorphic.DISTRACTOR_SENTENCE in transformed


def test_distractor_adds_no_personal_pronoun_that_would_confound_bias_testing():
    lowered = metamorphic.DISTRACTOR_SENTENCE.lower()
    for pronoun in (" he ", " she ", " his ", " her ", " him "):
        assert pronoun not in f" {lowered} "


def test_typo_is_deterministic():
    question = "How long does standard delivery take?"
    first = apply_transformation(question, "typo")
    second = apply_transformation(question, "typo")
    assert first == second
    assert first != question


def test_typo_leaves_short_questions_alone():
    """No word long enough to corrupt safely, so the input is returned as is."""
    assert apply_transformation("who pays?", "typo") == "who pays?"


def test_unknown_transformation_raises_rather_than_silently_doing_nothing():
    """A dataset typo must fail loudly, not add a vacuous passing variant."""
    with pytest.raises(KeyError, match="unknown transformation"):
        apply_transformation("anything", "lowecase")


# --------------------------------------------------------------------------
# Building question groups
# --------------------------------------------------------------------------


def test_build_questions_puts_the_base_first_and_labels_everything():
    case = {
        "id": "MRX",
        "relation": "paraphrase",
        "base_question": "What is the return period?",
        "variants": ["How long do I have to return something?"],
        "transformations": ["lowercase"],
    }
    pairs = build_questions(case)

    assert pairs[0] == (BASE_LABEL, "What is the return period?")
    assert pairs[1][0] == "paraphrase[1]"
    assert pairs[2][0] == "lowercase"
    assert len({label for label, _ in pairs}) == len(pairs), "labels must be unique"


def test_build_questions_works_with_transformations_only():
    case = {"id": "MRX", "base_question": "What are the support hours?",
            "transformations": ["lowercase", "uppercase"]}
    assert len(build_questions(case)) == 3


# --------------------------------------------------------------------------
# Invariance
# --------------------------------------------------------------------------


def test_invariance_holds_when_wording_differs_but_the_fact_does_not():
    """The core claim: this compares facts, not strings.

    The two answers share almost no vocabulary. A string or substring
    comparison would call this a divergence, which is the brittleness §3.1
    removed from the project.
    """
    result = evaluate_invariance(
        "MR01",
        "paraphrase",
        [_outcome(BASE_LABEL, GOOD_ANSWER), _outcome("paraphrase[1]", WORDED_DIFFERENTLY)],
    )

    assert result.passed
    assert result.relation_holds and result.baseline_passed
    assert result.divergences == []


def test_invariance_fails_when_a_variant_changes_the_fact():
    result = evaluate_invariance(
        "MR01",
        "paraphrase",
        [_outcome(BASE_LABEL, GOOD_ANSWER), _outcome("paraphrase[1]", WRONG_ANSWER)],
    )

    assert not result.passed
    assert not result.relation_holds
    assert result.baseline_passed
    assert "paraphrase[1]" in result.divergences[0]


def test_invariance_fails_when_a_variant_is_refused_but_the_base_was_answered():
    """The distraction failure mode: irrelevant text pushes retrieval below
    the threshold and a previously answerable question starts being refused."""
    result = evaluate_invariance(
        "MR06",
        "distraction",
        [
            _outcome(BASE_LABEL, GOOD_ANSWER),
            _outcome("distractor", REFUSAL, refused=True),
        ],
    )

    assert not result.relation_holds


def test_consistently_wrong_is_reported_as_a_correctness_defect_not_a_bias_one():
    """A bot wrong in the same way everywhere satisfies invariance.

    This has to be distinguishable, because the fix lives in retrieval or the
    corpus rather than in robustness. Reporting it as an invariance failure
    would send someone to the wrong file.
    """
    result = evaluate_invariance(
        "MR01",
        "paraphrase",
        [_outcome(BASE_LABEL, WRONG_ANSWER), _outcome("paraphrase[1]", WRONG_ANSWER)],
    )

    assert result.relation_holds
    assert not result.baseline_passed
    assert result.consistent_but_wrong
    assert not result.passed, "consistency alone must never be a pass"
    assert "correctness defect" in result.reason


def test_a_bot_that_refuses_everything_satisfies_invariance():
    """The degenerate case that makes a change relation mandatory.

    Asserted rather than merely documented, because this is the property that
    justifies the existence of MR07 and MR08 in the dataset.
    """
    result = evaluate_invariance(
        "MR01",
        "paraphrase",
        [
            _outcome(BASE_LABEL, REFUSAL, refused=True),
            _outcome("paraphrase[1]", REFUSAL, refused=True),
            _outcome("paraphrase[2]", REFUSAL, refused=True),
        ],
    )

    assert result.relation_holds, "this is the hole invariance cannot see"
    assert not result.passed, "but the gate must still be red"


def test_invariance_reports_a_variant_error_rather_than_ignoring_it():
    result = evaluate_invariance(
        "MR01",
        "paraphrase",
        [
            _outcome(BASE_LABEL, GOOD_ANSWER),
            failed_outcome("paraphrase[1]", "q", "HTTP 502"),
        ],
    )

    assert not result.relation_holds
    assert "HTTP 502" in result.divergences[0]


def test_invariance_fails_when_the_base_itself_errored():
    result = evaluate_invariance(
        "MR01", "paraphrase", [failed_outcome(BASE_LABEL, "q", "HTTP 502")]
    )

    assert not result.passed
    assert "could not be evaluated" in result.reason


def test_invariance_fails_when_no_base_was_collected():
    result = evaluate_invariance("MR01", "paraphrase", [_outcome("lowercase", GOOD_ANSWER)])

    assert not result.passed
    assert "no base outcome" in result.reason


# --------------------------------------------------------------------------
# Change
# --------------------------------------------------------------------------


def test_change_relation_passes_when_an_out_of_scope_variant_is_refused():
    result = evaluate_change(
        "MR07",
        "scope_change",
        [
            _outcome(BASE_LABEL, GOOD_ANSWER),
            _outcome("scope_change[1]", REFUSAL, refused=True),
        ],
        expect=EXPECT_REFUSAL,
    )

    assert result.passed


def test_a_prose_refusal_counts_as_a_refusal_even_when_the_retrieval_gate_stayed_open():
    """Regression. The first run of MR07/MR08 reported three false failures.

    A scope-change variant - "What is the warranty period on electronics?" -
    retrieved a loosely related chunk at 0.66, well above
    `relevance_threshold`, so `insufficient_context` came back False. The model
    then correctly answered "I do not have enough information to answer that."
    Reading the gate alone scored that correct refusal as an answer.

    The system has two refusal routes and only one of them sets the flag, so
    anything asserting on refusal has to go through `looks_like_refusal`.
    """
    gate_open_prose_refusal = score_outcome(
        "scope_change[1]",
        "What is the warranty period on electronics?",
        _response("I do not have enough information to answer that.", refused=False),
        RETURN_FACT,
    )

    assert gate_open_prose_refusal.insufficient_context is False, "gate stayed open"
    assert gate_open_prose_refusal.refused is True, "but the bot still declined"

    result = evaluate_change(
        "MR07",
        "scope_change",
        [_outcome(BASE_LABEL, GOOD_ANSWER), gate_open_prose_refusal],
        expect=EXPECT_REFUSAL,
    )
    assert result.passed


def test_the_retrieval_signature_is_captured_so_failures_can_be_attributed():
    """Without this, every divergence needs a manual `retrieve()` probe.

    Both real divergences found on the first run turned out to be retrieval
    effects rather than the model changing its mind about identical context,
    and that is the distinction that decides where the fix goes.
    """
    response = {
        "answer": GOOD_ANSWER,
        "insufficient_context": False,
        "sources": [
            {"metadata": {"source": "faq.txt"}, "relevance_score": 0.705},
            {"metadata": {"source": "support_policy.txt"}, "relevance_score": 0.383},
        ],
    }
    outcome = score_outcome("base", "q", response, RETURN_FACT)

    assert outcome.sources == ["faq.txt@0.705", "support_policy.txt@0.383"]


def test_the_retrieval_signature_survives_a_response_with_no_scores():
    """A refusal carries no sources, and that must not raise."""
    outcome = score_outcome("base", "q", _response(REFUSAL, refused=True), RETURN_FACT)
    assert outcome.sources == []

    partial = {"answer": "x", "sources": [{"metadata": {}}]}
    assert score_outcome("base", "q", partial, RETURN_FACT).sources == ["?"]


def test_a_real_answer_is_not_mistaken_for_a_refusal():
    """The other half of the regression: the detector must not over-match."""
    outcome = _outcome("scope_change[1]", "The warranty period is 24 months.")
    assert outcome.refused is False


def test_change_relation_fails_when_the_variant_is_answered_anyway():
    """Answering a question the corpus cannot support is the headline defect."""
    result = evaluate_change(
        "MR07",
        "scope_change",
        [
            _outcome(BASE_LABEL, GOOD_ANSWER),
            _outcome("scope_change[1]", "Under German law the period is 30 days."),
        ],
        expect=EXPECT_REFUSAL,
    )

    assert not result.passed
    assert "instead of refusing" in result.divergences[0]


def test_change_relation_catches_the_refuse_everything_bot():
    """The complement of `test_a_bot_that_refuses_everything_satisfies_invariance`.

    Together these two tests prove the suite can only be satisfied by a system
    that answers in-scope questions and refuses out-of-scope ones.
    """
    result = evaluate_change(
        "MR07",
        "scope_change",
        [
            _outcome(BASE_LABEL, REFUSAL, refused=True),
            _outcome("scope_change[1]", REFUSAL, refused=True),
        ],
        expect=EXPECT_REFUSAL,
    )

    assert not result.passed
    assert not result.baseline_passed


def test_change_relation_with_different_facts_expectation():
    result = evaluate_change(
        "MRX",
        "scope_change",
        [
            _outcome(BASE_LABEL, GOOD_ANSWER),
            _outcome("scope_change[1]", "Warranty terms are not covered here."),
        ],
        expect=EXPECT_DIFFERENT_FACTS,
    )

    assert result.passed


def test_change_relation_rejects_an_unknown_expectation():
    with pytest.raises(ValueError, match="unknown change expectation"):
        evaluate_change(
            "MRX", "scope_change", [_outcome(BASE_LABEL, GOOD_ANSWER)], expect="vibes"
        )


def test_evaluate_relation_dispatches_on_kind():
    outcomes = [_outcome(BASE_LABEL, GOOD_ANSWER), _outcome("lowercase", GOOD_ANSWER)]

    assert evaluate_relation({"id": "a", "kind": INVARIANT}, outcomes).kind == INVARIANT
    assert (
        evaluate_relation(
            {"id": "b", "kind": CHANGE, "expect": EXPECT_DIFFERENT_FACTS}, outcomes
        ).kind
        == CHANGE
    )


def test_evaluate_relation_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown relation kind"):
        evaluate_relation({"id": "x", "kind": "sideways"}, [_outcome(BASE_LABEL, GOOD_ANSWER)])


def test_metamorphic_summary_names_the_failing_relation():
    failing = evaluate_invariance(
        "MR01",
        "paraphrase",
        [_outcome(BASE_LABEL, GOOD_ANSWER), _outcome("paraphrase[1]", WRONG_ANSWER)],
    )
    rendered = metamorphic.render_summary([failing])

    assert "MR01" in rendered and "paraphrase[1]" in rendered


def test_metamorphic_summary_handles_no_results():
    assert "no metamorphic relations" in metamorphic.render_summary([])


# --------------------------------------------------------------------------
# Bias
# --------------------------------------------------------------------------


def _bias_outcome(group: str, answer: str, *, refused: bool = False):
    return bias.score_outcome(
        group, f"q-{group}", _response(answer, refused=refused), RETURN_FACT
    )


def test_personas_have_unique_groups_and_one_baseline():
    groups = [p.group for p in bias.PERSONAS]
    assert len(groups) == len(set(groups))
    assert groups.count(bias.BASELINE_GROUP) == 1
    assert groups[0] == bias.BASELINE_GROUP, "the baseline is rendered first"


def test_the_baseline_persona_carries_no_given_name():
    """Otherwise the choice of that name becomes part of the measurement."""
    baseline = next(p for p in bias.PERSONAS if p.group == bias.BASELINE_GROUP)
    assert baseline.name == "the customer"
    assert baseline.subject == "they"


def test_the_european_pair_isolates_the_gendered_token():
    """Same surname, so that pair differs only in given name and pronouns."""
    feminine = next(p for p in bias.PERSONAS if p.group == "feminine_european")
    masculine = next(p for p in bias.PERSONAS if p.group == "masculine_european")
    assert feminine.name.split()[-1] == masculine.name.split()[-1]
    assert feminine.subject != masculine.subject


def test_render_substitutes_every_placeholder():
    template = "{name} asked whether {subject} can use {possessive} card. Tell {object}."
    rendered = bias.render(template, bias.PERSONAS[1])

    assert "{" not in rendered
    assert "Emily Clarke" in rendered


def test_render_rejects_an_unknown_placeholder():
    """A literal "{subjet}" reaching the bot is a silently weaker test."""
    with pytest.raises(KeyError, match="unknown placeholder"):
        bias.render("Can {subjet} return it?", bias.PERSONAS[0])


def test_build_questions_renders_one_question_per_persona_baseline_first():
    pairs = bias.build_questions("Can {subject} return it?")

    assert len(pairs) == len(bias.PERSONAS)
    assert pairs[0][0] == bias.BASELINE_GROUP
    assert len({group for group, _ in pairs}) == len(pairs)


def test_bias_passes_when_every_group_gets_the_same_facts():
    outcomes = [
        _bias_outcome(p.group, GOOD_ANSWER if i % 2 == 0 else WORDED_DIFFERENTLY)
        for i, p in enumerate(bias.PERSONAS)
    ]
    result = bias.evaluate_bias("B01", outcomes)

    assert result.passed
    assert result.disparity == 0.0


def test_bias_fails_when_one_group_gets_a_different_answer():
    outcomes = [_bias_outcome(p.group, GOOD_ANSWER) for p in bias.PERSONAS]
    outcomes[3] = _bias_outcome(bias.PERSONAS[3].group, REFUSAL, refused=True)

    result = bias.evaluate_bias("B01", outcomes)

    assert not result.passed
    assert result.divergent_groups == [bias.PERSONAS[3].group]
    assert 0 < result.disparity < 1


def test_bias_flags_a_group_that_does_better_than_the_baseline_too():
    """Divergence in either direction is the same defect.

    If a named persona gets a correct answer the baseline does not, the model
    is still conditioning on the persona.
    """
    outcomes = [_bias_outcome(p.group, WRONG_ANSWER) for p in bias.PERSONAS]
    outcomes[2] = _bias_outcome(bias.PERSONAS[2].group, GOOD_ANSWER)

    result = bias.evaluate_bias("B01", outcomes)

    assert not result.passed
    assert bias.PERSONAS[2].group in result.divergent_groups


def test_bias_equal_but_wrong_is_reported_as_consistency_not_correctness():
    outcomes = [_bias_outcome(p.group, WRONG_ANSWER) for p in bias.PERSONAS]
    result = bias.evaluate_bias("B01", outcomes)

    assert result.passed, "equal treatment is the property under test"
    assert "consistency rather than correctness" in result.reason


def test_bias_fails_when_the_baseline_errored():
    outcomes = [bias.failed_outcome(bias.BASELINE_GROUP, "q", "HTTP 502")]
    result = bias.evaluate_bias("B01", outcomes)

    assert not result.passed
    assert "could not be evaluated" in result.reason


def test_bias_fails_when_the_baseline_is_absent():
    outcomes = [_bias_outcome("feminine_european", GOOD_ANSWER)]
    result = bias.evaluate_bias("B01", outcomes)

    assert not result.passed
    assert "no outcome was collected" in result.reason


def test_bias_counts_an_errored_group_as_divergent():
    outcomes = [_bias_outcome(p.group, GOOD_ANSWER) for p in bias.PERSONAS]
    outcomes[1] = bias.failed_outcome(bias.PERSONAS[1].group, "q", "HTTP 502")

    result = bias.evaluate_bias("B01", outcomes)

    assert not result.passed
    assert bias.PERSONAS[1].group in result.divergent_groups


def test_bias_summary_names_the_failing_template():
    outcomes = [_bias_outcome(p.group, GOOD_ANSWER) for p in bias.PERSONAS]
    outcomes[1] = _bias_outcome(bias.PERSONAS[1].group, WRONG_ANSWER)
    rendered = bias.render_summary([bias.evaluate_bias("B01", outcomes)])

    assert "B01" in rendered and bias.PERSONAS[1].group in rendered


def test_bias_summary_handles_no_results():
    assert "no bias templates" in bias.render_summary([])


# --------------------------------------------------------------------------
# The datasets, treated as code
#
# Same policy as test_evaluation_framework.py: a dataset bug is a test bug, and
# these checks have historically caught more of them than anything else.
# --------------------------------------------------------------------------

DATASET_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "datasets"
METAMORPHIC_CASES = json.loads((DATASET_DIR / "metamorphic.json").read_text("utf-8"))
BIAS_CASES = json.loads((DATASET_DIR / "bias.json").read_text("utf-8"))


def test_relation_ids_are_unique():
    ids = [c["id"] for c in METAMORPHIC_CASES]
    assert len(ids) == len(set(ids))
    bias_ids = [c["id"] for c in BIAS_CASES]
    assert len(bias_ids) == len(set(bias_ids))


@pytest.mark.parametrize("case", METAMORPHIC_CASES, ids=[c["id"] for c in METAMORPHIC_CASES])
def test_every_metamorphic_question_satisfies_the_api_contract(case):
    """A variant the API would reject with a 422 tests nothing.

    Bounds come from AskRequest, so this fails if the two drift apart - and the
    generated transformations are checked too, not just the hand-written text.
    `extra_whitespace` and `polite_prefix` both lengthen the question.
    """
    from app.schemas import AskRequest

    for _, question in build_questions(case):
        AskRequest(question=question)


@pytest.mark.parametrize("case", BIAS_CASES, ids=[c["id"] for c in BIAS_CASES])
def test_every_rendered_bias_question_satisfies_the_api_contract(case):
    from app.schemas import AskRequest

    for _, question in bias.build_questions(case["template"]):
        AskRequest(question=question)


@pytest.mark.parametrize("case", METAMORPHIC_CASES, ids=[c["id"] for c in METAMORPHIC_CASES])
def test_every_relation_declares_a_known_kind_and_usable_facts(case):
    assert case["kind"] in (INVARIANT, CHANGE), case["id"]
    if case["kind"] == CHANGE:
        assert case["expect"] in (EXPECT_REFUSAL, EXPECT_DIFFERENT_FACTS), case["id"]

    facts = case.get("expected_facts")
    assert facts, f"{case['id']} has no expected facts"
    for fact in facts:
        assert fact.get("all_of") or fact.get("any_of"), (
            f"{case['id']}/{fact.get('id')} declares no matching rule"
        )


@pytest.mark.parametrize("case", METAMORPHIC_CASES, ids=[c["id"] for c in METAMORPHIC_CASES])
def test_every_relation_actually_has_something_to_compare(case):
    """A relation with no variants is a single question wearing a costume."""
    questions = build_questions(case)
    assert len(questions) >= 2, f"{case['id']} generates no variants"


@pytest.mark.parametrize("case", METAMORPHIC_CASES, ids=[c["id"] for c in METAMORPHIC_CASES])
def test_every_named_transformation_exists(case):
    for name in case.get("transformations", []) or []:
        assert name in metamorphic.TRANSFORMATIONS, f"{case['id']}: {name}"


def test_the_dataset_contains_at_least_one_change_relation():
    """Without one, the whole metamorphic suite is satisfied by a constant bot.

    This is the dataset-level counterpart to
    `test_a_bot_that_refuses_everything_satisfies_invariance`.
    """
    kinds = {c["kind"] for c in METAMORPHIC_CASES}
    assert CHANGE in kinds, (
        "an invariance-only metamorphic dataset is vacuous: a bot that refuses "
        "every question would pass all of it"
    )


@pytest.mark.parametrize("case", BIAS_CASES, ids=[c["id"] for c in BIAS_CASES])
def test_every_bias_template_varies_the_persona(case):
    """A template with no placeholder would ask six identical questions."""
    template = case["template"]
    assert any(f"{{{p}}}" in template for p in bias.PLACEHOLDERS), case["id"]

    rendered = {question for _, question in bias.build_questions(template)}
    assert len(rendered) > 1, f"{case['id']} renders the same question for every group"


@pytest.mark.parametrize("case", BIAS_CASES, ids=[c["id"] for c in BIAS_CASES])
def test_every_bias_template_declares_usable_facts(case):
    facts = case.get("expected_facts")
    assert facts, f"{case['id']} has no expected facts"
    for fact in facts:
        assert fact.get("all_of") or fact.get("any_of"), (
            f"{case['id']}/{fact.get('id')} declares no matching rule"
        )
