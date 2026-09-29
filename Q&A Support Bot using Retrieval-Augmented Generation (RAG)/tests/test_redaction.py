"""Unit tests for the credential scrub on the way out of the service.

These need neither a model nor an index, so they run in the default suite. The
integration counterpart - a real key in a real indexed document, retrieved by a
real question - lives in `test_security.py`; this file covers the mechanics that
test depends on, and it is the mechanics that will break first when someone
touches a regular expression.

Marked `security` so `pytest -m security` picks the whole story up at once, and
left unmarked for `integration` so it stays in the fast run.
"""

from __future__ import annotations

import pytest

from app.services.redaction import (
    REDACTED,
    contains_secret,
    redact_response,
    redact_sources,
    redact_text,
)

pytestmark = [pytest.mark.security]


# The value in each pair is what must not survive. Written as (input, forbidden)
# rather than (input, expected output) on purpose: asserting the exact output
# string would pin the test to the current formatting of the replacement, and the
# property that matters is only that the secret is gone.
SECRETS = [
    pytest.param("api key: sk-poisoned-1234567890abcdef", "sk-poisoned", id="sk-key"),
    pytest.param("API_KEY=hunter2reallylongvalue", "hunter2reallylongvalue", id="env-style"),
    pytest.param("password = correcthorsebattery", "correcthorsebattery", id="password"),
    pytest.param("Authorization: Bearer abcdef0123456789xyz", "abcdef0123456789xyz", id="bearer"),
    pytest.param("secret: s3cr3t-value-here", "s3cr3t-value-here", id="secret-label"),
    pytest.param("token: ghp_abcdefghij0123456789", "ghp_abcdefghij0123456789", id="github-pat"),
    pytest.param(
        "digest 5d41402abc4b2a76b9719d911017c592aaaaaaaa",
        "5d41402abc4b2a76b9719d911017c592aaaaaaaa",
        id="long-hex",
    ),
]


@pytest.mark.parametrize("text,forbidden", SECRETS)
def test_a_credential_does_not_survive_redaction(text, forbidden):
    cleaned = redact_text(text)
    assert forbidden not in cleaned, cleaned
    assert REDACTED in cleaned


@pytest.mark.parametrize("text,forbidden", SECRETS)
def test_contains_secret_agrees_with_redact_text(text, forbidden):
    """The detector and the scrubber have to see the same thing.

    They are used for different purposes - one decides whether to log a warning,
    the other rewrites the string - so a pattern added to one and not the other
    would produce a silent redaction or a warning about nothing.
    """
    assert contains_secret(text)
    assert not contains_secret(redact_text(text))


# Answers the bot is supposed to give. Over-redaction is the failure mode that
# would go unnoticed: it does not raise, it just quietly mangles good answers.
CLEAN_ANSWERS = [
    "Customers can return products within 30 days of delivery if unused.",
    "You can contact support through email or the customer service portal.",
    "Standard shipping takes 5-7 business days and costs $4.99.",
    "Order status values are Processing, Shipped, and Delivered.",
    "I don't have enough information in the knowledge base to answer that.",
    # Numbers and tracking-shaped strings must pass through. The long-hex rule is
    # the one most likely to catch these by accident.
    "Your tracking number 1Z999AA10123456784 was issued on 2026-02-14.",
    "Reference 12345678 was refunded in full.",
]


@pytest.mark.parametrize("answer", CLEAN_ANSWERS)
def test_a_normal_answer_is_returned_untouched(answer):
    assert redact_text(answer) == answer
    assert not contains_secret(answer)


def test_the_word_password_alone_is_not_a_secret():
    """A sentence about passwords is not a password.

    The pattern needs a label *and* a value, so documentation that merely uses
    the word has to survive - otherwise a knowledge base with a
    'how do I reset my password' entry would answer in redaction markers.
    """
    sentence = "To reset your password, use the link on the sign-in page."
    assert redact_text(sentence) == sentence


def test_empty_and_missing_values_are_handled():
    assert redact_text("") == ""
    assert not contains_secret("")
    assert redact_sources(None) == []
    assert redact_sources([]) == []


def test_retrieved_chunks_are_scrubbed_as_well_as_the_answer():
    """`/api/v1/ask` returns the chunks it used, so they leak independently.

    Scrubbing only the answer would move the problem rather than fix it: the key
    would still be sitting in `sources[0].content` for anyone reading the JSON.
    """
    response = {
        "answer": "The addendum lists api key: sk-poisoned-1234567890abcdef.",
        "sources": [
            {
                "content": "Return authorisation. api key: sk-poisoned-1234567890abcdef",
                "metadata": {"source": "returns_addendum.txt"},
                "relevance_score": 0.71,
            }
        ],
        "insufficient_context": False,
    }

    cleaned = redact_response(response)

    assert "sk-poisoned" not in cleaned["answer"]
    assert "sk-poisoned" not in cleaned["sources"][0]["content"]
    # Everything else has to arrive intact, or the scrub has broken the contract
    # the API tests assert on.
    assert cleaned["sources"][0]["metadata"] == {"source": "returns_addendum.txt"}
    assert cleaned["sources"][0]["relevance_score"] == 0.71
    assert cleaned["insufficient_context"] is False


def test_redaction_does_not_mutate_the_original_response():
    """The service builds the dict, then redacts it; both must be usable.

    If redaction edited in place, a caller that logged the pre-redaction object -
    or the evaluation harness holding a reference - would see a different value
    than it expected, which is the kind of bug that only shows up under one
    code path.
    """
    source = {"content": "api key: sk-poisoned-1234567890abcdef", "metadata": {}}
    original = {"answer": "sk-poisoned-1234567890abcdef", "sources": [source]}

    redact_response(original)

    assert original["answer"] == "sk-poisoned-1234567890abcdef"
    assert source["content"] == "api key: sk-poisoned-1234567890abcdef"


def test_the_product_and_the_evaluator_detect_the_same_secrets():
    """The copy in `app` must not drift from the one in `evaluation`.

    `app.services.redaction` deliberately duplicates the patterns rather than
    importing them, so that the shipped service does not depend on the test
    harness. The cost of that choice is exactly this test: if the evaluator
    learns about a new credential shape and the product does not, the scorecard
    would report a leak the product had no ability to stop.
    """
    from evaluation.safety_evaluator import SECRET_PATTERNS as EVALUATOR_PATTERNS

    for text, _forbidden in ((p.values[0], p.values[1]) for p in SECRETS):
        detected_by_evaluator = any(p.search(text) for p in EVALUATOR_PATTERNS)
        if detected_by_evaluator:
            assert contains_secret(text), (
                f"the evaluator flags {text!r} as a credential but the service "
                "would return it unredacted"
            )


@pytest.mark.parametrize("text,forbidden", SECRETS)
def test_redaction_is_idempotent(text, forbidden):
    """Scrubbing twice must equal scrubbing once.

    Not a theoretical concern: the labelled pattern originally treated its own
    `[REDACTED]` marker as a value, so a second pass rewrote the marker and
    `contains_secret` kept returning True on clean text - which made the service
    log a warning every time it had successfully prevented a leak.
    """
    once = redact_text(text)
    assert redact_text(once) == once
