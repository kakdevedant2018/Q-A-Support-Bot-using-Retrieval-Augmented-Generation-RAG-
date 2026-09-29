"""Unit tests for output validation.

These hit each rejection branch directly. Driving them through the HTTP layer
would work, but every malformed payload would collapse into the same generic
500, so the test could not tell which rule actually fired.
"""

import pytest

from app.services.exceptions import InvalidAnswerError
from app.services.response_validator import (
    MAX_ANSWER_CHARS,
    validate_answer_payload,
)


def _valid_payload():
    return {
        "answer": "You can return products within 30 days of delivery.",
        "sources": [
            {"content": "Customers can return products within 30 days.", "metadata": {}}
        ],
        "insufficient_context": False,
    }


def test_valid_payload_passes_through_unchanged():
    payload = _valid_payload()
    assert validate_answer_payload(payload) is payload


def test_declined_answer_with_no_sources_is_allowed():
    """The honest refusal is the one legitimate case of zero sources."""
    payload = {
        "answer": "I don't have enough information to answer that.",
        "sources": [],
        "insufficient_context": True,
    }
    assert validate_answer_payload(payload)


@pytest.mark.parametrize("payload", ["a string", 42, None, ["a", "list"]])
def test_non_dict_payload_is_rejected(payload):
    with pytest.raises(InvalidAnswerError):
        validate_answer_payload(payload)


@pytest.mark.parametrize("answer", [None, 42, [], {}, True])
def test_non_string_answer_is_rejected(answer):
    payload = _valid_payload()
    payload["answer"] = answer
    with pytest.raises(InvalidAnswerError):
        validate_answer_payload(payload)


@pytest.mark.parametrize("answer", ["", "   ", "\n\t "])
def test_blank_answer_is_rejected(answer):
    payload = _valid_payload()
    payload["answer"] = answer
    with pytest.raises(InvalidAnswerError):
        validate_answer_payload(payload)


def test_absurdly_long_answer_is_rejected():
    """A runaway generation is a defect, not a response to forward."""
    payload = _valid_payload()
    payload["answer"] = "a" * (MAX_ANSWER_CHARS + 1)
    with pytest.raises(InvalidAnswerError):
        validate_answer_payload(payload)


def test_answer_at_the_length_limit_is_accepted():
    payload = _valid_payload()
    payload["answer"] = "a" * MAX_ANSWER_CHARS
    assert validate_answer_payload(payload)


@pytest.mark.parametrize("sources", [None, "not-a-list", 42, {}])
def test_non_list_sources_is_rejected(sources):
    payload = _valid_payload()
    payload["sources"] = sources
    with pytest.raises(InvalidAnswerError):
        validate_answer_payload(payload)


def test_source_that_is_not_an_object_is_rejected():
    payload = _valid_payload()
    payload["sources"] = ["just a string"]
    with pytest.raises(InvalidAnswerError):
        validate_answer_payload(payload)


def test_source_without_string_content_is_rejected():
    payload = _valid_payload()
    payload["sources"] = [{"content": 123, "metadata": {}}]
    with pytest.raises(InvalidAnswerError):
        validate_answer_payload(payload)


def test_source_with_non_object_metadata_is_rejected():
    payload = _valid_payload()
    payload["sources"] = [{"content": "text", "metadata": "faq.txt"}]
    with pytest.raises(InvalidAnswerError):
        validate_answer_payload(payload)


def test_source_missing_metadata_defaults_to_empty():
    """Metadata is optional; its absence is not a malformed source."""
    payload = _valid_payload()
    payload["sources"] = [{"content": "text"}]
    assert validate_answer_payload(payload)


def test_unsourced_confident_answer_is_rejected():
    """The core rule: prose with no citation and no admission of uncertainty."""
    payload = {
        "answer": "You can return items whenever you like.",
        "sources": [],
        "insufficient_context": False,
    }
    with pytest.raises(InvalidAnswerError, match="cites no sources"):
        validate_answer_payload(payload)


def test_error_message_names_the_offending_index():
    """A validator message should point at the specific bad source."""
    payload = _valid_payload()
    payload["sources"] = [
        {"content": "fine", "metadata": {}},
        {"content": None, "metadata": {}},
    ]
    with pytest.raises(InvalidAnswerError, match=r"sources\[1\]"):
        validate_answer_payload(payload)
