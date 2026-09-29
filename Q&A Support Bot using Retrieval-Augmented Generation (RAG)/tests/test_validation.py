"""Input validation tests (V02 to V09).

Everything here is mocked, so these tests exercise the validation layer without
ever reaching the model. That is the point: a validation test that triggers a
real generation call is slow, non-deterministic, and on a paid provider it also
costs money for input that should never have got past the schema.
"""

import pytest

pytestmark = pytest.mark.api

ASK_URL = "/api/v1/ask"


def test_empty_question_is_rejected(client):
    """V02: an empty string is below the minimum length."""
    response = client.post(ASK_URL, json={"question": ""})
    assert response.status_code == 422


def test_missing_question_field_is_rejected(client):
    """V03: the field is required."""
    response = client.post(ASK_URL, json={})
    assert response.status_code == 422


def test_null_question_is_rejected(client):
    """V04: null is not a string."""
    response = client.post(ASK_URL, json={"question": None})
    assert response.status_code == 422


@pytest.mark.parametrize("value", [12345, 3.14, True, ["a list"], {"a": "dict"}])
def test_wrong_type_is_rejected(client, value):
    """V05: non-string types are rejected rather than coerced."""
    response = client.post(ASK_URL, json={"question": value})
    assert response.status_code == 422


def test_question_at_minimum_length_is_accepted(client):
    """V06: three characters is the documented lower bound."""
    response = client.post(ASK_URL, json={"question": "abc"})
    assert response.status_code == 200


def test_question_at_maximum_length_is_accepted(client):
    """V07: one thousand characters is the documented upper bound."""
    response = client.post(ASK_URL, json={"question": "a" * 1000})
    assert response.status_code == 200


def test_question_over_maximum_length_is_rejected(client):
    """V08: one character past the bound is refused."""
    response = client.post(ASK_URL, json={"question": "a" * 1001})
    assert response.status_code == 422


@pytest.mark.parametrize("value", ["   ", "\t\t\t", "\n\n\n", "     \t  \n "])
def test_whitespace_only_question_is_rejected(client, value):
    """V09: this is the case a length check alone lets through.

    "   " is three characters, so min_length is satisfied. Only the business
    rule catches it.
    """
    response = client.post(ASK_URL, json={"question": value})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "question, expected_status",
    [
        ("a", 422),
        ("ab", 422),
        ("abc", 200),
        ("a" * 999, 200),
        ("a" * 1000, 200),
        ("a" * 1001, 422),
    ],
)
def test_length_boundaries(client, question, expected_status):
    """Boundary sweep either side of both limits."""
    response = client.post(ASK_URL, json={"question": question})
    assert response.status_code == expected_status


def test_validation_error_reports_the_offending_field(client):
    """A 422 should tell the caller what to fix."""
    response = client.post(ASK_URL, json={})

    detail = response.json()["detail"]
    assert any("question" in str(item.get("loc", "")) for item in detail)


def test_form_encoded_body_is_rejected(client):
    """A common early mistake: posting form data instead of JSON."""
    response = client.post(ASK_URL, data={"question": "What is the return policy?"})
    assert response.status_code == 422


def test_malformed_json_is_rejected(client):
    """Broken JSON is a client error, never a 500."""
    response = client.post(
        ASK_URL,
        content=b'{"question": "unterminated',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_extra_unknown_fields_are_ignored(client):
    """An unexpected extra key must not break a valid request."""
    response = client.post(
        ASK_URL,
        json={"question": "What is the return policy?", "unexpected": "value"},
    )
    assert response.status_code == 200
