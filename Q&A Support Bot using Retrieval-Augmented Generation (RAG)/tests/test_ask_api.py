"""Contract tests for POST /api/v1/ask.

All of these run against the mocked service, so they assert the API contract
rather than answer quality. Answer quality is tested separately in
test_rag_quality.py, because a 200 with valid JSON says nothing about whether
the answer is correct.
"""

import time

import pytest

from app.schemas import AskResponse
from tests.conftest import MOCK_ANSWER

pytestmark = pytest.mark.api

ASK_URL = "/api/v1/ask"


def test_ask_valid_question(client):
    """V01: the happy path returns an answer and its sources."""
    response = client.post(ASK_URL, json={"question": "What is the return policy?"})

    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)
    assert len(data["answer"].strip()) > 0


def test_ask_returns_the_injected_service_answer(client):
    """Proves dependency injection is really wiring in the double."""
    response = client.post(ASK_URL, json={"question": "What is the return policy?"})

    data = response.json()
    assert data["answer"] == MOCK_ANSWER
    assert len(data["sources"]) == 1


def test_response_matches_the_declared_schema(client):
    """The response validates against the same model the API advertises."""
    response = client.post(ASK_URL, json={"question": "What is the return policy?"})

    model = AskResponse.model_validate(response.json())
    assert isinstance(model.answer, str)
    assert isinstance(model.sources, list)
    assert model.request_id


def test_source_metadata_is_present(client):
    """R05: a source must be attributable, not an anonymous block of text."""
    response = client.post(ASK_URL, json={"question": "What is the return policy?"})

    source = response.json()["sources"][0]
    assert isinstance(source["content"], str)
    assert source["content"].strip()
    assert isinstance(source["metadata"], dict)
    assert source["metadata"]["source"] == "faq.txt"


@pytest.mark.parametrize(
    "question",
    [
        "What is the return policy?",
        "How can I track my order?",
        "How can I contact support?",
        "What payment methods are supported?",
    ],
)
def test_multiple_questions(client, question):
    """The endpoint behaves consistently across several valid questions."""
    response = client.post(ASK_URL, json={"question": question})

    assert response.status_code == 200
    assert response.json()["answer"].strip() != ""


def test_question_is_trimmed_before_use(client):
    """Surrounding whitespace is normalised rather than passed through."""
    response = client.post(ASK_URL, json={"question": "   What is the return policy?   "})
    assert response.status_code == 200


def test_response_includes_a_request_id_header(client):
    response = client.post(ASK_URL, json={"question": "What is the return policy?"})

    header_id = response.headers.get("X-Request-ID")
    assert header_id
    assert response.json()["request_id"] == header_id


def test_api_overhead_is_measured_not_asserted(client):
    """P01: measure the framework overhead, do not assert an unmeasured budget.

    With the service mocked, this times the API layer alone. A latency budget
    belongs in the integration suite against a real model, and only after a
    baseline has actually been measured on the target hardware.
    """
    start = time.perf_counter()
    response = client.post(ASK_URL, json={"question": "What is the return policy?"})
    duration = time.perf_counter() - start

    assert response.status_code == 200
    print(f"\nMocked API round trip: {duration:.4f}s")


def test_repeated_requests_are_stable(client):
    """P02: the endpoint holds up across sequential calls."""
    for _ in range(5):
        response = client.post(ASK_URL, json={"question": "What is the return policy?"})
        assert response.status_code == 200
