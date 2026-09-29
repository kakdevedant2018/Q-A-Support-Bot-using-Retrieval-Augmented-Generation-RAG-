"""Error handling tests (E01 to E04).

Each failure mode gets its own status code. That distinction is the reason the
service raises typed errors instead of letting everything collapse into one
broad `except Exception`: a caller can tell "come back later, the index is
missing" apart from "this request was wrong".

Every test also checks that nothing internal escapes. A stack trace, a file
path, or a config value in an error body is an information disclosure defect,
not a debugging aid.
"""

from typing import Any, Dict

import pytest

from app.api.routes import MSG_INTERNAL, MSG_LLM_DOWN, MSG_NO_KNOWLEDGE_BASE
from app.services.exceptions import (
    InvalidAnswerError,
    KnowledgeBaseUnavailableError,
    LLMUnavailableError,
)
from tests.conftest import FailingRAGService

pytestmark = pytest.mark.api

ASK_URL = "/api/v1/ask"
VALID_BODY = {"question": "What is the return policy?"}

# Strings that must never appear in a response body.
LEAK_MARKERS = [
    "Traceback",
    "secret_token",
    "/internal/path",
    "rag.py",
    "line ",
    "File \"",
]


class LLMDownService:
    def ask(self, question: str) -> Dict[str, Any]:
        raise LLMUnavailableError("ollama refused the connection on 127.0.0.1:11434")


class NoKnowledgeBaseService:
    def ask(self, question: str) -> Dict[str, Any]:
        raise KnowledgeBaseUnavailableError(
            "vector store empty at C:\\Users\\dev\\chroma_db"
        )


class BadPayloadService:
    """Returns a structurally invalid payload, so output validation must catch it."""

    def ask(self, question: str) -> Dict[str, Any]:
        return {"answer": "", "sources": "not-a-list"}


class UnsourcedAnswerService:
    """Produces prose attributable to nothing, without declaring low confidence."""

    def ask(self, question: str) -> Dict[str, Any]:
        return {"answer": "You can return items whenever you like.", "sources": []}


class InvalidAnswerRaisingService:
    def ask(self, question: str) -> Dict[str, Any]:
        raise InvalidAnswerError("answer failed an internal check")


def _assert_no_internal_leak(response) -> None:
    body = response.text
    for marker in LEAK_MARKERS:
        assert marker not in body, f"response leaked {marker!r}"


def test_unexpected_exception_returns_a_safe_500(make_client):
    """E03: an unforeseen error becomes a generic 500."""
    client = make_client(FailingRAGService)
    response = client.post(ASK_URL, json=VALID_BODY)

    assert response.status_code == 500
    assert response.json()["detail"] == MSG_INTERNAL


def test_unexpected_exception_does_not_leak_internals(make_client):
    """The simulated failure embeds a token and a path. Neither may escape."""
    client = make_client(FailingRAGService)
    response = client.post(ASK_URL, json=VALID_BODY)

    _assert_no_internal_leak(response)


def test_llm_failure_returns_502(make_client):
    """E01: the model being down is an upstream failure, not a client error."""
    client = make_client(LLMDownService)
    response = client.post(ASK_URL, json=VALID_BODY)

    assert response.status_code == 502
    assert response.json()["detail"] == MSG_LLM_DOWN
    _assert_no_internal_leak(response)


def test_llm_failure_hides_the_host_and_port(make_client):
    """Infrastructure detail stays in the log, not in the response."""
    client = make_client(LLMDownService)
    response = client.post(ASK_URL, json=VALID_BODY)

    assert "11434" not in response.text
    assert "127.0.0.1" not in response.text


def test_missing_knowledge_base_returns_503(make_client):
    """E02: no index means the service is not ready yet."""
    client = make_client(NoKnowledgeBaseService)
    response = client.post(ASK_URL, json=VALID_BODY)

    assert response.status_code == 503
    assert response.json()["detail"] == MSG_NO_KNOWLEDGE_BASE
    _assert_no_internal_leak(response)


def test_missing_knowledge_base_hides_the_filesystem_path(make_client):
    """A local path in an error body is an information disclosure."""
    client = make_client(NoKnowledgeBaseService)
    response = client.post(ASK_URL, json=VALID_BODY)

    assert "chroma_db" not in response.text
    assert "Users" not in response.text


def test_structurally_invalid_payload_never_reaches_the_user(make_client):
    """Output validation turns a malformed answer into a 500."""
    client = make_client(BadPayloadService)
    response = client.post(ASK_URL, json=VALID_BODY)

    assert response.status_code == 500
    assert response.json()["detail"] == MSG_INTERNAL


def test_answer_without_sources_is_blocked(make_client):
    """An unsourced claim is the exact failure mode RAG exists to prevent."""
    client = make_client(UnsourcedAnswerService)
    response = client.post(ASK_URL, json=VALID_BODY)

    assert response.status_code == 500
    assert "return items whenever you like" not in response.text


def test_invalid_answer_error_maps_to_500(make_client):
    client = make_client(InvalidAnswerRaisingService)
    response = client.post(ASK_URL, json=VALID_BODY)

    assert response.status_code == 500
    assert response.json()["detail"] == MSG_INTERNAL


@pytest.mark.parametrize(
    "service, expected_status",
    [
        (FailingRAGService, 500),
        (LLMDownService, 502),
        (NoKnowledgeBaseService, 503),
        (BadPayloadService, 500),
    ],
)
def test_error_bodies_are_always_json_with_a_detail_field(
    make_client, service, expected_status
):
    """A client can parse every error uniformly, whatever went wrong."""
    client = make_client(service)
    response = client.post(ASK_URL, json=VALID_BODY)

    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json()["detail"], str)


def test_missing_configuration_still_allows_import(client):
    """E04: the app runs with no .env present.

    Every setting has a default and no paid API key is required, so a missing
    configuration file cannot stop the service from starting. Startup failures
    caused by absent secrets are a whole class of problem this stack avoids.
    """
    from app.config import Settings

    fresh = Settings(_env_file=None)
    assert fresh.llm_model
    assert fresh.chroma_dir
    assert client.get("/health").status_code == 200
