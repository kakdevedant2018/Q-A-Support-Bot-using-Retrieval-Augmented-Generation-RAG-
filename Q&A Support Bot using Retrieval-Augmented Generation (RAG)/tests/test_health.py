"""Health endpoint tests (H01 to H03)."""

import pytest

from app import __version__

pytestmark = pytest.mark.api


def test_health_returns_200(client):
    """H01: the health endpoint is reachable."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema(client):
    """H02: the health payload has the documented shape."""
    response = client.get("/health")
    data = response.json()

    assert data["status"] == "healthy"
    assert data["version"] == __version__


def test_health_rejects_unsupported_method(client):
    """H03: POST to a GET-only route is a 405, not a 404 or a 500."""
    response = client.post("/health", json={})
    assert response.status_code == 405


def test_health_does_not_depend_on_the_knowledge_base(client):
    """Liveness must not fail just because ingestion has not run.

    If /health touched the vector store, a missing index would take the whole
    service out of a load balancer pool instead of returning an actionable 503
    from /ask.
    """
    response = client.get("/health")
    assert response.status_code == 200


def test_every_response_carries_a_request_id(client):
    """A user can quote this id and the log line can be found."""
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")
