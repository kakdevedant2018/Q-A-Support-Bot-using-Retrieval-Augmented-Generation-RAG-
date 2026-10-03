"""Shared test fixtures.

Two worlds live here, and keeping them apart is the whole point.

The default world is mocked. No Ollama, no embedding model, no vector store, no
network, no cost, and a deterministic answer so assertions can be exact. Every
health, validation, contract, and error-handling test runs here.

The integration world is opt-in behind the `integration` marker. It builds a
real index in a temporary directory and, for answer-quality tests, calls the
real local model.

The temporary directory is not a nicety. The reference implementation points
the service at the same directory the real ingestion script writes to, so
running the suite reads whatever the developer happened to ingest last and any
test that writes pollutes the real index. That makes results unreproducible on
a colleague's machine.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import settings

# Before `app.main` is imported, so the lifespan hook reads it as False. The
# default suite must not load torch, open Chroma, or reach Ollama, and a
# warmup thread would do all three behind the mocks' back.
settings.warmup_on_startup = False

from app.api.routes import get_rag_service  # noqa: E402
from app.main import app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = REPO_ROOT / "data" / "documents"
TEST_DATA_DIR = REPO_ROOT / "test_data"

MOCK_ANSWER = "Test answer from mock service."
MOCK_SOURCE_CONTENT = "Customers can return products within 30 days of delivery."


# --------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------


class MockRAGService:
    """Deterministic stand-in for the real service."""

    def ask(self, question: str) -> Dict[str, Any]:
        return {
            "answer": MOCK_ANSWER,
            "sources": [
                {
                    "content": MOCK_SOURCE_CONTENT,
                    "metadata": {"source": "faq.txt", "chunk_index": 0},
                }
            ],
            "insufficient_context": False,
        }


class FailingRAGService:
    """Raises an unexpected error, to prove nothing internal leaks out."""

    def ask(self, question: str) -> Dict[str, Any]:
        raise RuntimeError(
            "Simulated failure: secret_token=abc123 at /internal/path/rag.py:42"
        )


# --------------------------------------------------------------------------
# Mocked client fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def make_client() -> Iterator[Callable[[Callable[[], Any]], TestClient]]:
    """Build a TestClient backed by any service factory.

    Overrides are always cleared, so one test's double can never leak into the
    next test.
    """
    created: list[TestClient] = []

    def _factory(service_factory: Callable[[], Any]) -> TestClient:
        app.dependency_overrides[get_rag_service] = service_factory
        test_client = TestClient(app)
        created.append(test_client)
        return test_client

    yield _factory

    for test_client in created:
        test_client.close()
    app.dependency_overrides.clear()


@pytest.fixture
def client(make_client) -> TestClient:
    """The default client. Mocked, offline, free, deterministic."""
    return make_client(MockRAGService)


# --------------------------------------------------------------------------
# Integration fixtures (marker: integration)
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def seeded_chroma_dir() -> Iterator[str]:
    """Build a real vector index from the project documents in a temp dir.

    Session scoped because loading the embedding model and embedding the corpus
    is the slow part. This needs the embedding model but not Ollama, so
    retrieval-only tests can use it without a running language model.
    """
    from app.ingestion.ingest import ingest_documents

    temp_dir = tempfile.mkdtemp(prefix="rag_test_chroma_")
    try:
        ingest_documents(
            documents_dir=str(DOCUMENTS_DIR),
            chroma_dir=temp_dir,
            reset=True,
        )
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def integration_service(seeded_chroma_dir: str):
    """A real RAGService pointed at the temporary index."""
    from app.services.rag_service import RAGService

    return RAGService(chroma_dir=seeded_chroma_dir)


@pytest.fixture
def integration_client(integration_service, make_client) -> TestClient:
    """A client wired to the real service. Requires a running Ollama server."""
    return make_client(lambda: integration_service)


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Whether a local Ollama server is reachable with the configured model."""
    import httpx

    from app.config import settings

    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        response.raise_for_status()
    except Exception:
        return False

    names = [m.get("name", "") for m in response.json().get("models", [])]
    wanted = settings.llm_model
    return any(n == wanted or n.startswith(f"{wanted}:") for n in names)


@pytest.fixture
def require_ollama(ollama_available: bool) -> None:
    """Skip with a clear reason instead of failing when the model is absent."""
    if not ollama_available:
        from app.config import settings

        pytest.skip(
            f"Ollama is not reachable at {settings.ollama_base_url} with model "
            f"{settings.llm_model!r}. Start it and run: "
            f"ollama pull {settings.llm_model}"
        )
