"""Tests for the browser UI route.

The UI is a static file, so there is not much logic to test. What is worth
asserting is that it stays a *client* of the documented API: it must not ship
credentials, must not talk to anything other than /api/v1/ask, and must not
become a second, undocumented way into the system.

These run in the mocked suite. Serving a file needs no model and no index.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.api

INDEX_FILE = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def test_the_ui_is_served_at_the_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Support</title>" in response.text


def test_the_ui_is_excluded_from_the_api_schema(client):
    """The UI is not part of the API contract, so it should not appear in it.

    Someone reading /openapi.json is looking for endpoints to call, and an
    HTML page listed among them is noise.
    """
    schema = client.get("/openapi.json").json()
    assert "/" not in schema["paths"]
    assert set(schema["paths"]) == {"/health", "/api/v1/ask"}


def test_the_ui_calls_only_the_documented_endpoint():
    """No privileged or undocumented path into the system from the browser."""
    markup = INDEX_FILE.read_text(encoding="utf-8")

    called = set(re.findall(r'fetch\(\s*["\']([^"\']+)["\']', markup))
    called |= set(re.findall(r'^\s*const ASK_URL = "([^"]+)"', markup, re.M))

    assert called <= {"/api/v1/ask", "/health"}, (
        f"the UI fetches something unexpected: {called}"
    )


def test_the_ui_has_no_external_dependencies():
    """It must work offline, with no CDN and no build step.

    An external script or font would also be a third party watching support
    questions being typed.
    """
    markup = INDEX_FILE.read_text(encoding="utf-8")

    assert "http://" not in markup.replace("http://www.w3.org", "")
    assert "https://" not in markup
    assert "<script src" not in markup


def test_the_ui_ships_no_credentials():
    """A static file served to anyone is the worst place to keep a secret."""
    markup = INDEX_FILE.read_text(encoding="utf-8").lower()

    for marker in ("api_key", "apikey", "secret", "password", "bearer ", "token="):
        assert marker not in markup, f"the UI contains {marker!r}"


def test_the_ui_mirrors_the_request_contract():
    """The client-side limits must match AskRequest, or the UI lies to the user.

    A browser that allows 5000 characters when the API caps at 1000 produces a
    422 the user cannot explain.
    """
    from app.schemas import AskRequest

    markup = INDEX_FILE.read_text(encoding="utf-8")
    field = AskRequest.model_fields["question"]

    constraints = {
        getattr(meta, "max_length", None) for meta in field.metadata
    } | {
        getattr(meta, "min_length", None) for meta in field.metadata
    }

    assert f"const MAX_CHARS = {max(c for c in constraints if c)};" in markup
    assert f"const MIN_CHARS = {min(c for c in constraints if c)};" in markup
    assert 'maxlength="1000"' in markup


def test_a_missing_ui_file_does_not_break_the_api(monkeypatch, client):
    """The API must survive the UI being absent.

    Mounting static files at import time would turn a deleted file into a
    process that will not start at all.
    """
    from app import main

    monkeypatch.setattr(main, "INDEX_FILE", Path("does/not/exist.html"))

    assert client.get("/").status_code == 404
    assert client.get("/health").status_code == 200
