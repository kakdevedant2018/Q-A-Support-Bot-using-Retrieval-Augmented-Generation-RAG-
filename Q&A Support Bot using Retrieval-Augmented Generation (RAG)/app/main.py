"""FastAPI entrypoint.

Run it:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000 for the UI, or /docs for the API reference.
"""

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app import __version__
from app.api.routes import router
from app.config import settings
from app.schemas import HealthResponse
from app.utils.logger import configure_logging, get_logger

configure_logging(settings.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="RAG Support Bot API",
    version=__version__,
    description=(
        "A support question-answering API grounded in a local document "
        "knowledge base. Embeddings and the language model both run locally, "
        "so the service has no external API cost."
    ),
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Tag every request so a log line can be tied to a user report."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s status=%d latency_ms=%.1f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Final safety net.

    Logs the real cause internally and returns a generic message, so an
    unexpected failure cannot leak a file path, a config value, or a stack
    trace to the caller.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("unhandled error request_id=%s path=%s", request_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred while processing the question."},
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    """Liveness check.

    Deliberately does not touch the vector store or the model. It answers "is
    this process up", which is what a load balancer needs. Dependency problems
    surface as a 503 from /ask, where they are actionable.
    """
    return HealthResponse(status="healthy", version=__version__)


app.include_router(router, prefix="/api/v1", tags=["ask"])


# --- browser UI ------------------------------------------------------------
# One self-contained HTML file, served by this same process. It calls the same
# /api/v1/ask endpoint as curl and the CLI, so the browser gets no privileged
# path into the system and the UI cannot drift from the documented contract.

INDEX_FILE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    """Serve the UI, or say plainly that it is missing.

    Guarded rather than mounted at import time: a missing static file should
    cost you this one route, not the ability to start the API at all.
    """
    if not INDEX_FILE.is_file():
        return JSONResponse(
            status_code=404,
            content={"detail": "The UI is not installed. The API is at /docs."},
        )
    return FileResponse(INDEX_FILE, media_type="text/html")
