"""API endpoints.

The service is supplied through `Depends(get_rag_service)` rather than a module
level global. That is what lets the test suite swap in a mock and run the whole
validation and error-handling suite with no Ollama server, no vector store, and
no cost.
"""

import time
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas import AskRequest, AskResponse
from app.services.exceptions import (
    InvalidAnswerError,
    KnowledgeBaseUnavailableError,
    LLMUnavailableError,
)
from app.services.rag_service import RAGService
from app.services.response_validator import validate_answer_payload
from app.utils.logger import describe_question, get_logger

router = APIRouter()
logger = get_logger(__name__)

# Safe messages. None of these leak a path, a config value, a model name, or a
# stack trace. The detail needed for debugging goes to the log with the request
# id, not to the caller.
MSG_INTERNAL = "An internal error occurred while processing the question."
MSG_NO_KNOWLEDGE_BASE = (
    "The knowledge base is currently unavailable. Please try again later."
)
MSG_LLM_DOWN = (
    "The answering service is temporarily unavailable. Please try again later."
)


@lru_cache(maxsize=1)
def _shared_rag_service() -> RAGService:
    """Build the service once per process.

    Loading the embedding model takes seconds, so a fresh instance per request
    would make every call slow. The constructor is lazy, so this does not touch
    the vector store or Ollama until the first real question arrives.
    """
    return RAGService()


def get_rag_service() -> RAGService:
    """Dependency provider. Override this in tests."""
    return _shared_rag_service()


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a support question",
    responses={
        422: {"description": "Request failed schema or business validation"},
        500: {"description": "Unexpected internal error"},
        502: {"description": "The local model is unreachable"},
        503: {"description": "The knowledge base is missing or empty"},
    },
)
def ask_question(
    payload: AskRequest,
    request: Request,
    rag_service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    started = time.perf_counter()

    try:
        result = rag_service.ask(payload.question)
        validate_answer_payload(result)

    except KnowledgeBaseUnavailableError:
        logger.error(
            "ask failed request_id=%s reason=knowledge_base_unavailable %s",
            request_id,
            describe_question(payload.question),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MSG_NO_KNOWLEDGE_BASE,
        )

    except LLMUnavailableError:
        logger.error(
            "ask failed request_id=%s reason=llm_unavailable %s",
            request_id,
            describe_question(payload.question),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=MSG_LLM_DOWN,
        )

    except InvalidAnswerError:
        logger.error(
            "ask failed request_id=%s reason=invalid_answer %s",
            request_id,
            describe_question(payload.question),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=MSG_INTERNAL,
        )

    except Exception:
        # Anything unforeseen. Logged in full internally, reported generically.
        logger.exception(
            "ask failed request_id=%s reason=unexpected %s",
            request_id,
            describe_question(payload.question),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=MSG_INTERNAL,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "ask ok request_id=%s latency_ms=%.1f sources=%d insufficient=%s type=%s",
        request_id,
        elapsed_ms,
        len(result.get("sources", [])),
        bool(result.get("insufficient_context")),
        result.get("answer_type", "grounded"),
    )

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        request_id=request_id,
        insufficient_context=bool(result.get("insufficient_context", False)),
        answer_type=str(result.get("answer_type", "grounded")),
    )
