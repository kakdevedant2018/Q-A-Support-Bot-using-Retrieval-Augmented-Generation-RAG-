"""Typed service errors.

The routing layer maps each of these to a specific HTTP status. Without typed
errors, every failure collapses into one broad `except Exception` and the API
cannot tell "your knowledge base is missing" apart from "the model timed out".
"""


class RAGError(Exception):
    """Base class for every recoverable failure in the RAG pipeline."""


class KnowledgeBaseUnavailableError(RAGError):
    """The vector store is missing, empty, or unreadable.

    Maps to HTTP 503. Usually means the ingestion script has not been run.
    """


class LLMUnavailableError(RAGError):
    """The local model could not be reached or failed to respond.

    Maps to HTTP 502. Usually means the Ollama server is not running or the
    configured model has not been pulled.
    """


class InvalidAnswerError(RAGError):
    """The generated payload failed output validation.

    Maps to HTTP 500. The answer never reaches the user, because a malformed
    answer is a defect, not a response.
    """
