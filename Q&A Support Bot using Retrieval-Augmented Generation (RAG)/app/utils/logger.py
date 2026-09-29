"""Logging setup.

What we log: request id, endpoint, status, latency, how many chunks were
retrieved, and the error type when something fails.

What we deliberately do NOT log: the full user question. A support question
can contain an order number, an email address, or a customer name. We record
its length instead, which is enough to debug a validation or truncation bug
without turning the log file into a store of personal data.
"""

import logging
import sys
from typing import Optional

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install a single stdout handler. Safe to call more than once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # The embedding and vector store libraries are extremely chatty at INFO.
    for noisy in ("chromadb", "sentence_transformers", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def describe_question(question: Optional[str]) -> str:
    """Return a log-safe description of a user question.

    Length only. Never the content.
    """
    if question is None:
        return "question=<none>"
    return f"question_chars={len(question)}"
