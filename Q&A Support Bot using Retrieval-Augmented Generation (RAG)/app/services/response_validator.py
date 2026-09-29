"""Output validation.

A 200 with valid JSON does not mean the answer is usable. This module is the
last gate before a payload reaches the user. It checks structure and
non-emptiness, which are cheap and deterministic.

It does NOT check truthfulness. Groundedness is measured by the evaluation
tests against a labelled dataset, not asserted at request time.
"""

from typing import Any, Dict

from app.services.exceptions import InvalidAnswerError
from app.services.small_talk import ANSWER_SMALL_TALK
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_ANSWER_CHARS = 8000


def validate_answer_payload(payload: Any) -> Dict[str, Any]:
    """Validate a service payload, returning it unchanged when it is sound.

    Raises InvalidAnswerError on anything malformed, which the routing layer
    turns into a generic 500. A broken answer is a defect to investigate, not
    something to hand to a customer.
    """
    if not isinstance(payload, dict):
        raise InvalidAnswerError(
            f"Service returned {type(payload).__name__}, expected a dict"
        )

    answer = payload.get("answer")
    if not isinstance(answer, str):
        raise InvalidAnswerError(
            f"'answer' must be a string, got {type(answer).__name__}"
        )
    if not answer.strip():
        raise InvalidAnswerError("'answer' is empty")
    if len(answer) > MAX_ANSWER_CHARS:
        raise InvalidAnswerError(
            f"'answer' is {len(answer)} chars, over the {MAX_ANSWER_CHARS} limit"
        )

    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise InvalidAnswerError(
            f"'sources' must be a list, got {type(sources).__name__}"
        )

    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise InvalidAnswerError(
                f"sources[{index}] must be an object, got "
                f"{type(source).__name__}"
            )
        if not isinstance(source.get("content"), str):
            raise InvalidAnswerError(f"sources[{index}].content must be a string")
        metadata = source.get("metadata", {})
        if not isinstance(metadata, dict):
            raise InvalidAnswerError(f"sources[{index}].metadata must be an object")

    # An answer with no sources is only legitimate when the service explicitly
    # declined to answer. Otherwise it means the model produced prose that is
    # attributable to nothing, which is exactly the failure mode RAG exists to
    # prevent.
    #
    # Small talk is exempt, and narrowly so. The rule guards against *generated*
    # prose with nothing behind it; a greeting reply is a fixed string that never
    # reached the model and asserts nothing about the documents. The exemption is
    # keyed on the service's own outcome label, which no caller can set.
    unsourced_is_allowed = (
        payload.get("insufficient_context")
        or payload.get("answer_type") == ANSWER_SMALL_TALK
    )
    if not sources and not unsourced_is_allowed:
        raise InvalidAnswerError(
            "Answer cites no sources but was not flagged as insufficient context"
        )

    return payload
