"""Request and response models.

Layer 1 of validation lives here: field types, required fields, and length
bounds. Layer 2 (business rules such as whitespace-only input) lives in the
field validator below.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's support question.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject whitespace-only input.

        min_length alone accepts "   ", which is three characters but carries
        no question at all. Stripping here also normalises the text that
        reaches the retriever.
        """
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty or whitespace only.")
        return value


class Source(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Populated by the real retriever; absent from mocked responses.
    relevance_score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]

    # Echoed back so a user-reported problem can be found in the logs.
    request_id: Optional[str] = None

    # True when the relevance threshold rejected every candidate chunk, so
    # the bot declined to answer rather than guessing. It means exactly that,
    # and nothing else - see `answer_type` below.
    insufficient_context: bool = False

    # Which of the three outcomes produced this answer:
    #
    #   grounded      built from retrieved chunks
    #   out_of_scope  nothing cleared the relevance threshold; an honest refusal
    #   small_talk    a greeting or thanks, answered without touching the index
    #
    # A greeting is not a failed lookup, and collapsing the two would force a
    # client to render "not in knowledge base" over "hello". Kept separate from
    # `insufficient_context` so that flag keeps its single meaning and existing
    # clients are unaffected.
    answer_type: str = "grounded"


class HealthResponse(BaseModel):
    status: str
    version: str


class ErrorResponse(BaseModel):
    detail: str
