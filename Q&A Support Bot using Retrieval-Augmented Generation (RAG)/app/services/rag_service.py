"""Retrieval-augmented generation service.

Two design choices here are deliberate departures from the reference guide,
because the guide's own test suite assumes behaviour its sample code does not
implement:

1. `chroma_dir` is a constructor argument. The reference version reads the
   global setting unconditionally, which makes every test read and write the
   real developer knowledge base. Tests then depend on whatever happened to be
   ingested last and are not reproducible on another machine.

2. Retrieval applies a relevance threshold. The reference version passes
   whatever the retriever returns straight to the model, so an out-of-scope
   question still gets a confident answer built from weakly related chunks.
   The prompt asks the model not to do that, but a prompt is not a guarantee.
   The threshold is enforced in code, before the model is ever called.
"""

from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.services.exceptions import (
    KnowledgeBaseUnavailableError,
    LLMUnavailableError,
)
from app.services.redaction import contains_secret, redact_response
from app.services.small_talk import (
    ANSWER_GROUNDED,
    ANSWER_OUT_OF_SCOPE,
    ANSWER_SMALL_TALK,
    small_talk_reply,
)
from app.services.vector_store import collection_is_empty, get_vector_store
from app.utils.logger import describe_question, get_logger

logger = get_logger(__name__)

# Returned when no retrieved chunk clears the relevance threshold. Kept as a
# module constant so tests assert against one source of truth instead of a
# copy-pasted string.
# The first sentence is fixed: the refusal has to be recognisable, and both the
# evaluators and `tests/test_retrieval_threshold.py` key off it. Only the
# escalation route is configurable, because that is the part that is wrong in a
# new deployment - "contact a support agent" is not where a banking customer with
# a suspected fraud case should be sent.
INSUFFICIENT_CONTEXT_ANSWER = (
    "I don't have enough information in the knowledge base to answer that. "
    f"{settings.escalation_hint}"
)

# The rule ordering here is deliberate and was arrived at by measurement, not
# taste. An earlier version led with "if the context does not contain the answer,
# reply exactly: ..." and said nothing about applying stated conditions. A small
# local model took that escape hatch constantly: asked "what is the return policy,
# I have used the product" it declined, even though the retrieved chunk scored
# 0.649 and said returns require the product to be unused. The answer was right
# there; the question simply was not phrased like the FAQ heading.
#
# So the refusal is now the narrow case rather than the first thing offered, and
# the model is told explicitly that applying a stated rule to the user's own
# situation is not the same as inventing policy. "Use only the context" is
# unchanged — reasoning from the context is permitted, adding to it is not.
#
# Only the role line is configurable. The rules below it describe how to handle
# retrieved context and are true whatever the context is about, so they stay in
# the code where they can be reviewed - exposing the whole prompt as a setting
# would let a deployment silently drop the measured parts.
SYSTEM_PROMPT = f"""You are {settings.assistant_role}.

Answer the user's question using ONLY the context below. Follow these rules:
- Never invent policy details, time periods, prices, or contact channels.
- Never promise an action or outcome the context does not state. Describe what
  the policy says, not what you expect will be done about it.
- Quote figures such as time limits exactly as they appear in the context.
- The context states rules and conditions. When the user describes their own
  situation, apply those rules to it and tell them what follows - including when
  what follows is that they do not qualify.
- Reply exactly "I do not have enough information to answer that." only when the
  context says nothing about the subject of the question. Do not use it for a
  question the context answers only partly, or answers unfavourably.
- If the context attaches a requirement, deadline or condition to what the user
  asked about, include it. Leaving it out makes the answer wrong in practice.
- Keep the answer short and direct.

Context:
{{context}}"""


class RAGService:
    """Retrieves grounded context and generates an answer from it."""

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        k: Optional[int] = None,
        relevance_threshold: Optional[float] = None,
        llm: Any = None,
        store: Any = None,
    ) -> None:
        self.chroma_dir = chroma_dir or settings.chroma_dir
        self.collection_name = collection_name or settings.collection_name
        self.k = k if k is not None else settings.retrieval_k
        self.relevance_threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else settings.relevance_threshold
        )

        # Both of these are built on first use, not in the constructor.
        # Constructing the service must stay cheap and must not require a
        # running Ollama server, otherwise fixtures and health checks break.
        #
        # Passing `llm` or `store` explicitly skips the lazy build entirely,
        # which lets a unit test exercise the threshold logic with a stub and
        # no model downloads at all.
        self._llm = llm
        self._store: Any = store

    # -- lazily built collaborators ----------------------------------------

    @property
    def store(self) -> Any:
        if self._store is None:
            try:
                self._store = get_vector_store(
                    chroma_dir=self.chroma_dir,
                    collection_name=self.collection_name,
                )
            except Exception as exc:
                raise KnowledgeBaseUnavailableError(
                    f"Could not open the vector store at {self.chroma_dir!r}"
                ) from exc
        return self._store

    @property
    def llm(self) -> Any:
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama

                self._llm = ChatOllama(
                    model=settings.llm_model,
                    base_url=settings.ollama_base_url,
                    temperature=settings.llm_temperature,
                    client_kwargs={"timeout": settings.llm_timeout_seconds},
                )
            except Exception as exc:
                raise LLMUnavailableError(
                    "Could not initialise the local Ollama client"
                ) from exc
        return self._llm

    # -- pipeline stages ---------------------------------------------------

    def retrieve(self, question: str) -> List[Tuple[Any, float]]:
        """Return (document, relevance) pairs that clear the threshold.

        An empty list means the question is out of scope, which is a valid
        outcome and not an error.
        """
        store = self.store

        if collection_is_empty(store):
            raise KnowledgeBaseUnavailableError(
                "The vector store is empty. Run the ingestion script first: "
                "python -m app.ingestion.ingest"
            )

        try:
            scored = store.similarity_search_with_relevance_scores(
                question, k=self.k
            )
        except Exception as exc:
            raise KnowledgeBaseUnavailableError(
                "Similarity search against the vector store failed"
            ) from exc

        kept = [
            (doc, score)
            for doc, score in scored
            if score >= self.relevance_threshold
        ]

        logger.info(
            "retrieval candidates=%d kept=%d threshold=%.2f best=%.3f",
            len(scored),
            len(kept),
            self.relevance_threshold,
            scored[0][1] if scored else 0.0,
        )
        return kept

    def generate(self, question: str, context: str) -> str:
        """Call the local model and return its answer text."""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", "{question}")]
        )
        messages = prompt.format_messages(context=context, question=question)

        try:
            response = self.llm.invoke(messages)
        except LLMUnavailableError:
            raise
        except Exception as exc:
            # Covers a stopped Ollama server, a model that was never pulled,
            # and a timeout. All of them are "the model is unavailable".
            raise LLMUnavailableError(
                f"The local model {settings.llm_model!r} did not respond"
            ) from exc

        return (getattr(response, "content", "") or "").strip()

    # -- public entry point ------------------------------------------------

    def ask(self, question: str) -> Dict[str, Any]:
        """Answer a question from the knowledge base."""
        logger.info("rag.ask start %s", describe_question(question))

        # Answered before the index is opened, so a greeting costs neither a
        # retrieval nor a generation - and still works with an empty index,
        # which is the state a first-time user is most likely to greet it in.
        social = small_talk_reply(question)
        if social is not None:
            logger.info("rag.ask small talk")
            return {
                "answer": social,
                "sources": [],
                "insufficient_context": False,
                "answer_type": ANSWER_SMALL_TALK,
            }

        matches = self.retrieve(question)

        if not matches:
            # Refuse to answer rather than letting the model improvise from
            # weakly related text. This is the behaviour the test suite checks
            # for out-of-scope questions.
            logger.info("rag.ask no chunk cleared the relevance threshold")
            return {
                "answer": INSUFFICIENT_CONTEXT_ANSWER,
                "sources": [],
                "insufficient_context": True,
                "answer_type": ANSWER_OUT_OF_SCOPE,
            }

        context = "\n\n".join(doc.page_content for doc, _ in matches)
        answer = self.generate(question, context)

        if not answer:
            raise LLMUnavailableError("The local model returned an empty answer")

        sources = [
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata or {}),
                "relevance_score": round(float(score), 4),
            }
            for doc, score in matches
        ]

        logger.info("rag.ask done sources=%d", len(sources))

        # Last thing before the result leaves the service. A credential in the
        # knowledge base is retrieved like any other relevant text and needs no
        # attack to surface, so the answer and the returned chunks are scrubbed
        # together. Only the fact of a removal is logged, never the value.
        if contains_secret(answer):
            logger.warning("rag.ask redacted a credential-shaped string from the answer")

        return redact_response(
            {
                "answer": answer,
                "sources": sources,
                "insufficient_context": False,
                "answer_type": ANSWER_GROUNDED,
            }
        )
