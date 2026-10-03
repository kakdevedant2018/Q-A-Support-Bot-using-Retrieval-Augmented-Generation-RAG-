"""Application configuration.

Every setting has a usable default, so the app imports and starts with no
.env file present. That matters for two reasons:

1. CI can import the app and run the mocked test suite with no secrets.
2. There is no paid API key in this stack, so there is nothing that *must*
   be supplied by the operator.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Local LLM served by Ollama ---------------------------------------
    # This is the line that replaces the paid OpenAI dependency.
    llm_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.0
    llm_timeout_seconds: int = 120

    # --- Local LLM performance --------------------------------------------
    # These three are the difference between a 2-second answer and a
    # 60-second one on a machine without a usable GPU. All were left at
    # Ollama's defaults originally, and all three defaults are wrong for an
    # interactive Q&A bot.
    #
    # How long Ollama keeps the weights resident after a request. The default
    # is 5 minutes, so a user who reads an answer, thinks, and asks a second
    # question pays the whole model load again from disk. There is no reason
    # to evict a model this process is going to ask for again; "-1" pins it
    # for the lifetime of the Ollama server.
    llm_keep_alive: str = "30m"

    # Context window. llama3.1 advertises 128k and Ollama was allocating a
    # 32768-token KV cache here, which costs gigabytes of RAM and, on CPU,
    # slows prompt processing for every request. This bot sends a system
    # prompt plus `retrieval_k` chunks of `chunk_size` characters - well
    # under 2k tokens. 4096 is generous for that and much cheaper.
    llm_num_ctx: int = 4096

    # Hard cap on answer length. Unbounded generation is the single largest
    # latency term on CPU, and the system prompt already asks for a short
    # answer - this makes it enforceable rather than a request.
    llm_num_predict: int = 384

    # --- Startup ----------------------------------------------------------
    # Load the embedding model and pin the LLM in a background thread as soon
    # as the process starts, instead of on the first user question. The work
    # is identical; this only decides who waits for it. Off in tests, which
    # must never touch a real model.
    warmup_on_startup: bool = True

    # --- Local embedding model -------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector store -----------------------------------------------------
    chroma_dir: str = "chroma_db"
    collection_name: str = "support_documents"

    # --- Retrieval --------------------------------------------------------
    # Raising this from 3 is close to free, because the threshold below is
    # applied after retrieval: a larger k can only admit chunks that were
    # already judged relevant enough to keep. At 3 the answer to "I was charged
    # twice for my order" was invisible - it sat fourth at 0.352, above the
    # threshold and outside the window.
    retrieval_k: int = 5

    # Cosine relevance score in [-1, 1]; higher means more similar. Chunks
    # below this are dropped, which is what makes an out-of-scope question
    # return an honest "not enough information" answer instead of a
    # confident wrong one.
    #
    # The theoretical floor is -1, not 0, and it is reached in practice: an
    # unrelated question has been measured at -0.037 here. LangChain warns
    # "Relevance scores must be between 0 and 1" when that happens, which is
    # the warning being wrong, not the score.
    #
    # This default is a STARTING POINT, not a measured value. Tune it against
    # your own documents before trusting it.
    relevance_threshold: float = 0.35

    # --- Ingestion --------------------------------------------------------
    documents_dir: str = "data/documents"

    # These bound the size-based fallback only. Chunks are split on topic
    # boundaries first - see `app.ingestion.ingest.split_text_into_topics`.
    chunk_size: int = 500
    chunk_overlap: int = 100

    # --- Domain ------------------------------------------------------------
    # The only three places the shipped answer text mentions what this bot is
    # for. They are settings rather than literals so that pointing the system at
    # a different corpus - banking, HR, insurance, internal IT - is a change to
    # `.env` and `data/documents/`, not a change to `rag_service.py`.
    #
    # Nothing else in `app/` is domain-aware: ingestion, retrieval, the
    # threshold, redaction, and validation all operate on text without caring
    # what the text is about. The work of moving domains is almost entirely in
    # the corpus and in `evaluation/datasets/`, not in the application.
    #
    # `assistant_role` is interpolated into the system prompt. Keep it a plain
    # role description. The rules that follow it were arrived at by measurement
    # and are domain-independent, so they are deliberately not configurable -
    # a free-text prompt override is how a careful prompt quietly becomes a
    # careless one.
    assistant_role: str = "a helpful customer support assistant"

    # Where a user is sent when the knowledge base cannot answer. This is the
    # sentence most likely to be wrong in a new deployment: a banking bot that
    # tells people to "contact a support agent" when the real route is a branch
    # or a fraud line is giving bad advice politely.
    escalation_hint: str = "Please contact a support agent for help with this question."

    # Used by the greeting reply to say what the bot can be asked about.
    knowledge_domain: str = "the support documentation"

    # --- Observability ----------------------------------------------------
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
