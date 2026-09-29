# Architecture — what every part is for

This document explains the *why* behind each file. `README.md` covers setup and
commands; this one covers structure and the reasoning behind it, so that changing
something is a decision rather than a guess.

Companion documents: [AI_TESTING_STRATEGY.md](AI_TESTING_STRATEGY.md) — how a
non-deterministic system is tested automatically — and [TESTING.md](TESTING.md), the
runbook of commands for each of those methods.

---

## 1. What this system is

A support question-answering API. A question arrives over HTTP, the relevant
passages are retrieved from a local document index, a local language model writes
an answer using only those passages, and the answer is returned with the sources
it was built from.

Three constraints shaped every decision below:

| Constraint | Consequence throughout the codebase |
|---|---|
| **Zero cost** | No paid API anywhere. Embeddings run in-process (`sentence-transformers`), generation runs on local Ollama, the vector store is a directory on disk. An unbounded test loop cannot generate a bill. |
| **No Docker** | Nothing is a container. Every dependency is a `pip install` or a single Ollama installer, and the vector store is a folder rather than a database server. |
| **Windows 11** | No shell scripts, no `os.fork`, no POSIX paths in code, no native build steps beyond published wheels. Paths are built with `pathlib`. |

The same three constraints are the reason the *evaluation* layer (§6) is built
from deterministic text analysis rather than from RAGAS or DeepEval: both default
to a paid OpenAI key, which would put a bill behind every test run.

---

## 2. The two pipelines

RAG is two separate pipelines that meet at the vector store. Keeping them
distinct matters, because they fail differently and are fixed differently.

**Ingestion — run manually, ahead of time.**

```
data/documents/*.txt|*.pdf
        │
        │  load_documents()          read each file, attach source metadata
        ▼
   LangChain Documents
        │
        │  split_documents()         one topic per chunk (blank-line blocks)
        ▼
   chunks + chunk_index
        │
        │  get_embeddings()          MiniLM, local CPU, normalised vectors
        ▼
   384-dim vectors
        │
        │  Chroma.add_documents()    stable ids -> re-running replaces, not duplicates
        ▼
   chroma_db/  (on disk, cosine metric)
```

**Query — run per request.**

```
POST /api/v1/ask  {"question": "..."}
        │
        │  1. middleware        assign X-Request-ID, start the latency timer
        │  2. AskRequest        types, 3..1000 chars, reject whitespace-only
        │  3. Depends()         resolve the RAGService (overridable in tests)
        ▼
   RAGService.ask()
        │
        │  4. retrieve()        embed the question, cosine search, k=3
        │  5. threshold         drop anything below RELEVANCE_THRESHOLD
        │                       ── no chunk survives? ──► honest refusal, 200,
        │                                                  model never called
        │  6. generate()        Ollama, temperature 0, context-only prompt
        ▼
   {answer, sources, insufficient_context}
        │
        │  7. validate_answer_payload()   structural gate before the user sees it
        │  8. AskResponse                 serialise, attach request_id
        ▼
   200  {answer, sources[], request_id, insufficient_context}
```

Step 5 is the most consequential line of code in the project. It is what turns
"please don't make things up" from a request in a prompt into a property enforced
in code — and it is also the cheapest security control in the system, because a
prompt-injection attempt matches no support document and therefore never reaches
the model at all.

---

## 3. Application layer — file by file

### `app/main.py` — process entrypoint
Creates the FastAPI app, installs logging, mounts the router under `/api/v1`.

Two things live here rather than in the router because they must apply to *every*
request, including ones that never reach a route:

- **Request-ID middleware.** Honours an inbound `X-Request-ID` or generates one,
  echoes it in the response header, and logs method, path, status and latency
  against it. This is what makes "it failed at 2pm" a searchable log line.
- **Catch-all exception handler.** The final net. Logs the real cause with a
  traceback internally and returns a fixed generic message, so an unforeseen
  failure cannot leak a file path, a config value, or a stack trace.

`GET /health` is here too, and deliberately touches neither the vector store nor
the model: it answers "is this process alive", which is what a load balancer
needs. Dependency problems surface as a 502/503 from `/ask`, where they are
actionable.

### `app/api/routes.py` — HTTP boundary
Owns `POST /api/v1/ask` and one job: translate between HTTP and the service.

- The service is injected via `Depends(get_rag_service)`, never imported as a
  module-level global. That single choice is what lets the test suite substitute a
  mock and run the entire validation and error-handling suite with no Ollama, no
  embedding model, and no cost.
- `_shared_rag_service()` is `lru_cache`d so the embedding model loads once per
  process, not once per request.
- Each typed service error maps to its own status code (table in §5). The
  `detail` strings are module constants (`MSG_INTERNAL`, `MSG_LLM_DOWN`,
  `MSG_NO_KNOWLEDGE_BASE`) so tests assert against one source of truth, and none
  of them contains a path, a model name, or a config value.

### `app/schemas.py` — the contract
Pydantic models for request and response. Validation is in two layers because
they answer different questions:

- **Layer 1, field constraints.** `min_length=3`, `max_length=1000`. The cap is a
  cost and latency control as much as a correctness one: every token in a question
  is local compute.
- **Layer 2, business rule.** `question_must_not_be_blank` rejects `"   "`, which
  satisfies `min_length=3` while carrying no question. It also strips, so the text
  reaching the retriever is normalised.

`AskResponse.insufficient_context` is the structural signal that the bot declined
to answer. Exposing it as a boolean rather than expecting callers to pattern-match
on the answer text is what makes the refusal path assertable — see how
`evaluation/safety_evaluator.py` uses it.

### `app/services/rag_service.py` — the pipeline
The core. `retrieve()` / `generate()` / `ask()`.

- **`chroma_dir` is a constructor argument.** The reference design reads the
  global setting unconditionally, so every test reads and writes the developer's
  real index; results then depend on whatever was ingested last and are not
  reproducible on another machine. Here a test can point at a temp directory.
- **`llm` and `store` are lazy properties** and can be injected. Constructing the
  service must stay cheap and must not require a running Ollama — otherwise
  fixtures and health checks break. Injection lets the threshold logic be unit
  tested with stubs and zero downloads.
- **The relevance threshold is applied in `retrieve()`**, before generation.
- **`INSUFFICIENT_CONTEXT_ANSWER` is a module constant**, so tests assert one
  source of truth instead of a copy-pasted sentence.
- **`SYSTEM_PROMPT`** instructs context-only answering, forbids invented figures,
  and requires time limits be quoted exactly. Treated as a best effort, not a
  guarantee: the threshold, the output validator, and the evaluation suite are the
  enforcement.

### `app/services/vector_store.py` — one way to open the index
`get_vector_store()` is the single place that decides directory, collection name,
and distance metric. A mismatch in any of the three between ingestion and query
produces an empty result set with no error at all, which is a brutal bug to chase;
funnelling both paths through one function makes it impossible.

`collection_is_empty()` distinguishes "nothing was ingested" (a 503, run the
ingestion script) from "nothing matched" (a normal 200 refusal).

### `app/services/embeddings.py` — local vectors
`get_embeddings()` returns an `lru_cache`d MiniLM model.

- `normalize_embeddings=True` plus `hnsw:space: cosine` is what makes the
  relevance score land in `[-1, 1]`. With the default L2 metric the numbers are
  unbounded and a fixed threshold is not something you can reason about.
  The floor is `-1`, not `0`, and it is reached: an unrelated question has
  measured `-0.037` here. LangChain warns "Relevance scores must be between 0
  and 1" when that happens, which is the warning being wrong, not the score.
- Heavy imports sit inside the function so `import app.main` stays cheap and the
  mocked suite runs without torch installed.
- `_import_embeddings_class()` tries `langchain_huggingface` first and falls back
  to the deprecated `langchain_community` path, because LangChain moved this class
  and the two package layouts are both still in the wild.

### `app/services/response_validator.py` — the output gate
The last check before a payload reaches the user: types, non-emptiness, a length
cap, and one semantic rule — **an answer with no sources is only legitimate when
`insufficient_context` is set**. Otherwise the model produced prose attributable
to nothing, which is precisely the failure RAG exists to prevent.

It validates *structure*, never truthfulness. Truthfulness is measured offline
against a labelled dataset (§6); a per-request fact check would need a second
model call on every question.

### `app/services/exceptions.py` — typed failures
`KnowledgeBaseUnavailableError`, `LLMUnavailableError`, `InvalidAnswerError`.
Without them every failure collapses into one `except Exception` and the API
cannot tell "your index is missing" from "the model timed out" — so the caller
cannot tell whether retrying will help.

### `app/config.py` — settings
`pydantic-settings`, `.env` optional, every value defaulted so the app imports and
starts with no configuration at all. Full table in §7.

### `app/utils/logger.py` — observability with a privacy rule
One stdout handler, chatty libraries turned down, and `describe_question()`, which
returns `question_chars=42` and **never the question text**. A support question
routinely contains an order number, an email address, or a customer name; the
length is enough to debug a truncation or validation bug without turning the log
file into a store of personal data.

### `app/ingestion/ingest.py` — build the index
CLI: `python -m app.ingestion.ingest [--reset]`.

The notable property is **idempotence**. Each chunk id is derived from its source
file and its index within that file, so re-running replaces chunks in place
instead of appending. The naive version grows the collection on every run, which
skews retrieval toward whatever was ingested most often. Stale chunks from an
edited file are deleted explicitly. `--reset` drops the collection for a clean
rebuild.

---

## 4. Directory map

```
rag_support_bot/
├── app/                      the product
│   ├── main.py               FastAPI app, middleware, /health, error net
│   ├── config.py             settings + defaults
│   ├── schemas.py            request/response contract, input validation
│   ├── api/routes.py         POST /api/v1/ask, error -> status mapping
│   ├── services/
│   │   ├── rag_service.py    retrieve -> threshold -> generate
│   │   ├── vector_store.py   the one way to open Chroma
│   │   ├── embeddings.py     cached local MiniLM
│   │   ├── response_validator.py   structural gate on the way out
│   │   └── exceptions.py     typed failures
│   ├── ingestion/ingest.py   idempotent document -> chunk -> vector pipeline
│   └── utils/logger.py       logging, and the no-question-text rule
│
├── evaluation/               how good is it? (a measuring instrument)
│   ├── normalize.py          wording-independent text comparison
│   ├── fact_evaluator.py     did the answer state the expected facts
│   ├── groundedness_evaluator.py   is every figure/claim supported by context
│   ├── semantic_evaluator.py meaning-level similarity + negation guard
│   ├── safety_evaluator.py   injection, leakage, invented policy
│   ├── retrieval_metrics.py  Recall@k, Precision@k, MRR
│   ├── llm_judge.py          model-graded scoring (advisory only)
│   ├── results.py            result dataclasses; defines what "passed" means
│   ├── report.py             scorecard -> console / JSON / HTML
│   ├── run_eval.py           CLI harness
│   ├── calibrate.py          measure thresholds instead of guessing them
│   └── datasets/*.json       the labelled test data
│
├── tests/                    is it broken? (a gate)
│   ├── conftest.py           mocked and integration fixtures
│   ├── test_health.py        [api]
│   ├── test_validation.py    [api]
│   ├── test_ask_api.py       [api]
│   ├── test_error_handling.py[api]
│   ├── test_response_validator.py   output gate, unit
│   ├── test_retrieval_threshold.py  [rag] threshold logic with stubs
│   ├── test_evaluation_framework.py the evaluators' own unit tests
│   ├── test_rag_quality.py   [integration, rag] real index, round trip
│   ├── test_ai_quality.py    [integration, ai] dataset-driven scoring
│   └── test_security.py      [security] injection, leakage, refusal
│
├── data/documents/           the knowledge base (faq.txt, support_policy.txt)
├── test_data/questions.json  the original small labelled set
├── postman/                  importable collection + environment
├── docs/                     this file, the testing strategy, the test runbook
├── .github/workflows/        CI: mocked suite + byte-compile
├── pytest.ini                markers and the "mocked by default" policy
├── requirements.txt          full stack
└── requirements-test.txt     light set: enough for the mocked suite
```

---

## 5. Error model

Every failure has one status code, one safe message, and one place to look.

| Raised | Status | Cause | Fix |
|---|---|---|---|
| `AskRequest` validation | 422 | Empty, whitespace-only, too short, too long, wrong type, malformed JSON | Fix the request |
| `KnowledgeBaseUnavailableError` | 503 | `chroma_db/` missing, empty, or unreadable | `python -m app.ingestion.ingest` |
| `LLMUnavailableError` | 502 | Ollama not running, model not pulled, timeout, empty completion | Start Ollama; `ollama pull llama3.1` |
| `InvalidAnswerError` | 500 | Output failed the structural gate | A defect — read the log for the failing rule |
| anything else | 500 | Unforeseen | Log has the traceback and the request id |
| — | 200 + `insufficient_context: true` | No chunk cleared the threshold | Not an error. The correct answer to an out-of-scope question |

Two invariants hold across all of them: the response body never contains a path,
a token, a host/port, or a stack trace; and the log line always carries the
request id so a user report maps to a specific failure.

---

## 6. Evaluation layer — and why it is not in `tests/`

`tests/` and `evaluation/` answer different questions, and conflating them
produces a suite that is bad at both.

| | `tests/` | `evaluation/` |
|---|---|---|
| Question | Is it broken? | How good is it, compared with last week? |
| Output | pass / fail | a scorecard you keep and diff |
| Runs | on every change | deliberately, before a release |
| Right shape | assertions | metrics + artifacts |

So the evaluators are a library, `run_eval.py` is a CLI that produces a
scorecard, and `tests/test_ai_quality.py` imports the same library to turn
selected measurements into gates. One implementation, two consumers — an
evaluator bug cannot make the tests and the report disagree.

The dependency direction is strict: **`evaluation/` may import `app/`; `app/`
never imports `evaluation/`.** Measuring instruments do not ship inside the
product.

Internally the evaluators are layered cheapest-first, which is also
strongest-first:

```
normalize.py          no model, no network, no cost
   ├─ fact_evaluator.py          "did it say the thing"
   ├─ groundedness_evaluator.py  "is every figure in the context"   <- strongest free signal
   └─ safety_evaluator.py        "did it leak or comply"
semantic_evaluator.py  local embeddings (already loaded by the app)
llm_judge.py           a local model grading output — advisory, never a gate
```

`results.py` is where the policy lives: `CaseResult.passed` is computed from
facts, groundedness, semantics and safety, and **deliberately excludes the
judge**. A small local model grading its own family's output is not a defensible
build gate. When an evaluator cannot run, it reports "not measured" rather than
defaulting to a pass.

Full detail, including the seven testing methods and their limitations, is in
[AI_TESTING_STRATEGY.md](AI_TESTING_STRATEGY.md).

---

## 7. Configuration

All optional. Copy `env.example` to `.env` to override.

| Setting | Default | What it controls | When to change it |
|---|---|---|---|
| `LLM_MODEL` | `llama3.1` | Ollama model for generation | `llama3.2:3b` on 8 GB RAM |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama listens | Remote or non-default port |
| `LLM_TEMPERATURE` | `0.0` | Sampling randomness | Keep at 0 for support answers |
| `LLM_TIMEOUT_SECONDS` | `120` | Generation timeout | Raise on a slow CPU |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Vectoriser, ~90 MB | Changing it **requires re-ingestion** |
| `CHROMA_DIR` | `chroma_db` | Index location | Multiple knowledge bases |
| `COLLECTION_NAME` | `support_documents` | Chroma collection | Rarely |
| `RETRIEVAL_K` | `5` | Candidate chunks per query, before the threshold | Raising it is cheap: the threshold runs after, so a larger k can only admit chunks already judged relevant |
| `RELEVANCE_THRESHOLD` | `0.35` | Out-of-scope cutoff, cosine in `[-1, 1]` | **A starting point, not a measured value.** Measure it: `python -m evaluation.calibrate --retrieval` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `100` | Bounds the oversize fallback only; splitting is by topic | Re-ingest after changing |
| `LOG_LEVEL` | `INFO` | Verbosity | `DEBUG` when diagnosing retrieval |

Two of these are load-bearing and worth stating plainly. `RELEVANCE_THRESHOLD`
decides the entire out-of-scope guarantee: too low and the bot answers questions
it should decline, too high and it declines questions it could answer. And
`EMBEDDING_MODEL` is not hot-swappable — vectors from a different model are not
comparable with what is already in the index.

---

## 8. Test architecture

| Layer | Tested by | Needs |
|---|---|---|
| Middleware, `/health` | `test_health.py` | nothing |
| Input validation | `test_validation.py` | nothing |
| Response contract | `test_ask_api.py` | nothing |
| Error mapping and non-leakage | `test_error_handling.py` | nothing |
| Output gate | `test_response_validator.py` | nothing |
| Threshold logic | `test_retrieval_threshold.py` | nothing (stub store and LLM) |
| The evaluators themselves | `test_evaluation_framework.py` | nothing |
| Real retrieval and round trip | `test_rag_quality.py` | embeddings; Ollama for the last few |
| Scored answer quality | `test_ai_quality.py` | embeddings + Ollama |
| Injection, leakage, refusal | `test_security.py` | mostly nothing; some need both |

The default run — `pytest` — is fully mocked: **238 tests, under a second, no
network, no downloads, no cost.** That is possible because every heavy import in
`app/` is deferred into a function, so `requirements-test.txt` needs no torch and
no chromadb.

`conftest.py` provides the seam. `make_client()` installs a
`dependency_overrides` entry and always clears it, so one test's double cannot
leak into the next. `seeded_chroma_dir` is session-scoped and builds a real index
in a *temporary* directory — never the developer's real one, which would make
results depend on whatever was ingested last. `require_ollama` skips with an
actionable message instead of failing when the model is absent.

---

## 9. Decisions worth knowing about

**Where the reference design was departed from, and why.** The threshold is
enforced in code rather than requested in the prompt, because a prompt is a
request and cannot be asserted. `chroma_dir` is injectable, because otherwise
tests are not reproducible. Ingestion is idempotent, because appending duplicates
on every run quietly corrupts retrieval.

**Lazy imports everywhere.** Slightly unusual to read, and it buys three things:
fast app import, a mocked suite that runs without torch, and a broken native
wheel that cannot stop the API from importing.

**Cosine + normalised vectors.** Not a detail. It is the precondition for a
bounded relevance score in `[-1, 1]`, without which no fixed threshold means
anything.

**One topic per chunk.** Also not a detail, and the single highest-impact choice
measured here. A chunk is embedded as one vector, so a chunk covering three
subjects is stored as the average of three subjects and matches none of them
well. Splitting on the blank-line boundaries the documents already use, instead
of on a character budget, moved Recall@k from 88.2% to 100%, MRR from 0.824 to
1.000, and correctness from 79.4% to 100%.

**Refusal is a first-class outcome.** `insufficient_context` is a boolean in the
response, not a sentence to grep for. A support bot that declines clearly is
correct; one that improvises confidently is a liability.

**No judge in the pass gate.** Measured and reported, never decisive.

### Deliberately not built

Auth, rate limiting, conversation memory, streaming, reranking, hybrid search, a
UI, containers, a hosted deployment. Each would be defensible; none is needed to
demonstrate that this pipeline is correct, and every one of them would add a
moving part that the test suite would then have to cover honestly.
