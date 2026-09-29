# Q&A Support Bot using Retrieval-Augmented Generation

A support question-answering API that answers from your own documents instead of
from a language model's general training. Questions are embedded, matched
against a local vector index, and answered only from the retrieved text.

Everything runs locally and nothing here costs money. There is no OpenAI key, no
hosted vector database, and no Docker requirement.

| Layer | Choice | Cost |
|---|---|---|
| API | FastAPI with Pydantic validation | free |
| Embeddings | sentence-transformers all-MiniLM-L6-v2, on your CPU | free |
| Vector store | Chroma, persisted to a local folder | free |
| Language model | Ollama, running a model on your machine | free |
| Tests | pytest, Postman, Newman | free |
| Answer evaluation | deterministic checks + a local judge model | free |

## Documentation

| Document | What it answers |
|---|---|
| This file | How do I install and run it |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | What is every file for, and why is it built this way |
| [docs/AI_TESTING_STRATEGY.md](docs/AI_TESTING_STRATEGY.md) | How do you test a model that never returns the same string twice, automatically and for free |
| [docs/TESTING.md](docs/TESTING.md) | The runbook: exact commands for each of the seven methods, and how to read a failure |

---

## How it works

Two pipelines, deliberately separate.

**Ingestion, run once and again when documents change.** Documents are loaded,
split into one-topic chunks, embedded, and written to a Chroma directory.

**Query, run on every request.** The question is validated, embedded, and matched
against the stored index. Chunks that clear a relevance threshold become the
context for the model. The generated answer is validated before it is returned.

The API never re-embeds the knowledge base on startup or per request, so request
latency stays low and a restart is cheap.

```
POST /api/v1/ask
  -> schema validation      required field, type, length 3 to 1000
  -> business validation    reject whitespace-only input
  -> embed the question
  -> similarity search      top k, cosine relevance in [-1, 1]
  -> relevance threshold    drop weak chunks; if none survive, decline to answer
  -> build the prompt       only surviving chunks
  -> local model            Ollama
  -> output validation      non-empty answer, well-formed sources, must cite something
  -> JSON response          answer, sources, request_id, insufficient_context
```

---

## Setup on Windows 11

### 1. Install Python 3.12

Get it from python.org and tick **Add python.exe to PATH**.

Use 3.12 specifically. Chroma, sentence-transformers, and their native
dependencies lag behind the newest Python releases, and on 3.13 or 3.14 you can
hit a wheel that has to be compiled from source.

```powershell
py -3.12 --version
```

### 2. Create the virtual environment

```powershell
cd rag_support_bot
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This pulls torch as a dependency of sentence-transformers, so expect a download
of a couple of gigabytes. If you want the smaller CPU-only build explicitly:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If a Chroma dependency tries to compile and fails, install the **Desktop
development with C++** workload from the Visual Studio Build Tools installer,
then retry. Staying on Python 3.12 normally avoids this entirely.

### 3. Install Ollama and pull a model

Download Ollama for Windows from ollama.com, then:

```powershell
ollama pull llama3.1
ollama list
```

On a machine with 8 GB of RAM, use a smaller model instead and tell the app about
it:

```powershell
ollama pull llama3.2:3b
```

Then create a `.env` from the template and set `LLM_MODEL=llama3.2:3b`:

```powershell
copy env.example .env
```

Ollama runs as a background service on Windows once installed. Confirm it is up:

```powershell
curl http://localhost:11434/api/tags
```

### 4. Build the vector index

```powershell
python -m app.ingestion.ingest
```

Expect this to be slow the first time, because the embedding model downloads
about 90 MB of weights and caches them. Do not kill it. Every run after this is
fast.

Re-running is safe and does not create duplicates. For a guaranteed clean
rebuild:

```powershell
python -m app.ingestion.ingest --reset
```

### 5. Start the API

```powershell
uvicorn app.main:app --reload
```

- **UI: http://127.0.0.1:8000**
- Interactive API docs: http://127.0.0.1:8000/docs

The UI is one self-contained HTML file served by the same process — no npm, no
build step, no CDN, and it works offline. It calls the same `POST /api/v1/ask`
that `curl` and the CLI call, so the browser gets no privileged path into the
system and the UI cannot drift from the documented contract. Answers show their
retrieved chunks with relevance scores, latency, and the request id; a refusal is
tagged rather than left looking like a failure.

### 6. Try it

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/ask" -H "Content-Type: application/json" -d "{\"question\":\"What is the return policy?\"}"
```

In PowerShell the escaped double quotes above matter. The `curl` examples in
most guides are written for bash and will not work unchanged in PowerShell.

---

## Using it from the terminal

`curl` gets tedious. There is a terminal client:

```powershell
python -m app.cli                                   # interactive session
python -m app.cli "what is the return policy?"      # one-shot
python -m app.cli --sources "how long is delivery?"  # show chunks and scores
python -m app.cli --json "what is the refund window?"  # raw API response
```

Interactive mode takes three commands: `/sources` toggles chunk display,
`/help`, and `/quit` (Ctrl-D and Ctrl-C also work). A failed question does not
end the session.

It is scriptable, one question per input line, and a bad line does not abort the
batch:

```powershell
Get-Content questions.txt | python -m app.cli
```

By default it talks HTTP to a running server, which is the mode worth using: the
server already holds the embedding model in memory, so a question costs a
retrieval and a generation rather than a model load. It also goes through the
real request path, so what you see is what an API caller gets. The HTTP path uses
only the standard library, so it runs even on the light dependency set.

```powershell
python -m app.cli --local    # no server; reloads the model each start
```

`--local` drives the service in-process. Convenient when you do not want a second
terminal, but expect several seconds before the first answer.

Both modes validate the question through the API's own `AskRequest` model rather
than re-stating the rules, so the CLI cannot drift into accepting input the API
would reject. Errors are mapped to the action that fixes them — a 503 tells you to
run ingestion, a 502 tells you to check `ollama list`, and a refused connection
tells you to start the server or pass `--local`.

---

## Testing

Two things are measured here and they are kept apart on purpose. **Tests** answer
"is it broken" and produce pass/fail. **Evaluation** answers "how good is it now
compared with last week" and produces a scorecard you keep. Full reasoning, plus
the seven ways to test an LLM automatically, is in
[docs/AI_TESTING_STRATEGY.md](docs/AI_TESTING_STRATEGY.md). The commands for each of
those seven methods, and what to do when one goes red, are in
[docs/TESTING.md](docs/TESTING.md).

### The default suite: fast, offline, free

```powershell
pytest
```

296 tests in about four seconds: the health endpoint, input validation and boundaries,
the response contract, error handling and information disclosure, the relevance
threshold logic, output validation, API-level security, and the evaluation
framework's own unit tests.

It is this fast because the service is replaced with a test double through
FastAPI's dependency injection. No model, no index, no network. A validation test
that triggers a real generation call is slow, non-deterministic, and on a paid
provider it also bills you for input that should never have passed the schema.

You can run this suite with only the light dependency set, which is useful if a
native wheel will not build:

```powershell
pip install -r requirements-test.txt
pytest
```

### Choosing a suite

Markers describe what a test *needs*, because that is what decides whether it can
run on a given machine.

```powershell
pytest                              # mocked: everything that needs nothing
pytest -m api                       # HTTP contract, validation, error bodies
pytest -m "rag and not integration" # threshold logic, with stubs
pytest -m integration               # real embeddings and/or a real model
pytest -m "ai and not judge"        # scored answer quality, skipping the slow judge
pytest -m security                  # prompt injection, leakage, refusal behaviour
```

Combine them: `-m api` and `-m "rag and not integration"` run on the light
dependency set, anything with `integration` needs the full `requirements.txt`.

Integration tests need the embedding model; the generation ones also need Ollama
running with your configured model, and they skip with an actionable message
rather than failing when it is absent.

### Evaluating answer quality

Answer quality is not asserted with `assert "30 days" in answer` — that fails on
"thirty days", "a 30-day window" and "30 calendar days", all of which are correct.
Checks go through the `evaluation/` package, which compares normalised facts,
verifies every figure in an answer appears in the retrieved context, and detects
negated meaning ("customers **cannot** return products after 30 days" contains the
right number and states the wrong rule).

```powershell
python -m evaluation.run_eval                                   # all datasets, in-process
python -m evaluation.run_eval --dataset factual
python -m evaluation.run_eval --base-url http://127.0.0.1:8000  # through the real server
python -m evaluation.run_eval --judge --html reports\eval.html
python -m evaluation.run_eval --fail-under 0.9                  # non-zero exit below that
```

That prints a scorecard — correctness, groundedness, semantic match, hallucination
count, Recall@k, safety, latency percentiles — and can write it as JSON and as a
self-contained HTML report. The LLM judge is reported but deliberately does not
decide pass or fail.

### Measuring the thresholds instead of guessing them

```powershell
python -m evaluation.calibrate --retrieval   # in-domain vs out-of-domain scores
python -m evaluation.calibrate --semantic    # matched vs mismatched answer pairs
```

### Reports

```powershell
pytest --cov=app --cov-report=html      # open htmlcov\index.html
pytest --html=report.html --self-contained-html
```
###UI 
<img width="3024" height="1964" alt="image" src="https://github.com/user-attachments/assets/4397d51b-4408-4209-9ac6-61fbe92fd7aa" />





### Postman and Newman

```powershell
npm install -g newman
newman run postman\rag_support_bot_collection.json -e postman\rag_environment.json
```

Requests 1 and 3 to 7 need only a running server. Requests 2 and 8 also need an
ingested index and Ollama.

---

## Configuration

Every setting has a working default, so the app runs with no `.env` file at all.
Copy `env.example` to `.env` only to override something.

| Setting | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `llama3.1` | Must match a model you have pulled |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Runs locally |
| `CHROMA_DIR` | `chroma_db` | Persisted index location |
| `RETRIEVAL_K` | `5` | Chunks considered per question, before the threshold |
| `RELEVANCE_THRESHOLD` | `0.35` | Cosine, range `[-1, 1]`. See the warning below |
| `CHUNK_SIZE` | `500` | Characters. Only bounds the oversize fallback |
| `CHUNK_OVERLAP` | `100` | Characters. Only applies to that fallback |

**The threshold default is a starting point, not a measured value.** It controls
when the bot refuses to answer. Too high and it declines questions it could have
answered; too low and out-of-scope questions get confident answers built from
weakly related text. Measure it against your own documents:

```powershell
python -m evaluation.calibrate --retrieval
```

That prints the score distribution for questions the knowledge base can answer
against questions it cannot, and either suggests a threshold or tells you plainly
that the two groups overlap and no single value separates them.

---

## Where this deliberately departs from the build guide

Four changes. The first three are because the guide's own test suite assumes
behaviour its sample code does not implement; the fourth is because the guide's
approach was measured here and found to be the cause of most wrong answers.

**1. The relevance threshold is enforced in code.** The guide's tests assert that
an out-of-scope question returns an "insufficient information" answer, but its
service passes whatever the retriever returned straight to the model and relies
on the prompt to decline. A prompt is a request, not a constraint, so that
behaviour cannot be asserted. Here weak chunks are dropped before the model is
called, which also means the refusal path never spends a generation.

To make a fixed threshold meaningful at all, the Chroma collection is created
with the cosine distance metric and the embeddings are normalised to unit length.
With the default L2 metric the scores are unbounded and no fixed number can be
reasoned about.

**2. The vector store directory is a constructor argument.** The guide's service
reads the global setting unconditionally, so every test reads and writes the real
developer index. Tests then depend on whatever was ingested last and are not
reproducible on another machine. Here tests build a real index in a temporary
directory and delete it afterwards.

**3. Ingestion is idempotent.** The guide calls `Chroma.from_documents` on every
run, which appends and leaves the collection full of duplicates. Each chunk here
gets a stable id derived from its source file and position, so a second run
replaces chunks in place. Chunks left behind by a document that shrank are
deleted. There is a test that asserts the chunk count does not grow.

**4. Chunks are split on topic boundaries, not to a character budget.** The guide
splits with `RecursiveCharacterTextSplitter` at 500 characters, which packs text
until the budget is full. On these documents that put three unrelated FAQ entries
into one chunk — the return policy, order tracking and contact details — and a
chunk is embedded as a *single vector*, so that vector is the average of all
three. A question about any one of them then matches a blend and scores far below
what it should.

This was measured, not assumed. Asked "How can I contact support?", the chunk
containing the answer ranked **third at 0.3231**, below a chunk that only gives
opening hours; the payment-methods chunk peaked at **0.3451**, four thousandths
under the threshold, so the bot refused a question it could answer. Splitting on
the blank lines the documents already use took those to **0.7343** and **0.8552**,
both rank 1. Across the evaluation set Recall@k went from 88.2% to 100%, MRR from
0.824 to 1.000, correctness from 79.4% to 100%, and hallucinations from 1 to 0.

`CHUNK_SIZE` and `CHUNK_OVERLAP` still exist, but only as the fallback for a
topic too long to embed well on its own. Overlap exists to rescue a sentence cut
in half, and a blank-line boundary does not cut sentences.

Two smaller changes. Errors are typed, so a missing index is a 503, an
unreachable model is a 502, and only a genuine surprise is a 500; the guide
collapses everything into one broad handler. And `HuggingFaceEmbeddings` is
imported from `langchain_huggingface`, which is where LangChain moved it, with a
fallback to the old path.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 503 from `/ask` | The index is empty or missing | Run `python -m app.ingestion.ingest` |
| 502 from `/ask` | Ollama is not running, or the model was never pulled | `ollama list`, then `ollama pull llama3.1` |
| Every answer says it lacks information | The threshold is too high for your documents | Lower `RELEVANCE_THRESHOLD` and measure |
| First run appears to hang | The embedding model is downloading, about 90 MB | Wait; it is cached afterwards |
| `ImportError` on a `langchain_*` module | LangChain package layout drifted | `pip show langchain` and check the current import path |
| 422 on a request you believe is valid | Form data sent instead of JSON, or a missing content type | Send a JSON body with `Content-Type: application/json` |
| `curl` examples fail in PowerShell | The examples are written for bash | Escape inner quotes as `\"` |
| Tests pass locally, fail in CI | Integration tests leaked into the default run | CI runs the mocked suite only; keep the `integration` marker on real-model tests |
| A wheel tries to compile on Windows | No prebuilt wheel for your Python | Use Python 3.12, or install the Visual Studio C++ build tools |

---

## Intentionally not built

This is a prototype, and knowing what is missing matters more than pretending it
is complete. Absent by choice: authentication and authorization, an application
database, conversation history for multi-turn questions, rate limiting, hybrid
search and reranking, streaming responses, metrics and tracing, and a deployment
target.

The architecture does not block any of them. The language model sits behind a
service boundary, so it can be swapped without touching retrieval or the API,
which is exactly how the paid dependency was removed in the first place.

---

## Project layout

```
rag_support_bot/
├── app/
│   ├── main.py                     FastAPI entrypoint, request ids, safety net
│   ├── cli.py                      terminal client: interactive, one-shot, piped
│   ├── static/index.html           the browser UI, self-contained, no build step
│   ├── config.py                   settings, all with working defaults
│   ├── schemas.py                  request and response models, validation layers 1 and 2
│   ├── api/routes.py               endpoints, typed error mapping, injection point
│   ├── services/
│   │   ├── rag_service.py          retrieval, threshold, generation
│   │   ├── response_validator.py   output validation
│   │   ├── vector_store.py         Chroma access, one place only
│   │   ├── embeddings.py           local embedding model factory
│   │   └── exceptions.py           typed service errors
│   ├── ingestion/ingest.py         offline indexing, idempotent
│   └── utils/logger.py             logging that omits question content
├── evaluation/                     how good is it: a measuring instrument
│   ├── normalize.py                wording-independent text comparison
│   ├── fact_evaluator.py           did the answer state the expected facts
│   ├── groundedness_evaluator.py   is every figure supported by the context
│   ├── semantic_evaluator.py       meaning-level similarity + negation guard
│   ├── safety_evaluator.py         injection, leakage, invented policy
│   ├── retrieval_metrics.py        Recall@k, Precision@k, MRR
│   ├── llm_judge.py                model-graded scoring, advisory only
│   ├── report.py                   scorecard to console, JSON, HTML
│   ├── run_eval.py                 the CLI harness
│   ├── calibrate.py                measure thresholds from data
│   └── datasets/                   33 labelled cases in four files
├── docs/
│   ├── ARCHITECTURE.md             what every part is for
│   ├── AI_TESTING_STRATEGY.md      testing a non-deterministic system
│   └── TESTING.md                  runbook: commands per method, reading failures
├── data/documents/                 the knowledge base
├── tests/                          mocked suite plus opt-in integration suites
├── test_data/questions.json        the original small labelled set
├── postman/                        collection and environment
├── pytest.ini                      markers, and the mocked-by-default policy
├── requirements.txt                full stack
├── requirements-test.txt           light set for the mocked suite
└── env.example                     copy to .env to override a default
```
