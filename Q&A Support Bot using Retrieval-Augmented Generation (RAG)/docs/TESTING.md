# Testing runbook — the seven methods in practice

[AI_TESTING_STRATEGY.md](AI_TESTING_STRATEGY.md) explains *why* this project tests
the way it does. This document is the other half: the exact commands, in the order
to run them, what a good result looks like, and what to do when one goes red.

Commands are PowerShell (Windows 11). On macOS or Linux only the path separators
and the activation line change.

**In a hurry:** [the seven methods as one evaluation run](#the-seven-methods-as-one-evaluation-run)
is the command and the order to read the output in. [After a config
change](#after-a-config-change) is the same thing keyed by which setting you touched.

---

## Before anything: which suite can you even run?

Two dependency sets exist, and they decide which methods are available at all.

```powershell
venv\Scripts\activate
python -c "import chromadb, langchain_ollama; print('full stack')"
ollama list
```

| You have | Methods available | Methods blocked |
|---|---|---|
| `requirements-test.txt` only | 1, 2 (structure), 7 | 3, 4, 5, 6 (need embeddings / a model) |
| Full `requirements.txt`, no Ollama | 1, 2, 4, 7 | 3, 5, 6 (need generation) |
| Full stack + `ollama serve` | all seven | — |

Nothing silently degrades. Tests that need something absent are **skipped with a
reason**, and evaluation metrics that could not be computed print `not measured`
rather than a pass. If you see a green run, check whether it was green because it
passed or green because it skipped:

```powershell
pytest -m integration -rs      # -rs prints the reason for every skip
```

---

## The five-minute smoke run

Do this first, every time, before reaching for any of the seven methods.

```powershell
pytest                                            # 1. mocked suite, ~4 s, free
python -m app.ingestion.ingest                    # 2. (re)build the index
python -m evaluation.run_eval --limit 3           # 3. three real answers end to end
```

If step 1 fails, nothing below is meaningful — the contract is broken, not the
model. If step 3 produces answers but step 1 passed, the system is wired up and you
can start measuring quality.

---

## The seven methods as one evaluation run

You do not run seven tools. Six of the seven are wired into a single harness, so an
evaluation is one command and then reading seven different things out of one
scorecard. The per-method sections below are for when one of those numbers moves and
you need to know what it actually measured.

| # | Method | Command | Scorecard line |
|---|---|---|---|
| 1 | Rule-based fact checks | `run_eval --dataset factual` | `correctness` |
| 2 | Golden-dataset regression | `run_eval --json reports\after.json` | `cases / pass rate` + `failures` |
| 3 | LLM-as-a-judge | `run_eval --judge` | `llm judge` (0–2, advisory) |
| 4 | Retrieval metrics | `run_eval --dataset factual` | `recall@k`, `precision@k`, `mrr` |
| 5 | Hallucination | `run_eval --dataset hallucination` | `groundedness`, `hallucinations` |
| 6 | Prompt injection | `run_eval --dataset security` | `safety N/16` |
| 7 | Mocked LLM | `pytest` | pass/fail only — it runs first, and gates the rest |

### The run

```powershell
pytest                                         # 7 - gate everything else
python -m app.ingestion.ingest                 # the index must be current
python -m evaluation.run_eval --judge --html reports\eval.html --json reports\after.json
```

That one `run_eval` call executes methods 1, 2, 3, 4, 5 and 6 across all 33 cases.
`--judge` is the only opt-in; drop it and the run is several times faster.

### Read the output in this order — not the order it prints

The printed order is grouped for humans skimming a report. The *diagnostic* order is
different, because each of these rules out a cause for the ones below it.

1. **`recall@k`** (method 4). Not 100%? Retrieval broke, and everything below is
   uninterpretable. Fix chunking, `RETRIEVAL_K` or the threshold — not the prompt.
2. **`hallucinations`** (5). Must be `0`. Any other number is a defect, not a score.
3. **`safety`** (6). Must be `16/16`. Pass/fail, not a trade-off.
4. **`correctness`** (1) and the `failures` block, which names the exact fact that
   stopped being stated.
5. **`pass rate` against your before-file** (2). This is what makes the run an
   evaluation rather than a test.
6. **The judge's written reasons** (3), last and advisory. Read the sentences; ignore
   the averages until you have hand-calibrated them.

### Regression is the actual point

A single run tells you the current state. Two runs tell you whether a change helped.

```powershell
python -m evaluation.run_eval --json reports\before.json --quiet
# ... change chunking / the prompt / the model / config ...
python -m evaluation.run_eval --json reports\after.json --quiet
```

Diff the `summary` blocks. That is how the chunking rewrite was proven rather than
assumed: pass rate 84.8% → 97.0%, correctness 79.4% → 100%, MRR 0.824 → 1.000.
Without a before-file, "it seems better" is the only conclusion available.

### Two things outside the seven that decide whether they mean anything

```powershell
python -m evaluation.calibrate --retrieval    # is RELEVANCE_THRESHOLD still valid
python -m evaluation.calibrate --semantic     # is the 0.72 semantic threshold valid
```

A guessed threshold is the most common reason an evaluation suite either rejects
correct answers or waves through wrong ones. `calibrate` also says plainly when *no*
threshold separates the two groups — which is precisely why F01's `semantic match`
failure is the instrument and not the bot.

---

## Method 1 — Rule-based assertions

**Asserts on facts, not sentences.** `evaluation/normalize.py` +
`evaluation/fact_evaluator.py`.

```powershell
pytest tests/test_evaluation_framework.py -q      # the matchers themselves
python -m evaluation.run_eval --dataset factual   # the matchers against real answers
```

Read the `correctness` line on the scorecard, then the `failures` block, which names
the specific fact:

```
  F07  Do you ship internationally?
      - fact shipping_regions: required phrase 'European Union' is missing
```

**How to react.** The instinct is to widen the matcher until the test passes. Do
that twice and the evaluator stops detecting anything. Check in this order:

1. Is the answer actually wrong? Fix the bot.
2. Is the answer right and the *phrase* wrong (`"the EU"` vs `"European Union"`)?
   Add the alternative via `any_of`, not by loosening the gap.
3. Is the fact not in the documents at all? The dataset is wrong.

To add a case, edit `evaluation/datasets/factual.json`. The framework enforces that
**the reference answer must pass its own case's fact checks** — if you write a rule
your own model answer fails, `test_evaluation_framework.py` goes red immediately.
That check has caught more dataset bugs than anything else.

---

## Method 2 — Golden-dataset regression

**33 labelled cases, treated as code.** `evaluation/datasets/`.

```powershell
python -m evaluation.run_eval --html reports\eval.html --json reports\eval.json
python -m evaluation.run_eval --dataset factual --dataset open_ended
python -m evaluation.run_eval --only F01 --only O03          # one case at a time
python -m evaluation.run_eval --base-url http://127.0.0.1:8000
python -m evaluation.run_eval --fail-under 0.9               # non-zero exit below that
```

`--base-url` runs the same cases through a live server, so it also exercises
routing, validation and JSON serialisation; the in-process default needs no server
and is faster.

**How to use it as regression detection.** Keep the JSON. Before a change:

```powershell
python -m evaluation.run_eval --json reports\before.json --quiet
# ... change chunking / the prompt / the model ...
python -m evaluation.run_eval --json reports\after.json --quiet
```

Then compare `summary` blocks. This is how the chunking rewrite was verified: pass
rate 84.8% → 97.0%, correctness 79.4% → 100%, MRR 0.824 → 1.000. Without a
before-file, "it seems better" is the only available conclusion.

**Exit-code policy:** any crash or safety violation fails the run regardless of
pass rate. Those are defects, not moved numbers.

---

## Method 3 — LLM-as-a-judge

**A model scores the answer. Advisory, never a gate.** `evaluation/llm_judge.py`.

### Running it

```powershell
# whole suite, with judge scores added to the scorecard
python -m evaluation.run_eval --judge --html reports\eval.html

# faster: judge a few cases, skip the embedding model
python -m evaluation.run_eval --judge --only F01 --only O03 --no-semantic

# through pytest (slowest marker in the suite)
pytest -m judge -rs
pytest -m "ai and not judge"          # everything else, judge excluded
```

### Point it at a different model

The judge defaults to `settings.llm_model` — the same model that wrote the answer,
which is close to asking a student to mark their own paper. Override it:

```powershell
ollama pull qwen2.5:14b
$env:JUDGE_MODEL = "qwen2.5:14b"
python -m evaluation.run_eval --judge --dataset factual
```

`judge_model_name()` reports what was actually used, and the scorecard prints it,
so a run is never ambiguous about who did the grading. On this machine only
`llama3.1` is installed, so every judge number below is self-graded and should be
read with that discount.

### It only judges 17 of the 33 cases

Content cases — `factual.json` and `open_ended.json` — get a judge verdict. The 16
`hallucination.json` and `security.json` cases do not: they declare
`expected_behavior`, and they are scored on whether the bot refused and whether it
leaked, which is a deterministic behavioural question. Asking a model whether a
refusal was the right call reintroduces exactly the judgement those cases exist to
remove. So `scored 17/17` on the scorecard means "all the cases a judge is
appropriate for", not "half the run silently failed".

### What it returns

Three criteria, 0–2 each, plus one sentence of reasoning, constrained with
`format="json"` and then validated by `parse_judge_output()`:

```json
{
  "groundedness": 1,
  "relevance": 2,
  "correctness": 0,
  "reason": "the answer only partially addresses the question ..."
}
```

The reason is the part worth reading. The scores tell you a case moved; the
sentence tells you what to look at.

### Worked example — and why it is not a gate

Two cases, judged live on this machine:

| Case | Answer | Deterministic verdict | Judge |
|---|---|---|---|
| F01 | "The return period is 30 days from the date of delivery." | facts 1.0, groundedness 1.0, **pass** | G1 R2 **C1** — "omits the refund processing time" |
| O03 | "A request is treated as high priority when a payment has been charged twice. You should receive a first response within 4 hours." | facts 1.0, groundedness 1.0, **pass** | G1 R2 **C0** — "does not provide any information about how the double charge is handled" |

Both answers are correct, and on O03 the judge is **wrong in an instructive way**:
it wants the remedy for a double charge, which the corpus never states. Answering it
would have required inventing a policy — the exact failure the whole system exists
to prevent. A judge-gated build would have marked the correct, grounded, honest
answer as a regression and pushed a developer toward making the bot fabricate.

The pattern holds across the whole run. A full `--judge` pass over all 33 cases:

```
  quality
    correctness      100.0%      groundedness   100.0%      hallucinations  0

  llm judge (advisory, not a gate; scale 0-2)
    scored           17/17 cases
    groundedness     1.06        relevance      2.00        correctness     1.59
```

Deterministic groundedness is 100% — every figure in every answer traced back to the
retrieved context. The judge, grading the same 17 answers on the same property,
averages 1.06 out of 2. One of those numbers is measuring the text; the other is
measuring a small model's opinion of the text. Neither is useless, but only one of
them belongs in a gate.

That is the case for the design, stated in the docstring and enforced in code:

- **Advisory, never the gate.** `CaseResult.passed` ignores the judge entirely. It
  moves a reported average; it cannot fail your build.
- **Unavailability is not success.** No model reachable, malformed JSON, a score
  outside `{0,1,2}`, a boolean where a number belongs → `available=False` and the
  scorecard prints `not evaluated`. A metric that defaults to green is worse than no
  metric.
- **Two attempts, then give up.** `judge_answer(..., attempts=2)`. A small model
  occasionally emits prose around the JSON; the parser digs the object out, and one
  retry covers the rest without turning the run into a loop.
- **Testable without a model.** `judge_answer(..., invoke=my_callable)` injects the
  call, which is how `test_evaluation_framework.py` tests all the parsing paths in
  the default offline suite.

### Calibrate it before you believe a number

Absolute judge scores mean nothing until you know how often the judge agrees with
you. Once:

```powershell
python -m evaluation.run_eval --judge --json reports\judge_baseline.json
```

Open the JSON, read `cases_detail[].judge` next to `cases_detail[].answer`, and
hand-score 20 or so cases yourself on the same 0–2 scale. Then measure agreement.

- Agrees most of the time → the trend is usable. Watch it move; still don't gate.
- Disagrees like O03 above → the judge is measuring the corpus's silence, not the
  bot's quality. Use the deterministic checks and treat judge output as prompts for
  where to look.

Do not use a judge as the only check on anything that carries money or safety.

### When to reach for it

The judge earns its keep on exactly one thing the deterministic checks cannot do:
answers that are factually right and *unhelpful* — badly structured, burying the
condition, answering a different question than was asked. Everything about figures,
fabrication and retrieval is measured faster, cheaper and reproducibly by methods 1,
4 and 5.

---

## Method 4 — Retrieval metrics (RAGAS equivalents, no API key)

**Answer quality is capped by retrieval.** `evaluation/retrieval_metrics.py`.

```powershell
python -m evaluation.run_eval --dataset factual --no-semantic
pytest -m rag
```

Read `recall@k` **first**, before any answer-quality metric:

| Symptom | Meaning | Fix |
|---|---|---|
| Recall@k low | the answer chunk is never retrieved | chunking, `RETRIEVAL_K`, threshold — *not* the prompt |
| Recall@k 100%, correctness low | the context was there and the model fumbled it | the prompt or the model |
| MRR well below 1.0 | the right chunk is retrieved but ranked low | chunking; a diluted multi-topic chunk ranks badly |
| Precision@k low | some retrieved chunks are noise | usually acceptable; `k` is deliberately generous |

Precision@k around 0.75 is fine here and not worth optimising. The relevance
threshold is applied *after* retrieval, so a larger `k` can only admit chunks that
were already judged relevant — raising it is close to free.

### Measure thresholds, never guess them

```powershell
python -m evaluation.calibrate --retrieval    # in-domain vs out-of-domain scores
python -m evaluation.calibrate --semantic     # matched vs mismatched answer pairs
```

`calibrate` either suggests a threshold or says plainly that the two groups overlap
and no single threshold separates them. Both outputs are useful — and it is an
independent instrument, which is why it is worth running after any retrieval change.
It corroborated the chunking fix on its own: before, it refused to suggest a
retrieval threshold at all because in-domain scores (min 0.345) overlapped
out-of-domain (max 0.408); afterwards the groups separated cleanly (0.548 vs 0.402).

**Do not adopt a suggestion blindly.** It suggested 0.47; the configured value
stays 0.35, because a needed chunk for O03 sits at 0.3518 and 0.47 would cut it.
`calibrate`'s in-domain list contains well-phrased questions and no symptom-phrased
ones ("I was charged twice"), so its suggestion is over-confident on real traffic.

---

## Method 5 — Hallucination testing

**The strongest free signal.** `evaluation/groundedness_evaluator.py` +
`evaluation/datasets/hallucination.json`.

```powershell
python -m evaluation.run_eval --dataset hallucination
pytest -m "ai and not judge"
```

Two rules, deliberately asymmetric in strictness:

- **Numeric support (strict).** Every figure in the answer must appear in the
  retrieved context or in the question. Invented policy almost always carries an
  invented number, so this one rule catches most fabrication with no model and no
  cost.
- **Claim overlap (lenient).** Sentences whose vocabulary barely appears in the
  context are *reported*, not failed. A correct paraphrase can score low and a
  fabrication recycling source vocabulary can score high, so this is a pointer, not
  a verdict.

Read `hallucinations N case(s)` on the scorecard. **The target is 0, and it is not
negotiable the way a percentage is** — one fabricated policy statement matters more
than three points of correctness.

A failure here is usually not the prompt. Two real ones from this project:

- "How can I contact support?" → "submitting a request through our website". The
  website is not in the corpus. Root cause was retrieval: the chunk that *did*
  answer it ranked #3 at 0.3231 and got cut. Fixed by chunking.
- "I was charged twice" → "We will immediately investigate and reverse the duplicate
  charge." Grounded-sounding, and a promise the corpus never makes. Fixed by a
  prompt rule forbidding committed outcomes — describe what the policy says, not
  what you expect will be done.

---

## Method 6 — Prompt-injection testing

**Every rule is deterministic.** `evaluation/safety_evaluator.py` +
`tests/test_security.py`.

```powershell
pytest -m security
python -m evaluation.run_eval --dataset security
```

The scorecard's `safety passed 16/16` must stay at full marks. Unlike quality
metrics, this is pass/fail: a leak is a defect, and the exit code reflects that
regardless of pass rate.

What is checked: system-prompt fragments, internal paths, `chroma_db`, port 11434,
tracebacks, credential-shaped strings, PII absent from the context; refusal
behaviour (any refusal marker *or* the structural `insufficient_context` flag, since
wording is the model's choice and behaviour is not); and invented policy, which
reuses the numeric groundedness check because that is what the damage looks like.

**One thing to know before you edit the prompt.**
`test_no_answer_ever_echoes_the_system_prompt` feeds the application's *real*
`SYSTEM_PROMPT` through the leak detector. Change the prompt without updating
`PROMPT_LEAK_MARKERS` and that test fails — by design, so the detector cannot
silently go blind. When it fails after a prompt edit, add the new distinctive
phrases to the marker list; that is the fix, not an exemption.

Most attacks never reach the model at all: an injection string resembles no support
document, scores below the threshold, and no context is assembled, so generation is
skipped. `test_injection_prompts_are_mostly_stopped_before_the_model` measures how
many are stopped that way and **prints the number instead of asserting 100%**, so
the figure stays honest.

Passing this suite does not mean injection-proof. It means these specific attacks
did not succeed, and that a defence which used to hold is re-checked whenever the
prompt, model or chunking changes.

---

## Method 7 — Mocking the LLM

**The default suite, and the highest-value method on the list.**
`tests/conftest.py`.

```powershell
pytest                       # 296 passed in ~4 s
pytest -m api                # HTTP contract, validation, status codes, error bodies
pytest -q --lf               # last failures only, while iterating
pytest tests/test_small_talk.py -v
```

`MockRAGService` is installed through FastAPI's `dependency_overrides`, so the
whole request path runs with **exact** assertions and no model, index, network or
cost. This is where validation, boundaries, the response contract, error handling,
information disclosure, output validation and the evaluation framework's own unit
tests live.

It is also the right place for *logic* that happens to sit near the model. Small
talk is the example: `test_small_talk_never_touches_the_index` injects a store and
an LLM that raise on any access, which proves a greeting is answered without
spending a retrieval or a generation — an assertion that would be impossible to make
against the real stack.

This suite must stay fast and fully green. If it needs a model, a download or a
network call, it belongs behind the `integration` marker instead.

---

## After a config change

The one command that covers everything:

```powershell
venv\Scripts\activate

python -m app.ingestion.ingest              # 1. rebuild the index
pytest                                      # 2. nothing structural broke (~4 s)
python -m evaluation.calibrate --retrieval  # 3. are the score groups still separable
python -m evaluation.run_eval --json reports\after.json --html reports\eval.html
```

Run them in that order and stop at the first one that fails — step 4 cannot tell you
anything useful if step 1 or 2 is broken.

Then diff `reports\after.json` against the run from *before* the change. Without a
before-file there is nothing to compare to, so take one first:

```powershell
python -m evaluation.run_eval --json reports\before.json --quiet
```

### Which keys need what

Not every setting needs the full sequence. What matters is whether the change
invalidates the **stored vectors** — if it does, the index must be rebuilt or every
metric below it is measuring a stale index.

| Changed in `.env` / `config.py` | Rebuild index? | Run |
|---|---|---|
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | **yes** | `ingest` → `pytest tests/test_chunking.py` → `calibrate --retrieval` → `run_eval` |
| `EMBEDDING_MODEL` | **yes** | `ingest` → `calibrate --retrieval` **and** `--semantic` → `run_eval` |
| `DOCUMENTS_DIR`, `COLLECTION_NAME`, `CHROMA_DIR` | **yes** | `ingest` → `pytest -m rag` → `run_eval` |
| `RETRIEVAL_K` | no | `pytest -m rag` → `run_eval` (watch `recall@k`, `precision@k`, `mrr`) |
| `RELEVANCE_THRESHOLD` | no | `calibrate --retrieval` → `run_eval --dataset hallucination --dataset security` |
| `LLM_MODEL`, `LLM_TEMPERATURE` | no | `pytest -m integration` → `run_eval` before/after JSON |
| `LLM_TIMEOUT_SECONDS`, `OLLAMA_BASE_URL` | no | `pytest -m api` → `python -m evaluation.run_eval --limit 3` |
| `LOG_LEVEL` | no | `pytest` (the mocked suite asserts nothing leaks into logs) |

Two of these are worth a warning:

- **`RELEVANCE_THRESHOLD` is the one to be most careful with.** Raise it and
  out-of-scope questions get refused more reliably, but real questions start falling
  off the bottom — which shows up as a *correctness* drop, not as anything that looks
  like a threshold problem. Always run the `hallucination` and `security` datasets
  after touching it, because those are the two it trades against. And do not simply
  adopt what `calibrate` suggests: it suggested 0.47 here, and 0.47 would have cut a
  chunk O03 needs at 0.3518.
- **Changing `EMBEDDING_MODEL` invalidates both thresholds at once.** Different model,
  different score distribution, so `RELEVANCE_THRESHOLD` and the semantic threshold
  are both meaningless until re-measured. Run both halves of `calibrate`.

### What "still good" looks like

Compare against the last known-good run. These are the current values:

```
  cases 33   passed 32   pass rate 97.0%
  correctness 100.0%   groundedness 100.0%   semantic match 94.1%
  hallucinations 0
  recall@k 100.0%   precision@k 75.6%   mrr 1.000
  safety 16/16
  failures: F01 (semantic 0.672 below 0.72 - the instrument, not the bot)
```

Non-negotiable regardless of the config: `hallucinations 0` and `safety 16/16`. A
config change that moves either of those is not a tuning trade-off, it is a defect.
`recall@k` is the first number to read — if it dropped, the answer chunk stopped being
retrieved, and nothing downstream is worth interpreting until that is fixed.

Add `--fail-under 0.9` if you want a non-zero exit code rather than a number to read:

```powershell
python -m evaluation.run_eval --fail-under 0.9
```

---

## Which suite to run when

| Situation | Command | Time |
|---|---|---|
| Every save | `pytest` | ~4 s |
| Touched routes, schemas, validation | `pytest -m api` | ~1 s |
| Touched chunking, retrieval, the threshold | `python -m app.ingestion.ingest` then `pytest -m rag` and `calibrate --retrieval` | ~2 min |
| Touched the prompt | `pytest -m "ai and not judge"` then `pytest -m security` | ~2 min |
| Changed the model | full `pytest -m integration` + `run_eval` before/after JSON | ~10 min |
| Before a release | `pytest -m integration` then `python -m evaluation.run_eval --judge --html reports\eval.html` | tens of minutes |
| In CI | `pytest` only | seconds |

CI runs the mocked suite only. Integration tests need a local model server, which a
standard runner does not have — and pointed at a hosted model they would need a
manual trigger and a spend cap, because a test loop against a paid API on every push
is how a side project generates a real bill.

---

## Reading a failure, in order

Work down this list. Each step rules out a cause, and skipping to the bottom is how
people end up rewriting a prompt to fix a retrieval bug.

1. **Did the mocked suite pass?** No → the contract is broken. Ignore the model.
2. **Is the index current?** `python -m app.ingestion.ingest` after *any* document
   or chunking change. A stale index makes every quality metric meaningless.
3. **Is `recall@k` 100%?** No → retrieval. Chunking, `RETRIEVAL_K`, threshold.
4. **Is `hallucinations` 0?** No → fabrication. Read the flagged figure and find out
   whether retrieval starved the model or the prompt let it improvise.
5. **Is `safety` full marks?** No → stop and fix. Not a number to trade off.
6. **Only then** look at correctness, semantic match, and the judge's reasons.
7. **Read three answers yourself.** Deterministic checks catch wrong facts and
   invented figures. They do not catch an answer that is accurate, grounded and
   useless.

### Failures that are the instrument, not the bot

Worth recognising, because chasing them wastes a day:

- **`semantic match` on F01.** "The return period is 30 days from the date of
  delivery." scores 0.672 against a 0.72 threshold — and `calibrate --semantic`
  reports that matched pairs (min 0.672) and mismatched pairs (max 0.722) overlap on
  this dataset, so *no* threshold separates them. The answer is correct. The metric
  cannot tell.
- **`UserWarning: Relevance scores must be between 0 and 1`.** LangChain is wrong.
  Cosine relevance runs `[-1, 1]`; negative scores are measured routinely here and
  are normal for an unrelated chunk.
- **A judge score dropping while every deterministic check passes.** See the O03
  example above.

---

## The ladder

Cheapest and strongest first. Work down it only when the layer above cannot answer
the question.

```
mocked suite (7)            milliseconds, exact, catches contract regressions
normalise + facts (1,2)     free, deterministic, catches most quality regressions
numeric groundedness (5)    free, deterministic, catches most fabrication
injection rules (6)         free, deterministic, pass/fail
retrieval metrics (4)       local embeddings, tells you whether the fix is upstream
semantic similarity         local embeddings, catches meaning drift; calibrate it
llm judge (3)               advisory, never a gate, calibrate before believing it
human sampling              a few answers per release, irreplaceable
```

Most teams start at the judge and never build the first three, which is why their AI
suites are slow, flaky, and quietly untrusted.
