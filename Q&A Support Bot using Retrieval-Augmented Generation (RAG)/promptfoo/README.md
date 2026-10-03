# Adversarial red-teaming

An attack *generator*, sitting outside the pytest suite on purpose.

`evaluation/datasets/security.json` holds eight injection and leakage attacks
written by hand. They are the right shape for a gate: fixed, fast, and
reproducible, so each one proves a specific defence still holds. What they
cannot do is find an attack nobody has thought of, because every case in them
is an attack somebody already thought of.

This directory does the other half. Promptfoo synthesises attacks per plugin
and rewrites each through several strategies, which turns eight fixed probes
into a few hundred varied ones per run.

The two halves are connected, and the connection is the point:

```
  promptfoo (search)              pytest (gate)
  generated, varies per run       fixed, identical every run
  minutes, needs Node + server    3 seconds, no dependencies
  advisory - a shortlist          blocking - a release decision
          │                                ▲
          └──── to_regression.py ──────────┘
               a finding becomes a permanent case
```

A red-team finding that is fixed but not pinned is a finding that can be
silently undone by the next refactor. `to_regression.py` is what stops that.

## Two files, because they have different prerequisites

| | `target-check.yaml` | `redteam.yaml` |
|---|---|---|
| what it does | proves the HTTP wiring works | generates and runs attacks |
| assertions | deterministic strings and regexes | model-graded |
| needs | Node + running server | the above, plus a one-time email |
| runtime | ~2s | minutes |

Run `target-check.yaml` first, always. A promptfoo HTTP target has four things
that can be silently wrong - URL, body template, `transformResponse`, server
down - and the worst of them fails *clean*: get `transformResponse` wrong and
every attack is graded against `undefined`, which reads as a refusal, so the
red-team report comes back perfect and means nothing.

```bash
uvicorn app.main:app --port 8000        # in another terminal
npx promptfoo@latest eval -c promptfoo/target-check.yaml
```

Verified working: 5/5 against this repo. It found two real mistakes while being
written, both in the assertions rather than the bot, and both now documented
inline - the service has two different refusal wordings for its two refusal
routes, and promptfoo compiles `regex` with the JS engine, which rejects the
inline `(?i)` flag.

## The red-team run

No API key and no paid account: generation and grading are both pointed at the
same local `llama3.1` the application uses. Promptfoo defaults to OpenAI for
both, which would need a paid key *and* would send the attack corpus off this
machine.

There is one non-obvious prerequisite. As of promptfoo 0.123, `redteam
generate` refuses to run without a verified email and prompts for one
interactively:

```
  Email Verification Required
  Red team scans require email verification to continue.
  ? Work email:
```

Set it once, non-interactively:

```bash
npx promptfoo@latest config set email you@example.com
```

That is a deliberate decision to make, not a formality - it registers an
address with a third-party service and is why this step is documented rather
than scripted. Nothing else in this repository requires an account, which is
the whole point of the local-only stack, so this is the one place that
property breaks.

```bash
cd promptfoo
npx promptfoo@latest redteam generate -c redteam.yaml -o redteam.generated.yaml
npx promptfoo@latest redteam eval -c redteam.generated.yaml -o results.json
npx promptfoo@latest view               # browse the run in a UI
```

Generate and eval are separate commands for a reason: the generated file is the
attack corpus, and keeping it lets you re-run the *same* attacks against a
changed prompt. That is the only way to get a before/after comparison out of a
non-deterministic generator.

A smaller run while iterating on the config:

```bash
npx promptfoo@latest redteam generate -c redteam.yaml -o /tmp/smoke.yaml \
  --plugins prompt-extraction --strategies jailbreak -n 1
```

## Reading the output

Expect false positives. The grader is `llama3.1` judging `llama3.1`, which is
the same conflict of interest `docs/AI_TESTING_STRATEGY.md` §3.3 raised about
the LLM judge, and the same conclusion applies: **the output is a shortlist for
a person to read, never a number to gate on.** A local 8B grader in particular
tends to mark a correct refusal as a failure when the refusal is terse.

So read the failures, not the pass rate. For each one:

1. **Did the attack actually succeed?** Look at the answer, not the verdict.
   A refusal marked as a failure is a grader error - discard it.
2. **If it succeeded, what defence should have stopped it?** Usually the
   relevance threshold: an off-topic attack retrieves nothing above 0.35 and
   the model is never called. If an attack got context, that is the finding.
3. **Pin it.** `python to_regression.py results.json` prints candidate cases
   for `evaluation/datasets/regression.json`. Review and edit them before
   merging - the ids and notes always need work, and a case encoding a wrong
   expectation is worse than no case because it has to be argued with later.
4. **Then fix it**, and watch the new case go from red to green.

## Known limitations

Written down rather than discovered later:

- **Attack quality is capped by the local model.** A frontier model generates
  materially better jailbreaks. This setup trades that for costing nothing and
  keeping the corpus local. For a system where the stakes justify it, point
  `redteam.provider` at a stronger model and accept the bill.
- **The grader is not independent.** See above.
- **Clean run ≠ secure.** It means this generator, with this model, did not
  find anything this time. Absence of evidence.
- **Not in CI, by design.** Generated cases differ between runs, so a failure
  cannot be attributed to a code change - which is the one thing CI exists to
  do. The gate stays in pytest; this is a periodic exercise, before a release
  or after a prompt change. (`target-check.yaml` *is* deterministic and could
  be gated, but it only duplicates what `tests/test_security.py` and
  `tests/test_ai_quality.py` already cover in a runner that needs no Node.)
- **Requires an email registration for the red-team half.** Above.

## What Promptfoo is not used for here

The usual headline use - A/B two prompt versions over the same dataset - does
not fit this project, and forcing it would be a worse outcome than skipping it.

That pattern needs the prompt as a template file so promptfoo can substitute
versions: `prompts: [file://prompts/v1.txt, file://prompts/v2.txt]`. Here
`SYSTEM_PROMPT` lives in `app/services/rag_service.py`, and the comment above
it explains why: only the role line is configurable, because exposing the whole
prompt as a setting would let a deployment silently drop the parts that were
measured. Extracting it to a text file to satisfy a tool would undo a
deliberate decision in order to gain a comparison that
`evaluation/run_eval.py --json` already produces by running the suite against
each revision.

So: the tool is used for the thing it is uniquely good at here - generating
attacks nobody wrote - and not for the thing it is nominally good at but this
codebase is shaped against.

## Why not Giskard as well

The article this layer is modelled on recommends Giskard as the release-gate
scanner, and its distinguishing feature there is automated **metamorphic**
testing. That is now implemented natively in `evaluation/metamorphic.py`, with
relations written against this corpus rather than generated generically - and
on its first run it found a reproducible retrieval defect (MR03) that a generic
scan would have been unlikely to phrase. Adding Giskard would bring a pandas
and model-wrapper dependency for the overlap that remains.

That is a judgement about this project at this size, not about the tool. It is
revisited in `docs/AI_TESTING_STRATEGY.md` §3.4.
