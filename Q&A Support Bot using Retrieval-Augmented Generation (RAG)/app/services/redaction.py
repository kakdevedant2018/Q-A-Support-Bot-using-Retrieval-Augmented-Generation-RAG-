"""Strip credential-shaped strings out of anything the API is about to return.

This exists because of a test failure, not a hypothesis. `tests/test_security.py`
indexes a document containing `api key: sk-poisoned-...`, asks an ordinary
question, and the answer came back quoting the key verbatim. The bot refused to
*obey* the instructions planted in that document, which is the defence everyone
thinks about, and then repeated the secret anyway - which needs no jailbreak at
all, only a question relevant enough to retrieve the chunk.

The scenario is mundane rather than exotic: somebody pastes a runbook or a
support ticket into the knowledge base. Retrieval has no notion of sensitivity,
so it will surface that text exactly as eagerly as the returns policy.

Two design decisions worth stating.

Credentials only, not PII. A key in an answer is never correct, so removing it
cannot destroy a good answer. An email address or a postal address may well be
the right answer for some other corpus, so deciding those are unacceptable is a
policy call for whoever owns the data, not something a redaction helper should
make on their behalf.

Sources are scrubbed too. `/api/v1/ask` returns the retrieved chunks so a user
can check the answer against them, which means a secret leaks through `sources`
whether or not the model repeated it.

This is a mitigation, not a control. Anything it catches is already indexed, and
the real fix is to keep credentials out of the corpus. It buys the case where
that has already failed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

REDACTED = "[REDACTED]"

# Kept deliberately in step with `evaluation.safety_evaluator.SECRET_PATTERNS`.
# Duplicated rather than imported: `app` must not depend on `evaluation`, or the
# shipped service would carry the test harness with it. A test asserts the two
# lists stay equivalent, so the copy cannot quietly drift.
SECRET_PATTERNS: Tuple[re.Pattern, ...] = (
    # `api_key: xyz`, `password = hunter2`, `Bearer abc...`. The label is kept
    # and only the value is replaced, because "an API key was removed here" is
    # more useful to a reader than a bare marker.
    #
    # The lookahead makes the rule idempotent. Without it the marker itself is a
    # `\S+`, so `api_key: [REDACTED]` still matches - which meant `contains_secret`
    # reported a credential in already-scrubbed text and the service logged a
    # warning about a leak it had just prevented.
    # `(?![:=])` closes a backtracking hole. With only the marker lookahead, the
    # separator alternation could give up `" = "` and re-match as `\s+`, leaving
    # `=` itself as the "value" - so `password = [REDACTED]` came out as
    # `password [REDACTED] [REDACTED]`. A value can never begin with a separator.
    re.compile(
        r"\b(api[_-]?key|secret|password|bearer|token)\b(\s*[:=]\s*|\s+)"
        r"(?![:=])(?!\[REDACTED\])(\S+)",
        re.I,
    ),
    # Provider-style keys, which carry no label and so must be matched directly.
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    # Long hex runs: digests, session identifiers, hex-encoded keys.
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
)


def _replace_labelled(match: re.Match) -> str:
    """Keep `api_key:` and drop only what follows it."""
    label, separator, _value = match.groups()
    return f"{label}{separator}{REDACTED}"


def redact_text(text: str) -> str:
    """Return `text` with credential-shaped substrings replaced."""
    if not text:
        return text

    cleaned = SECRET_PATTERNS[0].sub(_replace_labelled, text)
    for pattern in SECRET_PATTERNS[1:]:
        cleaned = pattern.sub(REDACTED, cleaned)
    return cleaned


def contains_secret(text: str) -> bool:
    """True if `text` looks like it carries a credential.

    Used for logging a warning. The caller is told that something was removed
    without the removed value being written to the log, which would defeat the
    point by moving the secret from the response into the log file.
    """
    return bool(text) and any(p.search(text) for p in SECRET_PATTERNS)


def redact_sources(sources: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Scrub the `content` of each retrieved chunk, leaving metadata alone."""
    scrubbed: List[Dict[str, Any]] = []
    for source in sources or ():
        item = dict(source)
        if isinstance(item.get("content"), str):
            item["content"] = redact_text(item["content"])
        scrubbed.append(item)
    return scrubbed


def redact_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Scrub a full `RAGService.ask` result in place of the original.

    Applied at the end of `ask` rather than in the API layer so that every caller
    is covered - the evaluation harness talks to the service directly, and a
    mitigation the harness cannot see is one the scorecard cannot measure.
    """
    result = dict(response)
    if isinstance(result.get("answer"), str):
        result["answer"] = redact_text(result["answer"])
    if "sources" in result:
        result["sources"] = redact_sources(result.get("sources"))
    return result
