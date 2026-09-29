"""Text normalisation shared by every evaluator.

This module exists because of one problem: a correct answer can be worded in
many ways, and `"30 days" in answer` fails on all but one of them.

    30 days
    thirty days
    a 30-day window
    30 calendar days

All four say the same thing. Normalisation maps them onto a common form so a
deterministic check can still be used where a deterministic check is cheapest.

Two rules govern everything here:

1. Normalisation is applied identically to both sides of every comparison. It
   is never applied to only the answer or only the expectation.
2. It is deliberately conservative. Widening it to force a test to pass is how
   an evaluator stops detecting anything. Prefer a semantic check over a
   looser regular expression.

Nothing in this module imports a model, makes a network call, or costs money.
"""

from __future__ import annotations

import re
from typing import List, Set

# Number words up to a hundred. A support knowledge base deals in days, hours,
# and business days, so this range is enough; anything larger is written as
# digits in practice.
_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

# A number word is only converted to a digit when it quantifies something
# measurable. Without this gate, "one of our support agents" becomes "1 of our
# support agents", and the groundedness check then reports the figure 1 as an
# unsupported claim. Converting "thirty days" is useful; converting every "one"
# and "second" manufactures numbers that were never facts.
_QUANTIFIABLE = {
    "day",
    "hour",
    "minute",
    "week",
    "month",
    "year",
    "business",
    "working",
    "calendar",
    "time",
    "percent",
    "attempt",
    "item",
    "product",
    "order",
    "installment",
}

# Tokens allowed between a number word and its unit, so ranges like "five to
# seven business days" still convert.
_CONNECTORS = {"to", "and", "or", "through", "up", "until"}

# Words that flip the meaning of a nearby claim. "Customers cannot return
# products after 30 days" contains the phrase "30 day" and must not be scored
# as confirming "the return period is 30 days".
NEGATION_TOKENS: Set[str] = {
    "not",
    "no",
    "never",
    "cannot",
    "cant",
    "wont",
    "dont",
    "doesnt",
    "didnt",
    "isnt",
    "arent",
    "wasnt",
    "none",
    "without",
    "unable",
    "denied",
    "deny",
    "refused",
    "ineligible",
    "excluded",
    "prohibited",
    "unavailable",
}

_SMART_QUOTES = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
}

_KEEP = re.compile(r"[^a-z0-9%$:\s]")
_WS = re.compile(r"\s+")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_LIST_MARKER = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+", re.MULTILINE)


def strip_list_markers(text: str) -> str:
    """Remove leading bullet and numbered-list markers.

    An answer formatted as "1. ... 2. ..." contains the numbers 1 and 2, which
    are not factual claims. Groundedness checks on numbers would otherwise flag
    every well-formatted list as unsupported.
    """
    return _LIST_MARKER.sub("", text)


def _singularise(word: str) -> str:
    """A deliberately small plural rule, not a real stemmer.

    Handles the cases that actually appear in support answers: days, hours,
    cards, policies. Leaves anything ambiguous alone, because a wrong stem
    creates a false match and a false match is worse than a missed one.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if (
        len(word) > 3
        and word.endswith("s")
        and not word.endswith(("ss", "us", "is", "as", "os"))
    ):
        return word[:-1]
    return word


def _quantifies_something(tokens: List[str], start: int) -> bool:
    """Whether a measurable unit follows, allowing connectors and further numbers.

    "five to seven business days" qualifies: to -> seven -> business. "one of our
    agents" does not: the token after the number is "of".
    """
    index = start
    steps = 0
    while index < len(tokens) and steps < 4:
        token = tokens[index]
        if _singularise(token) in _QUANTIFIABLE:
            return True
        if (
            token in _CONNECTORS
            or token in _UNITS
            or token in _TENS
            or token.isdigit()
        ):
            index += 1
            steps += 1
            continue
        return False
    return False


def _words_to_numbers(tokens: List[str]) -> List[str]:
    """Convert quantifying number words to digits.

    "thirty days" -> "30 day", "twenty five days" -> "25 day", and
    "five to seven business days" -> "5 to 7 business day". A number word that
    quantifies nothing measurable is left alone; see `_QUANTIFIABLE`.
    """
    out: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]

        value: int | None = None
        consumed = 1

        if token in _TENS:
            value = _TENS[token]
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            if following in _UNITS and 1 <= _UNITS[following] <= 9:
                value += _UNITS[following]
                consumed = 2
        elif token in _UNITS:
            value = _UNITS[token]

        if value is not None and _quantifies_something(tokens, index + consumed):
            out.append(str(value))
            index += consumed
            continue

        out.append(token)
        index += 1

    return out


def normalize(text: str) -> str:
    """Canonical form used by every phrase comparison in the framework.

    lowercase -> straighten quotes -> hyphens to spaces -> drop punctuation ->
    number words to digits -> light singularisation -> collapse whitespace.
    """
    if not text:
        return ""

    lowered = text.lower()
    for fancy, plain in _SMART_QUOTES.items():
        lowered = lowered.replace(fancy, plain)

    # Hyphens join words that should compare as separate tokens: a "30-day
    # window" has to match "30 day".
    lowered = lowered.replace("-", " ").replace("/", " ")
    lowered = _KEEP.sub(" ", lowered)

    tokens = _WS.sub(" ", lowered).strip().split(" ")
    tokens = [t for t in tokens if t]
    tokens = _words_to_numbers(tokens)
    tokens = [_singularise(t) for t in tokens]

    return " ".join(tokens)


DEFAULT_MAX_GAP = 2


def contains_phrase(haystack: str, phrase: str, *, max_gap: int = DEFAULT_MAX_GAP) -> bool:
    """Whether a phrase appears in a text, both sides normalised first.

    `max_gap` allows intervening words between the phrase's tokens, because a
    correct answer inserts modifiers a fixed substring cannot anticipate:

        "30 days"   matches "30 calendar days" and "30 business days"
        "send back" matches "send the item back"

    Word-boundary aware at both ends, so "30 day" still does not match inside
    "130 days".

    The gap is bounded on purpose. Widen it and the check starts matching
    unrelated text that happens to share tokens; that is why the numeric
    groundedness check uses exact figures rather than phrase matching.
    """
    normalized_phrase = normalize(phrase)
    if not normalized_phrase:
        return False

    tokens = [re.escape(t) for t in normalized_phrase.split(" ") if t]
    if not tokens:
        return False

    gap = r"(?:\s+\w+){0," + str(max(0, max_gap)) + r"}\s+"
    pattern = r"(?<!\w)" + gap.join(tokens) + r"(?!\w)"
    return re.search(pattern, normalize(haystack)) is not None


def extract_numbers(text: str, *, ignore_list_markers: bool = True) -> List[str]:
    """Every numeric value in a text, as canonical strings.

    Number words are converted first, so "thirty days" yields ["30"]. Used by
    the groundedness check: a figure in an answer that appears nowhere in the
    retrieved context is the signature of an invented policy.
    """
    source = strip_list_markers(text) if ignore_list_markers else text
    found = _NUMBER.findall(normalize(source))

    canonical: List[str] = []
    for raw in found:
        value = float(raw)
        canonical.append(str(int(value)) if value.is_integer() else str(value))
    return canonical


def sentences(text: str) -> List[str]:
    """Split into sentences well enough for claim-level analysis.

    Not a linguistic parser. Claim-level granularity only has to be good enough
    to point a reviewer at the offending part of an answer.
    """
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def content_words(text: str) -> Set[str]:
    """Normalised tokens with stopwords removed, for overlap measurement."""
    return {t for t in normalize(text).split(" ") if t and t not in STOPWORDS}


def is_negated(text: str, phrase: str, *, window: int = 6) -> bool:
    """Whether a negation word sits just before a phrase in the same clause.

    This is the guard for the case that defeats both keyword matching and
    embedding similarity:

        Customers can return products within 30 days.
        Customers cannot return products after 30 days.

    Both contain "30 day" and their embeddings are very close, yet only one
    confirms the return window.

    The clause boundary matters. In "Returns are not free. The window is 30
    days." the negation belongs to the previous sentence and must not attach to
    the phrase.
    """
    normalized_phrase = normalize(phrase)
    if not normalized_phrase:
        return False

    first_token = normalized_phrase.split(" ")[0]

    for clause in re.split(r"[.!?;:,]|\bbut\b|\bhowever\b|\balthough\b", text):
        if not contains_phrase(clause, normalized_phrase):
            continue

        tokens = normalize(clause).split(" ")
        for start, token in enumerate(tokens):
            if token != first_token:
                continue
            preceding = tokens[max(0, start - window) : start]
            if any(word in NEGATION_TOKENS for word in preceding):
                return True
    return False


STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "you",
    "your",
    "we",
    "our",
    "they",
    "their",
    "can",
    "may",
    "will",
    "would",
    "should",
    "could",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "there",
    "here",
    "what",
    "which",
    "who",
    "when",
    "how",
    "please",
    "also",
    "any",
    "all",
    "within",
    "about",
}
