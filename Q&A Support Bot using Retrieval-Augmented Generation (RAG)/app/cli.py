"""Terminal client for the support bot.

    python -m app.cli                       interactive session
    python -m app.cli "how do I return?"    one-shot, prints the answer
    python -m app.cli --local               no server needed
    echo "how do I return?" | python -m app.cli    one answer per input line

Two modes, because they have different costs.

By default this talks HTTP to a running server. That is the better mode: the
server already holds the embedding model in memory, so each question costs a
retrieval and a generation rather than a model load. It also exercises the real
request path, so what you see here is what an API caller gets.

`--local` skips the server and drives RAGService in-process. Convenient when you
do not want a second terminal, but it reloads the embedding model on every
invocation, which takes several seconds before the first answer.

Both modes validate the question through `AskRequest` first. A CLI that accepted
input the API would reject would teach you the wrong thing about the system.

Only the standard library is used for the HTTP path, so this stays runnable even
when the heavy dependencies are not installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List

DEFAULT_URL = "http://127.0.0.1:8000"
ASK_PATH = "/api/v1/ask"

PROMPT = "You: "
ANSWER_PREFIX = "Bot: "
INDENT = " " * len(ANSWER_PREFIX)

HELP_TEXT = """Commands:
  /sources     show or hide the retrieved chunks and their relevance scores
  /help        this message
  /quit        leave (Ctrl-D and Ctrl-C also work)

Anything else is sent to the bot as a question."""


class CliError(Exception):
    """A failure with a message worth showing the operator verbatim.

    Every raise site is expected to say what to do next, not just what broke.
    """


# --- transport -------------------------------------------------------------


def _ask_over_http(base_url: str, timeout: float) -> Callable[[str], Dict[str, Any]]:
    url = base_url.rstrip("/") + ASK_PATH

    def ask(question: str) -> Dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps({"question": question}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            raise CliError(_explain_http_error(exc)) from exc

        except urllib.error.URLError as exc:
            raise CliError(
                f"No server responded at {base_url} ({exc.reason}).\n"
                f"Start one with:  uvicorn app.main:app\n"
                f"Or skip the server entirely:  python -m app.cli --local"
            ) from exc

        except TimeoutError as exc:
            raise CliError(
                f"The server did not answer within {timeout:.0f}s. A local model on "
                f"a busy CPU can be slow; retry with --timeout 300."
            ) from exc

    return ask


def _explain_http_error(exc: urllib.error.HTTPError) -> str:
    """Turn a status code into the action that fixes it.

    The API deliberately returns a distinct code per failure, so the CLI can be
    specific here instead of printing "request failed".
    """
    detail = _read_detail(exc)

    if exc.code == 422:
        return f"The question was rejected by validation: {detail}"
    if exc.code == 503:
        return (
            "The knowledge base is unavailable, which usually means it was never "
            "built.\nRun:  python -m app.ingestion.ingest"
        )
    if exc.code == 502:
        return (
            "The local model is unreachable.\n"
            "Check it is installed and running:  ollama list\n"
            "Pull the configured model if it is missing:  ollama pull llama3.1"
        )
    return f"The server returned HTTP {exc.code}: {detail}"


def _read_detail(exc: urllib.error.HTTPError) -> str:
    """Best-effort extraction of FastAPI's error body."""
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except Exception:
        return exc.reason or "no detail"

    detail = body.get("detail", body)
    if isinstance(detail, list):
        # Pydantic validation errors: one entry per failing field.
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail)


def _ask_locally() -> Callable[[str], Dict[str, Any]]:
    """Drive the service directly, with no HTTP layer and no server.

    Imported lazily: these modules pull in the embedding stack, and the HTTP
    path must stay usable on a machine that has only the light dependency set.
    """
    from app.services.exceptions import (
        KnowledgeBaseUnavailableError,
        LLMUnavailableError,
    )
    from app.services.rag_service import RAGService
    from app.services.response_validator import validate_answer_payload

    print("Loading the embedding model, this takes a few seconds...", file=sys.stderr)
    service = RAGService()

    def ask(question: str) -> Dict[str, Any]:
        try:
            result = service.ask(question)
            validate_answer_payload(result)

        except KnowledgeBaseUnavailableError as exc:
            raise CliError(
                "The knowledge base is missing or empty.\n"
                "Run:  python -m app.ingestion.ingest"
            ) from exc

        except LLMUnavailableError as exc:
            raise CliError(
                "The local model is unreachable.\n"
                "Check it is installed and running:  ollama list"
            ) from exc

        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "request_id": "local",
            "insufficient_context": bool(result.get("insufficient_context", False)),
        }

    return ask


# --- input validation ------------------------------------------------------


def validate_question(question: str) -> str:
    """Apply the API's own contract before sending anything.

    Reusing `AskRequest` rather than re-stating the rules means the CLI cannot
    drift away from what the API accepts.
    """
    from pydantic import ValidationError

    from app.schemas import AskRequest

    try:
        return AskRequest(question=question).question
    except ValidationError as exc:
        reasons = "; ".join(str(err.get("msg", err)) for err in exc.errors())
        raise CliError(f"That question was not accepted: {reasons}") from exc


# --- rendering -------------------------------------------------------------


def render_answer(
    payload: Dict[str, Any],
    elapsed: float,
    show_sources: bool,
    stream: Any = sys.stdout,
) -> None:
    answer = payload.get("answer", "")
    sources: List[Dict[str, Any]] = payload.get("sources") or []

    print(f"{ANSWER_PREFIX}{answer}", file=stream)

    if payload.get("insufficient_context"):
        # Worth calling out explicitly. This is the bot declining on purpose,
        # not a failure, and the distinction is invisible in the answer text.
        print(
            f"{INDENT}(declined: nothing in the knowledge base cleared the "
            f"relevance threshold)",
            file=stream,
        )

    if show_sources and sources:
        print(f"{INDENT}sources:", file=stream)
        for source in sources:
            metadata = source.get("metadata") or {}
            name = metadata.get("source", "unknown")
            index = metadata.get("chunk_index", "?")
            score = source.get("relevance_score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            print(f"{INDENT}  {name} chunk {index}  score {score_text}", file=stream)

    request_id = payload.get("request_id", "unknown")
    print(f"{INDENT}[{elapsed:.1f}s, request {request_id}]", file=stream)


# --- session modes ---------------------------------------------------------


def answer_one(
    ask: Callable[[str], Dict[str, Any]],
    question: str,
    show_sources: bool,
    as_json: bool,
) -> None:
    started = time.perf_counter()
    payload = ask(validate_question(question))
    elapsed = time.perf_counter() - started

    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        render_answer(payload, elapsed, show_sources)


def run_interactive(
    ask: Callable[[str], Dict[str, Any]],
    show_sources: bool,
) -> int:
    print("Support bot. Ask a question, or /help for commands.\n")

    while True:
        try:
            raw = input(PROMPT)
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        question = raw.strip()
        if not question:
            continue

        lowered = question.lower()
        if lowered in {"/quit", "/exit", "quit", "exit"}:
            return 0
        if lowered == "/help":
            print(HELP_TEXT)
            continue
        if lowered == "/sources":
            show_sources = not show_sources
            print(f"sources {'shown' if show_sources else 'hidden'}")
            continue

        try:
            started = time.perf_counter()
            payload = ask(validate_question(question))
            elapsed = time.perf_counter() - started
            render_answer(payload, elapsed, show_sources)

        except CliError as exc:
            # Keep the session alive. A typo or a briefly unavailable model is
            # not a reason to make someone restart and lose their place.
            print(f"{ANSWER_PREFIX}{exc}", file=sys.stderr)

        except KeyboardInterrupt:
            print("\n(cancelled)")

        print()


def run_piped(
    ask: Callable[[str], Dict[str, Any]],
    show_sources: bool,
    as_json: bool,
) -> int:
    """One question per input line, so the bot can be scripted.

    Failures are reported per line and do not abort the batch: when you pipe
    twenty questions in, losing the last nineteen to one bad line is useless.
    """
    exit_code = 0

    for line in sys.stdin:
        question = line.strip()
        if not question:
            continue

        try:
            answer_one(ask, question, show_sources, as_json)
        except CliError as exc:
            print(f"error: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


# --- entrypoint ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Ask the support bot questions from a terminal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m app.cli\n"
            '  python -m app.cli "what is the return policy?"\n'
            '  python -m app.cli --sources "how long does delivery take?"\n'
            "  python -m app.cli --local\n"
        ),
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="ask one question and exit; omit for an interactive session",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"base URL of a running server (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="answer in-process instead of calling a server (slower to start)",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="show the retrieved chunks and their relevance scores",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print the raw API response instead of formatted output",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="seconds to wait for an answer (default: 180)",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        ask = _ask_locally() if args.local else _ask_over_http(args.url, args.timeout)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        if args.question:
            answer_one(ask, " ".join(args.question), args.sources, args.as_json)
            return 0

        if not sys.stdin.isatty():
            return run_piped(ask, args.sources, args.as_json)

        return run_interactive(ask, args.sources)

    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
