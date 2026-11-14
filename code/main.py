#!/usr/bin/env python3
"""
Multi-Modal Evidence Review — Main Entry Point

Runs every claim in an input CSV through the five-stage pipeline (see
``pipeline.py``) and writes the predictions CSV.

Usage:
  python main.py                                          # Process claims.csv → output.csv
  python main.py --input ../dataset/sample_claims.csv     # Process a specific file
  python main.py --output my_output.csv                   # Custom output path
  python main.py --check                                  # Validate inputs offline, no API calls
  python main.py --limit 3 --quiet                        # Smoke-test the first 3 claims

Exit codes: 0 on success, 1 on a usage/input problem (missing or malformed
input, missing images under --check, OPENAI_API_KEY unset), 130 on Ctrl-C.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the code directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (  # noqa: E402
    CACHE_DIR,
    CLAIMS_CSV,
    IMAGE_ANALYSIS_MODEL,
    OUTPUT_CSV,
    USER_HISTORY_CSV,
    load_user_history,
)
from llm_client import LLMClient, TokenTracker  # noqa: E402
from pipeline import (  # noqa: E402
    InputError,
    load_claims,
    preflight,
    print_preflight,
    print_summary,
    process_claims,
    set_verbose,
    write_output,
)

EPILOG = """\
examples:
  python code/main.py                          run every claim in dataset/claims.csv -> output.csv
  python code/main.py --check                  validate the CSV and image paths (no API key needed)
  python code/main.py -i dataset/sample_claims.csv -o /tmp/sample_out.csv
  python code/main.py --limit 3 --quiet        smoke-test the first 3 claims, one line per claim

environment:
  OPENAI_API_KEY      required for a real run (read from the shell or a .env file)
  *_MODEL, IMAGE_DETAIL, SYNTHESIS_REASONING_EFFORT   optional overrides, see .env.example

Model responses are cached under code/.cache/, so re-running the same input is
free and reproducible. Delete that folder to force fresh calls.
"""


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or greater")
    return number


def _display(path: Path) -> str:
    """Show a path relative to the working directory when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python code/main.py",
        description=(
            "Verify damage claims from images, the claim conversation and user history.\n"
            "Reads a claims CSV and writes one structured prediction row per claim."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=CLAIMS_CSV,
        metavar="CSV",
        help="claims CSV with user_id, image_paths, user_claim, claim_object (default: dataset/claims.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=OUTPUT_CSV,
        metavar="CSV",
        help="where to write predictions; parent folders are created (default: output.csv at the repo root)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=_positive_int,
        metavar="N",
        help="only process the first N claims (handy for a quick, cheap smoke test)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the input and image paths, print a summary and exit without calling the API",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="print one line per claim instead of per-stage progress",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.is_file():
        print(f"❌ Input file not found: {input_path}", file=sys.stderr)
        print("   Pass a claims CSV with --input, e.g. --input dataset/sample_claims.csv", file=sys.stderr)
        return 1

    try:
        claims = load_claims(input_path)
    except InputError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    if args.limit:
        claims = claims[: args.limit]

    user_history = load_user_history()

    if args.check:
        report = preflight(claims, user_history)
        print_preflight(report, Path(_display(input_path)))
        return 0 if report.ok else 1

    if not claims:
        print(f"❌ {input_path} has no claim rows to process.", file=sys.stderr)
        return 1

    tracker = TokenTracker()
    try:
        client = LLMClient(tracker=tracker)
    except OSError as e:
        print(f"❌ {e}", file=sys.stderr)
        print(
            "   Export it in your shell or copy .env.example to .env and fill it in.\n"
            "   Tip: `python code/main.py --check` validates your inputs without a key.",
            file=sys.stderr,
        )
        return 1

    print("Multi-Modal Evidence Review — claim verification")
    print("─" * 60)
    print(f"📂 Claims:       {_display(input_path)} ({len(claims)} to process)")
    print(f"📂 User history: {_display(USER_HISTORY_CSV)} ({len(user_history)} users)")
    print(f"📝 Output:       {_display(output_path)}")
    print(f"🤖 Vision model: {IMAGE_ANALYSIS_MODEL} (response cache: {_display(CACHE_DIR)})")

    set_verbose(not args.quiet)
    start_time = time.time()
    try:
        results = process_claims(claims, user_history, client)
    except KeyboardInterrupt:
        print("\n⏹  Interrupted. No output written; finished calls are cached and re-used next run.")
        return 130
    finally:
        set_verbose(True)
    elapsed = time.time() - start_time

    write_output(results, output_path)
    print_summary(tracker, elapsed, len(claims))
    return 0


if __name__ == "__main__":
    sys.exit(main())
