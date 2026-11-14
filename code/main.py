#!/usr/bin/env python3
"""
Multi-Modal Evidence Review — Main Entry Point

Runs every claim in an input CSV through the five-stage pipeline (see
``pipeline.py``) and writes the predictions CSV.

Usage:
  python main.py                                          # Process claims.csv → output.csv
  python main.py --input ../dataset/sample_claims.csv     # Process a specific file
  python main.py --output my_output.csv                   # Custom output path

Exit codes: 0 on success, 1 if the input is missing or OPENAI_API_KEY is unset.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the code directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CACHE_DIR, CLAIMS_CSV, OUTPUT_CSV, load_user_history  # noqa: E402
from llm_client import LLMClient, TokenTracker  # noqa: E402
from pipeline import load_claims, print_summary, process_claims, write_output  # noqa: E402

# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review — Damage Claim Verification")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(CLAIMS_CSV),
        help=f"Path to input claims CSV (default: {CLAIMS_CSV})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(OUTPUT_CSV),
        help=f"Path to output CSV (default: {OUTPUT_CSV})",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"❌ Input file not found: {input_path}", file=sys.stderr)
        return 1

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Multi-Modal Evidence Review — Claim Verification       ║")
    print("║  Pipeline: Responses Vision → Rules/Policies → Output  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Load data
    print(f"📂 Loading claims from: {input_path}")
    claims = load_claims(input_path)
    print(f"   Found {len(claims)} claims to process")

    print("📂 Loading user history...")
    user_history = load_user_history()
    print(f"   Loaded history for {len(user_history)} users")

    # Initialize LLM client
    tracker = TokenTracker()
    try:
        client = LLMClient(tracker=tracker)
    except OSError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    print(f"🤖 LLM client initialized (cache dir: {CACHE_DIR})\n")

    # Process all claims
    start_time = time.time()
    results = process_claims(claims, user_history, client)

    elapsed = time.time() - start_time

    # Write output
    write_output(results, output_path)

    # Print summary
    print_summary(tracker, elapsed, len(claims))
    return 0


if __name__ == "__main__":
    sys.exit(main())
