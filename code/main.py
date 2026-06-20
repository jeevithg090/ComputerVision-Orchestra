#!/usr/bin/env python3
"""
Multi-Modal Evidence Review — Main Entry Point

Processes damage claims through a 5-stage pipeline:
  Stage 1: Claim Extraction (Responses API structured output)
  Stage 2: Image Analysis (Responses API vision)
  Stage 3: Evidence Matching (rule-based)
  Stage 4: Risk Assessment (rule-based)
  Stage 5: Decision Synthesis (deterministic policies + LLM fallback)

Usage:
  python main.py                                          # Process claims.csv → output.csv
  python main.py --input dataset/sample_claims.csv        # Process a specific file
  python main.py --output my_output.csv                   # Custom output path
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the code directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CLAIMS_CSV,
    OUTPUT_CSV,
    PROJECT_ROOT,
    DATASET_DIR,
    load_user_history,
)
from models import ClaimInput, ClaimResult
from llm_client import LLMClient, TokenTracker

# Pipeline stages
from stages.claim_extractor import extract_claim
from stages.image_analyzer import analyze_images
from stages.evidence_matcher import match_evidence
from stages.risk_assessor import assess_risk
from stages.decision_synthesizer import synthesize_decision


# ── Output CSV Column Order ─────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


# ── Pipeline ────────────────────────────────────────────────────────────────

def process_claim(
    claim: ClaimInput,
    user_history: Dict[str, dict],
    client: LLMClient,
    claim_index: int,
    total_claims: int,
) -> ClaimResult:
    """Process a single claim through all 5 pipeline stages."""

    prefix = f"[{claim_index + 1}/{total_claims}] {claim.user_id}"

    # Stage 1: Claim Extraction
    print(f"  {prefix} → Stage 1: Extracting claim...")
    extraction = extract_claim(claim, client)
    print(f"    Claim: {extraction.claim_summary[:80]}...")

    # Stage 2: Image Analysis
    num_images = len(claim.image_path_list)
    print(f"  {prefix} → Stage 2: Analyzing {num_images} image(s)...")
    image_analysis = analyze_images(claim, extraction, client)
    for img in image_analysis.images:
        status = "✓ damage" if img.damage_visible else "✗ no damage"
        print(f"    {img.image_id}: {img.object_visible}/{img.part_visible} [{status}]")

    # Stage 3: Evidence Matching
    print(f"  {prefix} → Stage 3: Checking evidence standards...")
    evidence_check = match_evidence(claim, extraction, image_analysis)
    met = "✓ met" if evidence_check.evidence_standard_met else "✗ not met"
    print(f"    Evidence standard: {met}")

    # Stage 4: Risk Assessment
    print(f"  {prefix} → Stage 4: Assessing risk...")
    risk_profile = assess_risk(claim, extraction, image_analysis)
    print(f"    Risk flags: {', '.join(risk_profile.risk_flags)}")

    # Stage 5: Decision Synthesis
    user_data = user_history.get(claim.user_id)
    print(f"  {prefix} → Stage 5: Synthesizing decision (policies + fallback)...")
    decision = synthesize_decision(
        claim, extraction, image_analysis,
        evidence_check, risk_profile,
        user_data, client,
    )
    print(f"    ═══ Decision: {decision.claim_status.upper()} "
          f"(severity={decision.severity}) ═══")

    return ClaimResult(
        input=claim,
        extraction=extraction,
        image_analysis=image_analysis,
        evidence_check=evidence_check,
        risk_profile=risk_profile,
        decision=decision,
    )


def load_claims(csv_path: Path) -> List[ClaimInput]:
    """Load claims from a CSV file."""
    claims = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append(ClaimInput(
                user_id=row["user_id"].strip(),
                image_paths=row["image_paths"].strip(),
                user_claim=row["user_claim"].strip(),
                claim_object=row["claim_object"].strip(),
            ))
    return claims


def write_output(results: List[ClaimResult], output_path: Path):
    """Write results to output CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for result in results:
            d = result.decision
            writer.writerow({
                "user_id": result.input.user_id,
                "image_paths": result.input.image_paths,
                "user_claim": result.input.user_claim,
                "claim_object": result.input.claim_object,
                "evidence_standard_met": str(d.evidence_standard_met).lower(),
                "evidence_standard_met_reason": d.evidence_standard_met_reason,
                "risk_flags": d.risk_flags,
                "issue_type": d.issue_type,
                "object_part": d.object_part,
                "claim_status": d.claim_status,
                "claim_status_justification": d.claim_status_justification,
                "supporting_image_ids": d.supporting_image_ids,
                "valid_image": str(d.valid_image).lower(),
                "severity": d.severity,
            })
    print(f"\n✅ Output written to: {output_path}")


def print_summary(tracker: TokenTracker, elapsed: float, num_claims: int):
    """Print operational summary."""
    summary = tracker.summary()
    print("\n" + "=" * 60)
    print("OPERATIONAL SUMMARY")
    print("=" * 60)
    print(f"  Claims processed: {num_claims}")
    print(f"  Total API calls:  {summary['total_calls']}")
    print(f"  Total time:       {elapsed:.1f}s ({elapsed/num_claims:.1f}s per claim)")
    print(f"\n  Token Usage:")
    print(f"    Input tokens:     {summary['total_input_tokens']:,}")
    print(f"    Output tokens:    {summary['total_output_tokens']:,}")
    print(f"    Reasoning tokens: {summary['total_reasoning_tokens']:,}")
    print(f"\n  Model Breakdown:")
    for model, stats in summary["model_breakdown"].items():
        cached = f" ({stats['cached']} cached)" if stats["cached"] else ""
        print(f"    {model}: {stats['calls']} calls{cached}, "
              f"{stats['input']:,} in / {stats['output']:,} out")
    print("=" * 60)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Evidence Review — Damage Claim Verification"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(CLAIMS_CSV),
        help=f"Path to input claims CSV (default: {CLAIMS_CSV})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(OUTPUT_CSV),
        help=f"Path to output CSV (default: {OUTPUT_CSV})",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Multi-Modal Evidence Review — Claim Verification       ║")
    print("║  Pipeline: Responses Vision → Rules/Policies → Output  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Load data
    print(f"📂 Loading claims from: {input_path}")
    claims = load_claims(input_path)
    print(f"   Found {len(claims)} claims to process")

    print(f"📂 Loading user history...")
    user_history = load_user_history()
    print(f"   Loaded history for {len(user_history)} users")

    # Initialize LLM client
    tracker = TokenTracker()
    client = LLMClient(tracker=tracker)

    from config import CACHE_DIR
    print(f"🤖 LLM client initialized (cache dir: {CACHE_DIR})\n")

    # Process all claims
    start_time = time.time()
    results: List[ClaimResult] = []

    for i, claim in enumerate(claims):
        print(f"\n{'─' * 60}")
        try:
            result = process_claim(claim, user_history, client, i, len(claims))
            results.append(result)
        except Exception as e:
            print(f"  ❌ Error processing {claim.user_id}: {e}")
            import traceback
            traceback.print_exc()
            # Create a fallback result
            from models import (
                ClaimExtraction, ImageAnalysisResult, EvidenceCheck,
                RiskProfile, ClaimDecision,
            )
            fallback = ClaimResult(
                input=claim,
                extraction=ClaimExtraction(
                    claimed_damage_type="unknown",
                    claimed_object_part="unknown",
                    claim_summary="Error during processing",
                    adversarial_signals=[],
                ),
                image_analysis=ImageAnalysisResult(images=[]),
                evidence_check=EvidenceCheck(
                    evidence_standard_met=False,
                    evidence_standard_met_reason="Processing error",
                ),
                risk_profile=RiskProfile(risk_flags=["manual_review_required"]),
                decision=ClaimDecision(
                    evidence_standard_met=False,
                    evidence_standard_met_reason="Processing error",
                    risk_flags="manual_review_required",
                    issue_type="unknown",
                    object_part="unknown",
                    claim_status="not_enough_information",
                    claim_status_justification="Claim could not be processed automatically.",
                    supporting_image_ids="none",
                    valid_image=False,
                    severity="unknown",
                ),
            )
            results.append(fallback)

    elapsed = time.time() - start_time

    # Write output
    write_output(results, output_path)

    # Print summary
    print_summary(tracker, elapsed, len(claims))

    return results


if __name__ == "__main__":
    main()
