"""
Shared claim-processing pipeline used by both the inference CLI (``main.py``)
and the evaluation harness (``evaluation/main.py``).

  Stage 1: Claim Extraction    (Responses API structured output)
  Stage 2: Image Analysis      (Responses API vision)
  Stage 3: Evidence Matching   (rule-based)
  Stage 4: Risk Assessment     (rule-based)
  Stage 5: Decision Synthesis  (deterministic policies + LLM fallback)
"""

from __future__ import annotations

import csv
import traceback
from collections.abc import Callable
from pathlib import Path

from llm_client import LLMClient, TokenTracker
from models import (
    ClaimDecision,
    ClaimExtraction,
    ClaimInput,
    ClaimResult,
    EvidenceCheck,
    ImageAnalysisResult,
    RiskProfile,
)
from stages.claim_extractor import extract_claim
from stages.decision_synthesizer import synthesize_decision
from stages.evidence_matcher import match_evidence
from stages.image_analyzer import analyze_images
from stages.risk_assessor import assess_risk

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
    user_history: dict[str, dict],
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
        claim,
        extraction,
        image_analysis,
        evidence_check,
        risk_profile,
        user_data,
        client,
    )
    print(f"    ═══ Decision: {decision.claim_status.upper()} (severity={decision.severity}) ═══")

    return ClaimResult(
        input=claim,
        extraction=extraction,
        image_analysis=image_analysis,
        evidence_check=evidence_check,
        risk_profile=risk_profile,
        decision=decision,
    )


def fallback_result(claim: ClaimInput, reason: str = "Processing error") -> ClaimResult:
    """Conservative result for a claim the pipeline could not process."""
    return ClaimResult(
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
            evidence_standard_met_reason=reason,
        ),
        risk_profile=RiskProfile(risk_flags=["manual_review_required"]),
        decision=ClaimDecision(
            evidence_standard_met=False,
            evidence_standard_met_reason=reason,
            risk_flags="manual_review_required",
            issue_type="unknown",
            object_part="unknown",
            claim_status="not_enough_information",
            claim_status_justification="Claim could not be processed automatically.",
            supporting_image_ids="none",
            valid_image=False,
            severity="unknown",
            rule_id="processing_error",
        ),
    )


def process_claims(
    claims: list[ClaimInput],
    user_history: dict[str, dict],
    client: LLMClient,
    process: Callable[..., ClaimResult] | None = None,
) -> list[ClaimResult]:
    """Process claims sequentially; a failing claim yields a fallback row, never a crash."""
    process = process or process_claim
    results: list[ClaimResult] = []
    for i, claim in enumerate(claims):
        print(f"\n{'─' * 60}")
        try:
            results.append(process(claim, user_history, client, i, len(claims)))
        except Exception as e:  # noqa: BLE001 - one bad claim must not abort the batch
            print(f"  ❌ Error processing {claim.user_id}: {e}")
            traceback.print_exc()
            results.append(fallback_result(claim))
    return results


def load_claims(csv_path: Path) -> list[ClaimInput]:
    """Load claims from a CSV file (extra columns, e.g. labels, are ignored)."""
    claims = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append(
                ClaimInput(
                    user_id=row["user_id"].strip(),
                    image_paths=row["image_paths"].strip(),
                    user_claim=row["user_claim"].strip(),
                    claim_object=row["claim_object"].strip(),
                )
            )
    return claims


def write_output(results: list[ClaimResult], output_path: Path):
    """Write results to output CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for result in results:
            d = result.decision
            writer.writerow(
                {
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
                }
            )
    print(f"\n✅ Output written to: {output_path}")


def print_summary(tracker: TokenTracker, elapsed: float, num_claims: int):
    """Print operational summary."""
    summary = tracker.summary()
    print("\n" + "=" * 60)
    print("OPERATIONAL SUMMARY")
    print("=" * 60)
    print(f"  Claims processed: {num_claims}")
    print(f"  Total API calls:  {summary['total_calls']}")
    per_claim = elapsed / num_claims if num_claims else 0.0
    print(f"  Total time:       {elapsed:.1f}s ({per_claim:.1f}s per claim)")
    print("\n  Token Usage:")
    print(f"    Input tokens:     {summary['total_input_tokens']:,}")
    print(f"    Output tokens:    {summary['total_output_tokens']:,}")
    print(f"    Reasoning tokens: {summary['total_reasoning_tokens']:,}")
    print("\n  Model Breakdown:")
    for model, stats in summary["model_breakdown"].items():
        cached = f" ({stats['cached']} cached)" if stats["cached"] else ""
        print(f"    {model}: {stats['calls']} calls{cached}, {stats['input']:,} in / {stats['output']:,} out")
    print("=" * 60)
