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
import time
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

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
from stages.image_analyzer import analyze_images, resolve_images
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


REQUIRED_INPUT_COLUMNS = ("user_id", "image_paths", "user_claim", "claim_object")
ALLOWED_CLAIM_OBJECTS = ("car", "laptop", "package")


class InputError(ValueError):
    """The input CSV is unreadable or does not match the expected schema."""


# ── Console output ──────────────────────────────────────────────────────────

_verbose = True


def set_verbose(verbose: bool) -> None:
    """Toggle per-stage progress lines (``--quiet`` turns them off)."""
    global _verbose
    _verbose = verbose


def _detail(message: str) -> None:
    if _verbose:
        print(message)


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
    _detail(f"  {prefix} → Stage 1: Extracting claim...")
    extraction = extract_claim(claim, client)
    _detail(f"    Claim: {extraction.claim_summary[:80]}...")

    # Stage 2: Image Analysis
    num_images = len(claim.image_path_list)
    _detail(f"  {prefix} → Stage 2: Analyzing {num_images} image(s)...")
    image_analysis = analyze_images(claim, extraction, client)
    for img in image_analysis.images:
        status = "✓ damage" if img.damage_visible else "✗ no damage"
        _detail(f"    {img.image_id}: {img.object_visible}/{img.part_visible} [{status}]")

    # Stage 3: Evidence Matching
    _detail(f"  {prefix} → Stage 3: Checking evidence standards...")
    evidence_check = match_evidence(claim, extraction, image_analysis)
    met = "✓ met" if evidence_check.evidence_standard_met else "✗ not met"
    _detail(f"    Evidence standard: {met}")

    # Stage 4: Risk Assessment
    _detail(f"  {prefix} → Stage 4: Assessing risk...")
    risk_profile = assess_risk(claim, extraction, image_analysis)
    _detail(f"    Risk flags: {', '.join(risk_profile.risk_flags)}")

    # Stage 5: Decision Synthesis
    user_data = user_history.get(claim.user_id)
    _detail(f"  {prefix} → Stage 5: Synthesizing decision (policies + fallback)...")
    decision = synthesize_decision(
        claim,
        extraction,
        image_analysis,
        evidence_check,
        risk_profile,
        user_data,
        client,
    )
    _detail(f"    ═══ Decision: {decision.claim_status.upper()} (severity={decision.severity}) ═══")

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
    total = len(claims)
    for i, claim in enumerate(claims):
        _detail(f"\n{'─' * 60}")
        started = time.time()
        try:
            result = process(claim, user_history, client, i, total)
        except Exception as e:  # noqa: BLE001 - one bad claim must not abort the batch
            print(f"  ❌ [{i + 1}/{total}] {claim.user_id}: {type(e).__name__}: {e}")
            print("     → wrote a conservative not_enough_information row; see traceback below")
            traceback.print_exc()
            result = fallback_result(claim)
        if not _verbose:
            d = result.decision
            print(
                f"  [{i + 1:>{len(str(total))}}/{total}] {claim.user_id:<10} {claim.claim_object:<8} "
                f"→ {d.claim_status:<22} {d.issue_type}/{d.object_part} "
                f"severity={d.severity} ({time.time() - started:.1f}s)"
            )
        results.append(result)
    return results


def load_claims(csv_path: Path) -> list[ClaimInput]:
    """
    Load claims from a CSV file (extra columns, e.g. labels, are ignored).

    Raises ``InputError`` with a human-readable message when the file is not
    valid UTF-8 CSV, is missing a required column, or a row has an invalid
    ``claim_object``.
    """
    claims = []
    try:
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in columns]
            if missing:
                raise InputError(
                    f"{csv_path} is missing required column(s): {', '.join(missing)}. "
                    f"Expected a header with: {', '.join(REQUIRED_INPUT_COLUMNS)}."
                )
            # Line 1 is the header, so the first data row is line 2.
            for line_no, row in enumerate(reader, start=2):
                try:
                    claims.append(
                        ClaimInput(
                            user_id=(row["user_id"] or "").strip(),
                            image_paths=(row["image_paths"] or "").strip(),
                            user_claim=(row["user_claim"] or "").strip(),
                            claim_object=(row["claim_object"] or "").strip(),
                        )
                    )
                except ValidationError:
                    value = (row.get("claim_object") or "").strip()
                    raise InputError(
                        f"{csv_path}, line {line_no}: claim_object must be one of "
                        f"{', '.join(ALLOWED_CLAIM_OBJECTS)} (got {value!r})."
                    ) from None
    except UnicodeDecodeError as e:
        raise InputError(f"{csv_path} is not valid UTF-8 text: {e}") from None
    return claims


# ── Preflight ───────────────────────────────────────────────────────────────


@dataclass
class PreflightReport:
    """Offline sanity check of an input file (no API calls)."""

    num_claims: int = 0
    num_images: int = 0
    objects: Counter = field(default_factory=Counter)
    missing_images: list[tuple[str, str]] = field(default_factory=list)  # (user_id, image_id)
    rows_without_images: list[str] = field(default_factory=list)
    users_without_history: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.num_claims > 0 and not self.missing_images and not self.rows_without_images


def preflight(claims: list[ClaimInput], user_history: dict[str, dict]) -> PreflightReport:
    """Check that every referenced image exists and summarise the input."""
    report = PreflightReport(num_claims=len(claims))
    for claim in claims:
        report.objects[claim.claim_object] += 1
        report.num_images += len(claim.image_path_list)
        if not claim.image_path_list:
            report.rows_without_images.append(claim.user_id)
        _present, missing = resolve_images(claim.image_ids, claim.image_path_list, warn=False)
        report.missing_images.extend((claim.user_id, image_id) for image_id in missing)
        if claim.user_id not in user_history:
            report.users_without_history.append(claim.user_id)
    return report


def print_preflight(report: PreflightReport, input_path: Path) -> None:
    """Human-readable preflight summary."""
    objects = ", ".join(f"{name}={count}" for name, count in sorted(report.objects.items())) or "none"
    print(f"Input:   {input_path}")
    print(f"Claims:  {report.num_claims} ({objects})")
    print(f"Images:  {report.num_images} referenced, {len(report.missing_images)} missing")
    if report.users_without_history:
        print(
            f"History: {len(report.users_without_history)} user(s) have no row in user_history.csv "
            f"(no history-based risk flags for them): {', '.join(report.users_without_history)}"
        )
    else:
        print("History: every user has a row in user_history.csv")
    for user_id, image_id in report.missing_images:
        print(f"  ✗ {user_id}: image {image_id} not found under dataset/")
    for user_id in report.rows_without_images:
        print(f"  ✗ {user_id}: row has no image_paths")
    if report.num_claims == 0:
        print("  ✗ input file has a header but no claim rows")
    print("\n✅ Preflight passed" if report.ok else "\n❌ Preflight found problems (see above)")


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
    print(f"\n✅ Wrote {len(results)} rows to: {output_path}")


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
