#!/usr/bin/env python3
"""
Evaluation Module — Run pipeline on sample_claims.csv and compare against ground truth.

Metrics computed:
  - Per-field exact match accuracy (claim_status, issue_type, object_part, etc.)
  - Multi-label F1 for risk_flags
  - Set overlap for supporting_image_ids
  - Confusion matrix for claim_status
  - Detailed error analysis

Usage:
  python -m evaluation.main
  python evaluation/main.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the code directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_CLAIMS_CSV, PROJECT_ROOT, load_user_history
from models import ClaimInput, ClaimResult
from llm_client import LLMClient, TokenTracker
from stages.claim_extractor import extract_claim
from stages.image_analyzer import analyze_images
from stages.evidence_matcher import match_evidence
from stages.risk_assessor import assess_risk
from stages.decision_synthesizer import synthesize_decision


# ── Load Ground Truth ───────────────────────────────────────────────────────

def load_sample_claims_with_labels(csv_path: Path) -> List[Tuple[ClaimInput, dict]]:
    """Load sample_claims.csv which has both input AND expected output columns."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claim = ClaimInput(
                user_id=row["user_id"].strip(),
                image_paths=row["image_paths"].strip(),
                user_claim=row["user_claim"].strip(),
                claim_object=row["claim_object"].strip(),
            )
            expected = {
                "evidence_standard_met": row.get("evidence_standard_met", "").strip().lower(),
                "evidence_standard_met_reason": row.get("evidence_standard_met_reason", "").strip(),
                "risk_flags": row.get("risk_flags", "none").strip(),
                "issue_type": row.get("issue_type", "").strip(),
                "object_part": row.get("object_part", "").strip(),
                "claim_status": row.get("claim_status", "").strip(),
                "claim_status_justification": row.get("claim_status_justification", "").strip(),
                "supporting_image_ids": row.get("supporting_image_ids", "").strip(),
                "valid_image": row.get("valid_image", "").strip().lower(),
                "severity": row.get("severity", "").strip(),
            }
            rows.append((claim, expected))
    return rows


# ── Metrics ─────────────────────────────────────────────────────────────────

def exact_match_accuracy(predicted: List[str], expected: List[str]) -> float:
    """Compute exact match accuracy."""
    if not expected:
        return 0.0
    correct = sum(1 for p, e in zip(predicted, expected) if p == e)
    return correct / len(expected)


def multi_label_f1(predicted_flags: List[str], expected_flags: List[str]) -> float:
    """Compute average F1 for multi-label risk flags (semicolon-separated)."""
    f1_scores = []
    for pred, exp in zip(predicted_flags, expected_flags):
        pred_set = set(f.strip() for f in pred.split(";") if f.strip())
        exp_set = set(f.strip() for f in exp.split(";") if f.strip())

        if not pred_set and not exp_set:
            f1_scores.append(1.0)
            continue

        tp = len(pred_set & exp_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(exp_set) if exp_set else 0.0

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def set_overlap_score(predicted_ids: List[str], expected_ids: List[str]) -> float:
    """Compute average Jaccard similarity for supporting_image_ids."""
    scores = []
    for pred, exp in zip(predicted_ids, expected_ids):
        pred_set = set(f.strip() for f in pred.split(";") if f.strip())
        exp_set = set(f.strip() for f in exp.split(";") if f.strip())

        if not pred_set and not exp_set:
            scores.append(1.0)
        elif pred_set == {"none"} and exp_set == {"none"}:
            scores.append(1.0)
        elif not pred_set or not exp_set:
            scores.append(0.0)
        else:
            intersection = len(pred_set & exp_set)
            union = len(pred_set | exp_set)
            scores.append(intersection / union if union else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def confusion_matrix(predicted: List[str], expected: List[str], labels: List[str]) -> Dict:
    """Build a confusion matrix dict."""
    matrix = {e: {p: 0 for p in labels} for e in labels}
    for p, e in zip(predicted, expected):
        if e in matrix and p in matrix[e]:
            matrix[e][p] += 1
    return matrix


def _split_set(value: str) -> set[str]:
    return {v.strip() for v in value.split(";") if v.strip() and v.strip() != "none"}


def build_failure_buckets(results: List[ClaimResult], expected_list: List[dict]) -> Dict[str, List[int]]:
    """Group failures by likely cause so calibration work is visible."""
    buckets: Dict[str, List[int]] = defaultdict(list)
    for idx, (result, expected) in enumerate(zip(results, expected_list), start=1):
        d = result.decision
        pred_flags = _split_set(d.risk_flags)
        exp_flags = _split_set(expected["risk_flags"])

        if d.claim_status != expected["claim_status"]:
            buckets["status_error"].append(idx)
        if d.issue_type != expected["issue_type"]:
            buckets["issue_type_error"].append(idx)
        if d.severity != expected["severity"]:
            buckets["severity_error"].append(idx)
        if "wrong_object" in exp_flags and "wrong_object" not in pred_flags:
            buckets["wrong_object_miss"].append(idx)
        if "text_instruction_present" in exp_flags and "text_instruction_present" not in pred_flags:
            buckets["adversarial_text_miss"].append(idx)
        if (
            {"claim_mismatch", "wrong_object_part"} & exp_flags and
            not ({"claim_mismatch", "wrong_object_part"} & pred_flags)
        ):
            buckets["part_mismatch_miss"].append(idx)
        if pred_flags != exp_flags:
            buckets["risk_flag_mismatch"].append(idx)

    return buckets


# ── Evaluation Report ───────────────────────────────────────────────────────

def generate_report(
    results: List[ClaimResult],
    expected_list: List[dict],
    tracker: TokenTracker,
    elapsed: float,
    output_path: Path,
):
    """Generate the evaluation report markdown file."""
    n = len(results)

    # Collect predicted values
    pred = {
        "claim_status": [],
        "issue_type": [],
        "object_part": [],
        "evidence_standard_met": [],
        "valid_image": [],
        "severity": [],
        "risk_flags": [],
        "supporting_image_ids": [],
    }
    exp = {k: [] for k in pred}

    for result, expected in zip(results, expected_list):
        d = result.decision
        pred["claim_status"].append(d.claim_status)
        pred["issue_type"].append(d.issue_type)
        pred["object_part"].append(d.object_part)
        pred["evidence_standard_met"].append(str(d.evidence_standard_met).lower())
        pred["valid_image"].append(str(d.valid_image).lower())
        pred["severity"].append(d.severity)
        pred["risk_flags"].append(d.risk_flags)
        pred["supporting_image_ids"].append(d.supporting_image_ids)

        exp["claim_status"].append(expected["claim_status"])
        exp["issue_type"].append(expected["issue_type"])
        exp["object_part"].append(expected["object_part"])
        exp["evidence_standard_met"].append(expected["evidence_standard_met"])
        exp["valid_image"].append(expected["valid_image"])
        exp["severity"].append(expected["severity"])
        exp["risk_flags"].append(expected["risk_flags"])
        exp["supporting_image_ids"].append(expected["supporting_image_ids"])

    # Compute metrics
    metrics = {}
    for field in ["claim_status", "issue_type", "object_part",
                   "evidence_standard_met", "valid_image", "severity"]:
        metrics[field] = exact_match_accuracy(pred[field], exp[field])

    metrics["risk_flags_f1"] = multi_label_f1(pred["risk_flags"], exp["risk_flags"])
    metrics["supporting_ids_overlap"] = set_overlap_score(
        pred["supporting_image_ids"], exp["supporting_image_ids"]
    )

    # Confusion matrix for claim_status
    status_labels = ["supported", "contradicted", "not_enough_information"]
    cm = confusion_matrix(pred["claim_status"], exp["claim_status"], status_labels)
    failure_buckets = build_failure_buckets(results, expected_list)

    # Token summary
    token_summary = tracker.summary()

    # ── Build Report ────────────────────────────────────────────────────

    lines = [
        "# Evaluation Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Sample size:** {n} claims",
        f"**Runtime:** {elapsed:.1f}s ({elapsed/n:.1f}s per claim)",
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | claim_status | issue_type | severity | Notes |",
        "|---|---:|---:|---:|---|",
        "| Baseline Chat Completions + LLM-first synthesis | 80.0% | 45.0% | 45.0% | Prior report before adjudication layer |",
        (
            f"| Responses API + structured outputs + deterministic adjudicator | "
            f"{metrics.get('claim_status', 0):.1%} | {metrics.get('issue_type', 0):.1%} | "
            f"{metrics.get('severity', 0):.1%} | Current run |"
        ),
        "",
        "## Accuracy Metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
    ]

    for field, score in metrics.items():
        emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        lines.append(f"| {field} | {emoji} {score:.1%} |")

    lines.extend([
        "",
        "## Claim Status Confusion Matrix",
        "",
        "| Expected \\ Predicted | supported | contradicted | not_enough_information |",
        "|---|---|---|---|",
    ])
    for e_label in status_labels:
        row_vals = [str(cm[e_label][p_label]) for p_label in status_labels]
        lines.append(f"| **{e_label}** | {' | '.join(row_vals)} |")

    # Per-case detail
    lines.extend([
        "",
        "## Per-Case Results",
        "",
        "| # | User | Object | Rule | Status E/P | Issue E/P | Severity E/P | Risk E/P | Valid E/P |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
    for i, (result, expected) in enumerate(zip(results, expected_list)):
        d = result.decision
        status_match = "✓" if d.claim_status == expected["claim_status"] else "x"
        issue_match = "✓" if d.issue_type == expected["issue_type"] else "x"
        severity_match = "✓" if d.severity == expected["severity"] else "x"
        risk_match = "✓" if d.risk_flags == expected["risk_flags"] else "x"
        valid_pred = str(d.valid_image).lower()
        valid_match = "✓" if valid_pred == expected["valid_image"] else "x"
        lines.append(
            f"| {i+1} | {result.input.user_id} | {result.input.claim_object} | {d.rule_id} | "
            f"{expected['claim_status']} / {d.claim_status} {status_match} | "
            f"{expected['issue_type']} / {d.issue_type} {issue_match} | "
            f"{expected['severity']} / {d.severity} {severity_match} | "
            f"{expected['risk_flags']} / {d.risk_flags} {risk_match} | "
            f"{expected['valid_image']} / {valid_pred} {valid_match} |"
        )

    # Error analysis
    errors = []
    for i, (result, expected) in enumerate(zip(results, expected_list)):
        d = result.decision
        if d.claim_status != expected["claim_status"]:
            errors.append({
                "case": i + 1,
                "user_id": result.input.user_id,
                "expected_status": expected["claim_status"],
                "predicted_status": d.claim_status,
                "expected_issue": expected["issue_type"],
                "predicted_issue": d.issue_type,
                "justification": d.claim_status_justification[:150],
            })

    if errors:
        lines.extend([
            "",
            "## Error Analysis",
            "",
        ])
        for err in errors:
            lines.extend([
                f"### Case {err['case']} ({err['user_id']})",
                f"- **Expected:** {err['expected_status']} / {err['expected_issue']}",
                f"- **Predicted:** {err['predicted_status']} / {err['predicted_issue']}",
                f"- **Justification:** {err['justification']}",
                "",
            ])

    lines.extend([
        "",
        "## Failure Buckets",
        "",
        "| Bucket | Cases |",
        "|---|---|",
    ])
    for bucket in [
        "status_error",
        "issue_type_error",
        "severity_error",
        "wrong_object_miss",
        "adversarial_text_miss",
        "part_mismatch_miss",
        "risk_flag_mismatch",
    ]:
        cases = failure_buckets.get(bucket, [])
        lines.append(f"| {bucket} | {', '.join(map(str, cases)) if cases else 'none'} |")

    # Operational analysis
    lines.extend([
        "",
        "## Operational Analysis",
        "",
        "### Model Calls",
        f"- Total API calls: {token_summary['total_calls']}",
        f"- Approximate calls per claim: {token_summary['total_calls'] / n:.1f}",
        "",
        "### Token Usage",
        f"- Input tokens: {token_summary['total_input_tokens']:,}",
        f"- Output tokens: {token_summary['total_output_tokens']:,}",
        f"- Reasoning tokens (o3): {token_summary['total_reasoning_tokens']:,}",
        "",
        "### Model Breakdown",
        "",
        "| Model | Calls | Cached | Input Tokens | Output Tokens |",
        "|---|---|---|---|---|",
    ])
    for model, stats in token_summary["model_breakdown"].items():
        lines.append(
            f"| {model} | {stats['calls']} | {stats['cached']} | "
            f"{stats['input']:,} | {stats['output']:,} |"
        )

    lines.extend([
        "",
        "### Cost Estimate",
        "",
        "| Model | Rate (input/output per 1M tokens) | Estimated Cost |",
        "|---|---|---|",
    ])

    # Cost estimation
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.0},
        "gpt-4o-mini": {"input": 0.150, "output": 0.60},
        "o3": {"input": 10.0, "output": 40.0},
    }
    total_cost = 0.0
    for model, stats in token_summary["model_breakdown"].items():
        rates = pricing.get(model, {"input": 5.0, "output": 15.0})
        cost = (stats["input"] / 1_000_000 * rates["input"] +
                stats["output"] / 1_000_000 * rates["output"])
        total_cost += cost
        lines.append(
            f"| {model} | ${rates['input']:.2f} / ${rates['output']:.2f} | ${cost:.4f} |"
        )
    lines.append(f"| **Total** | | **${total_cost:.4f}** |")

    lines.extend([
        "",
        "### Performance",
        f"- Total runtime: {elapsed:.1f}s",
        f"- Average per claim: {elapsed/n:.1f}s",
        f"- Images processed: {sum(len(r.image_analysis.images) for r in results)}",
        "",
        "### Rate Limit Strategy",
        "- Exponential backoff retry (up to 3 retries)",
        "- File-based response caching (re-runs cost $0)",
        "- Sequential processing (one claim at a time) to stay within TPM limits",
        "- All images per claim sent in a single API call to minimize request count",
    ])

    # Write report
    report_content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    write_sample_predictions(results, expected_list, output_path.parent / "predictions_sample.csv")
    write_intermediates(results, expected_list, output_path.parent / "intermediate_sample.jsonl")

    print(f"\n📊 Evaluation report written to: {output_path}")
    return metrics


def write_sample_predictions(
    results: List[ClaimResult],
    expected_list: List[dict],
    output_path: Path,
) -> None:
    """Write one row per sample with predicted and expected final fields."""
    fields = [
        "case",
        "user_id",
        "claim_object",
        "rule_id",
        "pred_claim_status",
        "expected_claim_status",
        "pred_issue_type",
        "expected_issue_type",
        "pred_object_part",
        "expected_object_part",
        "pred_severity",
        "expected_severity",
        "pred_risk_flags",
        "expected_risk_flags",
        "pred_valid_image",
        "expected_valid_image",
        "pred_evidence_standard_met",
        "expected_evidence_standard_met",
        "pred_supporting_image_ids",
        "expected_supporting_image_ids",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, (result, expected) in enumerate(zip(results, expected_list), start=1):
            d = result.decision
            writer.writerow({
                "case": idx,
                "user_id": result.input.user_id,
                "claim_object": result.input.claim_object,
                "rule_id": d.rule_id,
                "pred_claim_status": d.claim_status,
                "expected_claim_status": expected["claim_status"],
                "pred_issue_type": d.issue_type,
                "expected_issue_type": expected["issue_type"],
                "pred_object_part": d.object_part,
                "expected_object_part": expected["object_part"],
                "pred_severity": d.severity,
                "expected_severity": expected["severity"],
                "pred_risk_flags": d.risk_flags,
                "expected_risk_flags": expected["risk_flags"],
                "pred_valid_image": str(d.valid_image).lower(),
                "expected_valid_image": expected["valid_image"],
                "pred_evidence_standard_met": str(d.evidence_standard_met).lower(),
                "expected_evidence_standard_met": expected["evidence_standard_met"],
                "pred_supporting_image_ids": d.supporting_image_ids,
                "expected_supporting_image_ids": expected["supporting_image_ids"],
            })


def write_intermediates(
    results: List[ClaimResult],
    expected_list: List[dict],
    output_path: Path,
) -> None:
    """Write all intermediate stage outputs for diagnosis."""
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (result, expected) in enumerate(zip(results, expected_list), start=1):
            payload = {
                "case": idx,
                "expected": expected,
                "input": result.input.model_dump(),
                "extraction": result.extraction.model_dump(),
                "image_analysis": result.image_analysis.model_dump(),
                "evidence_check": result.evidence_check.model_dump(),
                "risk_profile": result.risk_profile.model_dump(),
                "decision": result.decision.model_dump(),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Multi-Modal Evidence Review — Evaluation Mode          ║")
    print("║  Running pipeline on sample_claims.csv with ground truth║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Load sample claims with labels
    sample_data = load_sample_claims_with_labels(SAMPLE_CLAIMS_CSV)
    claims = [s[0] for s in sample_data]
    expected = [s[1] for s in sample_data]
    print(f"📂 Loaded {len(claims)} sample claims with ground truth")

    # Load user history
    user_history = load_user_history()
    print(f"📂 Loaded history for {len(user_history)} users")

    # Initialize LLM client
    tracker = TokenTracker()
    client = LLMClient(tracker=tracker)
    print(f"🤖 LLM client initialized\n")

    # Process all sample claims
    start_time = time.time()
    results: List[ClaimResult] = []

    for i, claim in enumerate(claims):
        print(f"\n{'─' * 60}")
        try:
            from main import process_claim
            result = process_claim(claim, user_history, client, i, len(claims))
            results.append(result)
        except Exception as e:
            print(f"  ❌ Error processing {claim.user_id}: {e}")
            import traceback
            traceback.print_exc()
            # Create fallback
            from models import (
                ClaimExtraction, ImageAnalysisResult, EvidenceCheck,
                RiskProfile, ClaimDecision,
            )
            fallback = ClaimResult(
                input=claim,
                extraction=ClaimExtraction(
                    claimed_damage_type="unknown",
                    claimed_object_part="unknown",
                    claim_summary="Error",
                ),
                image_analysis=ImageAnalysisResult(images=[]),
                evidence_check=EvidenceCheck(
                    evidence_standard_met=False,
                    evidence_standard_met_reason="Error",
                ),
                risk_profile=RiskProfile(risk_flags=["manual_review_required"]),
                decision=ClaimDecision(
                    evidence_standard_met=False,
                    evidence_standard_met_reason="Error",
                    risk_flags="manual_review_required",
                    issue_type="unknown",
                    object_part="unknown",
                    claim_status="not_enough_information",
                    claim_status_justification="Error during processing.",
                    supporting_image_ids="none",
                    valid_image=False,
                    severity="unknown",
                ),
            )
            results.append(fallback)

    elapsed = time.time() - start_time

    # Generate evaluation report
    report_path = Path(__file__).resolve().parent / "evaluation_report.md"
    metrics = generate_report(results, expected, tracker, elapsed, report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for field, score in metrics.items():
        emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        print(f"  {emoji} {field}: {score:.1%}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
