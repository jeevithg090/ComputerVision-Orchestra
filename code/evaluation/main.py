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
  python code/evaluation/main.py                    # full run on dataset/sample_claims.csv
  python code/evaluation/main.py --limit 5 --quiet  # quick, cheap smoke run
  python code/evaluation/main.py --report-dir /tmp/eval

Writes evaluation_report.md, predictions_sample.csv and intermediate_sample.jsonl
into the report directory (default: code/evaluation/).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure the code directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_CLAIMS_CSV, load_user_history  # noqa: E402
from llm_client import LLMClient, TokenTracker  # noqa: E402
from models import ClaimInput, ClaimResult  # noqa: E402
from pipeline import process_claims, set_verbose  # noqa: E402

# ── Load Ground Truth ───────────────────────────────────────────────────────


def load_sample_claims_with_labels(csv_path: Path) -> list[tuple[ClaimInput, dict]]:
    """Load sample_claims.csv which has both input AND expected output columns."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
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


def exact_match_accuracy(predicted: list[str], expected: list[str]) -> float:
    """Compute exact match accuracy."""
    if not expected:
        return 0.0
    correct = sum(1 for p, e in zip(predicted, expected, strict=False) if p == e)
    return correct / len(expected)


def multi_label_f1(predicted_flags: list[str], expected_flags: list[str]) -> float:
    """Compute average F1 for multi-label risk flags (semicolon-separated)."""
    f1_scores = []
    for pred, exp in zip(predicted_flags, expected_flags, strict=False):
        pred_set = {f.strip() for f in pred.split(";") if f.strip()}
        exp_set = {f.strip() for f in exp.split(";") if f.strip()}

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


def set_overlap_score(predicted_ids: list[str], expected_ids: list[str]) -> float:
    """Compute average Jaccard similarity for supporting_image_ids."""
    scores = []
    for pred, exp in zip(predicted_ids, expected_ids, strict=False):
        pred_set = {f.strip() for f in pred.split(";") if f.strip()}
        exp_set = {f.strip() for f in exp.split(";") if f.strip()}

        if not pred_set and not exp_set or pred_set == {"none"} and exp_set == {"none"}:
            scores.append(1.0)
        elif not pred_set or not exp_set:
            scores.append(0.0)
        else:
            intersection = len(pred_set & exp_set)
            union = len(pred_set | exp_set)
            scores.append(intersection / union if union else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def confusion_matrix(predicted: list[str], expected: list[str], labels: list[str]) -> dict:
    """Build a confusion matrix dict."""
    matrix = {e: dict.fromkeys(labels, 0) for e in labels}
    for p, e in zip(predicted, expected, strict=False):
        if e in matrix and p in matrix[e]:
            matrix[e][p] += 1
    return matrix


def _split_set(value: str) -> set[str]:
    return {v.strip() for v in value.split(";") if v.strip() and v.strip() != "none"}


def build_failure_buckets(results: list[ClaimResult], expected_list: list[dict]) -> dict[str, list[int]]:
    """Group failures by likely cause so calibration work is visible."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, (result, expected) in enumerate(zip(results, expected_list, strict=False), start=1):
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
        if {"claim_mismatch", "wrong_object_part"} & exp_flags and not (
            {"claim_mismatch", "wrong_object_part"} & pred_flags
        ):
            buckets["part_mismatch_miss"].append(idx)
        if pred_flags != exp_flags:
            buckets["risk_flag_mismatch"].append(idx)

    return buckets


# ── Evaluation Report ───────────────────────────────────────────────────────


def generate_report(
    results: list[ClaimResult],
    expected_list: list[dict],
    tracker: TokenTracker,
    elapsed: float,
    output_path: Path,
):
    """Generate the evaluation report markdown file."""
    n = len(results)
    per_claim = elapsed / n if n else 0.0

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

    for result, expected in zip(results, expected_list, strict=False):
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
    for field in [
        "claim_status",
        "issue_type",
        "object_part",
        "evidence_standard_met",
        "valid_image",
        "severity",
    ]:
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
        f"**Runtime:** {elapsed:.1f}s ({per_claim:.1f}s per claim)",
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

    lines.extend(
        [
            "",
            "## Claim Status Confusion Matrix",
            "",
            "| Expected \\ Predicted | supported | contradicted | not_enough_information |",
            "|---|---|---|---|",
        ]
    )
    for e_label in status_labels:
        row_vals = [str(cm[e_label][p_label]) for p_label in status_labels]
        lines.append(f"| **{e_label}** | {' | '.join(row_vals)} |")

    # Per-case detail
    lines.extend(
        [
            "",
            "## Per-Case Results",
            "",
            "| # | User | Object | Rule | Status E/P | Issue E/P | Severity E/P | Risk E/P | Valid E/P |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for i, (result, expected) in enumerate(zip(results, expected_list, strict=False)):
        d = result.decision
        status_match = "✓" if d.claim_status == expected["claim_status"] else "x"
        issue_match = "✓" if d.issue_type == expected["issue_type"] else "x"
        severity_match = "✓" if d.severity == expected["severity"] else "x"
        risk_match = "✓" if d.risk_flags == expected["risk_flags"] else "x"
        valid_pred = str(d.valid_image).lower()
        valid_match = "✓" if valid_pred == expected["valid_image"] else "x"
        lines.append(
            f"| {i + 1} | {result.input.user_id} | {result.input.claim_object} | {d.rule_id} | "
            f"{expected['claim_status']} / {d.claim_status} {status_match} | "
            f"{expected['issue_type']} / {d.issue_type} {issue_match} | "
            f"{expected['severity']} / {d.severity} {severity_match} | "
            f"{expected['risk_flags']} / {d.risk_flags} {risk_match} | "
            f"{expected['valid_image']} / {valid_pred} {valid_match} |"
        )

    # Error analysis
    errors = []
    for i, (result, expected) in enumerate(zip(results, expected_list, strict=False)):
        d = result.decision
        if d.claim_status != expected["claim_status"]:
            errors.append(
                {
                    "case": i + 1,
                    "user_id": result.input.user_id,
                    "expected_status": expected["claim_status"],
                    "predicted_status": d.claim_status,
                    "expected_issue": expected["issue_type"],
                    "predicted_issue": d.issue_type,
                    "justification": d.claim_status_justification[:150],
                }
            )

    if errors:
        lines.extend(
            [
                "",
                "## Error Analysis",
                "",
            ]
        )
        for err in errors:
            lines.extend(
                [
                    f"### Case {err['case']} ({err['user_id']})",
                    f"- **Expected:** {err['expected_status']} / {err['expected_issue']}",
                    f"- **Predicted:** {err['predicted_status']} / {err['predicted_issue']}",
                    f"- **Justification:** {err['justification']}",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "## Failure Buckets",
            "",
            "| Bucket | Cases |",
            "|---|---|",
        ]
    )
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
    lines.extend(
        [
            "",
            "## Operational Analysis",
            "",
            "### Model Calls",
            f"- Total API calls: {token_summary['total_calls']}",
            f"- Approximate calls per claim: {token_summary['total_calls'] / n if n else 0.0:.1f}",
            "",
            "### Token Usage",
            f"- Input tokens: {token_summary['total_input_tokens']:,}",
            f"- Output tokens: {token_summary['total_output_tokens']:,}",
            f"- Reasoning tokens: {token_summary['total_reasoning_tokens']:,}",
            "",
            "### Model Breakdown",
            "",
            "| Model | Calls | Cached | Input Tokens | Output Tokens |",
            "|---|---|---|---|---|",
        ]
    )
    for model, stats in token_summary["model_breakdown"].items():
        lines.append(
            f"| {model} | {stats['calls']} | {stats['cached']} | {stats['input']:,} | {stats['output']:,} |"
        )

    lines.extend(
        [
            "",
            "### Cost Estimate",
            "",
            "| Model | Rate (input/output per 1M tokens) | Estimated Cost |",
            "|---|---|---|",
        ]
    )

    # Cost estimation
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.0},
        "gpt-4o-mini": {"input": 0.150, "output": 0.60},
        "o3": {"input": 10.0, "output": 40.0},
    }
    total_cost = 0.0
    for model, stats in token_summary["model_breakdown"].items():
        rates = pricing.get(model, {"input": 5.0, "output": 15.0})
        cost = stats["input"] / 1_000_000 * rates["input"] + stats["output"] / 1_000_000 * rates["output"]
        total_cost += cost
        lines.append(f"| {model} | ${rates['input']:.2f} / ${rates['output']:.2f} | ${cost:.4f} |")
    lines.append(f"| **Total** | | **${total_cost:.4f}** |")

    lines.extend(
        [
            "",
            "### Performance",
            f"- Total runtime: {elapsed:.1f}s",
            f"- Average per claim: {per_claim:.1f}s",
            f"- Images processed: {sum(len(r.image_analysis.images) for r in results)}",
            "",
            "### Rate Limit Strategy",
            "- Exponential backoff retry (up to 3 retries)",
            "- File-based response caching (re-runs cost $0)",
            "- Sequential processing (one claim at a time) to stay within TPM limits",
            "- All images per claim sent in a single API call to minimize request count",
        ]
    )

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
    results: list[ClaimResult],
    expected_list: list[dict],
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
        for idx, (result, expected) in enumerate(zip(results, expected_list, strict=False), start=1):
            d = result.decision
            writer.writerow(
                {
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
                }
            )


def write_intermediates(
    results: list[ClaimResult],
    expected_list: list[dict],
    output_path: Path,
) -> None:
    """Write all intermediate stage outputs for diagnosis."""
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (result, expected) in enumerate(zip(results, expected_list, strict=False), start=1):
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


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or greater")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python code/evaluation/main.py",
        description=(
            "Run the pipeline on a labelled sample CSV and score every output field "
            "against the ground truth (accuracy, risk-flag F1, image-ID overlap, confusion matrix)."
        ),
        epilog=(
            "Requires OPENAI_API_KEY. Responses are cached under code/.cache/, "
            "so a repeat run is fast and free."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=SAMPLE_CLAIMS_CSV,
        metavar="CSV",
        help="labelled claims CSV (default: dataset/sample_claims.csv)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        metavar="DIR",
        help="folder for the report, predictions and intermediates (default: code/evaluation/)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=_positive_int,
        metavar="N",
        help="only evaluate the first N rows",
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

    if not args.input.is_file():
        print(f"❌ Labelled input not found: {args.input}", file=sys.stderr)
        return 1

    print("Multi-Modal Evidence Review — evaluation")
    print("─" * 60)

    # Load sample claims with labels
    try:
        sample_data = load_sample_claims_with_labels(args.input)
    except (KeyError, ValueError) as e:
        print(f"❌ Could not read {args.input}: {e}", file=sys.stderr)
        return 1
    if args.limit:
        sample_data = sample_data[: args.limit]
    claims = [s[0] for s in sample_data]
    expected = [s[1] for s in sample_data]
    print(f"📂 Loaded {len(claims)} labelled claims from {args.input}")

    # Load user history
    user_history = load_user_history()
    print(f"📂 Loaded history for {len(user_history)} users")

    # Initialize LLM client
    tracker = TokenTracker()
    try:
        client = LLMClient(tracker=tracker)
    except OSError as e:
        print(f"❌ {e}", file=sys.stderr)
        print("   Export it in your shell or copy .env.example to .env and fill it in.", file=sys.stderr)
        return 1
    print("🤖 LLM client initialized")

    # Process all sample claims
    set_verbose(not args.quiet)
    start_time = time.time()
    try:
        results: list[ClaimResult] = process_claims(claims, user_history, client)
    except KeyboardInterrupt:
        print("\n⏹  Interrupted. No report written; finished calls are cached and re-used next run.")
        return 130
    finally:
        set_verbose(True)
    elapsed = time.time() - start_time

    # Generate evaluation report
    report_path = args.report_dir / "evaluation_report.md"
    metrics = generate_report(results, expected, tracker, elapsed, report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for metric, score in metrics.items():
        emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        print(f"  {emoji} {metric}: {score:.1%}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
