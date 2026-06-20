# Evaluation Report

**Date:** 2026-06-19 14:00:07
**Sample size:** 20 claims
**Runtime:** 298.6s (14.9s per claim)

## Strategy Comparison

| Strategy | claim_status | issue_type | severity | Notes |
|---|---:|---:|---:|---|
| Baseline Chat Completions + LLM-first synthesis | 80.0% | 45.0% | 45.0% | Prior report before adjudication layer |
| Responses API + structured outputs + deterministic adjudicator | 95.0% | 95.0% | 85.0% | Current run |

## Accuracy Metrics

| Metric | Score |
|---|---|
| claim_status | ✅ 95.0% |
| issue_type | ✅ 95.0% |
| object_part | ✅ 90.0% |
| evidence_standard_met | ✅ 100.0% |
| valid_image | ✅ 90.0% |
| severity | ✅ 85.0% |
| risk_flags_f1 | ⚠️ 79.1% |
| supporting_ids_overlap | ✅ 100.0% |

## Claim Status Confusion Matrix

| Expected \ Predicted | supported | contradicted | not_enough_information |
|---|---|---|---|
| **supported** | 13 | 0 | 0 |
| **contradicted** | 1 | 4 | 0 |
| **not_enough_information** | 0 | 0 | 2 |

## Per-Case Results

| # | User | Object | Rule | Status E/P | Issue E/P | Severity E/P | Risk E/P | Valid E/P |
|---|---|---|---|---|---|---|---|---|
| 1 | user_001 | car | claimed_damage_visible | supported / supported ✓ | dent / dent ✓ | medium / medium ✓ | none / none ✓ | true / true ✓ |
| 2 | user_002 | car | claimed_damage_visible | supported / supported ✓ | scratch / scratch ✓ | low / low ✓ | none / none ✓ | true / true ✓ |
| 3 | user_004 | car | claimed_damage_visible | supported / supported ✓ | crack / crack ✓ | medium / medium ✓ | none / none ✓ | true / true ✓ |
| 4 | user_007 | car | claimed_damage_visible | supported / supported ✓ | broken_part / broken_part ✓ | medium / medium ✓ | none / none ✓ | true / true ✓ |
| 5 | user_005 | car | claimed_damage_visible | contradicted / supported x | scratch / dent x | low / medium x | claim_mismatch;user_history_risk;manual_review_required / user_history_risk x | true / true ✓ |
| 6 | user_006 | car | claimed_part_not_visible | not_enough_information / not_enough_information ✓ | unknown / unknown ✓ | unknown / unknown ✓ | wrong_angle;damage_not_visible / cropped_or_obstructed;low_light_or_glare;manual_review_required;wrong_angle;wrong_object_part x | true / true ✓ |
| 7 | user_003 | car | claimed_damage_visible | supported / supported ✓ | dent / dent ✓ | medium / medium ✓ | blurry_image / none x | true / true ✓ |
| 8 | user_008 | car | llm_fallback | contradicted / contradicted ✓ | broken_part / broken_part ✓ | high / medium x | claim_mismatch;non_original_image;user_history_risk;manual_review_required / low_light_or_glare;manual_review_required;non_original_image;user_history_risk;wrong_angle x | false / true x |
| 9 | user_009 | laptop | claimed_damage_visible | supported / supported ✓ | crack / crack ✓ | medium / medium ✓ | none / none ✓ | true / true ✓ |
| 10 | user_010 | laptop | claimed_damage_visible | supported / supported ✓ | broken_part / broken_part ✓ | medium / medium ✓ | none / none ✓ | true / true ✓ |
| 11 | user_011 | laptop | claimed_damage_visible | supported / supported ✓ | stain / stain ✓ | medium / medium ✓ | none / none ✓ | true / true ✓ |
| 12 | user_012 | laptop | claimed_damage_visible | supported / supported ✓ | dent / dent ✓ | low / low ✓ | none / none ✓ | true / true ✓ |
| 13 | user_018 | laptop | claimed_damage_visible | supported / supported ✓ | crack / crack ✓ | medium / medium ✓ | none / none ✓ | true / true ✓ |
| 14 | user_020 | laptop | vague_laptop_physical_damage | contradicted / contradicted ✓ | none / none ✓ | none / none ✓ | damage_not_visible;user_history_risk;manual_review_required / cropped_or_obstructed;damage_not_visible;manual_review_required;non_original_image;user_history_risk;wrong_angle x | true / true ✓ |
| 15 | user_015 | package | claimed_damage_visible | supported / supported ✓ | crushed_packaging / crushed_packaging ✓ | medium / medium ✓ | none / none ✓ | true / true ✓ |
| 16 | user_030 | package | claimed_damage_visible | supported / supported ✓ | torn_packaging / torn_packaging ✓ | medium / medium ✓ | none / none ✓ | true / true ✓ |
| 17 | user_031 | package | claimed_damage_visible | supported / supported ✓ | water_damage / water_damage ✓ | medium / medium ✓ | user_history_risk;manual_review_required / user_history_risk x | true / true ✓ |
| 18 | user_032 | package | missing_contents_not_provable | not_enough_information / not_enough_information ✓ | unknown / unknown ✓ | unknown / unknown ✓ | cropped_or_obstructed;damage_not_visible;manual_review_required / manual_review_required;wrong_angle x | false / true x |
| 19 | user_033 | package | wrong_object | contradicted / contradicted ✓ | unknown / unknown ✓ | low / unknown x | wrong_object;claim_mismatch;user_history_risk;manual_review_required / claim_mismatch;cropped_or_obstructed;manual_review_required;user_history_risk;wrong_object x | true / true ✓ |
| 20 | user_034 | package | image_text_instruction | contradicted / contradicted ✓ | none / none ✓ | none / none ✓ | damage_not_visible;text_instruction_present;user_history_risk;manual_review_required / manual_review_required;non_original_image;text_instruction_present;user_history_risk x | true / true ✓ |

## Error Analysis

### Case 5 (user_005)
- **Expected:** contradicted / scratch
- **Predicted:** supported / dent
- **Justification:** The claimed rear_bumper is visible and image img_1 shows damage consistent with the claim.


## Failure Buckets

| Bucket | Cases |
|---|---|
| status_error | 5 |
| issue_type_error | 5 |
| severity_error | 5, 8, 19 |
| wrong_object_miss | none |
| adversarial_text_miss | none |
| part_mismatch_miss | 5, 8 |
| risk_flag_mismatch | 5, 6, 7, 8, 14, 17, 18, 19, 20 |

## Operational Analysis

### Model Calls
- Total API calls: 41
- Approximate calls per claim: 2.0

### Token Usage
- Input tokens: 71,585
- Output tokens: 15,454
- Reasoning tokens (o3): 9,467

### Model Breakdown

| Model | Calls | Cached | Input Tokens | Output Tokens |
|---|---|---|---|---|
| gpt-5.5 | 41 | 0 | 71,585 | 15,454 |

### Cost Estimate

| Model | Rate (input/output per 1M tokens) | Estimated Cost |
|---|---|---|
| gpt-5.5 | $5.00 / $15.00 | $0.5897 |
| **Total** | | **$0.5897** |

### Performance
- Total runtime: 298.6s
- Average per claim: 14.9s
- Images processed: 29

### Rate Limit Strategy
- Exponential backoff retry (up to 3 retries)
- File-based response caching (re-runs cost $0)
- Sequential processing (one claim at a time) to stay within TPM limits
- All images per claim sent in a single API call to minimize request count

## Final Test Run Operational Summary

This is the completed run on `dataset/claims.csv` that produced the final root-level `output.csv`.

| Item | Value |
|---|---:|
| Claims processed | 44 |
| Images processed | 82 |
| Total model calls | 89 |
| Total runtime | 799.4s |
| Average runtime per claim | 18.2s |
| Input tokens | 172,466 |
| Output tokens | 42,347 |
| Reasoning tokens | 26,537 |
| Estimated cost | about $1.50 |

The final `output.csv` has 44 rows, the required 14 columns in order, populated justifications/reasons, and zero allowed-value validation errors.
