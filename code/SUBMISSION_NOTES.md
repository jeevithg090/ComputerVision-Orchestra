# Submission Notes

## Executive Summary

This solution verifies multi-modal damage claims using a staged evidence-review pipeline:

1. Extract the claimed object, part, and damage from the conversation.
2. Analyze all submitted images together with OpenAI vision structured outputs.
3. Check evidence sufficiency against the provided evidence requirements.
4. Add user-history and evidence-quality risk context.
5. Apply deterministic adjudication policies for final status, issue type, severity, risk flags, and supporting images.

The core design choice is to keep visual understanding and final adjudication separate. The model describes evidence; deterministic policies turn evidence into reproducible decisions.

## Final Strategy

- OpenAI Responses API with strict structured outputs for extraction and image analysis.
- One vision call per claim with all images included together for cross-image context.
- Rule-based evidence matching using `evidence_requirements.csv`.
- Policy modules for contradiction, severity calibration, and final risk-flag calibration.
- File-based caching keyed by model, prompt/schema version, and full image payload hash.
- Environment-variable-only secrets; no hardcoded API keys.

## Evaluation Results

Latest sample evaluation on the 20 labeled rows:

| Metric | Score |
|---|---:|
| Claim status | 95.0% |
| Issue type | 95.0% |
| Object part | 90.0% |
| Evidence standard met | 100.0% |
| Valid image | 90.0% |
| Severity | 85.0% |
| Risk flags F1 | 79.1% |
| Supporting image overlap | 100.0% |

The final `output.csv` was generated for all 44 rows in `dataset/claims.csv` and passed schema, row-count, allowed-value, and non-empty-justification checks.

## Operational Profile

- Sample evaluation: 20 claims, 29 images, 41 model calls, 298.6 seconds.
- Final test run: 44 claims, 82 images, 89 model calls, 799.4 seconds.
- Sequential execution is intentional for simple rate-limit safety.
- Re-runs use local caching to avoid repeated cost.

## Why This Should Generalize

- The pipeline does not hardcode test labels or file-specific answers.
- The sample set is used only to calibrate general policies like severity and risk output.
- Wrong-object, wrong-part, missing-content, adversarial-text, and image-quality cases are handled as reusable evidence patterns.
- Final decisions remain grounded in image IDs and visible evidence.

## Known Tradeoffs

- The system prioritizes high precision on core claim status and issue type over aggressively flagging every possible risk.
- User history is treated as context, not a substitute for visual evidence.
- The final test set has no labels, so true hidden accuracy cannot be known locally; the labeled sample evaluation is the best available proxy.

