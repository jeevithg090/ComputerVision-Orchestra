# Interview Prep: Multi-Modal Evidence Review

## 1. What did you build?

I built a multi-modal damage-claim verification system. It reads claim conversations, submitted images, user history, and evidence requirements, then outputs structured decisions for each claim: whether evidence supports the claim, contradicts it, or is insufficient.

The system produces the exact required `output.csv` schema and includes an evaluation workflow on the labeled sample set.

## 2. What is the core architecture?

The pipeline has five stages:

1. **Claim extraction**: Extract claimed object part, damage type, summary, language, and adversarial signals from the conversation.
2. **Image analysis**: Analyze all images for a claim together using OpenAI vision structured outputs.
3. **Evidence matching**: Check whether the submitted image set is sufficient to evaluate the claimed object/part/damage.
4. **Risk assessment**: Add user-history, image-quality, mismatch, authenticity, and adversarial-text risk signals.
5. **Decision synthesis**: Apply deterministic policies for contradiction, severity, risk flags, and final decision.

The key architectural choice is that the model describes visual evidence, but deterministic policies make the final calibrated decision.

## 3. Why separate visual analysis from adjudication?

Because image models are good at describing what is visible, but final claim decisions need consistency. For example, a model may say “damage visible,” but the system still needs to decide whether that damage is on the claimed part, whether it matches the claimed issue, and whether image quality is sufficient.

Separating these responsibilities makes the system more reproducible, debuggable, and easier to calibrate.

## 4. What OpenAI API pattern did you use?

I used the OpenAI Responses API with strict structured outputs. Local images are sent as base64 image inputs, and all images for one claim are included in the same vision call.

Structured outputs are important because the evaluator requires exact allowed values and exact CSV columns. The strict schema reduces parsing failures and keeps downstream policy code stable.

## 5. Why send all images for a claim in one vision call?

Many claims include a wide shot plus a close-up, or one image that supports the claim and another that contradicts or clarifies it. Sending all claim images together lets the vision model compare them and assign image-level roles like supporting, contradicting, mismatch, or insufficient.

It also reduces request count compared with one call per image.

## 6. What were your sample evaluation results?

Latest labeled sample evaluation:

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

The final unlabeled test run processed all 44 claims and produced a valid `output.csv`.

## 7. Can you claim 95% accuracy on the hidden test set?

No. The 44-row test set has no ground truth labels in the repo, so true hidden-test accuracy cannot be computed locally.

The 95% figure applies to `claim_status` on the labeled sample set. For the final test set, I can validate schema, row count, allowed values, and output completeness, but not true accuracy.

## 8. How did you avoid hardcoding answers?

I did not hardcode test-row predictions or file-specific answers. The sample set was used to calibrate general policies, such as how to treat wrong-object evidence, missing contents, adversarial text, and severity labels.

The same policies apply to all rows in `claims.csv`.

## 9. What are the most important policy modules?

The main policy modules are:

- `contradiction_policy.py`: decides when visible evidence supports or contradicts the claim.
- `severity_policy.py`: calibrates final severity from status, issue type, object, part, and visual severity.
- `risk_policy.py`: decides which internal risk signals should appear in final output.

This keeps final adjudication clean instead of burying many rules inside prompts.

## 10. How do you handle wrong-object evidence?

If the submitted image clearly shows a different object than the claimed category, the system marks the claim as contradicted, uses `issue_type=unknown`, `object_part=unknown`, and adds risk flags such as `wrong_object`, `claim_mismatch`, and `manual_review_required` when appropriate.

Example: a package claim with a food can or retail product container is treated as wrong-object evidence.

## 11. How do you handle wrong-part evidence?

If damage is visible but on a different part than the claimed part, the system treats that as a claim mismatch. For example, if the user claims rear bumper damage but the image shows damage on the quarter panel, the claim is contradicted or flagged as mismatched depending on whether the claimed part is visible.

## 12. How do you handle missing-content package claims?

Missing contents are difficult to prove from images alone because the system does not know what should have been in the package. The policy generally returns `not_enough_information` unless the image clearly proves presence or absence in a way that resolves the claim.

This avoids falsely contradicting a missing-item claim just because the item is not visible.

## 13. How do you handle adversarial text in images?

The image prompt treats all image text as evidence, not instructions. If an image contains reviewer-directed text like “approve this claim,” the model records it as instructional text, and the deterministic policy adds `text_instruction_present`.

When the instruction is physically attached to claim evidence, the system can treat it as contradictory or manual-review-worthy evidence.

## 14. How do you use user history?

User history is risk context, not the primary source of truth. Clear visual evidence can still support a claim from a risky user. History can add flags like `user_history_risk` or `manual_review_required`, but it should not override visual evidence by itself.

## 15. Why not let the LLM make the whole final decision?

The baseline LLM-first synthesis had lower issue-type and severity accuracy. The final version uses deterministic policies for high-confidence decisions, with the LLM only as a fallback. This improved consistency and sample metrics.

## 16. What improved from the baseline?

The baseline sample report was:

- `claim_status`: 80%
- `issue_type`: 45%
- `severity`: 45%

The final version reached:

- `claim_status`: 95%
- `issue_type`: 95%
- `severity`: 85%

The improvement came from structured outputs, richer per-image evidence fields, deterministic adjudication, and calibrated risk/severity policies.

## 17. What was the hardest case type?

The hardest cases were mismatches where some damage is visible, but it does not fully match the claim. For example, a damaged car part may be visible, but the user claimed a different part or exaggerated the type of damage.

These cases required explicit wrong-part and claim-mismatch policies.

## 18. What is the biggest remaining weakness?

Risk flags are still the hardest field because they are multi-label and sometimes subjective. The system intentionally prioritizes core claim status and issue type over aggressively flagging every possible risk.

The risk policy suppresses non-blocking quality flags when a separate clear image supports the claim, but there is still room to tune risk-flag precision further.

## 19. How did you validate the final output?

I validated:

- `output.csv` has 44 rows, one for each test claim.
- Required columns are present in exact order.
- All values are in allowed sets.
- Justifications and evidence reasons are non-empty.
- No secrets or cache files are included in `code.zip`.

I also ran no-API regression checks for deterministic policy behavior.

## 20. What operational analysis did you include?

Sample evaluation:

- 20 claims
- 29 images
- 41 model calls
- 298.6 seconds
- About $0.59 using report assumptions

Final test run:

- 44 claims
- 82 images
- 89 model calls
- 799.4 seconds
- About $1.50 using the same assumptions

The system uses sequential processing, retries, and file-based caching.

## 21. Why sequential processing instead of parallel processing?

Sequential processing is slower, but it is simpler and safer for a hackathon submission. It avoids rate-limit complexity, makes logs easier to follow, and reduces the chance of partial failures.

If production throughput mattered, I would add bounded concurrency with retry-aware rate limiting.

## 22. How does caching work?

Responses are cached by a deterministic key that includes the model, prompt/schema version, and full image payload hash. Re-running the same evaluation or inference avoids repeated API cost.

The cache is excluded from the final zip because it is local runtime state, not source code.

## 23. How are secrets handled?

Secrets are read only from environment variables or a local `.env` file. The API key is never hardcoded. `.env` and cache files are excluded from `code.zip`.

## 24. What would you improve next with more time?

I would add:

- Bounded parallelism for faster inference.
- More systematic calibration for risk flags.
- A small local dashboard or HTML report for reviewing per-case evidence.
- Confidence scores for final decisions.
- More regression scenarios generated from sample failure patterns.

I would avoid making hidden-test-specific changes without labels.

## 25. How would you explain the project in one minute?

This project is a structured evidence-review system for damage claims. It uses OpenAI vision models to describe what is visible in all submitted images, then deterministic policies decide whether the evidence supports, contradicts, or is insufficient for the claim. The design separates perception from adjudication, which made the system more accurate and reproducible. On the labeled sample set, claim status reached 95%, issue type 95%, severity 85%, and supporting image selection 100%. The final output for all 44 test claims is schema-valid and ready for evaluation.

