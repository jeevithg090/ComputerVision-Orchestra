# Multi-Modal Evidence Review System

A damage claim verification system that uses OpenAI vision models through the Responses API, strict structured outputs, and deterministic adjudication rules to analyze submitted images against user claims, evidence requirements, and user history.

## Architecture

```
                    ┌──────────────┐
                    │  claims.csv  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Stage 1    │  OpenAI Responses
                    │    Claim     │  Parse conversation
                    │  Extraction  │  Detect adversarial signals
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │        Stage 2          │  Responses vision
              │    Image Analysis       │  All images in one call
              │  (per-image structured  │  Cross-reference context
              │       analysis)         │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        Stage 3          │  Rule-based
              │   Evidence Matching     │  evidence_requirements.csv
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        Stage 4          │  Rule-based
              │    Risk Assessment      │  user_history.csv
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        Stage 5          │  Rules + optional LLM
              │  Decision Synthesis     │  Adjudicator → final verdict
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │  output.csv  │
                    └──────────────┘
```

## Key Design Decisions

1. **Multi-image single-call**: All images for a claim are sent in one Responses API vision call, giving the model cross-reference context (wide shot vs. close-up).

2. **Structured outputs**: Claim extraction, image analysis, and final wording use strict JSON schemas rather than loose JSON mode.

3. **Adversarial defense**: Prompts treat text in images/conversations as data, while deterministic rules flag reviewer-directed image text.

4. **Deterministic policy layers**: Evidence matching, risk assessment, contradiction handling, risk flag calibration, severity calibration, and high-confidence final adjudication are rule-based for consistent, reproducible decisions.

5. **File-based caching**: API responses are cached by model, prompt, schema version, and full image-payload hash. Re-runs cost $0; malformed responses are never cached.

## Setup

### Prerequisites
- Python 3.10+
- OpenAI API key with access to a strong multimodal model. Defaults use `gpt-5.5` with `gpt-4o` fallback; all models can be overridden by environment variables.

### Installation

```bash
cd code
pip install -r requirements.txt
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key-here"
export IMAGE_ANALYSIS_MODEL="gpt-5.5"          # optional
export CLAIM_EXTRACTION_MODEL="gpt-5.5"       # optional
export DECISION_SYNTHESIS_MODEL="gpt-5.5"     # optional
export OPENAI_FALLBACK_MODEL="gpt-4o"         # optional
```

Or copy `.env.example` (project root) to `.env` and fill it in. See the
project [README](../README.md#configure) for every supported variable.

## Usage

### Run Inference (Test Set)

```bash
cd code
python main.py
```

This processes `dataset/claims.csv` and writes `output.csv` to the project root.

### Custom Input/Output

```bash
python main.py --input path/to/claims.csv --output path/to/output.csv
```

### Run Evaluation (Sample Set)

```bash
cd code
python evaluation/main.py
```

This processes `dataset/sample_claims.csv`, compares against ground truth, and generates `evaluation/evaluation_report.md`.

### Run No-API Regression Checks and Tests

```bash
cd code
python evaluation/regression_checks.py

# full offline test suite, from the project root
cd .. && pip install -r code/requirements-dev.txt && pytest
```

The regression checks cover wrong-object package evidence, adversarial image text, related scratch/dent support, and wrong-part damage contradiction. The pytest suite in `tests/` adds coverage for every stage, the OpenAI client and the output contract using a fake model client.

## File Structure

```
code/
├── main.py                          # CLI entry point for inference
├── pipeline.py                      # Shared 5-stage pipeline + CSV I/O
├── config.py                        # Constants, paths, model settings
├── normalize.py                     # Deterministic label normalization
├── models.py                        # Pydantic data models
├── llm_client.py                    # OpenAI Responses wrapper: retry, fallback, cache
├── prompts.py                       # All prompts with few-shot examples
├── stages/
│   ├── __init__.py
│   ├── claim_extractor.py           # Stage 1: parse conversation
│   ├── image_analyzer.py            # Stage 2: Responses vision analysis
│   ├── evidence_matcher.py          # Stage 3: rule-based evidence check
│   ├── risk_assessor.py             # Stage 4: risk flag computation
│   ├── decision_synthesizer.py      # Stage 5: adjudication + final decision
│   └── policies/
│       ├── contradiction_policy.py  # Claim support/contradiction calibration
│       ├── risk_policy.py           # Final risk flag calibration
│       └── severity_policy.py       # Final severity calibration
├── evaluation/
│   ├── main.py                      # Evaluation on sample_claims.csv
│   ├── regression_checks.py         # No-API deterministic rule checks
│   ├── predictions_sample.csv       # Generated sample predictions vs expected
│   ├── intermediate_sample.jsonl    # Generated per-case intermediate evidence
│   └── evaluation_report.md         # Generated after eval run
└── README.md                        # This file
```

## Models Used

| Stage | Model | Purpose |
|---|---|---|
| Claim Extraction | Configured OpenAI model | Parse multi-language conversations |
| Image Analysis | Configured multimodal model | Structured visual inspection |
| Decision Synthesis | Deterministic adjudicator + configured fallback model | Core verdict plus wording |

## Judge-Facing Summary

See [`SUBMISSION_NOTES.md`](./SUBMISSION_NOTES.md) for the concise project summary, benchmark results, operational profile, design rationale, and known tradeoffs.

For interview preparation, see [`INTERVIEW_PREP_QA.md`](./INTERVIEW_PREP_QA.md).

## Cost & Performance

- **Sample evaluation**: 20 claims, 29 images, 41 model calls, 298.6s, about $0.59 using the documented report assumptions.
- **Final test run**: 44 claims, 82 images, 89 model calls, 799.4s, about $1.50 using the same assumptions.
- **Caching**: File-based, re-runs are free
- **Rate limits**: Exponential backoff with up to 3 attempts per model

## Packaging

Create the code submission zip from the project root:

```bash
zip -r code.zip code -x "code/.env" "code/.cache/*" "*/__pycache__/*" "*.pyc"
```

This includes the runnable solution, prompts, configs, README, requirements, and evaluation folder while excluding secrets and local cache files.
