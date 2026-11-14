# ComputerVision-Orchestra — Multi-Modal Evidence Review

Automated review of damage claims for **cars**, **laptops** and **packages**. For each claim, the system reads the customer-support conversation, inspects every submitted photo with a vision model, checks the photos against minimum evidence requirements and the user's claim history, and decides whether the images **support**, **contradict** or **do not provide enough information** for the claim, with grounded justifications, risk flags and a severity estimate.

Built for the HackerRank Orchestrate (June 2026) "Multi-Modal Evidence Review" challenge. The full task specification is in [`problem_statement.md`](./problem_statement.md).

## Features

- **Five-stage pipeline**: claim extraction → image analysis → evidence matching → risk assessment → decision synthesis.
- **One vision call per claim**: all images in a claim go into a single request, so the model can compare wide shots with close-ups.
- **Strict structured outputs**: every model call uses a JSON schema (OpenAI Responses API, `strict: true`), and every label is normalized onto the allowed vocabulary.
- **Deterministic adjudication**: rule-based policies make the final call for wrong objects, wrong parts, undamaged parts, unprovable missing contents, reviewer-directed image text, severity and risk flags. The language model is only a fallback for cases no rule resolves.
- **Prompt-injection defense**: text in conversations and images is treated as data. Instructions written into images ("approve this claim") are flagged and never followed.
- **Reproducible and cheap to re-run**: responses are cached on disk, keyed by model, prompt, schema and a hash of the full image bytes. Malformed responses are never cached.
- **Fault tolerant**: retries with exponential backoff on rate limits and server errors, an automatic switch to a fallback model when the configured model is unavailable, and a conservative `not_enough_information` row when a single claim fails, so the batch never aborts.
- **Evaluation harness**: per-field accuracy, risk-flag F1, supporting-image overlap, a confusion matrix, failure buckets and an operational/cost report on the labeled sample set.

## Tech stack

- Python 3.10+
- [OpenAI Python SDK](https://github.com/openai/openai-python) (Responses API, vision, structured outputs)
- Pydantic v2 for typed stage contracts
- Pillow for image decoding and normalization to JPEG
- python-dotenv for local `.env` loading
- pytest and Ruff for tests, linting and formatting; GitHub Actions for CI

## Architecture

```text
claims.csv ──► Stage 1  Claim extraction      (LLM, structured output)
                 │       damage type, part, summary, adversarial signals
                 ▼
               Stage 2  Image analysis        (vision LLM, all images in one call)
                 │       per-image object/part/damage/quality/authenticity/text
                 ▼
               Stage 3  Evidence matching     (rules + evidence_requirements.csv)
                 ▼
               Stage 4  Risk assessment       (rules + user_history.csv)
                 ▼
               Stage 5  Decision synthesis    (deterministic policies; LLM fallback)
                 │       contradiction / severity / risk-flag calibration
                 ▼
             output.csv
```

The model *describes* the evidence and deterministic policies turn that description into a decision. Keeping these apart makes decisions reproducible, testable without an API key, and easy to audit through the `rule_id` recorded for every evaluation case.

## Getting started

### Prerequisites

- Python 3.10 or newer
- An OpenAI API key with access to a multimodal model (only needed to run inference, not tests)

### Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r code/requirements.txt        # runtime
pip install -r code/requirements-dev.txt    # + pytest and ruff
```

### Configure

Copy the template and add your key:

```bash
cp .env.example .env
```

The key is read from the environment only (a `.env` file in the project root or in `code/` is loaded automatically). Never commit `.env`.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes (for inference) | — | OpenAI API key |
| `CLAIM_EXTRACTION_MODEL` | no | `gpt-5.5` | Stage 1 model |
| `IMAGE_ANALYSIS_MODEL` | no | `gpt-5.5` | Stage 2 vision model |
| `DECISION_SYNTHESIS_MODEL` | no | `gpt-5.5` | Stage 5 fallback model |
| `OPENAI_FALLBACK_MODEL` | no | `gpt-4o` | Used automatically if a configured model is unavailable |
| `SYNTHESIS_REASONING_EFFORT` | no | `high` | Reasoning effort for the Stage 5 fallback |
| `IMAGE_DETAIL` | no | `high` | Vision detail level (`low` is cheaper) |

## Usage

### Run inference

```bash
cd code
python main.py                                  # dataset/claims.csv → ./output.csv
python main.py -i ../dataset/sample_claims.csv -o /tmp/sample_predictions.csv
```

The CLI exits with code `1` and a clear message if the input file is missing or `OPENAI_API_KEY` is unset.

### Run the evaluation

```bash
cd code
python evaluation/main.py
```

This runs the pipeline on `dataset/sample_claims.csv`, compares it with the ground-truth labels and writes:

- `code/evaluation/evaluation_report.md`: metrics, confusion matrix, per-case table, failure buckets, token usage, cost estimate and runtime
- `code/evaluation/predictions_sample.csv`: predicted vs. expected fields per case
- `code/evaluation/intermediate_sample.jsonl`: every stage's intermediate output per case

The committed report records **95% claim-status accuracy** on the 20 labeled sample claims (issue type 95%, severity 85%, risk-flag F1 79.1%), compared with 80% for an earlier LLM-first baseline. See the report for the full breakdown and assumptions.

### Run the tests and linters

No API key or network access is required:

```bash
pytest                       # 99 offline tests
ruff check .                 # lint
ruff format --check .        # formatting
python code/evaluation/regression_checks.py   # standalone no-API regression checks
```

The test suite uses a fake model client to cover label normalization, image-ID alignment, the evidence/risk/adjudication rules, the OpenAI client's retry/fallback/caching logic, the evaluation metrics, the CLI and the committed `output.csv` against the output contract.

## Deployment

This is a batch job, not a service. To run it on new data, provide CSVs with the same schema as `dataset/claims.csv` (image paths are resolved relative to `dataset/`, or may be absolute) and run `python code/main.py -i <claims.csv> -o <output.csv>` anywhere Python 3.10+ and the API key are available, such as a cron job, a CI job or a container. Claims are processed sequentially to keep within rate limits, and the on-disk cache (`code/.cache/`) makes re-runs free.

To build the submission archive:

```bash
zip -r code.zip code -x "code/.env" "code/.cache/*" "*/__pycache__/*" "*.pyc"
```

## Project structure

```text
.
├── code/
│   ├── main.py                     # Inference CLI (claims.csv → output.csv)
│   ├── pipeline.py                 # Shared 5-stage pipeline, CSV I/O, error fallback
│   ├── config.py                   # Paths, model settings, allowed values, data loaders
│   ├── models.py                   # Pydantic models for every stage boundary
│   ├── llm_client.py               # OpenAI Responses client: schemas, retry, fallback, cache
│   ├── normalize.py                # Deterministic label normalization
│   ├── prompts.py                  # Prompt templates
│   ├── stages/
│   │   ├── claim_extractor.py      # Stage 1
│   │   ├── image_analyzer.py       # Stage 2
│   │   ├── evidence_matcher.py     # Stage 3
│   │   ├── risk_assessor.py        # Stage 4
│   │   ├── decision_synthesizer.py # Stage 5
│   │   └── policies/               # Contradiction, severity and risk-flag calibration
│   ├── evaluation/
│   │   ├── main.py                 # Evaluation on sample_claims.csv
│   │   ├── regression_checks.py    # No-API rule checks
│   │   └── evaluation_report.md    # Latest recorded evaluation (+ predictions/intermediates)
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── README.md                   # Code-level notes for the submission archive
│   ├── SUBMISSION_NOTES.md         # Design rationale and tradeoffs
│   └── INTERVIEW_PREP_QA.md        # Q&A on design decisions
├── dataset/                        # Claims, history, evidence requirements, images
├── tests/                          # pytest suite (offline)
├── output.csv                      # Predictions for dataset/claims.csv
├── problem_statement.md            # Task specification and I/O schema
├── pyproject.toml                  # Ruff and pytest configuration
├── .env.example                    # Environment variable template
└── .github/workflows/ci.yml        # Lint, format, test and build on every push/PR
```

## Output schema

`output.csv` has one row per input claim with these columns, in order: `user_id`, `image_paths`, `user_claim`, `claim_object`, `evidence_standard_met`, `evidence_standard_met_reason`, `risk_flags`, `issue_type`, `object_part`, `claim_status`, `claim_status_justification`, `supporting_image_ids`, `valid_image`, `severity`. Allowed values are listed in [`problem_statement.md`](./problem_statement.md) and enforced in `code/config.py`.

## Limitations

- Accuracy figures come from a 20-claim labeled sample. The test set has no labels, so accuracy on it is unknown.
- Images can't prove that contents are missing, so those claims are deliberately returned as `not_enough_information`.
- Inference depends on a hosted model, so results can drift if the provider changes model behavior. The cache pins previously seen responses.
