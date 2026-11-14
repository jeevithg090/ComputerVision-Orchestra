"""
Configuration and constants for the Multi-Modal Evidence Review system.

All paths, model names, allowed values, and operational parameters
are centralized here so no magic strings leak into business logic.
"""

import os
import csv
from pathlib import Path
from typing import Dict, FrozenSet, List

# Load .env file if present (for OPENAI_API_KEY etc.)
try:
    from dotenv import load_dotenv
    # Look for .env in code/ dir, project root, or current dir
    _code_dir = Path(__file__).resolve().parent
    _project_root = _code_dir.parent
    for _env_path in [_code_dir / ".env", _project_root / ".env", Path(".env")]:
        if _env_path.exists():
            load_dotenv(_env_path)
            break
except ImportError:
    pass  # python-dotenv not installed, rely on shell env vars


# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CODE_DIR = PROJECT_ROOT / "code"
CACHE_DIR = CODE_DIR / ".cache"

CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
OUTPUT_CSV = PROJECT_ROOT / "output.csv"


# ── OpenAI Models ───────────────────────────────────────────────────────────

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

CLAIM_EXTRACTION_MODEL = os.environ.get("CLAIM_EXTRACTION_MODEL", "gpt-5.5")
IMAGE_ANALYSIS_MODEL = os.environ.get("IMAGE_ANALYSIS_MODEL", "gpt-5.5")
DECISION_SYNTHESIS_MODEL = os.environ.get("DECISION_SYNTHESIS_MODEL", "gpt-5.5")
FALLBACK_MODEL = os.environ.get("OPENAI_FALLBACK_MODEL", "gpt-4o")

# Reasoning effort for the final-decision fallback call: "low", "medium", or "high"
SYNTHESIS_REASONING_EFFORT = os.environ.get("SYNTHESIS_REASONING_EFFORT", "high")

# Image detail level sent to the vision model
IMAGE_DETAIL = os.environ.get("IMAGE_DETAIL", "high")  # "high" = best accuracy, "low" = cheaper / faster


# ── Retry & Rate-Limit ─────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 2.0
RETRY_MAX_DELAY_S = 60.0
REQUEST_TIMEOUT_S = 120  # per-request timeout


# ── Allowed Value Sets ──────────────────────────────────────────────────────

CLAIM_STATUS_VALUES: FrozenSet[str] = frozenset([
    "supported", "contradicted", "not_enough_information",
])

ISSUE_TYPE_VALUES: FrozenSet[str] = frozenset([
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
])

CAR_OBJECT_PARTS: FrozenSet[str] = frozenset([
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown",
])

LAPTOP_OBJECT_PARTS: FrozenSet[str] = frozenset([
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
    "port", "base", "body", "unknown",
])

PACKAGE_OBJECT_PARTS: FrozenSet[str] = frozenset([
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown",
])

OBJECT_PARTS: Dict[str, FrozenSet[str]] = {
    "car": CAR_OBJECT_PARTS,
    "laptop": LAPTOP_OBJECT_PARTS,
    "package": PACKAGE_OBJECT_PARTS,
}

RISK_FLAG_VALUES: FrozenSet[str] = frozenset([
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
])

SEVERITY_VALUES: FrozenSet[str] = frozenset([
    "none", "low", "medium", "high", "unknown",
])

CLAIM_OBJECTS: FrozenSet[str] = frozenset(["car", "laptop", "package"])


# ── Issue-type → Evidence-requirement family mapping ────────────────────────

ISSUE_TO_EVIDENCE_FAMILY: Dict[str, Dict[str, str]] = {
    "car": {
        "dent": "dent or scratch",
        "scratch": "dent or scratch",
        "crack": "crack, broken, or missing part",
        "glass_shatter": "crack, broken, or missing part",
        "broken_part": "crack, broken, or missing part",
        "missing_part": "crack, broken, or missing part",
    },
    "laptop": {
        "crack": "screen, keyboard, or trackpad",
        "stain": "screen, keyboard, or trackpad",
        "broken_part": "hinge, lid, corner, body, or port",
        "dent": "hinge, lid, corner, body, or port",
        "missing_part": "hinge, lid, corner, body, or port",
        "scratch": "hinge, lid, corner, body, or port",
    },
    "package": {
        "crushed_packaging": "crushed, torn, or seal damage",
        "torn_packaging": "crushed, torn, or seal damage",
        "water_damage": "water, stain, or label damage",
        "stain": "water, stain, or label damage",
        "missing_part": "contents or inner item",
        "broken_part": "contents or inner item",
    },
}


# ── Data Loading Helpers ────────────────────────────────────────────────────

def _to_int(value: object) -> int:
    """Parse a CSV count cell; blank or malformed cells count as 0."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def load_user_history(path: Path = USER_HISTORY_CSV) -> Dict[str, dict]:
    """Load user_history.csv into a dict keyed by user_id."""
    history: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = (row.get("user_id") or "").strip()
            if not uid:
                continue
            history[uid] = {
                "past_claim_count": _to_int(row.get("past_claim_count")),
                "accept_claim": _to_int(row.get("accept_claim")),
                "manual_review_claim": _to_int(row.get("manual_review_claim")),
                "rejected_claim": _to_int(row.get("rejected_claim")),
                "last_90_days_claim_count": _to_int(row.get("last_90_days_claim_count")),
                "history_flags": (row.get("history_flags") or "none").strip(),
                "history_summary": (row.get("history_summary") or "").strip(),
            }
    return history


def load_evidence_requirements(path: Path = EVIDENCE_REQUIREMENTS_CSV) -> List[dict]:
    """Load evidence_requirements.csv into a list of dicts."""
    reqs: List[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reqs.append({
                "requirement_id": row["requirement_id"].strip(),
                "claim_object": row["claim_object"].strip(),
                "applies_to": row["applies_to"].strip(),
                "minimum_image_evidence": row["minimum_image_evidence"].strip(),
            })
    return reqs
