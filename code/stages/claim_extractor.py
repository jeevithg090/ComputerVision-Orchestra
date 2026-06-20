"""
Stage 1 — Claim Extraction

Parses the user_claim conversation to extract:
  - Claimed damage type
  - Claimed object part
  - A brief summary of the claim
  - Any adversarial signals (prompt injection, manipulation)
  - Primary language of the conversation
"""

from __future__ import annotations

from models import ClaimInput, ClaimExtraction
from llm_client import LLMClient
from config import CLAIM_EXTRACTION_MODEL, EXTRACTION_TEMPERATURE, OBJECT_PARTS
from prompts import CLAIM_EXTRACTION_SYSTEM, build_claim_extraction_user_prompt


def extract_claim(claim: ClaimInput, client: LLMClient) -> ClaimExtraction:
    """
    Run Stage 1: Extract structured claim information from the conversation.

    Args:
        claim: Raw claim input row.
        client: LLM client for API calls.

    Returns:
        ClaimExtraction with parsed claim details.
    """
    user_prompt = build_claim_extraction_user_prompt(
        claim_object=claim.claim_object,
        user_claim=claim.user_claim,
    )

    result = client.call_text(
        model=CLAIM_EXTRACTION_MODEL,
        system_message=CLAIM_EXTRACTION_SYSTEM,
        user_message=user_prompt,
        temperature=EXTRACTION_TEMPERATURE,
    )

    # Validate and normalize the damage type
    claimed_damage = result.get("claimed_damage_type", "unknown")
    from config import ISSUE_TYPE_VALUES
    if claimed_damage not in ISSUE_TYPE_VALUES:
        claimed_damage = _find_closest_match(claimed_damage, ISSUE_TYPE_VALUES)

    # Validate and normalize the object part
    claimed_part = result.get("claimed_object_part", "unknown")
    valid_parts = OBJECT_PARTS.get(claim.claim_object, frozenset(["unknown"]))
    if claimed_part not in valid_parts:
        claimed_part = _find_closest_match(claimed_part, valid_parts)

    return ClaimExtraction(
        claimed_damage_type=claimed_damage,
        claimed_object_part=claimed_part,
        claim_summary=result.get("claim_summary", "Damage claim submitted."),
        adversarial_signals=result.get("adversarial_signals", []),
        language_detected=result.get("language_detected", "en"),
    )


def _find_closest_match(value: str, allowed: frozenset[str]) -> str:
    """
    Find the closest match from allowed values using substring matching.
    Falls back to 'unknown' if no reasonable match is found.
    """
    value_lower = value.lower().strip().replace(" ", "_").replace("-", "_")

    # Exact match after normalization
    if value_lower in allowed:
        return value_lower

    # Substring match
    for candidate in allowed:
        if candidate in value_lower or value_lower in candidate:
            return candidate

    # Common synonyms
    synonyms = {
        "cracked": "crack",
        "broken": "broken_part",
        "shattered": "glass_shatter",
        "torn": "torn_packaging",
        "crushed": "crushed_packaging",
        "water": "water_damage",
        "wet": "water_damage",
        "missing": "missing_part",
        "dented": "dent",
        "scratched": "scratch",
        "stained": "stain",
    }
    for syn, target in synonyms.items():
        if syn in value_lower and target in allowed:
            return target

    return "unknown"
