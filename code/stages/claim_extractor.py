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

from config import CLAIM_EXTRACTION_MODEL, ISSUE_TYPE_VALUES, OBJECT_PARTS
from llm_client import LLMClient
from models import ClaimExtraction, ClaimInput
from normalize import DAMAGE_SYNONYMS, normalize_label
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
    )

    # Validate and normalize the damage type
    claimed_damage = normalize_label(
        result.get("claimed_damage_type", "unknown"),
        ISSUE_TYPE_VALUES,
        synonyms=DAMAGE_SYNONYMS,
    )

    # Validate and normalize the object part
    valid_parts = OBJECT_PARTS.get(claim.claim_object, frozenset(["unknown"]))
    claimed_part = normalize_label(result.get("claimed_object_part", "unknown"), valid_parts)

    signals = result.get("adversarial_signals", [])
    if isinstance(signals, str):
        signals = [signals] if signals.strip() else []

    return ClaimExtraction(
        claimed_damage_type=claimed_damage,
        claimed_object_part=claimed_part,
        claim_summary=result.get("claim_summary", "Damage claim submitted."),
        adversarial_signals=[str(sig) for sig in signals if str(sig).strip()],
        language_detected=result.get("language_detected", "en"),
    )
