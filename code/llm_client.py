"""
OpenAI API client with retry logic, file-based caching, and token tracking.

Supports three calling modes:
  1. Text structured output through the Responses API
  2. Vision structured output with base64 local images
  3. Final-decision structured output fallback
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError

from config import (
    CACHE_DIR,
    FALLBACK_MODEL,
    MAX_RETRIES,
    RETRY_BASE_DELAY_S,
    RETRY_MAX_DELAY_S,
    REQUEST_TIMEOUT_S,
    IMAGE_DETAIL,
    OPENAI_API_KEY_ENV,
)


CLAIM_EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claimed_damage_type": {"type": "string"},
        "claimed_object_part": {"type": "string"},
        "claim_summary": {"type": "string"},
        "adversarial_signals": {"type": "array", "items": {"type": "string"}},
        "language_detected": {"type": "string"},
    },
    "required": [
        "claimed_damage_type",
        "claimed_object_part",
        "claim_summary",
        "adversarial_signals",
        "language_detected",
    ],
}


IMAGE_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "image_id": {"type": "string"},
                    "object_category": {"type": "string"},
                    "object_visible": {"type": "string"},
                    "is_expected_object": {"type": "boolean"},
                    "object_matches_claim": {"type": "boolean"},
                    "part_visible": {"type": "string"},
                    "part_matches_claim": {"type": "boolean"},
                    "damage_visible": {"type": "boolean"},
                    "claimed_damage_visible": {"type": "boolean"},
                    "other_damage_visible": {"type": "boolean"},
                    "damage_type": {"type": "string"},
                    "visible_damage_type": {"type": "string"},
                    "damage_confidence": {"type": "number"},
                    "damage_severity": {"type": "string"},
                    "quality_issues": {"type": "array", "items": {"type": "string"}},
                    "quality_blockers": {"type": "array", "items": {"type": "string"}},
                    "text_instruction_detected": {"type": "boolean"},
                    "text_instruction_content": {"type": "string"},
                    "instructional_text": {"type": "string"},
                    "appears_original": {"type": "boolean"},
                    "image_authenticity": {"type": "string"},
                    "relevance_to_claim": {"type": "string"},
                    "support_role": {"type": "string"},
                },
                "required": [
                    "image_id",
                    "object_category",
                    "object_visible",
                    "is_expected_object",
                    "object_matches_claim",
                    "part_visible",
                    "part_matches_claim",
                    "damage_visible",
                    "claimed_damage_visible",
                    "other_damage_visible",
                    "damage_type",
                    "visible_damage_type",
                    "damage_confidence",
                    "damage_severity",
                    "quality_issues",
                    "quality_blockers",
                    "text_instruction_detected",
                    "text_instruction_content",
                    "instructional_text",
                    "appears_original",
                    "image_authenticity",
                    "relevance_to_claim",
                    "support_role",
                ],
            },
        },
    },
    "required": ["images"],
}


CLAIM_DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claim_status": {"type": "string"},
        "claim_status_justification": {"type": "string"},
        "issue_type": {"type": "string"},
        "object_part": {"type": "string"},
        "supporting_image_ids": {"type": "string"},
        "valid_image": {"type": "boolean"},
        "severity": {"type": "string"},
        "evidence_standard_met": {"type": "boolean"},
        "evidence_standard_met_reason": {"type": "string"},
        "risk_flags": {"type": "string"},
    },
    "required": [
        "claim_status",
        "claim_status_justification",
        "issue_type",
        "object_part",
        "supporting_image_ids",
        "valid_image",
        "severity",
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
    ],
}


# ── Token Tracker ───────────────────────────────────────────────────────────

class TokenTracker:
    """Accumulates token usage across all API calls for the operational report."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0

    def record(self, model: str, input_tokens: int, output_tokens: int,
               reasoning_tokens: int = 0, cached: bool = False):
        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached": cached,
            "timestamp": time.time(),
        })
        if not cached:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_reasoning_tokens += reasoning_tokens

    def summary(self) -> Dict[str, Any]:
        model_breakdown: Dict[str, Dict[str, int]] = {}
        for c in self.calls:
            m = c["model"]
            if m not in model_breakdown:
                model_breakdown[m] = {"calls": 0, "input": 0, "output": 0, "reasoning": 0, "cached": 0}
            model_breakdown[m]["calls"] += 1
            if c["cached"]:
                model_breakdown[m]["cached"] += 1
            else:
                model_breakdown[m]["input"] += c["input_tokens"]
                model_breakdown[m]["output"] += c["output_tokens"]
                model_breakdown[m]["reasoning"] += c["reasoning_tokens"]
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "model_breakdown": model_breakdown,
        }


# ── Cache ───────────────────────────────────────────────────────────────────

class FileCache:
    """Simple file-based cache keyed by a hash of the request payload."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, payload: str) -> Optional[Dict[str, Any]]:
        key = self._key(payload)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set(self, payload: str, value: Dict[str, Any]):
        key = self._key(payload)
        path = self.cache_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)


# ── Image Encoding ──────────────────────────────────────────────────────────

from io import BytesIO
from PIL import Image

def encode_image_base64(image_path: str | Path) -> str:
    """Read an image file, convert to JPEG in memory, and return base64 string."""
    with Image.open(image_path) as img:
        # Convert to RGB to handle alpha channels and other color spaces
        rgb_img = img.convert("RGB")
        buffer = BytesIO()
        # Save as JPEG into the buffer
        rgb_img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _image_content_part(image_path: str | Path) -> Dict[str, Any]:
    """Build an OpenAI image_url content part from a local file, standardizing to JPEG."""
    b64 = encode_image_base64(image_path)
    # Since encode_image_base64 now converts everything to JPEG, we hardcode the media type.
    media_type = "image/jpeg"

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{b64}",
            "detail": IMAGE_DETAIL,
        },
    }


def _responses_image_content_part(image_path: str | Path) -> Dict[str, Any]:
    """Build a Responses API input_image part from a local image."""
    b64 = encode_image_base64(image_path)
    return {
        "type": "input_image",
        "image_url": f"data:image/jpeg;base64,{b64}",
        "detail": IMAGE_DETAIL,
    }


# ── LLM Client ─────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified OpenAI client for Responses API text, vision, and fallback calls.

    Usage:
        client = LLMClient()
        result = client.call_text(model, system_msg, user_msg)
        result = client.call_vision(model, system_msg, user_msg, image_paths)
        result = client.call_reasoning(model, developer_msg, user_msg)
    """

    def __init__(self, tracker: Optional[TokenTracker] = None):
        api_key = os.environ.get(OPENAI_API_KEY_ENV)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {OPENAI_API_KEY_ENV} is not set. "
                f"Please set it before running the pipeline."
            )
        self.client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_S)
        self.cache = FileCache(CACHE_DIR)
        self.tracker = tracker or TokenTracker()

    # ── Public Methods ──────────────────────────────────────────────────

    def call_text(
        self,
        model: str,
        system_message: str,
        user_message: str,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Call a text-only model, returns parsed JSON dict."""
        return self._call_responses(
            model=model,
            system_message=system_message,
            user_content=[{"type": "input_text", "text": user_message}],
            schema_name="claim_extraction",
            schema=CLAIM_EXTRACTION_SCHEMA,
        )

    def call_vision(
        self,
        model: str,
        system_message: str,
        user_text: str,
        image_paths: List[str | Path],
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """Call a vision model with text + images, returns parsed JSON dict."""
        user_content: List[Dict[str, Any]] = [
            {"type": "input_text", "text": user_text},
        ]
        for img_path in image_paths:
            user_content.append(_responses_image_content_part(img_path))

        return self._call_responses(
            model=model,
            system_message=system_message,
            user_content=user_content,
            schema_name="image_analysis",
            schema=IMAGE_ANALYSIS_SCHEMA,
        )

    def call_reasoning(
        self,
        model: str,
        developer_message: str,
        user_message: str,
        reasoning_effort: str = "high",
    ) -> Dict[str, Any]:
        """Call the final-decision fallback model, returns parsed JSON dict."""
        return self._call_responses(
            model=model,
            system_message=developer_message,
            user_content=[{"type": "input_text", "text": user_message}],
            schema_name="claim_decision",
            schema=CLAIM_DECISION_SCHEMA,
            reasoning_effort=reasoning_effort,
            is_reasoning=True,
        )

    # ── Internal ────────────────────────────────────────────────────────

    def _cache_key(self, model: str, messages: List[dict]) -> str:
        """Build a deterministic cache key from model + messages (text only)."""
        # For caching, we stringify messages but skip base64 image data
        # (we include a hash of the image URLs/paths instead)
        parts = [model]
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        # Hash just the first 100 chars of the data URL to identify the image
                        url = item.get("image_url", {}).get("url", "")
                        parts.append(hashlib.md5(url[:200].encode()).hexdigest())
        return json.dumps(parts, ensure_ascii=False)

    def _responses_cache_key(
        self,
        model: str,
        system_message: str,
        user_content: List[dict],
        schema_name: str,
    ) -> str:
        """Build a cache key that hashes full local-image payloads."""
        parts: List[Any] = ["responses-v2", model, schema_name, system_message]
        for item in user_content:
            if item.get("type") == "input_text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "input_image":
                url = item.get("image_url", "")
                parts.append(hashlib.sha256(url.encode("utf-8")).hexdigest())
                parts.append(item.get("detail", IMAGE_DETAIL))
        return json.dumps(parts, ensure_ascii=False)

    def _call_responses(
        self,
        model: str,
        system_message: str,
        user_content: List[dict],
        schema_name: str,
        schema: Dict[str, Any],
        reasoning_effort: str = "high",
        is_reasoning: bool = False,
    ) -> Dict[str, Any]:
        """Execute a Responses API call with strict structured outputs."""
        active_model = model
        tried_fallback = False
        last_error: Optional[Exception] = None

        while True:
            cache_key = self._responses_cache_key(
                active_model, system_message, user_content, schema_name
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.tracker.record(active_model, 0, 0, cached=True)
                return cached

            for attempt in range(MAX_RETRIES):
                try:
                    kwargs: Dict[str, Any] = {
                        "model": active_model,
                        "instructions": system_message,
                        "input": [{"role": "user", "content": user_content}],
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "schema": schema,
                                "strict": True,
                            }
                        },
                    }
                    if is_reasoning:
                        kwargs["reasoning"] = {"effort": reasoning_effort}

                    response = self.client.responses.create(**kwargs)
                    parsed = self._parse_json(getattr(response, "output_text", "") or "{}")

                    usage = getattr(response, "usage", None)
                    input_tok = getattr(usage, "input_tokens", 0) if usage else 0
                    output_tok = getattr(usage, "output_tokens", 0) if usage else 0
                    details = getattr(usage, "output_tokens_details", None) if usage else None
                    reasoning_tok = getattr(details, "reasoning_tokens", 0) if details else 0
                    self.tracker.record(active_model, input_tok, output_tok, reasoning_tok)

                    self.cache.set(cache_key, parsed)
                    return parsed

                except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                    last_error = e
                    delay = min(RETRY_BASE_DELAY_S * (2 ** attempt), RETRY_MAX_DELAY_S)
                    print(f"  ⚠ API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    print(f"    Retrying in {delay:.1f}s...")
                    time.sleep(delay)

                except APIError as e:
                    last_error = e
                    msg = str(e).lower()
                    model_unavailable = e.status_code in (400, 404) and (
                        "model" in msg or "unsupported" in msg or "not found" in msg
                    )
                    if model_unavailable and not tried_fallback and active_model != FALLBACK_MODEL:
                        tried_fallback = True
                        active_model = FALLBACK_MODEL
                        print(f"  ⚠ Model unavailable; falling back to {FALLBACK_MODEL}.")
                        break
                    if e.status_code and e.status_code >= 500:
                        delay = min(RETRY_BASE_DELAY_S * (2 ** attempt), RETRY_MAX_DELAY_S)
                        print(f"  ⚠ Server error {e.status_code} (attempt {attempt + 1}/{MAX_RETRIES})")
                        time.sleep(delay)
                    else:
                        raise
            else:
                raise RuntimeError(
                    f"All {MAX_RETRIES} API call attempts failed for model {active_model}. "
                    f"Last error: {last_error}"
                )

    def _call(
        self,
        model: str,
        messages: List[dict],
        temperature: float = 0.1,
        reasoning_effort: str = "high",
        is_reasoning: bool = False,
    ) -> Dict[str, Any]:
        """Execute an API call with retry, caching, and token tracking."""
        cache_key = self._cache_key(model, messages)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.tracker.record(model, 0, 0, cached=True)
            return cached

        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                }
                if is_reasoning:
                    kwargs["reasoning_effort"] = reasoning_effort
                    # o3 doesn't accept temperature
                else:
                    kwargs["temperature"] = temperature

                response = self.client.chat.completions.create(**kwargs)

                # Parse response
                content = response.choices[0].message.content or "{}"
                parsed = self._parse_json(content)

                # Track tokens
                usage = response.usage
                input_tok = usage.prompt_tokens if usage else 0
                output_tok = usage.completion_tokens if usage else 0
                reasoning_tok = 0
                if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                    reasoning_tok = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

                self.tracker.record(model, input_tok, output_tok, reasoning_tok)

                # Cache the result
                self.cache.set(cache_key, parsed)
                return parsed

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                delay = min(
                    RETRY_BASE_DELAY_S * (2 ** attempt),
                    RETRY_MAX_DELAY_S,
                )
                print(f"  ⚠ API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                print(f"    Retrying in {delay:.1f}s...")
                time.sleep(delay)

            except APIError as e:
                last_error = e
                if e.status_code and e.status_code >= 500:
                    delay = min(RETRY_BASE_DELAY_S * (2 ** attempt), RETRY_MAX_DELAY_S)
                    print(f"  ⚠ Server error {e.status_code} (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    raise

        raise RuntimeError(
            f"All {MAX_RETRIES} API call attempts failed for model {model}. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        """Parse JSON from model response, handling common edge cases."""
        content = content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  ⚠ JSON parse error: {e}")
            print(f"    Raw content (first 500 chars): {content[:500]}")
            # Return a minimal dict so the pipeline doesn't crash
            return {"parse_error": str(e), "raw_content": content[:1000]}
