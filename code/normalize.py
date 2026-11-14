"""
Deterministic normalization of free-text model labels onto allowed value sets.

Model outputs occasionally drift from the allowed vocabularies ("cracked",
"Rear Bumper", "no damage"). These helpers map them back onto the closed sets
defined in ``config`` and always return the same answer for the same input:
candidates are tried in a fixed order (longest first, then alphabetical) rather
than in ``frozenset`` iteration order, which varies between interpreter runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Ordered: earlier entries win when several synonyms occur in one value.
DAMAGE_SYNONYMS: Mapping[str, str] = {
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
    "no_damage": "none",
    "not_visible": "unknown",
}


def canonical(value: object) -> str:
    """Lower-case a label and convert spaces/hyphens to underscores."""
    if value is None:
        return ""
    return str(value).lower().strip().replace(" ", "_").replace("-", "_")


def _ordered(allowed: Iterable[str]) -> list[str]:
    return sorted(allowed, key=lambda c: (-len(c), c))


def normalize_label(
    value: object,
    allowed: Iterable[str],
    *,
    default: str = "unknown",
    synonyms: Mapping[str, str] | None = None,
    substring: bool = True,
) -> str:
    """
    Map ``value`` onto one of ``allowed``.

    Resolution order: exact match after canonicalization, then (optionally)
    substring containment in either direction, then synonym lookup. Returns
    ``default`` when nothing matches or the value is empty.
    """
    allowed_set = set(allowed)
    v = canonical(value)
    if not v:
        return default
    if v in allowed_set:
        return v
    if substring:
        for candidate in _ordered(allowed_set):
            if candidate in v or v in candidate:
                return candidate
    if synonyms:
        for syn, target in synonyms.items():
            if syn in v and target in allowed_set:
                return target
    return default


def normalize_severity(value: object, allowed: Iterable[str]) -> str:
    """Normalize a severity label onto ``allowed`` (none/low/medium/high/unknown)."""
    v = canonical(value)
    if v in set(allowed):
        return v
    for words, target in (
        (("minor", "slight", "light", "small"), "low"),
        (("moderate", "medium"), "medium"),
        (("severe", "major", "heavy", "significant"), "high"),
    ):
        if any(w in v for w in words):
            return target
    if "no" in v:
        return "none"
    return "unknown"
