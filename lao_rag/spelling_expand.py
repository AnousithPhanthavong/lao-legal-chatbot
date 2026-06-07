"""
Lao Legal RAG — Step 3b: Spelling-Variant Expansion
====================================================

Fixes the documented Q12 failure: variant spellings silently demote matching.
We EXPAND a query into its spelling variants, retrieve on each, and RRF-fuse —
retrieving whatever spelling the document happens to use. Frozen-corpus safe.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

EQUIVALENCE_CLASSES: tuple[tuple[str, ...], ...] = (
    ("ໝ", "ຫມ"),
    ("ໜ", "ຫນ"),
    ("ຫຼ", "ຫລ"),
    ("ສິ", "ສີ"),
    ("ກົດໝາຍ", "ກົດຫມາຍ"),
    ("ໜີ້", "ຫນີ້"),
    ("ໜ້າ", "ຫນ້າ"),
    ("ໝາຍ", "ຫມາຍ"),
    ("ພາສີ", "ພາສິ"),
)

MAX_VARIANT_FORMS = 4


def _variant_markers() -> set[str]:
    markers: set[str] = set()
    for cls in EQUIVALENCE_CLASSES:
        markers.update(cls)
    return markers


_MARKERS = _variant_markers()


def has_variant(query: str) -> bool:
    """Cheap gate: does the query contain any spelling-variant substring?"""
    return any(m in query for m in _MARKERS)


def canonical_form(text: str) -> str:
    """Collapse to a single canonical spelling (first element of each class)."""
    result = text
    pairs: list[tuple[str, str]] = []
    for cls in EQUIVALENCE_CLASSES:
        canon = cls[0]
        for variant in cls[1:]:
            pairs.append((variant, canon))
    for variant, canon in sorted(pairs, key=lambda p: len(p[0]), reverse=True):
        result = result.replace(variant, canon)
    return result


def expand_query(query: str, *, max_forms: int = MAX_VARIANT_FORMS) -> list[str]:
    """Return the query plus spelling variants (deduped, original first)."""
    forms: list[str] = [query]
    canon = canonical_form(query)
    if canon != query:
        forms.append(canon)
    for cls in EQUIVALENCE_CLASSES:
        present = [s for s in cls if s in query or s in canon]
        if not present:
            continue
        canon_token = cls[0]
        for alt in cls[1:]:
            flipped = canon.replace(canon_token, alt)
            if flipped not in forms:
                forms.append(flipped)
    seen: set[str] = set()
    out: list[str] = []
    for f in forms:
        if f not in seen:
            seen.add(f)
            out.append(f)
        if len(out) >= max_forms:
            break
    return out


def expand_if_needed(query: str, *, max_forms: int = MAX_VARIANT_FORMS) -> list[str]:
    """Gated expansion: only expand when a variant marker is present."""
    if not has_variant(query):
        return [query]
    return expand_query(query, max_forms=max_forms)
