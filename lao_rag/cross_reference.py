"""
Lao Legal RAG — Cross-Reference Extractor (Enrichment 1.5)
==========================================================

Builds the article->article citation graph powering the agent's
`follow_reference` tool. DETERMINISTIC: regex over article text, ZERO LLM calls,
so there is no hallucination risk — every edge is traceable to the literal
`context` substring that produced it (the grounding rule).

Corrected against REAL reference text (v2)
------------------------------------------
The first version failed on live data. Inspection revealed:
  1. Lao legal references appear in BOTH word orders:
       - article-first:  "ມາດຕາ 38 ... ຂອງກົດໝາຍວ່າດ້ວຍອາກອນລາຍໄດ້"
       - law-first:       "ຕາມກົດໝາຍວ່າດ້ວຍອາກອນລາຍໄດ້ ມາດຕາ 38"
     The original only handled law-first, so it MISSED the common article-first
     form and instead matched split-chunk citation headers.
  2. The ໝ/ຫມ spelling split (the Q12 problem) made references written
     "ກົດຫມາຍ" invisible to a "ກົດໝາຍ" pattern. We now NORMALIZE spelling
     (reusing spelling_expand.canonical_form) before matching.
  3. A reference whose target law == source law is SAME_LAW, not CROSS_LAW.
  4. Exact self-references (same law + same article) are filtered as noise.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel

try:
    # reuse the project's canonical spelling collapse for consistency
    from spelling_expand import canonical_form as _canonical
except Exception:  # pragma: no cover - fallback if module not on path
    def _canonical(t: str) -> str:
        return (t.replace("ກົດຫມາຍ", "ກົດໝາຍ")
                 .replace("ຫມ", "ໝ").replace("ຫນ", "ໜ").replace("ຫລ", "ຫຼ"))


class RefType(str, Enum):
    SAME_LAW = "same_law"
    CROSS_LAW = "cross_law"
    DECREE = "decree"


class CrossReference(BaseModel):
    from_law: str
    from_article: int
    to_law: Optional[str] = None
    to_article: Optional[int] = None
    ref_type: RefType
    resolved: bool
    context: str


# --------------------------------------------------------------------------- #
# Patterns (applied to SPELLING-NORMALIZED text)
# --------------------------------------------------------------------------- #
_LAO = r"\u0e80-\u0eff"

# article-first: "ມາດຕາ N ... ຂອງ ກົດໝາຍ(ວ່າດ້ວຍ) <name>"
# the gap between number and ຂອງກົດໝາຍ allows "ຂໍ້ 2", "ຂໍ້ທີ 4" etc.
_ART_FIRST = re.compile(
    r"ມາດຕາ\s*(\d+)[^ມ]{0,40}?ຂອງກົດໝາຍ(?:ວ່າດ້ວຍ)?\s*"
    r"((?:(?!ມາດຕາ)[" + _LAO + r"]){2,40})"
)
# law-first: "ກົດໝາຍ(ວ່າດ້ວຍ) <name> ມາດຕາ N"
_LAW_FIRST = re.compile(
    r"ກົດໝາຍ(?:ວ່າດ້ວຍ)?\s*((?:(?!ມາດຕາ)[" + _LAO + r"]){2,40})\s*ມາດຕາ\s*(\d+)"
)
# bare same-law article reference: "ມາດຕາ N" not tied to any law name
_BARE_ART = re.compile(r"ມາດຕາ\s*(\d+)")
# decree reference
_DECREE = re.compile(r"ດຳລັດ\s*((?:(?!ມາດຕາ)[" + _LAO + r"]){2,40})")

_CTX = 45


def _ctx(text: str, start: int, end: int) -> str:
    a, b = max(0, start - _CTX), min(len(text), end + _CTX)
    return text[a:b].replace("\n", " ").replace("\r", " ").strip()


def _norm_name(name: str) -> str:
    n = re.sub(r"[\s,.\-—]+", "", _canonical(name)).strip()
    # strip the boilerplate law-name prefix that extracted phrases never include
    for prefix in ("ກົດໝາຍວ່າດ້ວຍ", "ກົດໝາຍ", "ດຳລັດວ່າດ້ວຍ", "ດຳລັດ"):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    return n


def _match_score(phrase_norm: str, name_norm: str) -> float:
    """Bidirectional coverage score in [0,1]. Longest common substring length
    divided by the length of the LONGER string, so matching an entire short law
    name scores higher than partially matching a long one. This lets 'ວິສາຫະກິດ'
    resolve to the Enterprise law rather than tie with the longer SME law whose
    name merely contains the word."""
    if not phrase_norm or not name_norm:
        return 0.0
    a, b = phrase_norm, name_norm
    la, lb = len(a), len(b)
    prev = [0] * (lb + 1)
    best = 0
    for i in range(1, la + 1):
        cur = [0] * (lb + 1)
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    # divide by the LONGER length: rewards covering the whole law name, so a
    # phrase that IS a short law's full name beats one that is a fragment of a
    # long law's name.
    return best / max(la, lb)


def _resolve_law(
    phrase: str,
    registry: dict[str, str],
    *,
    min_score: float = 0.75,
    margin: float = 0.12,
) -> Optional[str]:
    """Resolve a referenced law phrase to a registry law_id.

    Spelling-normalizes BOTH sides, scores by longest-common-substring coverage,
    and returns the best match ONLY IF:
      * its score >= min_score (strong correspondence, not a shared word), AND
      * it beats the runner-up by >= margin (unambiguous).
    Otherwise returns None (honest unresolved beats a confident wrong guess).
    """
    np = _norm_name(phrase)
    if len(np) < 4:
        return None

    # exact full-name match wins outright (unambiguous even if np is a substring
    # of a longer law name): e.g. phrase 'ວິສາຫະກິດ' exactly == Enterprise law's
    # core name, so it resolves there, not to 'SME ... ວິສາຫະກິດ ...'.
    exact = [law_id for law_id, name in registry.items()
             if _norm_name(name) == np]
    if len(exact) == 1:
        return exact[0]

    scored: list[tuple[float, str]] = []
    for law_id, law_name in registry.items():
        s = _match_score(np, _norm_name(law_name))
        scored.append((s, law_id))
    scored.sort(reverse=True)
    if not scored:
        return None
    top_score, top_id = scored[0]
    if top_score < min_score:
        return None
    # ambiguity guard: reject if runner-up is too close
    if len(scored) > 1:
        runner_up = scored[1][0]
        if top_score - runner_up < margin:
            return None
    return top_id


def extract_references(
    from_law_id: str,
    from_article: int,
    article_text: str,
    registry: dict[str, str],
    *,
    drop_self: bool = True,
) -> list[CrossReference]:
    """Extract references from one article's (raw) text.

    Spelling is normalized internally before matching; `context` is taken from
    the normalized text so it reflects what was matched.
    """
    text = _canonical(article_text or "")
    refs: list[CrossReference] = []
    # spans of text consumed by a law-scoped match, so the bare-article pass
    # does not re-count an article number that already belongs to a law ref.
    consumed: list[tuple[int, int]] = []

    def _is_this_law(phrase: str) -> bool:
        """Detect 'this law' self-reference, robust to OCR noise:
        - spacing: 'ສະບັບ ນີ້' vs 'ສະບັບນີ້'
        - doubled vowels: 'ນີີ້' (OCR duplicated ີ) vs 'ນີ້'
        """
        p = re.sub(r"\s+", "", phrase)             # drop spaces
        p = re.sub(r"(.)\1+", r"\1", p)            # collapse repeated chars (ນີີ້->ນີ້)
        # 'this'/'this version' markers; bare 'ສະບັບ' arises when OCR spacing
        # truncates 'ສະບັບນີ້' at the space.
        return (("ນີ້" in p) or ("ສະບັບນີ້" in p) or ("ສະບັບ" in p)
                or p in ("ນີ", "ສະບັບນີ"))

    def _add_lawscoped(art_num: int, law_phrase: str, s: int, e: int):
        phrase = law_phrase.strip()
        # "ກົດໝາຍນີ້" / "ກົດໝາຍສະບັບນີ້" = "this (same) law" -> same-law ref
        if _is_this_law(phrase):
            refs.append(CrossReference(
                from_law=from_law_id, from_article=from_article,
                to_law=from_law_id, to_article=art_num,
                ref_type=RefType.SAME_LAW, resolved=True,
                context=_ctx(text, s, e),
            ))
            consumed.append((s, e))
            return
        to_law = _resolve_law(phrase, registry)
        same = (to_law == from_law_id)
        refs.append(CrossReference(
            from_law=from_law_id, from_article=from_article,
            to_law=to_law, to_article=art_num,
            ref_type=RefType.SAME_LAW if same else RefType.CROSS_LAW,
            resolved=to_law is not None,
            context=_ctx(text, s, e),
        ))
        consumed.append((s, e))

    # 1. article-first law-scoped refs
    for m in _ART_FIRST.finditer(text):
        _add_lawscoped(int(m.group(1)), m.group(2), m.start(), m.end())
    # 2. law-first law-scoped refs
    for m in _LAW_FIRST.finditer(text):
        _add_lawscoped(int(m.group(2)), m.group(1), m.start(), m.end())

    # 3. decree refs
    for m in _DECREE.finditer(text):
        refs.append(CrossReference(
            from_law=from_law_id, from_article=from_article,
            to_law=None, to_article=None, ref_type=RefType.DECREE,
            resolved=False, context=_ctx(text, m.start(), m.end()),
        ))

    # 4. bare same-law article refs (not inside a consumed law-scoped span)
    for m in _BARE_ART.finditer(text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        n = int(m.group(1))
        if drop_self and n == from_article:
            continue
        refs.append(CrossReference(
            from_law=from_law_id, from_article=from_article,
            to_law=from_law_id, to_article=n, ref_type=RefType.SAME_LAW,
            resolved=True, context=_ctx(text, m.start(), m.end()),
        ))

    # final safety net: drop exact self-references (same law + same article)
    if drop_self:
        refs = [r for r in refs
                if not (r.to_law == from_law_id and r.to_article == from_article)]
    return refs


def build_reference_graph(articles: list[dict], registry: dict[str, str]) -> dict:
    """Build the full graph. Each article dict: law_id, article_number, text."""
    all_refs: list[CrossReference] = []
    for art in articles:
        all_refs.extend(extract_references(
            from_law_id=art["law_id"],
            from_article=int(art["article_number"]),
            article_text=art["text"] or "",
            registry=registry,
        ))
    stats = {
        "total_references": len(all_refs),
        "same_law": sum(1 for r in all_refs if r.ref_type == RefType.SAME_LAW),
        "cross_law": sum(1 for r in all_refs if r.ref_type == RefType.CROSS_LAW),
        "decree": sum(1 for r in all_refs if r.ref_type == RefType.DECREE),
        "unresolved": sum(1 for r in all_refs if not r.resolved),
    }
    return {"references": [r.model_dump() for r in all_refs], "stats": stats}
