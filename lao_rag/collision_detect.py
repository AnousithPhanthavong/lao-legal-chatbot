"""
Lao Legal RAG — Title-Collision Detector
=========================================

A reusable data-quality tool that finds and CLASSIFIES articles sharing the same
title. Surfaced empirically: 27 colliding titles in the tax corpus, which
invalidated naive query-from-document eval questions (a question generated from a
repeated title cannot uniquely identify its source article).

Three collision classes, each handled differently downstream
------------------------------------------------------------
1. OCR_GARBAGE  — "title" is table pipes / punctuation / OCR noise, not prose.
                  -> exclude from eval set AND from the title index.
2. BOILERPLATE  — same title across DIFFERENT laws (ຈຸດປະສົງ "objectives",
                  ການຈັດຕັ້ງປະຕິບັດ "implementation", ຜົນສັກສິດ "effect").
                  Structurally guaranteed in every legal instrument.
                  -> eval needs law-context to disambiguate; set-credit within law.
3. CATEGORY_REPEAT — same title, SAME law, different articles (e.g. the income-tax
                  "ພື້ນຖານຄິດໄລ່, ອັດຕາ..." repeated per taxpayer category).
                  A genuine answer FAMILY.
                  -> set-credit: any family member counts unless question is specific.

Classification is DETERMINISTIC (character-class ratios + grouping by law), the
same orthographic-statistical approach as the garble detector — no LLM, fully
reproducible and defensible at defense.

Thesis contribution: "title-collision detection in query-from-document
evaluation for repeated-structure legal instruments."

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

import re
from collections import defaultdict
from enum import Enum
from typing import Optional

from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Title extraction (article header) — mirrors title_boost.extract_title intent
# --------------------------------------------------------------------------- #
_ARTICLE_HEADER = re.compile(r"^\s*ມາດຕາ\s*\d+\s*(.*)")
_LAO_LO, _LAO_HI = 0x0E80, 0x0EFF


def _is_lao(ch: str) -> bool:
    return _LAO_LO <= ord(ch) <= _LAO_HI


def extract_title_phrase(document_body: str, max_chars: int = 60) -> str:
    """First line's title phrase (after 'ມາດຕາ N'), trimmed."""
    first = ""
    for line in (document_body or "").splitlines():
        if line.strip():
            first = line.strip()
            break
    m = _ARTICLE_HEADER.match(first)
    phrase = m.group(1) if m else first
    return phrase[:max_chars].strip()


def lao_ratio(text: str) -> float:
    """Fraction of non-space characters that are Lao. Low => likely garbage."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    lao = sum(1 for c in chars if _is_lao(c))
    return lao / len(chars)


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
class CollisionClass(str, Enum):
    OCR_GARBAGE = "ocr_garbage"
    BOILERPLATE = "boilerplate"           # same title across different laws
    CATEGORY_REPEAT = "category_repeat"   # same title within one law
    UNIQUE = "unique"                     # not a collision


class TitleGroup(BaseModel):
    title: str
    chunk_ids: list[str]
    law_ids: list[Optional[str]]
    collision_class: CollisionClass

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    @property
    def distinct_laws(self) -> int:
        return len({l for l in self.law_ids if l})


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def _classify(title: str, law_ids: list[Optional[str]], *,
              garbage_lao_threshold: float = 0.5) -> CollisionClass:
    """Deterministic classification of one title group."""
    # 1. garbage: title has too little Lao content (pipes/punctuation/OCR noise)
    if lao_ratio(title) < garbage_lao_threshold:
        return CollisionClass.OCR_GARBAGE
    if len(title) < 2:
        return CollisionClass.OCR_GARBAGE

    n = len(law_ids)
    if n < 2:
        return CollisionClass.UNIQUE

    distinct = len({l for l in law_ids if l})
    # 2. boilerplate: spans multiple laws
    if distinct > 1:
        return CollisionClass.BOILERPLATE
    # 3. category-repeat: many chunks, one law
    return CollisionClass.CATEGORY_REPEAT


def detect_collisions(
    chunk_ids: list[str],
    documents: list[str],
    law_ids: list[Optional[str]],
    *,
    title_chars: int = 50,
    garbage_lao_threshold: float = 0.5,
) -> dict[str, TitleGroup]:
    """Group chunks by extracted title and classify each group.

    Inputs are parallel lists (as returned by chromadb .get). `law_ids` is the
    per-chunk parent_law_id (or any law-grouping key); pass None where unknown.

    Returns {title: TitleGroup} for groups of size >= 2 (the collisions). Unique
    titles are omitted (they are fine).
    """
    by_title: dict[str, list[int]] = defaultdict(list)
    titles: list[str] = []
    for i, doc in enumerate(documents):
        t = extract_title_phrase(doc or "", max_chars=title_chars)
        titles.append(t)
        if t:
            by_title[t].append(i)

    groups: dict[str, TitleGroup] = {}
    for title, idxs in by_title.items():
        if len(idxs) < 2:
            continue
        grp_chunk_ids = [chunk_ids[i] for i in idxs]
        grp_law_ids = [law_ids[i] for i in idxs]
        cls = _classify(title, grp_law_ids,
                        garbage_lao_threshold=garbage_lao_threshold)
        groups[title] = TitleGroup(
            title=title,
            chunk_ids=grp_chunk_ids,
            law_ids=grp_law_ids,
            collision_class=cls,
        )
    return groups


def build_answer_families(
    groups: dict[str, TitleGroup],
) -> dict[str, set[str]]:
    """Map each chunk_id -> the SET of chunk_ids that are valid equivalents.

    Used by the collision-aware evaluator for set-credit: a retrieval that finds
    ANY member of a chunk's family counts as correct for ambiguous questions.

    Garbage groups produce NO families (those chunks should be excluded, not
    set-credited). Boilerplate families are scoped within the same law (objectives
    of law A != objectives of law B). Category-repeat families include all members
    (same law, same title).
    """
    families: dict[str, set[str]] = {}
    for grp in groups.values():
        if grp.collision_class == CollisionClass.OCR_GARBAGE:
            continue
        if grp.collision_class == CollisionClass.BOILERPLATE:
            # group members by law; family = same-law members only
            by_law: dict[Optional[str], list[str]] = defaultdict(list)
            for cid, lid in zip(grp.chunk_ids, grp.law_ids):
                by_law[lid].append(cid)
            for members in by_law.values():
                fam = set(members)
                for cid in members:
                    families[cid] = fam
        else:  # CATEGORY_REPEAT
            fam = set(grp.chunk_ids)
            for cid in grp.chunk_ids:
                families[cid] = fam
    return families


def garbage_chunk_ids(groups: dict[str, TitleGroup]) -> set[str]:
    """All chunk_ids whose title was classified OCR_GARBAGE (exclude from eval)."""
    out: set[str] = set()
    for grp in groups.values():
        if grp.collision_class == CollisionClass.OCR_GARBAGE:
            out.update(grp.chunk_ids)
    return out


def summary(groups: dict[str, TitleGroup]) -> dict[str, int]:
    """Counts per collision class, for reporting / thesis."""
    counts: dict[str, int] = defaultdict(int)
    for grp in groups.values():
        counts[grp.collision_class.value] += 1
    return dict(counts)
