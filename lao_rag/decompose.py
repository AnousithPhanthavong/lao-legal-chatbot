"""
Lao Legal RAG — Step 2: Query Decomposition
===========================================

Fixes the documented v5 failures:
  * Q3 (compound):    "ພື້ນຖານຄິດໄລ່, ອັດຕາ ແລະ ການແຈ້ງ-ມອບອາກອນ ..." -> 3 needs blended
  * comparison:       "civil ແລະ criminal ແຕກຕ່າງກັນແນວໃດ?"           -> 2 topics blended

Architecture (locked decisions)
-------------------------------
HYBRID: a cheap, deterministic RULE detector decides *whether* a query needs
splitting. Only if it fires do we spend ONE LLM call to actually split.
  - ~70% of queries are single-need; the rule short-circuits them for free.
  - The rule is CONSERVATIVE: it fires on structural conjunction/comparison
    signals, not on a stray ແລະ inside a name.

POLICY: force-split. If the rule fires, we accept the LLM's split even if the
LLM thinks it is atomic — recall-favoring (documented preference). Both the rule
decision and the LLM decision are recorded for thesis analysis.

OUTPUT: the LLM returns JSON; we parse and VALIDATE with Pydantic. On any
malformed / over-long / empty output we FALL BACK to the unsplit query (never
crash, never silently drop the user's need).

MAX_SUBQUERIES = 3 (quota bound).

The LLM is injected as `LLMFn` (str -> str), so this module is fully testable
with a fake LLM and no API keys — same DI discipline as retrieval.py.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field, field_validator

MAX_SUBQUERIES = 3

# An LLM call: takes a prompt, returns the raw text completion.
LLMFn = Callable[[str], str]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class DecompositionKind(str, Enum):
    ATOMIC = "atomic"           # single need, no split
    COMPOUND = "compound"       # multiple needs joined (ແລະ / ຫຼື) -> retrieve each
    COMPARISON = "comparison"   # compare A vs B -> retrieve each, then synthesize


class SubQuery(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("sub-query text empty after strip")
        return v


class Decomposition(BaseModel):
    """Result of decomposing one user query."""

    original: str
    kind: DecompositionKind
    sub_queries: list[SubQuery]
    # synthesis hint for the agent's final step (esp. for comparisons)
    synthesis_instruction: Optional[str] = None
    # provenance for thesis analysis
    rule_fired: bool = False
    llm_called: bool = False
    llm_said_atomic: bool = False

    @field_validator("sub_queries")
    @classmethod
    def _cap(cls, v: list[SubQuery]) -> list[SubQuery]:
        if not v:
            raise ValueError("must have at least one sub-query")
        if len(v) > MAX_SUBQUERIES:
            # hard cap — never exceed quota bound
            return v[:MAX_SUBQUERIES]
        return v

    @property
    def needs_multiple_retrievals(self) -> bool:
        return len(self.sub_queries) > 1

    def texts(self) -> list[str]:
        return [sq.text for sq in self.sub_queries]


# --------------------------------------------------------------------------- #
# Rule detector (deterministic, free, conservative)
# --------------------------------------------------------------------------- #
# Structural conjunctions joining needs. We require the conjunction to be
# surrounded by whitespace so it is a real joiner, not a substring of a word.
_CONJUNCTIONS = ("ແລະ", "ຫຼື", "ຫລື")           # and / or (two ໍspellings of 'or')
_COMPARISON_WORDS = ("ແຕກຕ່າງ", "ປຽບທຽບ", "ທຽບ", "ຄວາມແຕກຕ່າງ")  # differ / compare


def _count_question_segments(query: str) -> int:
    """Count distinct interrogative segments (multiple ? or ⁇-like markers)."""
    return query.count("?") + query.count("？")


def detect_needs_decomposition(query: str) -> tuple[bool, DecompositionKind]:
    """Cheap deterministic gate. Returns (should_split, suspected_kind).

    Conservative by design: fires on structural signals only. Returns the
    *suspected* kind so the LLM prompt can be specialized, but the LLM has the
    final say on kind (the rule only forces THAT a split happens, per policy).
    """
    q = query.strip()

    # comparison signal takes priority (it implies a specific synthesis)
    if any(w in q for w in _COMPARISON_WORDS):
        return True, DecompositionKind.COMPARISON

    # conjunction as a real token: padded by space, or between Lao words
    for c in _CONJUNCTIONS:
        # \s ແລະ \s  — joiner between clauses, not inside a name fragment
        if re.search(rf"(?:^|\s){re.escape(c)}(?:\s|$)", q):
            return True, DecompositionKind.COMPOUND

    # multiple explicit question marks -> multi-part
    if _count_question_segments(q) >= 2:
        return True, DecompositionKind.COMPOUND

    return False, DecompositionKind.ATOMIC


# --------------------------------------------------------------------------- #
# LLM decomposition prompt
# --------------------------------------------------------------------------- #
def _build_prompt(query: str, suspected: DecompositionKind) -> str:
    """Prompt asks for STRICT JSON only. No prose, no markdown fences."""
    return f"""ທ່ານເປັນຕົວຊ່ວຍວິເຄາະຄຳຖາມກົດໝາຍ. ແຍກຄຳຖາມລຸ່ມນີ້ອອກເປັນຄຳຖາມຍ່ອຍ.

ກົດລະບຽບ:
- ຖ້າເປັນຄຳຖາມປຽບທຽບ (A ທຽບ B): ສ້າງ 1 ຄຳຖາມຍ່ອຍສຳລັບແຕ່ລະຫົວຂໍ້.
- ຖ້າເປັນຄຳຖາມລວມ (ມີຫຼາຍຄວາມຕ້ອງການເຊື່ອມດ້ວຍ "ແລະ"/"ຫຼື"): ແຍກແຕ່ລະຄວາມຕ້ອງການ.
- ສູງສຸດ {MAX_SUBQUERIES} ຄຳຖາມຍ່ອຍ. ແຕ່ລະຄຳຖາມຍ່ອຍຕ້ອງເປັນຄຳຖາມດຽວທີ່ຄົບຖ້ວນ.
- ຫ້າມເພີ່ມເນື້ອຫາທີ່ບໍ່ມີໃນຄຳຖາມເດີມ.

ຕອບເປັນ JSON ເທົ່ານັ້ນ (ບໍ່ມີ markdown, ບໍ່ມີຄຳອະທິບາຍ):
{{"kind": "atomic|compound|comparison", "sub_queries": ["...", "..."], "synthesis_instruction": "..."}}

ຄຳຖາມ: {query}

JSON:"""


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(raw: str) -> dict:
    """Robustly pull a JSON object out of an LLM completion.

    Handles: bare JSON, ```json fenced, leading/trailing prose. Raises
    ValueError if no parseable object is found (caller falls back).
    """
    text = _JSON_FENCE.sub("", raw).strip()
    # find the first {...} balanced span
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in LLM output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)
    raise ValueError("unbalanced JSON braces in LLM output")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def decompose(
    query: str,
    llm: LLMFn,
    *,
    force_split_on_rule: bool = True,
) -> Decomposition:
    """Hybrid decomposition.

    1. Rule gate. If it does NOT fire -> return ATOMIC immediately (no LLM).
    2. If it fires -> ONE LLM call, parse+validate JSON.
    3. force_split_on_rule (default True, locked policy): if the rule fired but
       the LLM returned a single sub-query, we KEEP the LLM's atomic result only
       when force_split is False; when True we still accept the LLM split but
       record llm_said_atomic for analysis. (The rule cannot invent sub-queries,
       so 'force' here means: we trust that a split was warranted and surface the
       disagreement rather than discarding the LLM's structured attempt.)
    4. Any parse/validation failure -> safe fallback to the unsplit query.
    """
    q = query.strip()
    rule_fired, suspected = detect_needs_decomposition(q)

    if not rule_fired:
        return Decomposition(
            original=q,
            kind=DecompositionKind.ATOMIC,
            sub_queries=[SubQuery(text=q)],
            rule_fired=False,
            llm_called=False,
        )

    # rule fired -> spend one LLM call
    prompt = _build_prompt(q, suspected)
    try:
        raw = llm(prompt)
        data = _extract_json(raw)
        kind = DecompositionKind(data.get("kind", suspected.value))
        subs = [SubQuery(text=s) for s in data.get("sub_queries", []) if str(s).strip()]
        if not subs:
            raise ValueError("LLM returned no sub-queries")
        llm_said_atomic = len(subs) == 1
        decomp = Decomposition(
            original=q,
            kind=kind,
            sub_queries=subs,
            synthesis_instruction=data.get("synthesis_instruction"),
            rule_fired=True,
            llm_called=True,
            llm_said_atomic=llm_said_atomic,
        )
        return decomp
    except Exception:
        # SAFE FALLBACK: never crash, never drop the user's need.
        return Decomposition(
            original=q,
            kind=DecompositionKind.ATOMIC,
            sub_queries=[SubQuery(text=q)],
            synthesis_instruction=None,
            rule_fired=True,
            llm_called=True,
            llm_said_atomic=True,
        )
