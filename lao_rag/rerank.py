"""
Lao Legal RAG — Step 4: LLM Reranker
====================================

Fixes the documented "high-confidence-wrong" failure (Part B #5): a high cosine
/ fusion rank can attach to the WRONG article. Observed live: the correct
article_64 sat at rank 5, BELOW article_55 (higher cosine, wrong topic) and a
0.000 title-only artifact. Fusion got the right answer into the candidate set
(recall) but could not tell it was the best one (precision).

The reranker judges relevance by READING each candidate against the question and
scoring it — replacing an inferred geometric signal (distance/rank) with a
judged one (does this text actually answer the question).

Design (locked decisions)
-------------------------
* SINGLE CALL, listwise: send all candidates in one prompt, ask Gemini to score
  each 0-10. One call (quota/latency friendly) and the model sees candidates
  TOGETHER so it can compare them, not just score in isolation.
* Pydantic-validated JSON output.
* GRACEFUL FALLBACK: on any failure (API error, malformed output, wrong count)
  return the ORIGINAL fusion order unchanged. The reranker is a refinement on a
  pipeline that already works; it must only ever improve or no-op, never break.

Injected LLM (LLMFn) -> unit-testable without keys.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from decompose import LLMFn, _extract_json  # reuse robust JSON extractor
from retrieval import RetrievedChunk

# How much candidate text to show the reranker (keep prompt bounded).
_SNIPPET_CHARS = 500


class RerankScore(BaseModel):
    index: int = Field(..., ge=0)
    score: float

    @field_validator("score")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(10.0, v))


class RerankResult(BaseModel):
    """Reranked chunks plus trace for UI/thesis."""

    query: str
    chunks: list[RetrievedChunk]
    reranked: bool = Field(..., description="True if LLM scores applied; False if fell back")
    scores: Optional[list[float]] = None  # aligned to returned chunks when reranked


def _build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """List all candidates with an index; ask for JSON scores 0-10."""
    lines = []
    for i, c in enumerate(chunks):
        title = c.citation.citation_string or "(no citation)"
        snippet = c.text.strip()[:_SNIPPET_CHARS].replace("\n", " ")
        lines.append(f"[{i}] {title}\n    {snippet}")
    candidates_block = "\n".join(lines)
    return f"""ທ່ານເປັນຜູ້ຊ່ຽວຊານກົດໝາຍລາວ. ໃຫ້ຄະແນນແຕ່ລະເອກະສານ ຕາມຄວາມກ່ຽວຂ້ອງ
ກັບຄຳຖາມ (0 = ບໍ່ກ່ຽວຂ້ອງເລີຍ, 10 = ຕອບຄຳຖາມໂດຍກົງ).

ຄຳຖາມ: {query}

ເອກະສານ:
{candidates_block}

ຕອບເປັນ JSON ເທົ່ານັ້ນ (ບໍ່ມີ markdown, ບໍ່ມີຄຳອະທິບາຍ). ໃຫ້ຄະແນນທຸກ index:
{{"scores": [{{"index": 0, "score": <0-10>}}, {{"index": 1, "score": <0-10>}}, ...]}}

JSON:"""


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    llm: LLMFn,
    *,
    top_n: Optional[int] = None,
) -> RerankResult:
    """Rerank `chunks` by LLM-judged relevance to `query`.

    Only the first `top_n` candidates are sent for reranking (cost bound); any
    beyond top_n keep their fusion order appended after the reranked block.
    Defaults to reranking all provided chunks.

    On ANY failure -> returns the original order (reranked=False).
    """
    if not chunks:
        return RerankResult(query=query, chunks=[], reranked=False)

    n = len(chunks) if top_n is None else min(top_n, len(chunks))
    head = chunks[:n]
    tail = chunks[n:]  # not reranked; appended unchanged

    try:
        raw = llm(_build_prompt(query, head))
        data = _extract_json(raw)
        raw_scores = data.get("scores", [])
        if not raw_scores:
            raise ValueError("reranker returned no scores")

        # parse + validate; build index->score map
        score_by_idx: dict[int, float] = {}
        for item in raw_scores:
            rs = RerankScore(**item)
            if 0 <= rs.index < len(head):
                score_by_idx[rs.index] = rs.score
        if not score_by_idx:
            raise ValueError("no valid indices in reranker output")

        # any head candidate the LLM didn't score: keep but rank last among head
        # by assigning -1 so it sinks below scored ones but stays in results.
        order = sorted(
            range(len(head)),
            key=lambda i: score_by_idx.get(i, -1.0),
            reverse=True,
        )
        reranked_head = [head[i] for i in order]
        aligned_scores = [score_by_idx.get(i, -1.0) for i in order]

        return RerankResult(
            query=query,
            chunks=reranked_head + tail,
            reranked=True,
            scores=aligned_scores + [None] * len(tail) if tail else aligned_scores,
        )
    except Exception:
        # GRACEFUL FALLBACK: original fusion order, untouched.
        return RerankResult(query=query, chunks=chunks, reranked=False)
