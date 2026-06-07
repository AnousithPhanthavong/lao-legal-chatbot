"""
Lao Legal RAG — Agent Retrieval Orchestration
==============================================

Combines every retrieval layer into ONE call, fixing the regression where
decomposition DISCARDED the original query and thereby lost the whole-phrase
title signal that recovers single-article answers (the article_64 case).

Key principle (learned empirically)
-----------------------------------
Decomposition helps when the answer is SPREAD across multiple articles
(true comparison/compound). It HURTS when the answer is a SINGLE article whose
title covers all sub-topics — splitting destroys the whole-phrase title match.

Since we cannot know which shape a query has until after retrieval, we ALWAYS
retrieve on BOTH:
    * the full original query  (catches single-article-titled-with-all-subtopics)
    * each sub-query           (catches answer-spread-across-articles)
each spelling-expanded, all fused by RRF. We never rely on confidence scores to
decide (they are documented as unreliable — "high-confidence-wrong").

The combined candidate set is intentionally WIDE (high recall). The Step 4
reranker is the precision cleanup that follows.

Depends only on the existing tested modules via injected callables
(EmbedQueryFn, LLMFn), so this stays unit-testable without keys.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

from typing import Optional

from dataclasses import dataclass, field

from decompose import Decomposition, LLMFn, decompose
from retrieval import (
    ChromaCollection,
    EmbedQueryFn,
    RetrievedChunk,
    SearchResult,
    _reciprocal_rank_fusion,
)
from spelling_expand import expand_if_needed
from title_boost import TitleIndex, search_with_title_boost


@dataclass
class AgentRetrievalResult:
    """Full retrieval result. Plain dataclass — no Pydantic class-identity
    validation — so it survives multi-folder import splits on Colab."""
    query: str
    decomposition: Decomposition
    retrieval_forms: list[str]
    chunks: list[RetrievedChunk] = field(default_factory=list)

    def ids(self) -> list[str]:
        return [c.id for c in self.chunks]


def _forms_for(query: str) -> list[str]:
    """Spelling-expanded forms for a single query string."""
    return expand_if_needed(query)


def agent_retrieve(
    query: str,
    embed_query: EmbedQueryFn,
    llm: LLMFn,
    collection: ChromaCollection,
    title_index: TitleIndex,
    k: int = 5,
    *,
    candidate_k: int = 10,
    decompose_query: bool = True,
    full_query_weight: float = 2.0,
    sub_query_weight: float = 1.0,
) -> AgentRetrievalResult:
    """Decompose + always-include-full-query + spelling-expand + title-boost, RRF.

    Retrieval forms = spelling-variants of (full query)
                    + spelling-variants of (each sub-query).
    Each form runs dense+title-boost; all resulting ranked lists are fused with
    WEIGHTED RRF.

    Over-fusion fix
    ---------------
    Fusing many sub-query lists can out-vote a concentrated strong signal from
    the full query (a chunk that appears once at excellent rank loses to chunks
    appearing in many lists at mediocre rank). We therefore weight the
    full-query lists higher (`full_query_weight`) than sub-query lists
    (`sub_query_weight`). The full query is the higher-fidelity signal because
    its title can match a single-article answer whose title covers all
    sub-topics. Weights are a TUNABLE knob — sweep on the eval set and report the
    value maximizing average Recall@k, rather than fitting one question.
    """
    decomp = (
        decompose(query, llm)
        if decompose_query
        else _atomic_decomposition(query)
    )

    # Build base queries, tagged by origin (full vs sub) so we can weight them.
    # (query_text, is_full_query)
    base_queries: list[tuple[str, bool]] = [(query, True)]
    if decomp.needs_multiple_retrievals:
        for sq in decomp.texts():
            if sq != query:
                base_queries.append((sq, False))

    # expand each into spelling forms; dedupe globally but remember origin
    forms: list[str] = []
    form_is_full: list[bool] = []
    seen: set[str] = set()
    for bq, is_full in base_queries:
        for form in _forms_for(bq):
            if form not in seen:
                seen.add(form)
                forms.append(form)
                form_is_full.append(is_full)

    # retrieve dense+title-boost per form; track each list's origin weight
    ranked_lists: list[list[RetrievedChunk]] = []
    list_weights: list[float] = []
    for form, is_full in zip(forms, form_is_full):
        res = search_with_title_boost(
            form, embed_query, collection, title_index,
            k=candidate_k, candidate_k=candidate_k,
        )
        if res.chunks:
            ranked_lists.append(res.chunks)
            list_weights.append(full_query_weight if is_full else sub_query_weight)

    if not ranked_lists:
        fused: list[RetrievedChunk] = []
    elif len(ranked_lists) == 1:
        fused = ranked_lists[0][:k]
    else:
        fused = _reciprocal_rank_fusion(
            ranked_lists, k=k, weights=list_weights
        )

    return AgentRetrievalResult(
        query=query,
        decomposition=decomp,
        retrieval_forms=forms,
        chunks=fused,
    )


def agent_retrieve_both(
    query: str,
    embed_query: EmbedQueryFn,
    llm: LLMFn,
    laws_collection: ChromaCollection,
    regulations_collection: ChromaCollection,
    laws_title_index: TitleIndex,
    regs_title_index: TitleIndex,
    k: int = 5,
    *,
    candidate_k: int = 10,
    decompose_query: bool = True,
    full_query_weight: float = 2.0,
    sub_query_weight: float = 1.0,
) -> AgentRetrievalResult:
    """Same orchestration as agent_retrieve, but searches BOTH collections
    (75 laws + 18 regulations) and fuses everything together.

    For each retrieval form we run title-boosted search on EACH collection and
    treat the two ranked lists as additional inputs to the weighted RRF. This
    lets a single query surface a regulation article and its parent-law article
    side by side — the full-corpus behaviour.

    NOTE: retrieval was formally evaluated on the tax-regulation slice; the
    laws collection is covered here but not separately benchmarked.
    """
    decomp = (
        decompose(query, llm) if decompose_query
        else _atomic_decomposition(query)
    )

    base_queries: list[tuple[str, bool]] = [(query, True)]
    if decomp.needs_multiple_retrievals:
        for sq in decomp.texts():
            if sq != query:
                base_queries.append((sq, False))

    forms: list[str] = []
    form_is_full: list[bool] = []
    seen: set[str] = set()
    for bq, is_full in base_queries:
        for form in _forms_for(bq):
            if form not in seen:
                seen.add(form)
                forms.append(form)
                form_is_full.append(is_full)

    ranked_lists: list[list[RetrievedChunk]] = []
    list_weights: list[float] = []
    for form, is_full in zip(forms, form_is_full):
        w = full_query_weight if is_full else sub_query_weight
        # search BOTH collections for this form
        for coll, tindex in ((regulations_collection, regs_title_index),
                             (laws_collection, laws_title_index)):
            res = search_with_title_boost(
                form, embed_query, coll, tindex,
                k=candidate_k, candidate_k=candidate_k,
            )
            if res.chunks:
                ranked_lists.append(res.chunks)
                list_weights.append(w)

    if not ranked_lists:
        fused: list[RetrievedChunk] = []
    elif len(ranked_lists) == 1:
        fused = ranked_lists[0][:k]
    else:
        fused = _reciprocal_rank_fusion(ranked_lists, k=k, weights=list_weights)

    return AgentRetrievalResult(
        query=query,
        decomposition=decomp,
        retrieval_forms=forms,
        chunks=fused,
    )


def _atomic_decomposition(query: str) -> Decomposition:
    """Build a no-split Decomposition (used when decompose_query=False)."""
    from decompose import DecompositionKind, SubQuery

    return Decomposition(
        original=query,
        kind=DecompositionKind.ATOMIC,
        sub_queries=[SubQuery(text=query)],
        rule_fired=False,
        llm_called=False,
    )
