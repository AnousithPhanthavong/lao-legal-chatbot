"""
Lao Legal RAG — Step 1: Retrieval Tool Layer
=============================================

Typed, testable wrappers over the two ChromaDB collections:
    - `lao_legal`          (75 laws, ~6,758 chunks)
    - `lao_regulations_tax`(18 tax regs, ~497 typed chunks)

Design principles
-----------------
1. **Dependency injection for the embedder.** Every search function takes an
   `embed_query: EmbedQueryFn` callable. In production you pass the Gemini
   `gemini-embedding-001` embedder (task_type=RETRIEVAL_QUERY, 768d, L2-norm).
   In tests you pass a deterministic fake. The retrieval logic never imports
   Gemini, so it is fully unit-testable without API keys.

2. **Typed results (Pydantic v2).** Chroma returns parallel lists of dicts;
   downstream agent steps must never touch that shape. `RetrievedChunk`
   re-assembles a flat Chroma metadata row back into a structured `Citation`
   + chunk, validated.

3. **RRF for cross-collection fusion.** Cosine scores from two collections are
   NOT comparable (different content distributions). `search_both` fuses by
   *rank* (Reciprocal Rank Fusion), which is scale-free and the correct way to
   merge heterogeneous ranked lists.

This module is the FOUNDATION layer. Steps 2-5 (decomposition, normalization,
rerank, agent loop) are built on top of these three functions and never call
ChromaDB directly.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Embedding contract (must match A.2 in the session handoff)
# --------------------------------------------------------------------------- #
EMBED_DIM = 768

# An embedder takes a query string and returns a 768-d, L2-normalized vector.
EmbedQueryFn = Callable[[str], Sequence[float]]


class ChromaCollection(Protocol):
    """Structural type for the bits of a chromadb Collection we use.

    Declaring a Protocol (instead of importing chromadb's concrete type) keeps
    this module import-light and lets tests pass a real in-memory collection or
    a stub interchangeably.
    """

    def query(  # noqa: D401 - matches chromadb signature
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int,
        where: Optional[dict] = ...,
        include: Sequence[str] = ...,
    ) -> dict: ...


# --------------------------------------------------------------------------- #
# Result models
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    """Re-assembled from the FLAT Chroma metadata keys (A.3).

    All fields optional because law chunks and regulation chunks carry
    different subsets of citation metadata; `citation_string` is the one field
    the handoff guarantees is ALWAYS populated.
    """

    citation_string: str = ""
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    regulation_name: Optional[str] = None
    document_number: Optional[str] = None
    issuing_authority: Optional[str] = None
    issuance_year: Optional[str] = None
    parent_law_id: Optional[str] = None
    parent_law_name: Optional[str] = None
    source_page_filenames: Optional[str] = None


class RetrievedChunk(BaseModel):
    """A single retrieval hit, fully typed for downstream agent steps."""

    # Same reload-trap guard as SearchResult: re-coerce nested models from
    # attributes rather than strict isinstance(), so a Citation built against a
    # previously-loaded copy of the class still validates after importlib.reload.
    model_config = {"revalidate_instances": "always"}

    id: str
    text: str = Field(..., description="content_verbatim, for display/citation")
    collection: str = Field(..., description="'lao_legal' or 'lao_regulations_tax'")
    score: float = Field(..., description="cosine similarity in [0,1]; 1.0 = identical")
    distance: float = Field(..., description="raw chroma cosine distance")
    chunk_type: Optional[str] = None
    is_regulation: bool = False
    provenance_tier: Optional[str] = None
    quarantine: bool = False
    citation: Citation

    @field_validator("citation", mode="before")
    @classmethod
    def _coerce_citation(cls, v):
        """Accept a Citation-like object from a stale-reloaded module by
        rebuilding it from its dumped fields."""
        if isinstance(v, Citation) or isinstance(v, dict):
            return v
        if hasattr(v, "model_dump"):
            return Citation(**v.model_dump())
        return v

    @field_validator("score")
    @classmethod
    def _score_in_range(cls, v: float) -> float:
        # Cosine distance can drift slightly outside [0,2] due to float error;
        # clamp the derived similarity so downstream code can trust [0,1].
        return max(0.0, min(1.0, v))


class SearchResult(BaseModel):
    """Container returned by every search_* function."""

    # `revalidate_instances='always'` makes Pydantic re-coerce each chunk from
    # its attributes rather than doing a strict identity isinstance() check.
    # This prevents the `importlib.reload` stale-class trap: if a chunk was
    # built against a previously-loaded copy of RetrievedChunk (because one
    # module was reloaded and a dependent wasn't), it is structurally identical
    # and will still validate here instead of raising a confusing
    # "RetrievedChunk is not a RetrievedChunk" error.
    model_config = {"revalidate_instances": "always"}

    query: str
    chunks: list[RetrievedChunk]

    @field_validator("chunks", mode="before")
    @classmethod
    def _coerce_chunks(cls, v):
        """Accept RetrievedChunk-like objects from a stale-reloaded module by
        rebuilding them from their dumped fields."""
        if not isinstance(v, (list, tuple)):
            return v
        out = []
        for item in v:
            if isinstance(item, RetrievedChunk):
                out.append(item)
            elif isinstance(item, dict):
                out.append(item)
            elif hasattr(item, "model_dump"):
                # a RetrievedChunk from a different class-version: rebuild it
                out.append(RetrievedChunk(**item.model_dump()))
            else:
                out.append(item)  # let pydantic raise a clear error
        return out

    def ids(self) -> list[str]:
        return [c.id for c in self.chunks]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _truthy(value) -> bool:
    """Chroma flattens booleans inconsistently (True / 'true' / 'True' / 1).

    Normalize defensively so `quarantine`/`is_regulation` are reliable.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _citation_from_meta(meta: dict) -> Citation:
    """Re-assemble a Citation from FLAT Chroma metadata keys."""
    return Citation(
        citation_string=meta.get("citation_string", "") or "",
        article_number=meta.get("article_number"),
        article_title=meta.get("article_title"),
        regulation_name=meta.get("regulation_name"),
        document_number=meta.get("document_number"),
        issuing_authority=meta.get("issuing_authority"),
        issuance_year=meta.get("issuance_year"),
        parent_law_id=meta.get("parent_law_id"),
        parent_law_name=meta.get("parent_law_name"),
        source_page_filenames=meta.get("source_page_filenames"),
    )


def _parse_chroma_response(resp: dict, collection_name: str) -> list[RetrievedChunk]:
    """Turn Chroma's parallel-list response into typed RetrievedChunk objects.

    Chroma returns {ids: [[...]], documents: [[...]], metadatas: [[...]],
    distances: [[...]]} — one inner list per query. We always query with a
    single embedding, so we take index [0]. Missing keys are handled so a
    collection queried with a narrower `include` still parses.
    """
    ids = (resp.get("ids") or [[]])[0]
    docs = (resp.get("documents") or [[]])[0]
    metas = (resp.get("metadatas") or [[]])[0]
    dists = (resp.get("distances") or [[]])[0]

    out: list[RetrievedChunk] = []
    for i, _id in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        distance = dists[i] if i < len(dists) else 0.0
        # cosine distance -> similarity. chroma cosine distance = 1 - cosine_sim.
        score = 1.0 - float(distance)
        out.append(
            RetrievedChunk(
                id=_id,
                text=(docs[i] if i < len(docs) else "") or "",
                collection=collection_name,
                score=score,
                distance=float(distance),
                chunk_type=meta.get("chunk_type"),
                is_regulation=_truthy(meta.get("is_regulation")),
                provenance_tier=meta.get("provenance_tier"),
                quarantine=_truthy(meta.get("quarantine")),
                citation=_citation_from_meta(meta),
            )
        )
    return out


def _embed_as_matrix(embed_query: EmbedQueryFn, query: str) -> list[list[float]]:
    """Embed a query and shape it as Chroma expects: a list of vectors."""
    vec = list(embed_query(query))
    if len(vec) != EMBED_DIM:
        raise ValueError(
            f"Embedder returned dim {len(vec)}, expected {EMBED_DIM}. "
            "Query embedding must match the document embedding contract (A.2)."
        )
    return [vec]


# --------------------------------------------------------------------------- #
# Public search functions
# --------------------------------------------------------------------------- #
def search_regulations(
    query: str,
    embed_query: EmbedQueryFn,
    collection: ChromaCollection,
    k: int = 5,
    *,
    filters: Optional[dict] = None,
    drop_quarantined: bool = True,
) -> SearchResult:
    """Search the `lao_regulations_tax` collection.

    Parameters
    ----------
    filters : optional Chroma `where` clause, e.g. {"chunk_type": "rate_table"}
        or {"parent_law_id": "..."}. Enables type-aware retrieval for the
        tax-calculator tool (Step 6) and law-scoped retrieval.
    drop_quarantined : exclude chunks flagged `quarantine=True` (garbled OCR).
        Done post-hoc rather than via `where` so it composes with any filter and
        does not depend on how the bool was flattened in metadata.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    embeddings = _embed_as_matrix(embed_query, query)
    # Over-fetch if we will drop quarantined, so we still return ~k good chunks.
    n = k * 2 if drop_quarantined else k
    resp = collection.query(
        query_embeddings=embeddings,
        n_results=n,
        where=filters,
        include=["documents", "metadatas", "distances"],
    )
    chunks = _parse_chroma_response(resp, "lao_regulations_tax")
    if drop_quarantined:
        chunks = [c for c in chunks if not c.quarantine]
    return SearchResult(query=query, chunks=chunks[:k])


def search_laws(
    query: str,
    embed_query: EmbedQueryFn,
    collection: ChromaCollection,
    k: int = 5,
    *,
    filters: Optional[dict] = None,
) -> SearchResult:
    """Search the `lao_legal` collection (75 laws)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    embeddings = _embed_as_matrix(embed_query, query)
    resp = collection.query(
        query_embeddings=embeddings,
        n_results=k,
        where=filters,
        include=["documents", "metadatas", "distances"],
    )
    chunks = _parse_chroma_response(resp, "lao_legal")
    return SearchResult(query=query, chunks=chunks[:k])


def _reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedChunk]],
    k: int,
    *,
    rrf_k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> list[RetrievedChunk]:
    """Merge multiple ranked lists by (optionally weighted) Reciprocal Rank Fusion.

    RRF score for a doc = sum over lists of weight_i / (rrf_k + rank), rank
    0-based. rrf_k=60 is the standard constant from Cormack et al. (2009); it
    damps the influence of very high ranks so a single list cannot dominate.
    Fusing by RANK (not raw cosine score) is essential because score
    distributions across lists/collections are not comparable.

    `weights` (optional): one weight per ranked list. A higher weight makes that
    list's votes count more — used to prioritize a high-fidelity source (e.g. the
    full-query result) over many lower-fidelity sub-query lists, preventing a
    concentrated strong signal from being out-voted by diffuse weak ones
    ("over-fusion"). Defaults to equal weight 1.0 for every list (unchanged
    behavior).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length {len(weights)} != number of lists {len(ranked_lists)}"
        )
    fused: dict[str, float] = {}
    best_obj: dict[str, RetrievedChunk] = {}
    for w, ranked in zip(weights, ranked_lists):
        for rank, chunk in enumerate(ranked):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + w / (rrf_k + rank)
            # Keep the instance with the higher cosine score for display.
            if chunk.id not in best_obj or chunk.score > best_obj[chunk.id].score:
                best_obj[chunk.id] = chunk
    ordered_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [best_obj[cid] for cid in ordered_ids[:k]]


def search_both(
    query: str,
    embed_query: EmbedQueryFn,
    laws_collection: ChromaCollection,
    regulations_collection: ChromaCollection,
    k: int = 5,
    *,
    drop_quarantined: bool = True,
) -> SearchResult:
    """Cross-instrument search: query BOTH collections, fuse with RRF.

    This is the function the agent uses when a (sub-)query may be answered by
    either a law or a regulation, or needs both (e.g. a regulation implementing
    a parent law). Returns a single RRF-merged, de-duplicated ranked list.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    law_res = search_laws(query, embed_query, laws_collection, k=k)
    reg_res = search_regulations(
        query, embed_query, regulations_collection, k=k,
        drop_quarantined=drop_quarantined,
    )
    fused = _reciprocal_rank_fusion([law_res.chunks, reg_res.chunks], k=k)
    return SearchResult(query=query, chunks=fused)
