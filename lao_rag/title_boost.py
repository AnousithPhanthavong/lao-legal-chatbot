"""
Lao Legal RAG — Step 3a: Title-Signal Retrieval Channel
=======================================================

Fixes the documented "long-chunk dilution" failure (Q3 / article_64):

A long article (e.g. ມາດຕາ 64, ~150 words covering 3 topics) embeds to a vector
whose "center of mass" is dominated by its body, so its short, high-signal
TITLE — which often mirrors the user's question almost verbatim — gets drowned.
Dense cosine search then ranks sharper single-topic articles above it.

This module adds a second, cheap retrieval channel that scores the QUERY against
chunk TITLES lexically (token overlap), then fuses it with the dense results via
Reciprocal Rank Fusion (already in retrieval.py). The title channel is exact-
keyword, which is exactly what recovers a near-verbatim title that the diluted
body-vector misses.

KEY DESIGN CONSTRAINT (from real data inspection)
-------------------------------------------------
In the live collection, `article_title` metadata is EMPTY and `article_number`
is None for many regulation chunks. The real title lives:
  1. at the START of the document body:  "ມາດຕາ 64 ພື້ນຖານຄິດໄລ່, ອັດຕາ ..."
  2. inside citation_string:              "ມາດຕາ 64, ຈັດຕັ້ງປະຕິບັດ..."
So we EXTRACT the title from the doc body's first line (+ citation_string),
never from the (empty) article_title field.

NO RE-EMBEDDING. The corpus stays frozen (thesis provenance intact). This is a
retrieval-layer enhancement built from metadata already in ChromaDB.

This is ONE layer of defense-in-depth: it helps when the answer chunk's title
shares vocabulary with the query. Queries phrased unlike any title are covered
by Step 3 expansion and the Step 4 reranker, not here.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from retrieval import (
    ChromaCollection,
    EmbedQueryFn,
    RetrievedChunk,
    SearchResult,
    _citation_from_meta,
    _reciprocal_rank_fusion,
    _truthy,
    search_regulations,
)

# --------------------------------------------------------------------------- #
# Title extraction
# --------------------------------------------------------------------------- #
# Matches a leading article header like "ມາດຕາ 64 <title text>".
# We capture the header line up to the first sentence-ish break so we get the
# TITLE, not the whole body.
_ARTICLE_HEADER = re.compile(r"^\s*(ມາດຕາ\s*\d+)\s*(.*)")


def extract_title(document_body: str, citation_string: str = "") -> str:
    """Best-effort title for a chunk, from body first line + citation_string.

    Strategy:
      * Take the first non-empty line of the body.
      * If it begins with 'ມາດຕາ N', keep that header + the short title phrase
        that follows on the same line (titles in this corpus are a comma-joined
        phrase BEFORE the body prose begins).
      * Always append citation_string tokens — they carry the article number and
        the instrument name, both useful lexical signals.
    """
    first_line = ""
    for line in document_body.splitlines():
        if line.strip():
            first_line = line.strip()
            break
    # Titles in this corpus run as a phrase then flow into body prose on the same
    # line; cut at the point where prose clearly starts. Heuristic: keep up to
    # the first ~120 chars of the header line — enough for the title phrase,
    # short enough to exclude most body dilution.
    m = _ARTICLE_HEADER.match(first_line)
    if m:
        header = first_line[:120]
    else:
        header = first_line[:120]
    return f"{header} {citation_string}".strip()


# --------------------------------------------------------------------------- #
# Lao tokenization for lexical overlap
# --------------------------------------------------------------------------- #
# Lao is scriptio continua (no spaces). For a LEXICAL title match we do not need
# linguistically perfect word segmentation — we need overlapping character-level
# signal. We use character n-grams (default trigrams) over Lao text, which is a
# robust, dependency-free way to measure surface similarity in spaceless scripts
# and is naturally tolerant to small morphological differences.
_LAO_RANGE = (0x0E80, 0x0EFF)


def _is_lao(ch: str) -> bool:
    return _LAO_RANGE[0] <= ord(ch) <= _LAO_RANGE[1]


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Character n-grams over the Lao+alphanumeric content of `text`.

    Whitespace/punctuation removed; ASCII letters/digits kept (for 'VAT', '5%').
    """
    kept = [c for c in text if _is_lao(c) or c.isalnum()]
    s = "".join(kept)
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def title_overlap_score(query: str, title: str, n: int = 3) -> float:
    """Jaccard overlap of character n-grams between query and title.

    Jaccard (intersection / union) in [0,1]; scale-free, so it composes with
    RRF the same way ranks do. A near-verbatim title scores very high.
    """
    q = char_ngrams(query, n)
    t = char_ngrams(title, n)
    if not q or not t:
        return 0.0
    inter = len(q & t)
    union = len(q | t)
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Title index
# --------------------------------------------------------------------------- #
class TitleEntry(BaseModel):
    id: str
    title: str
    quarantine: bool = False


class TitleIndex:
    """In-memory lexical index of chunk titles. Built once from a collection.

    Cheap: a single collection.get() of documents + metadatas, no embeddings.
    Rebuild on startup (same lifecycle as your ChromaDB rebuild-from-cache).
    """

    def __init__(self, entries: list[TitleEntry], ngram: int = 3):
        self._entries = entries
        self._ngram = ngram
        # precompute ngram sets for speed
        self._grams: dict[str, set[str]] = {
            e.id: char_ngrams(e.title, ngram) for e in entries
        }
        self._title_of: dict[str, str] = {e.id: e.title for e in entries}
        self._quarantined: set[str] = {e.id for e in entries if e.quarantine}

    @classmethod
    def from_collection(
        cls, collection: ChromaCollection, ngram: int = 3
    ) -> "TitleIndex":
        # pull everything; titles are tiny, this is a one-time startup cost
        raw = collection.get(include=["documents", "metadatas"])  # type: ignore[attr-defined]
        ids = raw.get("ids", [])
        docs = raw.get("documents", []) or []
        metas = raw.get("metadatas", []) or []
        entries: list[TitleEntry] = []
        for i, _id in enumerate(ids):
            body = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) and metas[i] else {}
            title = extract_title(body or "", meta.get("citation_string", "") or "")
            entries.append(
                TitleEntry(
                    id=_id,
                    title=title,
                    quarantine=_truthy(meta.get("quarantine")),
                )
            )
        return cls(entries, ngram=ngram)

    def __len__(self) -> int:
        return len(self._entries)

    def search(
        self, query: str, k: int = 10, *, drop_quarantined: bool = True
    ) -> list[tuple[str, float]]:
        """Return [(chunk_id, overlap_score)] ranked by title overlap, top-k."""
        q_grams = char_ngrams(query, self._ngram)
        if not q_grams:
            return []
        scored: list[tuple[str, float]] = []
        for cid, grams in self._grams.items():
            if drop_quarantined and cid in self._quarantined:
                continue
            if not grams:
                continue
            inter = len(q_grams & grams)
            if inter == 0:
                continue
            union = len(q_grams | grams)
            scored.append((cid, inter / union))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def title_of(self, chunk_id: str) -> Optional[str]:
        return self._title_of.get(chunk_id)


# --------------------------------------------------------------------------- #
# Fused search: dense + title channel
# --------------------------------------------------------------------------- #
def search_with_title_boost(
    query: str,
    embed_query: EmbedQueryFn,
    collection: ChromaCollection,
    title_index: TitleIndex,
    k: int = 5,
    *,
    candidate_k: int = 10,
    filters: Optional[dict] = None,
    drop_quarantined: bool = True,
) -> SearchResult:
    """Dense retrieval fused (RRF) with a lexical title channel.

    1. Dense search (the existing channel) over `candidate_k`.
    2. Title channel: lexical overlap of the query vs chunk titles, `candidate_k`.
    3. For any title-channel id NOT already a dense hit, fetch its chunk so we
       can return a full RetrievedChunk.
    4. RRF-fuse the two ranked lists; return top-k.

    Why RRF and not score-mixing: the dense cosine score and the Jaccard title
    score live on different scales; fusing by RANK is scale-free and is the same
    principled merge already used for cross-collection search.
    """
    dense = search_regulations(
        query, embed_query, collection, k=candidate_k,
        filters=filters, drop_quarantined=drop_quarantined,
    )
    dense_by_id = {c.id: c for c in dense.chunks}

    title_hits = title_index.search(
        query, k=candidate_k, drop_quarantined=drop_quarantined
    )

    # Build a RetrievedChunk list for title hits, fetching any not in dense set.
    missing_ids = [cid for cid, _ in title_hits if cid not in dense_by_id]
    fetched: dict[str, RetrievedChunk] = {}
    if missing_ids:
        raw = collection.get(  # type: ignore[attr-defined]
            ids=missing_ids, include=["documents", "metadatas"]
        )
        gids = raw.get("ids", [])
        gdocs = raw.get("documents", []) or []
        gmetas = raw.get("metadatas", []) or []
        for i, cid in enumerate(gids):
            meta = gmetas[i] if i < len(gmetas) and gmetas[i] else {}
            fetched[cid] = RetrievedChunk(
                id=cid,
                text=(gdocs[i] if i < len(gdocs) else "") or "",
                collection="lao_regulations_tax",
                score=0.0,  # no dense score; title channel contributes via rank
                distance=1.0,
                chunk_type=meta.get("chunk_type"),
                is_regulation=_truthy(meta.get("is_regulation")),
                provenance_tier=meta.get("provenance_tier"),
                quarantine=_truthy(meta.get("quarantine")),
                citation=_citation_from_meta(meta),
            )

    title_ranked: list[RetrievedChunk] = []
    for cid, _score in title_hits:
        if cid in dense_by_id:
            title_ranked.append(dense_by_id[cid])
        elif cid in fetched:
            title_ranked.append(fetched[cid])

    fused = _reciprocal_rank_fusion([dense.chunks, title_ranked], k=k)
    return SearchResult(query=query, chunks=fused)
