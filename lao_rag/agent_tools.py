"""
Lao Legal RAG — Agent Tools (wiring REAL enrichment assets)
===========================================================

Tools the agent can call, backed by the assets that ACTUALLY exist and are
clean:
  * get_summary(law_id)        <- summary_human_review.json (human-reviewed)
  * follow_reference(law, art) <- cross_references.json (deterministic graph)
  * lookup(law, article)       <- direct fetch from the article store
  * search(query)              <- the retrieval+rerank engine (separate module)
  * finish(answer)             <- terminates the agent loop

DELIBERATELY OMITTED (documented decisions, not gaps):
  * list_laws(topic): topic_index.json contained non-discriminating word
    fragments ('ແຫ່ງ', 'ເຂັ້ມແຂງ') from naive keyword extraction. Topic routing
    is replaced by direct semantic search. (Thesis: negative finding.)
  * check_concept: waits for the (small, grounding-validated) concepts KB.

All data is INJECTED (loaded dicts/lists passed in), so tools are unit-testable
without Drive or keys. Each tool returns a formatted STRING for the agent to
read, plus structured data is available where useful.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# Data containers (built once from the JSON assets)
# --------------------------------------------------------------------------- #
class EnrichmentStore(BaseModel):
    """Holds the loaded enrichment assets the tools read from."""

    # law_id -> reviewed summary text
    summaries: dict[str, str] = {}
    # list of cross-reference edges (the 'references' list from cross_references.json)
    references: list[dict] = []
    # law_id -> human-readable law name (from registry or topic_index)
    law_names: dict[str, str] = {}

    @classmethod
    def from_assets(
        cls,
        summary_review: list[dict],
        cross_refs: dict,
        law_names: Optional[dict[str, str]] = None,
    ) -> "EnrichmentStore":
        """Build from the raw loaded JSON objects.

        summary_review: the list from summary_human_review.json
        cross_refs: the dict from cross_references.json (has 'references')
        law_names: optional law_id -> law_name_lao

        `generated_summary` may be a plain string OR a nested dict (the real
        file nests it). We extract the summary text robustly from either shape.
        """
        summaries: dict[str, str] = {}
        for entry in summary_review:
            lid = entry.get("law_id", "")
            text = cls._extract_summary_text(entry.get("generated_summary"))
            if lid and text:
                summaries[lid] = text
        refs = cross_refs.get("references", []) if isinstance(cross_refs, dict) else []
        return cls(summaries=summaries, references=refs, law_names=law_names or {})

    @staticmethod
    def _extract_summary_text(gen_summary) -> str:
        """Pull a string summary from `generated_summary`, whatever its shape."""
        if gen_summary is None:
            return ""
        if isinstance(gen_summary, str):
            return gen_summary.strip()
        if isinstance(gen_summary, dict):
            # try common field names for the Lao summary text
            for key in ("summary_lao", "summary", "text", "summary_text",
                        "content", "lao"):
                v = gen_summary.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # fallback: join any string values found in the dict
            parts = [v.strip() for v in gen_summary.values()
                     if isinstance(v, str) and v.strip()]
            return "\n".join(parts)
        return str(gen_summary).strip()


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def get_summary(law_id: str, store: EnrichmentStore) -> str:
    """Return the human-reviewed summary of a law, if available."""
    # tolerate law_id with/without .json suffix
    candidates = {law_id, law_id + ".json", law_id.replace(".json", "")}
    for cid in candidates:
        if cid in store.summaries:
            name = store.law_names.get(cid, cid)
            return f"ສະຫຼຸບກົດໝາຍ ({name}):\n{store.summaries[cid]}"
    return f"ບໍ່ມີສະຫຼຸບສຳລັບ '{law_id}'."


def follow_reference(
    law_id: str, article: int, store: EnrichmentStore, *, max_refs: int = 10
) -> str:
    """List the references made BY a given article (outgoing edges)."""
    lid_norm = law_id.replace(".json", "")
    out = []
    for r in store.references:
        if r.get("from_law", "").replace(".json", "") == lid_norm \
                and r.get("from_article") == article:
            tgt_law = (r.get("to_law") or "?").replace(".json", "")
            tgt_art = r.get("to_article")
            rtype = r.get("ref_type", "")
            resolved = r.get("resolved", False)
            if resolved and tgt_art is not None:
                out.append(f"  → {tgt_law} ມາດຕາ {tgt_art} [{rtype}]")
            else:
                ctx = (r.get("context") or "")[:50]
                out.append(f"  → (ບໍ່ສາມາດລະບຸໄດ້) [{rtype}] ບໍລິບົດ: {ctx}")
            if len(out) >= max_refs:
                break
    if not out:
        return f"ມາດຕາ {article} ຂອງ {lid_norm} ບໍ່ມີການອ້າງອີງ."
    return f"ມາດຕາ {article} ຂອງ {lid_norm} ອ້າງອີງເຖິງ:\n" + "\n".join(out)


def lookup(
    law_id: str,
    article: int,
    article_fetch: Callable[[str, int], Optional[str]],
) -> str:
    """Direct article fetch via an injected fetch function.

    article_fetch(law_id, article) -> article text or None. (In production this
    wraps the article store / ChromaDB get; injected so this tool is testable.)
    """
    text = article_fetch(law_id, article)
    if not text:
        return f"ບໍ່ພົບ ມາດຕາ {article} ຂອງ {law_id}."
    return f"[{law_id}, ມາດຕາ {article}]\n{text}"


def finish(answer: str) -> str:
    """Terminal tool — the answer becomes the agent's final output."""
    return answer


# --------------------------------------------------------------------------- #
# Tool registry (for the agent loop / function-calling declaration)
# --------------------------------------------------------------------------- #
def build_tool_specs() -> list[dict]:
    """Lightweight tool descriptions for the agent's system prompt / function
    declarations. Kept data-only so the agent module can format them for Gemini."""
    return [
        {"name": "search",
         "desc": "ຄົ້ນຫາມາດຕາທີ່ກ່ຽວຂ້ອງ (semantic + rerank)",
         "args": {"query": "str"}},
        {"name": "lookup",
         "desc": "ເອົາມາດຕາສະເພາະ ໂດຍກົງ",
         "args": {"law_id": "str", "article": "int"}},
        {"name": "get_summary",
         "desc": "ເອົາສະຫຼຸບກົດໝາຍ (ກວດແລ້ວໂດຍຄົນ)",
         "args": {"law_id": "str"}},
        {"name": "follow_reference",
         "desc": "ຕິດຕາມການອ້າງອີງຂອງມາດຕາ",
         "args": {"law_id": "str", "article": "int"}},
        {"name": "finish",
         "desc": "ສົ່ງຄຳຕອບສຸດທ້າຍ",
         "args": {"answer": "str"}},
    ]
