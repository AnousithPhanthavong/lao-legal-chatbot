"""
Lao Legal RAG — Step 5: Agent Pipeline (fixed orchestration)
============================================================

The capstone. Given a question, runs a FIXED, explainable pipeline:

    decompose -> agent_retrieve (per sub-query, fused) -> rerank
              -> synthesize cited answer -> self-verify faithfulness

Design (locked decisions)
-------------------------
* FIXED pipeline, not function-calling: predictable, bounded latency, every step
  explainable at defense, and it uses the tested components exactly as designed.
* SELF-VERIFICATION: after drafting, an LLM faithfulness check confirms every
  claim traces to a retrieved chunk; unsupported claims are flagged/removed.
* GROUNDED synthesis: the answer is built ONLY from retrieved chunks, each cited
  with [law, article]. The LLM is told to use nothing outside the provided
  chunks (anti-hallucination), consistent with "every claim cites a source".
* Full TRACE recorded for the Streamlit panel + thesis screenshots.

All LLM/retrieval dependencies are injected, so the pipeline is unit-testable
with fakes (no keys, no DB).

NOTE: research tool, not legal advice. NOT a substitute for a licensed Lao
tax professional.
"""

from __future__ import annotations

import json
from typing import Callable, Optional

from pydantic import BaseModel

from decompose import LLMFn
from retrieval import RetrievedChunk
from agent_retrieve import AgentRetrievalResult
from rerank import RerankResult

# Injected callables ---------------------------------------------------------
# retrieve_fn(query) -> AgentRetrievalResult   (wraps agent_retrieve with bound args)
# rerank_fn(query, chunks) -> RerankResult      (wraps rerank with bound llm)
RetrieveFn = Callable[[str], AgentRetrievalResult]
RerankFn = Callable[[str, list[RetrievedChunk]], RerankResult]

DISCLAIMER = ("ໝາຍເຫດ: ນີ້ແມ່ນເຄື່ອງມືຄົ້ນຄວ້າ ບໍ່ແມ່ນຄຳແນະນຳທາງກົດໝາຍ "
              "ແລະ ບໍ່ທົດແທນ ນັກກົດໝາຍລາວ ທີ່ມີໃບອະນຸຍາດ.")


# --------------------------------------------------------------------------- #
# Trace + result models
# --------------------------------------------------------------------------- #
class TraceStep(BaseModel):
    step: str
    detail: str
    data: Optional[dict] = None


class AgentAnswer(BaseModel):
    query: str
    answer: str
    citations: list[str]              # [law, article] strings actually used
    chunks_used: list[RetrievedChunk]
    verified: bool                    # passed faithfulness check
    verification_notes: Optional[str] = None
    trace: list[TraceStep]

    def with_disclaimer(self) -> str:
        return f"{self.answer}\n\n{DISCLAIMER}"


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
def _format_chunks_for_prompt(chunks: list[RetrievedChunk], max_chars: int = 600) -> str:
    lines = []
    for i, c in enumerate(chunks):
        cite = c.citation.citation_string or f"chunk {c.id}"
        body = c.text.strip()[:max_chars].replace("\n", " ")
        lines.append(f"[{i}] ({cite})\n{body}")
    return "\n\n".join(lines)


def _synthesis_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    return f"""ທ່ານເປັນຕົວຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ. ຕອບຄຳຖາມໂດຍໃຊ້ ສະເພາະ ຂໍ້ມູນ
ຈາກມາດຕາທີ່ໃຫ້ມາລຸ່ມນີ້. ຫ້າມໃຊ້ຄວາມຮູ້ພາຍນອກ. ທຸກຄຳເວົ້າ ຕ້ອງອ້າງອີງ
[ກົດໝາຍ, ມາດຕາ] ຈາກຂໍ້ມູນທີ່ໃຫ້. ຖ້າຂໍ້ມູນບໍ່ພຽງພໍ ໃຫ້ບອກຊື່ສັດ.

ຄຳຖາມ: {query}

ມາດຕາທີ່ມີ:
{_format_chunks_for_prompt(chunks)}

ຕອບເປັນ JSON ເທົ່ານັ້ນ (ບໍ່ມີ markdown, ບໍ່ມີຄຳອະທິບາຍກ່ອນ ຫຼື ຫຼັງ).
ໃນ answer ຫ້າມໃຊ້ double-quote — ໃຊ້ ' ແທນຖ້າຈຳເປັນ.
{{"answer": "ຄຳຕອບພ້ອມ [ມາດຕາ]", "citations": ["ມາດຕາ"]}}

JSON:"""


def _verify_prompt(query: str, answer: str, chunks: list[RetrievedChunk]) -> str:
    return f"""ກວດສອບຄວາມຖືກຕ້ອງ. ແຕ່ລະຄຳເວົ້າໃນຄຳຕອບ ມີຫຼັກຖານຈາກມາດຕາທີ່ໃຫ້ບໍ?

ຄຳຖາມ: {query}
ຄຳຕອບ: {answer}

ມາດຕາທີ່ມີ:
{_format_chunks_for_prompt(chunks, max_chars=400)}

ຕອບ JSON: {{"faithful": true/false, "unsupported_claims": ["..."], "notes": "..."}}

JSON:"""


def _extract_json(raw: str) -> dict:
    from decompose import _extract_json as ej
    return ej(raw)


def _salvage_answer(raw: str, top_chunks) -> dict:
    """Recover an answer from LLM output that isn't clean JSON.

    Sometimes the model returns prose, markdown-fenced JSON, or JSON with a
    stray quote in the Lao text. Rather than discard a perfectly good answer,
    we salvage it:
      1. strip markdown fences and retry JSON
      2. regex-extract the "answer" field value if present
      3. fall back to using the raw text as the answer (cleaned)
    Citations are recovered from the cited chunks if not parseable.
    """
    import re, json

    text = raw.strip()

    # 1. strip ```json ... ``` fences and retry
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text,
                    flags=re.MULTILINE).strip()
    for candidate in (fenced, text):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and obj.get("answer"):
                return {"answer": str(obj["answer"]).strip(),
                        "citations": [str(c) for c in obj.get("citations", [])]}
        except Exception:
            pass

    # 2. regex-extract the "answer" field even from malformed JSON
    m = re.search(r'"answer"\s*:\s*"(.+?)"\s*(?:,|\})', text, re.DOTALL)
    if m:
        ans = m.group(1).replace('\\"', '"').replace("\\n", " ").strip()
        if ans:
            cites = re.findall(r'"(ມາດຕາ[^"]{0,80})"', text)
            return {"answer": ans, "citations": cites}

    # 3. last resort: the model wrote prose, not JSON. Use it as the answer,
    #    but only if it looks like real Lao content (not an error string).
    cleaned = re.sub(r"```", "", text).strip()
    has_lao = any('\u0e80' <= c <= '\u0eff' for c in cleaned)
    if cleaned and has_lao and len(cleaned) > 15:
        # attach citations from the retrieved chunks so it's still grounded
        cites = [c.citation.citation_string for c in top_chunks
                 if c.citation.citation_string][:3]
        return {"answer": cleaned[:1000], "citations": cites,
                "_salvaged": True}

    return {}  # genuinely nothing usable


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def answer_question(
    query: str,
    llm: LLMFn,
    retrieve_fn: RetrieveFn,
    rerank_fn: RerankFn,
    *,
    top_k: int = 5,
    verify: bool = True,
) -> AgentAnswer:
    """Run the fixed agent pipeline and return a cited, verified answer.

    retrieve_fn and rerank_fn are injected (bind your agent_retrieve / rerank
    with their embed_query/llm/collections/title_index already supplied).
    """
    trace: list[TraceStep] = []

    # 1. retrieve (decompose + per-subquery fusion happens inside retrieve_fn)
    ret = retrieve_fn(query)
    trace.append(TraceStep(
        step="retrieve",
        detail=f"decomposed into {len(ret.decomposition.sub_queries)} "
               f"sub-quer{'y' if len(ret.decomposition.sub_queries)==1 else 'ies'}; "
               f"{len(ret.chunks)} candidates",
        data={"kind": ret.decomposition.kind.value,
              "forms": ret.retrieval_forms,
              "candidate_ids": ret.ids()},
    ))

    if not ret.chunks:
        return AgentAnswer(
            query=query, answer="ບໍ່ພົບຂໍ້ມູນທີ່ກ່ຽວຂ້ອງ.",
            citations=[], chunks_used=[], verified=False,
            verification_notes="no candidates retrieved", trace=trace,
        )

    # 2. rerank for precision
    rr = rerank_fn(query, ret.chunks)
    top_chunks = rr.chunks[:top_k]
    trace.append(TraceStep(
        step="rerank",
        detail=f"reranked={rr.reranked}; kept top {len(top_chunks)}",
        data={"reranked": rr.reranked, "ranked_ids": [c.id for c in top_chunks]},
    ))

    # 3. synthesize a grounded, cited answer
    try:
        raw = llm(_synthesis_prompt(query, top_chunks))
        try:
            data = _extract_json(raw)
            answer_text = (data.get("answer") or "").strip()
            citations = [str(c) for c in data.get("citations", []) if str(c).strip()]
        except Exception:
            data = {}
            answer_text, citations = "", []

        # if clean JSON parse failed or gave nothing, try to SALVAGE the answer
        # from the raw output instead of discarding a possibly-good response
        if not answer_text:
            salvaged = _salvage_answer(raw, top_chunks)
            answer_text = (salvaged.get("answer") or "").strip()
            citations = [str(c) for c in salvaged.get("citations", []) if str(c).strip()]
            if answer_text and salvaged.get("_salvaged"):
                trace.append(TraceStep(
                    step="synthesize",
                    detail="recovered answer from non-JSON output (salvaged)",
                    data={"salvaged": True}))

        if not answer_text:
            raise ValueError("no JSON object in LLM output")
    except Exception as e:
        # Distinguish a TRANSIENT API error (server busy / rate limit) from a
        # genuine synthesis failure. The user asked: show the REAL reason, don't
        # disguise an API outage as "couldn't synthesize".
        err_str = str(e)
        transient_markers = ("503", "429", "500", "502", "504", "UNAVAILABLE",
                             "RESOURCE_EXHAUSTED", "high demand", "overload",
                             "quota", "rate limit")
        is_api_busy = any(m in err_str for m in transient_markers)

        citations = [c.citation.citation_string for c in top_chunks
                     if c.citation.citation_string]

        if is_api_busy:
            cause = ("503 ການບໍລິການ AI ບໍ່ວ່າງຊົ່ວຄາວ"
                     if ("503" in err_str or "UNAVAILABLE" in err_str
                         or "high demand" in err_str)
                     else "429 ຮອດຂີດຈຳກັດການເອີ້ນໃຊ້")
            answer_text = (
                f"⚠ ບໍ່ສາມາດສ້າງຄຳຕອບໄດ້ ເນື່ອງຈາກ {cause}. "
                f"ກະລຸນາລອງໃໝ່ໃນອີກ 1 ນາທີ. "
                f"(ພົບມາດຕາທີ່ກ່ຽວຂ້ອງແລ້ວ ແຕ່ບໍລິການ AI ບໍ່ວ່າງ)"
            )
            notes = f"API_TRANSIENT_ERROR: {err_str[:200]}"
        else:
            answer_text = (
                "ພົບມາດຕາທີ່ກ່ຽວຂ້ອງ ແຕ່ບໍ່ສາມາດສັງເຄາະຄຳຕອບໄດ້ "
                f"(synthesis error: {err_str[:100]})."
            )
            notes = f"SYNTHESIS_ERROR: {err_str[:200]}"

        trace.append(TraceStep(
            step="synthesize",
            detail=f"synthesis failed — {notes}",
            data={"error_type": "transient_api" if is_api_busy else "synthesis",
                  "raw_error": err_str[:300]},
        ))
        return AgentAnswer(
            query=query, answer=answer_text, citations=citations,
            chunks_used=top_chunks, verified=False,
            verification_notes=notes, trace=trace,
        )
    trace.append(TraceStep(
        step="synthesize",
        detail=f"answer drafted with {len(citations)} citation(s)",
        data={"citations": citations},
    ))

    # 4. self-verify faithfulness
    verified = True
    verify_notes = None
    if verify:
        try:
            vraw = llm(_verify_prompt(query, answer_text, top_chunks))
            vdata = _extract_json(vraw)
            verified = bool(vdata.get("faithful", True))
            unsupported = vdata.get("unsupported_claims", []) or []
            verify_notes = vdata.get("notes")
            trace.append(TraceStep(
                step="verify",
                detail=f"faithful={verified}; "
                       f"{len(unsupported)} unsupported claim(s)",
                data={"faithful": verified, "unsupported": unsupported},
            ))
        except Exception as e:
            # verification failure should NOT block the answer; flag it
            verified = False
            verify_notes = f"verification unavailable ({e})"
            trace.append(TraceStep(step="verify",
                                   detail=f"verification failed ({e})"))

    return AgentAnswer(
        query=query, answer=answer_text, citations=citations,
        chunks_used=top_chunks, verified=verified,
        verification_notes=verify_notes, trace=trace,
    )
