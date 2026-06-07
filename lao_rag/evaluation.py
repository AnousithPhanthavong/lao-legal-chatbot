"""
Lao Legal RAG — Retrieval Evaluation Harness
=============================================

Measures retrieval quality (Recall@k, MRR), COLLISION-AWARE, comparing:
  * single-shot  (v5-style: one embed, top-k)          [baseline]
  * agent        (decompose + title-boost + spelling + weighted RRF + rerank)

Headline design
---------------
The benchmark INCLUDES the named v5-failure cases (compound, spelling-variant,
high-confidence-wrong) as a tagged subset, so the report shows not just an
aggregate number but specifically that the agent fixes the documented failures.

Collision-aware scoring
------------------------
A question whose gold answer is one of several repeated-title sibling articles
is scored with SET-CREDIT: retrieving ANY member of the gold answer-family
counts as correct. This fixes the label-ambiguity discovered earlier (a question
generated from a repeated title cannot uniquely identify its source).

Metrics
-------
* Recall@k: gold (or any family member) appears in top-k
* MRR: 1/rank of the first gold/family hit (0 if absent)

Retrieve functions are injected, so this is testable with fakes and runs the
real system on Colab by binding the real retrievers.

NOTE: research tool, not legal advice.
"""

from __future__ import annotations

from typing import Callable, Optional

from pydantic import BaseModel

# A retrieve function: query -> ranked list of chunk_ids
RetrieveIdsFn = Callable[[str], list[str]]


class EvalQuestion(BaseModel):
    qid: str
    question: str
    gold_ids: list[str]                 # acceptable answer chunk ids (family)
    tag: str = "general"                # e.g. 'compound','spelling','hcw','general'


class QuestionResult(BaseModel):
    qid: str
    tag: str
    hit_rank: Optional[int]             # 1-based rank of first gold hit, or None
    recall_at_k: bool
    reciprocal_rank: float
    retrieved_ids: list[str]


class EvalReport(BaseModel):
    system: str
    k: int
    n: int
    recall_at_1: float
    recall_at_3: float
    recall_at_k: float
    mrr: float
    per_tag: dict[str, dict]            # tag -> {n, recall_at_k, mrr}
    results: list[QuestionResult]


def expand_gold_with_families(
    gold_ids: list[str], families: dict[str, set[str]]
) -> set[str]:
    """Expand each gold id to its full answer-family (set-credit)."""
    acceptable: set[str] = set()
    for gid in gold_ids:
        acceptable.add(gid)
        if gid in families:
            acceptable |= families[gid]
    return acceptable


def _first_hit_rank(retrieved: list[str], acceptable: set[str]) -> Optional[int]:
    for i, cid in enumerate(retrieved, start=1):
        if cid in acceptable:
            return i
    return None


def evaluate_one(
    q: EvalQuestion,
    retrieve: RetrieveIdsFn,
    families: dict[str, set[str]],
    k: int,
) -> QuestionResult:
    retrieved = retrieve(q.question)[:k]
    acceptable = expand_gold_with_families(q.gold_ids, families)
    rank = _first_hit_rank(retrieved, acceptable)
    return QuestionResult(
        qid=q.qid, tag=q.tag,
        hit_rank=rank,
        recall_at_k=(rank is not None and rank <= k),
        reciprocal_rank=(1.0 / rank if rank else 0.0),
        retrieved_ids=retrieved,
    )


def evaluate(
    system_name: str,
    questions: list[EvalQuestion],
    retrieve: RetrieveIdsFn,
    families: dict[str, set[str]],
    k: int = 5,
) -> EvalReport:
    """Run a retrieve function over the benchmark and compute metrics."""
    results = [evaluate_one(q, retrieve, families, k) for q in questions]
    n = len(results)

    def _recall_at(threshold: int) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results
                   if r.hit_rank is not None and r.hit_rank <= threshold) / n

    mrr = sum(r.reciprocal_rank for r in results) / n if n else 0.0

    # per-tag breakdown (this is where the v5-failure cases show up)
    per_tag: dict[str, dict] = {}
    tags = {r.tag for r in results}
    for tag in tags:
        tag_results = [r for r in results if r.tag == tag]
        tn = len(tag_results)
        per_tag[tag] = {
            "n": tn,
            "recall_at_k": sum(1 for r in tag_results if r.recall_at_k) / tn,
            "mrr": sum(r.reciprocal_rank for r in tag_results) / tn,
        }

    return EvalReport(
        system=system_name, k=k, n=n,
        recall_at_1=_recall_at(1),
        recall_at_3=_recall_at(3),
        recall_at_k=_recall_at(k),
        mrr=mrr,
        per_tag=per_tag,
        results=results,
    )


def compare_report(baseline: EvalReport, agent: EvalReport) -> str:
    """Format a side-by-side comparison table (thesis-ready text)."""
    lines = []
    lines.append(f"{'Metric':<16}{'Baseline':>12}{'Agent':>12}{'Δ':>10}")
    lines.append("-" * 50)
    for label, b, a in [
        ("Recall@1", baseline.recall_at_1, agent.recall_at_1),
        ("Recall@3", baseline.recall_at_3, agent.recall_at_3),
        (f"Recall@{baseline.k}", baseline.recall_at_k, agent.recall_at_k),
        ("MRR", baseline.mrr, agent.mrr),
    ]:
        lines.append(f"{label:<16}{b:>12.3f}{a:>12.3f}{(a-b):>+10.3f}")
    lines.append("")
    lines.append(f"By failure category (Recall@{baseline.k}):")
    lines.append(f"{'Tag':<16}{'Baseline':>12}{'Agent':>12}{'Δ':>10}")
    lines.append("-" * 50)
    for tag in sorted(set(baseline.per_tag) | set(agent.per_tag)):
        b = baseline.per_tag.get(tag, {}).get("recall_at_k", 0.0)
        a = agent.per_tag.get(tag, {}).get("recall_at_k", 0.0)
        lines.append(f"{tag:<16}{b:>12.3f}{a:>12.3f}{(a-b):>+10.3f}")
    return "\n".join(lines)
