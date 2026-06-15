"""
backend.py — FROZEN backend interface for the Lao Legal RAG app.
================================================================
The UI (app.py) imports ONLY this. One function: ask(question) -> AskResult.
The UI never touches the retrieval/generation core and cannot break it.

Reads API keys from GEMINI_KEYS (comma-separated) — your exact secret format —
with fallbacks to numbered GEMINI_KEY_1.. and single GEMINI_API_KEY.

DO NOT EDIT for UI work.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AskResult:
    answer: str
    citations: list = field(default_factory=list)   # [{law, article, title}]
    confidence: str = "NONE"
    is_general: bool = False
    steps: list = field(default_factory=list)        # ["ກຳລັງຄົ້ນຫາ...", ...] for the thinking panel
    error: Optional[str] = None


def _collect_keys() -> list[str]:
    """Read keys in your format first (GEMINI_KEYS comma string), then fallbacks."""
    # 1. GEMINI_KEYS = "key1,key2,..."  (your format)
    bulk = os.environ.get("GEMINI_KEYS", "")
    if not bulk:
        try:
            import streamlit as st
            bulk = st.secrets.get("GEMINI_KEYS", "")
        except Exception:
            pass
    if bulk:
        return [k.strip() for k in bulk.split(",") if k.strip()]
    # 2. numbered GEMINI_KEY_1..30
    keys = []
    try:
        import streamlit as st
        for i in range(1, 31):
            v = st.secrets.get(f"GEMINI_KEY_{i}", "")
            if v:
                keys.append(v.strip())
    except Exception:
        pass
    if keys:
        return keys
    # 3. single
    single = os.environ.get("GEMINI_API_KEY", "")
    if not single:
        try:
            import streamlit as st
            single = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    return [single.strip()] if single else []


_SYSTEM = None


def _load_once():
    global _SYSTEM
    if _SYSTEM is not None:
        return _SYSTEM
    # ensure keys are in the environment for core_system to find
    keys = _collect_keys()
    if not keys:
        raise RuntimeError("No API keys found. Set GEMINI_KEYS in Streamlit secrets.")
    os.environ["GEMINI_KEYS"] = ",".join(keys)
    os.environ.setdefault("GEMINI_API_KEY", keys[0])  # for cores that read a single key
    import core_system
    _SYSTEM = core_system.load_system()
    return _SYSTEM


def ask(question: str) -> AskResult:
    """Ask the system a question. Never raises — always returns a safe AskResult.
    `steps` lists the stages performed, for a Claude-style 'thinking' panel.
    """
    q = (question or "").strip()
    if len(q) < 3:
        return AskResult(answer="ກະລຸນາພິມຄຳຖາມໃຫ້ຍາວກວ່ານີ້ເລັກນ້ອຍ.", confidence="NONE")

    steps = []
    try:
        import core_system
        sys = _load_once()
        steps.append("ໂຫຼດລະບົບ ແລະ ຖານຂໍ້ມູນກົດໝາຍ")

        steps.append("ກຳລັງຄົ້ນຫາມາດຕາທີ່ກ່ຽວຂ້ອງ (semantic + keyword)")
        search_results, confidence, law_filter, article_num = core_system.search(q, sys)
        if law_filter:
            steps.append(f"ກວດພົບໝວດກົດໝາຍ: {law_filter}")
        if article_num:
            steps.append(f"ກວດພົບເລກມາດຕາ: {article_num}")
        steps.append(f"ພົບ {len(search_results) if search_results else 0} ຜົນການຄົ້ນຫາ (ຄວາມໝັ້ນໃຈ: {confidence})")

        steps.append("ກຳລັງວິເຄາະ ແລະ ສ້າງຄຳຕອບຈາກມາດຕາທີ່ພົບ")
        answer_text, citations_ok, is_general = core_system.generate_answer(
            q, search_results, confidence, sys
        )

        citations = []
        if not is_general and search_results:
            for r in search_results[:3]:
                m = r["chunk"]["metadata"]
                citations.append({
                    "law": m.get("law_name_lao", ""),
                    "article": m.get("article", ""),
                    "title": m.get("article_title", ""),
                })
        steps.append("ກວດສອບການອ້າງອີງ ແລະ ສຳເລັດ")

        return AskResult(answer=answer_text, citations=citations,
                         confidence=confidence, is_general=is_general, steps=steps)
    except Exception as e:
        msg = str(e)
        if "No API keys" in msg:
            friendly = "⚠ ບໍ່ພົບ API keys. ກະລຸນາຕັ້ງຄ່າ GEMINI_KEYS ໃນ Streamlit secrets."
        elif any(t in msg for t in ("503", "UNAVAILABLE", "overload", "high demand")):
            friendly = "⚠ ບໍລິການ AI ບໍ່ວ່າງຊົ່ວຄາວ. ກະລຸນາລອງໃໝ່ໃນອີກ 1 ນາທີ."
        elif any(t in msg for t in ("429", "RESOURCE_EXHAUSTED", "quota", "rate limit")):
            friendly = "⚠ ມີການໃຊ້ງານຫຼາຍເກີນໄປ. ກະລຸນາລອງໃໝ່ໃນອີກ 1 ນາທີ."
        else:
            friendly = "ຂໍອະໄພ, ເກີດຂໍ້ຜິດພາດ. ກະລຸນາລອງໃໝ່ອີກຄັ້ງ."
        steps.append(f"ຂໍ້ຜິດພາດ: {msg[:80]}")
        return AskResult(answer=friendly, confidence="NONE", steps=steps, error=msg[:200])


def get_system_info() -> dict:
    try:
        sys = _load_once()
        reg = sys.get("registry", {})
        return {"total_laws": reg.get("total_laws", "?"),
                "total_chunks": reg.get("total_chunks", "?"), "ready": True}
    except Exception as e:
        return {"total_laws": "?", "total_chunks": "?", "ready": False, "error": str(e)[:200]}
