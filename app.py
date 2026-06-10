"""
Lao Legal RAG — Streamlit App (Deployment Version)
====================================================
Reads GEMINI_KEYS from Streamlit Secrets (cloud) or environment (local).
ChromaDB is bundled in the repo at chroma_data/.

Run locally:  streamlit run app.py
Deployed at:  https://your-app.streamlit.app
"""

from __future__ import annotations
import os, sys, time
import numpy as np
import streamlit as st

# ------------------------------------------------------------------ #
# Path — add lao_rag modules (works both locally and on Streamlit Cloud)
# ------------------------------------------------------------------ #
ROOT = os.path.dirname(os.path.abspath(__file__))
LAO_RAG = os.path.join(ROOT, "lao_rag")
if LAO_RAG not in sys.path:
    sys.path.insert(0, LAO_RAG)

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
# ChromaDB bundled in the repo (committed via Git LFS or directly)
CHROMA_PATH = os.path.join(ROOT, "chroma_data")
REGS_COLLECTION = "lao_regulations_tax"
LAWS_COLLECTION  = "lao_legal"
EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL   = "gemini-2.5-flash"

DISCLAIMER = (
    "ໝາຍເຫດ: ນີ້ແມ່ນເຄື່ອງມືຄົ້ນຄວ້າ ບໍ່ແມ່ນຄຳແນະນຳທາງກົດໝາຍ "
    "ແລະ ບໍ່ທົດແທນ ນັກກົດໝາຍລາວ ທີ່ມີໃບອະນຸຍາດ."
)

EXAMPLE_QUERIES = [
    "ອັດຕາ VAT ແມ່ນເທົ່າໃດ?",
    "ວິສາຫະກິດ ໝາຍຄວາມວ່າແນວໃດ?",
    "ຜູ້ໃດຕ້ອງລົງທະບຽນ VAT?",
    "ການຟອກເງິນ ມີໂທດແນວໃດ?",
    "ວິທີຈົດທະບຽນວິສາຫະກິດ ເຮັດແນວໃດ?",
]

# ------------------------------------------------------------------ #
# Keys — Streamlit Secrets on cloud, env var locally
# ------------------------------------------------------------------ #
def _load_keys() -> list[str]:
    # Streamlit Cloud: keys in st.secrets
    raw = st.secrets.get("GEMINI_KEYS", "")
    # Local fallback: environment variable
    if not raw:
        raw = os.environ.get("GEMINI_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        st.error(
            "No GEMINI_KEYS found.\n\n"
            "On Streamlit Cloud: add GEMINI_KEYS to App Settings → Secrets.\n"
            "Locally: set the GEMINI_KEYS environment variable."
        )
        st.stop()
    return keys

# ------------------------------------------------------------------ #
# Cached resources (built once per server session)
# ------------------------------------------------------------------ #
def _decompress_db_if_needed():
    """If chroma.sqlite3 is missing but a .gz exists, decompress it once."""
    import gzip, shutil
    sqlite = os.path.join(CHROMA_PATH, "chroma.sqlite3")
    gz = sqlite + ".gz"
    if not os.path.exists(sqlite) and os.path.exists(gz):
        with gzip.open(gz, "rb") as f_in, open(sqlite, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


@st.cache_resource(show_spinner="ກຳລັງເຊື່ອມຕໍ່ ChromaDB...")
def _load_collections():
    import chromadb
    if not os.path.isdir(CHROMA_PATH):
        st.error(f"ChromaDB not found at {CHROMA_PATH}. "
                 "Make sure chroma_data/ is in the repo.")
        st.stop()
    _decompress_db_if_needed()
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    names  = [c.name for c in client.list_collections()]
    for needed in (REGS_COLLECTION, LAWS_COLLECTION):
        if needed not in names:
            st.error(f"Collection '{needed}' missing. Found: {names}")
            st.stop()
    laws = client.get_collection(LAWS_COLLECTION)
    regs = client.get_collection(REGS_COLLECTION)
    return laws, regs

@st.cache_resource(show_spinner="ກຳລັງສ້າງ title index...")
def _load_indexes(_laws, _regs):
    from title_boost import TitleIndex
    return TitleIndex.from_collection(_laws), TitleIndex.from_collection(_regs)

# ------------------------------------------------------------------ #
# Rotating client — fresh genai.Client per call, lazy import
# ------------------------------------------------------------------ #
class _Rotator:
    def __init__(self, keys: list[str]):
        self._keys = keys
        self._i    = 0

    def _call(self, fn):
        last = None
        for attempt in range(len(self._keys) + 1):
            try:
                import google.genai as genai
                client = genai.Client(
                    api_key=self._keys[self._i % len(self._keys)])
                self._i += 1
                return fn(client)
            except Exception as e:
                last = e
                if any(m in str(e) for m in ("429","503","UNAVAILABLE",
                       "RESOURCE_EXHAUSTED","high demand")) \
                        and attempt < len(self._keys):
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise last

    def embed(self, text: str) -> list[float]:
        def _fn(c):
            res = c.models.embed_content(
                model=EMBED_MODEL, contents=text,
                config={"task_type": "RETRIEVAL_QUERY",
                        "output_dimensionality": 768})
            v = np.asarray(res.embeddings[0].values, dtype=np.float32)
            return (v / (np.linalg.norm(v) or 1.0)).tolist()
        return self._call(_fn)

    def llm(self, prompt: str) -> str:
        return self._call(
            lambda c: c.models.generate_content(
                model=LLM_MODEL, contents=prompt).text)

@st.cache_resource(show_spinner=False)
def _load_rotator():
    return _Rotator(_load_keys())

# ------------------------------------------------------------------ #
# UI
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Lao Legal RAG",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ ຕົວຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ")
st.caption(
    "Agentic RAG · 75 laws + 18 tax regulations · "
    "decompose → retrieve → rerank → synthesize → verify"
)

# warm up (cached after first call)
laws, regs   = _load_collections()
li, ri       = _load_indexes(laws, regs)
rotator      = _load_rotator()

with st.sidebar:
    st.subheader("ຕົວຢ່າງຄຳຖາມ")
    for ex in EXAMPLE_QUERIES:
        if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["query"] = ex
    st.divider()
    st.caption(f"Laws: {laws.count()} chunks")
    st.caption(f"Regs: {regs.count()} chunks")
    st.caption(f"Keys: {len(_load_keys())} (rotating)")
    st.divider()
    st.caption(DISCLAIMER)

query = st.text_area(
    "ຄຳຖາມຂອງທ່ານ:",
    value=st.session_state.get("query", ""),
    height=80,
    placeholder=EXAMPLE_QUERIES[0],
)
run = st.button("ຄົ້ນຫາ", type="primary", use_container_width=True)

def _run_agent(q: str):
    import agent_retrieve as ar
    from rerank         import rerank
    from agent_pipeline import answer_question

    def retrieve_fn(query_text):
        if hasattr(ar, "agent_retrieve_both"):
            return ar.agent_retrieve_both(
                query_text, rotator.embed, rotator.llm,
                laws, regs, li, ri, k=8)
        return ar.agent_retrieve(
            query_text, rotator.embed, rotator.llm, regs, ri, k=8)

    def rerank_fn(query_text, chunks):
        return rerank(query_text, chunks, rotator.llm, top_n=8)

    with st.status("ກຳລັງດຳເນີນການ...", expanded=True) as status:
        st.write("🔍 ກຳລັງຄົ້ນຫາ ແລະ ວິເຄາະ...")
        try:
            result = answer_question(
                q, rotator.llm, retrieve_fn, rerank_fn, top_k=5)
        except Exception as e:
            status.update(label="ເກີດຂໍ້ຜິດພາດ", state="error")
            if any(m in str(e) for m in ("429","503","UNAVAILABLE","high demand")):
                st.warning(
                    "⚠ ການບໍລິການ AI ບໍ່ວ່າງຊົ່ວຄາວ. "
                    "ກະລຸນາລໍຖ້າ 1 ນາທີ ແລ້ວລອງໃໝ່.")
            else:
                st.error(f"Error: {e}")
            return None
        for step in result.trace:
            st.write(f"✓ **{step.step}** — {step.detail}")
        status.update(label="ສຳເລັດ ✓", state="complete")
    return result

if run and query.strip():
    result = _run_agent(query.strip())
    if result:
        st.divider()
        if result.verified:
            st.success("✓ ກວດສອບແລ້ວ — ຄຳຕອບອ້າງອີງຈາກມາດຕາທີ່ດຶງມາ")
        else:
            st.warning("⚠ ກວດສອບບໍ່ຄົບ — ກະລຸນາອ່ານມາດຕາຕົ້ນສະບັບ")
        st.markdown("### ຄຳຕອບ")
        st.markdown(result.answer)
        if result.citations:
            st.markdown("### ການອ້າງອີງ")
            for c in result.citations:
                st.markdown(f"- {c}")
        with st.expander("ມາດຕາທີ່ນຳໃຊ້"):
            for ch in result.chunks_used:
                cite = ch.citation.citation_string or ch.id
                st.markdown(f"**{cite}**")
                st.caption(ch.text[:400] + ("..." if len(ch.text) > 400 else ""))
                st.divider()
        st.info(DISCLAIMER)
elif run:
    st.warning("ກະລຸນາພິມຄຳຖາມກ່ອນ.")
