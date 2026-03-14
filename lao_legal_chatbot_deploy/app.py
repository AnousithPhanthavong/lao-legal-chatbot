import streamlit as st
import json
import os
import re
import math
import time
import pickle
import hashlib
from pathlib import Path

# ── Page config (MUST be first Streamlit command) ──
st.set_page_config(
    page_title="ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ",
    page_icon="⚖️",
    layout="centered",
)

# ── Custom CSS for Lao fonts ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;600;700&display=swap');

* { font-family: 'Noto Sans Lao', sans-serif !important; }

.main-title {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
.main-title h1 {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #1a73e8, #34a853);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.main-title p {
    color: #9aa0a6;
    font-size: 0.85rem;
}

.disclaimer-bar {
    background: rgba(234,67,53,0.08);
    border: 1px solid rgba(234,67,53,0.2);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    text-align: center;
    font-size: 0.75rem;
    color: #ea4335;
    margin-bottom: 1rem;
}

.source-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    font-size: 0.8rem;
}

.confidence-badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.conf-high { background: rgba(52,168,83,0.15); color: #34a853; }
.conf-medium { background: rgba(251,188,4,0.15); color: #fbbc04; }
.conf-low { background: rgba(234,67,53,0.15); color: #ea4335; }

.cached-tag {
    display: inline-block;
    background: rgba(26,115,232,0.1);
    color: #1a73e8;
    padding: 0.1rem 0.5rem;
    border-radius: 8px;
    font-size: 0.7rem;
    margin-left: 0.5rem;
}

.example-btn {
    margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# DATA & MODEL LOADING (cached)
# ═══════════════════════════════════════

@st.cache_resource(show_spinner="ກຳລັງໂຫຼດລະບົບ...")
def load_system():
    """Load all indexes and models. Cached across reruns."""
    from google import genai
    from google.genai import types as genai_types
    import chromadb
    from rank_bm25 import BM25Okapi

    # Paths — adjust if running locally vs Colab
    base = os.environ.get('LAO_LEGAL_BASE', os.path.join(os.path.dirname(__file__), 'data'))

    # Load Gemini
    api_key = os.environ.get('GEMINI_API_KEY', st.secrets.get('GEMINI_API_KEY', ''))
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found. Add it to Streamlit secrets.")
        st.stop()

    gclient = genai.Client(api_key=api_key)

    # Load ChromaDB
    chroma_path = f'{base}/db/chroma'
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_collection("lao_legal")

    # Load BM25
    bm25_path = f'{base}/db/bm25/bm25_index.pkl'
    with open(bm25_path, 'rb') as f:
        bm25_data = pickle.load(f)

    # Load all chunks
    all_chunks = []
    law_dir = f'{base}/laws/individual'
    for law_file in sorted(Path(law_dir).glob('*.json')):
        with open(law_file, 'r', encoding='utf-8') as f:
            law_data = json.load(f)
        all_chunks.extend(law_data['articles'])

    # Load article lookup
    lookup_path = f'{base}/db/article_lookup.json'
    with open(lookup_path, 'r', encoding='utf-8') as f:
        article_lookup = json.load(f)

    # Load registry
    reg_path = f'{base}/laws/registry.json'
    with open(reg_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    return {
        'gclient': gclient,
        'genai_types': genai_types,
        'collection': collection,
        'bm25_index': bm25_data['index'],
        'bm25_ids': bm25_data['ids'],
        'all_chunks': all_chunks,
        'chunks_by_id': {c['id']: c for c in all_chunks},
        'article_lookup': article_lookup,
        'registry': registry,
    }

# ═══════════════════════════════════════
# SEARCH & GENERATION FUNCTIONS
# ═══════════════════════════════════════

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
TARGET_DIM = 768

LAW_KEYWORDS = {
    'ປະກັນໄພ': 'Insurance Law',
    'ບັນຊີ': 'Accounting Law',
    'ວິສາຫະກິດ': 'Law on Enterprises',
    'ສົ່ງເສີມການລົງທຶນ': 'Law on Investment Promotion',
    'ການລົງທຶນ': 'Law on Investment Promotion',
    'ລົງທຶນ': 'Law on Investment Promotion',
}

LEGAL_TERMS = [
    'ກົດໝາຍ', 'ມາດຕາ', 'ວິສາຫະກິດ', 'ການລົງທຶນ', 'ປະກັນໄພ',
    'ບັນຊີ', 'ສັນຍາ', 'ນິຕິບຸກຄົນ', 'ຜູ້ລົງທຶນ', 'ສິດ',
    'ພັນທະ', 'ເງື່ອນໄຂ', 'ໃບອະນຸຍາດ', 'ທະບຽນ', 'ບໍລິສັດ',
    'ຄຸ້ມຄອງ', 'ຕິດຕາມ', 'ກວດກາ', 'ຕ່າງປະເທດ',
]

def normalize_vector(vec):
    mag = math.sqrt(sum(v**2 for v in vec))
    return [v/mag for v in vec] if mag > 0 else vec

def lao_tokenize(text):
    tokens = set()
    try:
        from laonlp.tokenize import word_tokenize
        for t in word_tokenize(text):
            if len(t.strip()) > 1:
                tokens.add(t.strip())
    except:
        pass
    for term in LEGAL_TERMS:
        if term in text:
            tokens.add(term)
    lao_only = re.sub(r'[^\u0e80-\u0eff]', '', text)
    for n in [3, 4, 5]:
        for i in range(len(lao_only) - n + 1):
            tokens.add(lao_only[i:i+n])
    for word in text.split():
        word = re.sub(r'[^\u0e80-\u0eff0-9a-z]', '', word.lower())
        if len(word) > 1:
            tokens.add(word)
    numbers = re.findall(r'\d+', text)
    tokens.update(numbers)
    return list(tokens)

def search(query, sys):
    """Full hybrid search: semantic + keyword + exact lookup."""
    # Detect article number and law
    article_match = re.search(r'ມາດຕາ\s*(\d+)', query)
    article_num = int(article_match.group(1)) if article_match else None

    law_filter = None
    for kw in sorted(LAW_KEYWORDS.keys(), key=len, reverse=True):
        if kw in query:
            law_filter = LAW_KEYWORDS[kw]
            break

    results = []
    seen_ids = set()

    # Exact lookup
    if article_num:
        for chunk in sys['all_chunks']:
            meta = chunk['metadata']
            if meta['article'] == article_num:
                if not law_filter or law_filter.lower() in meta['law_name_en'].lower():
                    if chunk['id'] not in seen_ids:
                        results.append({
                            'chunk': chunk,
                            'score': 1.0,
                            'source': 'exact',
                        })
                        seen_ids.add(chunk['id'])

    # Semantic search
    try:
        q_result = sys['gclient'].models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=sys['genai_types'].EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=TARGET_DIM,
            )
        )
        q_vec = normalize_vector(q_result.embeddings[0].values)
        where = {"law_name_en": law_filter} if law_filter else None
        chroma_results = sys['collection'].query(
            query_embeddings=[q_vec],
            n_results=10,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        for i in range(len(chroma_results['ids'][0])):
            cid = chroma_results['ids'][0][i]
            if cid not in seen_ids:
                sim = 1 - chroma_results['distances'][0][i]
                chunk = sys['chunks_by_id'].get(cid)
                if chunk:
                    results.append({
                        'chunk': chunk,
                        'score': sim,
                        'source': 'semantic',
                    })
                    seen_ids.add(cid)
    except:
        pass

    # BM25 keyword search
    try:
        tokens = lao_tokenize(query)
        scores = sys['bm25_index'].get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        for idx in top_idx:
            cid = sys['bm25_ids'][idx]
            if cid not in seen_ids and scores[idx] > 0:
                chunk = sys['chunks_by_id'].get(cid)
                if chunk:
                    results.append({
                        'chunk': chunk,
                        'score': scores[idx],
                        'source': 'keyword',
                    })
                    seen_ids.add(cid)
    except:
        pass

    # Sort: exact first, then by score type
    exact = [r for r in results if r['source'] == 'exact']
    semantic = sorted([r for r in results if r['source'] == 'semantic'], key=lambda x: -x['score'])
    keyword = sorted([r for r in results if r['source'] == 'keyword'], key=lambda x: -x['score'])

    # Interleave semantic and keyword
    merged = exact[:]
    for s, k in zip(semantic, keyword):
        if s['chunk']['id'] not in {r['chunk']['id'] for r in merged}:
            merged.append(s)
        if k['chunk']['id'] not in {r['chunk']['id'] for r in merged}:
            merged.append(k)
    # Add remaining
    for r in semantic + keyword:
        if r['chunk']['id'] not in {m['chunk']['id'] for m in merged}:
            merged.append(r)

    top = merged[:5]

    # Confidence
    if top and top[0]['source'] == 'exact':
        confidence = 'HIGH'
    elif top and isinstance(top[0]['score'], float) and top[0]['score'] > 0.75:
        confidence = 'HIGH'
    elif top and isinstance(top[0]['score'], float) and top[0]['score'] > 0.55:
        confidence = 'MEDIUM'
    elif top:
        confidence = 'LOW'
    else:
        confidence = 'NONE'

    return top, confidence, law_filter, article_num

SYSTEM_PROMPT = """ເຈົ້າແມ່ນ ຜູ້ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ (Lao Legal Research Assistant).

ກົດລະບຽບ:
1. ຕອບ ສະເພາະ ຈາກຂໍ້ມູນທີ່ສະໜອງໃຫ້ເທົ່ານັ້ນ. ຫ້າມ ສ້າງຂໍ້ມູນ.
2. ອ້າງອີງ ມາດຕາ ແລະ ຊື່ກົດໝາຍ ໃນຮູບແບບ [ຊື່ກົດໝາຍ, ມາດຕາ X].
3. ຖ້າບໍ່ມີຄຳຕອບ, ບອກວ່າ: "ຂ້ອຍບໍ່ພົບຂໍ້ມູນກ່ຽວກັບຄຳຖາມນີ້."
4. ຫ້າມ ສ້າງເລກມາດຕາ ທີ່ບໍ່ມີໃນຂໍ້ມູນ.
5. ຕອບເປັນ ພາສາລາວ.
6. ຮູບແບບ: 📋 ຄຳຕອບ → 📎 ຫຼັກຖານ → ⚠️ ຂໍ້ຄວນລະວັງ."""

def generate_answer(query, search_results, confidence, sys):
    """Generate grounded answer from search results."""
    if not search_results:
        return "ຂ້ອຍບໍ່ພົບຂໍ້ມູນກ່ຽວກັບຄຳຖາມນີ້ໃນເອກະສານທີ່ມີຢູ່.", True

    # Build context
    context = ""
    for i, r in enumerate(search_results[:3]):
        meta = r['chunk']['metadata']
        context += (
            f"\n--- ແຫຼ່ງ {i+1} ---\n"
            f"[{meta.get('law_name_lao','')}, ມາດຕາ {meta.get('article','')}]\n"
            f"ຫົວຂໍ້: {meta.get('article_title','')}\n"
            f"{r['chunk']['content']}\n"
        )

    conf_note = ""
    if confidence == 'LOW':
        conf_note = "\nໝາຍເຫດ: ຄວາມໝັ້ນໃຈຕ່ຳ. ແຈ້ງຜູ້ໃຊ້ວ່າຂໍ້ມູນອາດບໍ່ຄົບ."

    prompt = f"{SYSTEM_PROMPT}{conf_note}\n\nຂໍ້ມູນກົດໝາຍ:\n{context}\n\nຄຳຖາມ: {query}"

    try:
        response = sys['gclient'].models.generate_content(
            model=GEN_MODEL,
            contents=prompt,
            config=sys['genai_types'].GenerateContentConfig(
                max_output_tokens=1024,
                temperature=0.2,
            )
        )
        answer = response.text

        # Verify citations
        cited = re.findall(r'ມາດຕາ\s*(\d+)', answer)
        available = {str(r['chunk']['metadata'].get('article','')) for r in search_results}
        invalid = [c for c in cited if c not in available]
        citations_ok = len(invalid) == 0

        if not citations_ok:
            answer += f"\n\n⚠️ ໝາຍເຫດ: ມາດຕາ {', '.join(invalid)} ບໍ່ສາມາດຢືນຢັນໄດ້."

        return answer, citations_ok
    except Exception as e:
        return f"ຂໍອະໄພ, ລະບົບບໍ່ສາມາດສ້າງຄຳຕອບໄດ້. ({str(e)[:50]})", False

# ═══════════════════════════════════════
# RESPONSE CACHE
# ═══════════════════════════════════════
if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

def get_cached(query):
    qhash = hashlib.md5(query.strip().lower().encode()).hexdigest()
    cached = st.session_state.response_cache.get(qhash)
    if cached and time.time() - cached.get('time', 0) < 86400:
        return cached
    return None

def set_cache(query, data):
    qhash = hashlib.md5(query.strip().lower().encode()).hexdigest()
    data['time'] = time.time()
    st.session_state.response_cache[qhash] = data

# ═══════════════════════════════════════
# UI LAYOUT
# ═══════════════════════════════════════

# Header
st.markdown("""
<div class="main-title">
    <h1>⚖️ ຜູ້ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ</h1>
    <p>Lao Legal Research Assistant — Powered by AI</p>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer-bar">
    ⚠️ ນີ້ແມ່ນເຄື່ອງມືຄົ້ນຄວ້າເທົ່ານັ້ນ — ບໍ່ແມ່ນຄຳປຶກສາທາງກົດໝາຍ — ກະລຸນາປຶກສາທະນາຍຄວາມ
</div>
""", unsafe_allow_html=True)

# Load system
sys = load_system()

# Sidebar
with st.sidebar:
    st.markdown("### 📚 ກົດໝາຍທີ່ມີ")
    for law in sys['registry']['laws']:
        st.markdown(f"📜 **{law['law_name_lao']}**")
        st.caption(f"{law['law_name_en']} ({law['year']}) · {law['total_chunks']} ມາດຕາ")
    st.divider()
    st.markdown(f"📊 ລວມ: **{sys['registry']['total_laws']}** ກົດໝາຍ · **{sys['registry']['total_chunks']}** ມາດຕາ")
    st.divider()
    st.markdown("### ⚙️ ຕັ້ງຄ່າ")
    show_debug = st.toggle("ສະແດງ Debug", value=False)
    show_sources = st.toggle("ສະແດງແຫຼ່ງອ້າງອີງ", value=True)

# Example queries
st.markdown("##### 💡 ຕົວຢ່າງຄຳຖາມ:")
example_cols = st.columns(2)
examples = [
    "ມາດຕາ 52 ກົດໝາຍວິສາຫະກິດ",
    "ທຶນຈົດທະບຽນບໍລິສັດປະກັນໄພ",
    "ເງື່ອນໄຂການລົງທຶນຕ່າງປະເທດ",
    "ການຂຶ້ນທະບຽນວິສາຫະກິດ",
]
for i, ex in enumerate(examples):
    col = example_cols[i % 2]
    if col.button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state.pending_query = ex

# Chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if msg['role'] == 'assistant':
            st.markdown(msg['content'])
            if msg.get('sources') and show_sources:
                with st.expander("📎 ແຫຼ່ງອ້າງອີງ", expanded=False):
                    for s in msg['sources']:
                        st.markdown(f"""<div class="source-card">
                            📜 <b>{s.get('law_lao','')}</b>, ມາດຕາ {s.get('article','')} — {s.get('article_title','')}
                        </div>""", unsafe_allow_html=True)
            if msg.get('confidence'):
                conf = msg['confidence']
                cls = {'HIGH':'conf-high','MEDIUM':'conf-medium','LOW':'conf-low'}.get(conf,'')
                icon = {'HIGH':'🟢','MEDIUM':'🟡','LOW':'🔴'}.get(conf,'⚪')
                cached_tag = '<span class="cached-tag">💨 cached</span>' if msg.get('from_cache') else ''
                st.markdown(f'{icon} <span class="confidence-badge {cls}">{conf}</span>{cached_tag}',
                           unsafe_allow_html=True)
        else:
            st.markdown(msg['content'])

# Chat input
pending = st.session_state.pop('pending_query', None)
user_input = st.chat_input("ພິມຄຳຖາມກ່ຽວກັບກົດໝາຍລາວ...") or pending

if user_input:
    # Show user message
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Process
    with st.chat_message('assistant'):
        # Validate
        if len(user_input.strip()) < 3:
            response_text = "❌ ຄຳຖາມສັ້ນເກີນໄປ — ກະລຸນາໃສ່ລາຍລະອຽດເພີ່ມ"
            st.markdown(response_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response_text})
        else:
            # Check cache
            cached = get_cached(user_input)
            if cached:
                answer_text = cached['answer']
                sources = cached.get('sources', [])
                confidence = cached.get('confidence', 'MEDIUM')
                from_cache = True
                citations_ok = cached.get('citations_ok', True)
            else:
                with st.spinner("🔍 ກຳລັງຄົ້ນຫາ..."):
                    t0 = time.time()
                    search_results, confidence, law_filter, article_num = search(user_input, sys)
                    t_search = time.time() - t0

                with st.spinner("🤖 ກຳລັງສ້າງຄຳຕອບ..."):
                    t1 = time.time()
                    answer_text, citations_ok = generate_answer(
                        user_input, search_results, confidence, sys)
                    t_gen = time.time() - t1

                sources = []
                for r in search_results[:3]:
                    meta = r['chunk']['metadata']
                    sources.append({
                        'law_lao': meta.get('law_name_lao', ''),
                        'law_en': meta.get('law_name_en', ''),
                        'article': meta.get('article', 0),
                        'article_title': meta.get('article_title', ''),
                    })

                from_cache = False
                set_cache(user_input, {
                    'answer': answer_text,
                    'sources': sources,
                    'confidence': confidence,
                    'citations_ok': citations_ok,
                })

            # Display answer
            st.markdown(answer_text)

            # Sources
            if sources and show_sources:
                with st.expander("📎 ແຫຼ່ງອ້າງອີງ", expanded=False):
                    for s in sources:
                        st.markdown(f"""<div class="source-card">
                            📜 <b>{s.get('law_lao','')}</b>, ມາດຕາ {s.get('article','')} — {s.get('article_title','')}
                        </div>""", unsafe_allow_html=True)

            # Confidence
            conf_cls = {'HIGH':'conf-high','MEDIUM':'conf-medium','LOW':'conf-low'}.get(confidence,'')
            conf_icon = {'HIGH':'🟢','MEDIUM':'🟡','LOW':'🔴'}.get(confidence,'⚪')
            cache_html = '<span class="cached-tag">💨 cached</span>' if from_cache else ''
            st.markdown(f'{conf_icon} <span class="confidence-badge {conf_cls}">{confidence}</span>{cache_html}',
                       unsafe_allow_html=True)

            # Disclaimer
            st.caption("⚠️ ນີ້ແມ່ນເຄື່ອງມືຄົ້ນຄວ້າເທົ່ານັ້ນ — ບໍ່ແມ່ນຄຳປຶກສາທາງກົດໝາຍ — ກະລຸນາປຶກສາທະນາຍຄວາມ")

            # Debug
            if show_debug and not from_cache:
                with st.expander("🔧 Debug", expanded=False):
                    st.json({
                        'search_time': f"{t_search:.2f}s",
                        'generation_time': f"{t_gen:.2f}s",
                        'total_time': f"{t_search+t_gen:.2f}s",
                        'confidence': confidence,
                        'citations_verified': citations_ok,
                        'law_detected': law_filter,
                        'article_number': article_num,
                        'num_results': len(search_results),
                        'from_cache': from_cache,
                    })

            # Feedback
            col1, col2, col3 = st.columns([1,1,4])
            with col1:
                st.button("👍", key=f"up_{len(st.session_state.messages)}")
            with col2:
                st.button("👎", key=f"down_{len(st.session_state.messages)}")

            # Save to session
            st.session_state.messages.append({
                'role': 'assistant',
                'content': answer_text,
                'sources': sources,
                'confidence': confidence,
                'from_cache': from_cache,
            })
