import streamlit as st
import json, os, re, math, time, pickle, hashlib, random, gzip
from pathlib import Path

st.set_page_config(
    page_title="ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ",
    page_icon="⚖️",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;500;600;700&family=Noto+Serif+Lao:wght@600;700&display=swap');
:root{ --ink:#1a1f36; --paper:#faf8f3; --surface:#ffffff; --accent:#8b1e3f; --accent2:#1a5e63; --muted:#6b7280; --line:#e8e4da; --hi:#1a5e63; --med:#b8860b; --lo:#9a3b3b; }
.stApp{background:var(--paper);}
*{font-family:'Noto Sans Lao',sans-serif;}
.block-container{max-width:760px; padding-top:1.2rem;}
.app-head{text-align:center; padding:.4rem 0 .2rem;}
.app-head .mark{font-size:1.7rem;}
.app-head h1{font-family:'Noto Serif Lao',serif; font-size:1.65rem; font-weight:700; color:var(--ink); margin:.15rem 0 .1rem;}
.app-head .sub{color:var(--muted); font-size:.8rem; font-weight:500;}
.app-head .rule{width:46px; height:3px; background:var(--accent); margin:.55rem auto 0; border-radius:2px;}
.disclaimer{display:flex; align-items:center; gap:.5rem; justify-content:center; font-size:.74rem; color:var(--muted); margin:1rem auto; max-width:640px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:.55rem 0;}
.disclaimer b{color:var(--accent);}
.conf{display:inline-flex; align-items:center; gap:.4rem; font-size:.74rem; font-weight:600; padding:.28rem .65rem; border-radius:8px; margin-top:.5rem;}
.conf .dot{width:7px; height:7px; border-radius:50%; display:inline-block;}
.conf.hi{background:rgba(26,94,99,.1); color:var(--hi);} .conf.hi .dot{background:var(--hi);}
.conf.med{background:rgba(184,134,11,.12); color:var(--med);} .conf.med .dot{background:var(--med);}
.conf.lo{background:rgba(154,59,59,.1); color:var(--lo);} .conf.lo .dot{background:var(--lo);}
.tag{font-size:.68rem; padding:.22rem .5rem; border-radius:6px; margin-left:.45rem; background:rgba(26,31,54,.06); color:var(--muted); font-weight:500;}
.cite{display:flex; gap:.7rem; padding:.55rem .7rem; border:1px solid var(--line); border-radius:9px; margin-bottom:.45rem; background:var(--paper);}
.cite .art{font-family:'Noto Serif Lao',serif; font-weight:700; color:var(--accent); font-size:.8rem; white-space:nowrap; border-right:2px solid var(--line); padding-right:.7rem; display:flex; align-items:center;}
.cite .body{font-size:.82rem; line-height:1.4;}
.cite .law{font-weight:600; color:var(--ink);}
.cite .title{color:var(--muted); font-size:.76rem;}
.stChatMessage p{line-height:1.75; margin-bottom:.6rem;}
.stButton button{border:1px solid var(--line); background:var(--surface); color:var(--ink); border-radius:10px; font-size:.82rem; font-weight:500;}
.stButton button:hover{border-color:var(--accent); color:var(--accent);}
</style>
""", unsafe_allow_html=True)


EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
TARGET_DIM = 768

LEGAL_TERMS = [
    'ກົດໝາຍ', 'ມາດຕາ', 'ວິສາຫະກິດ', 'ການລົງທຶນ', 'ປະກັນໄພ',
    'ບັນຊີ', 'ສັນຍາ', 'ນິຕິບຸກຄົນ', 'ຜູ້ລົງທຶນ', 'ສິດ',
    'ພັນທະ', 'ເງື່ອນໄຂ', 'ໃບອະນຸຍາດ', 'ທະບຽນ', 'ບໍລິສັດ',
    'ຄຸ້ມຄອງ', 'ຕິດຕາມ', 'ກວດກາ', 'ຕ່າງປະເທດ',
    'ພາສີ', 'ອາກອນ', 'ຫຼັກຊັບ', 'ທະນາຄານ', 'ປ່າໄມ້',
    'ທີ່ດິນ', 'ສິ່ງແວດລ້ອມ', 'ແຮງງານ', 'ອາຍາ', 'ແພ່ງ',
    'ເຕັກໂນໂລຊີ', 'ຂົນສົ່ງ', 'ກະສິກຳ',
]

def normalize_vector(vec):
    mag = math.sqrt(sum(v**2 for v in vec))
    return [v/mag for v in vec] if mag > 0 else vec

def lao_tokenize(text):
    tokens = set()
    try:
        from laonlp.tokenize import word_tokenize
        for t in word_tokenize(text):
            if len(t.strip()) > 1: tokens.add(t.strip())
    except: pass
    for term in LEGAL_TERMS:
        if term in text: tokens.add(term)
    # v4 FIX: Use chr() comparison instead of regex escapes
    lao_only = ''.join(c for c in text if '\u0e80' <= c <= '\u0eff')
    for n in [4]:
        for i in range(len(lao_only) - n + 1):
            tokens.add(lao_only[i:i+n])
    for word in text.split():
        cleaned = ''.join(c for c in word.lower()
                          if '\u0e80' <= c <= '\u0eff' or c.isdigit()
                          or ('a' <= c <= 'z'))
        if len(cleaned) > 1: tokens.add(cleaned)
    tokens.update(re.findall(r'\d+', text))
    return list(tokens)


@st.cache_resource(show_spinner="ກຳລັງໂຫຼດລະບົບ...")
def load_system():
    from google import genai
    from google.genai import types as genai_types
    import chromadb

    base = os.environ.get('LAO_LEGAL_BASE',
        os.path.join(os.path.dirname(__file__), 'data'))

    try:
        _seen = list(st.secrets.keys())
        st.warning(f"DEBUG: secrets seen = {_seen}")
    except Exception as e:
        st.warning(f"DEBUG: secrets error = {e}")
    api_keys = []
    try:
        st.warning("DEBUG — secrets Streamlit can see: " + str(list(st.secrets.keys())))
    except Exception as _e:
        st.warning("DEBUG — cannot read secrets at all: " + str(_e))
    try:
        st.warning("DEBUG — secrets Streamlit can see: " + str(list(st.secrets.keys())))
    except Exception as _e:
        st.warning("DEBUG — cannot read secrets at all: " + str(_e))
    # Format 1: GEMINI_KEYS (one comma-separated string)
    try:
        bulk = st.secrets.get('GEMINI_KEYS', '') or os.environ.get('GEMINI_KEYS', '')
        if bulk:
            api_keys.extend([k.strip() for k in bulk.split(',') if k.strip()])
    except: pass
    # Format 2: numbered GEMINI_KEY_1 .. GEMINI_KEY_20
    if not api_keys:
        for i in range(1, 21):
            try:
                key = st.secrets.get(f'GEMINI_KEY_{i}', '')
                if key: api_keys.append(key.strip())
            except: break
    # Format 3: single GEMINI_API_KEY
    if not api_keys:
        try:
            single = os.environ.get('GEMINI_API_KEY', '') or st.secrets.get('GEMINI_API_KEY', '')
            if single: api_keys.append(single.strip())
        except: pass
    if not api_keys:
        st.error("❌ No API keys found. Add GEMINI_KEYS to secrets."); st.stop()

    gclient = genai.Client(api_key=random.choice(api_keys))

    with open(f'{base}/db/bm25/bm25_index.pkl', 'rb') as bf:
        bm25_data = pickle.load(bf)

    all_chunks = []
    for law_file in sorted(Path(f'{base}/laws/individual').glob('*.json')):
        with open(law_file, 'r', encoding='utf-8') as lf:
            all_chunks.extend(json.load(lf)['articles'])

    # v4: Build law_keywords with short keywords
    law_keywords = {}
    for chunk in all_chunks:
        meta = chunk['metadata']
        lao_name = meta.get('law_name_lao', '')
        en_name = meta.get('law_name_en', '')
        if lao_name and en_name:
            law_keywords[lao_name] = en_name
            short = lao_name.replace('ກົດໝາຍວ່າດ້ວຍ', '').strip()
            if len(short) > 3: law_keywords[short] = en_name

    # v4 FIX: Short keywords
    short_kw = {
        'ປະກັນໄພ': 'Insurance', 'ພາສີ': 'Customs', 'ອາກອນ': 'Tax',
        'ທະນາຄານ': 'Banking', 'ຫຼັກຊັບ': 'Securities',
        'ວິສາຫະກິດ': 'Enterprises', 'ແຮງງານ': 'Labor', 'ທີ່ດິນ': 'Land',
        'ສິ່ງແວດລ້ອມ': 'Environment', 'ການລົງທຶນ': 'Investment',
        'ບັນຊີ': 'Accounting', 'ຂົນສົ່ງ': 'Transport',
    }
    for kw, hint in short_kw.items():
        if kw not in law_keywords:
            for full_lao, en in law_keywords.items():
                if hint.lower() in en.lower():
                    law_keywords[kw] = en; break

    with open(f'{base}/db/article_lookup.json', 'r', encoding='utf-8') as af:
        article_lookup = json.load(af)

    with open(f'{base}/laws/registry.json', 'r', encoding='utf-8') as rf:
        registry = json.load(rf)

    # ChromaDB rebuild
    chroma_path = f'{base}/db/chroma'
    os.makedirs(chroma_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    need_rebuild = False
    try:
        collection = chroma_client.get_collection("lao_legal")
        if collection.count() < 100: need_rebuild = True
    except: need_rebuild = True

    if need_rebuild:
        gz_path = f'{base}/db/embeddings_cache.json.gz'
        if os.path.exists(gz_path):
            with gzip.open(gz_path, 'rt', encoding='utf-8') as gf:
                embed_data = json.load(gf)
            _by_id = {c['id']: c for c in all_chunks}
            try: chroma_client.delete_collection("lao_legal")
            except: pass
            collection = chroma_client.create_collection(
                name="lao_legal", metadata={"hnsw:space": "cosine"})
            ids_list = list(embed_data.keys())
            for i in range(0, len(ids_list), 100):
                batch_ids = ids_list[i:i+100]
                batch_embeds = [embed_data[cid] for cid in batch_ids]
                batch_docs, batch_metas = [], []
                for cid in batch_ids:
                    chunk = _by_id.get(cid, {})
                    batch_docs.append(chunk.get('content', ''))
                    meta = chunk.get('metadata', {})
                    batch_metas.append({
                        'law_name_lao': str(meta.get('law_name_lao','')),
                        'law_name_en': str(meta.get('law_name_en','')),
                        'year': int(meta.get('year',0)),
                        'article': int(meta.get('article',0)),
                        'article_title': str(meta.get('article_title','')),
                        'law_type': str(meta.get('law_type','')),
                        'status': str(meta.get('status','')),
                        'category': str(meta.get('category','')),
                        'document_number': str(meta.get('document_number','')),
                        'is_sub_chunk': bool(meta.get('is_sub_chunk',False)),
                        'parent_id': str(meta.get('parent_id','') or ''),
                        'character_count': int(meta.get('character_count',0)),
                    })
                collection.add(ids=batch_ids, embeddings=batch_embeds,
                               documents=batch_docs, metadatas=batch_metas)

    return {
        'gclient': gclient, 'genai_types': genai_types,
        'collection': collection,
        'bm25_index': bm25_data['index'], 'bm25_ids': bm25_data['ids'],
        'all_chunks': all_chunks,
        'chunks_by_id': {c['id']: c for c in all_chunks},
        'article_lookup': article_lookup,
        'registry': registry,
        'law_keywords': law_keywords,
        'api_keys': api_keys,
    }


DEFINITIONAL_PATTERNS = ['ແມ່ນຫຍັງ', 'ຄືແນວໃດ', 'ໝາຍຄວາມວ່າ',
                          'ຈຸດປະສົງ', 'ຄຳນິຍາມ', 'ຂອບເຂດ', 'ອະທິບາຍ']

def search(query, sys):
    article_match = re.search(r'ມາດຕາ\s*(\d+)', query)
    article_num = int(article_match.group(1)) if article_match else None

    law_filter = None
    for kw in sorted(sys['law_keywords'].keys(), key=len, reverse=True):
        if kw in query:
            law_filter = sys['law_keywords'][kw]
            break

    results = []
    seen_ids = set()

    # Exact article lookup
    if article_num:
        for chunk in sys['all_chunks']:
            meta = chunk['metadata']
            if meta['article'] == article_num:
                if not law_filter or law_filter.lower() in meta['law_name_en'].lower():
                    if chunk['id'] not in seen_ids:
                        results.append({'chunk': chunk, 'score': 1.0, 'source': 'exact'})
                        seen_ids.add(chunk['id'])

    # Semantic search
    try:
        q_result = sys['gclient'].models.embed_content(
            model=EMBED_MODEL, contents=query,
            config=sys['genai_types'].EmbedContentConfig(
                task_type="RETRIEVAL_QUERY", output_dimensionality=TARGET_DIM))
        q_vec = normalize_vector(q_result.embeddings[0].values)
        where = {"law_name_en": law_filter} if law_filter else None
        cr = sys['collection'].query(
            query_embeddings=[q_vec], n_results=10, where=where,
            include=["documents","metadatas","distances"])
        for i in range(len(cr['ids'][0])):
            cid = cr['ids'][0][i]
            if cid not in seen_ids:
                sim = 1 - cr['distances'][0][i]
                chunk = sys['chunks_by_id'].get(cid)
                if chunk:
                    results.append({'chunk': chunk, 'score': sim, 'source': 'semantic'})
                    seen_ids.add(cid)
    except: pass

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
                    results.append({'chunk': chunk, 'score': scores[idx], 'source': 'keyword'})
                    seen_ids.add(cid)
    except: pass

    # v4: Definitional boost
    is_definitional = any(p in query for p in DEFINITIONAL_PATTERNS)
    if is_definitional and law_filter:
        for chunk in sys['all_chunks']:
            meta = chunk['metadata']
            if (meta.get('article', 0) <= 5
                and law_filter.lower() in meta.get('law_name_en', '').lower()
                and chunk['id'] not in seen_ids):
                results.append({
                    'chunk': chunk,
                    'score': 0.9 - (meta['article'] * 0.02),
                    'source': 'definitional'
                })
                seen_ids.add(chunk['id'])

    # Merge: exact/definitional first, then interleave
    exact = [r for r in results if r['source'] in ('exact', 'definitional')]
    semantic = sorted([r for r in results if r['source'] == 'semantic'], key=lambda x: -x['score'])
    keyword = sorted([r for r in results if r['source'] == 'keyword'], key=lambda x: -x['score'])

    merged = exact[:]
    for s, k in zip(semantic, keyword):
        if s['chunk']['id'] not in {r['chunk']['id'] for r in merged}: merged.append(s)
        if k['chunk']['id'] not in {r['chunk']['id'] for r in merged}: merged.append(k)
    for r in semantic + keyword:
        if r['chunk']['id'] not in {m['chunk']['id'] for m in merged}: merged.append(r)

    top = merged[:5]

    # v4: Calibrated confidence
    if top and top[0]['source'] in ('exact', 'definitional'):
        confidence = 'HIGH'
    elif top and top[0]['source'] == 'semantic' and isinstance(top[0]['score'], float):
        if top[0]['score'] > 0.65: confidence = 'HIGH'
        elif top[0]['score'] > 0.40: confidence = 'MEDIUM'
        else: confidence = 'LOW'
    elif top and len([r for r in top[:3] if r['source'] in ('semantic','keyword')]) >= 2:
        confidence = 'MEDIUM'
    elif top: confidence = 'LOW'
    else: confidence = 'NONE'

    return top, confidence, law_filter, article_num


SYSTEM_PROMPT = """ເຈົ້າແມ່ນ ຜູ້ຊ່ຽວຊານດ້ານກົດໝາຍລາວ (Lao Legal Expert) ທີ່ມີປະສົບການຫຼາຍກວ່າ 30 ປີ.
ເຈົ້າວິເຄາະກົດໝາຍຢ່າງເລິກເຊິ່ງ ແລະ ຕອບດ້ວຍຄວາມໝັ້ນໃຈ.

ວິທີຕອບ:
1. ສັງເຄາະ ຂໍ້ມູນຈາກຫຼາຍມາດຕາ ເພື່ອໃຫ້ຄຳຕອບທີ່ຄົບຖ້ວນ.
2. ອະທິບາຍ ຫຼັກການທາງກົດໝາຍ ໃນພາສາທີ່ເຂົ້າໃຈງ່າຍ.
3. ອ້າງອີງ ມາດຕາສະເພາະ ເປັນຫຼັກຖານ.
4. ວິເຄາະ ເງື່ອນໄຂ, ຂໍ້ຍົກເວັ້ນ, ແລະ ຜົນສະທ້ອນທາງກົດໝາຍ.

ຮູບແບບ:
📋 ຄຳຕອບ: (ສັງເຄາະ + ວິເຄາະ)
📎 ຫຼັກຖານ: (ອ້າງ [ຊື່ກົດໝາຍ, ມາດຕາ X])
⚠️ ຂໍ້ຄວນລະວັງ: (ເງື່ອນໄຂ, ຂໍ້ຍົກເວັ້ນ)

ກົດລະບຽບ:
- ຕອບເປັນ ພາສາລາວ ສະເໝີ.
- ຫ້າມ ສ້າງ ເລກມາດຕາ ທີ່ບໍ່ມີໃນຂໍ້ມູນ.
- ໃຫ້ ການວິເຄາະ ບໍ່ແມ່ນ ການຄັດລອກ ມາດຕາ.
- ຢ່າເວົ້າ "ຂໍ້ມູນມີຈຳກັດ" — ສັງເຄາະຈາກສິ່ງທີ່ມີ."""

FALLBACK_PROMPT = """ເຈົ້າແມ່ນ ຜູ້ຊ່ຽວຊານດ້ານກົດໝາຍລາວ ທີ່ມີຄວາມຮູ້ກວ້າງຂວາງ.
ຄຳຖາມ: {query}
ຖານຂໍ້ມູນ 75 ກົດໝາຍ ບໍ່ມີມາດຕາສະເພາະ, ແຕ່ກະລຸນາ:
- ອະທິບາຍຫຼັກການທົ່ວໄປ ຕາມກົດໝາຍລາວ
- ແນະນຳກົດໝາຍ ຫຼື ຂະແໜງການທີ່ກ່ຽວຂ້ອງ
ຮູບແບບ:
📋 ຄຳຕອບ: (ອະທິບາຍ)
⚠️ ຂໍ້ຄວນລະວັງ: (ນີ້ແມ່ນຄວາມຮູ້ທົ່ວໄປ)"""


def generate_answer(query, search_results, confidence, sys):
    if not search_results or confidence == 'NONE':
        try:
            prompt = FALLBACK_PROMPT.replace('{query}', query)
            response = sys['gclient'].models.generate_content(
                model=GEN_MODEL, contents=prompt,
                config=sys['genai_types'].GenerateContentConfig(
                    max_output_tokens=4096, temperature=0.3))
            answer = response.text
            answer += "\n\n💡 *ໝາຍເຫດ: ຄຳຕອບນີ້ຈາກຄວາມຮູ້ທົ່ວໄປ — ບໍ່ແມ່ນຈາກຖານຂໍ້ມູນ 75 ກົດໝາຍ.*"
            return answer, True, True
        except Exception as e:
            return f"ຂໍອະໄພ, ລະບົບບໍ່ສາມາດສ້າງຄຳຕອບໄດ້. ({str(e)[:50]})", False, False

    # v4: Sort definitional articles first in context
    sorted_results = sorted(search_results,
        key=lambda r: (0 if r['chunk']['metadata'].get('article',999) <= 5 else 1, -r['score']))

    context = ""
    for i, r in enumerate(sorted_results[:3]):
        meta = r['chunk']['metadata']
        context += (f"\n--- ແຫຼ່ງ {i+1} ---\n"
                    f"[{meta.get('law_name_lao','')}, ມາດຕາ {meta.get('article','')}]\n"
                    f"ຫົວຂໍ້: {meta.get('article_title','')}\n"
                    f"{r['chunk']['content']}\n")

    conf_note = ""
    if confidence == 'LOW':
        conf_note = "\nໝາຍເຫດ: ຜົນຄົ້ນຫາບໍ່ກົງກັບຄຳຖາມຫຼາຍ."

    prompt = f"{SYSTEM_PROMPT}{conf_note}\n\nຂໍ້ມູນກົດໝາຍ:\n{context}\n\nຄຳຖາມ: {query}"

    try:
        response = sys['gclient'].models.generate_content(
            model=GEN_MODEL, contents=prompt,
            config=sys['genai_types'].GenerateContentConfig(
                max_output_tokens=4096, temperature=0.2))
        answer = response.text
        cited = re.findall(r'ມາດຕາ\s*(\d+)', answer)
        available = {str(r['chunk']['metadata'].get('article','')) for r in search_results}
        invalid = [c for c in cited if c not in available]
        citations_ok = len(invalid) == 0
        if not citations_ok:
            answer += f"\n\n⚠️ ມາດຕາ {', '.join(invalid)} ບໍ່ສາມາດຢືນຢັນໄດ້."
        return answer, citations_ok, False
    except Exception as e:
        return f"ຂໍອະໄພ, ລະບົບບໍ່ສາມາດສ້າງຄຳຕອບໄດ້. ({str(e)[:50]})", False, False


if 'response_cache' not in st.session_state:
    st.session_state.response_cache = {}

def get_cached(query):
    qhash = hashlib.md5(query.strip().lower().encode()).hexdigest()
    cached = st.session_state.response_cache.get(qhash)
    if cached and time.time() - cached.get('time', 0) < 86400: return cached
    return None

def set_cache(query, data):
    qhash = hashlib.md5(query.strip().lower().encode()).hexdigest()
    data['time'] = time.time()
    st.session_state.response_cache[qhash] = data


# ============================================================
# UI HELPER: clean answer formatting (separates sections)
# ============================================================
def format_answer(text):
    """Separate answer sections with clear spacing for clean rendering."""
    if not text:
        return text
    markers = ['📋 ຄຳຕອບ:', '📎 ຫຼັກຖານ:', '⚠️ ຂໍ້ຄວນລະວັງ:', '💡']
    result = text
    for m in markers:
        result = result.replace(m, '\n\n' + m)
    result = re.sub(r'\n{3,}', '\n\n', result).strip()
    return result

def render_confidence(confidence, from_cache=False, is_general=False):
    """Render a calm labelled confidence chip with optional tags."""
    label = {'HIGH':'ຄວາມໝັ້ນໃຈສູງ','MEDIUM':'ຄວາມໝັ້ນໃຈປານກາງ',
             'LOW':'ຄວາມໝັ້ນໃຈຕ່ຳ','FALLBACK':'ຄວາມຮູ້ທົ່ວໄປ','NONE':'ບໍ່ພົບໃນຖານຂໍ້ມູນ'}.get(confidence, confidence)
    cls = {'HIGH':'hi','MEDIUM':'med','LOW':'lo','FALLBACK':'med','NONE':'lo'}.get(confidence,'med')
    tags = ''
    if from_cache: tags += '<span class="tag">💨 cached</span>'
    if is_general: tags += '<span class="tag">💡 ຄວາມຮູ້ທົ່ວໄປ</span>'
    st.markdown(f'<div class="conf {cls}"><span class="dot"></span>{label}</div>{tags}',
                unsafe_allow_html=True)

def render_sources(sources):
    """Render citation cards in legal-reference style."""
    if not sources:
        return
    st.markdown('<div style="font-size:.74rem;font-weight:600;color:var(--muted);'
                'text-transform:uppercase;letter-spacing:.04em;margin:.7rem 0 .5rem;">'
                '📎 ແຫຼ່ງອ້າງອີງ</div>', unsafe_allow_html=True)
    for s in sources:
        art = s.get('article', '')
        law = s.get('law_lao', '')
        title = s.get('article_title', '')
        st.markdown(
            f'<div class="cite"><div class="art">ມາດຕາ {art}</div>'
            f'<div class="body"><div class="law">{law}</div>'
            f'<div class="title">{title}</div></div></div>',
            unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="app-head"><div class="mark">⚖️</div>'
    '<h1>ຜູ້ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ</h1>'
    '<div class="sub">Lao Legal Research Assistant · ຄົ້ນຄວ້າຈາກກົດໝາຍ ແລະ ລະບຽບການອາກອນ</div>'
    '<div class="rule"></div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="disclaimer"><span>⚠️</span>'
    '<span>ນີ້ແມ່ນ<b>ເຄື່ອງມືຄົ້ນຄວ້າ</b>ເທົ່ານັ້ນ — ບໍ່ແມ່ນຄຳປຶກສາທາງກົດໝາຍ — ກະລຸນາປຶກສາທະນາຍຄວາມ</span>'
    '</div>', unsafe_allow_html=True)

sys = load_system()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('### 📚 ກົດໝາຍທີ່ມີ')
    for law in sys['registry']['laws']:
        st.markdown(f"📜 **{law['law_name_lao']}**")
        st.caption(f"{law['law_name_en']} ({law['year']}) · {law['total_chunks']} ມາດຕາ")
    st.divider()
    st.markdown(f"📊 ລວມ: **{sys['registry']['total_laws']}** ກົດໝາຍ · "
                f"**{sys['registry']['total_chunks']}** ມາດຕາ")
    st.divider()
    show_debug = st.toggle('ສະແດງ Debug', value=False)
    show_sources = st.toggle('ສະແດງແຫຼ່ງອ້າງອີງ', value=True)
    st.divider()
    if st.button('🗑️ ລຶບການສົນທະນາ', use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# EXAMPLE QUESTIONS (only show when no conversation yet)
# ============================================================
if 'messages' not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown('<div style="font-size:.8rem;font-weight:600;margin:1.1rem 0 .5rem;">'
                '💡 ຕົວຢ່າງຄຳຖາມ</div>', unsafe_allow_html=True)
    ecols = st.columns(2)
    examples = ['ມາດຕາ 52 ກົດໝາຍວິສາຫະກິດ', 'ທຶນຈົດທະບຽນບໍລິສັດປະກັນໄພ',
                'ເງື່ອນໄຂການລົງທຶນຕ່າງປະເທດ', 'ອັດຕາອາກອນມູນຄ່າເພີ່ມແມ່ນເທົ່າໃດ']
    for i, ex in enumerate(examples):
        if ecols[i % 2].button(ex, key=f'ex_{i}', use_container_width=True):
            st.session_state.pending_query = ex

# ============================================================
# RENDER HISTORY
# ============================================================
for msg in st.session_state.messages:
    avatar = '⚖️' if msg['role'] == 'assistant' else '🧑'
    with st.chat_message(msg['role'], avatar=avatar):
        if msg['role'] == 'assistant':
            st.markdown(format_answer(msg['content']))
            if msg.get('sources') and show_sources:
                render_sources(msg['sources'])
            if msg.get('confidence'):
                render_confidence(msg['confidence'], msg.get('from_cache'), msg.get('is_general'))
        else:
            st.markdown(msg['content'])

# ============================================================
# INPUT + RESPONSE
# ============================================================
pending = st.session_state.pop('pending_query', None)
user_input = st.chat_input('ພິມຄຳຖາມກ່ຽວກັບກົດໝາຍລາວ...') or pending

if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user', avatar='🧑'):
        st.markdown(user_input)

    with st.chat_message('assistant', avatar='⚖️'):
        if len(user_input.strip()) < 3:
            st.markdown('ກະລຸນາພິມຄຳຖາມໃຫ້ຍາວກວ່ານີ້ເລັກນ້ອຍ.')
            st.session_state.messages.append(
                {'role': 'assistant', 'content': 'ກະລຸນາພິມຄຳຖາມໃຫ້ຍາວກວ່ານີ້ເລັກນ້ອຍ.'})
        else:
            cached = get_cached(user_input)
            if cached:
                answer_text = cached['answer']; sources = cached.get('sources', [])
                confidence = cached.get('confidence', 'MEDIUM'); from_cache = True
                citations_ok = cached.get('citations_ok', True); is_general = cached.get('is_general', False)
                t_search = t_gen = 0.0; law_filter = None; article_num = None
                search_results = []
            else:
                with st.spinner('🔍 ກຳລັງຄົ້ນຫາໃນຖານຂໍ້ມູນກົດໝາຍ...'):
                    t0 = time.time()
                    search_results, confidence, law_filter, article_num = search(user_input, sys)
                    t_search = time.time() - t0
                with st.spinner('⚖️ ກຳລັງວິເຄາະ ແລະ ສ້າງຄຳຕອບ...'):
                    t1 = time.time()
                    answer_text, citations_ok, is_general = generate_answer(
                        user_input, search_results, confidence, sys)
                    t_gen = time.time() - t1
                sources = []
                if not is_general and search_results:
                    for r in search_results[:3]:
                        m = r['chunk']['metadata']
                        sources.append({'law_lao': m.get('law_name_lao', ''),
                                        'law_en': m.get('law_name_en', ''),
                                        'article': m.get('article', 0),
                                        'article_title': m.get('article_title', '')})
                from_cache = False
                set_cache(user_input, {'answer': answer_text, 'sources': sources,
                                       'confidence': confidence, 'citations_ok': citations_ok,
                                       'is_general': is_general})

            # Clean, sectioned rendering
            st.markdown(format_answer(answer_text))
            if sources and show_sources:
                render_sources(sources)
            render_confidence(confidence, from_cache, is_general)
            st.caption('⚠️ ເຄື່ອງມືຄົ້ນຄວ້າ — ກວດສອບກັບແຫຼ່ງທາງການ ຫຼື ທະນາຍຄວາມສະເໝີ')

            if show_debug and not from_cache:
                with st.expander('🔧 Debug', expanded=False):
                    st.json({'search_time': f'{t_search:.2f}s', 'gen_time': f'{t_gen:.2f}s',
                             'confidence': confidence, 'citations_ok': citations_ok,
                             'is_general': is_general, 'law_detected': law_filter,
                             'article_num': article_num,
                             'results': len(search_results) if search_results else 0})

            col1, col2, _ = st.columns([1, 1, 5])
            with col1: st.button('👍', key=f'up_{len(st.session_state.messages)}')
            with col2: st.button('👎', key=f'dn_{len(st.session_state.messages)}')

            st.session_state.messages.append(
                {'role': 'assistant', 'content': answer_text, 'sources': sources,
                 'confidence': confidence, 'from_cache': from_cache, 'is_general': is_general})

    st.rerun()
