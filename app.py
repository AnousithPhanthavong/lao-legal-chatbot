import streamlit as st
import json, os, re, math, time, pickle, hashlib, random, gzip
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

st.set_page_config(page_title="ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ", page_icon="⚖️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;600;700&display=swap');
* { font-family: "Noto Sans Lao", sans-serif !important; }
.main-title { text-align: center; padding: 1rem 0 0.5rem; }
.main-title h1 { font-size: 1.8rem; background: linear-gradient(135deg, #1a73e8, #34a853); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem; }
.main-title p { color: #9aa0a6; font-size: 0.85rem; }
.disclaimer-bar { background: rgba(234,67,53,0.08); border: 1px solid rgba(234,67,53,0.2); border-radius: 8px; padding: 0.5rem 1rem; text-align: center; font-size: 0.75rem; color: #ea4335; margin-bottom: 1rem; }
.source-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 0.6rem 0.8rem; margin: 0.3rem 0; font-size: 0.8rem; }
.confidence-badge { display: inline-block; padding: 0.2rem 0.8rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
.conf-high { background: rgba(52,168,83,0.15); color: #34a853; }
.conf-medium { background: rgba(251,188,4,0.15); color: #fbbc04; }
.conf-low { background: rgba(234,67,53,0.15); color: #ea4335; }
.cached-tag { display: inline-block; background: rgba(26,115,232,0.1); color: #1a73e8; padding: 0.1rem 0.5rem; border-radius: 8px; font-size: 0.7rem; margin-left: 0.5rem; }
.general-tag { display: inline-block; background: rgba(251,188,4,0.1); color: #fbbc04; padding: 0.1rem 0.5rem; border-radius: 8px; font-size: 0.7rem; margin-left: 0.5rem; }
</style>
""", unsafe_allow_html=True)

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
TARGET_DIM = 768

LEGAL_TERMS = [
    'ກົດໝາຍ', 'ມາດຕາ', 'ລັດຖະທຳມະນູນ', 'ດຳລັດ', 'ລະບຽບການ',
    'ນິຕິກຳ', 'ສັນຍາ', 'ນິຕິບຸກຄົນ', 'ບຸກຄົນ', 'ວິສາຫະກິດ',
    'ການລົງທຶນ', 'ປະກັນໄພ', 'ບັນຊີ', 'ບໍລິສັດ', 'ຜູ້ຖືຮຸ້ນ',
    'ຮຸ້ນ', 'ທຶນ', 'ຜູ້ລົງທຶນ', 'ສໍາປະທານ', 'ໃບອະນຸຍາດ',
    'ທະບຽນ', 'ພາສີ', 'ອາກອນ', 'ທະນາຄານ', 'ຫຼັກຊັບ',
    'ໜີ້ສິນ', 'ງົບປະມານ', 'ການເງິນ', 'ສິນເຊື່ອ', 'ດອກເບ້ຍ',
    'ອາຍາ', 'ໂທດ', 'ຄະດີ', 'ຈຳຄຸກ', 'ປັບໃໝ',
    'ຜູ້ຖືກຫາ', 'ຜູ້ເສຍຫາຍ', 'ອາຊະຍາກຳ', 'ການກະທຳຜິດ', 'ແພ່ງ',
    'ກຳມະສິດ', 'ມໍລະດົກ', 'ຊັບສິນ', 'ສິດ', 'ພັນທະ',
    'ແຮງງານ', 'ຜູ້ອອກແຮງງານ', 'ຜູ້ໃຊ້ແຮງງານ', 'ຄ່າແຮງ', 'ສັນຍາແຮງງານ',
    'ປະກັນສັງຄົມ', 'ທີ່ດິນ', 'ຄອບຄອງ', 'ນຳໃຊ້ທີ່ດິນ', 'ອະສັງຫາລິມະຊັບ',
    'ສິ່ງແວດລ້ອມ', 'ປ່າໄມ້', 'ນ້ຳ', 'ບໍ່ແຮ່', 'ທຳມະຊາດ',
    'ເຕັກໂນໂລຊີ', 'ອີເລັກໂຕຣນິກ', 'ທຸລະກຳ', 'ຂໍ້ມູນ', 'ຄຸ້ມຄອງ',
    'ຕິດຕາມ', 'ກວດກາ', 'ການຄ້າ', 'ນຳເຂົ້າ', 'ສົ່ງອອກ',
    'ສິນຄ້າ', 'ຂົນສົ່ງ', 'ຍານພາຫະນະ', 'ເງື່ອນໄຂ', 'ຕ່າງປະເທດ',
    'ການສົ່ງເສີມ',
]

SITUATION_KEYWORDS = {
    'ໄລ່ອອກ': 'ແຮງງານ',
    'ເຮັດວຽກ': 'ແຮງງານ',
    'ຄ່າຈ້າງ': 'ແຮງງານ',
    'ຄ່າແຮງ': 'ແຮງງານ',
    'ພະນັກງານ': 'ແຮງງານ',
    'ນາຍຈ້າງ': 'ແຮງງານ',
    'ອາຍຸຕໍ່າສຸດ': 'ແຮງງານ',
    'ທີ່ດິນ': 'ທີ່ດິນ',
    'ຊື້ເຮືອນ': 'ທີ່ດິນ',
    'ກຳມະສິດ': 'ທີ່ດິນ',
    'ລັກ': 'ອາຍາ',
    'ຕີ': 'ອາຍາ',
    'ຂ້າ': 'ອາຍາ',
    'ໂທດ': 'ອາຍາ',
    'ຈຳຄຸກ': 'ອາຍາ',
    'ແຕ່ງງານ': 'ຄອບຄົວ',
    'ຢ່າ': 'ຄອບຄົວ',
    'ມໍລະດົກ': 'ຄອບຄົວ',
    'ບໍລິສັດ': 'ວິສາຫະກິດ',
    'ຈົດທະບຽນ': 'ວິສາຫະກິດ',
    'ຮຸ້ນ': 'ວິສາຫະກິດ',
}

DEFINITIONAL = ['ແມ່ນຫຍັງ', 'ຄືແນວໃດ', 'ໝາຍຄວາມວ່າ', 'ຈຸດປະສົງ', 'ຄຳນິຍາມ', 'ຂອບເຂດ', 'ນິຍາມ', 'ອະທິບາຍ']

def normalize_lao(text):
    """Normalize Lao text to canonical spelling."""
    if not text:
        return text
    result = text
    # Apply character-level normalization
    for old, new in _LAO_NORM_MAP.items():
        result = result.replace(old, new)
    # Normalize the specific ພາສິ → ພາສີ case
    # This is the #1 failure cause from testing
    result = result.replace('ພາສິ', 'ພາສີ')
    result = result.replace('ກົດຫມາຍ', 'ກົດໝາຍ')
    result = result.replace('ຫນີ້', 'ໜີ້')
    result = result.replace('ຫນ້າ', 'ໜ້າ')
    result = result.replace('ຫມາຍ', 'ໝາຍ')
    return result


def split_compound_query(query):
    """Split 'A ແລະ B' into ['A', 'B'] for separate searches."""
    # Detect compound connectors
    connectors = [' ແລະ ', ' ກັບ ', ' ຫຼື ', ' ພ້ອມ ']
    for conn in connectors:
        if conn in query:
            parts = query.split(conn)
            # Only split if both parts are substantial (>5 chars)
            parts = [p.strip() for p in parts if len(p.strip()) > 5]
            if len(parts) >= 2:
                return parts
    return [query]


def normalize_vector(vec):
    mag = math.sqrt(sum(v*v for v in vec))
    return [v/mag for v in vec] if mag > 0 else vec

def lao_tokenize(text):
    if not text: return []
    text = normalize_lao(text)
    tokens = set()
    try:
        from laonlp.tokenize import word_tokenize
        for t in word_tokenize(text):
            if len(t.strip()) > 1: tokens.add(t.strip())
    except: pass
    for term in LEGAL_TERMS:
        if term in text: tokens.add(term)
    lao_only = ''.join(c for c in text if '\u0e80' <= c <= '\u0eff')
    for i in range(len(lao_only) - 4 + 1):
        tokens.add(lao_only[i:i+4])
    for word in text.split():
        cleaned = ''.join(c for c in word.lower()
                          if '\u0e80' <= c <= '\u0eff' or c.isdigit() or ('a' <= c <= 'z'))
        if len(cleaned) > 1: tokens.add(cleaned)
    tokens.update(re.findall(r'\d+', text))
    return list(tokens)

@st.cache_resource(show_spinner='ກຳລັງໂຫຼດລະບົບ...')
def load_system():
    from google import genai
    from google.genai import types as genai_types
    import chromadb
    base = os.environ.get('LAO_LEGAL_BASE', os.path.join(os.path.dirname(__file__), 'data'))
    api_keys = []
    for i in range(1, 10):
        try:
            key = st.secrets.get(f'GEMINI_KEY_{i}', '')
            if key: api_keys.append(key)
        except: break
    if not api_keys:
        single = os.environ.get('GEMINI_API_KEY', st.secrets.get('GEMINI_API_KEY', ''))
        if single: api_keys.append(single)
    if not api_keys: st.error('No API keys.'); st.stop()
    gclient = genai.Client(api_key=random.choice(api_keys))
    with open(f'{base}/db/bm25/bm25_index.pkl', 'rb') as bf: bm25_data = pickle.load(bf)
    all_chunks = []
    for lf in sorted(Path(f'{base}/laws/individual').glob('*.json')):
        with open(lf, 'r', encoding='utf-8') as fh: all_chunks.extend(json.load(fh)['articles'])
    law_keywords = {}
    for chunk in all_chunks:
        meta = chunk['metadata']
        lao_n = meta.get('law_name_lao',''); en_n = meta.get('law_name_en','')
        if lao_n and en_n:
            law_keywords[lao_n] = en_n
            short = lao_n.replace('ກົດໝາຍວ່າດ້ວຍ','').strip()
            if len(short) > 3: law_keywords[short] = en_n
    if 'ປະກັນໄພ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Insurance'.lower() in en.lower(): law_keywords['ປະກັນໄພ'] = en; break
    if 'ພາສີ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Customs'.lower() in en.lower(): law_keywords['ພາສີ'] = en; break
    if 'ອາກອນ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Tax'.lower() in en.lower(): law_keywords['ອາກອນ'] = en; break
    if 'ທະນາຄານ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Banking'.lower() in en.lower(): law_keywords['ທະນາຄານ'] = en; break
    if 'ຫຼັກຊັບ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Securities'.lower() in en.lower(): law_keywords['ຫຼັກຊັບ'] = en; break
    if 'ວິສາຫະກິດ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Enterprises'.lower() in en.lower(): law_keywords['ວິສາຫະກິດ'] = en; break
    if 'ແຮງງານ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Labor'.lower() in en.lower(): law_keywords['ແຮງງານ'] = en; break
    if 'ທີ່ດິນ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Land'.lower() in en.lower(): law_keywords['ທີ່ດິນ'] = en; break
    if 'ສິ່ງແວດລ້ອມ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Environment'.lower() in en.lower(): law_keywords['ສິ່ງແວດລ້ອມ'] = en; break
    if 'ການລົງທຶນ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Investment'.lower() in en.lower(): law_keywords['ການລົງທຶນ'] = en; break
    if 'ບັນຊີ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Accounting'.lower() in en.lower(): law_keywords['ບັນຊີ'] = en; break
    if 'ຂົນສົ່ງ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Transport'.lower() in en.lower(): law_keywords['ຂົນສົ່ງ'] = en; break
    if 'ອາຍາ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Criminal'.lower() in en.lower(): law_keywords['ອາຍາ'] = en; break
    if 'ແພ່ງ' not in law_keywords:
        for fl, en in law_keywords.items():
            if 'Civil'.lower() in en.lower(): law_keywords['ແພ່ງ'] = en; break
    with open(f'{base}/db/article_lookup.json', 'r', encoding='utf-8') as af: article_lookup = json.load(af)
    with open(f'{base}/laws/registry.json', 'r', encoding='utf-8') as rf: registry = json.load(rf)
    chroma_path = f'{base}/db/chroma'
    os.makedirs(chroma_path, exist_ok=True)
    cc = chromadb.PersistentClient(path=chroma_path)
    rebuild = False
    try:
        col = cc.get_collection('lao_legal')
        if col.count() < 100: rebuild = True
    except: rebuild = True
    if rebuild:
        gz = f'{base}/db/embeddings_cache.json.gz'
        if os.path.exists(gz):
            with gzip.open(gz,'rt',encoding='utf-8') as gf: ed = json.load(gf)
            _bi = {c['id']:c for c in all_chunks}
            try: cc.delete_collection('lao_legal')
            except: pass
            col = cc.create_collection(name='lao_legal',metadata={'hnsw:space':'cosine'})
            ids = list(ed.keys())
            for i in range(0,len(ids),100):
                bi = ids[i:i+100]
                be = [ed[c] for c in bi]
                bd,bm = [],[]
                for c in bi:
                    ch = _bi.get(c,{}); m = ch.get('metadata',{})
                    bd.append(ch.get('content',''))
                    bm.append({k:str(m.get(k,'')) if isinstance(m.get(k,''),str) else int(m.get(k,0)) if isinstance(m.get(k,0),int) else bool(m.get(k,False)) for k in ['law_name_lao','law_name_en','year','article','article_title','law_type','status','category','document_number','is_sub_chunk','parent_id','character_count']})
                col.add(ids=bi,embeddings=be,documents=bd,metadatas=bm)
    return {'gclient':gclient,'genai_types':genai_types,'collection':col,
        'bm25_index':bm25_data['index'],'bm25_ids':bm25_data['ids'],
        'all_chunks':all_chunks,'chunks_by_id':{c['id']:c for c in all_chunks},
        'article_lookup':article_lookup,'registry':registry,'law_keywords':law_keywords,'api_keys':api_keys}


def search(query, sys):
    # Layer 1: Normalize
    nq = normalize_lao(query)
    sub_queries = split_compound_query(nq)

    # Detect article
    am = re.search(r'ມາດຕາ\s*(\d+)', nq)
    article_num = int(am.group(1)) if am else None

    # Detect law
    law_filter = None
    for kw in sorted(sys['law_keywords'].keys(), key=len, reverse=True):
        if kw in nq:
            law_filter = sys['law_keywords'][kw]; break
    # Try situation keywords if no law found
    if not law_filter:
        for sit_kw, law_kw in SITUATION_KEYWORDS.items():
            if sit_kw in nq:
                if law_kw in sys['law_keywords']:
                    law_filter = sys['law_keywords'][law_kw]; break

    is_def = any(p in nq for p in DEFINITIONAL)
    results = []
    seen = set()

    # Layer 2A: Exact lookup
    if article_num:
        for ch in sys['all_chunks']:
            m = ch['metadata']
            if m['article'] == article_num:
                if not law_filter or law_filter.lower() in m['law_name_en'].lower():
                    if ch['id'] not in seen:
                        results.append({'chunk':ch,'score':1.0,'source':'exact'}); seen.add(ch['id'])

    # Layer 2B: Semantic WITH filter
    try:
        qr = sys['gclient'].models.embed_content(model=EMBED_MODEL,contents=nq,
            config=sys['genai_types'].EmbedContentConfig(task_type="RETRIEVAL_QUERY",output_dimensionality=TARGET_DIM))
        qv = normalize_vector(qr.embeddings[0].values)
        if law_filter:
            cr = sys['collection'].query(query_embeddings=[qv],n_results=10,
                where={"law_name_en":law_filter},include=["documents","metadatas","distances"])
            for i in range(len(cr['ids'][0])):
                cid = cr['ids'][0][i]
                if cid not in seen:
                    results.append({'chunk':sys['chunks_by_id'].get(cid,{}),'score':1-cr['distances'][0][i],'source':'semantic'}); seen.add(cid)

        # Layer 2C: Semantic WITHOUT filter (cross-law)
        cr2 = sys['collection'].query(query_embeddings=[qv],n_results=10,
            include=["documents","metadatas","distances"])
        for i in range(len(cr2['ids'][0])):
            cid = cr2['ids'][0][i]
            if cid not in seen:
                ch = sys['chunks_by_id'].get(cid)
                if ch:
                    results.append({'chunk':ch,'score':(1-cr2['distances'][0][i])*0.9,'source':'semantic'}); seen.add(cid)

        # Layer 2D: Sub-query search for compound queries
        if len(sub_queries) > 1:
            for sq in sub_queries:
                sqr = sys['gclient'].models.embed_content(model=EMBED_MODEL,contents=sq,
                    config=sys['genai_types'].EmbedContentConfig(task_type="RETRIEVAL_QUERY",output_dimensionality=TARGET_DIM))
                sqv = normalize_vector(sqr.embeddings[0].values)
                scr = sys['collection'].query(query_embeddings=[sqv],n_results=5,include=["documents","metadatas","distances"])
                for i in range(len(scr['ids'][0])):
                    cid = scr['ids'][0][i]
                    if cid not in seen:
                        ch = sys['chunks_by_id'].get(cid)
                        if ch:
                            results.append({'chunk':ch,'score':(1-scr['distances'][0][i])*0.85,'source':'semantic'}); seen.add(cid)
    except: pass

    # Layer 2E: BM25 keyword
    try:
        tokens = lao_tokenize(nq)
        scores = sys['bm25_index'].get_scores(tokens)
        top_idx = sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)[:10]
        for idx in top_idx:
            cid = sys['bm25_ids'][idx]
            if cid not in seen and scores[idx] > 0:
                ch = sys['chunks_by_id'].get(cid)
                if ch: results.append({'chunk':ch,'score':scores[idx],'source':'keyword'}); seen.add(cid)
    except: pass

    # Layer 2F: Definitional boost
    if is_def and law_filter:
        for ch in sys['all_chunks']:
            m = ch['metadata']
            if m.get('article',0) <= 5 and law_filter.lower() in m.get('law_name_en','').lower() and ch['id'] not in seen:
                results.append({'chunk':ch,'score':0.9-(m['article']*0.02),'source':'definitional'}); seen.add(ch['id'])

    # Merge
    priority = [r for r in results if r['source'] in ('exact','definitional')]
    semantic = sorted([r for r in results if r['source']=='semantic'],key=lambda x:-x['score'])
    keyword = sorted([r for r in results if r['source']=='keyword'],key=lambda x:-x['score'])
    merged = priority[:]
    si,ki = 0,0
    while si < len(semantic) or ki < len(keyword):
        if si < len(semantic):
            r = semantic[si]; si += 1
            if r['chunk'].get('id','') not in {m['chunk'].get('id','') for m in merged}: merged.append(r)
        if ki < len(keyword):
            r = keyword[ki]; ki += 1
            if r['chunk'].get('id','') not in {m['chunk'].get('id','') for m in merged}: merged.append(r)
    top = merged[:5]

    # Layer 3: Relevance verification
    needs_fallback = False
    if law_filter and top:
        found_law = any(law_filter.lower() in r['chunk'].get('metadata',{}).get('law_name_en','').lower() for r in top)
        if not found_law: needs_fallback = True
    if top and all(r['source']=='semantic' and isinstance(r['score'],float) and r['score'] < 0.25 for r in top[:3]):
        needs_fallback = True

    # Confidence
    if needs_fallback: confidence = 'FALLBACK'
    elif not top: confidence = 'NONE'
    elif top[0]['source'] in ('exact','definitional'): confidence = 'HIGH'
    elif top[0]['source']=='semantic' and isinstance(top[0]['score'],float):
        if top[0]['score'] > 0.65: confidence = 'HIGH'
        elif top[0]['score'] > 0.40: confidence = 'MEDIUM'
        else: confidence = 'LOW'
    elif len([r for r in top[:3] if r['source'] in ('semantic','keyword')]) >= 2: confidence = 'MEDIUM'
    elif top: confidence = 'LOW'
    else: confidence = 'NONE'

    return top, confidence, law_filter, article_num

SYSTEM_PROMPT = (
    "ເຈົ້າແມ່ນ ລະບົບຄົ້ນຄວ້າກົດໝາຍລາວ (Lao Legal Research System) ທີ່ວິເຄາະກົດໝາຍ 75 ສະບັບ.\n"
    "ເຈົ້າໃຫ້ການວິເຄາະທາງກົດໝາຍທີ່ຖືກຕ້ອງ ໂດຍອີງໃສ່ຂໍ້ມູນກົດໝາຍທີ່ສະໜອງໃຫ້.\n"
    "\n"
    "ວິທີຕອບ:\n"
    "1. ສັງເຄາະ ຂໍ້ມູນຈາກຫຼາຍມາດຕາ ເພື່ອໃຫ້ຄຳຕອບທີ່ຄົບຖ້ວນ.\n"
    "2. ອະທິບາຍ ຫຼັກການທາງກົດໝາຍ ໃນພາສາທີ່ເຂົ້າໃຈງ່າຍ.\n"
    "3. ອ້າງອີງ ມາດຕາສະເພາະ ເປັນຫຼັກຖານ.\n"
    "4. ວິເຄາະ ເງື່ອນໄຂ, ຂໍ້ຍົກເວັ້ນ, ແລະ ຜົນສະທ້ອນທາງກົດໝາຍ.\n"
    "\n"
    "ຮູບແບບ:\n"
    "📋 ຄຳຕອບ: (ສັງເຄາະ + ວິເຄາະ)\n"
    "📎 ຫຼັກຖານ: (ອ້າງ [ຊື່ກົດໝາຍ, ມາດຕາ X])\n"
    "⚠️ ຂໍ້ຄວນລະວັງ: (ເງື່ອນໄຂ, ຂໍ້ຍົກເວັ້ນ)\n"
    "\n"
    "ກົດລະບຽບ:\n"
    "- ຕອບເປັນ ພາສາລາວ ສະເໝີ.\n"
    "- ຫ້າມ ສ້າງ ເລກມາດຕາ ທີ່ບໍ່ມີໃນຂໍ້ມູນ.\n"
    "- ໃຫ້ ການວິເຄາະ ບໍ່ແມ່ນ ການຄັດລອກ ມາດຕາ.\n"
    "- ຢ່າເວົ້າ \"ຂໍ້ມູນມີຈຳກັດ\" ຫຼື \"ບໍ່ມີຂໍ້ມູນພຽງພໍ\".\n"
    "- ຖ້າບໍ່ພົບຂໍ້ມູນສະເພາະ, ບອກສັ້ນໆ ແລ້ວໃຫ້ຂໍ້ມູນທີ່ກ່ຽວຂ້ອງ.\n"
    "- ຢ່າອ້າງວ່າເປັນທະນາຍ ຫຼື ມີປະສົບການ — ເຈົ້າແມ່ນເຄື່ອງມືຄົ້ນຄວ້າ.\n"
)

HYBRID_FALLBACK_PROMPT = (
    "ເຈົ້າແມ່ນ ລະບົບຄົ້ນຄວ້າກົດໝາຍລາວ.\n"
    "\n"
    "ຄຳຖາມ: {query}\n"
    "\n"
    "ຂໍ້ມູນກົດໝາຍທີ່ຄົ້ນພົບ (ອາດບໍ່ກົງກັບຄຳຖາມໂດຍກົງ):\n"
    "{context}\n"
    "\n"
    "ຄຳແນະນຳ:\n"
    "- ຖ້າຂໍ້ມູນຂ້າງເທິງກ່ຽວຂ້ອງ, ໃຊ້ມັນ ແລະ ອ້າງອີງມາດຕາ.\n"
    "- ຖ້າຂໍ້ມູນຂ້າງເທິງບໍ່ກ່ຽວຂ້ອງ, ຢ່າອ້າງອີງມັນ. ໃຫ້ຕອບຈາກຄວາມຮູ້ທົ່ວໄປ ກ່ຽວກັບກົດໝາຍລາວ ແລະ ບອກວ່າ \"ນີ້ແມ່ນຄວາມຮູ້ທົ່ວໄປ\".\n"
    "- ຕອບໃຫ້ເປັນປະໂຫຍດ. ຢ່າເວົ້າ \"ບໍ່ພົບຂໍ້ມູນ\" ຖ້າເຈົ້າສາມາດໃຫ້ຂໍ້ມູນທົ່ວໄປໄດ້.\n"
    "\n"
    "ຮູບແບບ:\n"
    "📋 ຄຳຕອບ: (ອະທິບາຍ)\n"
    "📎 ຫຼັກຖານ: (ຖ້າມີມາດຕາທີ່ກ່ຽວຂ້ອງ)\n"
    "⚠️ ຂໍ້ຄວນລະວັງ: (ລະບຸສ່ວນໃດເປັນຄວາມຮູ້ທົ່ວໄປ)\n"
)

PURE_FALLBACK_PROMPT = (
    "ເຈົ້າແມ່ນ ລະບົບຄົ້ນຄວ້າກົດໝາຍລາວ.\n"
    "ຄຳຖາມ: {query}\n"
    "ຖານຂໍ້ມູນ 75 ກົດໝາຍ ບໍ່ມີມາດຕາສະເພາະກ່ຽວກັບຄຳຖາມນີ້.\n"
    "ກະລຸນາ:\n"
    "- ອະທິບາຍຫຼັກການທົ່ວໄປ ຕາມກົດໝາຍລາວ\n"
    "- ແນະນຳກົດໝາຍ ຫຼື ຂະແໜງການທີ່ກ່ຽວຂ້ອງ\n"
    "ຮູບແບບ:\n"
    "📋 ຄຳຕອບ: (ອະທິບາຍ)\n"
    "⚠️ ຂໍ້ຄວນລະວັງ: (ນີ້ແມ່ນຄວາມຮູ້ທົ່ວໄປ — ຄວນກວດສອບກັບແຫຼ່ງທາງການ)\n"
)


def generate_answer(query, search_results, confidence, sys):
    is_general = False

    # Layer 4: Choose prompt strategy
    if confidence == 'NONE' or not search_results:
        prompt = PURE_FALLBACK_PROMPT.replace('{query}', query)
        is_general = True
    elif confidence == 'FALLBACK':
        ctx = ""
        for i,r in enumerate(search_results[:3]):
            m = r['chunk']['metadata']
            ctx += f"\n--- ແຫຼ່ງ {i+1} ---\n[{m.get('law_name_lao','')}, ມາດຕາ {m.get('article','')}]\n{r['chunk']['content']}\n"
        prompt = HYBRID_FALLBACK_PROMPT.replace('{query}', query).replace('{context}', ctx)
        is_general = True
    else:
        srt = sorted(search_results, key=lambda r: (0 if r['chunk']['metadata'].get('article',999) <= 5 else 1, -r['score']))
        ctx = ""
        for i,r in enumerate(srt[:3]):
            m = r['chunk']['metadata']
            ctx += f"\n--- ແຫຼ່ງ {i+1} ---\n[{m.get('law_name_lao','')}, ມາດຕາ {m.get('article','')}]\nຫົວຂໍ້: {m.get('article_title','')}\n{r['chunk']['content']}\n"
        extra = "\nໝາຍເຫດ: ຜົນຄົ້ນຫາບໍ່ກົງຫຼາຍ." if confidence == 'LOW' else ""
        prompt = f"{SYSTEM_PROMPT}{extra}\n\nຂໍ້ມູນກົດໝາຍ:\n{ctx}\n\nຄຳຖາມ: {query}"

    try:
        response = sys['gclient'].models.generate_content(model=GEN_MODEL,contents=prompt,
            config=sys['genai_types'].GenerateContentConfig(max_output_tokens=4096,temperature=0.2))
        answer = response.text
    except Exception as e:
        return f"ຂໍອະໄພ, ລະບົບບໍ່ສາມາດສ້າງຄຳຕອບໄດ້. ({str(e)[:50]})", False, False

    # Layer 5: Answer quality check
    bad_phrases = ['ບໍ່ມີການກ່າວເຖິງ','ບໍ່ໄດ້ລວມເອົາ','ບໍ່ໄດ້ລະບຸ','ບໍ່ມີການອະທິບາຍ','ບໍ່ສາມາດວິເຄາະ','ບໍ່ມີມາດຕາສະເພາະ']
    bad_count = sum(1 for p in bad_phrases if p in answer)
    nq = normalize_lao(query)
    off_topic = ('ພາສີ' in nq and 'ໄປສະນີ' in answer) or ('ແຮງງານ' in query and 'ເຂື່ອນ' in answer and 'ແຮງງານ' not in answer)

    if (bad_count >= 2 or off_topic) and confidence not in ('NONE','FALLBACK'):
        # Regenerate with hybrid fallback
        ctx = ""
        for i,r in enumerate(search_results[:3]):
            m = r['chunk']['metadata']
            ctx += f"\n[{m.get('law_name_lao','')}, ມາດຕາ {m.get('article','')}]\n{r['chunk']['content']}\n"
        retry_prompt = HYBRID_FALLBACK_PROMPT.replace('{query}', query).replace('{context}', ctx)
        try:
            retry = sys['gclient'].models.generate_content(model=GEN_MODEL,contents=retry_prompt,
                config=sys['genai_types'].GenerateContentConfig(max_output_tokens=4096,temperature=0.3))
            answer = retry.text
            is_general = True
        except: pass

    if is_general and '\U0001f4a1' not in answer:
        answer += "\n\n\U0001f4a1 *ໝາຍເຫດ: ບາງສ່ວນຂອງຄຳຕອບນີ້ມາຈາກຄວາມຮູ້ທົ່ວໄປ.*"

    # Citation check
    cited = re.findall(r'ມາດຕາ\s*(\d+)', answer)
    avail = {str(r['chunk']['metadata'].get('article','')) for r in search_results} if search_results else set()
    invalid = [c for c in cited if c not in avail] if not is_general else []
    if invalid:
        answer += f"\n\n⚠️ ມາດຕາ {', '.join(invalid)} ບໍ່ສາມາດຢືນຢັນໄດ້."

    return answer, len(invalid) == 0, is_general

if 'response_cache' not in st.session_state: st.session_state.response_cache = {}

def get_cached(q):
    h = hashlib.md5(q.strip().lower().encode()).hexdigest()
    c = st.session_state.response_cache.get(h)
    if c and time.time()-c.get('time',0)<86400: return c
    return None

def set_cache(q,d):
    h = hashlib.md5(q.strip().lower().encode()).hexdigest()
    d['time']=time.time()
    st.session_state.response_cache[h]=d

st.markdown('<div class="main-title"><h1>⚖️ ຜູ້ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ</h1><p>Lao Legal Research Assistant — 75 ກົດໝາຍ · Powered by AI</p></div>', unsafe_allow_html=True)
st.markdown('<div class="disclaimer-bar">⚠️ ນີ້ແມ່ນເຄື່ອງມືຄົ້ນຄວ້າເທົ່ານັ້ນ — ບໍ່ແມ່ນຄຳປຶກສາທາງກົດໝາຍ — ກະລຸນາປຶກສາທະນາຍຄວາມ</div>', unsafe_allow_html=True)

sys = load_system()

with st.sidebar:
    st.markdown('### 📚 ກົດໝາຍທີ່ມີ')
    for law in sys['registry']['laws']:
        st.markdown(f"📜 **{law['law_name_lao']}**")
        st.caption(f"{law['law_name_en']} ({law['year']}) · {law['total_chunks']} ມາດຕາ")
    st.divider()
    st.markdown(f"📊 ລວມ: **{sys['registry']['total_laws']}** ກົດໝາຍ · **{sys['registry']['total_chunks']}** ມາດຕາ")
    st.divider()
    show_debug = st.toggle('ສະແດງ Debug', value=False)
    show_sources = st.toggle('ສະແດງແຫຼ່ງອ້າງອີງ', value=True)

st.markdown('##### 💡 ຕົວຢ່າງຄຳຖາມ:')
ecols = st.columns(2)
examples = ['ມາດຕາ 52 ກົດໝາຍວິສາຫະກິດ', 'ທຶນຈົດທະບຽນບໍລິສັດປະກັນໄພ',
            'ເງື່ອນໄຂການລົງທຶນຕ່າງປະເທດ', 'ກົດໝາຍພາສີແມ່ນຫຍັງ']
for i,ex in enumerate(examples):
    if ecols[i%2].button(ex, key=f'ex_{i}', use_container_width=True):
        st.session_state.pending_query = ex

if 'messages' not in st.session_state: st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        if msg['role'] == 'assistant':
            st.markdown(msg['content'])
            if msg.get('sources') and show_sources:
                with st.expander('📎 ແຫຼ່ງອ້າງອີງ', expanded=False):
                    for s in msg['sources']:
                        st.markdown(f'<div class="source-card">📜 <b>{s.get("law_lao","")}</b>, ມາດຕາ {s.get("article","")} — {s.get("article_title","")}</div>', unsafe_allow_html=True)
            if msg.get('confidence'):
                conf = msg['confidence']
                cls = {'HIGH':'conf-high','MEDIUM':'conf-medium','LOW':'conf-low','FALLBACK':'conf-low'}.get(conf,'')
                icon = {'HIGH':'🟢','MEDIUM':'🟡','LOW':'🔴','FALLBACK':'🟡','NONE':'⚫'}.get(conf,'⚪')
                extra = ''
                if msg.get('from_cache'): extra = '<span class="cached-tag">💨 cached</span>'
                if msg.get('is_general'): extra = '<span class="general-tag">💡 ຄວາມຮູ້ທົ່ວໄປ</span>'
                st.markdown(f'{icon} <span class="confidence-badge {cls}">{conf}</span>{extra}', unsafe_allow_html=True)
        else:
            st.markdown(msg['content'])

pending = st.session_state.pop('pending_query', None)
user_input = st.chat_input('ພິມຄຳຖາມກ່ຽວກັບກົດໝາຍລາວ...') or pending

if user_input:
    st.session_state.messages.append({'role':'user','content':user_input})
    with st.chat_message('user'): st.markdown(user_input)
    with st.chat_message('assistant'):
        if len(user_input.strip()) < 3:
            st.markdown('❌ ຄຳຖາມສັ້ນເກີນໄປ')
            st.session_state.messages.append({'role':'assistant','content':'❌ ຄຳຖາມສັ້ນເກີນໄປ'})
        else:
            cached = get_cached(user_input)
            if cached:
                answer_text=cached['answer']; sources=cached.get('sources',[])
                confidence=cached.get('confidence','MEDIUM'); from_cache=True
                citations_ok=cached.get('citations_ok',True); is_general=cached.get('is_general',False)
            else:
                with st.spinner('🔍 ກຳລັງຄົ້ນຫາ...'):
                    t0=time.time()
                    search_results,confidence,law_filter,article_num = search(user_input,sys)
                    t_search=time.time()-t0
                with st.spinner('🤖 ກຳລັງສ້າງຄຳຕອບ...'):
                    t1=time.time()
                    answer_text,citations_ok,is_general = generate_answer(user_input,search_results,confidence,sys)
                    t_gen=time.time()-t1
                sources=[]
                if not is_general and search_results:
                    for r in search_results[:3]:
                        m=r['chunk']['metadata']
                        sources.append({'law_lao':m.get('law_name_lao',''),'law_en':m.get('law_name_en',''),
                            'article':m.get('article',0),'article_title':m.get('article_title','')})
                from_cache=False
                set_cache(user_input,{'answer':answer_text,'sources':sources,'confidence':confidence,
                    'citations_ok':citations_ok,'is_general':is_general})
            st.markdown(answer_text)
            if sources and show_sources:
                with st.expander('📎 ແຫຼ່ງອ້າງອີງ', expanded=False):
                    for s in sources:
                        st.markdown(f'<div class="source-card">📜 <b>{s.get("law_lao","")}</b>, ມາດຕາ {s.get("article","")} — {s.get("article_title","")}</div>', unsafe_allow_html=True)
            conf_cls={'HIGH':'conf-high','MEDIUM':'conf-medium','LOW':'conf-low','FALLBACK':'conf-low'}.get(confidence,'')
            conf_icon={'HIGH':'🟢','MEDIUM':'🟡','LOW':'🔴','FALLBACK':'🟡','NONE':'⚫'}.get(confidence,'⚪')
            extra=''
            if from_cache: extra='<span class="cached-tag">💨 cached</span>'
            if is_general: extra='<span class="general-tag">💡 ຄວາມຮູ້ທົ່ວໄປ</span>'
            st.markdown(f'{conf_icon} <span class="confidence-badge {conf_cls}">{confidence}</span>{extra}',unsafe_allow_html=True)
            st.caption('⚠️ ນີ້ແມ່ນເຄື່ອງມືຄົ້ນຄວ້າເທົ່ານັ້ນ — ບໍ່ແມ່ນຄຳປຶກສາທາງກົດໝາຍ — ກະລຸນາປຶກສາທະນາຍຄວາມ')
            if show_debug and not from_cache:
                with st.expander('🔧 Debug',expanded=False):
                    st.json({'search_time':f'{t_search:.2f}s','gen_time':f'{t_gen:.2f}s',
                        'confidence':confidence,'citations_ok':citations_ok,
                        'is_general':is_general,'law_detected':law_filter,
                        'article_num':article_num,'results':len(search_results) if search_results else 0})
            col1,col2,_=st.columns([1,1,4])
            with col1: st.button('👍',key=f'up_{len(st.session_state.messages)}')
            with col2: st.button('👎',key=f'dn_{len(st.session_state.messages)}')
            st.session_state.messages.append({'role':'assistant','content':answer_text,
                'sources':sources,'confidence':confidence,'from_cache':from_cache,'is_general':is_general})
