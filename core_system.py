"""
core_system.py — Lao Legal RAG backend (extracted from app.py).
================================================================
Contains all retrieval + generation logic: load_system, search,
generate_answer, and helpers. NO UI code. The UI (app.py) talks to this
only through backend.py. Do NOT edit for UI work.

Keys are read from GEMINI_KEYS (comma-separated) / numbered / single.
"""
import json, os, re, math, time, pickle, hashlib, random, gzip
from pathlib import Path

# load_system uses st.secrets for keys + st.cache_resource for caching.
# We import streamlit but use ONLY secrets/cache here — no UI rendering.
import streamlit as st

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
TARGET_DIM = 768

import time as _time
import itertools as _itertools

class RotatingGeminiClient:
    """Wraps multiple google.genai clients (one per key) and retries transient
    errors (503/429/500) by backing off and rotating to the next key. This
    spreads load across all keys and survives temporary Gemini outages."""
    def __init__(self, api_keys, genai_module):
        from google import genai
        self._clients = [genai.Client(api_key=k) for k in api_keys]
        self._cycle = _itertools.cycle(range(len(self._clients)))
        self._n = len(self._clients)

    @property
    def models(self):
        # return a proxy that routes .embed_content / .generate_content
        return _ModelProxy(self)

    def _call(self, method_name, **kwargs):
        transient = ("503", "429", "500", "502", "504", "UNAVAILABLE",
                     "RESOURCE_EXHAUSTED", "overload", "high demand", "quota",
                     "rate limit")
        last_err = None
        # try up to 2 full passes over all keys
        max_attempts = max(self._n * 2, 6)
        for attempt in range(max_attempts):
            idx = next(self._cycle)
            client = self._clients[idx]
            try:
                method = getattr(client.models, method_name)
                return method(**kwargs)
            except Exception as e:
                last_err = e
                msg = str(e)
                is_transient = any(t in msg for t in transient)
                if not is_transient:
                    raise  # a real error (bad request etc.) — don't retry
                # transient: brief backoff, then next key
                _time.sleep(min(0.5 * (attempt + 1), 3.0))
                continue
        raise last_err if last_err else RuntimeError("all retries failed")


class _ModelProxy:
    def __init__(self, rc): self._rc = rc
    def embed_content(self, **kwargs):
        return self._rc._call("embed_content", **kwargs)
    def generate_content(self, **kwargs):
        return self._rc._call("generate_content", **kwargs)

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

    api_keys = []
    # 1. Check os.environ FIRST (works in Colab AND Streamlit; never throws).
    bulk_env = os.environ.get('GEMINI_KEYS', '')
    if bulk_env:
        api_keys.extend([k.strip() for k in bulk_env.split(',') if k.strip()])
    # 2. Then Streamlit secrets (each access in its OWN try so one failure
    #    doesn't abort the rest). st.secrets raises outside a Streamlit session.
    if not api_keys:
        try:
            bulk = st.secrets.get('GEMINI_KEYS', '')
            if bulk:
                api_keys.extend([k.strip() for k in bulk.split(',') if k.strip()])
        except Exception:
            pass
    if not api_keys:
        for i in range(1, 21):
            try:
                key = st.secrets.get(f'GEMINI_KEY_{i}', '')
                if key: api_keys.append(key.strip())
            except Exception:
                break
    # 3. Single-key fallbacks
    if not api_keys:
        single = os.environ.get('GEMINI_API_KEY', '')
        if single:
            api_keys.append(single.strip())
    if not api_keys:
        try:
            single = st.secrets.get('GEMINI_API_KEY', '')
            if single: api_keys.append(single.strip())
        except Exception:
            pass
    if not api_keys:
        raise RuntimeError("No API keys found. Set GEMINI_KEYS or GEMINI_KEY_1..n.")

    # Rotating client across ALL keys, with 503/429 retry (spreads load,
    # survives transient Gemini outages).
    gclient = RotatingGeminiClient(api_keys, genai)

    # ---- Load FULL content (both collections) + embeddings from the new cache ----
    with gzip.open(f'{base}/db/content_full.json.gz', 'rt', encoding='utf-8') as cf:
        content_full = json.load(cf)   # {collection_name: {id: {content, metadata}}}
    with gzip.open(f'{base}/db/embeddings_cache_full.json.gz', 'rt', encoding='utf-8') as ef:
        embed_data = json.load(ef)     # {id: vector}

    # Flatten all chunks (both collections) into one list for BM25 + lookup
    all_chunks = []
    chunks_by_id = {}
    for coll_name, items in content_full.items():
        for cid, rec in items.items():
            chunk = {'id': cid, 'content': rec['content'],
                     'metadata': rec['metadata'], 'collection': coll_name}
            all_chunks.append(chunk)
            chunks_by_id[cid] = chunk

    # Build law keyword map from ALL law names. Map BOTH to law_name_en (when
    # present) AND keep the Lao name itself as the canonical key, so matching
    # never depends on law_name_en being populated. Every law becomes matchable
    # by the distinctive part of its own Lao name.
    law_keywords = {}          # keyword (Lao) -> canonical law id (Lao name)
    law_canonical = {}         # Lao name -> Lao name (identity, for lookup)
    for chunk in all_chunks:
        meta = chunk['metadata']
        lao_name = meta.get('law_name_lao', '')
        if not lao_name:
            continue
        law_canonical[lao_name] = lao_name
        # full name maps to itself
        law_keywords[lao_name] = lao_name
        # distinctive part: drop the common prefix "ກົດໝາຍວ່າດ້ວຍ"
        short = lao_name.replace('ກົດໝາຍວ່າດ້ວຍ', '').replace('ກົດໝາຍ', '').strip()
        if len(short) > 3:
            law_keywords[short] = lao_name

    # Load the PREBUILT BM25 index (fast startup; built by export_full_data.py)
    with open(f'{base}/db/bm25/bm25_index_full.pkl', 'rb') as bf:
        _bm25 = pickle.load(bf)
    bm25_index = _bm25['index']
    bm25_ids = _bm25['ids']

    # Registry (for sidebar counts)
    try:
        with open(f'{base}/laws/registry.json', 'r', encoding='utf-8') as rf:
            registry = json.load(rf)
    except Exception:
        registry = {'laws': [], 'total_laws': '?', 'total_chunks': len(all_chunks)}
    registry['total_chunks'] = len(all_chunks)

    # ---- Build BOTH ChromaDB collections from the cache ----
    chroma_path = f'{base}/db/chroma_runtime'
    os.makedirs(chroma_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    collections = {}
    for coll_name, items in content_full.items():
        try:
            c = chroma_client.get_collection(coll_name)
            if c.count() >= len(items) * 0.9:
                collections[coll_name] = c
                continue
        except: pass
        try: chroma_client.delete_collection(coll_name)
        except: pass
        c = chroma_client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
        ids_list = [cid for cid in items.keys() if cid in embed_data]
        for i in range(0, len(ids_list), 100):
            bids = ids_list[i:i+100]
            c.add(
                ids=bids,
                embeddings=[embed_data[cid] for cid in bids],
                documents=[items[cid]['content'] for cid in bids],
                metadatas=[_clean_meta(items[cid]['metadata']) for cid in bids],
            )
        collections[coll_name] = c

    return {
        'gclient': gclient, 'genai_types': genai_types,
        'collections': collections,
        'laws_collection': collections.get('lao_legal'),
        'regs_collection': collections.get('lao_regulations_tax'),
        'bm25_index': bm25_index, 'bm25_ids': bm25_ids,
        'all_chunks': all_chunks, 'chunks_by_id': chunks_by_id,
        'registry': registry, 'law_keywords': law_keywords,
        'api_keys': api_keys,
    }


def _clean_meta(meta):
    """Coerce metadata to Chroma-safe scalar types (str/int/float/bool)."""
    out = {}
    for k, v in meta.items():
        if v is None: out[k] = ''
        elif isinstance(v, (str, int, float, bool)): out[k] = v
        else: out[k] = str(v)
    return out

    for kw, hint in short_kw.items():
        if kw not in law_keywords:
            for full_lao, en in law_keywords.items():
                if hint.lower() in en.lower():
                    law_keywords[kw] = en; break

    with open(f'{base}/db/article_lookup.json', 'r', encoding='utf-8') as af:
        article_lookup = json.load(af)

    with open(f'{base}/laws/registry.json', 'r', encoding='utf-8') as rf:
        registry = json.load(rf)

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


def _normalize_lao(text):
    """Normalize Lao text for fuzzy matching: drop tone marks and common
    spelling-variant marks so 'ທີດິນ' matches 'ທີ່ດິນ', 'ພາສິ' matches 'ພາສີ'.
    These combining marks are frequently omitted/varied in user queries."""
    # Lao tone marks + some vowel-length marks that users drop or swap
    drop = {
        '\u0ec8',  # mai ek  ່
        '\u0ec9',  # mai tho ້
        '\u0eca',  # mai ti  ໊
        '\u0ecb',  # mai catawa ໋
    }
    out = ''.join(c for c in text if c not in drop)
    # normalize ສ/ສ and common i/ii swaps that break matching
    out = out.replace('\u0eb5', '\u0eb4')  # ີ (sara ii) -> ິ (sara i)
    return out


DEFINITIONAL_PATTERNS = ['ແມ່ນຫຍັງ', 'ຄືແນວໃດ', 'ໝາຍຄວາມວ່າ',
                          'ຈຸດປະສົງ', 'ຄຳນິຍາມ', 'ຂອບເຂດ', 'ອະທິບາຍ']

def search(query, sys):
    article_match = re.search(r'ມາດຕາ\s*(\d+)', query)
    article_num = int(article_match.group(1)) if article_match else None

    # Detect a named law via tone-mark-normalized keyword matching.
    # law_filter is now a canonical LAO law name (or None).
    law_filter = None
    _nq = _normalize_lao(query)
    for kw in sorted(sys['law_keywords'].keys(), key=len, reverse=True):
        if kw in query or _normalize_lao(kw) in _nq:
            law_filter = sys['law_keywords'][kw]
            break

    # Did the user clearly NAME a law? (mentions ກົດໝາຍ or ປະມວນກົດໝາຍ)
    user_named_law = ('ກົດໝາຍ' in query or 'ປະມວນ' in query)

    results = []
    seen_ids = set()

    def _law_matches(meta):
        """True if this chunk belongs to the law the user asked for."""
        if not law_filter:
            return True
        name = meta.get('law_name_lao', '') or meta.get('parent_law_name_lao', '')
        return _normalize_lao(law_filter) in _normalize_lao(name) or \
               _normalize_lao(name) in _normalize_lao(law_filter)

    # ---- Exact article lookup ----
    if article_num:
        for chunk in sys['all_chunks']:
            meta = chunk['metadata']
            try:
                a = int(meta.get('article', 0) or 0)
            except: a = 0
            if a == article_num and _law_matches(meta):
                if chunk['id'] not in seen_ids:
                    results.append({'chunk': chunk, 'score': 1.0, 'source': 'exact'})
                    seen_ids.add(chunk['id'])

    # If the user named a specific law + article, but we found NO exact match
    # in that law, do NOT fall back to other laws' articles. Signal not-found.
    law_named_but_missing = (
        user_named_law and law_filter and article_num and len(results) == 0
    )

    # ---- Semantic search across both collections ----
    try:
        q_result = sys['gclient'].models.embed_content(
            model=EMBED_MODEL, contents=query,
            config=sys['genai_types'].EmbedContentConfig(
                task_type="RETRIEVAL_QUERY", output_dimensionality=TARGET_DIM))
        q_vec = normalize_vector(q_result.embeddings[0].values)
        for coll_name, coll in sys['collections'].items():
            if coll is None: continue
            cr = coll.query(query_embeddings=[q_vec], n_results=10,
                            include=["documents", "metadatas", "distances"])
            for i in range(len(cr['ids'][0])):
                cid = cr['ids'][0][i]
                meta = cr['metadatas'][0][i] or {}
                if str(meta.get('quarantine', '')).lower() in ('true', '1'):
                    continue
                # if user named a law, prefer that law's chunks for semantic too
                if law_filter and user_named_law and not _law_matches(meta):
                    continue
                if cid not in seen_ids:
                    sim = 1 - cr['distances'][0][i]
                    chunk = sys['chunks_by_id'].get(cid)
                    if chunk:
                        results.append({'chunk': chunk, 'score': sim, 'source': 'semantic'})
                        seen_ids.add(cid)
    except: pass

    # If named-law filter removed everything semantic too, relax (so we can still
    # offer "closest thing" with a disclaimer) — but only if exact also missed.
    if user_named_law and law_filter and len(results) == 0:
        try:
            for coll_name, coll in sys['collections'].items():
                if coll is None: continue
                cr = coll.query(query_embeddings=[q_vec], n_results=5,
                                include=["documents", "metadatas", "distances"])
                for i in range(len(cr['ids'][0])):
                    cid = cr['ids'][0][i]; meta = cr['metadatas'][0][i] or {}
                    if str(meta.get('quarantine','')).lower() in ('true','1'): continue
                    if cid not in seen_ids:
                        chunk = sys['chunks_by_id'].get(cid)
                        if chunk:
                            results.append({'chunk': chunk, 'score': (1-cr['distances'][0][i])*0.5,
                                            'source': 'fallback'})
                            seen_ids.add(cid)
        except: pass

    # ---- BM25 keyword search ----
    try:
        tokens = lao_tokenize(query)
        scores = sys['bm25_index'].get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]
        for idx in top_idx:
            cid = sys['bm25_ids'][idx]
            if cid not in seen_ids and scores[idx] > 0:
                chunk = sys['chunks_by_id'].get(cid)
                if chunk:
                    meta = chunk['metadata']
                    if str(meta.get('quarantine','')).lower() in ('true','1'):
                        continue
                    if law_filter and user_named_law and not _law_matches(meta):
                        continue
                    results.append({'chunk': chunk, 'score': scores[idx], 'source': 'keyword'})
                    seen_ids.add(cid)
    except: pass

    # ---- Definitional boost (only within the named law if any) ----
    is_definitional = any(p in query for p in DEFINITIONAL_PATTERNS)
    if is_definitional:
        for chunk in sys['all_chunks']:
            meta = chunk['metadata']
            try: a = int(meta.get('article', 0) or 0)
            except: a = 0
            if a <= 5 and a > 0 and _law_matches(meta) and chunk['id'] not in seen_ids:
                if not law_filter:
                    continue  # don't boost random laws' early articles
                results.append({'chunk': chunk, 'score': 0.9 - (a*0.02), 'source': 'definitional'})
                seen_ids.add(chunk['id'])

    exact = [r for r in results if r['source'] in ('exact', 'definitional')]
    semantic = sorted([r for r in results if r['source'] == 'semantic'], key=lambda x: -x['score'])
    keyword = sorted([r for r in results if r['source'] == 'keyword'], key=lambda x: -x['score'])
    fallback = sorted([r for r in results if r['source'] == 'fallback'], key=lambda x: -x['score'])

    merged = exact[:]
    for s, k in zip(semantic, keyword):
        if s['chunk']['id'] not in {r['chunk']['id'] for r in merged}: merged.append(s)
        if k['chunk']['id'] not in {r['chunk']['id'] for r in merged}: merged.append(k)
    for r in semantic + keyword + fallback:
        if r['chunk']['id'] not in {m['chunk']['id'] for m in merged}: merged.append(r)

    top = merged[:5]

    # Confidence
    if law_named_but_missing and all(r['source'] == 'fallback' for r in top):
        confidence = 'NONE'   # named law's article not found; only loose fallback
    elif top and top[0]['source'] in ('exact', 'definitional'):
        confidence = 'HIGH'
    elif top and top[0]['source'] == 'semantic' and isinstance(top[0]['score'], float):
        if top[0]['score'] > 0.65: confidence = 'HIGH'
        elif top[0]['score'] > 0.40: confidence = 'MEDIUM'
        else: confidence = 'LOW'
    elif top and top[0]['source'] == 'fallback':
        confidence = 'LOW'
    elif top: confidence = 'MEDIUM'
    else: confidence = 'NONE'

    return top, confidence, law_filter, article_num


SYSTEM_PROMPT = """ເຈົ້າເປັນຜູ້ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ. ຕອບຄຳຖາມໂດຍກົງ ດ້ວຍຂໍ້ມູນຈາກມາດຕາທີ່ໃຫ້ມາ.

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
- ເລີ່ມຕອບດ້ວຍ "📋 ຄຳຕອບ:" ໂດຍກົງ — ຫ້າມແນະນຳຕົນເອງ, ຫ້າມເວົ້າເຖິງປະສົບການ ຫຼື ຄຳນຳໃດໆ.
- ຕອບເປັນ ພາສາລາວ ສະເໝີ.
- ຫ້າມ ສ້າງ ເລກມາດຕາ ທີ່ບໍ່ມີໃນຂໍ້ມູນ.
- ໃຫ້ ການວິເຄາະ ບໍ່ແມ່ນ ການຄັດລອກ ມາດຕາ.
- ຢ່າເວົ້າ "ຂໍ້ມູນມີຈຳກັດ" — ສັງເຄາະຈາກສິ່ງທີ່ມີ."""

FALLBACK_PROMPT = """ເຈົ້າເປັນຜູ້ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ. ຕອບໂດຍກົງ — ຫ້າມແນະນຳຕົນເອງ ຫຼື ເວົ້າເຖິງປະສົບການ.
ຄຳຖາມ: {query}
ຖານຂໍ້ມູນ 75 ກົດໝາຍ ບໍ່ມີມາດຕາສະເພາະ, ແຕ່ກະລຸນາ:
- ອະທິບາຍຫຼັກການທົ່ວໄປ ຕາມກົດໝາຍລາວ
- ແນະນຳກົດໝາຍ ຫຼື ຂະແໜງການທີ່ກ່ຽວຂ້ອງ
ຮູບແບບ:
📋 ຄຳຕອບ: (ອະທິບາຍ)
⚠️ ຂໍ້ຄວນລະວັງ: (ນີ້ແມ່ນຄວາມຮູ້ທົ່ວໄປ)"""


def generate_answer(query, search_results, confidence, sys):
    n_laws = sys.get('registry', {}).get('total_laws', '')
    n_txt = f"{n_laws} " if n_laws else ""

    # Case A: nothing found at all -> general knowledge, clearly flagged.
    if not search_results:
        try:
            prompt = FALLBACK_PROMPT.replace('{query}', query)
            response = sys['gclient'].models.generate_content(
                model=GEN_MODEL, contents=prompt,
                config=sys['genai_types'].GenerateContentConfig(
                    max_output_tokens=4096, temperature=0.3))
            answer = response.text
            answer += f"\n\n💡 *ໝາຍເຫດ: ຄຳຕອບນີ້ຈາກຄວາມຮູ້ທົ່ວໄປ — ບໍ່ພົບໃນຖານຂໍ້ມູນກົດໝາຍ.*"
            return answer, True, True
        except Exception as e:
            return f"ຂໍອະໄພ, ລະບົບບໍ່ສາມາດສ້າງຄຳຕອບໄດ້. ({str(e)[:50]})", False, False

    # Case B: results are only loose 'fallback' matches (the specific law/article
    # the user named was NOT found). Answer honestly: say it wasn't found, then
    # offer the closest related material WITH a clear disclaimer.
    only_fallback = all(r['source'] == 'fallback' for r in search_results)
    if confidence == 'NONE' or only_fallback:
        context = ""
        for i, r in enumerate(search_results[:3]):
            meta = r['chunk']['metadata']
            context += (f"\n--- ແຫຼ່ງ {i+1} ---\n"
                        f"[{meta.get('law_name_lao','')}, ມາດຕາ {meta.get('article','')}]\n"
                        f"{r['chunk']['content'][:500]}\n")
        prompt = (f"{SYSTEM_PROMPT}\n\n"
                  f"ສຳຄັນ: ບໍ່ພົບກົດໝາຍ ຫຼື ມາດຕາ ທີ່ຜູ້ໃຊ້ຖາມ ໂດຍກົງ ໃນຖານຂໍ້ມູນ. "
                  f"ໃຫ້ບອກຢ່າງຊື່ສັດວ່າ ບໍ່ພົບສິ່ງທີ່ຖາມໂດຍກົງ, ຈາກນັ້ນ ນຳສະເໜີ "
                  f"ຂໍ້ມູນທີ່ກ່ຽວຂ້ອງທີ່ສຸດ ທີ່ພົບ ພ້ອມລະບຸວ່າ ມັນອາດບໍ່ກົງກັບຄຳຖາມ.\n\n"
                  f"ຂໍ້ມູນທີ່ກ່ຽວຂ້ອງທີ່ສຸດ:\n{context}\n\nຄຳຖາມ: {query}")
        try:
            response = sys['gclient'].models.generate_content(
                model=GEN_MODEL, contents=prompt,
                config=sys['genai_types'].GenerateContentConfig(
                    max_output_tokens=4096, temperature=0.3))
            answer = response.text
            answer += ("\n\n⚠️ *ໝາຍເຫດ: ບໍ່ພົບກົດໝາຍ/ມາດຕາ ທີ່ທ່ານຖາມ ໂດຍກົງ "
                       "ໃນຖານຂໍ້ມູນ. ຂໍ້ມູນຂ້າງເທິງ ແມ່ນສິ່ງທີ່ກ່ຽວຂ້ອງທີ່ສຸດ "
                       "ເຊິ່ງອາດບໍ່ກົງກັບຄຳຖາມ — ກະລຸນາກວດສອບກັບແຫຼ່ງທາງການ.*")
            return answer, True, False
        except Exception as e:
            return f"ຂໍອະໄພ, ລະບົບບໍ່ສາມາດສ້າງຄຳຕອບໄດ້. ({str(e)[:50]})", False, False

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
