"""
app.py — Lao Legal RAG UI (clean, styled, no duplication)
=========================================================
Edit ONLY this file. Backend is one call: ask(question) -> result.
result.answer / .citations / .confidence / .is_general / .error
"""
import re
import streamlit as st
from backend import ask, get_system_info

st.set_page_config(page_title="ຜູ້ຊ່ວຍກົດໝາຍລາວ", page_icon="⚖️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;500;600;700&family=Noto+Serif+Lao:wght@600;700&display=swap');
:root{--ink:#1a1f36;--paper:#faf8f3;--surface:#fff;--accent:#8b1e3f;--accent2:#1a5e63;--muted:#6b7280;--line:#e8e4da;}
.stApp{background:var(--paper);}
*{font-family:'Noto Sans Lao',sans-serif;}
.block-container{max-width:740px;padding-top:1.2rem;}
.app-head{text-align:center;padding:.3rem 0 .1rem;}
.app-head .mark{font-size:1.8rem;}
.app-head h1{font-family:'Noto Serif Lao',serif;font-size:1.55rem;font-weight:700;color:var(--ink);margin:.1rem 0;}
.app-head .sub{color:var(--muted);font-size:.78rem;}
.app-head .rule{width:44px;height:3px;background:var(--accent);margin:.5rem auto 0;border-radius:2px;}
.disc{text-align:center;font-size:.72rem;color:var(--muted);margin:.9rem auto;max-width:560px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:.5rem 0;}
.disc b{color:var(--accent);}
.ans-section{margin:.6rem 0;line-height:1.75;}
.ans-head{font-family:'Noto Serif Lao',serif;font-weight:700;font-size:.95rem;color:var(--accent2);margin:.9rem 0 .35rem;display:flex;align-items:center;gap:.4rem;}
.ans-head.warn{color:var(--accent);}
.cite{display:flex;gap:.7rem;padding:.5rem .65rem;border:1px solid var(--line);border-radius:9px;margin-bottom:.4rem;background:var(--surface);}
.cite .art{font-family:'Noto Serif Lao',serif;font-weight:700;color:var(--accent);font-size:.78rem;white-space:nowrap;border-right:2px solid var(--line);padding-right:.7rem;display:flex;align-items:center;}
.cite .body .law{font-weight:600;color:var(--ink);font-size:.82rem;}
.cite .body .title{color:var(--muted);font-size:.75rem;}
.conf{display:inline-flex;align-items:center;gap:.4rem;font-size:.72rem;font-weight:600;padding:.25rem .6rem;border-radius:8px;margin-top:.5rem;}
.conf .dot{width:7px;height:7px;border-radius:50%;}
.conf.hi{background:rgba(26,94,99,.1);color:var(--accent2);} .conf.hi .dot{background:var(--accent2);}
.conf.med{background:rgba(184,134,11,.12);color:#b8860b;} .conf.med .dot{background:#b8860b;}
.conf.lo{background:rgba(154,59,59,.1);color:#9a3b3b;} .conf.lo .dot{background:#9a3b3b;}
.stButton button{border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:10px;font-size:.8rem;}
.stButton button:hover{border-color:var(--accent);color:var(--accent);}
</style>
""", unsafe_allow_html=True)


def render_answer(text: str):
    """Render the answer with its sections styled as headed blocks.
    The backend answer contains 📋/📎/⚠️/💡 markers. We split on them and
    render each as a clean section. We STRIP the in-text 📎 ຫຼັກຖານ block's
    raw citation lines because the structured citation cards show those —
    this removes the duplication.
    """
    if not text:
        st.markdown("—"); return

    # split into (marker, body) sections
    parts = re.split(r'(📋 ຄຳຕອບ:|📎 ຫຼັກຖານ:|⚠️ ຂໍ້ຄວນລະວັງ:|💡)', text)
    # parts[0] is any preamble before the first marker
    preamble = parts[0].strip()
    if preamble:
        st.markdown(f'<div class="ans-section">{preamble}</div>', unsafe_allow_html=True)

    i = 1
    while i < len(parts):
        marker = parts[i].strip()
        body = parts[i+1].strip() if i+1 < len(parts) else ""
        i += 2
        if marker.startswith("📋"):
            st.markdown('<div class="ans-head">📋 ຄຳຕອບ</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ans-section">{body}</div>', unsafe_allow_html=True)
        elif marker.startswith("📎"):
            # We SKIP rendering the raw evidence prose here — the citation cards
            # below the answer present the sources cleanly without duplication.
            continue
        elif marker.startswith("⚠️"):
            st.markdown('<div class="ans-head warn">⚠️ ຂໍ້ຄວນລະວັງ</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ans-section">{body}</div>', unsafe_allow_html=True)
        elif marker.startswith("💡"):
            st.caption(f"💡 {body}")


def render_citations(citations):
    if not citations:
        return
    # dedupe by (law, article)
    seen = set(); uniq = []
    for c in citations:
        key = (c.get("law",""), str(c.get("article","")))
        if key not in seen and c.get("law"):
            seen.add(key); uniq.append(c)
    if not uniq:
        return
    st.markdown('<div class="ans-head">📎 ແຫຼ່ງອ້າງອີງ</div>', unsafe_allow_html=True)
    for c in uniq:
        st.markdown(
            f'<div class="cite"><div class="art">ມາດຕາ {c["article"]}</div>'
            f'<div class="body"><div class="law">{c["law"]}</div>'
            + (f'<div class="title">{c["title"]}</div>' if c.get("title") else "")
            + '</div></div>', unsafe_allow_html=True)


def render_conf(confidence, is_general):
    if is_general:
        st.markdown('<div class="conf med"><span class="dot"></span>ຄວາມຮູ້ທົ່ວໄປ</div>',
                    unsafe_allow_html=True); return
    label = {"HIGH":"ຄວາມໝັ້ນໃຈສູງ","MEDIUM":"ຄວາມໝັ້ນໃຈປານກາງ",
             "LOW":"ຄວາມໝັ້ນໃຈຕ່ຳ","NONE":"ບໍ່ພົບໃນຖານຂໍ້ມູນ"}.get(confidence,"")
    cls = {"HIGH":"hi","MEDIUM":"med","LOW":"lo","NONE":"lo"}.get(confidence,"med")
    if label:
        st.markdown(f'<div class="conf {cls}"><span class="dot"></span>{label}</div>',
                    unsafe_allow_html=True)


# ---------- header ----------
st.markdown('<div class="app-head"><div class="mark">⚖️</div>'
            '<h1>ຜູ້ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ</h1>'
            '<div class="sub">Lao Legal Research Assistant</div>'
            '<div class="rule"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="disc">⚠️ ນີ້ແມ່ນ<b>ເຄື່ອງມືຄົ້ນຄວ້າ</b> — ບໍ່ແມ່ນຄຳປຶກສາທາງກົດໝາຍ — '
            'ກະລຸນາປຶກສາທະນາຍຄວາມ</div>', unsafe_allow_html=True)

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### ⚖️ ກົດໝາຍລາວ")
    if st.button("＋ ສົນທະນາໃໝ່", use_container_width=True):
        st.session_state.messages = []; st.rerun()
    info = get_system_info()
    st.divider()
    st.caption(f"📚 {info.get('total_laws','?')} ກົດໝາຍ · {info.get('total_chunks','?')} ມາດຕາ")
    if not info.get("ready"):
        st.error("ລະບົບຍັງບໍ່ພ້ອມ — ກວດ API keys")

# ---------- history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown('<div style="font-size:.8rem;font-weight:600;margin:1rem 0 .5rem;color:var(--muted);">'
                '💡 ຕົວຢ່າງຄຳຖາມ</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    examples = ['ມາດຕາ 52 ກົດໝາຍວິສາຫະກິດ','ອັດຕາອາກອນມູນຄ່າເພີ່ມ',
                'ເງື່ອນໄຂການລົງທຶນຕ່າງປະເທດ','ທຶນຈົດທະບຽນບໍລິສັດປະກັນໄພ']
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="⚖️" if msg["role"]=="assistant" else "🧑"):
        if msg["role"] == "assistant":
            render_answer(msg["content"])
            render_citations(msg.get("citations", []))
            render_conf(msg.get("confidence","NONE"), msg.get("is_general", False))
        else:
            st.markdown(msg["content"])

# ---------- input ----------
pending = st.session_state.pop("pending", None)
question = st.chat_input("ພິມຄຳຖາມກ່ຽວກັບກົດໝາຍລາວ...") or pending

if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("🔍 ກຳລັງຄົ້ນຫາ ແລະ ວິເຄາະ..."):
            result = ask(question)
        render_answer(result.answer)
        render_citations(result.citations)
        render_conf(result.confidence, result.is_general)
        st.caption("⚠️ ກວດສອບກັບແຫຼ່ງທາງການ ຫຼື ທະນາຍຄວາມສະເໝີ")
    st.session_state.messages.append({
        "role":"assistant","content":result.answer,
        "citations":result.citations,"confidence":result.confidence,
        "is_general":result.is_general,
    })
    st.rerun()
