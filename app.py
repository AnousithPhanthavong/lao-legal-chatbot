"""
app.py — Lao Legal RAG UI
=========================
Edit ONLY this file. Backend is one call: ask(question) -> result.
result.answer / .citations / .confidence / .is_general / .error
Citations include .preview (article text) for clickable references.
"""
import re
import streamlit as st
from backend import ask, get_system_info

st.set_page_config(page_title="ຜູ້ຊ່ວຍກົດໝາຍລາວ", page_icon="⚖️", layout="centered",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;500;600;700&family=Noto+Serif+Lao:wght@600;700&display=swap');
:root{--ink:#1a1f36;--paper:#faf8f3;--surface:#fff;--accent:#8b1e3f;--accent2:#1a5e63;--muted:#6b7280;--line:#e8e4da;}
.stApp{background:var(--paper);}
*{font-family:'Noto Sans Lao',sans-serif;}
body, .stApp, p, span, div, li{color:#1a1f36;}
.block-container{max-width:740px;padding-top:1.2rem;}
.app-head{text-align:center;padding:.3rem 0 .1rem;}
.app-head .mark{font-size:1.8rem;}
.app-head h1{font-family:'Noto Serif Lao',serif;font-size:1.55rem;font-weight:700;color:var(--ink);margin:.1rem 0;}
.app-head .sub{color:var(--muted);font-size:.78rem;}
.app-head .rule{width:44px;height:3px;background:var(--accent);margin:.5rem auto 0;border-radius:2px;}
.disc{text-align:center;font-size:.72rem;color:var(--muted);margin:.9rem auto;max-width:560px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:.5rem 0;}
.disc b{color:var(--accent);}
.ans-section{margin:.6rem 0;line-height:1.85;color:#1a1f36 !important;font-size:.95rem;}
.ans-section *{color:#1a1f36 !important;}
.stChatMessage, .stChatMessage p, .stMarkdown, .stMarkdown p{color:#1a1f36 !important;}
.ans-head{font-family:'Noto Serif Lao',serif;font-weight:700;font-size:.95rem;color:var(--accent2);margin:.9rem 0 .35rem;display:flex;align-items:center;gap:.4rem;}
.ans-head.warn{color:var(--accent);}
.cite-law{font-weight:600;color:var(--ink);font-size:.84rem;}
.cite-art{font-family:'Noto Serif Lao',serif;font-weight:700;color:var(--accent);font-size:.8rem;}
.cite-preview{color:#374151 !important;font-size:.82rem;line-height:1.6;background:var(--paper);border-left:3px solid var(--accent2);padding:.5rem .7rem;border-radius:0 6px 6px 0;margin-top:.3rem;}
.conf{display:inline-flex;align-items:center;gap:.4rem;font-size:.72rem;font-weight:600;padding:.25rem .6rem;border-radius:8px;margin-top:.5rem;}
.conf .dot{width:7px;height:7px;border-radius:50%;}
.conf.hi{background:rgba(26,94,99,.1);color:var(--accent2);} .conf.hi .dot{background:var(--accent2);}
.conf.med{background:rgba(184,134,11,.12);color:#b8860b;} .conf.med .dot{background:#b8860b;}
.conf.lo{background:rgba(154,59,59,.1);color:#9a3b3b;} .conf.lo .dot{background:#9a3b3b;}

/* ---- SIDEBAR (FIX 1: clear light text on dark bg) ---- */
section[data-testid="stSidebar"]{background:#1a1f36;}
section[data-testid="stSidebar"] *{color:#f0ede6 !important;}
section[data-testid="stSidebar"] .side-title{font-family:'Noto Serif Lao',serif;font-weight:700;font-size:1.1rem;color:#fff !important;}
section[data-testid="stSidebar"] .side-caption{color:#a8b0c0 !important;font-size:.74rem;}
section[data-testid="stSidebar"] .stButton button{background:rgba(255,255,255,.08);color:#f0ede6 !important;border:1px solid rgba(255,255,255,.15);border-radius:9px;font-size:.82rem;text-align:left;}
section[data-testid="stSidebar"] .stButton button:hover{background:rgba(255,255,255,.15);border-color:var(--accent);}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12);}
.hist-label{color:#a8b0c0 !important;font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;margin:.8rem 0 .3rem;}

/* ---- MAIN-AREA buttons (example questions) — readable on parchment ---- */
.block-container .stButton button{
  background:#fff !important;
  color:#1a1f36 !important;
  border:1px solid var(--line) !important;
  border-radius:10px !important;
  font-size:.88rem !important;
  font-weight:500 !important;
  padding:.7rem 1rem !important;
  text-align:left !important;
  transition:all .15s ease;
}
.block-container .stButton button:hover{
  border-color:var(--accent) !important;
  background:var(--paper) !important;
  color:var(--accent) !important;
}
.block-container .stButton button p{color:inherit !important;}
</style>
""", unsafe_allow_html=True)


def render_answer(text: str):
    if not text:
        st.markdown("—"); return
    parts = re.split(r'(📋 ຄຳຕອບ:|📎 ຫຼັກຖານ:|⚠️ ຂໍ້ຄວນລະວັງ:|💡)', text)
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
            continue  # sources shown as clickable cards below (no duplication)
        elif marker.startswith("⚠️"):
            st.markdown('<div class="ans-head warn">⚠️ ຂໍ້ຄວນລະວັງ</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ans-section">{body}</div>', unsafe_allow_html=True)
        elif marker.startswith("💡"):
            st.caption(f"💡 {body}")


def render_citations(citations, key_prefix=""):
    """FIX 5: clickable references — click to expand the actual article text."""
    if not citations:
        return
    seen = set(); uniq = []
    for c in citations:
        k = (c.get("law",""), str(c.get("article","")))
        if k not in seen and c.get("law"):
            seen.add(k); uniq.append(c)
    if not uniq:
        return
    st.markdown('<div class="ans-head">📎 ແຫຼ່ງອ້າງອີງ</div>', unsafe_allow_html=True)
    for idx, c in enumerate(uniq):
        title = c.get("title", "")
        art = c.get("article", "")
        if art:
            header = f"ມາດຕາ {art} · {c['law']}"
        else:
            header = f"{c['law']}"   # page-level regulation chunk, no article #
        if title:
            header += f" — {title}"
        # clickable expander showing the actual law/reg content
        with st.expander(f"📎 {header}"):
            preview = c.get("preview", "")
            if preview:
                st.markdown(f'<div class="cite-preview">{preview}</div>', unsafe_allow_html=True)
            else:
                st.caption("(ບໍ່ມີຕົວຢ່າງເນື້ອໃນ)")


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


# ---------- state ----------
if "conversations" not in st.session_state:
    st.session_state.conversations = {}   # id -> {title, messages}
if "current_id" not in st.session_state:
    st.session_state.current_id = None
if "sidebar_hidden" not in st.session_state:
    st.session_state.sidebar_hidden = False

def _new_conversation():
    import time
    cid = str(int(time.time()*1000))
    st.session_state.conversations[cid] = {"title": "ສົນທະນາໃໝ່", "messages": []}
    st.session_state.current_id = cid
    return cid

def _current():
    if st.session_state.current_id is None or st.session_state.current_id not in st.session_state.conversations:
        _new_conversation()
    return st.session_state.conversations[st.session_state.current_id]


# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="side-title">⚖️ ກົດໝາຍລາວ</div>', unsafe_allow_html=True)
    if st.button("＋ ສົນທະນາໃໝ່", use_container_width=True, key="newchat"):
        _new_conversation(); st.rerun()

    # FIX 3: show/hide history toggle
    st.markdown('<div class="hist-label">ປະຫວັດການສົນທະນາ</div>', unsafe_allow_html=True)
    show_history = st.toggle("ສະແດງປະຫວັດ", value=True, key="show_hist")

    # FIX 2: chat history list (session-only) — click to switch conversation
    if show_history:
        if not st.session_state.conversations:
            st.caption("ຍັງບໍ່ມີການສົນທະນາ")
        else:
            for cid, conv in reversed(list(st.session_state.conversations.items())):
                label = conv["title"][:28] + ("…" if len(conv["title"]) > 28 else "")
                marker = "▸ " if cid == st.session_state.current_id else ""
                if st.button(f"{marker}{label}", key=f"conv_{cid}", use_container_width=True):
                    st.session_state.current_id = cid; st.rerun()

    st.divider()
    info = get_system_info()
    st.markdown(f'<div class="side-caption">📚 {info.get("total_laws","?")} ກົດໝາຍ · '
                f'{info.get("total_chunks","?")} ມາດຕາ</div>', unsafe_allow_html=True)
    if not info.get("ready"):
        st.error("ລະບົບຍັງບໍ່ພ້ອມ — ກວດ API keys")
        # TEMP DIAGNOSTIC: show the real error so we can fix it
        if info.get("error"):
            st.caption(f"🔧 {info.get('error')}")


# ---------- header ----------
st.markdown('<div class="app-head"><div class="mark">⚖️</div>'
            '<h1>ຜູ້ຊ່ວຍຄົ້ນຄວ້າກົດໝາຍລາວ</h1>'
            '<div class="sub">Lao Legal Research Assistant</div>'
            '<div class="rule"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="disc">⚠️ ນີ້ແມ່ນ<b>ເຄື່ອງມືຄົ້ນຄວ້າ</b> — ບໍ່ແມ່ນຄຳປຶກສາທາງກົດໝາຍ — '
            'ກະລຸນາປຶກສາທະນາຍຄວາມ</div>', unsafe_allow_html=True)

conv = _current()

# ---------- example questions (empty state) ----------
if not conv["messages"]:
    st.markdown('<div style="font-size:.8rem;font-weight:600;margin:1rem 0 .5rem;color:var(--muted);">'
                '💡 ຕົວຢ່າງຄຳຖາມ</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    examples = ['ມາດຕາ 52 ກົດໝາຍວິສາຫະກິດ','ອັດຕາອາກອນມູນຄ່າເພີ່ມ',
                'ເງື່ອນໄຂການລົງທຶນຕ່າງປະເທດ','ທຶນຈົດທະບຽນບໍລິສັດປະກັນໄພ']
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex

# ---------- history ----------
for mi, msg in enumerate(conv["messages"]):
    with st.chat_message(msg["role"], avatar="⚖️" if msg["role"]=="assistant" else "🧑"):
        if msg["role"] == "assistant":
            render_answer(msg["content"])
            render_citations(msg.get("citations", []), key_prefix=f"h{mi}")
            render_conf(msg.get("confidence","NONE"), msg.get("is_general", False))
        else:
            st.markdown(msg["content"])

# ---------- input ----------
pending = st.session_state.pop("pending", None)
question = st.chat_input("ພິມຄຳຖາມກ່ຽວກັບກົດໝາຍລາວ...") or pending

if question:
    conv["messages"].append({"role":"user","content":question})
    # set the conversation title from the first question (FIX 2)
    if conv["title"] == "ສົນທະນາໃໝ່":
        conv["title"] = question[:40]
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("🔍 ກຳລັງຄົ້ນຫາ ແລະ ວິເຄາະ..."):
            result = ask(question)
        render_answer(result.answer)
        render_citations(result.citations, key_prefix="live")
        render_conf(result.confidence, result.is_general)
        st.caption("⚠️ ກວດສອບກັບແຫຼ່ງທາງການ ຫຼື ທະນາຍຄວາມສະເໝີ")
    conv["messages"].append({
        "role":"assistant","content":result.answer,
        "citations":result.citations,"confidence":result.confidence,
        "is_general":result.is_general,
    })
    st.rerun()
