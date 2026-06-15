"""
app.py — Lao Legal RAG UI (clean, with Claude-style thinking panel)
===================================================================
Edit ONLY this file. Backend is one call: ask(question) -> result.
result.answer / .citations / .confidence / .steps / .error
Run:    streamlit run app.py
Deploy: push to GitHub -> Streamlit Cloud auto-deploys.
"""
import re
import streamlit as st
from backend import ask, get_system_info

st.set_page_config(page_title="ບອດກົດໝາຍລາວ", page_icon="⚖️", layout="wide")

# ---------------- styling ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Lao:wght@400;500;600;700&display=swap');
*{font-family:'Noto Sans Lao',sans-serif;}
.block-container{padding-top:2rem; max-width:820px;}
.welcome{text-align:center; padding:3rem 1rem;}
.welcome .icon{font-size:3rem;}
.welcome h2{font-weight:700; margin:.5rem 0 .3rem;}
.welcome p{color:#6b7280; font-size:.9rem;}
section[data-testid="stSidebar"]{background:#fafafa; border-right:1px solid #eee;}
.newchat button{background:#2563eb !important; color:#fff !important; border:none !important;
  border-radius:10px !important; font-weight:600 !important;}
.stChatMessage{padding:.3rem 0;}
.cite-box{background:#f8fafc; border:1px solid #e8eef5; border-radius:8px;
  padding:.5rem .7rem; margin:.3rem 0; font-size:.85rem;}
.cite-box b{color:#1e3a5f;}
</style>
""", unsafe_allow_html=True)


def format_answer(text: str) -> str:
    """Separate the answer's sections so it reads cleanly, not as one block."""
    if not text:
        return text
    for m in ["📋 ຄຳຕອບ:", "📎 ຫຼັກຖານ:", "⚠️ ຂໍ້ຄວນລະວັງ:", "💡"]:
        text = text.replace(m, "\n\n" + m)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### ⚖️ ກົດໝາຍລາວ")
    st.markdown('<div class="newchat">', unsafe_allow_html=True)
    if st.button("＋ ສົນທະນາໃໝ່", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.text_input("ຄົ້ນຫາຄຳຖາມ...", key="search_box", label_visibility="collapsed",
                  placeholder="🔍 ຄົ້ນຫາຄຳຖາມ...")
    info = get_system_info()
    st.divider()
    st.caption(f"📚 {info.get('total_laws','?')} ກົດໝາຍ · {info.get('total_chunks','?')} ມາດຕາ")
    if not info.get("ready"):
        st.error("ລະບົບຍັງບໍ່ພ້ອມ — ກວດສອບ API keys")

# ---------------- main ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    # welcome screen (matches the reference image)
    st.markdown("""
    <div class="welcome">
      <div class="icon">⚖️</div>
      <h2>ຍິນດີຕ້ອນຮັບສູ່ ແຊັດບອດກົດໝາຍລາວ</h2>
      <p>ຖາມຄຳຖາມກ່ຽວກັບກົດໝາຍລາວ ແລະ ຮັບຄຳຕອບທີ່ຊັດເຈນ</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="⚖️" if msg["role"]=="assistant" else "🧑"):
            if msg["role"] == "assistant":
                # Claude-style thinking panel (collapsed) showing the steps
                if msg.get("steps"):
                    with st.expander("🧠 ຂັ້ນຕອນການຄິດ", expanded=False):
                        for s in msg["steps"]:
                            st.markdown(f"- {s}")
                st.markdown(format_answer(msg["content"]))
                for c in msg.get("citations", []):
                    st.markdown(
                        f'<div class="cite-box">📎 <b>{c["law"]}</b> ມາດຕາ {c["article"]}'
                        + (f' — {c["title"]}' if c.get("title") else "") + '</div>',
                        unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

# ---------------- input ----------------
question = st.chat_input("ກົດໝາຍແມ່ນຫຍັງ?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="⚖️"):
        # show steps live as a status while working (Claude-style)
        with st.status("ກຳລັງຄິດ...", expanded=True) as status:
            result = ask(question)
            for s in result.steps:
                st.write(s)
            status.update(label="ສຳເລັດ", state="complete", expanded=False)

        st.markdown(format_answer(result.answer))
        for c in result.citations:
            st.markdown(
                f'<div class="cite-box">📎 <b>{c["law"]}</b> ມາດຕາ {c["article"]}'
                + (f' — {c["title"]}' if c.get("title") else "") + '</div>',
                unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant", "content": result.answer,
        "citations": result.citations, "steps": result.steps,
    })
    st.rerun()
