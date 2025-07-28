import json
from pathlib import Path
from typing import List, Dict, Union

import streamlit as st
from dotenv import load_dotenv

from chatbot import FAQChatbot, ChatResult

# ------------ Setup ------------
load_dotenv()

@st.cache_resource
def get_bot() -> FAQChatbot:
    return FAQChatbot(faq_path="ques.json", use_groq=True, groq_threshold=0.55)

bot = get_bot()

def load_first_20(path: str | Path = "ques.json") -> List[Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data[:20]

# ------------ Page Config ------------
st.set_page_config(page_title="QuickAI Chatbot", page_icon="💬", layout="wide")

# ------------ Sidebar ------------
MENU_CHAT   = "Chat"
MENU_FAQS   = "Example AI, ML, DL FAQs"
MENU_SET    = "Settings / Info"

section = st.sidebar.radio("Navigate", [MENU_CHAT, MENU_FAQS, MENU_SET], index=0)

if st.sidebar.button("🗑️ Delete chat history"):
    st.session_state.pop("messages", None)
    st.rerun()

# ------------ Session state ------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------ Renderers ------------
def render_chat():
    st.title("💬 QuickAI Chatbot")
    st.caption("Ask a question. This is an AI/ML/DL learning assistant.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Type your question…")
    if not user_q:
        return

    st.session_state.messages.append({"role": "user", "content": user_q})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result: Union[ChatResult, List[ChatResult]] = bot.ask(user_q, top_k=1)
            if isinstance(result, list) and result:
                result = result[0]

            answer = getattr(result, "answer", "Sorry, something went wrong.")
            if getattr(result, "source", "") == "groq_llm":
                answer = f"*(LLM answer)*\n{answer}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_faqs():
    st.title("📚 Example AI / ML / DL FAQs")
    st.caption("First 20 entries from ques.json")
    for i, item in enumerate(load_first_20(), 1):
        st.markdown(f"**{i}. {item['question']}**")
        st.write(item["answer"])
        st.markdown("---")


def render_settings():
    st.title("⚙️ Settings / Info")
    st.write(f"Similarity threshold: `{bot.groq_threshold}`")
    st.write("Embedding model: `all-MiniLM-L6-v2`")
    st.write(f"Total FAQs loaded: {len(bot.questions)}")
    st.write(f"Groq enabled: {bot.use_groq}")

# ------------ Router ------------
if section == MENU_CHAT:
    render_chat()
elif section == MENU_FAQS:
    render_faqs()
elif section == MENU_SET:
    render_settings()
else:            # Fallback so the screen never stays blank
    render_chat()
