# app.py
import json
from pathlib import Path
from typing import List, Dict, Union

import streamlit as st
from dotenv import load_dotenv

from chatbot import FAQChatbot, ChatResult


# ---------------- Setup ----------------
load_dotenv()  # GROQ_API_KEY / GROQ_MODEL etc.

@st.cache_resource
def get_bot() -> FAQChatbot:
    return FAQChatbot(faq_path="ques.json", use_groq=True, groq_threshold=0.55)

bot = get_bot()


def load_first_20(path: str | Path = "ques.json") -> List[Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data[:20]


# ---------------- Page Config ----------------
st.set_page_config(page_title="QuickAI Chatbot", page_icon="💬", layout="wide")

# ---------------- Sidebar (navigation + delete button) ----------------
section = st.sidebar.radio(
    "Navigate",
    ["💬 Chat", "📚 Example AI, ML, DL Faqs", "⚙️ Settings / Info"],
    index=0,
)

if st.sidebar.button("🗑️ Delete chat history"):
    st.session_state.pop("messages", None)
    st.rerun()

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role": "user"/"assistant", "content": "..." }]


# ---------------- Renderers ----------------
def render_chat():
    st.title("💬 QuickAI Chatbot")
    st.caption("Ask a question. This is an AI/ML/DL learning assistant")

    # Show previous turns
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # New user input
    user_q = st.chat_input("Type your question…")
    if not user_q:
        return

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_q})

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result: Union[ChatResult, List[ChatResult], None] = bot.ask(user_q, top_k=1)

            # Normalize result
            if isinstance(result, list) and result:
                result = result[0]

            # Safely extract answer/source
            answer = "Sorry, something went wrong."
            source = "unknown"

            # If ChatResult
            if isinstance(result, ChatResult):
                answer = result.answer
                source = result.source
            # If dict (in case you changed bot.ask to return dict)
            elif isinstance(result, dict):
                answer = result.get("answer", answer)
                source = result.get("source", source)

            # Tag LLM answers
            if source == "groq_llm":
                answer = f"*(LLM answer)*\n{answer}"

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_faqs():
    st.title("📚 Example ML / AI / DL FAQs")
    st.caption("Showing the first 20 entries from ques.json")
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


# ---------------- Router ----------------
if section == "Chat":
    render_chat()
elif section == "EXAMPLE ML AI DL FAQS":
    render_faqs()
elif section == "Settings / Info":
    render_settings()
