# 💬 QuickAI – Intelligent FAQ Chatbot

**QuickAI** is a smart, responsive chatbot that intelligently answers user questions using a hybrid system:
- It matches the most relevant FAQs from a custom knowledge base.
- Falls back to the **Groq LLaMA3 8B** language model when a close match isn't found.
- Is a learning platform for most frequent AI questions.

---

## 🚀 Features

- ✅ Match user queries to predefined FAQ questions using semantic similarity (via `sentence-transformers`)
- ✅ Use **Groq LLaMA3-8B-8192** for fallback answers
- ✅ Streamlit-powered UI with chat experience
- ✅ Shows FAQ list and system settings from sidebar
- ✅ Editable `ques.json` file to define your own FAQs
- ✅ Delete chat history with one click
- ✅ Fast and lightweight performance

---

## 📁 Project Structure

# 💬 QuickAI – Intelligent FAQ Chatbot

**QuickAI** is a smart, responsive chatbot that intelligently answers user questions using a hybrid system:
- It matches the most relevant FAQs from a custom knowledge base.
- Falls back to the **Groq LLaMA3 8B** language model when a close match isn't found.

---

## 🚀 Features

- ✅ Match user queries to predefined FAQ questions using semantic similarity (via `sentence-transformers`)
- ✅ Use **Groq LLaMA3-8B-8192** for fallback answers
- ✅ Streamlit-powered UI with chat experience
- ✅ Shows FAQ list and system settings from sidebar
- ✅ Editable `ques.json` file to define your own FAQs
- ✅ Delete chat history with one click
- ✅ Fast and lightweight performance

---

## 📁 Project Structure

```
QuickAI/
│
├── app.py # Main Streamlit application
├── chatbot.py # Core chatbot logic
├── llm_groq.py # Handles Groq LLM API integration
├── ques.json # Custom FAQ question-answer list
├── requirements.txt # Python dependencies
└── .env # Contains your GROQ_API_KEY and model ```


---

## 🧠 How It Works

1. **Similarity Matching**  
   Compares user input to all questions in `ques.json` using vector embeddings.

2. **LLM Fallback**  
   If no FAQ match exceeds the similarity threshold (default `0.55`), the prompt is forwarded to **Groq LLaMA3 8B** for a generative response.

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/minaal-rizz/QuickAI.git
cd QuickAI

# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

#YOUR .ENV
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=llama3-8b-8192

Acknowledgements
Groq for blazing-fast inference

Streamlit for the elegant UI

SentenceTransformers for semantic matching