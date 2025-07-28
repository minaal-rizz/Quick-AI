# chatbot.py
"""
FAQ Chatbot core:
- Load Q/A pairs from ques.json
- Embed questions once
- Retrieve best match via cosine similarity
- (Optional) Use Groq LLM to polish answers or handle low-similarity queries
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm_groq import GroqLLM  # optional LLM polish/fallback


# ------------------------- DATA TYPES ---------------------- #

@dataclass
class FAQItem:
    question: str
    answer: str


@dataclass
class ChatResult:
    matched_question: Optional[str]
    answer: str
    score: float
    source: str  # "faq", "faq+groq", "groq_llm", "fallback"

    def to_dict(self):
        return asdict(self)


# ------------------------- MAIN CLASS ---------------------- #

class FAQChatbot:
    """
    Load FAQs, embed questions, and answer user queries by semantic similarity.
    Optionally uses Groq LLM:
      - to rewrite answers more clearly
      - to answer when similarity is low (fallback)
    """

    def __init__(
        self,
        faq_path: Union[str, Path] = "ques.json",
        embed_model_name: str = "all-MiniLM-L6-v2",
        use_groq: bool = True,
        groq_threshold: float = 0.55,
    ):
        self.faq_path = Path(faq_path)
        self.embed_model = SentenceTransformer(embed_model_name)
        self.use_groq = use_groq
        self.groq_threshold = groq_threshold
        self.groq = GroqLLM() if use_groq else None

        self.faq_items: List[FAQItem] = self._load_faqs(self.faq_path)
        self.questions = [item.question for item in self.faq_items]
        self.answers = [item.answer for item in self.faq_items]
        self.embeddings = self.embed_model.encode(self.questions, show_progress_bar=False)

    @staticmethod
    def _load_faqs(path: Path) -> List[FAQItem]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return [FAQItem(q["question"], q["answer"]) for q in data]

    def ask(self, user_query: str, top_k: int = 1) -> Union[ChatResult, List[ChatResult]]:
        """Return the best answer(s) for the user's query."""
        user_emb = self.embed_model.encode([user_query], show_progress_bar=False)
        sims = cosine_similarity(user_emb, self.embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        # Low similarity → ask Groq instead (if enabled)
        if self.use_groq and best_score < self.groq_threshold:
            llm_ans = self._groq_answer_fallback(user_query)
            if llm_ans:
                return ChatResult(None, llm_ans, best_score, "groq_llm")
            return ChatResult(None, "Sorry, I couldn't find a good match.", best_score, "fallback")

        # Otherwise return FAQ match(es)
        if top_k == 1:
            return self._build_single_result(best_idx, best_score)
        return self._build_multi_results(sims, top_k)

    # --------------------- Helpers -------------------------- #

    def _build_single_result(self, idx: int, score: float) -> ChatResult:
        q, a = self.questions[idx], self.answers[idx]
        if self.use_groq:
            polished = self._groq_polish(a)
            return ChatResult(q, polished or a, score, "faq+groq")
        return ChatResult(q, a, score, "faq")

    def _build_multi_results(self, sims: np.ndarray, k: int) -> List[ChatResult]:
        idxs = np.argsort(sims)[::-1][:k]
        results: List[ChatResult] = []
        for i in idxs:
            q, a, score = self.questions[i], self.answers[i], float(sims[i])
            if self.use_groq:
                a = self._groq_polish(a) or a
                src = "faq+groq"
            else:
                src = "faq"
            results.append(ChatResult(q, a, score, src))
        return results

    def _groq_polish(self, answer: str) -> Optional[str]:
        if not self.groq:
            return None
        prompt = f"Rewrite this answer clearly and concisely:\n\n{answer}"
        return self.groq.generate(prompt)

    def _groq_answer_fallback(self, user_q: str) -> Optional[str]:
        if not self.groq:
            return None
        prompt = (
            "You are an FAQ assistant. If you cannot answer, say you don't know.\n\n"
            f"User question: {user_q}"
        )
        return self.groq.generate(prompt)
