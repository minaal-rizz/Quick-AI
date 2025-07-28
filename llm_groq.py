# groq_llm.py
"""
Tiny wrapper around Groq's chat completions API.

Reads:
  - GROQ_API_KEY
  - GROQ_MODEL  (e.g. "llama3-8b-8192")

Usage:
    from groq_llm import GroqLLM
    llm = GroqLLM()
    text = llm.generate("Rewrite this clearly.")
"""

import os
from typing import Optional

from groq import Groq, BadRequestError, NotFoundError


class GroqLLM:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")

        self.client = Groq(api_key=api_key)
        self.model = model or os.getenv("GROQ_MODEL", "llama3-8b-8192")

    def generate(
        self,
        prompt: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> Optional[str]:
        """
        Returns the model's reply text, or None if the request fails.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()

        except (BadRequestError, NotFoundError) as e:
            # Model name wrong, schema mismatch, etc.
            print(f"[GroqLLM] Request failed ({e.__class__.__name__}): {e}")
            return None
        except Exception as e:
            print(f"[GroqLLM] Unexpected error: {e}")
            return None
