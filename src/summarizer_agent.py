# src/summarizer_agent.py
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PROMPT = """You are a fact-grounded summarizer.
Given SOURCE documents (each has an ID), produce 3–6 bullet-point CLAIMS.
Each claim MUST include citations like [ID:xxxxx].
If a claim is not supported, write UNSUPPORTED.

SOURCES:
{sources}

OUTPUT (bullet claims with citations):
"""

class SummarizerAgent:
    """
    Local FLAN-T5 summarizer (CPU-friendly). Use flan-t5-small if needed.
    """
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _format_sources(self, docs: List[Dict[str, Any]]) -> str:
        # Keep input small; truncate doc text
        lines = []
        for d in docs:
            text = d["doc"]
            lines.append(f"ID:{d['id']}\n{text[:1200]}")
        return "\n\n".join(lines)

    def generate_summary(self, candidate_docs: List[Dict[str, Any]], max_new_tokens: int = 256) -> str:
        prompt = PROMPT.format(sources=self._format_sources(candidate_docs))
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
        return self.tok.decode(out[0], skip_special_tokens=True)
