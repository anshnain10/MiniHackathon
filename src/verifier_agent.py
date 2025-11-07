# src/verifier_agent.py
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re

class VerifierAgent:
    """
    Factual precision: max cosine similarity >= threshold against source docs.
    Contradiction: NLI on (premise=evidence, hypothesis=claim).
    """
    def __init__(self, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 nli_model: str = "facebook/bart-large-mnli"):
        self.embed = SentenceTransformer(embed_model)
        self.nli = pipeline("text-classification", model=nli_model, device=-1)  # CPU

    @staticmethod
    def _split_claims(text: str) -> List[str]:
        lines = [l.strip("- ").strip() for l in text.splitlines() if l.strip().startswith("-")]
        return [re.sub(r"\[ID:[^\]]+\]", "", l).strip() for l in lines if l]

    def verify(self, summary: str, source_docs: List[Dict[str, Any]], sim_threshold: float = 0.8) -> Dict[str, Any]:
        claims = self._split_claims(summary)
        if not claims:
            return {"per_claim": [], "factual_precision": 0.0, "contradiction_rate": 0.0}

        evid = [d["doc"] for d in source_docs]
        evid_emb = self.embed.encode(evid, convert_to_numpy=True)

        precise, contradict = 0, 0
        per_claim = []

        for c in claims:
            c_emb = self.embed.encode([c], convert_to_numpy=True)[0]
            sims = cosine_similarity([c_emb], evid_emb)[0]
            best_i = int(np.argmax(sims))
            best_sim = float(sims[best_i])

            if best_sim >= sim_threshold:
                precise += 1

            # NLI: premise (evidence) vs hypothesis (claim)
            prem = evid[best_i][:900]
            out = self.nli({"text": prem, "text_pair": c})
            label = max(out, key=lambda x: x["score"])["label"]
            if "CONTRADICTION" in label.upper():
                contradict += 1

            per_claim.append({
                "claim": c, "best_doc_id": source_docs[best_i]["id"],
                "similarity": best_sim, "nli_label": label
            })

        n = max(1, len(claims))
        return {
            "per_claim": per_claim,
            "factual_precision": round(precise / n, 3),
            "contradiction_rate": round(contradict / n, 3)
        }
