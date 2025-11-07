# src/retriever_hybrid.py
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever:
    """
    Sparse (BM25) + Dense (MiniLM) hybrid.
    score = alpha*dense + (1-alpha)*sparse, both min-max normalized.
    """
    def __init__(self, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed = SentenceTransformer(embed_model_name)
        self.ids: List[str] = []
        self.docs: List[str] = []
        self.bm25 = None
        self.doc_emb = None

    def build_index(self, docs: List[str], ids: List[str]):
        self.docs = docs
        self.ids  = ids
        tok = [d.lower().split() for d in docs]
        self.bm25 = BM25Okapi(tok)
        self.doc_emb = self.embed.encode(docs, convert_to_numpy=True, show_progress_bar=True)

    @staticmethod
    def _norm(a: np.ndarray) -> np.ndarray:
        a = a.astype(float)
        return np.zeros_like(a) if a.max() == a.min() else (a - a.min()) / (a.max() - a.min())

    def hybrid_query(self, query: str, top_k: int = 10, alpha: float = 0.6) -> List[Dict[str, Any]]:
        if self.bm25 is None or self.doc_emb is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # sparse
        sparse = self.bm25.get_scores(query.lower().split())
        s_norm = self._norm(np.array(sparse))

        # dense
        q = self.embed.encode([query], convert_to_numpy=True)[0]
        d_norm = self._norm(cosine_similarity([q], self.doc_emb)[0])

        hybrid = alpha * d_norm + (1 - alpha) * s_norm
        top_idx = np.argsort(-hybrid)[:top_k]

        results = []
        for i in top_idx:
            results.append({
                "id": self.ids[i],
                "doc": self.docs[i],
                "sparse_score": float(s_norm[i]),
                "dense_score": float(d_norm[i]),
                "hybrid_score": float(hybrid[i]),
            })
        return results
