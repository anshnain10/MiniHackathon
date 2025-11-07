# src/orchestrator_langgraph.py
from __future__ import annotations
from typing import TypedDict, Dict, Any, List
from pathlib import Path
import time

from langgraph.graph import StateGraph, END
from utils import DATA_DIR, RESULTS_DIR, save_json, load_json, normalize_corpus_csv, compact_text
from classical_agent import ClassicalAgent
from retriever_hybrid import HybridRetriever
from summarizer_agent import SummarizerAgent
from verifier_agent import VerifierAgent

class State(TypedDict, total=False):
    query_id: str
    query: str
    top_k: int
    alpha: float
    retry_threshold: float
    retrieved: List[Dict[str, Any]]
    summary: str
    metrics: Dict[str, Any]
    attempt: int
    trace: List[Dict[str, Any]]

def log(state: State, step: str, payload: Dict[str, Any]):
    entry = {"step": step, "payload": payload, "ts": time.time()}
    state.setdefault("trace", []).append(entry)
    save_json(state["trace"], RESULTS_DIR / "trace.json")

# Heavy objects once
CLASSICAL = ClassicalAgent()
RETRIEVER = HybridRetriever()
SUMMARIZER = SummarizerAgent(model_name="google/flan-t5-small")  # smaller = faster
VERIFIER = VerifierAgent()

def node_classical(s: State) -> State:
    qa = CLASSICAL.analyze_documents([s["query"]])[0]
    log(s, "classical_query", {"ents": qa["ner"], "sent": qa["sentiment"]})
    return s

def node_retrieve(s: State) -> State:
    res = RETRIEVER.hybrid_query(s["query"], top_k=s["top_k"], alpha=s["alpha"])
    s["retrieved"] = res
    log(s, "retrieve", {"k": s["top_k"], "alpha": s["alpha"], "ids": [r["id"] for r in res]})
    return s

def node_summarize(s: State) -> State:
    s["summary"] = SUMMARIZER.generate_summary(s["retrieved"])
    (RESULTS_DIR / f"generated_summary_{s['query_id']}.txt").write_text(s["summary"], "utf-8")
    log(s, "summarize", {"preview": s["summary"][:300]})
    return s

def node_verify(s: State) -> State:
    m = VERIFIER.verify(s["summary"], s["retrieved"], sim_threshold=0.8)
    s["metrics"] = m
    save_json(m, RESULTS_DIR / f"metrics_{s['query_id']}.json")
    log(s, "verify", m)
    return s

def decide_retry(s: State) -> str:
    m = s.get("metrics", {}) or {}
    prec = float(m.get("factual_precision", 0.0))
    contra = float(m.get("contradiction_rate", 0.0))
    conf = max(0.0, prec * (1.0 - contra))
    s["attempt"] = int(s.get("attempt", 0))
    log(s, "confidence", {"proxy_conf": conf, "attempt": s["attempt"]})
    if conf < s["retry_threshold"] and s["attempt"] < 1:
        s["top_k"] = min(int(s.get("top_k", 10)) * 2, 30)
        s["alpha"] = min(float(s.get("alpha", 0.6)) + 0.1, 0.9)
        s["attempt"] += 1
        log(s, "retry", {"new_k": s["top_k"], "new_alpha": s["alpha"]})
        return "retry"
    return "done"


def build_graph():
    g = StateGraph(State)
    g.add_node("classical", node_classical)
    g.add_node("retrieve", node_retrieve)
    g.add_node("summarize", node_summarize)
    g.add_node("verify", node_verify)
    g.set_entry_point("classical")
    g.add_edge("classical", "retrieve")
    g.add_edge("retrieve", "summarize")
    g.add_edge("summarize", "verify")
    g.add_conditional_edges("verify", decide_retry, {"retry": "retrieve", "done": END})
    return g.compile()

def setup_indexes(csv_path: Path):
    df = normalize_corpus_csv(csv_path)
    texts = [compact_text(r.title, r.content) for _, r in df.iterrows()]
    ids = df["id"].tolist()
    CLASSICAL.fit_tfidf_vectorizer(texts)
    RETRIEVER.build_index(texts, ids)

def run_all(data_csv="data/news_corpus.csv", queries_json="data/queries.json",
            alpha=0.7, k=10, retry_threshold=0.7):
    setup_indexes(Path(data_csv))
    queries = load_json(Path(queries_json))
    graph = build_graph()

    index = []
    for q in queries:
        s: State = {
            "query_id": q["id"], "query": q["query"],
            "top_k": k, "alpha": alpha, "retry_threshold": retry_threshold,
            "attempt": 0, "trace": []
        }
        graph.invoke(s, config={"recursion_limit": 3})

        save_json({"query": q["query"], "alpha": s["alpha"], "top_k": s["top_k"],
                   "results": s["retrieved"]}, RESULTS_DIR / f"retrieval_{q['id']}.json")
        index.append({
            "query_id": q["id"],
            "retrieval_file": f"retrieval_{q['id']}.json",
            "summary_file": f"generated_summary_{q['id']}.txt",
            "metrics_file": f"metrics_{q['id']}.json",
        })
    save_json(index, RESULTS_DIR / "trace_index.json")
    print("Done. Check results/")

if __name__ == "__main__":
    run_all()
