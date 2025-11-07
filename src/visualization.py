import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

Path("results").mkdir(exist_ok=True)
G = nx.DiGraph()
stages = ["Classical NLP", "Retrieve", "Summarize", "Verify"]
for i in range(len(stages)-1):
    G.add_edge(stages[i], stages[i+1])

plt.figure()
nx.draw(G, with_labels=True, node_size=3000)
plt.title("LangGraph Pipeline")
plt.savefig("results/pipeline_graph.png", bbox_inches="tight")
print("Saved results/pipeline_graph.png")
