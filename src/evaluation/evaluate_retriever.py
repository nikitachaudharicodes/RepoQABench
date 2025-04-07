import os
import json
from typing import List, Dict

def load_retriever_outputs(folder: str) -> List[Dict]:
    """Load all retriever output JSON files from a folder."""
    outputs = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), 'r') as f:
                outputs.append(json.load(f))
    return outputs

def compute_recall_at_k(results: List[Dict], k: int = 3) -> float:
    """Compute Recall@k."""
    hits = 0
    for entry in results:
        golden = entry["golden_context"]
        retrieved = entry["retrieved"][:k]
        match_found = any(
            r["file_path"] == golden["file_path"] and r["name"] == golden["name"]
            for r in retrieved
        )
        hits += int(match_found)
    return hits / len(results)

def compute_mrr_at_k(results: List[Dict], k: int = 3) -> float:
    """Compute Mean Reciprocal Rank at k."""
    total_rr = 0.0
    for entry in results:
        golden = entry["golden_context"]
        retrieved = entry["retrieved"][:k]
        for rank, r in enumerate(retrieved, start=1):
            if r["file_path"] == golden["file_path"] and r["name"] == golden["name"]:
                total_rr += 1 / rank
                break
    return total_rr / len(results)

def compute_avg_cosine_similarity(results: List[Dict], k: int = 3) -> float:
    """Compute average cosine similarity of top-k retrieved items."""
    total = 0.0
    count = 0
    for entry in results:
        for r in entry["retrieved"][:k]:
            total += r.get("similarity", 0.0)
            count += 1
    return total / count if count > 0 else 0.0


 

if __name__ == "__main__":
    folder = "retriever_outputs"  # path to your JSON files
    results = load_retriever_outputs(folder)
    
    for k in [1, 3, 5]:
        recall = compute_recall_at_k(results, k)
        mrr = compute_mrr_at_k(results, k)
        avg_sim = compute_avg_cosine_similarity(results, k)
        print(f"Avg Cosine Similarity@{k}: {avg_sim:.4f}")
        print(f"Recall@{k}: {recall:.4f}")
        print(f"MRR@{k}: {mrr:.4f}")
