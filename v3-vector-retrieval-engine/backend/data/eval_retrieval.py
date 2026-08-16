"""Offline retrieval quality evaluation.

For ~15 held-out queries from the corpus, run each of the 3 individual
strategies plus the ensemble, and check whether the passage the dataset
itself marked as relevant (is_selected == 1) shows up in the top-5 results.

Prints a simple hit-rate table per strategy so we have a defensible reason
to state which retrieval approach we chose.
"""
import io
import json
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval import RetrievalEngine


def load_eval_queries(corpus_path: Path, n_queries: int = 15) -> list[dict]:
    """Load held-out queries that have at least one relevant passage
    (is_selected == 1).  Group by query_id and collect relevant passage
    texts so we can check if retrieval surfaces them."""

    docs: list[dict] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    # Group by query_id
    query_groups: dict[str, dict] = {}
    for doc in docs:
        qid = str(doc.get("query_id", ""))
        if qid not in query_groups:
            query_groups[qid] = {
                "query_id": qid,
                "query_text": doc.get("query_text", ""),
                "relevant_passages": [],
                "all_doc_ids": [],
            }
        query_groups[qid]["all_doc_ids"].append(doc.get("doc_id", ""))
        if doc.get("is_selected") == 1:
            query_groups[qid]["relevant_passages"].append(
                doc.get("passage_text", "")[:100]
            )

    # Keep only queries with at least one relevant passage
    valid = [q for q in query_groups.values() if q["relevant_passages"]]

    # Skip the very first few queries (those may overlap with test fixtures)
    # and pick from the middle of the corpus as held-out
    offset = min(5, len(valid))
    selected = valid[offset : offset + n_queries]

    if len(selected) < n_queries:
        selected = valid[:n_queries]

    return selected


def passage_in_results(
    relevant_passages: list[str],
    results: list,
    query_id: str,
) -> bool:
    """Check if any retrieved chunk contains text from one of the relevant
    passages, or shares the same query_id with is_selected metadata."""
    for chunk in results:
        meta = chunk.metadata or {}
        # Direct metadata match: same query_id and is_selected == 1
        if str(meta.get("query_id")) == query_id and meta.get("is_selected") == 1:
            return True
        # Text overlap fallback: check if any relevant passage prefix is
        # contained in the chunk text
        for rp in relevant_passages:
            if rp[:60] in chunk.text or chunk.text[:60] in rp:
                return True
    return False


def main():
    corpus_path = Path("backend/data/corpus.jsonl")
    if not corpus_path.exists():
        corpus_path = Path("data/corpus.jsonl")

    print("=" * 70, flush=True)
    print("       RETRIEVAL QUALITY EVALUATION (TOP-5 HIT RATE)", flush=True)
    print("=" * 70, flush=True)

    print("\nLoading RetrievalEngine (all strategies)...", flush=True)
    engine = RetrievalEngine()

    strategies = engine.available_strategies + ["ensemble"]

    print(f"Loading {15} held-out evaluation queries...", flush=True)
    eval_queries = load_eval_queries(corpus_path, n_queries=15)
    print(f"Loaded {len(eval_queries)} queries with relevant passages.\n", flush=True)

    # Evaluate
    hits: dict[str, int] = {s: 0 for s in strategies}
    total = len(eval_queries)

    for i, q in enumerate(eval_queries):
        qid = q["query_id"]
        query_text = q["query_text"]
        relevant = q["relevant_passages"]

        print(f"[{i+1}/{total}] QID={qid} Query: {query_text[:60]}...", flush=True)

        for strat in strategies:
            if strat == "ensemble":
                results = engine.search_ensemble(query=query_text, top_k=5)
            else:
                results = engine.search(
                    query=query_text, strategy=strat, top_k=5
                )

            hit = passage_in_results(relevant, results, qid)
            if hit:
                hits[strat] += 1

    # Print results table
    print("\n" + "=" * 70, flush=True)
    print(f"{'RETRIEVAL HIT RATE TABLE (TOP-5)':^70}", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Strategy':<30} | {'Hits':>6} / {'Total':>6} | {'Hit Rate':>10}", flush=True)
    print("-" * 70, flush=True)

    for strat in strategies:
        rate = (hits[strat] / total * 100) if total > 0 else 0.0
        marker = " <-- BEST" if hits[strat] == max(hits.values()) else ""
        print(
            f"{strat:<30} | {hits[strat]:>6} / {total:>6} | {rate:>9.1f}%{marker}",
            flush=True,
        )

    print("=" * 70, flush=True)

    best = max(hits, key=hits.get)
    print(f"\nRecommendation: Use '{best}' for production retrieval.", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
