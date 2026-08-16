import io
import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chunking import CHUNKING_STRATEGIES


def run_demo():
    print("=" * 70, flush=True)
    print("           HHG VOICE RAG - PROMPT 1 PIPELINE DEMO", flush=True)
    print("=" * 70, flush=True)

    corpus_path = Path("backend/data/corpus.jsonl")
    if not corpus_path.exists():
        corpus_path = Path("data/corpus.jsonl")

    # 1. Load Sample Document
    print("\n1. SAMPLE DOCUMENT FROM CORPUS:", flush=True)
    print("-" * 50, flush=True)
    with open(corpus_path, "r", encoding="utf-8") as f:
        sample_doc = json.loads(f.readline())

    print(f"Doc ID      : {sample_doc['doc_id']}", flush=True)
    print(f"Language    : {sample_doc['language']}", flush=True)
    print(f"Query ID    : {sample_doc['query_id']}", flush=True)
    print(f"Hindi Query : {sample_doc['query_text']}", flush=True)
    print(f"Passage Text: {sample_doc['passage_text'][:200]}...", flush=True)

    # 2. Run All 3 Chunking Strategies Live
    print("\n\n2. CHUNKING STRATEGIES COMPARISON (LIVE EXECUTED ON PASSAGE):", flush=True)
    print("=" * 70, flush=True)

    metadata = {
        "doc_id": sample_doc["doc_id"],
        "query_id": sample_doc["query_id"],
        "query_text": sample_doc["query_text"],
        "language": sample_doc["language"],
        "source_lang": sample_doc["source_lang"],
        "target_lang": sample_doc["target_lang"],
        "is_selected": sample_doc["is_selected"],
        "passage_position": sample_doc["passage_position"],
    }

    for strat_name, strat_fn in CHUNKING_STRATEGIES.items():
        print(f"\n---> STRATEGY: '{strat_name}'", flush=True)
        chunks = strat_fn(sample_doc["passage_text"], metadata)
        print(f"     Chunks produced: {len(chunks)}", flush=True)
        for idx, c in enumerate(chunks[:2]):
            print(f"     [Chunk {idx}] ID: {c.chunk_id} | Words: {len(c.text.split())}", flush=True)
            print(f"                  Text: {c.text[:120]}...", flush=True)
            if strat_name == "metadata_aware":
                print(f"                  Enriched Meta: {c.metadata}", flush=True)

    # 3. Load Warm FAISS Vector Index & Perform Semantic Retrieval Query
    print("\n\n3. WARM FAISS VECTOR SEARCH DEMO:", flush=True)
    print("=" * 70, flush=True)
    
    index_dir = Path("backend/data/index/sentence_semantic")
    if not index_dir.exists():
        index_dir = Path("data/index/sentence_semantic")

    faiss_file = index_dir / "faiss.index"
    chunks_file = index_dir / "chunks.jsonl"
    meta_file = index_dir / "meta.json"

    print(f"Loading FAISS Index from {faiss_file}...", flush=True)
    index = faiss.read_index(str(faiss_file))
    print(f"Loaded FAISS Index! Total Indexed Vectors: {index.ntotal}", flush=True)

    chunks_list = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks_list.append(json.loads(line))

    with open(meta_file, "r", encoding="utf-8") as f:
        idx_meta = json.load(f)
    print(f"Index Metadata: Model={idx_meta['embedding_model']}, Dim={idx_meta['dimension']}", flush=True)

    # Perform Sample Vector Query
    test_query = sample_doc.get("query_text") or "भारत का इतिहास"
    print(f"\nQuerying vector index with Hindi query: '{test_query}'", flush=True)

    model = SentenceTransformer("intfloat/multilingual-e5-base")
    query_emb = model.encode([f"query: {test_query}"], normalize_embeddings=True)
    query_emb = np.array(query_emb, dtype=np.float32)

    k = 3
    scores, indices = index.search(query_emb, k)

    print(f"\nTop-{k} Retrieved Results (Cosine Similarity Scores):", flush=True)
    print("-" * 70, flush=True)
    for rank, (score, idx_pos) in enumerate(zip(scores[0], indices[0]), start=1):
        matched_chunk = chunks_list[idx_pos]
        print(f"Rank {rank} | Cosine Similarity Score: {score:.4f}", flush=True)
        print(f"       Chunk ID  : {matched_chunk['chunk_id']}", flush=True)
        print(f"       Strategy  : {matched_chunk['strategy']}", flush=True)
        print(f"       Text      : {matched_chunk['text'][:150]}...", flush=True)
        print("-" * 70, flush=True)

    print("\n" + "=" * 70, flush=True)
    print("                  PROMPT 1 DEMO COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    run_demo()
