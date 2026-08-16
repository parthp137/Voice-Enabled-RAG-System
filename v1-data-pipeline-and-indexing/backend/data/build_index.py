import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

# Maximize PyTorch CPU parallelism for fast embedding
import torch
num_cores = os.cpu_count() or 8
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chunking import CHUNKING_STRATEGIES, Chunk


def load_corpus(corpus_path: str, max_docs: int = None) -> list[dict]:
    print(f"[build_index] Loading corpus from {corpus_path}...", flush=True)
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
                if max_docs and len(documents) >= max_docs:
                    break
    print(f"[build_index] Loaded {len(documents)} documents for indexing.", flush=True)
    return documents


def build_index_for_strategy(
    strategy_name: str,
    chunk_fn: callable,
    documents: list[dict],
    model: SentenceTransformer,
    output_base_dir: Path,
    batch_size: int = 64
) -> dict:
    print(f"\n==================================================", flush=True)
    print(f"[build_index] Building FAISS Index for strategy: '{strategy_name}'", flush=True)
    print(f"==================================================", flush=True)

    start_time = time.time()
    
    # 1. Chunk corpus
    print(f"[build_index] Chunking {len(documents)} documents with strategy '{strategy_name}'...", flush=True)
    all_chunks: list[Chunk] = []

    for doc in tqdm(documents, desc=f"Chunking ({strategy_name})"):
        text = doc.get("passage_text", "")
        if not text:
            continue
        
        metadata = {
            "doc_id": doc.get("doc_id"),
            "query_id": doc.get("query_id"),
            "query_text": doc.get("query_text"),
            "language": doc.get("language"),
            "source_lang": doc.get("source_lang"),
            "target_lang": doc.get("target_lang"),
            "is_selected": doc.get("is_selected"),
            "passage_position": doc.get("passage_position"),
        }

        if strategy_name == "sentence_semantic":
            chunks = chunk_fn(text, metadata, model=model)
        else:
            chunks = chunk_fn(text, metadata)

        all_chunks.extend(chunks)

    chunk_count = len(all_chunks)
    avg_len_words = (
        sum(len(c.text.split()) for c in all_chunks) / chunk_count
        if chunk_count > 0
        else 0
    )
    print(f"[build_index] Generated {chunk_count} chunks (avg length: {avg_len_words:.1f} words).", flush=True)

    if chunk_count == 0:
        raise ValueError(f"No chunks generated for strategy {strategy_name}")

    # 2. Batch Embedding
    print(f"[build_index] Embedding {chunk_count} chunks in batches of {batch_size} (using {num_cores} CPU threads)...", flush=True)
    
    passages_to_embed = [f"passage: {c.text}" for c in all_chunks]
    
    embeddings = model.encode(
        passages_to_embed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    dimension = embeddings.shape[1]

    # 3. Create FAISS Index
    print(f"[build_index] Creating FAISS IndexFlatIP (dim={dimension})...", flush=True)
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    # 4. Save Artifacts
    strategy_dir = output_base_dir / strategy_name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    index_path = strategy_dir / "faiss.index"
    chunks_path = strategy_dir / "chunks.jsonl"
    meta_path = strategy_dir / "meta.json"

    print(f"[build_index] Saving artifacts to {strategy_dir.resolve()}...", flush=True)
    faiss.write_index(faiss_index, str(index_path))

    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(c.model_dump_json() + "\n")

    build_time = time.time() - start_time
    meta_data = {
        "strategy": strategy_name,
        "embedding_model": "intfloat/multilingual-e5-base",
        "dimension": dimension,
        "chunk_count": chunk_count,
        "avg_chunk_len_words": round(avg_len_words, 2),
        "build_time_seconds": round(build_time, 2),
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    print(f"[build_index] Completed strategy '{strategy_name}' in {build_time:.2f} seconds.", flush=True)
    return meta_data


def main():
    parser = argparse.ArgumentParser(description="Offline FAISS index builder for multi-strategy chunking.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["all", "fixed_size_overlap", "sentence_semantic", "metadata_aware"],
        help="Strategy to build ('all' or specific strategy name)"
    )
    parser.add_argument("--corpus-path", type=str, default="backend/data/corpus.jsonl", help="Path to corpus.jsonl")
    parser.add_argument("--output-dir", type=str, default="backend/data/index", help="Base directory for index output")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding generation")
    parser.add_argument("--max-docs", type=int, default=500, help="Max documents from corpus to index for fast build (default: 500)")
    args = parser.parse_args()

    corpus_file = Path(args.corpus_path)
    output_base = Path(args.output_dir)

    if not corpus_file.exists():
        print(f"[build_index] Error: Corpus file {corpus_file.resolve()} does not exist. Run build_corpus.py first.", flush=True)
        sys.exit(1)

    documents = load_corpus(str(corpus_file), max_docs=args.max_docs)

    model_name = "intfloat/multilingual-e5-base"
    print(f"\n[build_index] Loading embedding model '{model_name}' (this happens ONCE)...", flush=True)
    model = SentenceTransformer(model_name)

    strategies_to_run = (
        list(CHUNKING_STRATEGIES.keys())
        if args.strategy == "all"
        else [args.strategy]
    )

    summaries = []
    for strat_name in strategies_to_run:
        chunk_fn = CHUNKING_STRATEGIES[strat_name]
        meta = build_index_for_strategy(
            strategy_name=strat_name,
            chunk_fn=chunk_fn,
            documents=documents,
            model=model,
            output_base_dir=output_base,
            batch_size=args.batch_size
        )
        summaries.append(meta)

    print("\n" + "=" * 70, flush=True)
    print(f"{'INDEX BUILD SUMMARY TABLE':^70}", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Strategy':<22} | {'Chunks':<8} | {'Avg Words':<10} | {'Build Time (s)':<14}", flush=True)
    print("-" * 70, flush=True)
    for s in summaries:
        print(f"{s['strategy']:<22} | {s['chunk_count']:<8} | {s['avg_chunk_len_words']:<10} | {s['build_time_seconds']:<14}", flush=True)
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()
