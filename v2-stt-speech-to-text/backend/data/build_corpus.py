import argparse
import json
import os
import sys
from pathlib import Path
import duckdb
from huggingface_hub import hf_hub_download


def load_msmarco_rows(limit: int = 3000) -> list[dict]:
    """
    Loads MSMARCO-XI Hindi split rows. Downloads or uses cached parquet file
    via hf_hub_download and queries it using DuckDB for instant row retrieval.
    """
    files_to_try = [
        ("validation/hinval.parquet", "validation"),
        ("train/hintrain.parquet", "train")
    ]

    for filename, split_name in files_to_try:
        print(f"[build_corpus] Requesting {filename} via Hugging Face Hub download/cache...", flush=True)
        try:
            local_path = hf_hub_download(
                repo_id="ai4bharat/MSMARCO-XI",
                filename=filename,
                repo_type="dataset"
            )
            print(f"[build_corpus] Local parquet path: {local_path}", flush=True)
            print(f"[build_corpus] Querying first {limit} rows using DuckDB...", flush=True)
            
            con = duckdb.connect()
            # Normalize Windows backslashes for DuckDB query
            escaped_path = local_path.replace("\\", "/")
            query = f"SELECT * FROM '{escaped_path}' LIMIT {limit}"
            res = con.execute(query).fetchall()
            cols = [desc[0] for desc in con.description]
            
            rows = []
            for r in res:
                rows.append(dict(zip(cols, r)))
            print(f"[build_corpus] Successfully loaded {len(rows)} rows from {split_name} split.", flush=True)
            return rows
        except Exception as e:
            print(f"[build_corpus] Failed to load {filename}: {e}", flush=True)

    raise RuntimeError("Failed to load MSMARCO-XI dataset from Hugging Face Hub.")


def build_corpus(limit: int, output_path: str):
    print(f"[build_corpus] Starting corpus extraction (limit={limit})...", flush=True)
    raw_rows = load_msmarco_rows(limit=limit)

    documents = []
    skipped_passages = 0

    for row in raw_rows:
        query_id = str(row.get("query_id", ""))
        query_text = row.get("query") or row.get("Eng_Query") or ""
        eng_query = row.get("Eng_Query") or ""
        source_lang = row.get("source_lang") or "en"
        target_lang = row.get("target_lang") or "hi"

        passages_obj = row.get("passages") or {}
        
        if isinstance(passages_obj, dict):
            trans_passages = passages_obj.get("Translated_passages") or []
            eng_passages = passages_obj.get("English_passages") or []
            is_sel_list = passages_obj.get("is_selected") or []
        else:
            trans_passages = []
            eng_passages = []
            is_sel_list = []

        num_passages = max(len(trans_passages), len(eng_passages))
        
        for idx in range(num_passages):
            passage_text = ""
            if idx < len(trans_passages) and trans_passages[idx]:
                passage_text = str(trans_passages[idx]).strip()
            elif idx < len(eng_passages) and eng_passages[idx]:
                passage_text = str(eng_passages[idx]).strip()

            if not passage_text:
                skipped_passages += 1
                continue

            is_sel = 0
            if idx < len(is_sel_list):
                try:
                    is_sel = int(is_sel_list[idx])
                except (ValueError, TypeError):
                    is_sel = 0

            doc_id = f"{query_id}_{idx}"
            doc = {
                "doc_id": doc_id,
                "language": target_lang,
                "query_id": query_id,
                "query_text": query_text,
                "eng_query": eng_query,
                "passage_text": passage_text,
                "is_selected": is_sel,
                "passage_position": idx,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
            documents.append(doc)

    print(f"[build_corpus] Extracted {len(documents)} document passages from {len(raw_rows)} dataset rows.", flush=True)
    if skipped_passages > 0:
        print(f"[build_corpus] Skipped {skipped_passages} empty passages.", flush=True)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"[build_corpus] Written corpus to {out_file.resolve()} ({out_file.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build MSMARCO-XI flattened corpus for RAG backend.")
    parser.add_argument("--limit", type=int, default=3000, help="Number of dataset rows to load (default: 3000)")
    parser.add_argument("--output", type=str, default="backend/data/corpus.jsonl", help="Output corpus jsonl path")
    args = parser.parse_args()

    build_corpus(limit=args.limit, output_path=args.output)
