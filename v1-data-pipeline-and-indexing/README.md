# HHG Voice RAG - Multilingual Voice AI System

A production-grade, offline-indexed Multilingual Voice Search and RAG (Retrieval-Augmented Generation) system built with Next.js 14, FastAPI, FAISS vector search, and Indic NLP models.

---

## 🏗 Repository Structure

```text
hhg-voice-rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── schemas.py              <- Shared Pydantic contract definitions
│   ├── data/
│   │   ├── build_corpus.py         <- Offline dataset extraction script
│   │   ├── chunking.py             <- Multi-strategy chunking implementations
│   │   ├── build_index.py          <- FAISS index builder & artifact exporter
│   │   ├── corpus.jsonl            <- Flattened corpus file
│   │   └── index/                  <- Multi-strategy FAISS index artifacts
│   │       ├── fixed_size_overlap/
│   │       ├── sentence_semantic/
│   │       └── metadata_aware/
│   └── requirements.txt            <- Python dependencies
└── frontend/
    ├── app/                        <- Next.js 14 App Router layout & components
    ├── package.json
    ├── tailwind.config.ts
    └── tsconfig.json
```

---

## 🚀 Reproducible Offline Data Pipeline

### 1. Corpus Extraction
To extract and flatten passage documents from `ai4bharat/MSMARCO-XI` (Hindi split `hinval.parquet`):
```bash
cd backend
python data/build_corpus.py --limit 300
```
*Outputs:* `backend/data/corpus.jsonl`

### 2. Multi-Strategy FAISS Index Construction
To chunk, embed with `intfloat/multilingual-e5-base`, and build FAISS vector indices across all three chunking strategies (`fixed_size_overlap`, `sentence_semantic`, `metadata_aware`):
```bash
cd backend
python data/build_index.py --strategy all
```
*Outputs artifacts per strategy under `backend/data/index/<strategy>/`:*
- `faiss.index`: L2-normalized FAISS IndexFlatIP file
- `chunks.jsonl`: JSON line chunk map (maps index row position to chunk text + metadata)
- `meta.json`: Embedding model, vector dimension, chunk count, and build timestamp metadata
