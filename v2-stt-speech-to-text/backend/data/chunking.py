import re
from typing import Callable, Optional
import numpy as np
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict = Field(default_factory=dict)
    strategy: str


def split_sentences_indic(text: str) -> list[str]:
    """
    Split text into sentences handling both standard punctuation (., ?, !)
    and Indic punctuation like Devanagari danda ('।') and double danda ('॥').
    """
    if not text or not text.strip():
        return []
    
    # Try NLTK sent_tokenize if available
    sentences = []
    try:
        import nltk
        try:
            raw_sents = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            raw_sents = nltk.sent_tokenize(text)
        
        # Further split any NLTK sentence that contains Devanagari danda '।' or '॥'
        for s in raw_sents:
            sub_sents = re.split(r'[।॥\n]+', s)
            for sub in sub_sents:
                cleaned = sub.strip()
                if cleaned:
                    sentences.append(cleaned)
    except Exception:
        # Regex fallback splitting on standard punctuation, dandas, and newlines
        parts = re.split(r'[.!?।॥\n]+', text)
        sentences = [p.strip() for p in parts if p.strip()]

    return sentences if sentences else [text.strip()]


def fixed_size_overlap(
    text: str,
    metadata: dict,
    chunk_size: int = 220,
    overlap: int = 40
) -> list[Chunk]:
    """
    Naive fixed-size word-count window chunking with overlap.
    """
    words = text.strip().split()
    if not words:
        return []
    
    if len(words) <= chunk_size:
        chunk_id = f"{metadata.get('doc_id', 'doc')}_fixed_0"
        return [Chunk(chunk_id=chunk_id, text=" ".join(words), metadata=metadata.copy(), strategy="fixed_size_overlap")]

    chunks = []
    step = max(1, chunk_size - overlap)
    idx = 0
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_id = f"{metadata.get('doc_id', 'doc')}_fixed_{idx}"
        chunk_text = " ".join(chunk_words)
        
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            metadata=metadata.copy(),
            strategy="fixed_size_overlap"
        ))
        
        if end >= len(words):
            break
        start += step
        idx += 1

    return chunks


def sentence_semantic(
    text: str,
    metadata: dict,
    model=None,
    similarity_threshold: float = 0.55
) -> list[Chunk]:
    """
    Semantic chunking: split into sentences, embed each sentence, and greedily
    merge adjacent sentences while cosine similarity to the running chunk centroid
    remains above similarity_threshold.
    """
    sentences = split_sentences_indic(text)
    if not sentences:
        return []
    
    doc_id = metadata.get("doc_id", "doc")

    # If only 1 sentence or model not supplied, return standard sentence chunk(s)
    if len(sentences) == 1 or model is None:
        chunks = []
        for idx, s in enumerate(sentences):
            chunk_id = f"{doc_id}_semantic_{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, text=s, metadata=metadata.copy(), strategy="sentence_semantic"))
        return chunks

    # Encode sentences
    # Prefix with 'passage: ' if using e5 model, otherwise direct string
    model_name = getattr(model, "model_name_or_path", "") or ""
    encoded_texts = [f"passage: {s}" if "e5" in model_name.lower() else s for s in sentences]
    
    embeddings = model.encode(encoded_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    chunks = []
    current_sentences = [sentences[0]]
    current_embeddings = [embeddings[0]]
    chunk_idx = 0

    def compute_centroid(emb_list):
        vec = np.mean(emb_list, axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    for i in range(1, len(sentences)):
        next_sent = sentences[i]
        next_emb = embeddings[i]

        centroid = compute_centroid(current_embeddings)
        sim = float(np.dot(centroid, next_emb))

        if sim >= similarity_threshold:
            current_sentences.append(next_sent)
            current_embeddings.append(next_emb)
        else:
            # Finalize current chunk
            chunk_text = " ".join(current_sentences)
            chunk_id = f"{doc_id}_semantic_{chunk_idx}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=metadata.copy(),
                strategy="sentence_semantic"
            ))
            chunk_idx += 1
            current_sentences = [next_sent]
            current_embeddings = [next_emb]

    # Append remaining chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_id = f"{doc_id}_semantic_{chunk_idx}"
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            metadata=metadata.copy(),
            strategy="sentence_semantic"
        ))

    return chunks


def metadata_aware(
    text: str,
    metadata: dict
) -> list[Chunk]:
    """
    Metadata-aware strategy: one chunk per passage (no size reduction), but rich
    metadata annotations preserved and enriched for filtering/boosting.
    """
    doc_id = metadata.get("doc_id", "doc")
    chunk_id = f"{doc_id}_meta_0"

    enriched_meta = metadata.copy()
    enriched_meta.update({
        "query_id": str(metadata.get("query_id", "")),
        "language": metadata.get("language", "hi"),
        "source_lang": metadata.get("source_lang", "en"),
        "target_lang": metadata.get("target_lang", "hi"),
        "is_selected": metadata.get("is_selected", 0),
        "passage_position": metadata.get("passage_position", 0),
        "query_text": metadata.get("query_text", ""),
        "doc_id": doc_id,
    })

    return [Chunk(
        chunk_id=chunk_id,
        text=text.strip(),
        metadata=enriched_meta,
        strategy="metadata_aware"
    )]


CHUNKING_STRATEGIES: dict[str, Callable] = {
    "fixed_size_overlap": fixed_size_overlap,
    "sentence_semantic": sentence_semantic,
    "metadata_aware": metadata_aware,
}
